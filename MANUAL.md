# PyCamTRT Python Manual

This is the manual for building your own pipelines with PyCamTRT once the
[README](README.md) has convinced you it's worth trying. It documents every
class and function in the public API (`python/pycamtrt/__init__.py` is the
canon this manual is checked against — see `python/_qa_manual_gate.py`), and
teaches each concept against a real, shipped example rather than a toy
snippet invented for this document. Performance numbers below are quoted
verbatim from the README's measured tables; nothing here is invented.

**Getting the library built**: everything compiles and runs inside a
CUDA/TensorRT container — see [BUILD.md](BUILD.md) for the full story
(prerequisites, the Dockerfile, `pip install .` vs. the classic
cmake/`PYTHONPATH` workflow, and the trtexec appendix). This manual assumes
you already have a working `import pycamtrt` — either via `pip install .`
inside the container, or `PYTHONPATH=/workspace/build:/workspace/python`
pointed at an in-tree build.

## 1. The mental model

PyCamTRT is an open-source, modular alternative to NVIDIA DeepStream: the
same class of performance — zero-copy, multi-stream, cascade inference over
RTSP — without the GStreamer element graph or the closed internals. The
split that makes this possible is architectural, not incidental: **Python is
the control plane, C++/CUDA is the data plane.** You describe a processing
graph in Python — which streams, which TensorRT engines, which postprocess
decode to run, how cascade stages connect — and that description compiles
down to index-based step descriptors (`StepDesc`/`LayerDesc`/
`PipelineConfig`, in `src/core/graph.h`) that a compiled C++ executor
(`pycamtrt::Pipeline`, bound as `_pycamtrt`) actually runs. No Python function
you write ever executes on a per-frame, per-detection, or per-pixel basis.

Concretely, one frame's trip through the pipeline never leaves VRAM until
you ask it to:

```
RTSP → NVDEC decode → fused GPU preprocess (letterbox/normalize) →
batched TensorRT inference → GPU postprocess (NMS/CTC/argmax/embedding) →
[only compact results cross into Python]
```

A frame's pixels are decoded straight into a CUDA buffer, preprocessed and
batched across every stream sharing an engine, run through TensorRT, and
decoded by a GPU kernel into a short list of survivors (boxes, text,
labels — tens of items, kilobytes) — all of it async, on the GPU, before a
single byte reaches your process. What crosses the C++/Python boundary per
frame is that compact result, not pixels. If you explicitly ask for pixels
(`fetch_frame()`, `frame_cuda()` — §3.3), you get them; otherwise they never
move.

"Declarative graph" means exactly what the two example snippets in the
package docstring show: you build a small object graph out of three generic
stage kinds — `Process` (preprocess), `Engine` (a TensorRT engine call), and
`Postprocess` (a decode kernel) — grouped into named `Layer`s, and hand the
whole thing plus a `Streams` source to `Pipeline`. `Pipeline.__init__` is a
*compiler*: it walks your Python objects once, resolves every step's `input`
reference into a flat array index, and hands that index graph to the C++
side, which validates it (`Validate()` in `pipeline.cpp`) and loads/builds
whatever TensorRT engines it names. You never describe *how* frames flow
through memory — only *what* runs on what. Section 2 covers building that
graph; §2.7 covers exactly what happens when `Pipeline()` compiles and
`start()`s it.

This is also why the library reads as small from the outside despite doing
a lot underneath: the public surface (`python/pycamtrt/__init__.py`) is
almost entirely *description* — a dozen small classes, most of them just
validating and holding onto arguments — while the actual frame-by-frame
work (NVDEC session management, CUDA stream scheduling, batching across
cameras that happen to share an engine, the postprocess kernels themselves)
lives in `src/core/` and never surfaces as something you call. Comparing
this to DeepStream is fair specifically here: DeepStream gets its
performance from a GStreamer pipeline of plugin elements you wire together
with capsfilters and properties; PyCamTRT gets the same class of
performance from a much smaller, Python-native object graph compiling to a
purpose-built (not general-purpose-media-framework) executor. You give up
GStreamer's element ecosystem; you gain a graph you can read top to bottom
in a normal Python file, and a debugger that can step into your own
control-plane code without stepping into a plugin's opaque state machine.

## 2. Building a graph

### 2.1 Streams and Stream: where every graph starts

Every step graph is rooted at a `Streams` object:

```python
streams = pycamtrt.Streams(["rtsp://cam1", "rtsp://cam2"])
```

`Streams(urls)` takes a list where each entry is either a plain URL string
(inherits the pipeline-wide `skip`/`decode` — see §3.1) or a `Stream` object
overriding those two load dials for just that camera:

```python
streams = pycamtrt.Streams([
    "rtsp://entrance",                 # pipeline defaults
    pycamtrt.Stream("rtsp://parking", skip=4),   # infer 1-in-4 for this cam
])
```

`Stream(url, skip=None, decode=None)` validates `skip >= 1` and
`decode in ("all", "key")` at construction. Capacity planning is
*per-camera*: an entrance camera you want at full fidelity can sit right
next to a parking camera you're happy to under-sample — mixing dials on one
`Streams` list is how you express that. A step whose `input` is the
`Streams` object itself compiles to `StepDesc.input = -1`, the C++
convention for "raw stream source" (`graph.h`).

### 2.2 The three generic stages

Every layer is built from `Process`, `Engine`, and `Postprocess` — a
`_Step` base class holding one thing: the `input` reference that IS the
dataflow edge. There's no separate "wire this to that" API; you wire the
graph by passing one step as another's `input=`.

**`Process(input)`** is an explicit preprocess stage — optional, since an
`Engine` fed `Streams` (or another step) directly assumes the implicit
preprocess. You'd only add it to name/reuse that stage explicitly; none of
the shipped examples need it.

**`Engine(input, engine_path, norm=None, color=None, max_batch=16, fp16=True, shape=None)`**
runs a TensorRT engine. `input` is `Streams` (a layer-0 detector reading
whole frames) or a `Postprocess` step from an earlier layer — feeding an
`Engine` a `Postprocess` step is the cross-layer edge that auto-crops each
survivor detection on the GPU before this engine sees it (the mechanism
every cascade in this manual uses). `engine_path` can point at a prebuilt
`.engine` or straight at a `.onnx` — more in §2.5. `norm`/`color` are
per-engine input transform knobs (see below); `max_batch`/`fp16`/`shape`
only matter when auto-building from an `.onnx`.

`norm` takes `(offset, scale)` applied per channel as
`(pixel[c] + offset[c]) * scale[c]`, in two shapes: a scalar pair broadcast
to all three channels, or a genuine per-channel
`((o0,o1,o2), (s0,s1,s2))` pair. Channel index 0/1/2 follow `color`'s order
(`color="rgb"` → 0=R; `"bgr"` → 0=B). This is how every classifier/re-ID
example in this manual states ImageNet normalization — torchvision's
mean/std are RGB-ordered, converted as `offset[c] = -mean[c]*255`,
`scale[c] = 1/(std[c]*255)`:

```python
IMAGENET_NORM = ((-123.675, -116.28, -103.53),
                  (1 / 58.395, 1 / 57.12, 1 / 57.375))
```

(`examples/route_and_reid/route_and_reid.py`). Leaving `norm`/`color` as
`None` inherits a per-position family default: layer 0 (whole-frame/SAHI
tile, YOLO-family input) is `(0, 1/255)`, RGB; a cascade crop layer is
`(-127.5, 1/128)`, BGR (today's LPRNet/OCR convention) — override for
anything trained differently.

**`Postprocess(input, family, score=0.4, iou=0.45)`** decodes an `Engine`'s
raw output tensor. `input` is the `Engine` step whose output this reads;
`family` selects which GPU decode kernel runs (§2.3); `score`/`iou` are
honored or ignored depending on family. This is the step whose *output*
other layers' `Engine`s crop from, and the step `Result.outputs` (§3.2)
keys its data off of.

### 2.3 Postprocess and the six families

`family=` selects one of six compiled decode kernels (`_FAMILIES` in
`__init__.py`, mapped to `pycamtrt::Family`). Two are layer-0 *detector*
families (they read a whole-frame/tile tensor and produce `Detection`
objects); four are shapes a *cascade child* can produce (three of them —
`ctc`/`argmax`/`embedding` — read a per-crop tensor and produce something
aligned with the parent layer's detections):

| family | position | reads | produces |
|---|---|---|---|
| `yolo` | layer 0 | anchor-based YOLO head; NMS applied (`iou` honored) | `Detection` list |
| `yolo-e2e` | layer 0 | NMS-free end-to-end head (yolo26-style `[N,300,6]`); `iou` ignored | `Detection` list |
| `rtdetr` | layer 0 | RT-DETR's `[N,300,6]`, normalized cx,cy,w,h; `iou` ignored | `Detection` list |
| `ctc` | cascade child | OCR sequence head (LPRNet-style) | `list[str]`, aligned with detections |
| `argmax` | cascade child | classifier `[N,classes]` logits, plain max — no softmax | `list[(label:int, score:float)]` |
| `embedding` | cascade child | raw `[N,D]` feature head — no decode at all | one `[D]`-float list per detection |

`yolo-e2e`/`rtdetr` both reject SAHI at `Pipeline` construction (a named
`RuntimeError`) — cross-tile NMS merge is undefined for an already-NMS-free
head. `embedding` is a genuine pass-through: "there is no kernel, the
pass-through IS the decode" (module docstring) — whatever your re-ID/feature
engine's second-to-last layer outputs comes back raw, for you to
cosine/L2/cluster however you like (`route_and_reid.py`, §6.5, does exactly
that in plain Python).

Custom decodes are possible but deliberately not a runtime plugin: adding a
seventh family means writing a CUDA kernel and recompiling
(`docs/ADDING_A_FAMILY.md` walks through it using `argmax` as the template).
If your logic only needs the *survivors* a family already produced — zone
filters, alerting, temporal smoothing — it belongs in plain Python on the
consumer thread instead (§6.6's `zone_filter.py`); that tier needs no
recompile.

### 2.4 SAHI: tiled inference for small/far objects

A whole-frame 640×640 letterbox shrinks a distant license plate or a small
person to a handful of pixels — SAHI (Slicing Aided Hyper Inference) trades
GPU cost for recall on exactly that case, as a per-layer option:

```python
L1 = pycamtrt.Layer("detect", sahi=dict(tile=640, overlap=0.2))
eng = L1.add(pycamtrt.Engine(streams, "detector.engine"))
L1.add(pycamtrt.Postprocess(eng, family="yolo"))
pipe = pycamtrt.Pipeline(streams, layers=[L1])
```

Instead of one whole-frame pass, the frame is cut into overlapping
`tile`-pixel tiles, each run through the *same* engine, then merged back
into one detection list (cross-tile NMS at threshold `merge_iou`). A
full-frame pass runs too by default (`full_frame=True`), so objects larger
than one tile are still caught. The dict's keys — `tile` (default 640),
`overlap` (default 0.2), `merge_iou` (default 0.5), `full_frame` (default
`True`), `serial` (default `False`) — are all optional; *presence* of the
dict, even `{}`, is what turns SAHI on for that layer. `Layer(sahi=...)` is
the preferred, per-layer form; `Pipeline(sahi=...)` is a legacy
position-based alias that always means "layer 0," useful only for code that
predates per-layer `Layer(sahi=...)` — giving both with genuinely different
dicts raises `ValueError` (ambiguous which one wins), the same dict on both
is redundant but harmless. The v1 executor only accepts SAHI on the `yolo`
detection layer (today, always layer 0) — putting it on a cascade layer, or
on a `yolo-e2e`/`rtdetr` layer (§2.3), raises a named `RuntimeError` at
`Pipeline` construction.

**Pooled batching is the default; `serial=True` is an A/B escape hatch.**
Pooling batches tile jobs from every in-flight camera slot into shared
engine calls (capped at the engine's max batch) — never slower, and usually
much faster, than `serial`, which keeps each frame's tiles in their own
chunk and exists mainly for isolating a regression. Do not change the 640
default lightly: it's a recall/cost trade, not a pure speed knob — tile
1280 (fewer, bigger tiles) trades small-object recall for throughput, and
the README's atlas shows exactly how much (pooled tile 640 tops out around
a 60 fps whole-GPU fleet ceiling at 4K vs. ~208 fps at tile 1280). Rather
than guess, run a candidate configuration through `pycamtrt.recommend()`
(§5) first — its `sahi=` argument models this exact cost curve
(`_capacity.py`'s `sahi_tile_count()` derives the same tile grid the C++
side actually cuts, so the tile count `T` behind the `a + b*T` cost model
isn't a guess either).

### 2.5 Engine auto-build and the cache

Point `Engine()` at an `.onnx` instead of a `.engine` and the TensorRT
engine is built once per GPU, in-process, and cached next to the `.onnx`
with the naming `<stem>_b1-<max_batch>_<fp16|fp32>_sm<arch>.engine` — e.g.
`yolov8n_plates_b1-16_fp16_sm86.engine` for `max_batch=16, fp16=True` on an
sm_86 GPU (`models/README.md`). `max_batch` sets the dynamic-batch profile's
upper bound (min is always 1 — a TensorRT profile with `kMIN` batch > 1 is
rejected with a named error); `shape=(H, W)` only matters for exports with
symbolic spatial dims (`ultralytics(dynamic=True)` exports); `fp16=True` is
the default and matches every measured number in the README.

Build cost is real and worth planning around — and it is entirely avoidable
if `max_batch` is a constant instead of derived from the live stream count.
An earlier version of this repo's app example sized `max_batch=max(4,
len(sources))`, so every camera-count change that crossed a batch-size
boundary triggered a genuinely COLD TensorRT build (measured: 219s, versus
9s when reusing an already-cached profile). `fleet_demo` instead
**pins `max_batch=32` forever** — legal because streams exceeding an
engine's max batch legally CHUNK rather than reject (§2.7 covers the
`Validate()` rules this doesn't run into; the batching itself just logs a
warning and proceeds). Every rebuild, at any fleet size from 1 to 65
streams, now targets the SAME cached engine file: the acceptance run's
9-stream-to-65-stream scale-out rebuilt within a single ~2s status-poll
tick, and a same-fleet-size dial change (one scenario's `decode` flipped
from `"all"` to `"key"`) rebuilt in **~2.7s** — both warm, neither a fresh
TensorRT build. The cost of pinning: batches larger than 32 streams simply
chunk (a logged warning, not a rejection), capping per-tick throughput at
full (`skip=1, decode="all"`) rate past that point — acceptable when a
Performance Lab's own capacity model (§5) is telling the operator to turn
down `skip`/`decode` before the fleet gets that large anyway. A production
rebuild path (§6.7) should treat "pin the batch profile, accept chunking"
as the default, and "size `max_batch` to the live fleet" as the exception
that buys a bigger single-tick batch at the cost of occasional cold builds.

### 2.6 Select: routing detections to cascade children

`Select(input, classes=None, min_score=None, min_size=None)` is a
declarative filter sitting between a detector layer's `Postprocess` and a
cascade child's `Engine`: only detections passing *every* active criterion
get cropped and sent to that child.

```python
person = pycamtrt.Layer("person")
sel = person.add(pycamtrt.Select(det, classes={0}, min_size=32))
eng = person.add(pycamtrt.Engine(sel, "models/resnet18emb...", ...))
person.add(pycamtrt.Postprocess(eng, family="embedding"))
```

(adapted from the `Select` docstring; `route_and_reid.py`, §6.5, is the full
runnable version with two routed branches.) `classes` takes integer class
ids — the library is deliberately model-agnostic here: "cls 0 means person
only if YOUR detector says so," never a class *name*. `min_score` filters on
detection confidence; `min_size` filters on the smaller side (in source
pixels) of the crop the child would actually see.

The contract that makes routing safe to reason about: **results stay
aligned even when routing is on.** `outputs[layer][i]` still describes
`detections[i]` — a routed-away detection simply gets that child's empty
entry (`""` for `ctc`, `[]` for `embedding`, an empty pair for `argmax`).
You never need to know which detections were routed away to index
correctly; `route_and_reid.py`'s loop below shows the pattern (checking
truthiness of the entry, not tracking indices separately). `Select`'s input
must be the *detector* layer's `Postprocess` step — feeding it a cascade
child's own output is the depth-3 (chain-of-chains) case v1 rejects (§2.7).
Multiple children may share one `Select` handle; routing can only reduce
work, never add it (measured: −36% per-child GPU time with half the
detections routed, per the README).

Concretely, this means a cascade child `Layer` only ever takes one of two
shapes: `[Engine, Postprocess]` (unrouted — every detection reaches this
child, exactly what `read_plates.py`'s `read` layer and
`classify_detections.py`'s `classify` layer look like, §6.1/§6.2) or
`[Select, Engine, Postprocess]` (routed — `route_and_reid.py`'s `person`
and `vehicle` layers, §6.5). Both shapes coexist freely in one pipeline; a
`Select` is purely additive to the plain two-step form, never a different
kind of layer.

### 2.7 How the graph actually becomes a pipeline

`Layer(name, sahi=None)` is just a named ordered container —
`layer.add(step)` appends and returns the step (so you can pass it as
another step's `input=`); `layer.steps` is a read-only view. Nothing is
compiled yet at this point; `Layer`/`Streams`/steps are plain Python objects
holding references to each other.

Compilation happens inside `Pipeline.__init__` (`python/pycamtrt/__init__.py`).
It walks `layers` in order, and for every step it builds a `_c.StepDesc`,
resolving `step.input` to a flat integer index via an `id(python object) ->
compiled index` dict (seeded with `{id(streams): -1}`). Each step kind fills
in its own fields on the descriptor — `Engine` sets `engine_path`,
`norm_offset`/`norm_scale`, `color`, `build_max_batch`, `build_fp16`,
`build_h`/`build_w`; `Postprocess` sets `family`, `score_thresh`,
`iou_thresh`; `Select` sets `sel_classes`/`sel_min_score`/`sel_min_size`.
Layers become `_c.LayerDesc`s carrying their step indices and any SAHI
dict (`Layer(sahi=...)`, applied via `_apply_sahi`); sinks compile last
(events sinks reference an already-compiled `Postprocess` index).

The assembled `_c.PipelineConfig` is then handed to `_c.Pipeline(cfg)` —
**this call is where engine load or auto-build actually happens**, under
the GIL released on the C++ side. This is also where the C++ `Validate()`
runs: the single source of truth for which step shapes are legal (e.g. "v1
executor supports a depth-2 tree... chained cascades (a child of a child)
are not yet executable"). Any rejection surfaces as a Python `RuntimeError`
carrying that C++ message verbatim — the Python layer above does *not*
re-validate graph shape beyond mapping `family=` strings.

`pipe.start()` is the second, separate step: it spawns the producer
(demux/decode) and GPU worker threads. It's single-shot — call it twice and
it raises, mirroring `pycamtrt::Pipeline::Start`. Iterating a `Pipeline`
directly (`for r in pipe:`) calls `start()` for you if you haven't already;
using it as a context manager (`with pycamtrt.Pipeline(...) as pipe:`) calls
`stop()` on exit.

**Streams are fixed at construction.** There is no `pipe.add_stream()` —
the source list you pass to `Streams()` is baked into the `Pipeline` object
at compile time, and so is every engine/layer downstream of it. Adding or
removing a camera means building an entirely new `Streams` + `Layer` graph
and a new `Pipeline` object, then stopping the old one. This sounds like a
limitation until you see it done for real: `fleet_demo` does exactly
this, live, every time a slider is applied (§6.7 walks through its rebuild
loop — the short version is `pipe.stop()` from the web thread unblocks the
consumer thread's `for r in pipe:`, which then calls `build_pipeline()`
again with the new dials and `pipe.start()`s the replacement).

## 3. Running: Pipeline and Result

### 3.1 Pipeline kwargs

Beyond `streams`/`layers`, `Pipeline(...)` takes:

| kwarg | default | controls |
|---|---|---|
| `skip` | `1` | infer every Kth *decoded* frame (pipeline-wide; `Stream(skip=...)` overrides per camera) |
| `decode` | `"all"` | `"all"` or `"key"` (keyframes only) — pipeline-wide decode rate, overridable per stream |
| `max_frames` | `0` | decoded frames per stream before auto-stop; `0` = run until `stop()` |
| `verify` | `False` | per-frame CPU-reference check (sets `r.verify_ok`) |
| `queue_capacity` | `256` | result-queue depth |
| `backpressure` | `"block"` | `"block"` (GPU thread waits on a full queue) or `"drop_oldest"` (evicts the oldest result instead — see `dropped_results()`) |
| `hold_frames` | `False` | enables tier-3 frame access (`frame_cuda()`) — see §3.3 |
| `ring_depth` | `4` | frames-in-flight slack per stream; raise when holding frames |
| `ring_seconds` | `10` | seconds of *compressed* packets kept per stream, feeding `Sink(kind="stream")` and `clip()`; `0` disables the ring |
| `sinks` | `()` | `Sink` objects (§4) |
| `cascade_serial` | `False` | `True` runs cascade children sequentially instead of on their own CUDA streams — an A/B escape hatch |
| `sahi` | `None` | legacy, position-based SAHI dict (prefer `Layer(sahi=...)`, §2.4) |
| `log` | `None` | callable taking one `str`, invoked from C++ worker threads — keep it cheap, it holds the GIL |

`skip` and `decode` are the two capacity dials the README's atlas table is
built from: at or below a fleet's design point, turning either dial buys
GPU/NVDEC *headroom*, not lower latency — see §5 for `recommend()`, which
turns these same knobs into a prediction.

### 3.2 Iterating results

```python
with pipe:
    for r in pipe:
        print(r.stream_id, r.frame_no, len(r.detections))
```

`for r in pipe:` calls `_pipe.poll(500)` internally — a 500ms-timeout poll
that releases the GIL while waiting (so Ctrl-C and other Python threads stay
responsive), looping past a timeout and raising `StopIteration` once the
pipeline reports `Finished` (every stream's producer gave up, or
`max_frames` was reached everywhere). Each ready poll wraps one
`_pycamtrt.FrameResult` in a `Result`.

`Result.outputs` is a dict keyed by layer name. `outputs[layer_names[0]]`
(the detector layer) is always `r.detections`; every layer past it (a
sibling cascade child — see §2.6/§6.3's depth-2 tree) is resolved to the
shape its family implies: `list[str]` for `ctc`, `list[(label, score)]` for
`argmax`, one `[D]`-float list per detection for `embedding`. This is
exactly the shape table from §2.3, already unpacked by name — you index
`outputs["read"]` instead of hunting through `r.children`.

Every raw `FrameResult` field is also exposed directly on `Result` (via
`__getattr__`, so `r.stream_id` just reads through):

- `stream_id`, `pts_us`, `frame_no`, `batch_size`, `batch_seq`
- `detections` — the detector layer's `Detection` list (`.x`, `.y`, `.w`,
  `.h`, `.score`, `.cls`)
- `texts`, `labels`, `label_scores` — pre-tree back-compat fields (first
  `ctc`/first `argmax` child only); `children` is the full per-child list
  (`ChildOutput`, one per sibling, in layer order) that `outputs` above is
  built from
- `ms_pop_to_ready`, `ms_ready_to_take`, `ms_take_to_done` — the three-way
  timing split (pop-off-queue-to-ready, ready-to-consumer-take,
  take-to-decode-done); a cascade child's own cost is
  `ChildOutput.ms_gpu`. None of the three sees the decoder itself: set the
  environment variable `PYCAMTRT_DIAG_DEMUX=1` and every stream prints its
  mean demuxer-to-pop time (parser plus hardware-decoder queue) when it
  exits — about the decode time at low load, growing only when NVDEC
  saturates
- `verified`, `verify_ok`, `verify_cpu_dets` — populated only when
  `Pipeline(verify=True)`
- `frame_addr`, `frame_pitch`, `frame_width`, `frame_height` — tier-1 memory
  fields, always present, printable/loggable, never a live pointer (see
  §3.3)

`read_plates.py` (§6.1) uses exactly this surface: `r.outputs["read"]` for
the aligned OCR strings, `r.stream_id`/`r.frame_no` for the print line, and
`r.verify_ok` when `--verify` is passed.

Three of the names on this surface are direct pybind11 bindings, exported
at the top of `pycamtrt/__init__.py` rather than wrapped: `Detection =
_c.Detection` (the `.x/.y/.w/.h/.score/.cls` struct every `detections`
entry is), `StreamInfo = _c.StreamInfo` (`.decoded`/`.reconnects`/`.failed`
— what `get_stream_info()` returns, §3.4), and `PollStatus =
_c.Pipeline.PollStatus` (the enum `__next__` checks internally — you won't
normally touch it yourself unless you're driving `_pipe.poll()` by hand).
None of the three need a Python wrapper class of their own; they're plain
structs, so the binding *is* the API.

### 3.3 The three-tier frame-memory access

Every `Result` carries its underlying full-res NV12 frame at three
escalating levels of commitment:

**Tier 1 — always on, informational.** `r.frame_addr`/`frame_pitch`/
`frame_width`/`frame_height` are populated on every result. Fine to print or
log; the memory they describe may be recycled by another frame at any
moment — never treat them as a live pointer.

**Tier 2 — `r.fetch_frame()`.** A plain device-to-host copy into a fresh
numpy array, shape `(height*3/2, width)` uint8 (NV12: luma plane then
interleaved UV). Safe from any thread; costs a copy; requires
`hold_frames=True` (same requirement as tier 3, since both need the ring
slot to still be alive when they run).

**Tier 3 — `Pipeline(hold_frames=True)` + `r.frame_cuda()`.** Keeps this
result's ring slot alive until every copy of the `Result` is garbage
collected or `r.release_frame()` is called. `frame_cuda()` then hands back a
zero-copy CUDA view (`__cuda_array_interface__`) usable directly with
`torch.as_tensor(r.frame_cuda(), device="cuda")` or `cupy.asarray(...)` —
no copy, no format conversion.

Holding a frame means its ring slot can't be reused until you let go of it,
so `ring_depth` (default 4, fine when not holding) must be sized to how
many frames of *one stream* you plan to hold concurrently — a consumer
holding `ring_depth - 1` frames stalls that stream's producer (intended
backpressure, not a bug). A consumer that holds frames should call
`r.release_frame()` unconditionally at the end of every iteration — exactly
the discipline that keeps a hold-frames consumer from silently exhausting
its ring. (`fleet_demo` never reads pixels, so it leaves `hold_frames=False`
and the default `ring_depth`.)

### 3.4 clip() and stream stats

`pipe.clip(stream_id, seconds_back, path)` writes a raw Annex-B `.h264`
elementary stream covering roughly the last `seconds_back` seconds (from the
newest keyframe at or before that point, to now) to `path` — pulled from the
same packet ring `Sink(kind="stream")` replays from (§4.2), independent of
whether any sink is configured. Returns `False` (and logs why) if the
stream id is invalid, the ring is disabled (`ring_seconds=0`) or still
empty, or no keyframe is available yet. Safe from any thread while running.

`pipe.get_stream_info(i)` returns a `StreamInfo` with `.decoded`,
`.reconnects`, `.failed` — every example in §6 prints this per stream at
the end of its run. `pipe.dropped_results()` counts results evicted under
`backpressure="drop_oldest"` (always 0 under the default `"block"`).
`pipe.sink_dropped()` is the sink-side equivalent (§4.1). `pipe.max_batch()`/
`.classes()`/`.anchors()` expose the compiled detector engine's own
dimensions.

## 4. Getting data out: Sinks

A `Sink` is a strictly downstream tap: **it never backpressures the
pipeline.** Each sink drains through its own bounded, drop-oldest queue on a
dedicated thread — a slow consumer of a sink loses its *own* oldest data,
never a frame from the pipeline proper. `Sink(input, kind, target,
stream=None)` comes in two kinds.

### 4.1 `kind="events"` — NDJSON over TCP

```python
events = pycamtrt.Sink(det, kind="events", target="tcp://0.0.0.0:9000")
```

`input` must be one of the pipeline's `Postprocess` steps (the detector, or
any cascade child — results are per-frame, not per-layer). `target` is
`"tcp://host:port"`, `"file:///abs/path"`, or `"stdout"`. Each consumed
`FrameResult` becomes one compact NDJSON line:

```json
{"stream": 0, "frame": 1234, "pts_us": 41133333, "batch": 4,
 "dets": [{"x": 12.0, "y": 8.0, "w": 40.0, "h": 20.0,
           "score": 0.91, "cls": 0}],
 "texts": ["ABC123"],
 "ms": {"pre": 0.8, "queue": 0.2, "gpu": 1.1},
 "frame_addr": "0x7f0a1c000000"}
```

`ms` mirrors the `Result` timing split (§3.2); `frame_addr` is the same
tier-1 informational address as `Result.frame_addr` — printable/loggable at
the receiving end, never dereferenceable there. `stream=None` (default)
emits every stream; pass an index to filter to one camera. Notice the
schema's own shape: it carries `dets`/`texts`, not arbitrary per-family
payloads — an `embedding` family's raw vectors have no field here (§4.3
below is exactly why that matters).

`target`'s three forms cover the three places you'd actually want this
stream: `"tcp://host:port"` for a real network consumer (a dashboard
backend, a log shipper); `"file:///abs/path"` for a plain NDJSON log file on
disk (handy for offline analysis or a quick `tail -f`); `"stdout"` when
you just want to eyeball events while developing a new graph, with no
consumer process at all. All three are checked at `Pipeline` construction
time, so a typo'd target fails fast rather than silently dropping every
line at runtime.

**The pipeline connects *out*, not in** — a design choice, not an oversight:
nothing on the pipeline side ever listens for a connection — a fleet-inference
process shouldn't need an open listening port just to report what it saw;
the consumer (dashboard, log aggregator, whatever) owns the socket and the
pipeline dials it.

A full/stalled receiver never slows inference: `pipe.sink_dropped()`
reports the total NDJSON lines dropped across every events sink's own
queue once it falls behind its target — 0 if no sink has ever fallen
behind, always 0 with no events sink configured at all.

### 4.2 `kind="stream"` — RTSP relay, no re-encode

```python
relay = pycamtrt.Sink(streams, kind="stream", stream=0,
                       target="rtsp://localhost:8554/cam1_relay")
```

`input` must be the pipeline's `Streams` object — it forwards the
*original* compressed video, upstream of any inference step, republished
with `-c copy` (no re-encode). `stream=` is **required** here: v1 restricts
one relay sink to one concrete camera (one relay target is one RTSP path;
multiplexing "all streams" onto one path has no defined behavior). A
viewer's first connection replays that stream's ring history (up to
`Pipeline(ring_seconds=...)` of it, §3.1) at realtime pace, then goes live
— so connecting mid-run still gets a continuous stream from a few seconds
back, not a jump straight to "now." That replay is exactly why `fleet_demo`
does NOT use one for its tiles: measured on 2026-09-12, the relay's first
connection came ~8 s after its source opened, replayed that backlog at
realtime pace, and left the relayed video a fixed 7.4 s (74 frames) behind
the detections — an offset a per-frame overlay cannot bridge. The demo
plays the publishers' own mediamtx paths instead (§6.7); a relay that
starts from the newest keyframe rather than the oldest would remove the
offset and is the library-side fix to make.

### 4.3 Sinks vs. in-process consumption

Sinks are the right tool when the payload a `Postprocess` step already
produces is exactly what a downstream consumer needs, and that consumer can
live in a separate process. They're wrong the moment your logic needs
either (a) a payload shape the events schema doesn't carry, or (b) state
that persists *across* frames.

`fleet_demo`'s tracking is the second kind. Associating this frame's
detections with last frame's tracks — greedy IoU matching, a
tentative-then-confirmed lifecycle, aging out unmatched tracks — is
inherently *stateful*: it has to run, in order, on every frame, against
per-camera memory that survives from one frame to the next, and its output
(a track id per box) is not a field the `kind="events"` schema carries. A
fire-and-forget NDJSON tee can't do that. So the demo consumes results
**in-process** — the same `for r in pipe:` loop the earlier examples use,
calling its `consume()` (§6.7) instead of printing — and sends its *own*
frame events (box + class + track id) to the browser over SSE. It uses the
library's `Sink(kind="stream")` only for the one thing that genuinely has
no state — the raw video relay — and nothing else.

The rule this generalizes to: reach for a `Sink` when you're tapping a
result stream a separate process can consume as-is; reach for the
`for r in pipe:` loop itself when your logic needs a payload shape the
schema doesn't carry, or memory across frames.

## 5. Capacity planning: recommend()

`pycamtrt.recommend()` is a pure-Python, dependency-free cost *model* — not
a live benchmark — built from the measured capacity atlas the README's
table is drawn from (RTX 3060 Ti, sm_86, fp16 cascade engines). It needs no
GPU and no filesystem access to run:

```pycon
>>> r = pycamtrt.recommend(streams=16, resolution="4k", fps=30,
...                        sahi={"tile": 640})
>>> r.holds_realtime, r.suggestion
(False, 'over capacity: raise skip to 9 / use decode="key" / ...')
```

`recommend(streams, resolution="720p", fps=30, skip=1, decode="all",
key_gop=30, sahi=None)` mirrors the same dials `Pipeline`/`Stream` accept:
`resolution` is one of `"720p"`/`"1080p"`/`"1440p"`/`"4k"` (only affects the
NVDEC ceiling and, with SAHI, tile count — inference cost itself is
resolution-invariant, everything letterboxes to 640×640); `skip`/`decode`
scale the inference/decode load exactly as `Pipeline(skip=..., decode=...)`
would; `sahi` mirrors `Layer(sahi=...)`'s dict (minus `merge_iou`, which has
no cost-model effect).

What comes back is a `Recommendation` dataclass — `predicted_gpu_ms_per_frame`,
`gpu_capacity_fps`/`nvdec_capacity_fps`, `required_gpu_fps`/
`required_nvdec_fps`, `gpu_headroom_pct`/`nvdec_headroom_pct`,
`binding_resource` (`"gpu-inference"`, `"nvdec"`, or `"none"`),
`holds_realtime` (both ratios ≤ 1.0), a human `suggestion` string naming
concrete dials to turn when over capacity, and a `notes` list carrying the
model's own provenance and caveats — always non-empty, because a
`Recommendation` is meant to travel on its own once returned (it's a plain
dataclass — `dataclasses.asdict()` serializes it trivially).

The model's own biggest limit, stated in every call's `notes`: every
constant was measured on **one GPU**. A different SM count or NVDEC
generation needs its own atlas run before these numbers mean anything
there — treat `recommend()` as a seed, not a portable model, on different
hardware.

`fleet_demo` calls it once a second from its stats thread —
`recommend(streams=N, resolution="1080p", fps=10, skip=skip, decode=decode)`
for whatever the three sliders currently say — and prints the prediction on
the same line as the measured decoded fps, inferred fps and GPU ms per
frame, so the two can be read side by side while the dials move (§6.7).
`recommend()` is UNIFORM-only (one resolution/fps/skip/decode per call); a
fleet whose cameras run different dials needs one call per homogeneous
group with the `required_gpu_fps`/`required_nvdec_fps` summed against the
GPU's capacity — the "group-and-sum" pattern the demo doesn't need because
its dials are global.

## 6. Worked examples

`examples/` ships six matched library-mechanics demos (each a `.py`, a
`.cpp` port with the identical step graph, a `README.md` with exact run
commands and a captured expected-output sample) plus one live application,
`fleet_demo`. Run any of the six with `tools/stream_farm/farm.sh` in
place of real cameras — most need
`CLIP=media/atlas_plate_g30.mp4 ./farm.sh up 2` so the plate detector has
something to find (see `examples/CLAUDE.md`). Full run commands are each
example's own README; what follows is the graph each one builds and why.

### 6.1 read_plates — the basic cascade

```python
detect = pycamtrt.Layer("detect")
eng = detect.add(pycamtrt.Engine(streams, DETECT_ENGINE))
detect.add(pycamtrt.Postprocess(eng, family="yolo", score=0.4))

read = pycamtrt.Layer("read")
# Cross-layer edge: an Engine fed a Postprocess step auto-crops each
# detection from the earlier layer before running the OCR engine on it.
oeng = read.add(pycamtrt.Engine(detect.steps[-1], READ_ENGINE))
read.add(pycamtrt.Postprocess(oeng, family="ctc"))
```

(`examples/read_plates/read_plates.py`) This is the two-layer detect→crop→OCR
shape from §2.2's cross-layer edge, as small as it gets: one detector, one
`ctc` child, `skip=2`. Expected output (captured against the plate farm):
a dominant plate string appearing hundreds of times per run
(`<wan>Y2444` in the shipped capture) with a handful of near-miss variants
— the LPRNet engine is Chinese-charset-trained, so non-Chinese plates read
garbled by design; the point is the crop→OCR *mechanics*, not charset fit.

### 6.2 classify_detections — the argmax family, real per-channel norm

Same shape as §6.1, with the OCR child swapped for a classifier:

```python
classify = pycamtrt.Layer("classify")
ceng = classify.add(pycamtrt.Engine(detect.steps[-1], CLASSIFIER,
                                    norm=CLASSIFIER_NORM, color="rgb"))
classify.add(pycamtrt.Postprocess(ceng, family="argmax"))
```

(`examples/classify_detections/classify_detections.py`) This is the
worked example for §2.2's per-channel `norm=` — true ImageNet normalization,
not the scalar-approximation shortcut. `CLASSIFIER_NORM` is the exact
`IMAGENET_NORM` tuple from §2.2, spelled out again here because this script
predates `route_and_reid.py` and the two are kept in sync by hand, a small
"replicate the constant" tax the manual's own §2.2 explicitly calls out.
Expected output: `outputs["classify"]` as `(label:int, logit:float)` pairs,
aligned with `r.detections`; since the plate detector's crops get run
through an *ImageNet* classifier, labels are semantically nonsensical
("street sign," "digital clock") — again, the cascade mechanics are what's
under test, not classification accuracy. The `argmax` score returned is a
raw logit, not a softmax probability — `LaunchArgmaxBatched`'s own
documented contract (§2.3) — so don't compare it across models or treat it
as a confidence percentage.

### 6.3 read_and_classify — a depth-2 tree, two siblings

```python
detect = pycamtrt.Layer("detect")
eng = detect.add(pycamtrt.Engine(streams, DETECT_ENGINE))
root = detect.add(pycamtrt.Postprocess(eng, family="yolo", score=0.4))

read = pycamtrt.Layer("read")
reng = read.add(pycamtrt.Engine(root, READ_ENGINE))          # <- root
read.add(pycamtrt.Postprocess(reng, family="ctc"))

classify = pycamtrt.Layer("classify")
ceng = classify.add(pycamtrt.Engine(root, CLASSIFIER_ENGINE,  # <- root, NOT read
                                    norm=CLASSIFIER_NORM, color="rgb"))
classify.add(pycamtrt.Postprocess(ceng, family="argmax"))
```

(`examples/read_and_classify/read_and_classify.py`) This is §2.6's depth-2
tree made concrete: **both** children's `Engine`s take `root` — the
detector's own `Postprocess` step — as `input`, never each other's output.
That's the whole shape v1 executes: sibling recognizers, not a 3-deep chain
(feeding `read`'s output into another `Engine` is exactly the "child of a
child" case `Validate()` rejects by name). It's built entirely from the
previous two examples' unchanged engines — proof the tree generalization is
additive. Expected output: `r.outputs["read"]` and `r.outputs["classify"]`
both populated per detection, same index into both.

### 6.4 detect_rtdetr — a transformer detector, no cascade

```python
detect = pycamtrt.Layer("detect")
eng = detect.add(pycamtrt.Engine(streams, ENGINE))
detect.add(pycamtrt.Postprocess(eng, family="rtdetr", score=0.5))
```

(`examples/detect_rtdetr/detect_rtdetr.py`) The simplest possible graph —
one layer, no cascade — teaching the `rtdetr` family from §2.3: NMS-free,
`score` honored, `iou` silently ignored (there's no NMS stage to threshold).
Because `rtdetr_l` is an ultralytics-licensed (AGPL-3.0) checkpoint, its
`.onnx` isn't distributed with the repo — this is also the example that
exercises `python/export_ultralytics.py` and `Engine()`'s `.onnx` auto-build
path (§2.5) end to end, rather than pointing straight at a prebuilt
`.engine` like the other five. Expected output: `r.detections` with COCO
class ids, `d.cls`/`d.score`/`d.x,y,w,h` printed for the first few frames —
a plain single-stage detector loop with no crop step at all.

### 6.5 route_and_reid — Select routing + embedding, cosine matching in Python

```python
person = pycamtrt.Layer("person")
psel = person.add(pycamtrt.Select(det, classes=PERSON_CLASSES))
peng = person.add(pycamtrt.Engine(psel, EMBED_ENGINE,
                                  norm=IMAGENET_NORM, color="rgb"))
person.add(pycamtrt.Postprocess(peng, family="embedding"))

vehicle = pycamtrt.Layer("vehicle")
vsel = vehicle.add(pycamtrt.Select(det, classes=VEHICLE_CLASSES,
                                   min_size=VEHICLE_MIN_SIZE))
veng = vehicle.add(pycamtrt.Engine(vsel, VEHICLE_ENGINE,
                                   norm=IMAGENET_NORM, color="rgb"))
vehicle.add(pycamtrt.Postprocess(veng, family="argmax"))
```

(`examples/route_and_reid/route_and_reid.py`) This is THE showcase for
`Select` (§2.6) and `embedding` (§2.3) together — one detector routed to two
specialist siblings by class id, each running its own engine on only the
crops that matched. The consumer loop then does its own cosine matching, in
plain Python, entirely on the *survivors* tier:

```python
v = np.asarray(vec, dtype=np.float32)
v /= np.linalg.norm(v) + 1e-9
sims = np.stack(gal) @ v          # cosine against a per-stream gallery deque
```

Expected output: `MATCH person s0 frame N ~ s1 (cos 0.97X)` lines — the same
person detected on two cameras, matched purely from 512-d vectors this
script keeps in a small per-stream `deque` gallery. (The demo's honest
caveat: resnet18-on-ImageNet was never trained for re-ID, so treat the
matches as illustrative of the *mechanism*, not a production re-ID model —
this repo's own live app, `fleet_demo` (§6.7), deliberately stays
detect-plus-track only, after a real cross-camera re-ID model's measured
precision/recall tradeoffs made it the wrong story to tell live.)

### 6.6 zone_filter — Python-on-results, no raw tensor in sight

```python
detect = pycamtrt.Layer("detect")
eng = detect.add(pycamtrt.Engine(streams, DETECT_ENGINE))
detect.add(pycamtrt.Postprocess(eng, family="yolo", score=0.4))
pipe = pycamtrt.Pipeline(streams, layers=[detect], skip=1, max_frames=0,
                         backpressure="drop_oldest")
```

(`examples/zone_filter/zone_filter.py`) The graph is one plain detector
layer; the actual lesson is the consumer loop, not the graph — this is the
worked example for §2.3's "tier 1" custom postprocessing: point-in-polygon
zone occupancy counting over `r.outputs["detect"]`, entirely on the
consumer thread, using nothing but the already-compact detection list (no
shapely, ~20 lines of ray-casting). It also demonstrates the
`backpressure="drop_oldest"` + `pipe.dropped_results()` pattern from §3.1:
"a live-video consumer should skip stale frames rather than lag behind."
Expected output: `ZONE 'right' ENTER`/`LEAVE` transition lines plus final
per-zone counts — the one example in the suite with byte-identical output
between Python and C++ runs (a fixed first-60-results window against a
fresh farm start).

### 6.7 fleet_demo — a minimal live app

Every example above is a mechanics demo run to completion and summarized.
`fleet_demo` is a long-running service instead: up to 65 AIC22 street
cameras, detect → track → draw, with the library's three capacity dials on
sliders. It is deliberately small — one Python file, one HTML page, one
`run.sh` — so the whole path from a slider to a rebuilt pipeline fits on a
screen. Its graph, straight from `demo.py`:

```python
def build_pipeline(n, skip, decode):
    cams = MANIFEST[:n]
    streams = pycamtrt.Streams([f"{RTSP}/{c['rtsp_path']}" for c in cams])
    detect = pycamtrt.Layer("detect")
    eng = detect.add(pycamtrt.Engine(streams, ONNX, max_batch=MAX_BATCH))
    detect.add(pycamtrt.Postprocess(eng, family="yolo", score=SCORE))
    return pycamtrt.Pipeline(streams, layers=[detect], skip=skip, decode=decode,
                             backpressure="drop_oldest", ring_seconds=0)
```

Every choice traces back to a section above. `MAX_BATCH = 32`, pinned
regardless of N, is §2.5's rebuild-cost lesson: every generation targets
the same cached engine, so every rebuild is warm (about a second at any N
from 4 to 65). `skip` and `decode` are the pipeline-wide dials of §2.1, set
once per build from the sliders. `hold_frames` stays at its default `False`
(§3.3): nothing here reads pixels. There are no sinks: the tiles play the
publishers' own mediamtx paths over WebRTC (§4.2 explains why not the
relay), so `ring_seconds=0` opts out of the packet ring and its per-packet
copy entirely. `backpressure="drop_oldest"` (§3.2) keeps
the consumer from ever stalling the GPU thread, and the stats line shows
`dropped_results()` so a lagging consumer is visible rather than hidden.

**Consumption is in-process** (§4.3): `consume()` runs inside the real
`for r in pipe:` loop, keeps only vehicle classes, feeds them through a
per-camera greedy-IoU tracker, and publishes one frame event per result —
boxes, classes, track ids, and a wall-clock stamp the page uses to pair each
displayed video frame with its detections. Only the displayed cameras'
events go over SSE; tracking still runs for all N.

**The rebuild** is §2.7's "streams fixed at construction" pattern, live:
`POST /config` stores the new dials, then calls `pipe.stop()` from the web
thread. That ends the consumer thread's `for r in pipe:`; the thread loops
around, builds the next generation from the same cached engine, starts it,
and the request returns. The page then reloads itself — fresh WebRTC and
SSE connections are the whole reconnect strategy. The pipeline object and
its stream count are swapped as one `Gen` object, so the stats thread can
never read a new N against an old pipeline.

**Measured** (RTX 3060 Ti, read off the demo's own stats line): 4 cameras
at 10 fps decode at demand with GPU 1.65 ms per frame against 2.05
predicted; 16 cameras at `skip=2` infer 80 results/s from 160 decoded fps;
all 65 at `decode="key"` hold real time at 17.5 decoded fps; all 65 at
`decode="all"` are over capacity, and the stats line shows both the
prediction saying so and the measured decoded rate falling short of the
650 fps demand. Every rebuild took about a second.

## 7. Gotchas and reference appendix

**Streams are fixed at construction.** No API changes a running
`Pipeline`'s source list — rebuild, as §2.7/§6.7 show, by constructing a new
`Streams`+`Layer` graph and a new `Pipeline`, stopping the old one only
after the new one is running if you want zero-downtime relay paths.

**The 640-input assumption is gone, but its history is worth knowing.**
Earlier internal builds silently assumed a 640×640 detector input; a
non-640 engine caused silent wrong inference and an out-of-bounds GPU write
before that was fixed. Since v0.2.0, detector input size is read straight
from the engine (640/416/1280/non-square are all first-class) — but the
`recommend()` cost model (§5) is still measured at 640×640 specifically;
non-640 nets are measured separately (416 ≈ 0.57×, 1280 ≈ 3.25× the 640
baseline, per the README) but not yet folded into the model itself.

**Engine builds are per-GPU and not always cheap.** A cached hit loads in
low single-digit seconds; a genuinely new batch profile is a full TensorRT
build — the measured **219s** in §2.5/§6.7 is not a fluke, it's what
happens whenever `max_batch` (or `shape`) changes to a value never built
before on this GPU. Plan any auto-scaling logic (like
`max_batch=max(4, len(sources))`) around that asymmetry.

**File inputs are real but younger than the RTSP path.** MP4/H.264/HEVC
files became first-class sources in v0.2.0 (an inline AVCC→Annex-B
bitstream filter handles the container framing NVDEC's parser needs, with
clean end-of-file semantics distinct from a live-camera disconnect — see
`CHANGELOG.md`). Every measured number in this manual and the README,
though, is against live RTSP — the stream farm, every shipped example, and
the entire capacity atlas all target `rtsp://`. Treat file input as
correct per the changelog, but the less-exercised of the two paths if you
hit something unexpected.

**`hold_frames` sizes `ring_depth`, not the other way around.** A consumer
holding `N` frames of one stream needs `ring_depth >= N + 1`
(§3.3) — undersizing it doesn't crash, it silently stalls that stream's
producer. Release with `r.release_frame()` (or let every copy of `r` be
garbage collected) as soon as you're done reading, every iteration,
unconditionally.

**Everything runs inside the `tensorrt-dev` container.** The host needs
only the NVIDIA driver and docker; there's no supported bare-metal build —
see `BUILD.md` for the Dockerfile, the `NVIDIA_DRIVER_CAPABILITIES=video`
requirement, and the full trap list (missing `libnvcuvid.so` symlink, a
stale `apt` index on a fresh container, a CUDA/driver version mismatch).

**Two ways to get `import pycamtrt` working, don't mix them up.**
`pip install .` inside the container needs no `PYTHONPATH` afterward — the
compiled module lives inside the installed package. The classic
`cmake`/`make` dev workflow needs
`PYTHONPATH=/workspace/build:/workspace/python` set explicitly, and needs
cmake's `-DPython3_EXECUTABLE=` to point at the *exact* interpreter you'll
run with (a mismatch here is BUILD.md's "trap #2" — the module silently
builds against the wrong Python and refuses to import). Every example in
this manual assumes one of these two is already working.

**`Pipeline(verify=True)` is a debugging dial, not a production one.** It
turns on a per-frame CPU-reference recomputation the GPU result is diffed
against (`r.verify_ok`/`r.verify_cpu_dets`) — invaluable while you're
onboarding a new engine or chasing a suspected postprocess bug (every
example's `--verify` flag in §6 uses it this way), but it's extra
CPU/GPU-sync work paid on every single frame. Leave it off once a graph is
trusted; none of the README's measured latency numbers include it.

**`Pipeline.__del__` calls `stop()` for a reason you'll only notice if you
skip it.** `stop()` is idempotent and safe to call from any thread, and
`Pipeline` calls it from its own `__del__` as a deadlock guard: pybind11
runs the C++ destructor *while holding the GIL*, and if a `log=` callback
happens to be mid-call on a worker thread at that exact moment, that thread
is blocked waiting for the GIL the destructor also needs — a real deadlock
if the destructor's own thread-joins didn't go through the GIL-releasing
`stop()` binding first. You don't need to think about this if you always
`stop()` explicitly (or use `with pycamtrt.Pipeline(...) as pipe:`) — it's
worth knowing about mainly because it explains why a `Pipeline` object with
a `log=` callback set is safe to simply let fall out of scope instead.

**Known scope limits worth knowing before you design around them** (see the
README's own "Known limitations" for the full, current list): single GPU;
cascades are depth-2 only (one detector, N sibling children — never a child
of a child); no persistent object tracker in the library itself (a per-frame
detection list, not tracks — `zone_filter.py`'s "occupancy, not per-object
identity" design note in §6.6 is a direct consequence); layer-0 engines need
a TensorRT profile with `kMIN` batch 1.

### Quick reference: every public class

| class | constructor | what it is |
|---|---|---|
| `Streams` | `Streams(urls)` | the graph's root — a list of URLs / `Stream` objects |
| `Stream` | `Stream(url, skip=None, decode=None)` | per-camera override of the two load dials |
| `Process` | `Process(input)` | optional, explicit preprocess stage |
| `Select` | `Select(input, classes=None, min_score=None, min_size=None)` | declarative per-child detection filter |
| `Engine` | `Engine(input, engine_path, norm=None, color=None, max_batch=16, fp16=True, shape=None)` | runs a TensorRT engine (`.engine`, or auto-built from `.onnx`) |
| `Postprocess` | `Postprocess(input, family, score=0.4, iou=0.45)` | decodes an `Engine`'s raw tensor (six families, §2.3) |
| `Layer` | `Layer(name, sahi=None)` | a named, ordered group of steps; `.add(step)` |
| `Sink` | `Sink(input, kind, target, stream=None)` | downstream NDJSON/RTSP tee that never backpressures the pipeline |
| `Pipeline` | `Pipeline(streams, layers, **kwargs)` | compiles and runs the graph; iterate with `for r in pipe:` |
| `Result` | *(returned, not constructed)* | one frame's compact output — `.outputs`, three-tier frame access |
| `recommend` | `recommend(streams, resolution="720p", fps=30, skip=1, decode="all", key_gop=30, sahi=None)` | capacity-planning estimate; returns a `Recommendation` |
