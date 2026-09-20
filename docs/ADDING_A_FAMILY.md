# Adding a compiled postprocess family (M3b)

This is a walkthrough for a competent C++/CUDA contributor who needs to add
a NEW raw-tensor postprocess family to PyCamTRT — a genuinely custom model
head, not something `Postprocess(family="yolo"/"ctc"/"argmax")` already
covers. It uses **Argmax as the worked example**, because that family was
added in M1a *for exactly this purpose*: a template a future family can be
copied from. Every file/symbol named below is real, current code — verify
any of them yourself with `grep`; this document names files and function
symbols, never line numbers (they rot).

If you only want to filter/count/alert on a model's already-decoded
detections (survivors, not the raw tensor), you don't need any of this —
see the package docstring's "Custom postprocessing — two tiers" section
and `examples/zone_filter/zone_filter.py` instead. Section 1 below is the actual
decision test.

## 1. When you need this vs. when Python-on-results suffices

PyCamTRT's custom-processing split is by **tensor size**, not by
preference (a settled design rule from the research repo, "Custom
models -> custom postprocess — split by TENSOR SIZE"):

- **Postprocess that reads the RAW OUTPUT TENSOR** — a model's raw head,
  e.g. `[4+classes, anchors]` for a detector or `[N, classes]` for a
  classifier, before anyone has thrown away the anchors/pixels that
  didn't matter — is **GPU-only, compiled**. This is what this document
  covers. The reason isn't taste: that tensor is megabytes, every frame,
  and Python callback latency (plus the GIL) in that path is exactly the
  per-frame Python involvement PyCamTRT's whole architecture exists to
  avoid (the house rule: "Python = control plane, C++ = data plane. No
  Python in the per-frame path, ever.").
- **Postprocess on the COMPACT RESULT** — the tens-of-survivors detection
  list, OCR strings, or classifier labels a `Postprocess` step has
  *already* produced — is legal, encouraged, ordinary Python on the
  consumer thread. It cannot touch the per-frame GPU path because by
  construction it runs after the GPU thread has moved on. See
  `examples/zone_filter/zone_filter.py` for a full worked example of that tier, and
  the package docstring (`python/pycamtrt/__init__.py`) for the contract.

The litmus test: does your code need every anchor / every pixel of a raw
model output (this document), or does it only need the short list of
things that survived decode+NMS/CTC/argmax (the other tier)? When in
doubt, the "survivors vs. raw tensor" phrasing is the same one both docs
use — it's the one test that matters.

## 2. The contract

`src/postprocess.h`'s own top-of-file comment states the design contract
every family (built-in or yours) must honor, verbatim:

> "Contract kept deliberately minimal so other/experimental models can
> plug in their own decode without architectural change: a plain launch
> function, caller owns every device buffer, everything async on the
> caller's stream. This file is yolov8-family specific in *layout* only
> ... A different model family gets a sibling launcher next to this one,
> not a plugin framework."

Concretely, that means:

- **A plain launcher function**, not a class, not a registered callback,
  not a vtable — a normal C++ function with a stable signature living
  next to the existing launchers in `src/postprocess.h`/`.cu`.
- **Caller-owned device buffers.** Your launcher takes raw device
  pointers the *pipeline* allocated (in `Setup()`) and sized once at
  startup; it never allocates or frees VRAM itself. This is why step (c)
  below happens once, in `Pipeline::Setup`, not per frame.
- **Async on the caller's stream, no host sync inside the launcher.**
  Every existing launcher (`LaunchBoxDecode`, `LaunchNms`,
  `LaunchArgmaxBatched`, …) takes a `cudaStream_t stream` and returns
  immediately; the caller (`GpuLoop`, in `src/core/pipeline.cpp`)
  decides when — and whether — to synchronize before reading results.
  Your kernel launch does the same: queue work on `stream`, return.

This is a **compile-time extension mechanism**, not a runtime plugin
system — see §4 for why that's a deliberate, current-state honesty note,
not an oversight.

## 3. The steps, traced through Argmax (M1a)

Argmax is a plain classifier postprocess: one thread per batch item,
sequential max-logit scan over that item's class row, no softmax (see
`postprocess.h`'s `LaunchArgmaxBatched` WHY-comment for why raw-logit
confidence is the documented contract rather than a probability). Follow
these same seven touch points for a new family.

### (a) Kernel + launcher — `src/postprocess.cu` / `src/postprocess.h`

The header declares the launcher's public contract; the `.cu` file holds
the `__global__` kernel and the launcher that invokes it.

- `src/postprocess.h`: `void LaunchArgmaxBatched(const float* d_logits, int batch, int classes, int* d_labels, float* d_scores, cudaStream_t stream);` —
  declared under the "Classifier family: argmax (M1a)" comment block,
  right after the existing `LaunchBoxDecodeBatched`/`LaunchNmsBatched`
  declarations. Note the shape of the contract: raw device pointers in
  (`d_logits`), raw device pointers out (`d_labels`, `d_scores`), a
  `cudaStream_t` last.
- `src/postprocess.cu`: the `__global__ void ArgmaxKernel(...)` kernel
  (one thread per batch item, sequential scan over `classes`), and the
  `LaunchArgmaxBatched` function that computes the launch geometry
  (`block = 256`, `grid = (batch + block - 1) / block`) and launches it
  on the caller's `stream`.

A new family adds its own kernel + launcher pair here, next to Argmax's —
not a template/generic dispatcher, a sibling.

### (b) Family enum value — `src/core/graph.h`

`enum class Family { YoloDetect, Ctc, Argmax };` — this is the single
enum every layer of the stack threads through (Python string ->
`_FAMILIES` dict -> this enum -> `pybind11` -> `Validate()`/`GpuLoop`
branches). Adding a family means adding one more enumerator here (e.g.
`Family::MyFamily`), plus updating every `switch`/`if`-chain over
`Family` that needs to know about it (steps c/d below are exactly those
chains).

### (c) `Validate()`/`ExecPlan` acceptance + engine I/O-shape check — `src/core/pipeline.cpp` (`Setup`)

Two separate touch points inside `pipeline.cpp`, both inside the
`Pipeline::Impl` construction path:

- **Graph acceptance**, in the function that builds `ExecPlan` from a
  `PipelineConfig` (the `Validate()`-style logic near the layer-1 check):
  `(q.family != Family::Ctc && q.family != Family::Argmax)` is the exact
  guard that currently accepts a layer-1 cascade `Postprocess` step —
  extending this to `q.family != Family::Ctc && q.family != Family::Argmax && q.family != Family::MyFamily`
  (or equivalent) is what lets a graph carrying your family past
  construction instead of hitting `FailGraph("layer 1 step 1 (want
  Postprocess(Ctc|Argmax) fed from the Engine step)")`. `ExecPlan::l1_family`
  (declared right after `ocr_engine_path` in the same struct) is where the
  resolved family value is stashed for `GpuLoop` to read later — set via
  `plan.l1_family = q.family;`.
- **Engine I/O-shape check**, in `Setup()`'s cascade-engine block (the
  `if (!plan.ocr_engine_path.empty())` section): `if (plan.l1_family == Family::Argmax) { ... }`
  validates the OCR/classifier engine's `OutputDims` shape — for Argmax,
  "2D `[N,classes]` (or `[N,classes,1,1,...]` with every trailing dim
  squeezed to 1)", raising `std::runtime_error("argmax engine: want 2D
  output [N,classes] ...")` otherwise; it also allocates the
  family-specific scratch here (`cudaMalloc(&d_argmax_labels, ...)`,
  `cudaMalloc(&d_argmax_scores, ...)`, sized `ocr->MaxBatch()`) — this is
  the ONE place device buffers for your family get allocated, honoring
  §2's "caller owns every device buffer."

### (d) `GpuLoop` cascade-block branch — `src/core/pipeline.cpp`

Inside `GpuLoop`'s "Stage 2 (sequential cascade)" block (same function
that runs the per-batch OCR/classifier chunk loop), `const bool is_argmax
= plan.l1_family == Family::Argmax;` gates two things per chunk:

- **Scratch shape**: `slot_labels[s]`/`slot_label_scores[s]` are resized
  per-slot only `if (is_argmax)`, mirroring `slot_texts[s]` for the `Ctc`
  branch (see the block's own comment: "left as n empty per-slot vectors
  otherwise").
- **Where results are produced + D2H**: after `ocr->Infer(gpu_stream)`,
  `if (is_argmax) { LaunchArgmaxBatched(ocr->OutputPtr(), nc, ocr_classes, d_argmax_labels, d_argmax_scores, gpu_stream); cudaMemcpyAsync(...); cudaMemcpyAsync(...); cudaStreamSynchronize(gpu_stream); ... }`
  is the launcher call site — straight off the engine's own output
  binding (`ocr->OutputPtr()`), D2H only the compact `(label, score)`
  pairs (not the full logits), then one `cudaStreamSynchronize` before the
  CPU loop that writes `slot_labels[owner.first][owner.second]` /
  `slot_label_scores[...]`. This is the ONE call site pattern to copy: your
  launcher call replaces `LaunchArgmaxBatched`'s, your own D2H copies
  replace its two `cudaMemcpyAsync` calls, and you write into your own
  per-slot vectors here instead.

### (e) `FrameResult` field(s) + `bindings.cpp` exposure — `src/core/result.h` / `src/python/bindings.cpp`

- `src/core/result.h`: `std::vector<int> labels; std::vector<float> label_scores;`
  on `FrameResult`, declared right after `texts` with a WHY-comment
  explaining the alignment contract ("`labels[i]`/`label_scores[i]`
  describe `detections[i]`'s stage-2 crop"; populated only when a
  layer-1 `Postprocess(Argmax)` cascade exists, empty otherwise). A new
  family that produces a different SHAPE of per-detection result (not a
  `(label, score)` pair) adds its own field(s) here, with the same
  "empty unless this family's cascade exists" contract.
- `src/python/bindings.cpp`: inside the `py::class_<FrameResult>(m,
  "FrameResult")` block, `.def_readonly("labels", &FrameResult::labels)`
  and `.def_readonly("label_scores", &FrameResult::label_scores)` expose
  those fields read-only to Python, right after `.def_readonly("texts",
  &FrameResult::texts)`. Also update `src/python/bindings.cpp`'s
  `py::enum_<Family>(m, "Family")` block (`.value("Argmax",
  Family::Argmax)`) to add your new enumerator's binding.

### (f) Python: `_FAMILIES` entry + `Result.outputs` mapping — `python/pycamtrt/__init__.py`

- `_FAMILIES = {"yolo": _c.Family.YoloDetect, "ctc": _c.Family.Ctc, "argmax": _c.Family.Argmax}` —
  the string-to-enum table `Postprocess(family=...)` validates against
  (`if family not in _FAMILIES: raise ValueError(...)`). Add your
  family's string here, pointed at the `Family` enumerator from (e).
- `Result.__init__` (the `class Result` wrapping one `FrameResult`): the
  block that builds `self.outputs` reads `l1_argmax` (resolved once at
  `Pipeline` construction — see `self._l1_argmax` in `class Pipeline`,
  derived from `layers[1].steps[-1].family == "argmax"`) to decide
  whether `outputs[layer_names[1]]` is `list(zip(r.labels,
  r.label_scores))` or `r.texts`. A new family with its own result shape
  adds a third branch here (its own `_l1_<family>`-style flag threaded
  the same way, or a small generalization if a second cascade-shaped
  family arrives) so `Result.outputs` hands back the right Python shape
  for your family's payload.

### (g) THE GATE — bit-exact CPU-reference checkpoint (non-negotiable)

This is the house rule ("Nothing is 'done' without its gate
green"; `HANDOFF.md` §5: "Verification as architecture. Every
stage/feature ships with a checkpoint diffing it against a reference,
bit-exact where possible."). No family is done without both of these:

- **`src/postprocess_batch_test.cpp`**: the "M1a: argmax classifier
  family checkpoint" section (search the file for that exact banner
  comment) builds synthetic `[batch, classes]` logits with a *planted*,
  strictly-greater-than-everything-else maximum per row (`kArgmaxBatch =
  37`, `kArgmaxClasses = 1000`, plus two boundary rows forcing the max to
  land at class index 0 and at the last class index — "exercises the
  loop's initial-best and final-iteration edges, not just interior
  hits"), computes a CPU reference (`ArgmaxRef CpuArgmax(const float*
  row, int classes)`), asserts the CPU reference itself agrees with what
  was planted (a construction sanity check, not the kernel test), then
  calls `LaunchArgmaxBatched` and compares GPU vs. CPU labels AND scores
  with **exact equality** (`ax_got_labels[i] == ax_ref[i].label &&
  ax_got_scores[i] == ax_ref[i].score` — no tolerance, because "the GPU
  and CPU read the identical planted float value, no arithmetic is
  performed on it by either side"). Prints `✓ PASS: argmax checkpoint.`
  or `✗ FAIL: ...`. A new family's checkpoint belongs in this same file,
  same shape: planted synthetic inputs (never uniform noise — see the
  file's own comment on why: nondeterministic compaction/ordering makes
  a real comparison meaningless), a from-scratch CPU reference, boundary
  cases, bit-exact comparison.
- **`python/qa_matrix.py` section G** (`section_g_classifier`, printed as
  `[G] M1b classifier cascade (argmax family)`): the Python-level
  end-to-end gate, in two parts —
  - `section_g1_demo` (`[G1]`): runs the real cascade shape
    (`examples/classify_detections/classify_detections.py`'s pipeline — detect layer feeding
    an `Engine`+`Postprocess(family="argmax")` classify layer) for 60
    results and asserts `len(r.outputs["classify"]) == len(r.detections)`
    on every single result (alignment, not correctness).
  - `section_g2_parity` (`[G2]`): a **subprocess**
    (`python/_qa_classifier_parity_subprocess.py`) independently
    recomputes the classifier's argmax label in plain torch/cv2 on the
    SAME live crops (via `r.fetch_frame()` plus the same per-channel
    normalization) and gates on a logit-gap invariant — any label
    disagreement between the pipeline and the reference must be a
    near-tied decision (small logit gap on the reference side), not a
    confident one, or it's flagged as a mechanics bug rather than
    tolerated as model near-ties.
  A new family's qa_matrix section should follow G1's alignment-style
  check at minimum; G2's independent-recompute style is the pattern to
  reach for whenever your family's output can be recomputed in plain
  Python/torch against the same live data — an alignment check alone
  proves shape, not correctness.

## 4. Honesty note: this is v1, on purpose

Everything above is a **recompile-required, compiled-in family**: adding
one means touching seven files/functions and rebuilding
`libpycamtrt_core`/`_pycamtrt` inside the `tensorrt-dev` container. There is no
`.so`-loaded plugin ABI today — no `dlopen`, no stable C ABI boundary for
third-party kernels, no registry a family can drop itself into at
runtime. That's a deliberate, currently-accepted scope limit, not an
oversight: `HANDOFF.md`'s backlog explicitly lists a "custom-postprocess
plugin ABI" as a later item, deferred until the compiled-family path
above proves insufficient in practice (i.e., until real usage shows the
seven-touch-point recompile cost is actually a problem worth a plugin
ABI's added complexity — a stable cross-DSO device-pointer/stream
contract, versioning, ABI compatibility). Argmax (M1a) is the second data
point after Yolo/Ctc that this compiled-family shape scales to a genuinely
different kind of model (detector -> classifier, not detector -> OCR);
until a THIRD or later family finds the recompile cost prohibitive, this
document — not a plugin system — is the supported way to extend PyCamTRT's
raw-tensor postprocess.

---

## 5. Addendum: layer-0 DETECTOR families (learned from yolo-e2e, M4a)

This recipe was written from a layer-1 cascade family (Argmax). The first layer-0 detector
family (`yolo-e2e`, M4a) surfaced these divergences — read this before adding a detector:

- **(c)** there is an `ExecPlan::l0_family` (added in M4a); BOTH layer-0 acceptance forms
  (2-step and 3-step) need widening; think about SAHI compatibility explicitly — if your
  family has no NMS, cross-tile merge is undefined: reject SAHI with a named error.
- **(d)** your branch point is Stage 1 (the layer-0 decode call site in `GpuLoop`), not the
  cascade block. Consider writing directly into `d_kept`/`d_kept_counts` (budget check:
  your max detections ≤ `kMaxNmsCandidates`) instead of allocating scratch.
- **(e)/(f)** usually N/A: a detector's output IS `FrameResult::detections` — generic,
  no new fields, no `Result.outputs` branch.
- **(g)** if your family compacts atomically WITHOUT a later sort (no NMS), the checkpoint
  must canonicalize order itself: plant strictly-distinct scores and sort both sides before
  the bit-exact compare (see the yolo-e2e checkpoint in `postprocess_batch_test.cpp`).
- **Extra touch point the 7-step list missed**: `cfg.verify`'s `CpuReference()` assumes the
  yolo layout. Either write a from-scratch CPU reference for your family or guard verify off
  for it *visibly* (M4a chose the documented no-op) — never leave it silently wrong.
- **Free win**: section A's "valid families" error text updates itself from `_FAMILIES`.

## §6 addendum (report 11 follow-up): the FMA contraction trap

Learned adding the `rtdetr` variant (the first family whose decode multiplies before it
adds): a mul-add shape like `cx - 0.5f * bw` in device code gets CONTRACTED into a fused
multiply-add by nvcc (default `--fmad=true`), which rounds once, not twice — your CPU
reference, compiled without FMA, rounds per-op and diverges in the last ulp. The existing
families never hit this because their decode math is all `(a - b) / c` shapes, which have
no mul-add to contract. Symptoms: the bit-exact checkpoint fails ONLY on geometries whose
intermediate values are inexact (our non-square 802×543 image), with equal counts and
near-equal values.

Fix pattern (see `YoloE2EBatchedKernel`'s `norm_cxcywh` branch): write the device math with
`__fmul_rn`/`__fadd_rn`/`__fsub_rn` intrinsics — CUDA guarantees these are never fused —
so the kernel's op sequence is pinned to exactly what your CPU reference does. Do NOT
"fix" it by making the CPU side use `std::fma` (compiler contraction choices are not a
contract), and do NOT flip `--fmad=false` file-wide (it perturbs codegen for every other
family's already-verified kernels). If your new family's decode multiplies, use the
intrinsics from the start.
