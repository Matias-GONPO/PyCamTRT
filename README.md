# PyCamTRT

**Python Camera Fleet TensorRT inference Library** — an open-source, fully-modular library for
zero-copy cascade inference over many concurrent RTSP streams. A Pythonic alternative to
NVIDIA DeepStream: the same class of performance, without the closed internals or the
GStreamer learning curve.

```python
import pycamtrt

streams = pycamtrt.Streams(["rtsp://cam1", "rtsp://cam2"])

detect = pycamtrt.Layer("detect")
eng  = detect.add(pycamtrt.Engine(streams, "models/yolov8n_plates_b1-16_fp16_sm86.engine"))
dets = detect.add(pycamtrt.Postprocess(eng, family="yolo", score=0.4))

read = pycamtrt.Layer("read")                      # cascade: crops each detection on the GPU
oeng = read.add(pycamtrt.Engine(dets, "models/lprnet_b1-32_fp32_sm86.engine"))
read.add(pycamtrt.Postprocess(oeng, family="ctc"))

with pycamtrt.Pipeline(streams, layers=[detect, read], skip=2) as pipe:
    for r in pipe:                                  # ~3 ms/frame at 16 cameras
        print(r.stream_id, r.outputs["read"])
```

Python **describes** the processing graph; a fused C++/CUDA data plane **executes** it. No
Python code ever runs in the per-frame path — pixels go RTSP → NVDEC → inference → GPU
postprocess entirely in VRAM, and only compact results (boxes, text, labels; ~KB/frame)
cross into Python.

## Measured (RTX 3060 Ti, 720p H.264 live RTSP — same hour, same machine, reproducible via `bench/`)

Post-decode latency per frame (batching wait + inference + cascade; the clock-fair figure —
see `bench/METHODOLOGY.md`):

| | workload | 1 cam | 16 cams (480 fps offered) |
|---|---|---|---|
| Naive Python (OpenCV + torch fp32) | detect | 13.1 ms | process-per-camera |
| Modern stack (ultralytics + PaddleOCR) | cascade | 13.6 ms | collapses at 4 cams (~86 fps ceiling) |
| NVIDIA DIY (PyNvVideoCodec + TensorRT, thread/stream) | detect only¹ | 1.8 ms | 2.2 ms |
| DeepStream 8.0, default config | cascade | 3.2 ms | 43.5 ms |
| DeepStream 8.0, latency-tuned fp16 | cascade | 1.8 ms | 4.1 ms |
| **PyCamTRT** | **full cascade** | **1.7 ms** | **2.5 ms** (2.95 end-to-end), GPU 64% busy |

¹ the DIY row runs no OCR cascade (building crop machinery is where 150-line DIY ends), needed
a third-party demuxer for live RTSP, and OOM'd until its engine profile was hand-tuned —
details and raw logs in `bench/`.

Plate-content cascade at 16 cams: PyCamTRT 3.6 ms vs tuned DeepStream 9.0 ms (2.5×).
Endurance: 45-min parallel-cascade soak — 647,867 results, zero drops, flat RSS.
Capacity: ~42 full-rate 720p cams (NVDEC-bound), 100 cams in keyframe mode at 3.7 ms —
the full four-resolution capacity atlas feeds `pycamtrt.recommend()`.
Every feature ships with a verification gate (bit-exact CPU references where possible):
`python/qa_matrix.py` A–L and the `src/*_test.cpp` checkpoints.

## Features

- **Any TensorRT model** (clean ONNX→engine export is the only contract): per-engine
  normalization/color knobs, six postprocess families (`yolo`, `yolo-e2e` NMS-free heads,
  `rtdetr`, `ctc`, `argmax`, and `embedding` raw-vector pass-through for re-ID/feature
  heads) — and a documented recipe for adding your own (`docs/ADDING_A_FAMILY.md`).
  Detector input size is read from the engine (640, 416, 1280 … all first-class).
- **Depth-2 cascade trees**: one detector feeding N sibling recognizers (OCR + classifier
  simultaneously), crops cut from full-res frames in VRAM.
- **Detection routing** (`Select`): declarative, model-agnostic per-child filters —
  `Select(dets, classes={0}, min_size=32)` sends only matching crops to that branch
  (person→re-ID, vehicle→classifier), strictly reducing work; results stay aligned.
- **Engine auto-build**: point `Engine()` at a `.onnx` — the TensorRT engine is built once
  per GPU and cached (no hand-typed trtexec incantations).
- **SAHI tiled inference** (per-layer, pooled cross-camera batching) for small/far objects.
- **Per-camera load dials**: `skip`, `decode="key"` (keyframe-only; the 100-camera mode),
  mixable per stream.
- **Endpoint sinks that cannot backpressure the pipeline**: NDJSON event streams, live RTSP
  republish of the original video (no re-encode — compressed frames are teed host-side),
  pre-roll event clips.
- **Three-tier frame access**: printable VRAM addresses, D2H numpy fetch, and zero-copy
  `__cuda_array_interface__` views consumable by torch/CuPy. *(Tier 2/3 currently
  experimental — see Known limitations.)*
- **`pycamtrt.recommend()`**: capacity planning from a measured atlas — streams × decode rate
  × inference rate → predicted latency and which resource binds first.

## Getting started

Everything builds and runs inside a CUDA/TensorRT container (host needs only the NVIDIA
driver + docker); the provided `Dockerfile` is the zero-thought environment. One manual
prerequisite: the NVIDIA Video Codec SDK unpacked into `third_party/Video_Codec_SDK/`
(free download; its license forbids us redistributing it). **`BUILD.md` is the full build
manual** — short form:

```bash
docker build -t pycamtrt-dev .
docker run --rm --gpus all --network host -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v $PWD:/workspace -w /workspace pycamtrt-dev bash -c "
  mkdir -p build && cd build
  cmake -DPython3_EXECUTABLE=$(which python3) -DPython_EXECUTABLE=$(which python3) ..
  make -j"
# engines build THEMSELVES: point Engine()/--engine at a .onnx and it is
# built once per GPU and cached (models/README.md; trtexec appendix in BUILD.md).
# ultralytics-derived exports are recreated locally: python/export_ultralytics.py
# no cameras? tools/stream_farm/farm.sh simulates an RTSP fleet from a recorded clip.
# then: PYTHONPATH=build:python python3 examples/read_plates.py rtsp://localhost:8554/cam1
```

## Repository layout

```
src/            C++/CUDA data plane (src/core/ = the pipeline library; *_test.cpp = checkpoints)
src/python/     pybind11 binding (_pycamtrt)
python/         the pycamtrt package + the acceptance matrix (qa_matrix.py) and harnesses
examples/       read_plates, route_and_reid (routing + re-ID showcase), detect_rtdetr,
                classify_detections, read_and_classify (3-model tree), zone_filter
tools/          RTSP stream farm, SAHI parity checker
docs/           the book (Zero-Copy Vision Part 3), ADDING_A_FAMILY.md; full manual to follow
models/         .onnx exports + auto-built engine cache — provenance/licenses in models/README.md
bench/          the DeepStream head-to-head harness + METHODOLOGY.md (reproduce our numbers)
BUILD.md        the build manual (Dockerfile route, traps, trtexec appendix)
THIRD_PARTY.md  third-party components and model licensing
```

**Read `docs/book/CORDERO_BOOK_PART3.txt`** for the full story: architecture, the build, the
benchmarks, and the capacity-atlas method (historical narrative through v0.1.0 — later
changes live in `CHANGELOG.md`).

## Known limitations (v0.2.0 — honest list)

No object tracker yet (frames between `skip` detections carry no boxes) · no annotated-video
output yet (planned: GPU draw + NVENC sink) · `hold_frames` tier-2/3 frame access
had a rare exit-time crash in earlier internal builds — not reproduced in v0.2.0 across
140 hunt attempts (see FINDINGS), but the write site was never localized, so treat it as
suppressed rather than proven-fixed · single-GPU ·
depth-2 trees only · layer-0 engines
need a TensorRT profile with kMIN batch 1 (min>1 profiles are rejected with a named error) ·
`pycamtrt.recommend()`'s cost constants were measured at 640×640; non-640 nets run
first-class and their costs are measured (416 ≈ 0.57×, 1280 ≈ 3.25× the 640 baseline) but
not yet modeled per-size ·
all performance constants measured on one GPU (RTX 3060 Ti) — build
your own atlas for other hardware (the book's chapter 7 shows how).

## Roadmap

Object tracker (SORT-lite in the core; predicted boxes on skipped frames) · annotated-video
output (GPU draw + NVENC RTSP sink) · chained cascades (depth-3+: detector-inside-crop, e.g.
car → plate detector → OCR) · multi-GPU · pip packaging · a `--deterministic` fixed-batching
mode for byte-exact file replay.

## Provenance

PyCamTRT is the library release of the **CORDERO** research project — a from-the-silicon-up
zero-copy pipeline built step by verified step (frame tracing, GPU postprocess migration,
multi-stream batching, capacity atlases, a DeepStream head-to-head). The development history,
reports, and measurement campaigns live in the CORDERO repository.

Licensed under the MIT License — see `LICENSE`.
