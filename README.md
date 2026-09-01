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
    for r in pipe:                                  # ~4 ms/frame at 16 cameras
        print(r.stream_id, r.outputs["read"])
```

Python **describes** the processing graph; a fused C++/CUDA data plane **executes** it. No
Python code ever runs in the per-frame path — pixels go RTSP → NVDEC → inference → GPU
postprocess entirely in VRAM, and only compact results (boxes, text, labels; ~KB/frame)
cross into Python.

## Measured (RTX 3060 Ti, 720p H.264, full detect→OCR cascade)

| | end-to-end / frame | at 16 cameras |
|---|---|---|
| Naive Python (OpenCV + torch) | 12.0 ms | process-per-camera, ~1.3 GB RAM each |
| Modern conventional stack (ultralytics + PaddleOCR) | 13.6 ms | collapses at 4 cameras (~86 fps ceiling, GPU 40% idle) |
| NVIDIA DeepStream (latency-tuned, fp16) | 6.8 ms post-decode | holds |
| **PyCamTRT** | **2.2 ms** (1 cam) | **4.8 ms, full 480 fps offered rate held, sinks on** |

Endurance: 3.5 h soak, 16 streams — 3,023,750 results, zero drops, 3 MB RSS drift.
Every feature in this repo ships with a verification gate (bit-exact CPU references where
possible); run them via `python/qa_matrix.py` and the `src/*_test.cpp` checkpoints.

## Features

- **Any TensorRT model** (clean ONNX→engine export is the only contract): per-engine
  normalization/color knobs, four postprocess families (`yolo`, `yolo-e2e` NMS-free heads,
  `ctc`, `argmax`) — and a documented recipe for adding your own (`docs/ADDING_A_FAMILY.md`).
- **Depth-2 cascade trees**: one detector feeding N sibling recognizers (OCR + classifier
  simultaneously), crops cut from full-res frames in VRAM.
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
driver + docker). Prerequisites: an image with CUDA 12.x + TensorRT 10.x + OpenCV dev
(`tensorrt-dev` in our setup), the NVIDIA Video Codec SDK unpacked into
`third_party/Video_Codec_SDK/` (free download; its license forbids us redistributing it),
and Python ≥3.11 with numpy for the binding.

```bash
docker run --rm --gpus all --network host -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v $PWD:/workspace -w /workspace tensorrt-dev bash -c "
  apt-get update -qq && apt-get install -y -qq pkg-config libavformat-dev libavcodec-dev libavutil-dev git
  ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
  mkdir -p build && cd build
  cmake -DPython3_EXECUTABLE=$(which python3) -DPython_EXECUTABLE=$(which python3) ..
  make -j"
# engines: build once per GPU from the committed .onnx exports, e.g.:
#   trtexec --onnx=models/yolov8n_plates_dynamic.onnx --fp16 \
#     --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:16x3x640x640 \
#     --saveEngine=models/yolov8n_plates_b1-16_fp16_sm86.engine
# no cameras? tools/stream_farm/farm.sh simulates an RTSP fleet from a recorded clip.
# then: PYTHONPATH=build:python python3 examples/read_plates.py rtsp://localhost:8554/cam1
```

## Repository layout

```
src/            C++/CUDA data plane (src/core/ = the pipeline library; *_test.cpp = checkpoints)
src/python/     pybind11 binding (_pycamtrt)
python/         the pycamtrt package + the acceptance matrix (qa_matrix.py) and harnesses
examples/       read_plates, classify_detections, read_and_classify (3-model tree), zone_filter
tools/          RTSP stream farm, SAHI parity checker
docs/           the book (Zero-Copy Vision Part 3), ADDING_A_FAMILY.md; full manual to follow
models/         committed .onnx model exports (engines are per-GPU: build locally, gitignored)
```

**Read `docs/book/CORDERO_BOOK_PART3.txt`** for the full story: architecture, the build, the
benchmarks, the capacity-atlas method, and the roadmap.

## Known limitations (v0.1.0 — honest list)

No object tracker yet (frames between `skip` detections carry no boxes) · no annotated-video
output yet (planned: GPU draw + NVENC sink) · `hold_frames` tier-2/3 frame access has a rare
unresolved exit-time crash (~5%/process; do not rely on it in production) · MP4 *file* inputs
need Annex-B conversion (`ffmpeg -bsf:v h264_mp4toannexb`; RTSP unaffected) · single-GPU ·
depth-2 trees only · all performance constants measured on one GPU (RTX 3060 Ti) — build
your own atlas for other hardware (the book's chapter 7 shows how).

## Provenance

PyCamTRT is the library release of the **CORDERO** research project — a from-the-silicon-up
zero-copy pipeline built step by verified step (frame tracing, GPU postprocess migration,
multi-stream batching, capacity atlases, a DeepStream head-to-head). The development history,
reports, and measurement campaigns live in the CORDERO repository.

Licensed under the MIT License — see `LICENSE`.
