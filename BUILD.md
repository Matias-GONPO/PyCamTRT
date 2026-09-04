# Building PyCamTRT

Everything builds and runs inside a CUDA/TensorRT container; the host needs
only the NVIDIA driver and docker (with the NVIDIA container toolkit). This
is the full build manual — the README's Getting started is the short form.

## 1. Prerequisites

- Host: NVIDIA driver (recent enough for CUDA 12.x), docker + NVIDIA
  container toolkit.
- **NVIDIA Video Codec SDK** (free download, NVIDIA account required — its
  license forbids us redistributing it): unpack so that
  `third_party/Video_Codec_SDK/Interface/` exists. Without it, only the
  non-decode targets build.
- A container image with CUDA 12.x + TensorRT 10.x + build tools. Use the
  provided `Dockerfile`:

```bash
docker build -t pycamtrt-dev .
```

(or keep your own equivalent image — the project only assumes `nvinfer`,
`nvonnxparser`, OpenCV dev, ffmpeg dev, cmake; the repo's canonical
container invocations name the image `tensorrt-dev`.)

## 2. Build (C++ core, CLI, checkpoints, Python binding)

```bash
docker run --rm --gpus all --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v $PWD:/workspace -w /workspace pycamtrt-dev bash -c "
  ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
  mkdir -p build && cd build
  cmake -DPython3_EXECUTABLE=$(which python3) -DPython_EXECUTABLE=$(which python3) ..
  make -j"
```

Notes:
- `NVIDIA_DRIVER_CAPABILITIES` **must include `video`** or the container
  won't get `libnvcuvid` at all (trap #0).
- The `ln -sf` line exists because containers ship `libnvcuvid.so.1` but no
  unversioned linker symlink (trap #1). The Dockerfile pre-creates it, but
  driver-mounted setups may need it per-run.
- **GPU architecture**: defaults to `sm_86` (RTX 3060 Ti). Other GPUs:
  `cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..` (Ada), `=120` (Blackwell), etc.
- **Python binding**: cmake's `-DPython3_EXECUTABLE=` must point at the
  interpreter you will RUN with, or pybind11 silently builds against the
  container's default python and the module won't import (trap #2). To use
  a host conda env, mount it (`-v ~/anaconda3:/home/<user>/anaconda3`) and
  pass its python path; numpy must be installed there.
- Run Python with `PYTHONPATH=/workspace/build:/workspace/python`.

## 3. Models and engines

The repo commits redistributable `.onnx` exports; ultralytics-derived ones
are recreated locally (`python/export_ultralytics.py` — see
`models/README.md` and `THIRD_PARTY.md`). Engines are per-GPU and build
themselves: point `pycamtrt.Engine()` (or `--engine`/`--ocr`) at a `.onnx`
and the engine is built once and cached next to it
(`Engine("models/foo.onnx", max_batch=16, fp16=True, shape=(640, 640))` —
`shape=` only matters for exports with symbolic H/W).

<details><summary>trtexec appendix (manual builds)</summary>

```bash
trtexec --onnx=models/yolov8n_plates_dynamic.onnx --fp16 \
  --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 \
  --maxShapes=images:16x3x640x640 \
  --saveEngine=models/yolov8n_plates_b1-16_fp16_sm86.engine
```

The input tensor name differs per exporter (`images` for ultralytics,
`input` for the torchvision exports here) — `trtexec` will tell you if you
guess wrong; the auto-builder discovers it itself.
</details>

## 4. No cameras? The stream farm

`tools/stream_farm/farm.sh` replays a clip as N independent RTSP cameras
(mediamtx + ffmpeg `-c copy` loops — decoder-side identical to real
cameras at ~zero CPU):

```bash
cd tools/stream_farm
mkdir -p bin media
# fetch mediamtx (MIT) into bin/ from https://github.com/bluenviron/mediamtx/releases
# put any H.264/H.265 clip at media/webcam_60s.mp4 (record your own - no
# clips ship with the repo)
./farm.sh up 4        # rtsp://localhost:8554/cam1 .. cam4
./farm.sh down
```

## 5. Verify the build (gates)

```bash
# inside the container, from /workspace:
./build/postprocess_batch_test                       # bit-exact GPU-vs-CPU checkpoints
./build/rtsp_infer_multi <clip.mp4> --frames 200 --verify   # per-frame CPU reference
PYTHONPATH=build:python python3 python/qa_matrix.py  # full acceptance matrix
```

## 6. Known traps, collected

| symptom | cause / fix |
|---|---|
| `libnvcuvid.so` not found at link time | trap #1: create the symlink (see §2) |
| decoder init fails / no NVDEC in container | trap #0: `NVIDIA_DRIVER_CAPABILITIES` missing `video` |
| `import _pycamtrt` fails | trap #2: cmake used a different python than you run (§2) |
| piped container stdout appears frozen | docker pipes are block-buffered — prefer log files/markers over live greps |
| relative paths break | binaries expect cwd = repo root (`/workspace`) |
| engine builds/loads fail after a TensorRT upgrade | delete the cached `.engine`; auto-built caches rebuild themselves once automatically |

> **Sanity check after configuring**: the cmake output must say
> `CMAKE_BUILD_TYPE: Release` (the default since v0.2.0). An empty build
> type compiles without optimization — functionally correct, up to ~45×
> slower on sparse workloads, and it looks fine on dense ones (we learned
> this the hard way; see the project FINDINGS ledger).
