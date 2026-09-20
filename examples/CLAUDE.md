# examples/ — agent guide

You're helping a user run or modify one of the PyCamTRT examples: six
matched library-mechanics examples, plus a seventh, full app example.
Each of the six (`read_plates/`, `classify_detections/`, `read_and_classify/`,
`detect_rtdetr/`, `route_and_reid/`, `zone_filter/`) has a `.py`, a `.cpp`
(a direct `pycamtrt::Pipeline` port - identical step graph, not a different
example), a `README.md` (what it shows + the three run commands + a
captured expected-output sample), and an optional `Dockerfile`. Read the
target example's own `README.md` first - it has the exact copy-paste
commands and what success looks like; this file is the cross-cutting
context that doesn't fit there.

The seventh, `fleet_demo/`, is different on purpose: a minimal live app
(one Python process = pycamtrt consumer thread + a small FastAPI/SSE
server, plus one plain `index.html`) that puts the library's three capacity
dials - N streams, decode rate, inference rate - on sliders over all 65
AIC22 cameras, with detection boxes and track ids drawn in the browser and
measured performance next to `recommend()`'s prediction. **Python-only, no
C++ port**, no database, no zones/alerts, its own `run.sh up|down|status`
lifecycle script instead of a bare `docker run` line. Sections below that
reference "the six examples" or their `.py`/`.cpp`/docker-run pattern don't
apply to it as-is - see its own `README.md`; the one-line pointer is in the
run matrix below.

## 1. Prerequisites checklist

- **NVIDIA GPU + driver** recent enough for CUDA 12.x (`nvidia-smi` should
  work on the host). Everything else runs inside docker.
- **The docker image**: build the root `Dockerfile` (`docker build -t
  pycamtrt-dev .`) or use any CUDA 12.x + TensorRT 10.x image with
  `nvinfer`/`nvonnxparser`, OpenCV dev, and ffmpeg dev installed - the
  repo's own canonical invocations name the image `tensorrt-dev`. See
  `BUILD.md` §1-2.
- **NVIDIA Video Codec SDK**: unpack so `third_party/Video_Codec_SDK/
  Interface/` exists (free download, NVIDIA account required - its
  license forbids redistributing it, so it is never baked into any image
  here). Without it, `pycamtrt_core`/`rtsp_infer_multi`/the Python binding/every
  C++ example all fail to configure - `CMakeLists.txt` gates on this path.
- **Engines**: point `pycamtrt.Engine()` (Python) or a `.cpp`'s
  `kSomeEngine` constant at a `.onnx` and it builds+caches a TensorRT
  engine once per GPU automatically (see `models/README.md` and
  `BUILD.md` §3) - no hand-typed `trtexec` required, though the appendix
  exists if you want it. Most example engines already exist under
  `models/*.engine` in a checked-out repo; if one is missing, check the
  per-example `README.md`'s Prerequisites section for which `.onnx` to
  point at (some upstream models, e.g. `rtdetr_l`, are ultralytics-
  licensed and NOT distributed - `models/README.md` has the full table).
- **The simulated camera farm** (`tools/stream_farm/farm.sh`): `cd
  tools/stream_farm && ./farm.sh up 2` serves a recorded clip as
  `rtsp://localhost:8554/cam1`/`cam2`. Plate-content examples
  (`read_plates`, `classify_detections`, `read_and_classify`,
  `zone_filter`) need `CLIP=media/atlas_plate_g30.mp4 ./farm.sh up 2`
  instead of the default webcam loop - set `CLIP` BEFORE `up`, not after.
  `./farm.sh down` tears it down; `./farm.sh status` checks it.

## 2. Run matrix

Every command below assumes cwd = repo root. The Python/C++ rows still
need the `docker run ... tensorrt-dev bash -c '...'` wrapper unless you're
already inside a container shell - copy it from the target example's own
`README.md` (or `BUILD.md`) rather than retyping it here.

| example | Python | C++ (after `cmake`/`make`) | Docker |
|---|---|---|---|
| read_plates | `python3 examples/read_plates/read_plates.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `./build/examples/read_plates rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `docker build -t pycamtrt-read-plates examples/read_plates && docker run --rm --gpus all --network host -v $PWD:/workspace pycamtrt-read-plates` |
| classify_detections | `python3 examples/classify_detections/classify_detections.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `./build/examples/classify_detections rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `docker build -t pycamtrt-classify examples/classify_detections && docker run --rm --gpus all --network host -v $PWD:/workspace pycamtrt-classify` |
| read_and_classify | `python3 examples/read_and_classify/read_and_classify.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `./build/examples/read_and_classify rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `docker build -t pycamtrt-read-and-classify examples/read_and_classify && docker run --rm --gpus all --network host -v $PWD:/workspace pycamtrt-read-and-classify` |
| detect_rtdetr | `python3 examples/detect_rtdetr/detect_rtdetr.py rtsp://localhost:8554/cam1` | `./build/examples/detect_rtdetr rtsp://localhost:8554/cam1` | `docker build -t pycamtrt-rtdetr examples/detect_rtdetr && docker run --rm --gpus all --network host -v $PWD:/workspace pycamtrt-rtdetr` |
| route_and_reid | `python3 examples/route_and_reid/route_and_reid.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `./build/examples/route_and_reid rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `docker build -t pycamtrt-route-reid examples/route_and_reid && docker run --rm --gpus all --network host -v $PWD:/workspace pycamtrt-route-reid` |
| zone_filter | `python3 examples/zone_filter/zone_filter.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `./build/examples/zone_filter rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `docker build -t pycamtrt-zone-filter examples/zone_filter && docker run --rm --gpus all --network host -v $PWD:/workspace pycamtrt-zone-filter` |
| **fleet_demo** (app example - see note above) | `cd examples/fleet_demo && ./run.sh up` (then browse `http://localhost:8000/`); `./run.sh status` / `./run.sh down` | *(no C++ port)* | `run.sh` drives the bind-mounted `tensorrt-dev` image; no per-example Dockerfile |

Building the C++ ports: `mkdir -p build && cd build && cmake
-DPython3_EXECUTABLE=$(which python3) .. && make -j <example-name>`
(binaries land in `build/examples/<name>`; all six are `ON` by default via
`-DPYCAMTRT_BUILD_EXAMPLES`, gated on the Video Codec SDK like `pycamtrt_core`).

`fleet_demo` ports (mediamtx RTSP/WHEP and the page+SSE server):
`8554`/`8889`/`8000` - see `examples/fleet_demo/README.md`. Its
prerequisites (dataset prep via `tools/prep_aic22.py` for all 65 AIC22
cameras, `models/yolov8n_dynamic.onnx`, the `tensorrt-dev` image, the
stream farm's `mediamtx` binary) are also its own README's job, not this
file's - don't duplicate them here.

## 3. Troubleshooting

| symptom | fix |
|---|---|
| `libnvcuvid.so` not found at link/run time | `ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so` inside the container - the driver mounts the versioned `.so.1` at CONTAINER run time, not image build time, so this needs to be (re-)run per `docker run`, not baked in once. |
| `import pycamtrt` / `import _pycamtrt` fails | Either cmake used a different Python than you're running (`-DPython3_EXECUTABLE=` must match; BUILD.md trap #2), or you forgot `PYTHONPATH=/workspace/build:/workspace/python`. Alternative that sidesteps both: `python3 -m pip install .` inside the container (BUILD.md §2b) - installs `pycamtrt` into site-packages, no `PYTHONPATH` needed afterward. |
| RTSP connect refused / times out | The farm isn't up (`cd tools/stream_farm && ./farm.sh status`) or you forgot `--network host` on `docker run` (the farm listens on the HOST's localhost). |
| Detector finds nothing on a plate example | Farm was started without `CLIP=media/atlas_plate_g30.mp4` - the default `webcam_60s.mp4` has no plates. Bring the farm down and back up with the override. |
| Missing `.engine` file | Point `Engine()` (Python) or the example's `kSomeEngine`-adjacent `.onnx` (C++ - see the CMakeLists/README for which `.onnx`) straight at the `.onnx` instead; it auto-builds and caches per-GPU on first use (`models/README.md`, `BUILD.md` §3). Some `.onnx` exports are not distributed (ultralytics AGPL models) - see `models/README.md`'s provenance table and `python/export_ultralytics.py`. |
| `apt-get install` fails / stale package index in a fresh container | Fresh `tensorrt-dev`-style containers need `apt-get update -qq` BEFORE any `apt-get install` - the base image's index is stale. Do this once per fresh container, before the `libavformat-dev`/etc. install line in the canonical incantation. |
| `cmake` reports "Could NOT find Python3 (missing: Development ...)" on a bare `nvcr.io/nvidia/tensorrt` image | The base image ships a python3 interpreter but not its headers/lib (`python3-dev`/`libpython3.NN-dev`) - install that package first (the root `Dockerfile` doesn't; every `examples/*/Dockerfile` does, for exactly this reason). Once a build directory has cached a NOTFOUND result for this, re-running `cmake` in the SAME directory won't retry even after installing the package - delete/recreate that build directory (this is why the example Dockerfiles use their own dedicated `.docker-build/`, never BUILD.md's `build/`, which a dev checkout may have already configured against a different Python). |
| TensorRT/CUDA prints `This container was built for NVIDIA Driver Release ... but version ... was detected` and inference segfaults with `CUDA initialization failure` | The pinned base image tag's CUDA toolkit is newer than this host's NVIDIA driver supports (check `nvidia-smi`'s reported `CUDA Version` against what the container's own banner demands). This is a HOST/image-tag compatibility issue, not a code bug - it reproduces identically with a bare, unmodified upstream image and affects the root `Dockerfile` on the same host exactly as it does any example's `Dockerfile` (same `FROM` line). Fixing it means either updating the host driver or the owner deliberately re-pinning the base image tag - flag it, don't silently change the pin yourself. |
| CMake configures but skips `_pycamtrt`/examples | Check the `Video Codec SDK found at ...` line in the `cmake` output - if it says "not found", `third_party/Video_Codec_SDK/Interface/` is missing (see §1 above); nothing under that gate (including all six example binaries) will build. |

## 4. Rules for you (the agent)

- **Never run `git commit`/`git push`** - the owner commits personally.
  Uncommitted changes across the repo are normal and expected.
- **Verify before declaring success**: run the example, then compare its
  output against the "Expected output" sample in that example's own
  `README.md` - same dominant plate string / top label / class mix / zone
  counts, not necessarily byte-identical counts (RTSP timing is not fully
  deterministic across separate farm runs - `zone_filter` is the one
  exception that IS byte-identical, since it only consumes the farm's
  first 60 results). A run that produces no detections at all on a
  plate-content example almost always means the farm was started without
  the `CLIP=` override (§3 above), not a code bug.
- Match repo conventions if you edit a `.cpp` here: `kConstants`,
  WHY-comments, and keep these files linear/pedagogical (no factoring into
  helper headers) - see any existing example `.cpp` for the house style.
