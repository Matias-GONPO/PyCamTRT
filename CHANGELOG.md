# Changelog

All notable changes to PyCamTRT. The project follows semantic versioning;
every feature listed here shipped with a verification gate (bit-exact CPU
references where possible — see `python/qa_matrix.py` and `src/*_test.cpp`).

## v0.4.0 — first public release (2026-09)

### Added
- **Examples as directories**: each of the six mechanics examples is now
  `examples/<name>/` with the Python script, a C++ port built directly on the
  core (`-DPYCAMTRT_BUILD_EXAMPLES`, ON by default), a README with the run
  commands and a captured expected-output sample, and a convenience Dockerfile.
- **`examples/fleet_demo/`**: a minimal live app — one Python process (pycamtrt
  consumer thread + FastAPI/SSE) and one plain HTML page — that puts the three
  capacity dials (streams, decode rate, inference rate) on sliders over 65
  traffic cameras, draws detection boxes with per-camera track ids, and shows
  measured throughput next to `recommend()`'s prediction. Its dataset prep
  (`tools/prep_aic22.py`) gained `--gop` to re-encode at other keyframe intervals.
- **`MANUAL.md`**: the Python manual — every class and function, how a graph
  compiles into a running pipeline, and a worked tour of the examples.
- README: a versions / OS / requirements section (the exact stack every number
  was measured on).
- **The capacity atlas tooling lives in the repo** (`bench/atlas/`: content build,
  farm, orchestrator with `full`/`mini`/`smoke` grids, resume, comparison and
  analysis). A 96-cell validation atlas on the v0.4.0 decoder
  (`bench/atlas/runs/2026-09-19_mini/`) confirms the new every-frame ceilings
  (50/16/8/4 cameras at 720p/1080p/1440p/4K on the atlas grid) and that the
  keyframe cells and the VRAM wall are unchanged from the v2 atlas.
- **DeepStream on the same dials** (`bench/deepstream/ds_dials.sh`,
  `gen_app_config_dials.sh`: nvinfer `interval` and `intra-decode-enable`):
  capacity is identical at every resolution and inference rate; latency at full
  rate below the wall is 2 to 4 times lower on PyCamTRT. Figure
  `docs/img/dials_vs_deepstream.png` (`bench/atlas/atlas_vs_deepstream.py`).

### Fixed
- **Decoder no longer waits for inference.** `cuvidMapVideoFrame` ran its
  post-processing on the legacy default CUDA stream (the `output_stream` field
  was never set), which synchronizes with the blocking inference stream, and one
  NVDEC context lock was shared by every camera and held during that map. Above
  ~24 cameras at 720p every camera waited for the running TensorRT batch and for
  each other, capping full-inference decode at ~815 fps with the hardware decoder
  half idle. The map now runs on the producer's own non-blocking stream and each
  decoder owns its lock (NVIDIA's sample pattern). Measured the same night against
  DeepStream 8.0: the full-inference ceiling moved from 24 to 48 cameras at 720p
  (end of the grid, NVDEC 97%) and from 16 to 24 at 1080p — the NVDEC wall — with
  post-decode latency equal or lower at every camera count.
- **Deterministic detections.** The same bug let the letterbox kernel, on a
  non-blocking stream, read a mapped surface whose post-processing on stream 0
  had not finished: three runs of the previous build gave three slightly
  different detection sets (sub-pixel jitter). Detections are now byte-identical
  run to run at 1, 4 and 16 streams (`bench/smoke_digest.py`).
- **One frame period less end-to-end latency per camera.** Packets are fed to
  the NVDEC parser with `CUVID_PKT_ENDOFPICTURE` (each demuxed packet is a whole
  access unit), so a picture is decoded when its packet arrives instead of when
  the next packet's start code shows up. Demuxer-to-pop fell from 53 to 19 ms at
  1080p/24 cameras and from 46 to 10 ms at 720p/48; throughput and detections
  unchanged.

### Changed
- The C++ core is namespaced `pycamtrt::` and its CMake target is
  `pycamtrt_core` (previously named after the research project).
- `pip install .` no longer needs `git`: pybind11 is a build requirement in
  `pyproject.toml`, CMake uses an installed pybind11 when the interpreter has
  one and otherwise fetches the pinned release tarball (hash-checked).
- README figures: `ceiling_systems`, `ceiling_latency_pairs`, `ceiling_curves`,
  `max_streams_every_vs_keyframes`, `dials_vs_deepstream` and the hero image are
  built from the 19 September data; the two v2-atlas dial figures whose
  every-frame bars predate the decoder fix were retired.
- The head-to-head numbers in the README come from a same-night five-system
  campaign (`bench/results/ceiling_2026-09-19/`); the capacity atlas behind
  `recommend()` predates the decoder fix and is conservative above the old
  every-frame ceilings.
- All machine-specific paths removed from scripts and docs: the interpreter and
  ffmpeg/ffprobe come from `PYTHON3` / `FFMPEG` / `FFPROBE` (defaulting to
  `PATH`), and dataset locations are explicit arguments.

### Fixed
- `Sink(kind="stream")` relay: a live follower resumes across reconnects by a
  monotonic packet sequence instead of source pts, so a looping publisher no
  longer starves the relay at the loop boundary.
- Per-stream `decoded` counters read from another thread could report 0
  (missing store ordering in the producer loop).

### Known
- A relay's *first* connection replays its ring backlog at realtime pace, so
  relayed video stays a fixed few seconds behind the detections; `fleet_demo`
  plays its sources directly for that reason (see MANUAL §4.2).
- The decoder holds each picture until the next packet arrives (one frame of
  latency at the source frame rate).

### Repository
- History rewritten to drop the ultralytics-derived ONNX exports that the
  v0.1.0 commit had carried; `v0.2.0` and `v0.3.0` tags re-created on the
  rewritten commits.

## v0.3.0 — pip packaging, phase 0 (2026-09)

### Added
- **`pip install .`** (scikit-build-core over the existing CMake build):
  installs the `pycamtrt` package with the compiled `_pycamtrt` module
  inside it — no more `PYTHONPATH=build:python`. Build-time environment
  requirements are unchanged (run it inside the CUDA/TensorRT container,
  see BUILD.md §2b); a prebuilt wheel needing no build environment is a
  later phase.

### Changed
- The binding import is layout-aware: installed packages load
  `pycamtrt._pycamtrt`; the in-tree dev workflow (`PYTHONPATH=build:python`)
  keeps working via fallback.

## v0.2.0 — first public release (2026-09)

### Added
- **`Select` detection routing**: a first-class graph node filtering which
  detections feed each cascade child (`classes=`, `min_score=`, `min_size=`;
  ints only — model-agnostic by construction). Results stay aligned;
  non-routed detections get empty entries. Routing strictly reduces child
  work. qa section L.
- **Engine auto-build**: `Engine("model.onnx", max_batch=, fp16=, shape=)`
  builds the TensorRT engine once per GPU (in-process nvonnxparser) and
  caches it with the house naming; input tensor name discovered
  automatically; stale caches rebuild themselves once.
- **`embedding` postprocess family**: raw `[N,D]` pass-through for
  re-ID/feature heads — per-detection float vectors in
  `Result.outputs`/`ChildOutput.vectors`. Gated bit-exactly against the
  argmax family on the same engine (qa section K).
- **`rtdetr` postprocess family**: RT-DETR's NMS-free `[N,300,6]` head with
  normalized-cxcywh decode, sharing the yolo-e2e kernel (FMA-pinned;
  synthetic bit-exact checkpoint).
- **CP1 cascade parallelism**: sibling cascade children run on their own
  CUDA streams by default (`cascade_serial=True` is the measured A/B
  escape hatch); per-child `ms_gpu` timing on every result.
- **MP4/file inputs**: H.264/HEVC files are first-class deterministic
  sources (AVCC→Annex-B conversion inline; clean EOF completion).
- Examples: `route_and_reid.py` (routing + cross-camera re-ID showcase),
  `detect_rtdetr.py`. Docs: `BUILD.md`, `Dockerfile`, `THIRD_PARTY.md`,
  `models/README.md`, ADDING_A_FAMILY §6 (the FMA-contraction lesson).

### Changed
- **Detector input size is read from the engine** (640/416/1280/non-square
  all first-class) — previously a silent-failure 640 assumption (found as
  a silent wrong-inference + GPU OOB write; fixed in two stages:
  fail-fast, then engine-derived dims).
- Ultralytics-derived ONNX exports removed from the repository (AGPL-3.0
  vs MIT) — recreate locally via `python/export_ultralytics.py`; see
  `THIRD_PARTY.md`.

### Fixed
- Non-640 detector engines: silent wrong inference + out-of-bounds GPU
  write (see Changed).
- min>1 / static-batch TensorRT profiles now rejected with named errors.

### Known issues
- `hold_frames` tier-2/3 frame access remains experimental: a rare
  (~5%/process) exit-time crash from an unlocalized stray write in the
  hold path. Production paths without `hold_frames` are unaffected.

## v0.1.0 — internal baseline (2026-09-01)

The born-verified extraction of the research project's pipeline into this
repository: zero-copy RTSP→NVDEC→TensorRT→GPU-postprocess data plane,
declarative Python graph API, depth-2 cascade trees (`yolo`/`yolo-e2e`
detectors, `ctc`/`argmax` children), SAHI tiled inference, per-stream load
dials, endpoint sinks (NDJSON events, RTSP relay, pre-roll clips),
three-tier frame access, `pycamtrt.recommend()` capacity planning, and the
full acceptance matrix. History and measurement campaigns live in the
the research project's repository.
