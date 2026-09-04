# Changelog

All notable changes to PyCamTRT. The project follows semantic versioning;
every feature listed here shipped with a verification gate (bit-exact CPU
references where possible — see `python/qa_matrix.py` and `src/*_test.cpp`).

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

The born-verified extraction of the CORDERO research pipeline into this
repository: zero-copy RTSP→NVDEC→TensorRT→GPU-postprocess data plane,
declarative Python graph API, depth-2 cascade trees (`yolo`/`yolo-e2e`
detectors, `ctc`/`argmax` children), SAHI tiled inference, per-stream load
dials, endpoint sinks (NDJSON events, RTSP relay, pre-roll clips),
three-tier frame access, `pycamtrt.recommend()` capacity planning, and the
full acceptance matrix. History and measurement campaigns live in the
CORDERO research repository.
