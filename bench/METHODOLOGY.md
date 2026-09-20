# Benchmark methodology — PyCamTRT vs NVIDIA DeepStream

This folder makes the README's comparison table **reproducible**: pinned
versions, committed configs, the measurement scripts, and this contract.
Run it yourself; challenge it with data.

## What is compared

The same workload on the same machine, same models, same live RTSP content:
a YOLOv8 plate detector (fp16) feeding an LPRNet CTC OCR cascade, at
N = 1/4/8/16 concurrent streams.

- **DeepStream**: `nvcr.io/nvidia/deepstream:8.0-triton-multiarch`
  (DeepStream 8.0, TensorRT 10.x, CUDA 12.x), `deepstream-app` with
  PGIE=plates + SGIE=LPRNet (custom parsers in `deepstream/parsers/`,
  built from source in the container). Two config sets: **default**
  (stock knobs) and **tuned** (fp16 GIEs, streammux batched-push-timeout
  5 ms, RTSP jitter buffer 30 ms, `attach-sys-ts=1`, `sync=0` — the
  latency-oriented settings a competent operator would set; committed as
  `gen_app_config_tuned.sh`).
- **PyCamTRT**: the shipped defaults (`bench/pycamtrt_side.py`), skip=1.

## The clock-start contract (the honesty rule)

The two systems start their latency clocks in different places, and the
naive comparison of "total latency" is therefore WRONG in DeepStream's
disfavor:

- DeepStream's "Frame latency" starts at the SOURCE timestamp and includes
  ~1 GOP of RTSP delivery/jitter buffering that PyCamTRT's clock never
  sees (its clock starts at `PopFrame`, when a decoded frame exists).
- The fair figure is **post-decode latency**: what each pipeline adds
  AFTER a frame is decoded — batching wait + inference + postprocess.

Component mapping (from `deepstream/parse_latency.py`, recorded as an
approximation, not an identity):

```
pre    <- nvv4l2decoder* component latency        (decode)
queue  <- nvstreammux src_bin_muxer latency        (wait to close batch)
gpu    <- primary_gie + secondary_gie_0 latency    (stage1 + stage2)
post-decode = queue + gpu    <=> PyCamTRT's (ready_to_take + take_to_done)
```

Warmup: the first 60 frames per source (~2 s at 30 fps) are discarded on
both sides. GPU utilization is sampled at 1 Hz (`nvidia-smi`) and averaged
after the sampler has fully exited (an earlier in-flight read raced the
file flush and produced NA/garbage — fixed).

## Content

Both content types are served by mediamtx with `-c copy` (zero transcode —
decoder-side identical to real cameras):

- **cam** — a recorded webcam clip looped on `cam1..N`
  (`tools/stream_farm/farm.sh up N`).
- **plate** — a real temporal plate recording looped on `plate1..N`
  (`deepstream/plate_farm.sh up N`). An earlier campaign used a looped
  STILL image here; its keyframe-burst delivery produced bimodal decode
  readings (0.98 ms vs 66.9 ms at different N — a loop/GOP timing
  artifact, not the pipeline) and was discarded. Use real temporal clips
  only.

## Running it

```bash
# 1. content up (host):
cd tools/stream_farm && ./farm.sh up 16          # cam1..16
cd ../../bench/deepstream && ./plate_farm.sh up 16   # plate1..16
# 2. DeepStream side (pulls the DS container; parsers build on first use):
docker run ... # run_bench.sh wraps this - see its header
./run_sweep.sh results/ds_$(date +%F) 60 both
# 3. PyCamTRT side (tensorrt-dev/pycamtrt-dev container, repo mounted):
PYTHONPATH=build:python python3 bench/pycamtrt_side.py \
    results/pycamtrt_$(date +%F).csv 1,4,8,16 cam,plate 900
```

Raw logs, latency splits, GPU CSVs and parsed tables land under
`bench/results/<date>/` — the archive behind any number we publish.

## The dials, head-to-head (19 September 2026)

DeepStream exposes the same two capacity dials PyCamTRT does, so the
validation atlas (`bench/atlas/capacity_atlas.sh <res> <dir> mini`) was run
for both systems on the same farm, GOP content and hour:

- inference rate: PyCamTRT `skip=k` <=> nvinfer `interval=k-1` in the
  `[primary-gie]` group (infer every k-th decoded frame);
- decode rate: PyCamTRT `decode="key"` <=> `intra-decode-enable=1` on every
  `[sourceN]` (the decoder outputs keyframes only, one per GOP).

`bench/deepstream/gen_app_config_dials.sh` writes the latency-tuned config
with both knobs; `bench/deepstream/ds_dials.sh` runs the grid;
`bench/atlas/atlas_vs_deepstream.py` compares and draws the figure.

Two caveats the figure states: DeepStream's post-decode latency under
`interval>0` is a per-buffer median over inferred AND skipped frames, so
latency is compared at full rate only; in keyframe mode its latency line does
not parse and is recorded as NA. Hold on the PyCamTRT side is measured by
`rtsp_infer_multi` from process start (ramp included) and tops out at 96-97 %,
while DeepStream's PERF counter is steady-state; the 95 % threshold absorbs
that.

## Known asymmetries (declared, not hidden)

- DeepStream's SGIE batching, OSD and tracker stages are disabled/absent
  in both its configs and ours — this compares the decode→detect→OCR
  spine only.
- PyCamTRT numbers use its shipped defaults (parallel cascade, engine
  auto-build allowed); DeepStream default numbers use ITS shipped
  defaults — the "tuned" column exists precisely so the comparison is not
  default-vs-expert.
- SAHI tiled inference is EXCLUDED from the comparison: PyCamTRT ships it
  as a per-layer feature; DeepStream has no native equivalent (its
  `nvdspreprocess` ROIs are static rectangles without tile-grid
  generation or cross-tile merge, and `nvmultistreamtiler` is a DISPLAY
  mosaic, not inference tiling). Benchmarking a hand-built DS tiling
  contraption would measure our DS programming, not DeepStream — so both
  sides run whole-frame, and SAHI is priced separately in the capacity
  atlas series.
- Single machine, single GPU (config in the report accompanying each
  results folder).
