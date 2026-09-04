# deepstream_bench — the DeepStream side of the CORDERO benchmark

DeepStream 8.0 reference pipeline mirroring CORDERO's cascade
(N RTSP -> batched yolov8n plate detector -> LPRNet OCR on crops), built
from the SAME ONNX files, fed by the SAME mediamtx farm, config-driven
(`deepstream-app`, zero app code). Purpose: the head-to-head measurement
proposed in report 5 §9.

## Layout

- `parsers/` — the only compiled code: two custom nvinfer parsers.
  - `yolov8_parser.cpp` — PGIE bbox parser for the [4+nc, anchors] head;
    our own by decision (same decode math CORDERO verifies bit-exact).
  - `lprnet_parser.cpp` — SGIE classifier parser; wraps the checkpointed
    `src/lprnet_ctc.h` greedy-CTC decoder (NVIDIA's lpr parser targets a
    different charset, so "reuse theirs" bought nothing — deviation from
    the original lean, flagged here).
- `configs/` — nvinfer configs (`pgie_plates.txt`, `sgie_lprnet.txt`) and
  generated app configs.
- `gen_app_config.sh N [prefix]` — emits `configs/app_<prefix><N>.txt`
  for the stream sweep (1/4/8/16), prefix `plate` or `cam`.
- `run_bench.sh <config> [--latency]` — runs deepstream-app in the DS 8.0
  container (host network, repo at /workspace).

## First-time setup

```
# 1. Build the parsers (inside the DS container, repo mounted):
docker run --rm --gpus all -v ~/Desktop/PROYECTO/CORDERO:/workspace \
  nvcr.io/nvidia/deepstream:8.0-triton-multiarch \
  make -C /workspace/bench/deepstream/parsers

# 2. Publish streams (host): mediamtx via tools/stream_farm/farm.sh, then
#    plate publishers: ffmpeg -re -stream_loop -1 \
#      -i tools/stream_farm/media/plate_test.mp4 -c copy -f rtsp \
#      rtsp://localhost:8554/plateN

# 3. Generate a config and run:
./gen_app_config.sh 1 plate
./run_bench.sh app_plate1.txt
```

First run per batch size builds TRT engines from the ONNX (minutes);
they cache next to the ONNX files (gitignored like CORDERO's engines).

## Measurement parity (the methodology contract)

- **Throughput**: `deepstream-app -t` prints per-stream fps; compare to
  CORDERO's inferred/s per stream.
- **Latency**: `--latency` sets NVDS_ENABLE_LATENCY_MEASUREMENT (+
  component-level). DS timestamps buffers at source and reports per-
  component latency; mapping onto CORDERO's pre/queue/gpu splits is the
  open methodological task — validate before quoting numbers.
- **GPU util / VRAM / CPU**: `nvidia-smi dmon` + `pidstat` on the host,
  identical procedure for both systems.
- **Content**: same farm streams; sweep 1/4/8/16; include one
  webcam_60s.mp4 run (plate loop sits below the knee — report 5 caveat).

## Latency-tuned variant

The default configs are DeepStream's idiomatic happy path. For the
latency-matched comparison (see Reports/report 5/FP16_AND_DEEPSTREAM_AUDIT):

```
./gen_app_config_tuned.sh 16 cam         # fp16 GIEs + tuned muxer/jitter
./run_bench.sh app_cam16_tuned.txt 0 --latency   # 0 = run until stopped
```

Tuned config differences (gen_app_config_tuned.sh + *_fp16.txt):
- fp16 GIEs (network-mode=2) — matches CORDERO fp16; biggest lever.
- streammux batched-push-timeout 5 ms (was 33), source jitter 30 ms (100).
First tuned run builds fp16 nvinfer engines (minutes; cached after). Use
duration 0, not a short timeout — the fp16 max-batch-16 engine build
takes longer than 60 s and a short `timeout` SIGINTs it mid-build.

Measured effect (webcam N=16): post-decode latency 33 -> 6.8 ms (5x).

## Audit findings (Reports/report 5/FP16_AND_DEEPSTREAM_AUDIT)

Three bugs found + fixed: parse_latency.py matched only nvv4l2decoder0
(dropped decode for sources >0 at N>1); farm cam8 stale PID (dead stream);
PGIE parser clamped box origin but not extent (edge-clipped boxes). All
parser math, normalization, color format, thresholds, and cascade wiring
audited and confirmed correct. Smoke test re-verified after the parser fix.

## Known parity deviations (all deliberate, all recorded)

1. **Letterbox pad value**: nvinfer pads black; CORDERO pads YOLO's 114
   gray (not configurable in DS). Marginal detector-input difference.
2. **TF32**: no --noTF32 equivalent in nvinfer, so DS LPRNet runs with
   TF32 on. Report-5 data: CTC strings identical under TF32 drift —
   compare strings, not logits.
3. **Batching policy**: DS closes batches on batched-push-timeout
   (33 ms = one frame interval); CORDERO closes when the GPU thread is
   ready. Similar in spirit, not identical — affects latency at low N.
4. **NMS**: nvinfer clusters post-parser (cluster-mode=2, IOU 0.45) on
   CPU; CORDERO's NMS is a GPU kernel. Same algorithm, different
   executor — part of what's being benchmarked, not a flaw.
