# fleet_demo

The smallest PyCamTRT live app: up to 65 AIC22 traffic cameras through one YOLOv8n
detector, per-camera IoU tracking, and boxes with track ids drawn in a browser. Its
only purpose is to put the library's three capacity dials on sliders and show what
they do to measured performance next to `pycamtrt.recommend()`'s prediction:

| dial | slider | what it changes |
|---|---|---|
| **N streams** | 1..65 | the first N cameras of `data/manifest.json` are decoded and inferred; the page shows the first 16 |
| **decode rate** | all / key | every frame, or keyframes only: one decode per keyframe interval (GOP), i.e. one every 3 s with the default 30-frame transcode; the option label shows the actual rate |
| **inference rate** | skip 1..10 | infer every k-th *decoded* frame; with keyframes only that is every k-th keyframe, i.e. one inference per 3k seconds per camera |

Apply rebuilds the pipeline (warm: the TensorRT engine is cached) and the page reloads.
Between inferences a tile keeps its last boxes; once they are more than a second old
they are dimmed and the tile prints their age, so at keyframes-only with skip 10 you
see boxes refresh every 30 s rather than boxes that look stuck.

## Files

```
demo.py        one process: pycamtrt consumer thread + 1 Hz stats thread + FastAPI/SSE server
index.html     plain page: WebRTC (WHEP) tiles straight from mediamtx, canvas overlay, 3 sliders, stats line
run.sh         up | down | status
tools/         prep_aic22.py (one-time transcode of the dataset), publish_aic22.sh (mediamtx + publishers)
data/          manifest.json + aic22_h264/*.mp4 (generated, gitignored, ~14 GB)
```

No database, no zones or alerts, no reconnect or error handling: if a tile is black,
reload the page.

## Prerequisites

1. The `tensorrt-dev` docker image (root `Dockerfile`) and an NVIDIA GPU.
   `run.sh` and the publishers take `python3` and `ffmpeg` from `PATH` unless `PYTHON3=` and
   `FFMPEG=` say otherwise; the interpreter must be the one the extension in `build/` was
   built with (BUILD.md). If they point at the wrong tools the symptoms are silent: the 65
   publishers print a pid and exit at once, and the demo container disappears (`--rm`) on
   its first import. `PYTHON3=/path/to/python3 FFMPEG=/path/to/ffmpeg ./run.sh up` is the
   safe form.
2. `models/yolov8n_dynamic.onnx`: `python3 python/export_ultralytics.py yolov8n`.
   The first run builds and caches `models/yolov8n_dynamic_b1-32_fp16_sm86.engine`.
3. The AIC22 / CityFlowV2 clips transcoded to H.264 (the raw dataset is not part of the repo):
   `python3 tools/prep_aic22.py --scenarios S01,S02,S03,S04,S05,S06 --dataset-root <CityFlowV2 dir>`
   writes `data/manifest.json` and `data/aic22_h264/`.
   `--gop N` sets the keyframe interval of the transcode (default 30 frames = one keyframe
   every 3 s at 10 fps) and records it per camera in the manifest; the demo reads it for the
   keyframes-only label and for `recommend(key_gop=...)`. Re-running with a different `--gop`
   re-encodes the selected scenarios (a few minutes for S01, hours for all 65 cameras) and
   replaces the clips in place, so stop the publishers first (`./run.sh down`).
   **Only re-encode to imitate a camera setting you actually expect**: real IP cameras set
   their own keyframe interval (typically 1 to 2 s, configurable per stream), so an
   intermediate decode rate on a real fleet is a camera setting, not a dataset change. A
   short GOP raises bitrate and file size; a long one makes keyframes-only very sparse.
4. mediamtx at `tools/stream_farm/bin/mediamtx` (the stream farm's own binary).

## Quickstart

```bash
cd examples/fleet_demo
./run.sh up              # mediamtx + 65 publishers, then the demo container
docker logs -f fleet_demo   # wait for "ready gen 1" (about a minute: apt in the container + engine load)
# open http://localhost:8000/
./run.sh down
```

Ports: 8554 RTSP (publishers at `aic/<slug>`; the engine reads them and the browser plays
the same paths over WebRTC), 8889 WHEP, 8000 the page + SSE.

## The stats line

`decoded X fps of Y demanded` sums every stream's decoded-frame counter per second
against N x 10 fps. `inferred /s` counts results consumed. `GPU ms/frame measured` is an
EMA of `ms_take_to_done / batch_size`, comparable with `recommend()`'s
`predicted_gpu_ms_per_frame`. `dropped` is `Pipeline.dropped_results()` (the consumer
falling behind under `backpressure="drop_oldest"`). `GPU %` and `NVDEC %` come from
`nvidia-smi`. The second line is `recommend(streams=N, resolution="1080p", fps=10, skip, decode)`:
whether the configuration holds real time, the headroom on each resource, which one binds,
and what to turn.

## How Apply works

`POST /config` stores the new dials, stops the running pipeline from the web thread
(`Pipeline.stop()` is safe from any thread and ends the consumer's `for r in pipe` loop),
the consumer thread builds the next generation from the same cached engine, and the
request returns when it is running. The page then reloads: fresh WebRTC and SSE
connections are the whole reconnect strategy.

## Known limitations

- The AIC22 clips are 10 fps, so boxes update 10 times a second; that is the source, not the pipeline.
- The tiles play the publishers' streams directly, not the engine's `Sink(kind="stream")`
  relay. Measured 2026-09-12: the relay's first connection replays its ring backlog at
  real-time pace, which left the relayed video a fixed 7.4 s behind the detections (74
  frames, matched by re-detecting cars in screenshots). With direct streams the boxes
  land within about one frame of the video; that residual is the decoder's own latency
  and is the next thing to tune.
- The engine processes all N cameras but only the first 16 are relayed and displayed.
- `max_batch=32` is pinned so every rebuild is warm; above 32 streams the C++ core logs a
  chunk warning once per build and proceeds.
- `recommend()` is calibrated on an RTX 3060 Ti.
- The tracker is per-camera greedy IoU with no motion model; ids restart on every rebuild.
  With `decode=key` each detection is 3 s apart, longer than the 1.5 s track age-out, so
  every detection gets a new id.
- Each `up` spends 30-60 s installing libav inside the container.
- A second browser tab keeps drawing over a frozen video after another tab's Apply; reload it.
