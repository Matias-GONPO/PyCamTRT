# zone_filter

A worked example of the "Python on compact results" consumer-side tier
(the OTHER tier from a compiled postprocess family - see
`docs/ADDING_A_FAMILY.md`): per-stream polygon "zones" filter an already-
compact `Result`'s detections, no raw tensor in sight. This code runs on
the consumer thread, on the compact per-frame result the data plane has
already produced (tens of boxes at most) - it can never slow the GPU path.

Attaches "left"/"right" polygon zones per stream (pixel-space, SOURCE-frame
coordinates), counts detections whose box CENTER lands inside each zone,
and prints an ENTER/LEAVE line whenever a zone's occupancy changes. Uses
`backpressure="drop_oldest"` (a live-video consumer should skip stale
frames rather than lag behind) and reports dropped-result counts at the
end.

Content note (per the plate farm's known content - see the Python file's
module docstring): `atlas_plate_g30.mp4`'s plate detection is essentially
STATIC (a stationary vehicle), so no single zone polygon is ever CROSSED
by real motion in this clip - the fallback two-zone split (at `x=557`) is
chosen so the COUNTS DIFFER instead, and the ENTER transition still fires
honestly exactly once per stream (a freshly attached zone's occupancy
starts unoccupied).

## Prerequisites

- `models/yolov8n_plates_b1-16_fp16_sm86.engine` on disk.
- The stream farm serving plate content:
  `cd tools/stream_farm && CLIP=media/atlas_plate_g30.mp4 ./farm.sh up 2`.

`ZONE_FILTER_RESULTS` (env var, default 60) caps how many results this
demo consumes before printing its summary and exiting.

## Run it

**Python**:

```bash
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace -v ~/anaconda3:/home/<user>/anaconda3 \
    -w /workspace tensorrt-dev bash -c '
        ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
        PYTHONPATH=/workspace/build:/workspace/python \
        python3 \
        examples/zone_filter/zone_filter.py \
        rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
    '
```

**C++**:

```bash
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace -w /workspace tensorrt-dev bash -c '
        ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
        ./build/examples/zone_filter rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
    '
```

**Docker**:

```bash
docker build -t pycamtrt-zone-filter examples/zone_filter
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace pycamtrt-zone-filter
```

## Expected output (captured against the 2-cam plate farm)

Python and C++ produce **byte-identical** output for this example (the
first 60 results of a fresh farm start are deterministic - same clip
position, same detections):

```
[s1 f1] ZONE 'right' ENTER (1 detection(s); running count=1)
[s0 f1] ZONE 'right' ENTER (1 detection(s); running count=1)

---- zone counts ----
s0 zone 'left': 0 detection(s) counted, occupied=False
s0 zone 'right': 26 detection(s) counted, occupied=True
s1 zone 'left': 0 detection(s) counted, occupied=False
s1 zone 'right': 34 detection(s) counted, occupied=True
results consumed: 60  dropped (backpressure): 0
```
