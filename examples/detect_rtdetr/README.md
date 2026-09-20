# detect_rtdetr

RT-DETR as the detection layer: one stage ("detect"), an RT-DETR-L engine
under `family="rtdetr"` - the NMS-free transformer detector family. Same
`[N,300,6]` head contract as `family="yolo-e2e"`, but RT-DETR's
ultralytics export emits NORMALIZED `cx,cy,w,h` box columns (its pixel
scaling normally lives in ultralytics' own Python postprocess) - the
`rtdetr` family does that scaling in the GPU decode kernel instead.
`score=`/`score_thresh` is honored; `iou=`/`iou_thresh` is ignored (no NMS
stage exists); SAHI is rejected on this family at construction (cross-tile
merge NMS is undefined for an already-NMS-free head).

## Prerequisites

- `models/rtdetr_l_b1-8_fp16_sm86.engine` on disk. `rtdetr_l` is an
  ultralytics-licensed (AGPL-3.0) model, so its `.onnx` export is **not
  distributed** in this repo - create it once with
  `python/export_ultralytics.py` (downloads via the `ultralytics` package
  under your own acceptance of its license), then build the engine per
  `models/README.md` (or point `Engine()`/the `.cpp`'s `kEngine` straight
  at the `.onnx` for auto-build).
- The stream farm - any content works (this is a general COCO-class
  detector, not plate-specific): `cd tools/stream_farm && ./farm.sh up 1`.

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
        examples/detect_rtdetr/detect_rtdetr.py rtsp://localhost:8554/cam1
    '
```

**C++**:

```bash
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace -w /workspace tensorrt-dev bash -c '
        ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
        ./build/examples/detect_rtdetr rtsp://localhost:8554/cam1
    '
```

**Docker**:

```bash
docker build -t pycamtrt-rtdetr examples/detect_rtdetr
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace pycamtrt-rtdetr
```

## Expected output (captured against the plate farm, cam1 only)

Python:

```
s0 frame 10: cls=0 score=0.91 box=(34,343,467x351)

---- summary ----
frames: 300  detections: 963
top COCO class ids:
    300  cls 0
    278  cls 39
    232  cls 62
    144  cls 72
      9  cls 46
s0: decoded=600 reconnects=0
```

C++:

```
s0 frame 10: cls=62 score=0.91 box=(687,478,371x239)

---- summary ----
frames: 300  detections: 1008
top COCO class ids:
    300  cls 0
    283  cls 39
    266  cls 62
    148  cls 72
     10  cls 46
      1  cls 28
s0: decoded=600 reconnects=0
```

Both runs agree on the dominant class mix (0=person, 39=bottle, 62=tv,
72=refrigerator - the plate clip is an indoor scene, so RT-DETR sees
furniture/indoor objects, not plates; per-frame detection lists differ in
detail run-to-run since box order isn't guaranteed stable, but the class
histogram shape matches).
