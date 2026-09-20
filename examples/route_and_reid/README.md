# route_and_reid

THE showcase for detection routing (`Select`) and the `embedding` family
together. Stage 1 ("detect") runs a COCO YOLOv8 detector; its detections
are ROUTED by class to two specialist siblings, each a
`Select -> Engine -> Postprocess` layer:

- `Select(classes={0})` (person) → "person": a resnet18 penultimate-feature
  engine (`family="embedding"`) turns each person crop into a 512-d
  vector; this example cosine-matches vectors ACROSS streams - "the person
  on cam1 is the person on cam2".
- `Select(classes={2,3,5,7}, min_size=48)` (car/motorcycle/bus/truck) →
  "vehicle": a mobilenet ImageNet classifier (`family="argmax"`) labels
  the vehicle type.

Routing is exact and strictly reduces work: each sibling only ever infers
the crops its `Select` passes; every other detection keeps that sibling's
empty entry - results stay aligned regardless.

Honest caveats: resnet18-on-ImageNet is a MECHANICS demo for re-ID - never
trained for person re-identification, so treat matches as illustrative.
Likewise the vehicle labels are loose ImageNet stand-ins. The demo farm
clip is an indoor scene: expect person matches and no vehicles there -
point it at real street content to see both branches light up.

## Prerequisites

- `models/yolov8n_b1-16_fp16_sm86.engine` (COCO), 
  `models/resnet18emb_b1-32_fp16_sm86.engine`, and
  `models/mobilenet_v3s_b1-32_fp16_sm86.engine` on disk (see
  `models/README.md` for provenance/licensing and
  `python/export_resnet18_embed.py`/`python/export_classifier.py` for how
  the latter two are exported).
- The stream farm: `cd tools/stream_farm && ./farm.sh up 2` (any content
  works - a COCO detector, not plate-specific; the plate clip happens to
  contain people, which is what exercises the "person" branch below).

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
        examples/route_and_reid/route_and_reid.py \
        rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
    '
```

**C++**:

```bash
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace -w /workspace tensorrt-dev bash -c '
        ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
        ./build/examples/route_and_reid rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
    '
```

**Docker**:

```bash
docker build -t pycamtrt-route-reid examples/route_and_reid
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace pycamtrt-route-reid
```

## Expected output (captured against the 2-cam plate farm)

Python:

```
MATCH person s0 frame 12 ~ s1 (cos 0.975)

---- summary ----
frames: 600  person crops embedded: 538  vehicle crops classified: 0
cross-camera person matches (cos >= 0.85): 532
best match: s1 ~ s0 (cos 0.999)
s0: decoded=600 reconnects=0
s1: decoded=600 reconnects=0
```

C++:

```
MATCH person s1 frame 19 ~ s0 (cos 0.961)

---- summary ----
frames: 600  person crops embedded: 551  vehicle crops classified: 0
cross-camera person matches (cos >= 0.85): 546
best match: s1 ~ s0 (cos 1.000)
s0: decoded=600 reconnects=0
s1: decoded=600 reconnects=0
```

Both runs agree qualitatively: a large majority of person crops match
across streams (cos ≥ 0.85, best match ≈ 1.0 - same stationary person
visible on both cams of the farm clip) and `vehicle crops classified: 0`
(the indoor clip has no vehicles, exactly the documented caveat above).
