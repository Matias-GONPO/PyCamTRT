# read_and_classify

The depth-2 TREE showcase: one detector root ("detect") feeding TWO
SIBLING recognition children that both crop the SAME root detections
directly - "read" (`family="ctc"`, LPRNet OCR, identical engine to
`read_plates/`) and "classify" (`family="argmax"`, mobilenet ImageNet
classifier, identical engine/norm to `classify_detections/`). Neither
child depends on the other's output; that's the whole point of a tree as
opposed to a 3-deep chain (which the executor rejects by name - see
`src/core/pipeline.cpp`'s `Validate()`). This demo reuses both single-child
cascades UNCHANGED, just wired as two siblings under one root - proof the
tree generalization is additive.

Note on output semantics (same mechanics-not-fit disclaimer both parent
demos carry): the LPRNet OCR charset is Chinese (garbled text on
non-Chinese plates is expected), and mobilenet classifies into 1000
ImageNet object categories, none of which is "license plate". This
exercises the TREE MECHANICS - one detection cropped independently by two
unrelated engines, both outputs aligned with the same detection list - not
plate-reading or object-classification correctness.

## Prerequisites

- `models/yolov8n_plates_b1-16_fp16_sm86.engine`,
  `models/lprnet_b1-32_fp32_sm86.engine`, and
  `models/mobilenet_v3s_b1-32_fp16_sm86.engine` on disk (see
  `read_plates/README.md` and `classify_detections/README.md` for how each
  is built/auto-built).
- The stream farm serving plate content:
  `cd tools/stream_farm && CLIP=media/atlas_plate_g30.mp4 ./farm.sh up 2`.

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
        examples/read_and_classify/read_and_classify.py \
        rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
    '
```

**C++**:

```bash
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace -w /workspace tensorrt-dev bash -c '
        ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
        ./build/examples/read_and_classify rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
    '
```

**Docker**:

```bash
docker build -t pycamtrt-read-and-classify examples/read_and_classify
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace pycamtrt-read-and-classify
```

## Expected output (captured against the 2-cam plate farm)

Python:

```
s0 frame 600 det0: '<wan>Y2444' | 919(9.80)

---- summary ----
frames seen: 600  detections: 599  non-empty reads: 599
classify label counts (top 10):
    415  label=919
    184  label=530
s0: decoded=600 reconnects=0
s1: decoded=600 reconnects=0
```

C++:

```
s0 frame 600 det0: '<wan>Y2444' | 919(10.23)

---- summary ----
frames seen: 600  detections: 599  non-empty reads: 599
classify label counts (top 10):
    481  label=919
    118  label=530
s0: decoded=600 reconnects=0
s1: decoded=600 reconnects=0
```

Both runs agree on the plate text (`<wan>Y2444`) and the dominant classify
label (919) - one detection, two independent recognitions, same alignment,
same answers as the two single-child demos running alone.
