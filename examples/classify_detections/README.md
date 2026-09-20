# classify_detections

The same detect → crop → recognize cascade shape as `read_plates/`, with
the OCR (`family="ctc"`) child swapped for an ImageNet-1k classifier
(`family="argmax"`, plain max-logit, no softmax). Exercises true
per-channel normalization (`Engine(norm=((o0,o1,o2),(s0,s1,s2)), color=
"rgb")`) - torchvision's ImageNet mean/std, converted to pycamtrt's
`(pixel+offset)*scale` convention.

Note on output semantics: the plate detector finds LICENSE PLATES, but
`mobilenet_v3_small` classifies into the 1000 ImageNet object categories
(none of which is "license plate"). Printed labels look semantically wrong
(e.g. a plate crop classified as "street sign") - expected. This exercises
the CASCADE MECHANICS (crop → classify → aligned labels), not a meaningful
plate/object classifier.

**Python vs. C++ difference**: the Python script optionally looks up
human-readable ImageNet category names via `torchvision` (a control-plane,
print-time-only convenience - falls back to bare ids if unavailable); the
C++ port always prints bare numeric label ids (no torchvision dependency
in the data plane). The mechanics under test - crop, classify, alignment -
are identical either way.

## Prerequisites

- `models/yolov8n_plates_b1-16_fp16_sm86.engine` and
  `models/mobilenet_v3s_b1-32_fp16_sm86.engine` on disk (build the latter
  via `trtexec` from the committed `models/mobilenet_v3s_dynamic.onnx` -
  see `python/export_classifier.py` and `BUILD.md`'s `trtexec` appendix -
  or just point `Engine()`/the `.cpp`'s `kClassifierEngine` at the `.onnx`
  for auto-build).
- The stream farm serving plate content (so stage 1 has detections to feed
  stage 2): `cd tools/stream_farm && CLIP=media/atlas_plate_g30.mp4
  ./farm.sh up 2`.

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
        examples/classify_detections/classify_detections.py \
        rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
    '
```

**C++**:

```bash
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace -w /workspace tensorrt-dev bash -c '
        ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
        ./build/examples/classify_detections rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
    '
```

**Docker**:

```bash
docker build -t pycamtrt-classify examples/classify_detections
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace pycamtrt-classify
```

## Expected output (captured against the 2-cam plate farm)

Python:

```
s0 frame 600: label=919 ('street sign') logit=9.75

---- summary ----
frames seen: 600  detections classified: 600
label counts:
    460  'street sign'
    140  'digital clock'
s0: decoded=600 reconnects=0
s1: decoded=600 reconnects=0
```

C++ (bare numeric labels, no category names):

```
s0 frame 600: label=919 logit=10.00

---- summary ----
frames seen: 600  detections classified: 599
label counts:
    488  label=919
    111  label=530
s0: decoded=600 reconnects=0
s1: decoded=600 reconnects=0
```

Both runs agree on the **top label** (919, "street sign") against the same
near-tied runner-up (530, "digital clock") - a documented near-tie between
independent bilinear resizers (Python/cv2 vs. the GPU crop kernel), not a
mechanics bug (see `python/_qa_classifier_parity_subprocess.py`'s
`GAP_LIMIT` discussion for the full story).
