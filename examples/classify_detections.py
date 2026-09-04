#!/usr/bin/env python3
"""classify_detections.py - a two-stage detect -> classify cascade built on
pycamtrt, exercising the M1a "argmax" classifier family end to end.

Stage 1 ("detect") runs the existing YOLOv8 plate detector over each stream.
Stage 2 ("classify") crops every detection and runs an ImageNet-1k
mobilenet_v3_small classifier (see python/export_classifier.py) over the
crop, via `Postprocess(family="argmax")` - plain max-logit argmax, no
softmax (see src/postprocess.h's `LaunchArgmaxBatched`). This mirrors
read_plates.py's detect -> crop -> OCR shape exactly, with the OCR
(`family="ctc"`) stage swapped for a classifier (`family="argmax"`) one.

Note on output semantics (the mechanics-not-fit disclaimer, same spirit as
read_plates.py's LPRNet-charset note): the plate detector finds LICENSE
PLATES, but mobilenet_v3_small classifies into the 1000 ImageNet object
categories (dogs, vehicles, appliances, ...) - none of which is "license
plate". Printed labels will therefore look semantically wrong (e.g. a plate
crop classified as "handkerchief" or "envelope") - that's expected. This
script exercises the CASCADE MECHANICS (crop -> classify -> aligned labels
alongside detections), not a meaningful plate/object classifier; see
examples/classify_detections.py's own PARITY_GATE note below for what IS
being validated numerically.

Normalization (M3a, see python/export_classifier.py's docstring for the
full derivation): pycamtrt's `Engine(norm=...)` now accepts a genuine
PER-CHANNEL (offset, scale) pair - this demo uses TRUE ImageNet
normalization, norm=((-123.675, -116.28, -103.53),
(1/58.395, 1/57.12, 1/57.375)), color="rgb" (index 0 = R). This replaces
M1b's single-scalar approximation (norm=(-114.0, 1/58.6), recorded as a
known limitation in manual/FINDINGS.md) now that the underlying knob is
per-channel; the parity gate in python/qa_matrix.py section G2 uses this
same per-channel norm on both the pipeline's engine and its reference
torch model.

Run it (from the repo root, inside the tensorrt-dev docker container, with
the Python-dev conda env providing the interpreter - see
examples/read_plates.py's docstring for the full canonical incantation):

    docker run --rm --gpus all --network host \\
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \\
        -v $(pwd):/workspace \\
        -v ~/anaconda3:/home/<user>/anaconda3 \\
        -w /workspace tensorrt-dev bash -c '
            ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
            PYTHONPATH=/workspace/build:/workspace/python \\
            ~/anaconda3/envs/Python-dev/bin/python3 \\
            examples/classify_detections.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
        '

The RTSP sources come from tools/stream_farm/farm.sh; plate-shaped content
(so stage 1 actually finds detections to feed stage 2) needs:

    cd tools/stream_farm && CLIP=media/atlas_plate_g30.mp4 ./farm.sh up 2

Requires models/mobilenet_v3s_b1-32_fp16_sm86.engine (build via trtexec from the
committed models/mobilenet_v3s_dynamic.onnx - see python/export_classifier.py's
docstring and BUILD.md's trtexec appendix (or just point Engine() at the .onnx)).
"""
import sys
from collections import Counter

import pycamtrt

DETECT_ENGINE = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
CLASSIFIER = "models/mobilenet_v3s_b1-32_fp16_sm86.engine"

# True per-channel ImageNet normalization (M3a) - see this file's module
# docstring and python/export_classifier.py's derivation.
CLASSIFIER_NORM = ((-123.675, -116.28, -103.53), (1 / 58.395, 1 / 57.12, 1 / 57.375))

# skip=2 (infer every other decoded frame) + max_frames=600 (decoded frames
# per stream) -> ~300 inferred frames/stream, ~20s wall at 30fps source
# (same cadence as read_plates.py).
SKIP = 2
MAX_FRAMES = 600

# Best-effort human-readable ImageNet class names (torchvision ships them
# alongside the pretrained weights) - purely a print-time convenience in
# this CONTROL-PLANE script; falls back to bare numeric indices if
# torchvision isn't importable. Never touches the per-frame path (see
# the house rule: "Python = control plane, C++ = data plane").
def _load_imagenet_categories():
    try:
        import torchvision
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        return weights.meta["categories"]
    except Exception:
        return None


def build_pipeline(urls, verify=False):
    streams = pycamtrt.Streams(urls)

    detect = pycamtrt.Layer("detect")
    eng = detect.add(pycamtrt.Engine(streams, DETECT_ENGINE))
    detect.add(pycamtrt.Postprocess(eng, family="yolo", score=0.4))

    classify = pycamtrt.Layer("classify")
    # Cross-layer edge: an Engine fed a Postprocess step auto-crops each
    # detection from the earlier layer before running the classifier on it
    # (same mechanism read_plates.py uses for its OCR stage).
    ceng = classify.add(pycamtrt.Engine(detect.steps[-1], CLASSIFIER,
                                        norm=CLASSIFIER_NORM, color="rgb"))
    classify.add(pycamtrt.Postprocess(ceng, family="argmax"))

    return pycamtrt.Pipeline(
        streams, layers=[detect, classify], skip=SKIP, max_frames=MAX_FRAMES,
        verify=verify,
    )


def main():
    args = sys.argv[1:]
    verify = "--verify" in args
    urls = [a for a in args if a != "--verify"] or ["rtsp://localhost:8554/cam1"]

    categories = _load_imagenet_categories()

    frames_seen = 0
    dets_seen = 0
    label_counts = Counter()
    verify_ok = 0
    verify_total = 0

    with build_pipeline(urls, verify) as pipe:
        for r in pipe:
            frames_seen += 1
            pairs = r.outputs["classify"]  # [(label:int, raw_logit:float), ...]
            if pairs:
                dets_seen += len(pairs)
                for label, score in pairs:
                    name = categories[label] if categories else str(label)
                    label_counts[name] += 1
                    print(f"s{r.stream_id} frame {r.frame_no}: "
                          f"label={label} ({name!r}) logit={score:.2f}")
            if verify:
                verify_total += 1
                if r.verify_ok:
                    verify_ok += 1

        n_streams = len(urls)
        per_stream = [pipe.get_stream_info(i) for i in range(n_streams)]

    print("\n---- summary ----")
    print(f"frames seen: {frames_seen}  detections classified: {dets_seen}")
    print("label counts:")
    for name, count in label_counts.most_common(15):
        print(f"  {count:5d}  {name!r}")
    for i, si in enumerate(per_stream):
        print(f"s{i}: decoded={si.decoded} reconnects={si.reconnects}")

    if verify:
        print(f"PYTHON VERIFY {verify_ok}/{verify_total}")
        if verify_ok != verify_total:
            sys.exit(1)


if __name__ == "__main__":
    main()
