#!/usr/bin/env python3
"""read_and_classify.py - M4b showcase: a depth-2 TREE with one detector
root feeding TWO SIBLING recognition children, built on pycamtrt. See
python/pycamtrt/__init__.py's module docstring ("Depth-2 tree" section) for
the general shape this demonstrates concretely.

Root layer ("detect"): the same YOLOv8 plate detector read_plates.py and
classify_detections.py already use.

Sibling children - BOTH crop the SAME root detections directly (neither
depends on the other's output; that is the whole point of a depth-2 tree,
as opposed to a 3-deep chain, which v1 does not execute - see
``Validate()``'s named RuntimeError in pipeline.cpp):

  - "read"      (family="ctc")    - LPRNet OCR. Identical engine to
                                    read_plates.py's single-child cascade.
  - "classify"  (family="argmax") - mobilenet_v3_small ImageNet-1k
                                    classifier, true per-channel ImageNet
                                    norm + RGB. Identical engine/norm to
                                    classify_detections.py's single-child
                                    cascade.

This demo deliberately reuses BOTH of those single-child cascades
UNCHANGED, just wired as two siblings under one root - proving the tree
generalization (M4b) is additive: nothing about read_plates.py's or
classify_detections.py's own pipeline shape had to change for this to work.

Note on output semantics (same mechanics-not-fit disclaimer both parent
demos carry): the LPRNet OCR charset is Chinese (garbled text on non-Chinese
plates is expected - see read_plates.py), and mobilenet_v3_small classifies
into 1000 ImageNet object categories, none of which is "license plate" (see
classify_detections.py). This script exercises the TREE MECHANICS - one
detection cropped independently by two unrelated engines, both outputs
aligned with the same detection list - not plate-reading or object-
classification correctness.

Run it (see read_plates.py's docstring for the full canonical docker
incantation)::

    docker run --rm --gpus all --network host \\
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \\
        -v $(pwd):/workspace \\
        -v ~/anaconda3:/home/<user>/anaconda3 \\
        -w /workspace tensorrt-dev bash -c '
            ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
            PYTHONPATH=/workspace/build:/workspace/python \\
            ~/anaconda3/envs/Python-dev/bin/python3 \\
            examples/read_and_classify.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
        '

The RTSP sources come from tools/stream_farm/farm.sh; plate-shaped content
(so the root actually finds detections to feed both children) needs:

    cd tools/stream_farm && CLIP=media/atlas_plate_g30.mp4 ./farm.sh up 2

Requires both models/lprnet_b1-32_fp32_sm86.engine and
models/mobilenet_v3s_b1-32_fp16_sm86.engine on disk (see read_plates.py's and
classify_detections.py's docstrings for how each is built).
"""
import sys
from collections import Counter

import pycamtrt

DETECT_ENGINE = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
READ_ENGINE = "models/lprnet_b1-32_fp32_sm86.engine"
CLASSIFIER_ENGINE = "models/mobilenet_v3s_b1-32_fp16_sm86.engine"

# True per-channel ImageNet normalization (M3a) - copied verbatim from
# examples/classify_detections.py (kept in sync by hand, same convention as
# qa_matrix.py's CLASSIFIER_NORM).
CLASSIFIER_NORM = ((-123.675, -116.28, -103.53), (1 / 58.395, 1 / 57.12, 1 / 57.375))

# skip=2 (infer every other decoded frame) + max_frames=600 (decoded frames
# per stream) -> ~300 inferred frames/stream, ~20s wall at 30fps source
# (same cadence as read_plates.py/classify_detections.py).
SKIP = 2
MAX_FRAMES = 600


def build_pipeline(urls, verify=False):
    streams = pycamtrt.Streams(urls)

    # Root: the plate detector. `root` (its Postprocess step) is the SHARED
    # input both sibling children below crop from - never each other's
    # output (that would be a 3-deep chain, which v1's executor rejects by
    # name - see pipeline.cpp's Validate()).
    detect = pycamtrt.Layer("detect")
    eng = detect.add(pycamtrt.Engine(streams, DETECT_ENGINE))
    root = detect.add(pycamtrt.Postprocess(eng, family="yolo", score=0.4))

    # Sibling 1: OCR ("read", ctc) - crops `root`'s detections.
    read = pycamtrt.Layer("read")
    reng = read.add(pycamtrt.Engine(root, READ_ENGINE))
    read.add(pycamtrt.Postprocess(reng, family="ctc"))

    # Sibling 2: classifier ("classify", argmax) - ALSO crops `root`
    # directly (not `read`'s output).
    classify = pycamtrt.Layer("classify")
    ceng = classify.add(pycamtrt.Engine(root, CLASSIFIER_ENGINE,
                                        norm=CLASSIFIER_NORM, color="rgb"))
    classify.add(pycamtrt.Postprocess(ceng, family="argmax"))

    return pycamtrt.Pipeline(
        streams, layers=[detect, read, classify], skip=SKIP,
        max_frames=MAX_FRAMES, verify=verify,
    )


def main():
    args = sys.argv[1:]
    verify = "--verify" in args
    urls = [a for a in args if a != "--verify"] or ["rtsp://localhost:8554/cam1"]

    frames_seen = 0
    dets_seen = 0
    read_nonempty = 0
    classify_counts = Counter()
    verify_ok = 0
    verify_total = 0

    with build_pipeline(urls, verify) as pipe:
        for r in pipe:
            frames_seen += 1
            texts = r.outputs["read"]      # list[str], aligned with detections
            pairs = r.outputs["classify"]  # list[(label, score)], same alignment
            dets_seen += len(r.detections)
            for i in range(len(r.detections)):
                text = texts[i] if i < len(texts) else ""
                label, score = pairs[i] if i < len(pairs) else (-1, 0.0)
                if text:
                    read_nonempty += 1
                classify_counts[label] += 1
                print(f"s{r.stream_id} frame {r.frame_no} det{i}: "
                      f"{text!r} | {label}({score:.2f})")
            if verify:
                verify_total += 1
                if r.verify_ok:
                    verify_ok += 1

        n_streams = len(urls)
        per_stream = [pipe.get_stream_info(i) for i in range(n_streams)]

    print("\n---- summary ----")
    print(f"frames seen: {frames_seen}  detections: {dets_seen}  "
          f"non-empty reads: {read_nonempty}")
    print("classify label counts (top 10):")
    for label, count in classify_counts.most_common(10):
        print(f"  {count:5d}  label={label}")
    for i, si in enumerate(per_stream):
        print(f"s{i}: decoded={si.decoded} reconnects={si.reconnects}")

    if verify:
        print(f"PYTHON VERIFY {verify_ok}/{verify_total}")
        if verify_ok != verify_total:
            sys.exit(1)


if __name__ == "__main__":
    main()
