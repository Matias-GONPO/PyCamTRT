#!/usr/bin/env python3
"""detect_rtdetr.py - RT-DETR as the detection layer, built on pycamtrt.

One stage ("detect"): an RT-DETR-L engine under ``family="rtdetr"`` - the
NMS-free transformer detector family. Same [N,300,6] head contract as
``yolo-e2e``, but RT-DETR's ultralytics export emits NORMALIZED cx,cy,w,h
box columns (its pixel scaling normally lives in ultralytics' own python
postprocess) - the ``rtdetr`` family does that scaling in the GPU decode
kernel instead. ``score=`` is honored; ``iou=`` is ignored (no NMS stage
exists); SAHI is rejected on this family (cross-tile merge NMS is
undefined for an already-NMS-free head).

Model note: rtdetr_l is an ultralytics-licensed (AGPL-3.0) model, so its
export is NOT distributed in this repo - create it once with
``python/export_ultralytics.py`` (downloads via the ultralytics package
under your own acceptance of its license), then build the engine per
models/README.md. With engine auto-build you can also point ENGINE below
straight at the .onnx.

Run it (full canonical docker incantation in examples/read_plates.py):

    PYTHONPATH=/workspace/build:/workspace/python \\
    ~/anaconda3/envs/Python-dev/bin/python3 \\
    examples/detect_rtdetr.py rtsp://localhost:8554/cam1
"""
import sys
from collections import Counter

import pycamtrt

ENGINE = "models/rtdetr_l_b1-8_fp16_sm86.engine"
SKIP = 2
MAX_FRAMES = 600


def build_pipeline(urls):
    streams = pycamtrt.Streams(urls)
    detect = pycamtrt.Layer("detect")
    eng = detect.add(pycamtrt.Engine(streams, ENGINE))
    detect.add(pycamtrt.Postprocess(eng, family="rtdetr", score=0.5))
    return pycamtrt.Pipeline(streams, layers=[detect], skip=SKIP,
                             max_frames=MAX_FRAMES)


def main():
    urls = sys.argv[1:] or ["rtsp://localhost:8554/cam1"]
    frames = dets = 0
    class_counts = Counter()

    with build_pipeline(urls) as pipe:
        for r in pipe:
            frames += 1
            dets += len(r.detections)
            for d in r.detections:
                class_counts[d.cls] += 1
            if frames <= 5 and r.detections:
                d = r.detections[0]
                print(f"s{r.stream_id} frame {r.frame_no}: cls={d.cls} "
                      f"score={d.score:.2f} box=({d.x:.0f},{d.y:.0f},"
                      f"{d.w:.0f}x{d.h:.0f})")
        per_stream = [pipe.get_stream_info(i) for i in range(len(urls))]

    print("\n---- summary ----")
    print(f"frames: {frames}  detections: {dets}")
    print("top COCO class ids:")
    for cls, count in class_counts.most_common(8):
        print(f"  {count:5d}  cls {cls}")
    for i, si in enumerate(per_stream):
        print(f"s{i}: decoded={si.decoded} reconnects={si.reconnects}")


if __name__ == "__main__":
    main()
