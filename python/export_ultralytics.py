#!/usr/bin/env python3
"""export_ultralytics.py - recreate the ultralytics-derived ONNX exports.

WHY this script exists instead of committed .onnx files: ultralytics
licenses its models (yolov8n, yolo26n, rtdetr-l, ...) under AGPL-3.0,
which conflicts with redistributing their exports inside this MIT repo -
so PyCamTRT ships this RECIPE, not the files (see THIRD_PARTY.md).
Running it downloads the checkpoints through the ultralytics package
under YOUR OWN acceptance of its license and writes the same exports the
examples/docs reference into models/. Engines then build per-GPU either
automatically (point Engine() at the .onnx - see EngineBuilder.h) or via
the trtexec appendix in BUILD.md.

Exports produced (all dynamic=True: batch AND spatial dims symbolic; the
auto-builder pins H/W at build time - 640x640 unless shape= says
otherwise):

    models/yolov8n_dynamic.onnx        yolov8n.pt        COCO detector
    models/yolov8n_416_dynamic.onnx    yolov8n.pt @416   (imgsz=416)
    models/yolov8n_1280_dynamic.onnx   yolov8n.pt @1280  (imgsz=1280)
    models/yolo26n.onnx                yolo26n.pt        NMS-free e2e head
    models/rtdetr_l_dynamic.onnx       rtdetr-l.pt       family="rtdetr"

Run (conda Python-dev env, no GPU needed - exports on CPU):

    ~/anaconda3/envs/Python-dev/bin/python3 python/export_ultralytics.py
    ~/anaconda3/envs/Python-dev/bin/python3 python/export_ultralytics.py rtdetr

With no argument every export above is produced; pass any of
yolov8n / yolov8n_416 / yolov8n_1280 / yolo26n / rtdetr to export just
those.
"""
import shutil
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"

# name -> (checkpoint, imgsz, output filename, ultralytics class)
EXPORTS = {
    "yolov8n": ("yolov8n.pt", 640, "yolov8n_dynamic.onnx", "YOLO"),
    "yolov8n_416": ("yolov8n.pt", 416, "yolov8n_416_dynamic.onnx", "YOLO"),
    "yolov8n_1280": ("yolov8n.pt", 1280, "yolov8n_1280_dynamic.onnx",
                     "YOLO"),
    "yolo26n": ("yolo26n.pt", 640, "yolo26n.onnx", "YOLO"),
    "rtdetr": ("rtdetr-l.pt", 640, "rtdetr_l_dynamic.onnx", "RTDETR"),
}


def export(name):
    ckpt, imgsz, out_name, cls_name = EXPORTS[name]
    import ultralytics  # deferred: give the no-package error a clear line

    model_cls = getattr(ultralytics, cls_name)
    print(f"[{name}] loading {ckpt} (downloads to the ultralytics cache "
          f"on first use, under the ultralytics AGPL-3.0 license)")
    model = model_cls(ckpt)
    onnx_path = model.export(format="onnx", imgsz=imgsz, dynamic=True,
                             device="cpu")
    dst = MODELS / out_name
    shutil.move(str(onnx_path), dst)
    print(f"[{name}] -> {dst}")


def main():
    names = sys.argv[1:] or list(EXPORTS)
    unknown = [n for n in names if n not in EXPORTS]
    if unknown:
        sys.exit(f"unknown export(s) {unknown}; valid: {list(EXPORTS)}")
    for n in names:
        export(n)


if __name__ == "__main__":
    main()
