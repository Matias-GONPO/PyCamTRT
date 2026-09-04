#!/usr/bin/env python3
"""export_classifier.py - torchvision mobilenet_v3_small -> ONNX (dynamic
batch), the M1b classifier-cascade demo's stage-2 engine source.

House pattern: exports are committed (.onnx), engines are NOT (gitignored,
built per-GPU - auto-built from this .onnx by Engine(), or via BUILD.md's
trtexec appendix).

Model: torchvision's pretrained mobilenet_v3_small (ImageNet-1k, 1000
classes), used AS-IS (no forward-graph surgery needed - unlike LPRNet, this
model has no batch-coupling ops (BatchNorm running stats, not batch
reductions) and exports cleanly).

Normalization (M3a - upgrade of the earlier M1b scalar-approximation
limitation): pycamtrt's Engine(norm=...) knob now accepts a genuinely
PER-CHANNEL (offset, scale) pair - see python/pycamtrt/__init__.py's Engine
docstring. torchvision's ImageNet preprocessing is per-channel
(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], RGB order, on a 0..1
float image); this demo (and the M1b classifier-cascade demos generally -
see examples/classify_detections.py, python/qa_matrix.py's CLASSIFIER_NORM,
python/_qa_classifier_parity_subprocess.py) now uses the TRUE per-channel
values, converted to pycamtrt's own (pixel+offset)*scale form on a 0..255
pixel:
    offset[c] = -mean[c] * 255   -> [-123.675, -116.28, -103.53]
    scale[c]  = 1 / (std[c] * 255) -> [1/58.395, 1/57.12, 1/57.375]
i.e. norm=((-123.675, -116.28, -103.53), (1/58.395, 1/57.12, 1/57.375)),
color="rgb" (index 0 = R, matching the mean/std order above). This
replaces the earlier scalar approximation (single averaged offset/scale
applied to all 3 channels, e.g. norm=(-114.0, 1/58.6)) - a recorded M1b
finding (manual/FINDINGS.md's "M1 model-generality findings" section) that
per-channel norm (M3a) was expected to improve. Real accuracy against
labeled ImageNet data is still not measured here (no labeled eval set in
this demo); what M3a's qa_matrix.py section G2 measures instead is
agreement between the pipeline's GPU classifier and a plain torch/cv2
reference given the SAME (now per-channel) normalization on both sides.

Usage (conda Python-dev env; no docker/TensorRT needed for this step - pure
torch/torchvision/onnx):
    /home/matiasu/anaconda3/envs/Python-dev/bin/python3 \\
        python/export_classifier.py

Writes models/mobilenet_v3s_dynamic.onnx (input "images" [N,3,224,224] float32,
output "logits" [N,1000] float32, dynamic batch) to the repo root, matching
the existing models/yolov8n_dynamic.onnx / models/lprnet_dynamic.onnx naming convention.
"""
import os

import torch
import torchvision

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "models/mobilenet_v3s_dynamic.onnx")


def main():
    torch.manual_seed(0)

    weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = torchvision.models.mobilenet_v3_small(weights=weights).eval()
    print(f"loaded mobilenet_v3_small ({weights}), "
          f"{sum(p.numel() for p in model.parameters())} params")

    x1 = torch.randn(1, 3, 224, 224)
    x4 = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        y1 = model(x1)
        print(f"batch-1 sanity: output shape {tuple(y1.shape)}, "
              f"argmax class {y1.argmax(dim=1).item()}")
        assert tuple(y1.shape) == (1, 1000)

        # Batch-decoupling sanity (same spirit as the LPRNet export's check in the CORDERO repo,
        # cheaper here since mobilenet_v3_small has no batch-coupled ops to
        # begin with - this just confirms that fact rather than fixing
        # anything): slot 0's output must be identical regardless of its
        # batchmates.
        a = model(torch.cat([x1, x4[:3]]))[0]
        b = model(torch.cat([x1, torch.randn(3, 3, 224, 224)]))[0]
        d = (a - b).abs().max().item()
        print(f"batch decoupling (same slot, different batchmates): "
              f"max abs diff {d:.2e}")
        assert d == 0.0, "unexpected batch coupling in mobilenet_v3_small"

    torch.onnx.export(
        model, x1, OUT,
        input_names=["images"], output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # legacy exporter (torch 2.12's dynamo default needs onnxscript)
    )
    print(f"ONNX written: {OUT} (input Nx3x224x224, output Nx1000, "
          f"dynamic batch)")


if __name__ == "__main__":
    main()
