#!/usr/bin/env python3
"""export_resnet18_embed.py - torchvision resnet18 with its classifier
head removed -> ONNX [batch, 512] penultimate-feature ("embedding")
export, the `family="embedding"` demo engine source.

House pattern (see export_classifier.py): exports are committed (.onnx -
torchvision weights are BSD-3, redistribution is fine, see
THIRD_PARTY.md), engines are NOT (per-GPU; auto-built from the .onnx or
via BUILD.md's trtexec appendix). This script exists so the committed
file is reproducible from a one-liner.

Model: torchvision resnet18, IMAGENET1K_V1 weights, ``fc`` replaced by
Identity so the output is the 512-d post-avgpool feature vector. Use with
ImageNet normalization, same values as export_classifier.py documents:
norm=((-123.675, -116.28, -103.53), (1/58.395, 1/57.12, 1/57.375)),
color="rgb". This is a MECHANICS demo for the embedding family - a real
re-ID deployment would export a purpose-trained checkpoint (e.g. OSNet)
through this same shape contract ([N, D], 2D).

Run (conda Python-dev env, CPU only):

    ~/anaconda3/envs/Python-dev/bin/python3 python/export_resnet18_embed.py
"""
from pathlib import Path

import torch
import torchvision

OUT = (Path(__file__).resolve().parent.parent / "models" /
       "resnet18_embed_dynamic.onnx")


def main():
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()  # [N,512] penultimate feature out
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, str(OUT),
        input_names=["input"], output_names=["features"],
        dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17, dynamo=False)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
