# models/ — ONNX exports and per-GPU engines

Two kinds of files live here, with different rules:

- **`.onnx` exports** — portable model graphs. The redistributable ones are
  committed; the ultralytics-derived ones are NOT (license, see below) and
  are recreated locally with one command.
- **`.engine` files** — TensorRT engines, always **gitignored**: they are
  per-GPU, per-TensorRT-version artifacts. Build them by pointing
  `pycamtrt.Engine()` (or the CLI's `--engine`/`--ocr`) straight at a
  `.onnx` — the engine is built once and cached here automatically
  (`<stem>_b1-<max_batch>_<fp16|fp32>_sm<arch>.engine`) — or by hand with
  the trtexec appendix in `BUILD.md`.

## Provenance and licensing (per file)

| file | origin | license | in git? |
|---|---|---|---|
| `yolov8n_plates*.onnx` | project-trained plate detector (YOLOv8 architecture); its training lineage could not be ruled AGPL-clean, so it is treated as ultralytics-derived | **treated as AGPL-3.0 — NOT distributed** | **no** |
| `lprnet_dynamic.onnx` | project-trained LPRNet OCR (Chinese-charset — a mechanics demo) | project (MIT) | yes |
| `mobilenet_v3s_dynamic.onnx` | torchvision `mobilenet_v3_small`, ImageNet-1k weights (`python/export_classifier.py`) | BSD-3 (torchvision weights) | yes |
| `resnet18_embed_dynamic.onnx` | torchvision `resnet18` minus its head → 512-d features (`python/export_resnet18_embed.py`) | BSD-3 (torchvision weights) | yes |
| `yolov8n_dynamic.onnx`, `yolov8n_416/1280_dynamic.onnx`, `yolo26n.onnx`, `rtdetr_l_dynamic.onnx` | ultralytics checkpoints | **AGPL-3.0 — NOT distributed here** | **no** |
| `veri_sbs_r50ibn_dynamic.onnx` | fast-reid (JDAI-CV) `Baseline` (ResNet50-IBN + NL + GeM + BNneck), VeRi-776-trained, weights v0.1.1 release asset [`veri_sbs_R50-ibn.pth`](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth) → 2048-d vehicle re-ID embedding, recreated locally via `export_veri_reid.py` (removed with the retired `camera_fleet_reid` example; see git history) | Apache-2.0 (fast-reid) | **no** (~90 MB; not committed — kept out of git the same way the ultralytics-derived exports are, see `.gitignore`) |

To create the ultralytics-derived exports locally (downloads their
checkpoints through the `ultralytics` package under your own acceptance of
its license — see `THIRD_PARTY.md`):

```bash
python3 python/export_ultralytics.py          # all of them
python3 python/export_ultralytics.py rtdetr   # just one
```

The plate detector itself is not distributed either (same caution — see the
table). The plate examples run with **any** detector export you point them
at: bring your own plate model, or fine-tune one and export it with a plain
`torch.onnx.export` / your framework's exporter — the pipeline only sees the
`[N,3,H,W]`-in, yolo-head-out contract below.

## Input contracts the pipeline enforces

- Exactly **1 input + 1 output tensor** per engine (named error otherwise).
- Detector (layer-0) engines: 4D `[N,3,H,W]`; H/W are read from the engine
  (640, 416, 1280 … all first-class). The TensorRT profile needs
  **kMIN batch 1** (min>1 profiles are rejected with a named error).
- Cascade child engines are free-size (LPRNet 94×24, mobilenet 224×224 and
  resnet18 224×224 all coexist).
