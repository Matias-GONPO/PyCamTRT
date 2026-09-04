# Third-party components and models

PyCamTRT itself is MIT-licensed (see `LICENSE`). It builds against, and its
docs reference, the following third-party pieces — none of which are
redistributed in this repository unless stated:

- **torchvision pretrained weights** (mobilenet_v3_small, resnet18 —
  ImageNet-1k): BSD-3-Clause. The committed `models/mobilenet_v3s_dynamic.onnx`
  and `models/resnet18_embed_dynamic.onnx` are exports of these weights,
  redistributed under that license's terms (reproducible via
  `python/export_classifier.py` / `python/export_resnet18_embed.py`).
- **Ultralytics models** (yolov8n, yolo26n, rtdetr-l): **AGPL-3.0**. Their
  exports are deliberately NOT distributed in this repository. Recreate
  them locally with `python/export_ultralytics.py`, which downloads the
  checkpoints through the `ultralytics` package under your own acceptance
  of its license. If AGPL is a problem for your deployment, substitute any
  cleanly-exporting detector — the library routes on class ids and reads
  input dims from the engine, nothing is yolov8-specific at the API.
- **NVIDIA Video Codec SDK**: license forbids redistribution — download it
  yourself into `third_party/Video_Codec_SDK/` (see `BUILD.md`).
- **NVIDIA TensorRT / CUDA**: used via NVIDIA's containers/packages under
  their own licenses (see `BUILD.md` for the container route).
- **mediamtx** (RTSP server used by the demo stream farm): MIT; its binary
  is not distributed here — `BUILD.md` covers fetching it.
- **FFmpeg** (libavformat/avcodec/avutil): LGPL/GPL components installed
  from your distribution; linked, not distributed.

The project-trained OCR model (`lprnet_dynamic`) is released under this
repository's MIT license. The project's plate detector (`yolov8n_plates*`)
is NOT distributed: its training lineage could not be positively ruled free
of ultralytics (AGPL-3.0) checkpoint heritage, so it is kept out of the
repository on the cautious side (see `models/README.md`).
