# The capacity atlas

One cell = one (cameras, decode rate, inference rate) point run through the full plate cascade
(`build/rtsp_infer_multi`, plates detector b1-48 fp16 + LPRNet) for about 40 s on the live RTSP farm:
decoded fps against the offered rate (hold %), the post-decode timing split (pop-to-ready, ready-to-take,
GPU per batch), NVDEC and SM utilisation (nvidia-smi dmon medians), peak VRAM, plates read, and a
bottleneck label. Columns are the same in every atlas file here.

- `atlas_<res>_v2.csv`: the four-resolution series measured 2026-07 on the pre-v0.4.0 decoder. Its keyframe
  and low-camera-count cells are still right; its every-frame cells above the old ceilings are not
  (the decoder waited on inference, see CHANGELOG v0.4.0), and `recommend()` is fitted on them.
- `runs/2026-09-19_mini/<res>/atlas.csv`: the validation grid on the v0.4.0 build, 96 cells, with a cell-by-cell
  comparison against v2 in `compare_v2.txt`. Every-frame ceilings moved to the NVDEC wall: 720p 32 -> 50 cameras
  (11.8 ms at 50), 1080p 16 (the wall lies between 16 and 32; the ceiling campaign put it at 24), 1440p 8, 4K 4.
  Keyframe cells and the VRAM wall (1440p/4K at 100 keyframe cameras) are unchanged.
- A full v3 series is the next step: `bench/atlas/run_atlas_series.sh full 720p 1080p 1440p 4k` (about 4 to 5 hours per
  resolution, resume-capable), then refit `python/pycamtrt/_capacity.py`.

DeepStream on the same grid: `../deepstream/ds_dials.sh <res> <dir>` writes `ds_dials.csv` next to `atlas.csv`;
`atlas_vs_deepstream.py <runs dir> [out_prefix]` prints the shared-cell comparison and draws `docs/img/dials_vs_deepstream*.png`.

Tooling: `run_atlas.sh` (content + one resolution), `capacity_atlas.sh` (the grid; `full`, `mini`, `smoke`),
`atlas_farm.sh` (GOP-variant publishers), `atlas_compare.py`, `atlas_analyze.py` (bottleneck census and capacity fit).
Content: `tools/stream_farm/media/atlas_<res>_g{15,30,60,150}.mp4`, the ceiling campaign's recipe at four keyframe
intervals; built by `run_atlas.sh` from `plate_<res>_g30.mp4` or, with `SRC_DIR`, from the 4K source recordings.
