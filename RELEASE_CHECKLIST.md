# v0.2.0 public-release checklist (owner-executed; nothing here is automated)

All work is uncommitted on top of the single local commit `568ef93` (tag
`v0.1.0`, the born-verified state). Every phase below was gated green before
this checklist was written (qa matrix A–L, verify anchors, standalone gates —
see `manual/FINDINGS.md` in the CORDERO repo for the ledger). This file
itself need not be committed — delete it after the push, or keep it as
release notes raw material.

## 0. One decision before anything: the plates model's heritage

⚠ `yolov8n_plates*` — was it fine-tuned FROM ultralytics' `yolov8n.pt`
checkpoint? If YES, its weights arguably inherit AGPL-3.0 and it should
move to the not-distributed side (export recipe/notice instead of the
committed .onnx, like the other ultralytics derivatives). If it was trained
from scratch (or from non-ultralytics weights), the current MIT labeling in
`models/README.md` stands. Only you know — adjust `models/README.md` +
`THIRD_PARTY.md` if needed.

## 1. Remove ultralytics-derived exports from git (files stay on disk)

```bash
git rm --cached models/yolov8n.onnx models/yolov8n_dynamic.onnx models/yolo26n.onnx
```

Never `git add`: `models/yolov8n_416_dynamic.onnx`, `models/yolov8n_1280_dynamic.onnx`,
`models/rtdetr_l_dynamic.onnx` (131 MB!) — they are untracked; keep them so.
Consider adding these exact names to `.gitignore` so they can't slip in.

## 2. Sanity before staging

```bash
git status --porcelain | grep -E '\.engine|\.onnx'   # expect ONLY the rm-cached trio + kept torchvision/project onnx
git status --porcelain | awk '{print $2}' | xargs -r du -sh 2>/dev/null | sort -rh | head  # nothing huge
```

`.gitignore` already covers: engines, python logs/CSVs/ndjson, farm
media/bin/run, Video Codec SDK, lock files.

## 3. Suggested commit slices (working tree → v0.2.0)

1. **core: CP1 cascade parallelism + MP4/file input (T0+T2)** — src/core/pipeline.cpp
   (ChildScratch/streams/events, ProducerLoop EOF), src/FFmpegDemuxer.h (AVBSF),
   qa section J, `python/_cascade_grid.py`.
2. **core: report-11 fixes + families** — layer-0 dims engine-derived (net_w/net_h),
   `embedding` + `rtdetr` families (graph.h, pipeline.cpp, postprocess.cu/.h FMA-pinned
   variant, bindings, package), postprocess_batch_test rtdetr section,
   `python/_qa_embedding_gate.py`.
3. **feature: Select detection routing (R1)** — graph.h StepKind/StepDesc, pipeline.cpp
   Validate + GpuLoop pass lists, bindings, package `Select`, `python/_qa_select_gate.py`.
4. **feature: engine auto-build (R3) + build-config fix** — src/EngineBuilder.h, TrtEngine.h
   onnx path, CMakeLists (nvonnxparser, project name/VERSION, arch default, **Release
   build-type default — the unoptimized-release incident fix, see FINDINGS**), Engine kwargs,
   `python/export_ultralytics.py`, `python/export_resnet18_embed.py`.
5. **examples** — route_and_reid.py, detect_rtdetr.py, reference fixes in the older four.
6. **docs + licensing** — README, BUILD.md, Dockerfile, THIRD_PARTY.md, models/README.md,
   .gitignore, ADDING_A_FAMILY §6 (FMA lesson), qa sections K/L, the §1 removals.
7. **test runners** (optional, or fold into 1/2) — `python/_t3_probes.py`,
   `python/_t4_edges.py`, `python/_t5_soak.py`.

## 4. Tag and push

```bash
git tag -a v0.2.0 -m "first public release: 6 postprocess families, Select routing, engine auto-build, examples+docs"
git remote add origin <your-github-url>     # if not already wired
git push origin main --tags
```

## 5. Separate, still pending

- The CORDERO research repo's own 13-slice `COMMIT_PLAN.md` (its history
  covers everything the book/reports reference).
- Post-push niceties: GitHub description/topics, a release entry pasting
  the README's Measured table + Roadmap.

## Addendum (campaign complete, 2026-09-04)

New/changed since the checklist was written — fold into the slices:
- CMakeLists **Release build-type default** (slice 4 note already added) + packet_ring.h
  arrival-trim + SizeBytes/RINGDBG (slice 1 or its own robustness slice).
- `bench/` (slice 6 or its own): deepstream configs/scripts/parsers-src, METHODOLOGY.md,
  pycamtrt_side.py, diy_pynvc_trt.py, plate_farm.sh. `bench/results/` + `.diyenv/` are
  gitignored (add `.diyenv/` to .gitignore before staging).
- `_capacity.py` v2 constants + qa H1/K/L updates; CHANGELOG.md; `__version__`;
  README Measured-table refresh; `_routing_bench.py`, `_b3_*.py`, `_hold_hunt*`,
  `_qa_select_gate.py` runners.
- CORDERO side (its own commit plan): Atlas v2 folders ×4 + SAHI Refresh v2 + reports 11/12
  + tools patches (ATLAS_* env overrides, sahi_refresh_v2.sh).
