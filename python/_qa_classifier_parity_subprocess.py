#!/usr/bin/env python3
"""_qa_classifier_parity_subprocess.py - helper for qa_matrix.py's section
G2 (M1b classifier-cascade parity gate), run as a SEPARATE PROCESS rather
than in-line - same reason as python/_qa_hold_subprocess.py: this uses
hold_frames=True, which carries a reproducible (nondeterministic,
~5%/process baseline, amplified by torch presence - see
manual/FINDINGS.md's "Multi-Pipeline + torch frame_cuda() exit-time crash"
entry) native exit-time crash. Every assertion/measurement below completes
and this script prints its RESULT line BEFORE any teardown begins, so the
parent's subprocess.run() captures the real result regardless of how (or
whether) this process exits - exactly _qa_hold_subprocess.py's pattern.

What this checks (the M1b parity gate): runs the SAME detect->classify
cascade as examples/classify_detections/classify_detections.py on one stream with
hold_frames=True, and for each result independently recomputes the
classifier's answer in plain torch/cv2/numpy from the SAME live frame
(r.fetch_frame(), a D2H NV12 copy - tier 2, not tier 3 - see
python/pycamtrt/__init__.py's module docstring) and the SAME detection
boxes, then compares argmax labels. The two paths necessarily differ in
resize implementation (the pipeline's GPU Nv12CropResizeKernel - bilinear,
back-projected sampling - vs. cv2.resize's own bilinear - see
src/preprocess.cu), so bit-exact (100%) agreement is not expected. The
ACTUAL gate (see GAP_LIMIT/AGREE_FLOOR below): every disagreement's
logit-gap < 1.0 AND overall agreement >= 60%.

M3a RESULT (per-channel ImageNet norm, replacing M1b's scalar
approximation norm=(-114.0, 1/58.6)): agreement measurably improved and
comfortably clears both the current gate and M1b's original (superseded)
>= 90% aspiration on most runs - 4 consecutive measurements on this demo's
content: 78.3% (47/60), 91.7% (55/60), 93.3% (56/60), 98.3% (59/60), all
with max mismatch logit-gap <= 0.263 (comfortably under GAP_LIMIT=1.0).
This is a real improvement over M1b's repeatable ~70-80% scalar baseline,
but NOT a full fix: the SAME 919<->530 ("street sign" <-> "digital clock")
near-tied decision boundary that motivated the GAP_LIMIT design (see
manual/FINDINGS.md's "M1 model-generality findings" section A) still
appears in every run's mismatches, at similarly small gaps (0.02-0.26) -
per-channel norm shifts this out-of-distribution classifier's logits
somewhat but does not resolve the underlying near-tie for this demo's
plate-crop content. Two real bugs found and fixed in an earlier
investigation (a missing Result.release_frame() under hold_frames=True,
and a crop-rect rounding mismatch against pipeline.cpp's actual
cascade-crop builder) remain fixed; this is a normalization-precision
effect, not a mechanics bug.

Prints one line ``RESULT agreement=<f> total=<i> matched=<i>`` on success
(computed over every detection seen across all consumed results - not just
per-result), or ``ERROR <message>`` on failure, then exits. The parent
(qa_matrix.py section G2) parses stdout and does not treat a nonzero/crashed
exit code as failure BY ITSELF - only a missing RESULT line is a failure.
"""
import sys

import numpy as np

import pycamtrt

DET_ENGINE = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
CLASSIFIER_ENGINE = "models/mobilenet_v3s_b1-32_fp16_sm86.engine"

# M3a: TRUE per-channel ImageNet normalization (upgrade of M1b's single
# scalar-pair approximation, norm=(-114.0, 1/58.6)) - see
# python/export_classifier.py's docstring for the mean/std -> offset/scale
# derivation and examples/classify_detections/classify_detections.py. Both sides of this parity
# check use this EXACT per-channel pair (that's the point: parity is about
# the crop/resize/argmax mechanics agreeing, not about how good the norm
# itself is) - RGB order, index 0 = R, matching color="rgb" below.
NORM_OFFSET = (-123.675, -116.28, -103.53)
NORM_SCALE = (1 / 58.395, 1 / 57.12, 1 / 57.375)

MAX_RESULTS = 60  # widened from 30 for gate stability
# Logit-gap gate (orchestrator ruling 2026-08-31, second design after the
# margin-threshold version proved to be whack-a-mole): on a MISMATCH, compute
# gap = ref_top_logit - ref_logits[pipeline_label]. Legitimate cross-resizer
# flips on decision-boundary crops keep the pipeline's answer among the
# reference's near-top classes (observed gaps 0.04-0.6 on the one
# boundary-parked box, always the same 919<->530 pair); a real crop/norm/
# argmax MECHANICS bug lands the pipeline on a class the reference scores
# units lower. Gate: max_gap < GAP_LIMIT and overall agreement >= AGREE_FLOOR
# (floor guards the degenerate all-mismatch case).
GAP_LIMIT = 1.0
AGREE_FLOOR = 0.6

# skip=7 (infer every 7th decoded frame), not skip=1: at ~30 fps, 30
# CONSECUTIVE decoded frames (skip=1) span ~1 second - close to one single
# physical object passing through frame, i.e. ~30 near-duplicate crops of
# the SAME plate rather than 30 independent samples. Investigated as the
# root cause of a real, reproduced flake (100% on one connection window,
# as low as 17% on another - see manual/FINDINGS.md's M1 section): the
# out-of-distribution classifier's decision margin for a given object is
# sometimes a near-tie between two classes (logit deltas as small as
# ~0.05, on raw logits ~9.3-9.6), and *when that happens*, the crop/resize
# implementation delta this whole check is measuring (GPU kernel vs cv2 -
# same BT.601 coefficients, different resize/convert order) is large
# enough to flip EVERY frame of that one near-tied object the same way -
# an N~1 sample of "objects", not an N=30 sample of "frames". skip=7
# spreads 30 results over ~7s of source video (~210 decoded frames),
# sampling multiple distinct objects/moments instead of one repeated ~30x.
SKIP = 7


def main():
    urls = sys.argv[1:] or ["rtsp://localhost:8554/cam1"]

    try:
        import cv2
        import torch
        import torchvision
    except ImportError as e:
        print(f"ERROR import failed: {e}")
        return 1
    if not torch.cuda.is_available():
        print("ERROR torch.cuda unusable")
        return 1

    device = torch.device("cuda")
    weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    ref_model = torchvision.models.mobilenet_v3_small(weights=weights)
    ref_model.eval().to(device)

    streams = pycamtrt.Streams(urls[:1])

    detect = pycamtrt.Layer("detect")
    eng = detect.add(pycamtrt.Engine(streams, DET_ENGINE))
    detect.add(pycamtrt.Postprocess(eng, family="yolo", score=0.4))

    classify = pycamtrt.Layer("classify")
    ceng = classify.add(pycamtrt.Engine(detect.steps[-1], CLASSIFIER_ENGINE,
                                        norm=(NORM_OFFSET, NORM_SCALE),
                                        color="rgb"))
    classify.add(pycamtrt.Postprocess(ceng, family="argmax"))

    pipe = pycamtrt.Pipeline(streams, layers=[detect, classify],
                             hold_frames=True, ring_depth=6, skip=SKIP,
                             max_frames=600)  # decoded-frame cap; comfortably
                                               # covers 30 results * SKIP=7 -
                                               # early break below once
                                               # MAX_RESULTS is reached

    total = 0
    matched = 0
    max_gap = 0.0
    results_seen = 0

    with pipe:
        for r in pipe:
            results_seen += 1
            print(f"  .. result {results_seen} s{r.stream_id} "
                  f"frame {r.frame_no} dets={len(r.detections)}",
                  file=sys.stderr)
            sys.stderr.flush()
            pairs = r.outputs["classify"]  # [(label, raw_logit), ...],
                                            # aligned with r.detections
            dets = r.detections
            if pairs and dets:
                frame = r.fetch_frame()  # tier 2: D2H NV12 copy, shape
                                          # (h*3/2, w) uint8 - see the
                                          # Result.fetch_frame() docstring
                h, w = r.frame_height, r.frame_width
                bgr = cv2.cvtColor(frame[: h * 3 // 2, :w],
                                   cv2.COLOR_YUV2BGR_NV12)

                for det, (pipe_label, _pipe_score) in zip(dets, pairs):
                    # Reproduce pipeline.cpp's EXACT crop-rect rounding (the
                    # cascade crop builder just above `LaunchNv12CropResizeBatched`
                    # in src/core/pipeline.cpp): x/y/w/h are each rounded
                    # INDEPENDENTLY (lroundf), then x/y are clamped to the
                    # frame, THEN w/h are clamped against the ALREADY-clamped
                    # x/y (`rw = max(1, min(round(w), sw - rx))`, not
                    # `round(x+w) - round(x)`). Rounding x0 and x1 as a SUM
                    # (the earlier, WRONG version of this script) can land a
                    # different integer than round(x)+round(w) whenever the
                    # fractional parts don't align - a silent 1px-per-edge
                    # crop-geometry mismatch against the C++ side, invisible
                    # on comfortably-sized crops but large relative to a
                    # small plate crop's own size. See manual/FINDINGS.md's
                    # M1 section for the investigation this fixed.
                    x0 = max(0, min(int(round(det.x)), w - 1))
                    y0 = max(0, min(int(round(det.y)), h - 1))
                    cw = max(1, min(int(round(det.w)), w - x0))
                    ch = max(1, min(int(round(det.h)), h - y0))
                    x1, y1 = x0 + cw, y0 + ch

                    crop_bgr = bgr[y0:y1, x0:x1]
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(crop_rgb, (224, 224),
                                         interpolation=cv2.INTER_LINEAR)
                    # M3a: per-channel norm broadcasts over the last (RGB
                    # channel) axis of `resized` - NORM_OFFSET/NORM_SCALE[i]
                    # lines up with channel i because crop_rgb (via
                    # COLOR_BGR2RGB above) and the pipeline's color="rgb"
                    # both put R at index 0.
                    normed = ((resized.astype(np.float32) + np.float32(NORM_OFFSET))
                             * np.float32(NORM_SCALE))
                    chw = np.transpose(normed, (2, 0, 1))[None]  # 1x3x224x224
                    x = torch.from_numpy(chw).to(device)

                    with torch.no_grad():
                        logits = ref_model(x)
                    ref_label = int(logits.argmax(dim=1).item())

                    total += 1
                    if ref_label == pipe_label:
                        matched += 1
                    else:
                        # The gate quantity: how far below the reference's
                        # top choice does it score the PIPELINE's choice?
                        # Small gap = boundary flip (fine); big gap = the
                        # pipeline classified a different image = mechanics.
                        row = logits[0]
                        gap = float(row[ref_label] - row[pipe_label])
                        max_gap = max(max_gap, gap)
                        print(f"MISMATCH gap={gap:.3f} ref={ref_label} "
                              f"pipe={pipe_label} "
                              f"box=({det.x:.1f},{det.y:.1f},"
                              f"{det.w:.1f},{det.h:.1f})", flush=True)

            # Explicit release (matches _qa_hold_subprocess.py's pattern) -
            # hold_frames=True holds EVERY consumed result's ring slot, not
            # just ones we called fetch_frame() on, so relying on implicit
            # refcounting to free the previous result on each `for r in
            # pipe` reassignment stalls this stream's producer once more
            # than ring_depth-1 results have accumulated unreleased.
            r.release_frame()

            if results_seen >= MAX_RESULTS:
                break

    if total == 0:
        print("ERROR no detections seen across consumed results - cannot "
              "measure parity (check farm has plate content up)")
        return 1

    agreement = matched / total
    print(f"RESULT agreement={agreement:.4f} total={total} matched={matched} "
          f"max_gap={max_gap:.4f} results_seen={results_seen}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
