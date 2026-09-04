"""hold_frames exit-crash hunt (campaign sidecar): ONE variant per process.

The bug (FINDINGS "Multi-Pipeline + torch frame_cuda() exit-time crash",
recharacterized session C2-2): a nondeterministic stray write somewhere in
the hold_frames path corrupts the glibc heap; ptmalloc's bin checks
detonate at interpreter exit ~5%/process without torch, ~58% with torch
consuming a frame_cuda() view. compute-sanitizer clean (host-side write).

Variants (each invocation runs exactly one, then exits - the EXIT is the
experiment):
  vnohold  1 pipeline, hold_frames=False, drain 20    (control, was 0/5)
  v0       1 pipeline, hold_frames=True, drain 20, no torch (was 1/10)
  v2       v0 + torch.as_tensor(frame_cuda()) consume       (was 3/4)
  v0long   v0 with max_frames=2000  (per-process vs per-frame discriminator)

Run under MALLOC_CHECK_=3 (abort at FIRST detected corruption - closer to
the write site) and MALLOC_PERTURB_=42; the driver loops reps and runs gdb
on crashers. Usage: _hold_hunt.py <variant> <src>
"""
import sys

import pycamtrt

DET = "models/yolov8n_plates_b1-16_fp16_sm86.engine"


def run(variant, src):
    hold = variant != "vnohold"
    frames = 2000 if variant == "v0long" else 20
    streams = pycamtrt.Streams([src])
    det = pycamtrt.Layer("detect")
    e = det.add(pycamtrt.Engine(streams, DET))
    det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
    pipe = pycamtrt.Pipeline(streams, layers=[det], skip=1,
                             max_frames=frames, hold_frames=hold,
                             ring_depth=6)
    first = None
    n = 0
    with pipe:
        for r in pipe:
            n += 1
            if first is None:
                first = r
                if variant == "v2":
                    import torch
                    t = torch.as_tensor(first.frame_cuda(), device="cuda")
                    print(f"gpu_mean={float(t.float().mean()):.3f}",
                          flush=True)
                if hold:
                    first.release_frame()
                first = None
    print(f"DRAINED variant={variant} n={n}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
    print("CLEAN_BODY_EXIT", flush=True)  # anything after this is teardown
