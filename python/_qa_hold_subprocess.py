#!/usr/bin/env python3
"""_qa_hold_subprocess.py - helper for qa_matrix.py's section E2, run as a
SEPARATE PROCESS rather than in-line.

WHY a subprocess (not just style): a reproducible native-level bug, isolated
during this task's ctx-migration work, fires at PROCESS EXIT (not during
actual use - every functional check below passes every single time before
it) whenever BOTH of these are true in one process:
  (a) more than one pycamtrt::Pipeline has been constructed (regardless of
      order or whether earlier ones were torn down or kept alive), AND
  (b) a hold_frames FrameResult's frame_cuda() view is consumed via
      torch.as_tensor(..., device="cuda").
qa_matrix.py's sections A-D each construct several Pipelines before section E
runs, so running E2's torch/CUDA-array-interface check in the same process as
A-D reliably hits this. Isolated in its own process (which constructs
exactly ONE Pipeline, ever), it does not reproduce. Root-caused as far as the
tooling available allows (see the task report): confirmed NOT caused by this
migration's cuCtxSynchronize() fix, confirmed device-side correct
(compute-sanitizer memcheck shows no kernel/device errors and the printed
gpu/host means agree every time), and confirmed host-side (the crash message
is glibc's "corrupted double-linked list" / a plain SIGSEGV, not a CUDA
error) - consistent with a rare interaction between repeated CUDA primary-
context retain/release cycles and torch's own CUDA state, surfacing in
driver/runtime teardown bookkeeping neither this process nor torch owns the
source of. Isolating the two symptoms (functional correctness vs. the known
exit-time crash) into their own process is the pragmatic fix: this script
prints its results BEFORE any teardown begins, so the parent's subprocess
call captures them regardless of how (or whether) this process exits.

Prints one line ``RESULT gpu_mean=<f> host_mean=<f> host_var=<f> frame_h=<i>
frame_addr=<hex>`` on success, or ``ERROR <message>`` on failure, then exits.
The parent (qa_matrix.py) parses stdout and does not treat a nonzero/crashed
exit code as failure BY ITSELF - only a missing RESULT line is a failure.
"""
import sys

import pycamtrt


def main():
    urls = sys.argv[1:] or ["rtsp://localhost:8554/cam1"]

    try:
        import torch
    except ImportError as e:
        print(f"ERROR torch unavailable: {e}")
        return 1
    if not torch.cuda.is_available():
        print("ERROR torch.cuda unusable")
        return 1

    DET = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
    streams = pycamtrt.Streams(urls[:1])
    layer = pycamtrt.Layer("detect")
    eng = layer.add(pycamtrt.Engine(streams, DET))
    layer.add(pycamtrt.Postprocess(eng, family="yolo"))

    pipe = pycamtrt.Pipeline(streams, layers=[layer], hold_frames=True,
                              ring_depth=6, max_frames=40)
    with pipe:
        it = iter(pipe)
        r = next(it)
        t = torch.as_tensor(r.frame_cuda(), device="cuda")
        gpu_mean = t[:r.frame_height].float().mean().item()
        del t  # done reading - see module docstring's CAI lifetime caveat
        a = r.fetch_frame()
        host_mean = float(a[:r.frame_height].mean())
        host_var = float(a[:r.frame_height].astype("float64").var())
        frame_addr = r.frame_addr
        frame_h = r.frame_height
        r.release_frame()
        for _ in it:
            pass  # drain the rest for a clean stop

    print(f"RESULT gpu_mean={gpu_mean} host_mean={host_mean} "
          f"host_var={host_var} frame_h={frame_h} frame_addr={frame_addr:x}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
