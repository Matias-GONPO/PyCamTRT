"""Embedding-family gate (T3.1 follow-up): bit-exact cross-family check.

The SAME engine file (resnet18, 2D [N,512] head) runs as TWO siblings off
one detector: an `embedding` child (raw pass-through) and an `argmax`
child (GPU max-logit kernel), identical norm/color. For every detection,
argmax() over the embedding child's raw vector must reproduce the argmax
child's (label, score) BIT-EXACTLY - same tensor, two independent read
paths. Run in BOTH cascade modes (CP1 parallel + serial escape hatch).

Usage: _qa_embedding_gate.py <src> [frames]
"""
import os
import sys

import pycamtrt

DET = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
# EMB env override lets the R3 auto-build gate point this at the .onnx.
EMB = os.environ.get("EMB", "models/resnet18emb_b1-32_fp16_sm86.engine")
IMAGENET_NORM = ((-123.675, -116.28, -103.53),
                 (1 / 58.395, 1 / 57.12, 1 / 57.375))


def run(src, frames, serial):
    streams = pycamtrt.Streams([src])
    det = pycamtrt.Layer("detect")
    e = det.add(pycamtrt.Engine(streams, DET))
    d = det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
    emb = pycamtrt.Layer("embed")
    ee = emb.add(pycamtrt.Engine(d, EMB, norm=IMAGENET_NORM, color="rgb"))
    emb.add(pycamtrt.Postprocess(ee, family="embedding"))
    cls = pycamtrt.Layer("cls")
    ce = cls.add(pycamtrt.Engine(d, EMB, norm=IMAGENET_NORM, color="rgb"))
    cls.add(pycamtrt.Postprocess(ce, family="argmax"))

    n = dets = mismatch = bad_dim = 0
    with pycamtrt.Pipeline(streams, layers=[det, emb, cls], skip=1,
                           max_frames=frames,
                           cascade_serial=serial) as pipe:
        for r in pipe:
            n += 1
            vecs = r.outputs["embed"]
            pairs = r.outputs["cls"]
            if len(vecs) != len(pairs) or len(vecs) != len(r.detections):
                mismatch += 1
                continue
            for v, (lab, sc) in zip(vecs, pairs):
                dets += 1
                if len(v) != 512:
                    bad_dim += 1
                    continue
                py_lab = max(range(len(v)), key=lambda i: (v[i], -i))
                if py_lab != lab or v[py_lab] != sc:
                    mismatch += 1
                    if mismatch <= 3:
                        print(f"  MISMATCH: py=({py_lab},{v[py_lab]!r}) "
                              f"gpu=({lab},{sc!r})", flush=True)
    mode = "serial" if serial else "parallel"
    ok = n > 0 and dets > 0 and mismatch == 0 and bad_dim == 0
    print(f"EMBED_GATE[{mode}] results={n} dets={dets} "
          f"mismatches={mismatch} bad_dim={bad_dim} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    src = sys.argv[1]
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    ok_p = run(src, frames, serial=False)
    ok_s = run(src, frames, serial=True)
    print("EMBED_GATE overall:", "PASS" if ok_p and ok_s else "FAIL",
          flush=True)
    sys.exit(0 if ok_p and ok_s else 3)


if __name__ == "__main__":
    main()
