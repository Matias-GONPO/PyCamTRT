"""B2 (campaign night 5): the routing-benefit measurement.

Same 2-child tree run with Select OFF (every child sees every crop - the
pre-R1 behavior) vs ROUTED (person-class crops to the embedding child,
indoor-class crops to the argmax child). Captures per-child mean ms_gpu +
the standard latency split. CSV rows:
    mode,N,frames,n,wall,pre,queue,gpu,e2e,emb_ms,cls_ms,crops_emb,crops_cls

Usage: _routing_bench.py out.csv N1,N2 frames src [src2 ...]
  (srcs are repeated round-robin to reach each N - file inputs fine)
"""
import sys
import time

import pycamtrt

COCO = "models/yolov8n_b1-16_fp16_sm86.engine"
EMB = "models/resnet18emb_b1-32_fp16_sm86.engine"
CLS = "models/mobilenet_v3s_b1-32_fp16_sm86.engine"
IMAGENET_NORM = ((-123.675, -116.28, -103.53),
                 (1 / 58.395, 1 / 57.12, 1 / 57.375))
PERSON = {0}
INDOOR = {56, 62, 63, 66, 73}


def tree(urls, routed):
    streams = pycamtrt.Streams(urls)
    det = pycamtrt.Layer("detect")
    e = det.add(pycamtrt.Engine(streams, COCO))
    d = det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
    emb = pycamtrt.Layer("embed")
    esrc = emb.add(pycamtrt.Select(d, classes=PERSON)) if routed else d
    ee = emb.add(pycamtrt.Engine(esrc, EMB, norm=IMAGENET_NORM, color="rgb"))
    emb.add(pycamtrt.Postprocess(ee, family="embedding"))
    cls = pycamtrt.Layer("cls")
    csrc = cls.add(pycamtrt.Select(d, classes=INDOOR)) if routed else d
    ce = cls.add(pycamtrt.Engine(csrc, CLS, norm=IMAGENET_NORM, color="rgb"))
    cls.add(pycamtrt.Postprocess(ce, family="argmax"))
    return streams, [det, emb, cls]


def run_cell(urls, routed, frames):
    streams, layers = tree(urls, routed)
    n = batches = 0
    pre = queue = gpu = 0.0
    kid = [0.0, 0.0]
    kb = 0
    crops = [0, 0]
    last_seq = -1
    t0 = time.time()
    with pycamtrt.Pipeline(streams, layers=layers, skip=1,
                           max_frames=frames) as pipe:
        for r in pipe:
            n += 1
            pre += r.ms_pop_to_ready
            queue += r.ms_ready_to_take
            vecs = r.outputs["embed"]
            pairs = r.outputs["cls"]
            crops[0] += sum(1 for v in vecs if v)
            crops[1] += sum(1 for p in pairs if p and p[1] != 0.0)
            if r.batch_seq != last_seq:
                last_seq = r.batch_seq
                batches += 1
                gpu += r.ms_take_to_done
                if r.children:
                    kb += 1
                    for i, c in enumerate(r.children[:2]):
                        kid[i] += c.ms_gpu
    wall = time.time() - t0
    b, k, m = max(batches, 1), max(kb, 1), max(n, 1)
    return dict(n=n, wall=wall, pre=pre / m, queue=queue / m, gpu=gpu / b,
                e2e=(pre + queue) / m + gpu / b, emb=kid[0] / k,
                cls=kid[1] / k, ce=crops[0], cc=crops[1])


def main():
    out, ns, frames = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
    srcs = sys.argv[4:]
    with open(out, "a") as f:
        for N in ns:
            N = int(N)
            urls = [srcs[i % len(srcs)] for i in range(N)]
            for routed in (False, True):
                m = run_cell(urls, routed, frames)
                row = (f"{'routed' if routed else 'off'},{N},{frames},"
                       f"{m['n']},{m['wall']:.1f},{m['pre']:.3f},"
                       f"{m['queue']:.3f},{m['gpu']:.3f},{m['e2e']:.3f},"
                       f"{m['emb']:.3f},{m['cls']:.3f},{m['ce']},{m['cc']}")
                print("CELL", row, flush=True)
                f.write(row + "\n")
                f.flush()


if __name__ == "__main__":
    main()
