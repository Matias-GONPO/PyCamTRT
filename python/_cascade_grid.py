"""T1 (6h test run): cascade stream-parallelism benchmark grid.
Cells: mode {serial,parallel} x children {1,2,3} x N. Children: 1=ctc(read),
2=+argmax(classify), 3=+argmax(classify2, duplicate engine instance).
Emits one CSV row per cell incl. per-child mean ms_gpu.
Usage: _cascade_grid.py out.csv N1,N2,.. modes children_counts frames url1.."""
import sys
import time

import pycamtrt

DET = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
OCR = "models/lprnet_b1-32_fp32_sm86.engine"
CLS = "models/mobilenet_v3s_b1-32_fp16_sm86.engine"
IMAGENET_NORM = ((-123.675, -116.28, -103.53),
                 (1 / 58.395, 1 / 57.12, 1 / 57.375))


def build(urls, n_children, serial):
    streams = pycamtrt.Streams(urls)
    det = pycamtrt.Layer("detect")
    e = det.add(pycamtrt.Engine(streams, DET))
    d = det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
    layers = [det]
    if n_children >= 1:
        read = pycamtrt.Layer("read")
        oe = read.add(pycamtrt.Engine(d, OCR))
        read.add(pycamtrt.Postprocess(oe, family="ctc"))
        layers.append(read)
    for k in range(2, n_children + 1):
        cl = pycamtrt.Layer(f"classify{k - 1}")
        ce = cl.add(pycamtrt.Engine(d, CLS, norm=IMAGENET_NORM, color="rgb"))
        cl.add(pycamtrt.Postprocess(ce, family="argmax"))
        layers.append(cl)
    return pycamtrt.Pipeline(streams, layers=layers, skip=1,
                             max_frames=int(sys.argv[5]),
                             cascade_serial=serial)


def run_cell(urls, n_children, serial):
    n = batches = 0
    pre = queue = gpu = 0.0
    child_ms = [0.0] * n_children
    child_batches = 0
    last_seq = -1
    t0 = time.time()
    with build(urls, n_children, serial) as pipe:
        for r in pipe:
            n += 1
            pre += r.ms_pop_to_ready
            queue += r.ms_ready_to_take
            if r.batch_seq != last_seq:
                last_seq = r.batch_seq
                batches += 1
                gpu += r.ms_take_to_done
                if r.children:
                    child_batches += 1
                    for i, c in enumerate(r.children[:n_children]):
                        child_ms[i] += c.ms_gpu
    wall = time.time() - t0
    cb = max(child_batches, 1)
    return dict(n=n, wall=wall, pre=pre / n, queue=queue / n,
                gpu=gpu / batches, e2e=(pre + queue) / n + gpu / batches,
                child_ms=[m / cb for m in child_ms])


def main():
    out, ns, modes, ccounts, frames = (sys.argv[1], sys.argv[2].split(","),
                                       sys.argv[3].split(","),
                                       sys.argv[4].split(","), sys.argv[5])
    urls = sys.argv[6:]
    with open(out, "a") as f:
        for N in ns:
            for nc in ccounts:
                for mode in modes:
                    t0 = time.time()
                    m = run_cell(urls[: int(N)], int(nc), mode == "serial")
                    kids = "|".join(f"{v:.3f}" for v in m["child_ms"])
                    row = (f"{N},{nc},{mode},{frames},{m['n']},"
                           f"{m['wall']:.1f},{m['pre']:.3f},{m['queue']:.3f},"
                           f"{m['gpu']:.3f},{m['e2e']:.3f},{kids}")
                    print("CELL", row, flush=True)
                    f.write(row + "\n")
                    f.flush()


if __name__ == "__main__":
    main()
