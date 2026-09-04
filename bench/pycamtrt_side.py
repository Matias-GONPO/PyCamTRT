"""bench/pycamtrt_side.py - PyCamTRT's side of the DeepStream head-to-head.

Runs the SAME workload the DeepStream sweep runs (yolov8n_plates fp16
detector -> LPRNet fp32 OCR cascade, skip=1) over the same live farm
content (cam1..N webcam, plate1..N real plate clip - see
deepstream/plate_farm.sh), and emits one CSV row per cell in the house
grid-runner convention (append + "CELL <row>" echo, like
python/_cascade_grid.py).

Columns: content,N,frames,n_results,wall_s,pre_ms,queue_ms,gpu_ms,e2e_ms
  pre   = ms_pop_to_ready   (preprocess + sync; starts at PopFrame -
          see deepstream/../METHODOLOGY.md for the clock-start contract)
  queue = ms_ready_to_take  (batch formation wait)
  gpu   = ms_take_to_done   (whole-batch GPU cycle, per batch)
  e2e   = pre + queue + gpu (the post-decode figure compared against
          DeepStream's streammux + GIE component latencies)

Usage: pycamtrt_side.py out.csv N1,N2,.. cam,plate frames
"""
import sys
import time

import pycamtrt

DET = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
OCR = "models/lprnet_b1-32_fp32_sm86.engine"


def run_cell(urls, frames):
    streams = pycamtrt.Streams(urls)
    det = pycamtrt.Layer("detect")
    e = det.add(pycamtrt.Engine(streams, DET))
    d = det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
    read = pycamtrt.Layer("read")
    oe = read.add(pycamtrt.Engine(d, OCR))
    read.add(pycamtrt.Postprocess(oe, family="ctc"))

    n = batches = 0
    pre = queue = gpu = 0.0
    last_seq = -1
    t0 = time.time()
    with pycamtrt.Pipeline(streams, layers=[det, read], skip=1,
                           max_frames=frames) as pipe:
        for r in pipe:
            n += 1
            pre += r.ms_pop_to_ready
            queue += r.ms_ready_to_take
            if r.batch_seq != last_seq:
                last_seq = r.batch_seq
                batches += 1
                gpu += r.ms_take_to_done
    wall = time.time() - t0
    b = max(batches, 1)
    m = max(n, 1)
    return dict(n=n, wall=wall, pre=pre / m, queue=queue / m, gpu=gpu / b,
                e2e=(pre + queue) / m + gpu / b)


def main():
    out, ns, contents, frames = (sys.argv[1], sys.argv[2].split(","),
                                 sys.argv[3].split(","), int(sys.argv[4]))
    with open(out, "a") as f:
        for content in contents:
            for N in ns:
                urls = [f"rtsp://localhost:8554/{content}{i}"
                        for i in range(1, int(N) + 1)]
                m = run_cell(urls, frames)
                row = (f"{content},{N},{frames},{m['n']},{m['wall']:.1f},"
                       f"{m['pre']:.3f},{m['queue']:.3f},{m['gpu']:.3f},"
                       f"{m['e2e']:.3f}")
                print("CELL", row, flush=True)
                f.write(row + "\n")
                f.flush()


if __name__ == "__main__":
    main()
