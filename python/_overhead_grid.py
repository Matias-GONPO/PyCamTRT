"""Endpoint sink-overhead grid (overnight Phase C). Cells: N x sink-config,
each a fresh Pipeline over the plate cascade, skip=1, fixed frames. Emits one
CSV row per cell. Configs isolate each cost layer:
  off    ring_seconds=0, no sinks   (the true pre-endpoint baseline)
  ring   default ring, no sinks     (cost of the new ring_seconds=10 default)
  events ring + events sink (file)
  stream ring + relay of stream 0 (rtsp -> mediamtx)
  both   ring + events + relay
Usage: _overhead_grid.py out.csv N1,N2,.. cfg1,cfg2,.. frames url1 url2 ...
"""
import sys
import time

import pycamtrt

EVENTS_PATH = "/workspace/_grid_events.ndjson"


def run_cell(urls, cfg, frames):
    streams = pycamtrt.Streams(urls)
    det = pycamtrt.Layer("detect")
    e1 = det.add(pycamtrt.Engine(streams,
                                 "models/yolov8n_plates_b1-16_fp16_sm86.engine"))
    d1 = det.add(pycamtrt.Postprocess(e1, family="yolo", score=0.4))
    read = pycamtrt.Layer("read")
    e2 = read.add(pycamtrt.Engine(d1, "models/lprnet_b1-32_fp32_sm86.engine"))
    read.add(pycamtrt.Postprocess(e2, family="ctc"))

    sinks = []
    ring = 10
    if cfg == "off":
        ring = 0
    if cfg in ("events", "both"):
        sinks.append(pycamtrt.Sink(d1, kind="events",
                                   target=f"file://{EVENTS_PATH}"))
    if cfg in ("stream", "both"):
        sinks.append(pycamtrt.Sink(streams, kind="stream", stream=0,
                                   target="rtsp://localhost:8554/gridrelay"))

    n = batches = 0
    pre = queue = gpu = 0.0
    last_seq = -1
    t0 = time.time()
    with pycamtrt.Pipeline(streams, layers=[det, read], sinks=sinks,
                           ring_seconds=ring, skip=1,
                           max_frames=frames) as pipe:
        for r in pipe:
            n += 1
            pre += r.ms_pop_to_ready
            queue += r.ms_ready_to_take
            if r.batch_seq != last_seq:
                last_seq = r.batch_seq
                batches += 1
                gpu += r.ms_take_to_done
        dropped = pipe.sink_dropped()
    wall = time.time() - t0
    return dict(n=n, wall=wall, pre=pre / n, queue=queue / n,
                gpu=gpu / batches, dropped=dropped,
                e2e=(pre + queue) / n + gpu / batches)


def main():
    out, ns, cfgs, frames = (sys.argv[1], sys.argv[2].split(","),
                             sys.argv[3].split(","), int(sys.argv[4]))
    urls = sys.argv[5:]
    with open(out, "a") as f:
        for N in ns:
            for cfg in cfgs:
                t0 = time.time()
                m = run_cell(urls[: int(N)], cfg, frames)
                row = (f"{N},{cfg},{frames},{m['n']},{m['wall']:.1f},"
                       f"{m['pre']:.3f},{m['queue']:.3f},{m['gpu']:.3f},"
                       f"{m['e2e']:.3f},{m['dropped']},{time.time() - t0:.1f}")
                print("CELL", row, flush=True)
                f.write(row + "\n")
                f.flush()


if __name__ == "__main__":
    main()
