"""T5 (6h test run): parallel-cascade soak. 16 streams, 2-child tree
(ctc + argmax), cascade_serial=False, events + relay sinks, skip=2.
Per-minute heartbeats; gates printed at the end.

Args: duration_seconds url1 url2 ...
"""
import os
import sys
import threading
import time

import pycamtrt

DET = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
OCR = "models/lprnet_b1-32_fp32_sm86.engine"
CLS = "models/mobilenet_v3s_b1-32_fp16_sm86.engine"
IMAGENET_NORM = ((-123.675, -116.28, -103.53),
                 (1 / 58.395, 1 / 57.12, 1 / 57.375))
EVENTS_PATH = "/workspace/python/_t5_events.ndjson"


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return -1


def main():
    duration = int(sys.argv[1])
    urls = sys.argv[2:]
    if os.path.exists(EVENTS_PATH):
        os.remove(EVENTS_PATH)

    streams = pycamtrt.Streams(urls)
    det = pycamtrt.Layer("detect")
    e1 = det.add(pycamtrt.Engine(streams, DET))
    d1 = det.add(pycamtrt.Postprocess(e1, family="yolo", score=0.4))
    read = pycamtrt.Layer("read")
    e2 = read.add(pycamtrt.Engine(d1, OCR))
    read.add(pycamtrt.Postprocess(e2, family="ctc"))
    cls = pycamtrt.Layer("classify")
    e3 = cls.add(pycamtrt.Engine(d1, CLS, norm=IMAGENET_NORM, color="rgb"))
    cls.add(pycamtrt.Postprocess(e3, family="argmax"))
    sinks = [
        pycamtrt.Sink(d1, kind="events", target=f"file://{EVENTS_PATH}"),
        pycamtrt.Sink(streams, kind="stream", stream=0,
                      target="rtsp://localhost:8554/t5relay"),
    ]
    pipe = pycamtrt.Pipeline(streams, layers=[det, read, cls], skip=2,
                             max_frames=0, cascade_serial=False, sinks=sinks)
    timer = threading.Timer(duration, pipe.stop)
    timer.start()

    n = plates = batches = 0
    minute_n = 0
    minute_kid = [0.0, 0.0]
    minute_kb = 0
    rates = []
    rss0 = rss_last = rss_mb()
    last_seq = -1
    t0 = tmin = time.time()
    try:
        for r in pipe:
            n += 1
            if n == 1:
                # baseline AFTER Start(): the gate measures steady-state
                # drift, not the one-time startup allocations (NVDEC
                # sessions, PacketRing, sink threads land between ctor
                # and first result — ~300 MB at 16 streams).
                rss0 = rss_mb()
            minute_n += 1
            plates += len(r.texts)
            if r.batch_seq != last_seq:
                last_seq = r.batch_seq
                batches += 1
                if r.children:
                    minute_kb += 1
                    for i, c in enumerate(r.children[:2]):
                        minute_kid[i] += c.ms_gpu
            now = time.time()
            if now - tmin >= 60.0:
                rss_last = rss_mb()
                kb = max(minute_kb, 1)
                rate = minute_n / (now - tmin)
                rates.append(rate)
                print(f"HEARTBEAT t={now - t0:6.0f}s rate={rate:6.1f}/s "
                      f"child_ms={minute_kid[0] / kb:.3f}|"
                      f"{minute_kid[1] / kb:.3f} rss={rss_last}MB "
                      f"dropped={pipe.dropped_results()} "
                      f"sink_dropped={pipe.sink_dropped()}", flush=True)
                minute_n, minute_kb = 0, 0
                minute_kid = [0.0, 0.0]
                tmin = now
    finally:
        timer.cancel()
        pipe.stop()
    wall = time.time() - t0
    time.sleep(0.5)
    with open(EVENTS_PATH) as f:
        lines = sum(1 for _ in f)

    drift = rss_last - rss0
    rate_ok = (len(rates) < 3 or
               abs(rates[-1] - rates[0]) / max(rates[0], 1e-9) < 0.10)
    print(f"SOAK n={n} plates={plates} wall={wall:.0f}s "
          f"rate={n / wall:.1f}/s events_lines={lines} "
          f"rss_drift={drift}MB dropped={pipe.dropped_results()} "
          f"sink_dropped={pipe.sink_dropped()}", flush=True)
    gates = {
        "rss_drift<64MB": abs(drift) < 64,
        "rate_held": rate_ok,
        "events==results": lines == n,
        "clean_stop": True,
    }
    print("SOAK GATES:", {k: ("PASS" if v else "FAIL")
                          for k, v in gates.items()}, flush=True)
    sys.exit(0 if all(gates.values()) else 3)


if __name__ == "__main__":
    main()
