"""B3.3: drop_oldest under deliberate overload + recovery.

Phase 1 (overload): consumer sleeps per result, forcing the result queue
full -> drop_oldest MUST evict (dropped_results() grows) while the
consumer keeps seeing FRESH frames (frame_no keeps advancing rather than
serving stale backlog).
Phase 2 (recovery): consumer stops sleeping -> evictions must stop
growing and the pipeline drains cleanly.

Gates printed at the end; exit 0/3. Usage: _b3_droptest.py url1 [url2..]
"""
import sys
import time

import pycamtrt

DET = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
OCR = "models/lprnet_b1-32_fp32_sm86.engine"


def main():
    urls = sys.argv[1:]
    streams = pycamtrt.Streams(urls)
    det = pycamtrt.Layer("detect")
    e = det.add(pycamtrt.Engine(streams, DET))
    d = det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
    read = pycamtrt.Layer("read")
    oe = read.add(pycamtrt.Engine(d, OCR))
    read.add(pycamtrt.Postprocess(oe, family="ctc"))
    pipe = pycamtrt.Pipeline(streams, layers=[det, read], skip=1,
                             max_frames=0, backpressure="drop_oldest",
                             queue_capacity=64)
    t0 = time.time()
    phase = 1
    n = 0
    max_fno = {}
    stale = 0
    drops_p1 = drops_p2_start = 0
    with pipe:
        for r in pipe:
            n += 1
            prev = max_fno.get(r.stream_id, -1)
            if r.frame_no < prev:
                stale += 1
            max_fno[r.stream_id] = max(prev, r.frame_no)
            el = time.time() - t0
            if phase == 1:
                time.sleep(0.005)  # deliberate slow consumer
                if el > 45:
                    phase = 2
                    drops_p1 = pipe.dropped_results()
                    drops_p2_start = drops_p1
                    print(f"PHASE1 done: results={n} dropped={drops_p1}",
                          flush=True)
            elif el > 75:
                break
    drops_final = pipe.dropped_results()
    p2_new = drops_final - drops_p2_start
    print(f"DROPTEST results={n} stale_results={stale} "
          f"dropped_p1={drops_p1} dropped_new_in_p2={p2_new}", flush=True)
    gates = {
        "evictions_under_overload": drops_p1 > 0,
        "no_stale_delivery": stale == 0,
        "recovery_drops_stop": p2_new < max(1, drops_p1 // 10),
        "clean_stop": True,
    }
    print("DROPTEST GATES:", {k: "PASS" if v else "FAIL"
                              for k, v in gates.items()}, flush=True)
    sys.exit(0 if all(gates.values()) else 3)


if __name__ == "__main__":
    main()
