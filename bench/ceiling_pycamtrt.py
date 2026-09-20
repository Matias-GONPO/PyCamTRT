"""bench/ceiling_pycamtrt.py - PyCamTRT cell for the ceiling campaign.

Same plate cascade as pycamtrt_side.py (yolov8n_plates fp16 -> LPRNet fp32,
skip=1), but measured the way the capacity atlas measures "holds": the
per-stream decoded-frame counters over a steady-state window, against the
offered rate, plus the post-decode latency split over the same window.

Usage: ceiling_pycamtrt.py out.csv N prefix seconds
Row:   N,offered_fps,decoded_fps,hold_pct,results_fps,pre_ms,queue_ms,gpu_ms,post_decode_ms,dropped
"""
import sys, time
import pycamtrt

DET = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
OCR = "models/lprnet_b1-32_fp32_sm86.engine"
FPS = 30.0
WARMUP_S = 12.0   # stream open + engine load + queue settle


def main():
    out, n, prefix, secs = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
    urls = [f"rtsp://localhost:8554/{prefix}{i}" for i in range(1, n + 1)]
    streams = pycamtrt.Streams(urls)
    det = pycamtrt.Layer("detect")
    e = det.add(pycamtrt.Engine(streams, DET))
    d = det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
    read = pycamtrt.Layer("read")
    oe = read.add(pycamtrt.Engine(d, OCR))
    read.add(pycamtrt.Postprocess(oe, family="ctc"))
    pipe = pycamtrt.Pipeline(streams, layers=[det, read], skip=1, backpressure="drop_oldest")
    pipe.start()
    t0 = time.time()
    res = pre = queue = gpu = 0.0
    batches = 0; last_seq = -1; dec0 = None; t_win = None
    for r in pipe:
        now = time.time()
        if now - t0 < WARMUP_S:
            continue
        if dec0 is None:
            dec0 = sum(pipe.get_stream_info(i).decoded for i in range(n)); t_win = now; continue
        res += 1; pre += r.ms_pop_to_ready; queue += r.ms_ready_to_take
        if r.batch_seq != last_seq:
            last_seq = r.batch_seq; batches += 1; gpu += r.ms_take_to_done
        if now - t_win >= secs:
            break
    dec1 = sum(pipe.get_stream_info(i).decoded for i in range(n)); dt = time.time() - t_win
    dropped = pipe.dropped_results()
    pipe.stop()
    offered = n * FPS; decoded_fps = (dec1 - dec0) / dt; results_fps = res / dt
    m = max(res, 1); b = max(batches, 1)
    post = queue / m + gpu / b
    row = f"{n},{offered:.0f},{decoded_fps:.1f},{100*decoded_fps/offered:.1f},{results_fps:.1f},{pre/m:.3f},{queue/m:.3f},{gpu/b:.3f},{post:.3f},{dropped}"
    print("CELL", row, flush=True)
    with open(out, "a") as f: f.write(row + "\n")


if __name__ == "__main__":
    main()
