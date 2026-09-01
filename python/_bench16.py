"""Ephemeral benchmark: 16-stream plate cascade THROUGH the Python binding,
config matched to report 5's C++ fp16-cascade measurement (every frame,
plates-fp16 detector + lprnet-fp32 OCR). Prints the same latency split the
C++ CLI reports, computed from per-result fields."""
import sys
import pycamtrt

urls = sys.argv[1:]
streams = pycamtrt.Streams(urls)
det = pycamtrt.Layer("detect")
e1 = det.add(pycamtrt.Engine(streams, "models/yolov8n_plates_b1-16_fp16_sm86.engine"))
det.add(pycamtrt.Postprocess(e1, family="yolo", score=0.4))
read = pycamtrt.Layer("read")
e2 = read.add(pycamtrt.Engine(det.steps[-1], "models/lprnet_b1-32_fp32_sm86.engine"))
read.add(pycamtrt.Postprocess(e2, family="ctc"))

n = plates = 0
pre = queue = gpu = 0.0
batches = 0
last_seq = -1
import time
t0 = time.time()
with pycamtrt.Pipeline(streams, layers=[det, read], skip=1,
                       max_frames=300) as pipe:
    for r in pipe:
        n += 1
        plates += len(r.texts)
        pre += r.ms_pop_to_ready
        queue += r.ms_ready_to_take
        if r.batch_seq != last_seq:
            last_seq = r.batch_seq
            batches += 1
            gpu += r.ms_take_to_done
wall = time.time() - t0
print(f"PY16: {n} results, {plates} plates, {wall:.1f}s wall, "
      f"{n / wall:.1f} results/s")
print(f"PY16 latency split: {pre / n:.2f} pop->ready, {queue / n:.2f} "
      f"ready->take, {gpu / batches:.2f} take->done (avg/batch) | "
      f"end-to-end {(pre + queue) / n + gpu / batches:.2f} ms approx")
