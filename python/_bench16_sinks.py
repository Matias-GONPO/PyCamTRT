"""Phase B (overnight plan): first 16-stream run through the Python binding,
WITH endpoint sinks attached. Closes two gaps at once: the binding was never
benchmarked past 8 streams, and sinks were never run at scale.

Args: urls... [--skip K] [--frames N]. Prints the same latency-split line as
_bench16.py plus sink accounting (events lines written vs results, drops)."""
import os
import sys
import time

import pycamtrt

EVENTS_PATH = "/workspace/_b16_events.ndjson"

args = []
skip, frames = 2, 300
argv = sys.argv[1:]
i = 0
while i < len(argv):  # flags consume their value; everything else is a URL
    if argv[i] == "--skip":
        skip = int(argv[i + 1]); i += 2
    elif argv[i] == "--frames":
        frames = int(argv[i + 1]); i += 2
    else:
        args.append(argv[i]); i += 1

if os.path.exists(EVENTS_PATH):
    os.remove(EVENTS_PATH)

streams = pycamtrt.Streams(args)
det = pycamtrt.Layer("detect")
e1 = det.add(pycamtrt.Engine(streams, "models/yolov8n_plates_b1-16_fp16_sm86.engine"))
d1 = det.add(pycamtrt.Postprocess(e1, family="yolo", score=0.4))
read = pycamtrt.Layer("read")
e2 = read.add(pycamtrt.Engine(d1, "models/lprnet_b1-32_fp32_sm86.engine"))
read.add(pycamtrt.Postprocess(e2, family="ctc"))

sinks = [
    pycamtrt.Sink(d1, kind="events", target=f"file://{EVENTS_PATH}"),
    pycamtrt.Sink(streams, kind="stream", stream=0,
                  target="rtsp://localhost:8554/b16relay"),
]

n = plates = batches = 0
pre = queue = gpu = 0.0
last_seq = -1
t0 = time.time()
with pycamtrt.Pipeline(streams, layers=[det, read], sinks=sinks,
                       skip=skip, max_frames=frames) as pipe:
    for r in pipe:
        n += 1
        plates += len(r.texts)
        pre += r.ms_pop_to_ready
        queue += r.ms_ready_to_take
        if r.batch_seq != last_seq:
            last_seq = r.batch_seq
            batches += 1
            gpu += r.ms_take_to_done
    dropped = pipe.sink_dropped()
wall = time.time() - t0

time.sleep(0.5)  # sink file flush margin
with open(EVENTS_PATH) as f:
    ndjson_lines = sum(1 for _ in f)
os.remove(EVENTS_PATH)

print(f"B16S: {len(args)} streams skip={skip}: {n} results, {plates} plates, "
      f"{wall:.1f}s wall, {n / wall:.1f} results/s")
print(f"B16S latency split: {pre / n:.2f} pop->ready, {queue / n:.2f} "
      f"ready->take, {gpu / batches:.2f} take->done | end-to-end "
      f"{(pre + queue) / n + gpu / batches:.2f} ms approx")
print(f"B16S sinks: events_lines={ndjson_lines} results={n} "
      f"sink_dropped={dropped} "
      f"{'OK 1:1' if ndjson_lines == n and dropped == 0 else 'MISMATCH'}")
