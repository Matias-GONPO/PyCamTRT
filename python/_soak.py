"""Overnight soak driver (Phase C3): N streams, both sinks, skip=2,
unbounded run for --seconds S. Prints one heartbeat line per minute:
minute, results in that minute, cumulative, sink_dropped, dropped_results,
own VmRSS MB. Exit 0 only if the run stayed healthy and stop was clean.
Events go to a file sink; the final line reports events-vs-results accounting."""
import os
import sys
import time

import pycamtrt

EVENTS_PATH = "/workspace/_soak_events.ndjson"

urls = [a for a in sys.argv[1:] if not a.startswith("--")]
seconds = 10800
if "--seconds" in sys.argv:
    seconds = int(sys.argv[sys.argv.index("--seconds") + 1])
    urls = [u for u in urls if u != str(seconds)]

if os.path.exists(EVENTS_PATH):
    os.remove(EVENTS_PATH)


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return -1


streams = pycamtrt.Streams(urls)
det = pycamtrt.Layer("detect")
e1 = det.add(pycamtrt.Engine(streams, "models/yolov8n_plates_b1-16_fp16_sm86.engine"))
d1 = det.add(pycamtrt.Postprocess(e1, family="yolo", score=0.4))
read = pycamtrt.Layer("read")
e2 = read.add(pycamtrt.Engine(d1, "models/lprnet_b1-32_fp32_sm86.engine"))
read.add(pycamtrt.Postprocess(e2, family="ctc"))
sinks = [
    pycamtrt.Sink(d1, kind="events", target=f"file://{EVENTS_PATH}"),
    pycamtrt.Sink(streams, kind="stream", stream=0,
                  target="rtsp://localhost:8554/soakrelay"),
]

total = minute_count = 0
t0 = time.time()
next_beat = t0 + 60
print(f"SOAK start: {len(urls)} streams, skip=2, {seconds}s target, "
      f"rss={rss_mb()}MB", flush=True)
pipe = pycamtrt.Pipeline(streams, layers=[det, read], sinks=sinks,
                         skip=2, max_frames=0)
pipe.start()
healthy = True
while time.time() - t0 < seconds:
    status, r = pipe._pipe.poll(500)
    if status == pycamtrt.PollStatus.Finished:
        print("SOAK ERROR: pipeline finished early", flush=True)
        healthy = False
        break
    if status == pycamtrt.PollStatus.Ok:
        total += 1
        minute_count += 1
    now = time.time()
    if now >= next_beat:
        m = int((now - t0) // 60)
        print(f"SOAK m{m}: {minute_count} results/min (cum {total}) "
              f"sink_dropped={pipe.sink_dropped()} "
              f"dropped={pipe.dropped_results()} rss={rss_mb()}MB", flush=True)
        minute_count = 0
        next_beat += 60

pipe.stop()
recon = sum(pipe.get_stream_info(i).reconnects for i in range(len(urls)))
failed = sum(1 for i in range(len(urls)) if pipe.get_stream_info(i).failed)
decoded = sum(pipe.get_stream_info(i).decoded for i in range(len(urls)))
time.sleep(1)
with open(EVENTS_PATH) as f:
    ev = sum(1 for _ in f)
os.remove(EVENTS_PATH)
wall = time.time() - t0
print(f"SOAK end: {total} results / {wall:.0f}s = {total / wall:.1f}/s | "
      f"decoded={decoded} reconnects={recon} failed={failed} | "
      f"events={ev} vs results={total} "
      f"({'OK' if abs(ev - total) <= pipe.sink_dropped() + 2 else 'MISMATCH'}) "
      f"sink_dropped={pipe.sink_dropped()} rss={rss_mb()}MB", flush=True)
sys.exit(0 if healthy and failed == 0 else 1)
