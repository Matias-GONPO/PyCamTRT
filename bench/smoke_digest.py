"""bench/smoke_digest.py - determinism gate: N copies of one clip file through the plate cascade
(same graph as bench/ceiling_pycamtrt.py). Writes per-frame detection digests
so builds can be compared byte for byte. Two runs of the same build must produce the same digest (sha printed on the SMOKE line).
Usage (inside the container, repo root): PYTHONPATH=build:python python3 bench/smoke_digest.py tools/stream_farm/media/atlas_plate_g30.mp4 4 300 /tmp/digest.txt"""
import sys, time, hashlib, statistics
import pycamtrt

DET = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
OCR = "models/lprnet_b1-32_fp32_sm86.engine"
clip, n, frames, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
streams = pycamtrt.Streams([clip] * n)
det = pycamtrt.Layer("detect")
e = det.add(pycamtrt.Engine(streams, DET))
d = det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
read = pycamtrt.Layer("read")
oe = read.add(pycamtrt.Engine(d, OCR))
read.add(pycamtrt.Postprocess(oe, family="ctc"))
pipe = pycamtrt.Pipeline(streams, layers=[det, read], skip=1, max_frames=frames)
pipe.start()
rows, pre, q, g = [], [], [], []
t0 = time.time()
for r in pipe:
    dets = sorted((int(dd.cls), round(float(dd.x), 1), round(float(dd.y), 1), round(float(dd.w), 1),
                   round(float(dd.h), 1), round(float(dd.score), 3)) for dd in r.detections)
    rows.append((r.stream_id, r.frame_no, dets))
    pre.append(r.ms_pop_to_ready); q.append(r.ms_ready_to_take); g.append(r.ms_take_to_done)
wall = time.time() - t0
pipe.stop()
rows.sort()
h = hashlib.sha256()
with open(out, "w") as f:
    for sid, fr, dets in rows:
        line = f"{sid} {fr} {dets}\n"; h.update(line.encode()); f.write(line)
ndet = sum(len(dd) for _, _, dd in rows)
print(f"SMOKE results={len(rows)} dets={ndet} sha={h.hexdigest()[:16]} wall={wall:.1f}s "
      f"pre_med={statistics.median(pre):.3f} queue_med={statistics.median(q):.3f} gpu_med={statistics.median(g):.3f} ms", flush=True)
