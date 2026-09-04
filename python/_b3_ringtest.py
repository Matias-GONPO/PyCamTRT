"""B3.5: ring-vs-looped-PTS discriminator for the mixed-res RSS growth.

Hypothesis: the compressed-packet ring (ring_seconds>0) fails to prune
across the PTS discontinuities a LOOPED 4K source produces, growing RSS
~MB/min; with ring_seconds=0 the growth should vanish.

Runs a detect-only pipeline on the given (looped-4K) streams for
`seconds`, printing RSS once a minute. Usage:
    _b3_ringtest.py <ring_seconds> <seconds> url1 [url2..]
"""
import sys
import threading
import time

import pycamtrt

DET = "models/yolov8n_plates_b1-16_fp16_sm86.engine"


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return -1


def main():
    ring = int(sys.argv[1])
    seconds = int(sys.argv[2])
    urls = sys.argv[3:]
    streams = pycamtrt.Streams(urls)
    det = pycamtrt.Layer("detect")
    e = det.add(pycamtrt.Engine(streams, DET))
    det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
    pipe = pycamtrt.Pipeline(streams, layers=[det], skip=2, max_frames=0,
                             ring_seconds=ring)
    timer = threading.Timer(seconds, pipe.stop)
    timer.start()
    n = 0
    rss0 = None
    t0 = tmin = time.time()
    try:
        for r in pipe:
            n += 1
            if rss0 is None:
                rss0 = rss_mb()
            now = time.time()
            if now - tmin >= 60:
                print(f"RING{ring} t={now - t0:5.0f}s rss={rss_mb()}MB",
                      flush=True)
                tmin = now
    finally:
        timer.cancel()
        pipe.stop()
    drift = rss_mb() - (rss0 or 0)
    print(f"RINGTEST ring_seconds={ring} results={n} "
          f"drift={drift}MB over {seconds}s", flush=True)


if __name__ == "__main__":
    main()
