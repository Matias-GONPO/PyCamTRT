#!/usr/bin/env python3
"""Parses deepstream-app --latency stdout into CORDERO-comparable splits.

Component mapping to CORDERO's pre/queue/gpu latency split (report 3
methodology), recorded as an approximation, not an identity:
  pre   <- nvv4l2decoder0 component latency  (decode)
  queue <- nvstreammux-src_bin_muxer component_latency (wait to close batch)
  gpu   <- primary_gie + secondary_gie_0 component latency (stage1+stage2)
  total <- "Frame latency" (source-timestamp to last-probe, includes decode
           queueing that CORDERO's clock doesn't start until PopFrame)

Usage: docker run ... deepstream-app -c CFG -t | parse_latency.py [--warmup N]
Discards the first `warmup` frames per source (default 60 = ~2s at 30fps)
to exclude the engine-build/pipeline-ramp transient, same discipline
CORDERO's own benchmarks use.
"""
import re
import sys
import statistics as st
from collections import defaultdict

LINE_COMP = re.compile(
    r"Comp name = (\S+).*?component[_ ]latency\s*=\s*([\d.]+)")
LINE_FRAME = re.compile(
    r"Source id = (\d+) Frame_num = (\d+) Frame latency = ([\d.]+)")


def main():
    warmup = 60
    if "--warmup" in sys.argv:
        warmup = int(sys.argv[sys.argv.index("--warmup") + 1])

    per_source_frame = defaultdict(int)
    decode, queue, gpu1, gpu2, total = (defaultdict(list) for _ in range(5))
    pending_decode = {}
    pending_queue = {}
    pending_gpu1 = {}
    pending_gpu2 = {}

    for line in sys.stdin:
        m = LINE_COMP.search(line)
        if m:
            name, lat = m.group(1), float(m.group(2))
            if name.startswith("nvv4l2decoder"):
                pending_decode["cur"] = lat
            elif name.startswith("nvstreammux"):
                pending_queue["cur"] = lat
            elif name == "primary_gie":
                pending_gpu1["cur"] = lat
            elif name == "secondary_gie_0":
                pending_gpu2["cur"] = lat
            continue
        m = LINE_FRAME.search(line)
        if m:
            src, fnum, lat = int(m.group(1)), int(m.group(2)), float(m.group(3))
            per_source_frame[src] = fnum
            if fnum >= warmup:
                if "cur" in pending_decode:
                    decode[src].append(pending_decode["cur"])
                if "cur" in pending_queue:
                    queue[src].append(pending_queue["cur"])
                g = pending_gpu1.get("cur", 0.0) + pending_gpu2.get("cur", 0.0)
                gpu1[src].append(g)
                total[src].append(lat)
            pending_decode.clear()
            pending_queue.clear()
            pending_gpu1.clear()
            pending_gpu2.clear()

    if not total:
        print("No post-warmup frames captured (stream too short or "
              "--latency not enabled).", file=sys.stderr)
        sys.exit(1)

    srcs = sorted(total.keys())
    print(f"{'src':>4} {'n':>5} {'decode_ms':>10} {'queue_ms':>9} "
          f"{'gpu_ms':>8} {'total_ms':>9}")

    def med(lst):
        return st.median(lst) if lst else 0.0

    all_decode, all_queue, all_gpu, all_total = [], [], [], []
    for s in srcs:
        d, q, g, t = med(decode[s]), med(queue[s]), med(gpu1[s]), med(total[s])
        print(f"{s:>4} {len(total[s]):>5} {d:>10.2f} {q:>9.2f} {g:>8.2f} {t:>9.2f}")
        all_decode += decode[s]
        all_queue += queue[s]
        all_gpu += gpu1[s]
        all_total += total[s]

    print("-" * 50)
    print(f"{'ALL':>4} {len(all_total):>5} {med(all_decode):>10.2f} "
          f"{med(all_queue):>9.2f} {med(all_gpu):>8.2f} {med(all_total):>9.2f}"
          f"   (median across all sources, warmup={warmup} frames dropped)")


if __name__ == "__main__":
    main()
