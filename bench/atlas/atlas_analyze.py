#!/usr/bin/env python3
"""Analyze the capacity atlas CSV -> tables, bottleneck map, model fit.

Produces the material for Report 8:
  1. Per-N pivot: hold% and total latency vs (decode, inference).
  2. Redundant-pair isolation: same inference rate via different decode
     -> the pure cost of decode load (NVDEC) vs inference.
  3. Bottleneck map: which resource binds where.
  4. Capacity model fit: total_ms ~ f(N, decode_rate, inf_rate) from a
     few constants; report fit quality.
Usage: atlas_analyze.py atlas.csv
"""
import sys, csv, statistics as st
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
def f(r, k):
    try: return float(r[k])
    except: return 0.0

print(f"# Capacity Atlas — {len(rows)} cells\n")

# ---- 1. Bottleneck census ------------------------------------------------
census = defaultdict(int)
for r in rows: census[r['bottleneck']] += 1
print("## Bottleneck census")
for k, v in sorted(census.items(), key=lambda x: -x[1]):
    print(f"  {k:24s} {v:4d} cells ({100*v/len(rows):.0f}%)")
print()

# ---- 2. Max sustained N (hold>=95) per (decode, inference) ---------------
print("## Max N holding >=95% by (decode, skip) -> inference rate")
best = {}
for r in rows:
    key = (r['decode'], r['skip'], r['inf_per_s_per_stream'])
    if f(r, 'hold_pct') >= 95:
        best[key] = max(best.get(key, 0), int(r['N']))
for (dec, skip, inf), n in sorted(best.items(), key=lambda x: (x[0][0], -float(x[0][2]))):
    print(f"  decode={dec:7s} skip={skip:4s} inf={float(inf):6.3f}/s -> max {n} streams @>=95%")
print()

# ---- 3. Redundant pairs: same inference rate, different decode ----------
# group by (N, inference rate); within a group compare decode variants.
print("## Redundant-pair isolation (same inference rate, N; vary decode)")
groups = defaultdict(list)
for r in rows:
    groups[(r['N'], round(f(r, 'inf_per_s_per_stream'), 3))].append(r)
shown = 0
for (n, inf), cells in sorted(groups.items(), key=lambda x: (int(x[0][0]), -x[0][1])):
    if len(cells) < 2: continue
    if int(n) not in (32, 50, 75, 100): continue
    line = f"  N={n:>3} inf={inf:6.3f}/s: "
    parts = []
    for c in sorted(cells, key=lambda c: -f(c, 'decoded_per_s')):
        parts.append(f"{c['decode']}(nvdec {c['nvdec_pct']}%,tot {c['total_ms']}ms,hold {c['hold_pct']}%)")
    print(line + " | ".join(parts))
    shown += 1
    if shown >= 20: break
print()

# ---- 4. Capacity model fit ----------------------------------------------
# Serial batch cycle model: gpu_ms ~ a + b*batch; decode headroom from
# nvdec. Fit gpu_ms = base + per_img*(effective batch) using cells where
# the system is NOT ramp-limited (hold>=90).
print("## Capacity model — GPU batch time vs offered inference load")
# effective inference load per second = N * inf_rate
pts = [(f(r,'N')*f(r,'inf_per_s_per_stream'), f(r,'gpu_ms')) for r in rows
       if f(r,'hold_pct') >= 90 and f(r,'gpu_ms') > 0]
if len(pts) > 5:
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    n = len(pts); sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
    slope = (n*sxy - sx*sy)/(n*sxx - sx*sx) if (n*sxx-sx*sx) else 0
    inter = (sy - slope*sx)/n
    resid = [y-(inter+slope*x) for x,y in zip(xs,ys)]
    rms = (sum(e*e for e in resid)/n)**0.5
    print(f"  gpu_ms ~= {inter:.2f} + {slope:.5f} * (N*inf_rate)  [RMS resid {rms:.2f} ms, {n} pts]")
    print(f"  interpretation: base batch cost {inter:.2f} ms; each +1 inference/s of")
    print(f"  aggregate load adds {slope*1000:.2f} us of GPU cycle.")
print()

# ---- 5. NVDEC vs decode load (the ceiling) -------------------------------
print("## NVDEC utilization vs aggregate decode rate")
dec_pts = defaultdict(list)
for r in rows:
    agg_decode = f(r,'decoded_per_s')
    dec_pts[r['decode']].append((agg_decode, f(r,'nvdec_pct')))
for dec, pts in sorted(dec_pts.items()):
    pts = [p for p in pts if p[0] > 0]
    if not pts: continue
    hi = max(pts, key=lambda p: p[0])
    print(f"  decode={dec:7s}: peak {hi[0]:.0f} decoded/s at NVDEC {hi[1]:.0f}%")
