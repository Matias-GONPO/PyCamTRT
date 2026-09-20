"""bench/atlas/atlas_compare.py <old.csv> <new.csv> - the cells present in both atlases (keyed N,decode,skip),
side by side: hold %, total post-decode ms, NVDEC %, bottleneck. Ends with a count of cells that changed class."""
import sys, csv
old = {(r["N"], r["decode"], r["skip"]): r for r in csv.DictReader(open(sys.argv[1]))}
new = {(r["N"], r["decode"], r["skip"]): r for r in csv.DictReader(open(sys.argv[2]))}
keys = [k for k in new if k in old]
keys.sort(key=lambda k: (k[1], int(k[0]), int(k[2])))
print(f"{'N':>4} {'decode':7} {'skip':>4} | {'hold old':>8} {'hold new':>8} | {'ms old':>7} {'ms new':>7} | {'nvdec old':>9} {'nvdec new':>9} | bottleneck old -> new")
gained = lost = 0
for k in keys:
    o, n = old[k], new[k]
    ho, hn = float(o["hold_pct"]), float(n["hold_pct"])
    if ho < 95 <= hn: gained += 1
    if hn < 95 <= ho: lost += 1
    print(f"{k[0]:>4} {k[1]:7} {k[2]:>4} | {ho:8.1f} {hn:8.1f} | {float(o['total_ms']):7.1f} {float(n['total_ms']):7.1f} | {o['nvdec_pct']:>9} {n['nvdec_pct']:>9} | {o['bottleneck']} -> {n['bottleneck']}")
print(f"\n{len(keys)} shared cells; now hold but did not: {gained}; held but no longer: {lost}")
