"""figures_final.py <campaign_final/cells.csv> <out_prefix>
Same-night five-system campaign: per resolution, decoded % of offered and post-decode latency versus cameras,
for PyCamTRT original, PyCamTRT (Option A), DeepStream tuned, DeepStream default, NVIDIA DIY. Prints the
held-ceiling table (>=95 %). Light + dark PNGs."""
import sys, csv
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELLS, OUT = sys.argv[1], sys.argv[2]
import os
# (system key, label, colour slot). DeepStream's default fp32 config is in cells.csv but not charted.
SYS = [("pycamtrt", "PyCamTRT (full cascade)", 0)] + ([("pycamtrt_original", "PyCamTRT before the v0.4.0 decoder fix", 4)] if os.environ.get("INCLUDE_ORIGINAL") == "1" else []) + [
       ("deepstream_tuned", "DeepStream 8.0, latency-tuned fp16", 1), ("diy_detect_only", "NVIDIA DIY, detect only", 3)]
RES = [("720p", "720p"), ("1080p", "1080p"), ("4k", "4K")]
THEMES = {"light": dict(surface="#fcfcfb", ink="#0b0b0b", sec="#52514e", muted="#898781", grid="#e1e0d9", s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]),
          "dark":  dict(surface="#1a1a19", ink="#ffffff", sec="#c3c2b7", muted="#898781", grid="#2c2c2a", s=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"])}
rows = [r for r in csv.DictReader(open(CELLS))]
cells = defaultdict(dict)
for r in rows:
    try: cells[(r["res"], r["system"])][int(r["N"])] = r
    except ValueError: pass
def f(r, k):
    try: return float(r[k])
    except (KeyError, ValueError, TypeError): return float("nan")
RES = [x for x in RES if any(k[0] == x[0] for k in cells)]

print("held ceiling (>=95 % of offered), post-decode ms there, NVDEC % there")
for rk, rl in RES:
    for sk, sl, _ in SYS:
        held = [(n, r) for n, r in sorted(cells[(rk, sk)].items()) if f(r, "hold_pct") >= 95]
        if held: n, r = held[-1]; print(f"  {rl:5} {sl:36} {n:3}  {f(r,'post_decode_ms'):6.1f} ms  nvdec {f(r,'nvdec_util'):5.1f}")
        else: print(f"  {rl:5} {sl:36}   0")

for mode, t in THEMES.items():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
                         "text.color": t["ink"], "axes.edgecolor": t["grid"], "xtick.color": t["sec"], "ytick.color": t["muted"]})
    fig, axes = plt.subplots(2, len(RES), figsize=(5.6 * len(RES), 8.6), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.08, hspace=0.42, wspace=0.18)
    fig.text(0.04, 0.955, "Every frame decoded and inferred at 30 fps per camera: how far each system goes on one RTX 3060 Ti", fontsize=17, fontweight="bold", va="center")
    fig.text(0.04, 0.915, "Plate cascade (detector + LPRNet) on the same real plate recording per resolution, same farm and hour · holds = ≥95 % of the offered rate · DIY = detector only, no cascade",
             fontsize=10, color=t["sec"], va="center")
    for c, (rk, rl) in enumerate(RES):
        for rrow, (key, ylabel, ref) in enumerate([("hold_pct", "decoded, % of offered rate", 95), ("post_decode_ms", "post-decode latency, ms (median)", None)]):
            ax = axes[rrow][c]; ax.grid(axis="y", color=t["grid"], lw=0.8, zorder=0); ax.set_axisbelow(True)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            if ref: ax.axhline(ref, color=t["muted"], lw=1, ls=(0, (4, 4)), zorder=1)
            for j, (sk, sl, ci) in enumerate(SYS):
                pts = sorted((n, f(r, key)) for n, r in cells[(rk, sk)].items())
                pts = [p for p in pts if p[1] == p[1]]
                if not pts: continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=t["s"][ci], lw=2, zorder=3, label=sl if (rrow == 0 and c == 0) else None)
                ax.plot(xs, ys, "o", ms=7, mfc=t["s"][ci], mec=t["surface"], mew=2, zorder=4)
            allx = sorted({n for sk, _, _ in SYS for n in cells[(rk, sk)]})
            ax.set_xticks(allx); ax.set_xlim(min(allx) - 1, max(allx) + 2)
            ax.set_xlabel("cameras", fontsize=10, color=t["sec"]); ax.set_ylabel(ylabel, fontsize=10, color=t["sec"]); ax.tick_params(labelsize=9)
            if key == "hold_pct": ax.set_ylim(0, 105)
            else: ax.set_yscale("log"); ax.set_ylim(0.5, 300); ax.set_yticks([0.5, 1, 3, 10, 30, 100, 300]); ax.set_yticklabels(["0.5", "1", "3", "10", "30", "100", "300"])
            ax.set_title(rl, fontsize=12, loc="left")
    axes[0][0].legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.savefig(f"{OUT}{'-dark' if mode == 'dark' else ''}.png", dpi=160); print("wrote", f"{OUT}{'-dark' if mode == 'dark' else ''}.png")
