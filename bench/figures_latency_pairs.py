"""bench/figures_latency_pairs.py <ceiling cells.csv> <out_prefix>
Post-decode latency, PyCamTRT vs latency-tuned DeepStream, at every camera count where BOTH hold
(>= 95 % of the offered rate), one panel per resolution. Bars are labelled with the ms value and each
pair with DeepStream's latency divided by PyCamTRT's. Writes <out_prefix>.png and -dark.png at 2400x1350."""
import sys, csv
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CELLS, OUT = sys.argv[1], sys.argv[2]
A, B = ("pycamtrt", "PyCamTRT, full detect → crop → OCR cascade"), ("deepstream_tuned", "DeepStream 8.0, latency-tuned fp16, same cascade")
RES = [("720p", "720p"), ("1080p", "1080p"), ("4k", "4K")]
THEMES = {"light": dict(surface="#fcfcfb", ink="#0b0b0b", sec="#52514e", muted="#898781", grid="#e1e0d9", s=["#2a78d6", "#eb6834"]),
          "dark":  dict(surface="#1a1a19", ink="#ffffff", sec="#c3c2b7", muted="#898781", grid="#2c2c2a", s=["#3987e5", "#d95926"])}
cells = defaultdict(dict)
for r in csv.DictReader(open(CELLS)):
    try: cells[(r["res"], r["system"])][int(r["N"])] = (float(r["hold_pct"]), float(r["post_decode_ms"]))
    except ValueError: pass
pairs = {}
for rk, _ in RES:
    ns = sorted(n for n in cells[(rk, A[0])] if n in cells[(rk, B[0])] and cells[(rk, A[0])][n][0] >= 95 and cells[(rk, B[0])][n][0] >= 95)
    pairs[rk] = [(n, cells[(rk, A[0])][n][1], cells[(rk, B[0])][n][1]) for n in ns]
    print(rk, " ".join(f"{n}:{a:.1f}/{b:.1f}({b/a:.1f}x)" for n, a, b in pairs[rk]))
ratios = [b / a for rk in pairs for _, a, b in pairs[rk]]
for mode, t in THEMES.items():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
                         "text.color": t["ink"], "axes.edgecolor": t["grid"], "xtick.color": t["sec"], "ytick.color": t["muted"]})
    fig = plt.figure(figsize=(12, 6.75))
    fig.text(0.04, 0.93, "Post-decode latency where both systems hold — one RTX 3060 Ti", fontsize=18, fontweight="bold", va="center")
    fig.text(0.04, 0.875, f"Every frame of every camera decoded and inferred at 30 fps · same clip, farm, machine and hour · PyCamTRT is {min(ratios):.1f}× to {max(ratios):.1f}× lower, median {np.median(ratios):.1f}×",
             fontsize=10.5, color=t["sec"], va="center")
    widths = [max(len(pairs[rk]), 1) for rk, _ in RES]
    axes = fig.subplots(1, len(RES), gridspec_kw=dict(width_ratios=widths, wspace=0.12), squeeze=False)[0]
    fig.subplots_adjust(left=0.06, right=0.985, top=0.76, bottom=0.21)
    ymax = max(b for rk in pairs for _, a, b in pairs[rk]) * 1.3
    for ax, (rk, rl) in zip(axes, RES):
        rows = pairs[rk]; x = np.arange(len(rows)); w = 0.36
        ba = ax.bar(x - w / 2, [a for _, a, _ in rows], w - 0.03, color=t["s"][0], label=A[1], zorder=3)
        bb = ax.bar(x + w / 2, [b for _, _, b in rows], w - 0.03, color=t["s"][1], label=B[1], zorder=3)
        for bars, idx in ((ba, 1), (bb, 2)):
            for bar, row in zip(bars, rows):
                v = row[idx]; ax.text(bar.get_x() + bar.get_width() / 2, v + ymax * 0.012, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=t["ink"])
        for xi, (n, a, b) in zip(x, rows):
            ax.text(xi, max(a, b) + ymax * 0.09, f"{b / a:.1f}×", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=t["sec"])
        ax.set_xticks(x); ax.set_xticklabels([str(n) for n, _, _ in rows], fontsize=11)
        ax.set_xlabel("cameras", fontsize=10, color=t["sec"]); ax.set_title(rl, loc="left", fontsize=12.5, fontweight="bold")
        ax.set_ylim(0, ymax); ax.grid(axis="y", color=t["grid"], lw=1, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
    axes[0].set_ylabel("post-decode latency per frame, ms (median)", fontsize=10.5, color=t["sec"])
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.985, 0.845), frameon=False, fontsize=9.5, ncol=2)
    fig.text(0.06, 0.1, "Post-decode = time from a decoded frame to its result (queue + GPU); decode and network delivery, identical for both, are excluded.\n"
                         "Counts shown are the grid points where both systems sustain ≥95 % of the offered rate; 48 is the end of the 720p grid.\n"
                         "DeepStream tuned per NVIDIA's latency guidance (fp16 GIEs, 5 ms batch timeout, 30 ms jitter buffer); same plate cascade on both sides.",
             fontsize=8.6, color=t["muted"], va="top")
    fig.text(0.06, 0.022, "github.com/Matias-GONPO/PyCamTRT  ·  bench/results/ceiling_2026-09-19/cells.csv  ·  bench/METHODOLOGY.md", fontsize=9.5, color=t["sec"], va="center")
    fig.savefig(f"{OUT}{'-dark' if mode == 'dark' else ''}.png", dpi=200, facecolor=t["surface"]); plt.close(fig)
print("wrote", OUT + ".png", OUT + "-dark.png")
