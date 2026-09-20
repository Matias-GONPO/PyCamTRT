"""bench/figures_max_streams.py [out_prefix]
Cameras held per resolution, every frame decoded and inferred versus keyframes only, on the v0.4.0 decoder.
Every-frame bars: the finest grid available - the ceiling campaign (bench/results/ceiling_2026-09-19/cells.csv,
8-camera steps) for 720p/1080p/4K and the validation atlas for 1440p; keyframe bars: the validation atlas
(bench/atlas/runs/2026-09-19_mini/<res>/atlas.csv, key30 rows). Holds = >= 95 % of the offered rate.
Writes <out_prefix>.png and -dark.png (default docs/img/max_streams_every_vs_keyframes)."""
import sys, os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "docs", "img", "max_streams_every_vs_keyframes")
CAMPAIGN = os.path.join(REPO, "bench", "results", "ceiling_2026-09-19", "cells.csv")
MINI = os.path.join(REPO, "bench", "atlas", "runs", "2026-09-19_mini")
RES = [("720p", "720p"), ("1080p", "1080p"), ("1440p", "1440p"), ("4k", "4K")]
THEMES = {"light": dict(surface="#fcfcfb", ink="#0b0b0b", sec="#52514e", muted="#898781", grid="#e1e0d9", s=["#2a78d6", "#eb6834"]),
          "dark":  dict(surface="#1a1a19", ink="#ffffff", sec="#c3c2b7", muted="#898781", grid="#2c2c2a", s=["#3987e5", "#d95926"])}
camp = [r for r in csv.DictReader(open(CAMPAIGN)) if r["system"] == "pycamtrt"]
rows = []
for key, label in RES:
    atlas = list(csv.DictReader(open(os.path.join(MINI, key, "atlas.csv"))))
    if key == "1440p":
        held = [(int(r["N"]), float(r["total_ms"])) for r in atlas if r["decode"] == "all" and r["skip"] == "1" and float(r["hold_pct"]) >= 95]
    else:
        held = [(int(r["N"]), float(r["post_decode_ms"])) for r in camp if r["res"] == key and float(r["hold_pct"]) >= 95]
    n_all, ms_all = max(held)
    kept = [(int(r["N"]), float(r["total_ms"])) for r in atlas if r["decode"] == "key30" and r["skip"] == "1" and float(r["hold_pct"]) >= 95]
    n_key, ms_key = max(kept)
    rows.append(dict(res=label, all=n_all, all_ms=ms_all, key=n_key, key_ms=ms_key)); print(rows[-1])
for mode, t in THEMES.items():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
                         "text.color": t["ink"], "axes.edgecolor": t["grid"], "xtick.color": t["sec"], "ytick.color": t["muted"]})
    fig = plt.figure(figsize=(12, 6.75))
    fig.text(0.04, 0.93, "How many cameras one GPU holds in real time — every frame vs. keyframes only", fontsize=18, fontweight="bold", va="center")
    fig.text(0.04, 0.875, "PyCamTRT v0.4.0 on one RTX 3060 Ti · live H.264 RTSP at 30 fps per camera · plate detector + OCR cascade · measured, not extrapolated",
             fontsize=10.5, color=t["sec"], va="center")
    ax = fig.add_axes([0.07, 0.225, 0.90, 0.565])
    x = np.arange(len(rows)); w = 0.36
    b1 = ax.bar(x - w/2, [r["all"] for r in rows], w - 0.03, color=t["s"][0], label="every frame decoded and inferred (30 inferences/s per camera)", zorder=3)
    b2 = ax.bar(x + w/2, [r["key"] for r in rows], w - 0.03, color=t["s"][1], label="keyframes only decoded and inferred (1 inference/s per camera)", zorder=3)
    for bars, k in ((b1, "all"), (b2, "key")):
        for b, r in zip(bars, rows):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{r[k]}", ha="center", va="bottom", fontsize=15, fontweight="bold", color=t["ink"])
            ms = r[k + "_ms"]; ax.text(b.get_x() + b.get_width()/2, b.get_height() + 9, f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms", ha="center", va="bottom", fontsize=9.5, color=t["muted"])
    ax.set_xticks(x); ax.set_xticklabels([r["res"] for r in rows], fontsize=13)
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("cameras held (≥95% of the offered rate)", fontsize=11, color=t["sec"])
    ax.grid(axis="y", color=t["grid"], lw=1, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    fig.text(0.07, 0.12, "Bars: the highest measured grid point still sustaining ≥95% of the offered rate; grey = post-decode latency there. Every-frame bars from the ceiling campaign\n"
                         "(8-camera steps; 48 is the end of that grid) and, for 1440p, the validation atlas; keyframe bars from the validation atlas (grid 32, 50, 75, 100). Every-frame decode\n"
                         "stops where the NVDEC hardware does (97–100% decoder load); keyframes-only decode moves that wall: 1.5–5× more cameras for tasks that need one look per second.",
             fontsize=8.8, color=t["muted"], va="top")
    fig.text(0.07, 0.03, "github.com/Matias-GONPO/PyCamTRT  ·  bench/results/ceiling_2026-09-19/cells.csv  ·  bench/atlas/runs/2026-09-19_mini", fontsize=9.5, color=t["sec"], va="center")
    fig.savefig(f"{OUT}{'-dark' if mode == 'dark' else ''}.png", dpi=200, facecolor=t["surface"]); plt.close(fig)
print("wrote", OUT + ".png")
