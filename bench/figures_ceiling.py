"""bench/figures_ceiling.py - the cross-system ceiling figure from a ceiling campaign's cells.csv.

Usage: python3 bench/figures_ceiling.py bench/results/ceiling_<date>/cells.csv [out_prefix]
For each resolution and system: the highest camera count in the grid that sustained >=95% of the
offered rate (30 fps per camera, every frame decoded and inferred). Bars are labelled with that
count and the post-decode latency measured there. Writes <out_prefix>.png and <out_prefix>-dark.png
(default docs/img/ceiling_systems) plus a LinkedIn-sized copy next to the light one.
"""
import sys, os, csv, glob
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CELLS = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "img", "ceiling_systems")
HOLD = 95.0
# Colour follows the system (slot 1 blue, slot 2 orange, slot 4 yellow), whichever subset is drawn.
# SYSTEMS=pycamtrt,deepstream_tuned narrows the chart (LinkedIn version). DeepStream's default fp32
# configuration is in cells.csv but not charted: it is GPU-bound and erratic above 32 cameras.
ALL_SYSTEMS = [("pycamtrt", "PyCamTRT (full cascade)", 0), ("deepstream_tuned", "DeepStream 8.0, latency-tuned fp16 (cascade)", 1),
               ("diy_detect_only", "NVIDIA DIY: PyNvVideoCodec + TensorRT (detect only)", 3)]
_want = os.environ.get("SYSTEMS")
SYSTEMS = [x for x in ALL_SYSTEMS if not _want or x[0] in _want.split(",")]
RES = [("720p", "720p"), ("1080p", "1080p"), ("4k", "4K")]
THEMES = {"light": dict(surface="#fcfcfb", ink="#0b0b0b", sec="#52514e", muted="#898781", grid="#e1e0d9", s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
          "dark":  dict(surface="#1a1a19", ink="#ffffff", sec="#c3c2b7", muted="#898781", grid="#2c2c2a", s=["#3987e5", "#d95926", "#199e70", "#c98500"])}

rows = list(csv.DictReader(open(CELLS)))
cells = defaultdict(list)
for r in rows:
    try: cells[(r["res"], r["system"])].append((int(r["N"]), float(r["hold_pct"]), float(r["post_decode_ms"]) if r["post_decode_ms"] not in ("NA", "") else float("nan")))
    except ValueError: pass
def ceiling(res, sysk):
    pts = sorted(cells.get((res, sysk), []))
    held = [p for p in pts if p[1] >= HOLD]
    if not held: return 0, float("nan"), pts[0][0] if pts else None
    n, h, ms = held[-1]
    nxt = [p for p in pts if p[0] > n]
    return n, ms, (nxt[0][0] if nxt else None)   # (max held, latency there, next grid point that failed or None=grid-limited)

table = {(rk, sk): ceiling(rk, sk) for rk, _ in RES for sk, _, _ in SYSTEMS}
for rk, rl in RES:
    print(rl, " | ".join(f"{sl.split(' (')[0]}: {table[(rk, sk)][0]} ({table[(rk, sk)][1]:.1f} ms, next {table[(rk, sk)][2]})" for sk, sl, _ in SYSTEMS))

for mode, t in THEMES.items():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
                         "text.color": t["ink"], "axes.edgecolor": t["grid"], "xtick.color": t["sec"], "ytick.color": t["muted"]})
    fig = plt.figure(figsize=(12, 6.75))
    fig.text(0.04, 0.93, "Cameras held with full inference at 30 fps, per system — one RTX 3060 Ti", fontsize=18, fontweight="bold", va="center")
    fig.text(0.04, 0.875, "Same real plate recording at each resolution, same machine, same hour · every frame decoded and inferred · holds = ≥95% of the offered rate",
             fontsize=10.5, color=t["sec"], va="center")
    ax = fig.add_axes([0.07, 0.225, 0.90, 0.575])
    ns = len(SYSTEMS); w = 0.8 / max(ns, 1) if ns > 2 else 0.3; x = np.arange(len(RES))
    ymax = max(v[0] for v in table.values()) or 10
    for j, (sk, sl, ci) in enumerate(SYSTEMS):
        vals = [table[(rk, sk)][0] for rk, _ in RES]
        bars = ax.bar(x + (j - (ns - 1) / 2) * w, vals, w - 0.03, color=t["s"][ci], label=sl, zorder=3)
        for b, (rk, _) in zip(bars, RES):
            n, ms, nxt = table[(rk, sk)]
            ax.text(b.get_x() + b.get_width() / 2, n + ymax * 0.012, str(n) if n else (f"<{nxt}" if nxt else "0"), ha="center", va="bottom", fontsize=13, fontweight="bold", color=t["ink"])
            if n and ms == ms:
                ax.text(b.get_x() + b.get_width() / 2, n + ymax * 0.075, f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms", ha="center", va="bottom", fontsize=8.5, color=t["muted"])
    ax.set_xticks(x); ax.set_xticklabels([rl for _, rl in RES], fontsize=13)
    ax.set_ylim(0, ymax * 1.25); ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("cameras held (highest measured grid point)", fontsize=11, color=t["sec"])
    ax.grid(axis="y", color=t["grid"], lw=1, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    fig.text(0.07, 0.115, "Grey = post-decode latency per frame at that count. Grid: 720p 8–48 · 1080p 8–32 · 4K 2–8 cameras; a bar at the top of its grid is grid-limited, not a ceiling.\n"
                          "DeepStream tuned per NVIDIA's latency guidance (fp16 GIEs, 5 ms batch timeout, 30 ms jitter buffer); its default fp32 config held 24/24/6 at 63/82/18 ms. DIY runs no OCR cascade.\n"
                          "Python stacks (OpenCV + torch, ultralytics + PaddleOCR) collapse below 4 cameras at 720p and are not on the chart.",
             fontsize=8.6, color=t["muted"], va="top")
    fig.text(0.07, 0.03, "github.com/Matias-GONPO/PyCamTRT  ·  bench/run_ceiling_campaign.sh  ·  bench/METHODOLOGY.md", fontsize=9.5, color=t["sec"], va="center")
    fig.savefig(f"{OUT}{'-dark' if mode == 'dark' else ''}.png", dpi=200, facecolor=t["surface"]); plt.close(fig)
print("wrote", OUT + ".png", OUT + "-dark.png")
