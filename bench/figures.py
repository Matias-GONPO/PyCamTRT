"""bench/figures.py - regenerates the README figures (docs/img/*.png, light + dark).

Inputs: the head-to-head numbers below (bench/results/*, see METHODOLOGY.md) and the
capacity atlas v2 grids in bench/atlas/ (336 measured cells per resolution, RTX 3060 Ti).
Run from the repo root: python3 bench/figures.py   (needs pandas + matplotlib)
"""
import os
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "img")
ATLAS = os.path.join(ROOT, "bench", "atlas")
THEMES = {
  "light": dict(surface="#fcfcfb", ink="#0b0b0b", sec="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], gray="#c3c2b7"),
  "dark":  dict(surface="#1a1a19", ink="#ffffff", sec="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                s=["#3987e5", "#d95926", "#199e70", "#c98500"], gray="#52514e"),
}
def style(t):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 9.5,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["sec"], "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.edgecolor": t["axis"], "axes.linewidth": 1, "grid.color": t["grid"], "grid.linewidth": 1, "grid.linestyle": "-",
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
        "xtick.major.size": 0, "ytick.major.size": 0, "axes.titlelocation": "left", "axes.titleweight": "semibold"})
def clean(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
def line(ax, x, y, color, t, label=None, ls="-"):
    ax.plot(x, y, color=color, lw=2, ls=ls, solid_joinstyle="round", solid_capstyle="round", label=label, zorder=3)
    ax.plot(x, y, "o", ms=8, mfc=color, mec=t["surface"], mew=2, zorder=4)
def save(fig, name, mode):
    fig.savefig(f"{OUT}/{name}{'-dark' if mode=='dark' else ''}.png", dpi=160, bbox_inches="tight", pad_inches=0.25); plt.close(fig)

# ---------------- data ----------------
CAMS = [1, 4, 8, 16]
pyc = [0.056+1.624, 0.151+1.728, 0.435+1.771, 0.705+1.835]        # bench/results/pycamtrt_2026-09-04.csv queue+gpu
ds_tuned = [0.04+1.71, 5.10+2.22, 5.10+1.75, 1.72+2.41]            # bench/results/ds_sweep_console.log queue+gie
ds_default = [0.03+3.15, 8.87+6.62, 5.33+11.52, 22.05+21.49]
diy = {1: 1.771, 16: 2.188}                                         # bench/results/diy_2026-09-04.csv (detect only)
atlas = {r: pd.read_csv(f"{ATLAS}/atlas_{r}_v2.csv") for r in ["720p", "1080p", "1440p", "4k"]}

for mode, t in THEMES.items():
    style(t)
    # ---- Fig 1: head-to-head latency vs cameras ----
    fig, ax = plt.subplots(figsize=(8.2, 4.4)); clean(ax)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(CAMS)); ax.xaxis.set_major_formatter(FixedFormatter([str(c) for c in CAMS]))
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50])); ax.yaxis.set_major_formatter(FixedFormatter(["1", "2", "5", "10", "20", "50"]))
    line(ax, CAMS, ds_default, t["s"][2], t, "DeepStream 8.0, default config (cascade)")
    line(ax, CAMS, ds_tuned, t["s"][1], t, "DeepStream 8.0, latency-tuned fp16 (cascade)")
    line(ax, [1, 16], [diy[1], diy[16]], t["s"][3], t, "NVIDIA DIY: PyNvVideoCodec + TensorRT (detect only)", ls=(0, (2, 2)))
    line(ax, CAMS, pyc, t["s"][0], t, "PyCamTRT (full cascade)")
    for y, s in [(ds_default[-1], "43.5 ms"), (ds_tuned[-1], "4.1 ms"), (pyc[-1], "2.5 ms"), (diy[16], "2.2 ms")]:
        ax.annotate(s, (16, y), xytext=(8, 0), textcoords="offset points", va="center", color=t["sec"], fontsize=9)
    ax.set_xlim(0.9, 22); ax.set_ylim(1, 70)
    ax.set_xlabel("concurrent cameras (720p H.264, 30 fps each)"); ax.set_ylabel("post-decode latency per frame, ms (median)")
    ax.set_title("Same GPU, same hour: post-decode latency vs. camera count")
    ax.text(0.0, -0.2, "Not on the chart: naive Python (OpenCV + torch) 13.1 ms at 1 camera, one process per camera; the modern Python stack\n"
            "(ultralytics + PaddleOCR) 13.6 ms at 1 camera and collapses at 4 (~86 fps ceiling). RTX 3060 Ti, live RTSP, bench/METHODOLOGY.md.",
            transform=ax.transAxes, fontsize=8, color=t["muted"], va="top")
    ax.legend(loc="upper left", handlelength=1.6); save(fig, "bench_latency_vs_cameras", mode)

    # ---- Fig 2: the decode dial moves the wall (small multiples per resolution) ----
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
    for ax, (res, d) in zip(axes, atlas.items()):
        clean(ax); ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator([1, 4, 16, 50, 100])); ax.xaxis.set_major_formatter(FixedFormatter(["1", "4", "16", "50", "100"]))
        ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 50, 200])); ax.yaxis.set_major_formatter(FixedFormatter(["2", "5", "10", "50", "200"]))
        for dec, ci, lab in [("all", 0, "all frames decoded"), ("key30", 1, "keyframes only (1 per 30)")]:
            r = d[(d.decode == dec) & (d.skip == 1) & (d.total_ms > 0)].sort_values("N")
            ax.plot(r.N, r.total_ms, color=t["s"][ci], lw=2, label=lab if ax is axes[0] else None, zorder=3)
            hold = r[r.hold_pct >= 95]; miss = r[r.hold_pct < 95]
            ax.plot(hold.N, hold.total_ms, "o", ms=8, mfc=t["s"][ci], mec=t["surface"], mew=2, zorder=4)
            ax.plot(miss.N, miss.total_ms, "o", ms=8, mfc=t["surface"], mec=t["s"][ci], mew=2, zorder=4)
        ax.set_title(res.upper() if res == "4k" else res); ax.set_xlim(0.8, 130); ax.set_ylim(1.5, 300)
        ax.set_xlabel("cameras")
    axes[0].set_ylabel("post-decode ms per frame")
    fig.suptitle("The decode dial moves the wall: every frame vs. keyframes only, per resolution (RTX 3060 Ti, capacity atlas v2)", x=0.01, ha="left", fontsize=11, fontweight="semibold", color=t["ink"])
    fig.legend(loc="lower left", bbox_to_anchor=(0.01, -0.06), ncol=2, handlelength=1.6)
    fig.text(0.99, -0.04, "filled marker = holds ≥95% of the offered rate · hollow = falls behind · missing = out of memory", ha="right", fontsize=8, color=t["muted"])
    fig.subplots_adjust(wspace=0.12, top=0.82)
    # v2-atlas figure: its every-frame lines predate the v0.4.0 decoder fix. Not built by default;
    # set ATLAS_V2_FIGURES=1 to regenerate it (or point `atlas` at a v3 atlas once it exists).
    save(fig, "dials_decode_wall", mode) if os.environ.get("ATLAS_V2_FIGURES") else plt.close(fig)

    # ---- Fig 3: inference dial buys headroom, not speed ----
    d = atlas["1080p"]; r = d[(d.decode == "all") & (d.N == 16)].sort_values("skip")
    fig, ax = plt.subplots(figsize=(8.2, 3.9)); clean(ax); ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 15, 30, 60, 150])); ax.xaxis.set_major_formatter(FixedFormatter(["1", "2", "4", "8", "15", "30", "60", "150"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    line(ax, r.skip, r.sm_pct, t["s"][0], t, "GPU SM utilization (inference)")
    line(ax, r.skip, r.nvdec_pct, t["s"][1], t, "NVDEC utilization (decode)")
    ax.set_ylim(0, 100); ax.set_xlim(0.9, 180)
    ax.set_xlabel("inference rate: infer every k-th decoded frame (skip)"); ax.set_ylabel("utilization, %")
    ax.set_title("The inference dial buys headroom, not speed (16 cameras, 1080p, all frames decoded)")
    ax.text(0.99, 0.6, f"post-decode latency across all 14 points:\n{r.total_ms.min():.1f} – {r.total_ms.max():.1f} ms", transform=ax.transAxes, ha="right", color=t["sec"], fontsize=9)
    ax.legend(loc="upper right", handlelength=1.6); save(fig, "dials_inference_headroom", mode)

    # ---- Fig 4: cameras that hold real time, by profile ----
    profiles = [("all frames · every frame inferred", "all", 1), ("all frames · infer every 4th", "all", 4), ("all frames · infer every 30th", "all", 30),
                ("keyframes only (1 per 30) · every keyframe", "key30", 1), ("keyframes only (1 per 15) · every keyframe", "key15", 1)]
    def max_hold(d, dec, sk):
        ok = d[(d.decode == dec) & (d.skip == sk) & (d.hold_pct >= 95) & (d.total_ms > 0) & (d.total_ms < 10)]
        return int(ok.N.max()) if len(ok) else 0
    v720 = [max_hold(atlas["720p"], dec, sk) for _, dec, sk in profiles]; v1080 = [max_hold(atlas["1080p"], dec, sk) for _, dec, sk in profiles]
    fig, ax = plt.subplots(figsize=(8.2, 4.2)); clean(ax); ax.grid(axis="x"); ax.grid(False, axis="y")
    y = np.arange(len(profiles)); h = 0.30
    b1 = ax.barh(y + h/2, v720, h - 0.03, color=t["s"][0], label="720p", zorder=3)
    b2 = ax.barh(y - h/2, v1080, h - 0.03, color=t["s"][1], label="1080p", zorder=3)
    for bars, vals in [(b1, v720), (b2, v1080)]:
        for b, v in zip(bars, vals): ax.text(v + 1.5, b.get_y() + b.get_height()/2, str(v), va="center", fontsize=9, color=t["sec"])
    ax.set_yticks(y); ax.set_yticklabels([p[0] for p in profiles]); ax.invert_yaxis(); ax.set_xlim(0, 112)
    ax.set_xlabel("cameras that hold real time on one RTX 3060 Ti (highest atlas grid point at ≥95% of the offered rate)")
    ax.set_title("Turning the dials: cameras per GPU for tasks that need few inferences")
    ax.legend(loc="upper right", handlelength=1.2)
    save(fig, "dials_capacity_profiles", mode) if os.environ.get("ATLAS_V2_FIGURES") else plt.close(fig)  # v2-atlas figure, see above
print("figures written to", OUT)
