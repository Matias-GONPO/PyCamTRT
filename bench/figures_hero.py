"""bench/figures_hero.py [out.png] - the at-a-glance figure: post-decode latency at 16 cameras (4 September campaign,
bench/results/*_2026-09-04*) and cameras held per GPU as the dials are turned (validation atlas 2026-09-19).
Numbers are inlined; regenerate after re-measuring. Default output docs/img/hero_release.png."""
"""One LinkedIn feed image (1200x675 @2x) for the PyCamTRT 0.4.0 release post.
Numbers: bench/results (head-to-head, 720p live RTSP) and the 720p capacity atlas v2."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "img", "hero_release.png")
SURF, INK, SEC, MUTED, GRID, ACCENT, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": SEC})
fig = plt.figure(figsize=(12, 6.75))
fig.text(0.04, 0.93, "PyCamTRT 0.4.0 — zero-copy cascade inference over RTSP camera fleets", fontsize=18, fontweight="bold", va="center")
fig.text(0.04, 0.875, "Python describes the graph; a fused C++/CUDA data plane runs it.  RTX 3060 Ti · live 720p H.264 RTSP · every number measured on one machine, same hour",
         fontsize=10.5, color=SEC, va="center")

def panel(rect):
    ax = fig.add_axes(rect)
    ax.grid(axis="x", color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    return ax

# ---- left: head-to-head at 16 cameras ----
ax = panel([0.17, 0.27, 0.30, 0.47])
systems = ["DeepStream 8.0\ndefault config", "DeepStream 8.0\nlatency-tuned fp16", "PyCamTRT"]
vals = [43.5, 4.1, 2.5]; cols = [GRAY, GRAY, ACCENT]
bars = ax.barh(range(3), vals, height=0.52, color=cols, zorder=3)
for b, v in zip(bars, vals):
    ax.text(v + 0.9, b.get_y() + b.get_height()/2, f"{v:.1f} ms", va="center", fontsize=13, fontweight="bold" if v == 2.5 else "normal", color=INK)
ax.set_yticks(range(3)); ax.set_yticklabels(systems, fontsize=12); ax.invert_yaxis()
ax.set_xlim(0, 52); ax.set_xticks([0, 10, 20, 30, 40, 50])
ax.set_title("16 cameras, detect → crop → OCR cascade\npost-decode latency per frame, ms", loc="left", fontsize=12.5, fontweight="bold", pad=10)
fig.text(0.17, 0.165, "Python stacks do not reach 16 cameras: OpenCV + torch 13.1 ms at one\n"
                      "camera, one process per camera; ultralytics + PaddleOCR 13.6 ms at one\n"
                      "camera, collapses at four. DeepStream tuned per NVIDIA's latency guidance.",
         fontsize=8.6, color=MUTED, va="top")

# ---- right: cameras per GPU by profile (720p atlas) ----
ax2 = panel([0.68, 0.27, 0.27, 0.47])
profiles = ["every frame decoded,\nevery frame inferred", "every frame decoded,\ninfer 1 in 3", "keyframes only (1 per 30),\nevery keyframe inferred", "keyframes only (1 per 15),\nevery keyframe inferred"]
cams = [50, 50, 100, 100]   # validation atlas 2026-09-19 (v0.4.0 decoder), 720p grid points; 1 per 15 from the v2 atlas
bars = ax2.barh(range(4), cams, height=0.52, color=ACCENT, zorder=3)
for b, v in zip(bars, cams):
    ax2.text(v + 2, b.get_y() + b.get_height()/2, f"{v} cameras", va="center", fontsize=13, fontweight="bold", color=INK)
ax2.set_yticks(range(4)); ax2.set_yticklabels(profiles, fontsize=10.5); ax2.invert_yaxis()
ax2.set_xlim(0, 138); ax2.set_xticks([0, 25, 50, 75, 100])
fig.text(0.585, 0.785, "Same GPU, dials set for low inference rates\ncameras held in real time, at ~2 ms per frame", fontsize=12.5, fontweight="bold", va="center")
fig.text(0.57, 0.165, "720p, capacity atlas grid points (1 per 15: v2 atlas); every bar holds ≥95% of\n"
                      "the offered rate. Occupancy, counting and presence tasks need\n"
                      "one look per second, or less.",
         fontsize=8.6, color=MUTED, va="top")

fig.text(0.04, 0.05, "github.com/Matias-GONPO/PyCamTRT   ·   pip install .   ·   bench/METHODOLOGY.md reproduces every number", fontsize=10, color=SEC, va="center")
fig.savefig(OUT, dpi=200, facecolor=SURF)
print("wrote", OUT)
