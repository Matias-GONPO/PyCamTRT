"""bench/atlas/atlas_vs_deepstream.py <runs dir> [out_prefix]
For each resolution folder under <runs dir> that has both atlas.csv (PyCamTRT) and ds_dials.csv (DeepStream):
a shared-cell table (hold %, post-decode ms, NVDEC %) and, per inference rate per camera, the highest camera
count each system held (>= 95 % of the offered rate). With out_prefix, draws the dials chart: cameras held
versus inferences per second per camera, one line per system, one panel per resolution (light + dark PNG)."""
import sys, os, csv
from collections import defaultdict
RUNS = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
RES = [r for r in ("720p", "1080p", "1440p", "4k") if os.path.isfile(f"{RUNS}/{r}/atlas.csv") and os.path.isfile(f"{RUNS}/{r}/ds_dials.csv")]
_only = os.environ.get("RES_FILTER")  # e.g. RES_FILTER=720p,1080p for a two-panel version
if _only: RES = [r for r in RES if r in _only.split(",")]
def load(path, fps_key, ms_key):
    d = {}
    for r in csv.DictReader(open(path)):
        try: d[(int(r["N"]), r["decode"], int(r["skip"]))] = (float(r["hold_pct"]), float(r[ms_key]) if r[ms_key] not in ("NA", "") else float("nan"), float(r["nvdec_pct"]), float(r["inf_per_s_per_stream"]))
        except ValueError: pass
    return d
held = {}   # (res, system, inf_rate) -> max N held
raw = {}    # res -> (py cells, ds cells)
below = {}  # (res, system, inf_rate) -> smallest tested N when nothing held
for res in RES:
    py = load(f"{RUNS}/{res}/atlas.csv", "decoded_per_s", "total_ms"); ds = load(f"{RUNS}/{res}/ds_dials.csv", "achieved_fps", "post_decode_ms"); raw[res] = (py, ds)
    print(f"\n=== {res}: shared cells (hold %, post-decode ms, NVDEC %)  PyCamTRT | DeepStream tuned   [DeepStream ms at skip>1 averages inferred AND skipped frames; compare latency at skip 1 only]")
    print(f"{'N':>4} {'decode':6} {'skip':>4} {'inf/s':>6} | {'hold':>6} {'ms':>6} {'nvdec':>5} | {'hold':>6} {'ms':>6} {'nvdec':>5}")
    for k in sorted(set(py) & set(ds), key=lambda k: (k[1], k[2], k[0])):
        a, b = py[k], ds[k]
        print(f"{k[0]:>4} {k[1]:6} {k[2]:>4} {a[3]:6.1f} | {a[0]:6.1f} {a[1]:6.1f} {a[2]:5.0f} | {b[0]:6.1f} {b[1]:6.1f} {b[2]:5.0f}")
    for name, d in (("pycamtrt", py), ("deepstream", ds)):
        by_rate = defaultdict(list)
        for (n, dec, skip), (hold, ms, nv, inf) in d.items(): by_rate[round(inf, 2)].append((n, hold))
        for rate, cells in by_rate.items():
            ok = [n for n, h in cells if h >= 95]; held[(res, name, rate)] = max(ok) if ok else 0
            if not ok: below[(res, name, rate)] = min(n for n, _ in cells)   # nothing held: label "<smallest tested N"
    rates = sorted({r for (rr, _, r) in held if rr == res}, reverse=True)
    print(f"  cameras held per inference rate (inf/s per camera): " + "  ".join(f"{r:g}/s: Py {held.get((res,'pycamtrt',r),'-')} DS {held.get((res,'deepstream',r),'-')}" for r in rates))
if OUT:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    THEMES = {"light": dict(surface="#fcfcfb", ink="#0b0b0b", sec="#52514e", muted="#898781", grid="#e1e0d9", s=["#2a78d6", "#eb6834"]),
              "dark":  dict(surface="#1a1a19", ink="#ffffff", sec="#c3c2b7", muted="#898781", grid="#2c2c2a", s=["#3987e5", "#d95926"])}
    for mode, t in THEMES.items():
        plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
                             "text.color": t["ink"], "axes.edgecolor": t["grid"], "xtick.color": t["sec"], "ytick.color": t["muted"]})
        fig, grid = plt.subplots(2, len(RES), figsize=(max(12.0, 4.6 * len(RES) + 1.6), 10.5), squeeze=False); axes = grid[0]; laxes = grid[1]
        fig.subplots_adjust(left=0.07, right=0.985, top=0.835, bottom=0.14, wspace=0.18, hspace=0.42)
        fig.text(0.04, 0.955, "PyCamTRT vs DeepStream on the same dials — capacity, then latency — one RTX 3060 Ti", fontsize=17, fontweight="bold", va="center")
        fig.text(0.04, 0.92, "Plate cascade · top: cameras held at each inference rate (every frame decoded, or K = keyframes only) · bottom: post-decode latency at full rate where both hold · same farm, content and hour",
                 fontsize=10, color=t["sec"], va="center")
        for lax, res in zip(laxes, RES):
            py, ds = raw[res]
            ns = sorted(n for (n, dec, skip) in py if dec == "all" and skip == 1 and (n, "all", 1) in ds and py[(n, "all", 1)][0] >= 95 and ds[(n, "all", 1)][0] >= 95)
            xs = list(range(len(ns))); w = 0.38
            for j, (cells, label) in enumerate(((py, "PyCamTRT"), (ds, "DeepStream"))):
                ys = [cells[(n, "all", 1)][1] for n in ns]; px = [x + (j - 0.5) * w for x in xs]
                lax.bar(px, ys, w - 0.04, color=t["s"][j], zorder=3)
                for x, y in zip(px, ys): lax.text(x, y + 0.25, f"{y:.1f}", ha="center", va="bottom", fontsize=8.5, color=t["ink"])
            lax.set_xticks(xs); lax.set_xticklabels([str(n) for n in ns], fontsize=10); lax.set_xlabel("cameras, every frame inferred at 30 fps", fontsize=9.5, color=t["sec"])
            lax.set_title(res.upper() if res == "4k" else res, loc="left", fontsize=12.5, fontweight="bold")
            lax.grid(axis="y", color=t["grid"], lw=1, zorder=0); lax.set_axisbelow(True); lax.set_ylim(0, 14)
            for sp in ("top", "right"): lax.spines[sp].set_visible(False)
        laxes[0].set_ylabel("post-decode latency per frame, ms", fontsize=10, color=t["sec"])
        for ax, res in zip(axes, RES):
            rates = sorted({r for (rr, _, r) in held if rr == res}, reverse=True)
            xs = list(range(len(rates))); w = 0.38
            for j, (name, label) in enumerate((("pycamtrt", "PyCamTRT"), ("deepstream", "DeepStream 8.0, latency-tuned fp16"))):
                pts = [(x, held[(res, name, r)]) for x, r in zip(xs, rates) if (res, name, r) in held]
                if not pts: continue
                px, py_ = zip(*pts); px = [x + (j - 0.5) * w for x in px]
                ax.bar(px, py_, w - 0.04, color=t["s"][j], zorder=3, label=label if ax is axes[0] else None)
                for x, y, r in zip(px, py_, [r for r in rates if (res, name, r) in held]):
                    ax.text(x, y + 1.5, str(y) if y else f"<{below[(res, name, r)]}", ha="center", va="bottom", fontsize=8.5, color=t["ink"])
            ax.set_xticks(xs); ax.set_xticklabels([("K" if abs(r - 1.0) < 0.01 else f"{r:g}") for r in rates], fontsize=10)
            ax.set_xlabel("inferences per second per camera", fontsize=9.5, color=t["sec"]); ax.set_title(res.upper() if res == "4k" else res, loc="left", fontsize=12.5, fontweight="bold")
            ax.grid(axis="y", color=t["grid"], lw=1, zorder=0); ax.set_axisbelow(True); ax.set_ylim(0, 112)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        axes[0].set_ylabel("cameras held (highest measured grid point)", fontsize=10, color=t["sec"])
        h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.985, 0.895), frameon=False, fontsize=9.5, ncol=2)
        fig.text(0.07, 0.085, "Top: grid points 1, 8, 16, 32, 50, 75 (100 for keyframes) cameras; a value at the top of its grid is grid-limited, not a ceiling; inference rate = 30 fps / skip; DeepStream: nvinfer interval = skip - 1, keyframes only = intra-decode-enable.\n"
                             "Bottom, how latency is measured: both clocks start when a frame is already decoded and stop when its results are on the host, so network, jitter buffer and decoding are excluded on both sides.\n"
                             "PyCamTRT = mean over inferred frames of pop-to-ready + ready-to-take + take-to-done (bench/atlas, rtsp_infer_multi). DeepStream = median per-buffer component latency of nvstreammux + primary_gie + secondary_gie\n"
                             "(NVDS latency measurement, attach-sys-ts=1, 60 warm-up frames per source dropped, bench/deepstream/parse_latency.py). Only full-rate cells are shown: with the skip dial DeepStream's figure averages skipped frames too.",
                 fontsize=8.4, color=t["muted"], va="top")
        fig.text(0.07, 0.02, "github.com/Matias-GONPO/PyCamTRT  ·  bench/atlas/atlas_vs_deepstream.py  ·  bench/atlas/runs/2026-09-19_mini/<res>/{atlas,ds_dials}.csv", fontsize=9.5, color=t["sec"], va="center")
        fig.savefig(f"{OUT}{'-dark' if mode == 'dark' else ''}.png", dpi=200); plt.close(fig)
    print("wrote", OUT + ".png")
