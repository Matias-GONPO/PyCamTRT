"""pycamtrt._capacity - the arithmetic behind ``pycamtrt.recommend()``, the
M2 capacity-planning seed.

WHAT THIS IS: a pure-Python, dependency-free cost MODEL, not a live
benchmark. It has ZERO runtime dependency on ``Reports/`` - every constant
below is a number copied out of a specific Atlas measurement (cited inline)
at the point this module was written, not read from a CSV at call time.
That means ``recommend()`` works standalone (no filesystem access, no GPU
needed) but also means it goes STALE the moment the underlying pipeline
changes in a way that would move one of these numbers - re-derive the
constants from a fresh Atlas run if that ever happens.

HOW A PREDICTION IS DERIVED (read top to bottom, matches the code order):

1. Two load dials scale the requested ``fps``:
   - ``decode="key"`` divides the DECODE rate by ``key_gop`` (NVDEC only
     decodes 1-in-``key_gop`` frames); ``skip`` then divides the INFERENCE
     rate on top of whatever was decoded (frames are decoded but not all
     of them are inferred). This mirrors ``StreamDesc``/``PipelineConfig``
     in graph.h: decode and skip are independent dials.
   - Aggregate decode load = ``streams * decode_fps_per_stream``
     (drives the NVDEC estimate). Aggregate inference load =
     ``streams * decode_fps_per_stream / skip`` (drives the GPU estimate).
2. ``predicted_gpu_ms_per_frame`` is looked up (no SAHI: a flat,
   resolution-specific baseline) or computed (SAHI: ``a + b*T`` where ``T``
   is the tile count - see ``sahi_tile_count()``, a direct Python port of
   ``sahi_tiles.h``'s ``MakeTileGrid``/``TileOffsets1D``, so ``T`` is
   derived from resolution/tile/overlap, never hardcoded).
3. GPU capacity (fps) = ``1000 / predicted_gpu_ms_per_frame``; NVDEC
   capacity (decoded fps) = a fixed pixel-rate budget (see
   ``_NVDEC_BASE_PIXEL_RATE``) divided by this resolution's pixel count.
4. Whichever of (required load / capacity) is larger determines
   ``binding_resource``; both <= 1.0 means ``holds_realtime=True``.
5. ``offered_fps`` is ``fps`` scaled down by the worse of the two ratios
   when over capacity (a simple proportional-degradation approximation,
   not a queueing simulation - see the WHY-comment on ``offered_fps``
   below for its limits).

THE BIGGEST LIMIT: every constant here was measured on ONE GPU (RTX 3060
Ti, sm_86, fp16 engines, ``yolov8n_plates`` + ``lprnet`` cascade). A
different GPU (different SM count, different NVDEC generation) needs its
OWN Atlas run before these numbers mean anything there - ``recommend()``
says this in every ``Recommendation.notes`` list, not just here, because
the dataclass is meant to travel on its own once returned.

Sources (all under ``Reports/Atlas/`` in the repo this module ships with):
  - ``README.md`` - what an atlas cell measures, the resolution-invariance
    claim (inference always letterboxes to 640x640).
  - ``Atlas <res>/ATLAS_<res>_analysis.txt`` - the per-resolution "Capacity
    model" (baseline gpu_ms intercept) and "NVDEC utilization" (peak
    decoded/s) sections, one per {720p, 1080p, 1440p, 4k}.
  - ``SAHI Night/SAHI_NIGHT_SUMMARY.md`` - the decode=key (offered-rate-
    throttled) SAHI cost model, `3.185 + 0.482*T` (v2 re-fit).
  - ``SAHI Refresh/SAHI_REFRESH_SUMMARY.md`` - the decode=all pooled/serial
    SAHI cost models at N=8, and the finding that SAHI Night's model does
    NOT generalize to decode=all (different regime, different fixed
    overhead - see that file's §3b).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["recommend", "Recommendation"]

# ---------------------------------------------------------------------------
# Measured constants (RTX 3060 Ti / sm_86, fp16 engines - see module
# docstring). Each is a plain Python literal so the model stays inspectable
# (``python3 -c "import pycamtrt._capacity as c; print(c._BASELINE_MS_PER_FRAME)"``)
# and diffable against the next Atlas refresh.
# ---------------------------------------------------------------------------

# Pixel dimensions for the four resolutions the atlas series measured.
_RESOLUTION_PX: Dict[str, Tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
}

# Baseline (no-SAHI) cascade GPU-batch cost, ms/frame, fp16 - the additive
# intercept `a` of each resolution's own "Capacity model" fit
# (`gpu_ms ~= a + b*(N*inf_rate)`), from
# Reports/Atlas/Atlas <res>/ATLAS_<res>_analysis.txt's "## Capacity model"
# section:
# v2 (2026-09, "Atlas <res> v2" series - measured on the v0.2.0 Release
# binary; drift tables in each folder, content md5-pinned per run):
#   720p:  "gpu_ms ~= 1.94 + 0.00445 * (N*inf_rate)"  [RMS 0.42]
#   1080p: "gpu_ms ~= 2.05 + -0.00074 * (N*inf_rate)" [RMS 0.17]
#   1440p: "gpu_ms ~= 2.06 + -0.00137 * (N*inf_rate)" [RMS 0.17]
#   4k:    "gpu_ms ~= 2.09 + -0.00363 * (N*inf_rate)" [RMS 0.19]
# (v1, July, for the record: 1.98 / 2.00 / 1.87 / 2.01 - v2 supersedes;
# 1440p's v1 1.87 was the series outlier, see its drift table.)
# Only the intercept is used: all four land within +-0.07 ms of each other,
# confirming the Atlas README's own claim ("inference cost is
# resolution-invariant - everything letterboxes to 640x640"). The slope
# term `b` each fit reports (0.5-11 us per aggregate inf/s, sign
# inconsistent across resolutions - i.e. noise, not a real per-load-unit
# compute cost) is deliberately NOT used: extrapolated to fleet-scale
# aggregate rates (hundreds of inf/s) it would occasionally predict a
# NEGATIVE ms/frame, which is not a real limitation to encode as precision.
_BASELINE_MS_PER_FRAME: Dict[str, float] = {
    "720p": 1.94,
    "1080p": 2.05,
    "1440p": 2.06,
    "4k": 2.09,
}

# SAHI per-frame GPU cost models: ms/frame = a + b*T, T = tile count
# (see sahi_tile_count()). Three regimes - SAHI Refresh's central finding
# is that the FIXED overhead `a` is regime-dependent (how much the outer
# whole-frame batcher amortises it across frames in flight), while the
# per-tile marginal cost `b` is essentially constant (~0.48-0.50 ms/tile)
# across every regime and campaign:
#
#  - "pooled" (decode="all", the shipped default - cross-slot tile
#    batching, pipeline.cpp's non-serial SAHI path): Reports/Atlas/
#    SAHI Refresh/SAHI_REFRESH_SUMMARY.md §4, fit at N=8 (this campaign's
#    fleet-relevant operating point): "pooled | 8 | 0.323 | 0.4982 | 1.0000".
#  - "serial" (decode="all", the --sahi-serial A/B escape hatch - each
#    slot's tiles chunked on their own, never mixed across slots): same
#    §4, "serial | 8 | 1.094 | 0.4823 | 0.9998".
#  - "key" (decode="key" - an offered-rate-throttled regime where the
#    outer batcher rarely groups more than ~1 frame per grouping, so fixed
#    overhead is paid close to once per frame regardless of pooled/serial):
#    Reports/Atlas/SAHI Night/SAHI_NIGHT_SUMMARY.md §2, "gpu_ms per frame
#    ~= 2.66 + 0.488 x T (R^2=0.9978)". SAHI Refresh §3b/§6.3 explicitly
#    warns this key30-regime number must NOT be reused for decode="all"
#    (that regime amortises far more fixed overhead) - recommend()
#    therefore branches on `decode`, not on the `serial` flag alone.
#
# CAVEAT the module docstring also states: the N=8 pooled/serial fits are
# applied here regardless of the actual `streams` argument. Both
# intercepts fall further with N in the raw data (pooled: 3.327 @N=1 ->
# 1.061 @N=4 -> 0.323 @N=8 - see SAHI Refresh §4's full table), so this
# UNDER-states real cost for small fleets (a true N=1 pooled run costs
# closer to the N=1 fit, ~3.3 ms fixed overhead, not 0.32) and is only a
# steady-state approximation at/above roughly N=8. Noted in
# Recommendation.notes whenever SAHI is in play.
_SAHI_MODEL: Dict[str, Tuple[float, float]] = {
    # v2 re-fit (SAHI Refresh v2, 2026-09, R^2 >= 0.9999 all three;
    # v1 for the record: pooled (0.323, 0.498), serial (1.094, 0.482),
    # key (2.66, 0.488) - slopes drifted <2%).
    "pooled": (0.305, 0.4963),
    "serial": (0.998, 0.4900),
    "key": (3.185, 0.4820),
}

# NVDEC ceiling: a pixel-rate SCALING from one measured point (Atlas 720p:
# "decode=all: peak 1499 decoded/s at NVDEC 99%", i.e. ~50 streams @
# 30 fps/stream - Reports/Atlas/Atlas 720p/ATLAS_720p_analysis.txt's
# "## NVDEC utilization" section) - this is the task-specified
# approximation (treat NVDEC as a fixed decoded-PIXELS-per-second budget,
# scale by this resolution's pixel count), not a per-resolution measured
# fit. See the NOTE below for how this compares to the atlas's own
# per-resolution numbers, which it does NOT match exactly.
# v2 (2026-09): 1276 decoded/s at NVDEC 100% -> ~42.5 streams @ 30fps.
# NOTE the four-point v2 series read (Atlas 4k v2/DRIFT_TABLE_4k.md):
# 720p/1440p/4k peaks all came in ~15% below v1 while 1080p reproduced v1
# EXACTLY - the "peak" is an observed operating point, not a hardware
# constant; v2 values are self-consistent (one encoder, one day,
# md5-pinned content) and conservative.
_NVDEC_BASE_STREAMS_720P_AT_30FPS = 42.5
_NVDEC_BASE_PIXEL_RATE = (
    _NVDEC_BASE_STREAMS_720P_AT_30FPS * 30.0 * (1280 * 720)
)  # ~1.3824e9 px/s, treated as a fixed hardware decode budget.

# NOTE (measured, for context/future refinement - NOT used by the
# pixel-rate model above, which is what the task spec asked for): the
# atlas series actually ran all four resolutions, and NVDEC does NOT scale
# purely with pixel count on this hardware - fixed per-stream session
# overhead matters too, so every resolution above 720p sustains MORE
# decoded/s than pure area-scaling predicts:
# v2 (2026-09) measured peaks; pixel-scaling deltas recomputed from the
# v2 base (1276 @720p):
#   720p:  1276 decoded/s (the base point itself)
#   1080p:  787 decoded/s (pixel-scaling predicts ~567 -> +39%)
#   1440p:  400 decoded/s (pixel-scaling predicts ~319 -> +25%)
#   4k:     186 decoded/s (pixel-scaling predicts ~142 -> +31%)
# (v1 for the record: 1499/786/472/219 - see the v2 drift tables for the
# -15%/exact/-15%/-15% series read.)
# (all from each resolution's own ATLAS_<res>_analysis.txt "## NVDEC
# utilization" section, decode=all peak row). The pixel-rate model is
# therefore CONSERVATIVE (under-states real NVDEC headroom) by roughly
# that margin at 1080p/1440p/4k - surfaced in Recommendation.notes
# whenever NVDEC is the binding resource.
_NVDEC_MEASURED_PEAK_DECODED_PER_S: Dict[str, float] = {
    "720p": 1276.0,
    "1080p": 787.0,
    "1440p": 400.0,
    "4k": 186.0,
}

# Ratio above which a resource is reported as the/a "binding" one even
# while still holding (i.e. worth watching) - see binding_resource's
# WHY-comment in recommend() below. Purely a reporting threshold; it does
# not affect holds_realtime (that is always exactly ratio <= 1.0).
_DOMINANT_RATIO_THRESHOLD = 0.5

_SAHI_KEYS = {"tile", "overlap", "full_frame", "serial"}


# ---------------------------------------------------------------------------
# SAHI tile geometry - a direct Python port of src/sahi_tiles.h's
# TileOffsets1D/MakeTileGrid, so T (tile count) is DERIVED from
# resolution/tile/overlap here, never hardcoded (matches how the C++ side
# actually computes it - see pipeline.cpp's sahi_grids cache). Verified by
# hand against the Atlas tables this module cites: tile 640 -> T=32 tiles
# @4K (+1 full-frame pass = 33, matching every "640 | 33 | ..." row in
# SAHI Night/Refresh); tile 1280 -> 8 tiles (+1 = 9); tile 960 -> 15 tiles
# (+1 = 16).
# ---------------------------------------------------------------------------


def _tile_offsets_1d(frame: int, tile: int, stride: int) -> List[int]:
    if frame <= tile:
        return [0]
    offs: List[int] = []
    o = 0
    while True:
        if o + tile >= frame:
            offs.append(frame - tile)  # flush with the far edge
            break
        offs.append(o)
        o += stride
    return offs


def sahi_tile_count(resolution: str, tile_px: int, overlap: float = 0.2,
                     full_frame: bool = True) -> int:
    """Number of SAHI tiles (+1 if ``full_frame``) MakeTileGrid would cut
    ``resolution`` into at ``tile_px``/``overlap`` - the ``T`` in every
    ``a + b*T`` cost model in this module. Pure geometry, no GPU/model
    needed; matches ``sahi_tiles.h`` bit-for-bit (same clamp, same
    "last tile flush with the far edge" rule).
    """
    if resolution not in _RESOLUTION_PX:
        raise ValueError(
            f"resolution must be one of {sorted(_RESOLUTION_PX)}, got "
            f"{resolution!r}"
        )
    w, h = _RESOLUTION_PX[resolution]
    overlap = min(max(overlap, 0.0), 0.9)
    stride = max(1, int(tile_px * (1.0 - overlap)))
    xs = _tile_offsets_1d(w, tile_px, stride)
    ys = _tile_offsets_1d(h, tile_px, stride)
    return len(xs) * len(ys) + (1 if full_frame else 0)


@dataclass
class Recommendation:
    """The output of ``recommend()`` - a capacity-planning ESTIMATE, not a
    guarantee (see the module docstring for how each field is derived and
    its limits). Every field is a plain value (no methods needed) so it
    serializes trivially (``dataclasses.asdict()``, ``repr()``, logging).
    """

    #: Modeled GPU cost of one inferred frame, ms (already includes the
    #: SAHI tile cost, amortized per whole frame, when ``sahi`` was given).
    predicted_gpu_ms_per_frame: float
    #: The per-stream fps this configuration can actually sustain - equal
    #: to the requested ``fps`` when it holds, scaled down proportionally
    #: to the worse of the two capacity ratios otherwise (see
    #: recommend()'s WHY-comment on this field for the model's limits).
    offered_fps: float
    #: Aggregate inference load requested of the GPU (streams x offered
    #: inference rate per stream), frames/s.
    required_gpu_fps: float
    #: (capacity - required) / capacity * 100 for the GPU-inference
    #: resource specifically; negative when over capacity.
    gpu_headroom_pct: float
    #: Which resource the configuration is bound by: "gpu-inference",
    #: "nvdec", or "none" (comfortable headroom on both - see
    #: _DOMINANT_RATIO_THRESHOLD).
    binding_resource: str
    #: True iff BOTH the GPU-inference and NVDEC ratios are <= 1.0.
    holds_realtime: bool
    #: One human sentence: "holds with N% headroom", or "over capacity: "
    #: plus concrete dials to turn (raise skip, decode="key", bigger tile,
    #: fewer streams).
    suggestion: str
    #: Model provenance (which constants/regime were used, their
    #: citations) + the this-GPU caveat - always non-empty.
    notes: List[str] = field(default_factory=list)
    #: Aggregate NVDEC decode load requested, decoded frames/s (extra
    #: transparency field, not in the M2 spec's minimum list but cheap to
    #: expose - see the module docstring's derivation walkthrough).
    required_nvdec_fps: float = 0.0
    #: Modeled NVDEC ceiling for this resolution, decoded frames/s.
    nvdec_capacity_fps: float = 0.0
    #: (capacity - required) / capacity * 100 for NVDEC specifically.
    nvdec_headroom_pct: float = 0.0
    #: Modeled GPU-inference ceiling, frames/s (== 1000 /
    #: predicted_gpu_ms_per_frame).
    gpu_capacity_fps: float = 0.0


def recommend(
    streams: int,
    resolution: str = "720p",
    fps: float = 30,
    skip: int = 1,
    decode: str = "all",
    key_gop: int = 30,
    sahi: Optional[dict] = None,
) -> Recommendation:
    """Predict whether ``streams`` cameras at ``resolution``/``fps`` hold
    real-time through the fp16 plate-detect cascade, with or without SAHI
    tiling, on the SAME GPU the Atlas series measured (RTX 3060 Ti,
    sm_86) - see the module docstring for the full derivation and the
    ACTUAL Atlas citations behind every constant used here.

    Args:
        streams: camera count (fleet size), >= 1.
        resolution: ``"720p"``, ``"1080p"``, ``"1440p"``, or ``"4k"`` - the
            four resolutions the Atlas series measured. Only affects the
            NVDEC ceiling (pixel-rate scaled) and, when ``sahi`` is given,
            the tile count ``T`` - GPU-inference cost itself is
            resolution-invariant (everything letterboxes to 640x640).
        fps: the source stream's native frame rate.
        skip: infer every Kth DECODED frame (mirrors
            ``Pipeline(skip=...)``) - divides the inference rate only,
            never the decode/NVDEC rate.
        decode: ``"all"`` (decode every frame) or ``"key"`` (keyframes
            only, mirrors ``Pipeline(decode=...)``) - divides BOTH the
            decode rate and (before ``skip``) the inference rate by
            ``key_gop``.
        key_gop: GOP length in frames, used only when ``decode="key"``.
        sahi: ``None`` (no tiling - the flat per-resolution baseline cost
            applies), or a dict with a required ``"tile"`` key (pixel
            tile size) and optional ``"overlap"`` (default 0.2, matching
            ``LayerDesc.sahi_overlap``), ``"full_frame"`` (default True),
            ``"serial"`` (default False - pooled cross-slot batching is
            the shipped default; True selects the ``--sahi-serial`` A/B
            model instead). Mirrors ``Layer(sahi=...)``/
            ``Pipeline(sahi=...)``'s own dict shape (minus ``merge_iou``,
            which has no cost-model effect).

    Returns:
        A ``Recommendation`` - see that dataclass for each field's meaning
        and ``Recommendation.notes`` for this call's specific provenance.

    Raises:
        ValueError: unknown ``resolution``/``decode``, ``streams``/``skip``
            < 1, or an ``sahi`` dict missing ``"tile"`` / carrying an
            unknown key.
    """
    if resolution not in _RESOLUTION_PX:
        raise ValueError(
            f"resolution must be one of {sorted(_RESOLUTION_PX)}, got "
            f"{resolution!r}"
        )
    if decode not in ("all", "key"):
        raise ValueError(f'decode must be "all" or "key", got {decode!r}')
    if streams < 1:
        raise ValueError(f"streams must be >= 1, got {streams}")
    if skip < 1:
        raise ValueError(f"skip must be >= 1, got {skip}")

    notes: List[str] = [
        "All constants measured on RTX 3060 Ti (sm_86), fp16 engines "
        "(yolov8n_plates + lprnet cascade) - see Reports/Atlas/. A "
        "different GPU (SM count, NVDEC generation) needs its own atlas "
        "run before these numbers mean anything there; treat this as a "
        "seed, not a portable model.",
    ]

    # ---- Load dials (WHY: see module docstring §1) --------------------
    decode_fps_per_stream = fps / key_gop if decode == "key" else fps
    inference_fps_per_stream = decode_fps_per_stream / skip
    required_decode_fps = streams * decode_fps_per_stream
    required_gpu_fps = streams * inference_fps_per_stream

    # ---- GPU-inference cost model (WHY: see module docstring §2) ------
    tile_px: Optional[int] = None
    if sahi is not None:
        unknown = set(sahi) - _SAHI_KEYS
        if unknown:
            raise ValueError(
                f"unknown sahi key(s) {sorted(unknown)}; valid: "
                f"{sorted(_SAHI_KEYS)}"
            )
        if "tile" not in sahi:
            raise ValueError('sahi dict requires a "tile" key, e.g. '
                              '{"tile": 640}')
        tile_px = int(sahi["tile"])
        overlap = float(sahi.get("overlap", 0.2))
        full_frame = bool(sahi.get("full_frame", True))
        serial = bool(sahi.get("serial", False))
        T = sahi_tile_count(resolution, tile_px, overlap, full_frame)
        if decode == "key":
            regime = "key"
        else:
            regime = "serial" if serial else "pooled"
        a, b = _SAHI_MODEL[regime]
        predicted_gpu_ms_per_frame = a + b * T
        notes.append(
            f"SAHI: tile {tile_px} @ {resolution} (overlap={overlap}, "
            f"full_frame={full_frame}) -> T={T} tiles (MakeTileGrid rule, "
            f"see sahi_tile_count()); '{regime}' cost model "
            f"ms/frame = {a} + {b}*T = {predicted_gpu_ms_per_frame:.3f} "
            "(see this module's _SAHI_MODEL comment for the Atlas "
            "citation and the N=8 steady-state caveat)."
        )
        if regime == "key":
            notes.append(
                "decode=\"key\" SAHI cost uses SAHI Night's key30-regime "
                "model (fixed overhead paid ~once per frame, since the "
                "outer batcher rarely groups more than ~1 frame at this "
                "offered rate) - pooled vs serial makes no measured "
                "difference in this regime."
            )
    else:
        predicted_gpu_ms_per_frame = _BASELINE_MS_PER_FRAME[resolution]
        notes.append(
            f"no SAHI: baseline cascade cost for {resolution} = "
            f"{predicted_gpu_ms_per_frame} ms/frame (Atlas <res> analysis's "
            "'Capacity model' intercept) - the Atlas series' own finding "
            "is that this is resolution-invariant (everything letterboxes "
            "to 640x640); all four measured resolutions land within "
            "+-0.07 ms of each other."
        )

    gpu_capacity_fps = 1000.0 / predicted_gpu_ms_per_frame

    # ---- NVDEC ceiling (WHY: see module docstring §3 / _NVDEC_* consts)
    w, h = _RESOLUTION_PX[resolution]
    nvdec_capacity_fps = _NVDEC_BASE_PIXEL_RATE / (w * h)
    if resolution != "720p":
        measured = _NVDEC_MEASURED_PEAK_DECODED_PER_S[resolution]
        pct = (measured / nvdec_capacity_fps - 1.0) * 100.0
        notes.append(
            f"NVDEC ceiling for {resolution} ({nvdec_capacity_fps:.0f} "
            "decoded/s) is a pixel-rate SCALING from the 720p measurement "
            f"(~50 streams @30fps) - the atlas actually measured "
            f"{measured:.0f} decoded/s here ({pct:+.0f}% vs this "
            "approximation), so this UNDER-states real NVDEC headroom at "
            "this resolution; see _NVDEC_MEASURED_PEAK_DECODED_PER_S's "
            "comment if you need the measured number instead."
        )

    # ---- Ratios, binding resource, holds_realtime ----------------------
    gpu_ratio = required_gpu_fps / gpu_capacity_fps
    nvdec_ratio = required_decode_fps / nvdec_capacity_fps
    holds_realtime = gpu_ratio <= 1.0 and nvdec_ratio <= 1.0

    gpu_headroom_pct = (1.0 - gpu_ratio) * 100.0
    nvdec_headroom_pct = (1.0 - nvdec_ratio) * 100.0

    # WHY 0.5 as the "dominant" cutoff (not "> 1.0"): a resource can be
    # worth naming ("this is what to watch") well before it actually
    # blows the budget - "none" is reserved for genuinely comfortable
    # headroom on BOTH sides, not merely "not yet over capacity".
    max_ratio = max(gpu_ratio, nvdec_ratio)
    if max_ratio < _DOMINANT_RATIO_THRESHOLD:
        binding_resource = "none"
    elif gpu_ratio >= nvdec_ratio:
        binding_resource = "gpu-inference"
    else:
        binding_resource = "nvdec"

    # WHY this formula (a simple proportional-degradation approximation,
    # NOT a queueing simulation): over capacity, the offered/achieved rate
    # does not fall to zero, it falls roughly in proportion to how far
    # over budget the binding resource is (backpressure/frame-dropping
    # shares the deficit across the offered load) - matches the Atlas
    # data's own qualitative shape (e.g. SAHI Refresh's T=33/N=8 cell held
    # ~23% of offered rate at ~4.3x over its own compute ceiling, the same
    # order of magnitude 1/max_ratio would predict) without claiming
    # queueing-model precision this module was never measured against.
    offered_fps = fps / max(1.0, gpu_ratio, nvdec_ratio)

    if holds_realtime:
        headroom = min(gpu_headroom_pct, nvdec_headroom_pct)
        suggestion = f"holds with {headroom:.0f}% headroom"
    else:
        dials: List[str] = []
        if gpu_ratio > 1.0:
            needed_skip = math.ceil(skip * gpu_ratio)
            dials.append(f"raise skip to {needed_skip}")
            if decode != "key":
                dials.append('use decode="key"')
            if tile_px is not None and tile_px < 1280:
                dials.append(
                    "use tile 1280 (fewer/bigger tiles - lower recall on "
                    "small objects, see the package docstring's SAHI "
                    "section)"
                )
        if nvdec_ratio > 1.0:
            needed_streams = max(1, math.floor(streams / nvdec_ratio))
            dials.append(
                f'cut to <= {needed_streams} streams or use decode="key" '
                "to shed NVDEC load"
            )
        suggestion = "over capacity: " + " / ".join(dials)

    return Recommendation(
        predicted_gpu_ms_per_frame=predicted_gpu_ms_per_frame,
        offered_fps=offered_fps,
        required_gpu_fps=required_gpu_fps,
        gpu_headroom_pct=gpu_headroom_pct,
        binding_resource=binding_resource,
        holds_realtime=holds_realtime,
        suggestion=suggestion,
        notes=notes,
        required_nvdec_fps=required_decode_fps,
        nvdec_capacity_fps=nvdec_capacity_fps,
        nvdec_headroom_pct=nvdec_headroom_pct,
        gpu_capacity_fps=gpu_capacity_fps,
    )
