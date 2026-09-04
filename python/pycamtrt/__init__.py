"""pycamtrt - the declarative Python API for CORDERO's multi-stream
TensorRT pipeline.

This package is a thin, pure-Python compiler on top of the compiled
extension module ``_pycamtrt`` (a near-1:1 pybind11 binding of
``cordero::Pipeline`` - see src/python/bindings.cpp and src/core/{pipeline,
graph,result}.h). Users build a small object graph out of generic stages
(``Process``, ``Engine``, ``Postprocess``) grouped into ``Layer``s, and this
module compiles that graph into the ``StepDesc``/``LayerDesc``/
``PipelineConfig`` index graph the C++ executor understands.

Minimal one-layer (detection-only) example::

    import pycamtrt

    streams = pycamtrt.Streams(["rtsp://cam1", "rtsp://cam2"])
    L1 = pycamtrt.Layer("detect")
    eng = L1.add(pycamtrt.Engine(streams, "models/yolov8n_b1-16_fp32_sm86.engine"))
    det = L1.add(pycamtrt.Postprocess(eng, family="yolo", score=0.4, iou=0.45))

    pipe = pycamtrt.Pipeline(streams, layers=[L1])
    with pipe:
        for r in pipe:
            print(r.stream_id, r.frame_no, len(r.detections))

Two-layer cascade (detect -> crop -> read), cross-layer edge auto-crops::

    L2 = pycamtrt.Layer("read")
    eng2 = L2.add(pycamtrt.Engine(det, "ocr.engine"))
    txt = L2.add(pycamtrt.Postprocess(eng2, family="ctc"))
    pipe = pycamtrt.Pipeline(streams, layers=[L1, L2])

Depth-2 TREE (M4b): one detector root feeding N SIBLING recognition
children (any mix of ``family="ctc"``/``family="argmax"``), each with its
OWN engine/norm/color - every child's ``Engine`` takes the SAME root
``Postprocess`` step as ``input`` (a sibling edge, not a chain)::

    L1 = pycamtrt.Layer("detect")
    eng = L1.add(pycamtrt.Engine(streams, "detector.engine"))
    det = L1.add(pycamtrt.Postprocess(eng, family="yolo"))

    read = pycamtrt.Layer("read")
    reng = read.add(pycamtrt.Engine(det, "lprnet.engine"))       # <- det
    read.add(pycamtrt.Postprocess(reng, family="ctc"))

    classify = pycamtrt.Layer("classify")
    ceng = classify.add(pycamtrt.Engine(det, "mobilenet.engine",  # <- det
                                        norm=IMAGENET_NORM, color="rgb"))
    classify.add(pycamtrt.Postprocess(ceng, family="argmax"))

    pipe = pycamtrt.Pipeline(streams, layers=[L1, read, classify])
    with pipe:
        for r in pipe:
            r.outputs["read"]      # -> list[str], aligned with r.detections
            r.outputs["classify"]  # -> list[(label, score)], same alignment

LIMITATION (depth-2 only, deliberate - not yet a chain executor): every
child's ``Engine`` must take the DETECTOR layer's own ``Postprocess`` step
as ``input`` - siblings, all cropping the same root. Feeding one child's
cascade output into ANOTHER ``Engine`` (a "child of a child" - e.g.
``Engine(read.steps[-1], ...)`` instead of ``Engine(det, ...)`` above) is
rejected with a named ``RuntimeError`` from the C++ executor's
``Validate()`` (pipeline.cpp): *"v1 executor supports a depth-2 tree: one
detector layer feeding sibling recognition layers; chained cascades (a
child of a child) are not yet executable"*. See
``examples/read_and_classify.py`` for the full runnable 3-model version of
the tree above.

Per-stream load-dial overrides (capacity planning is per-camera - see
``Stream``): mix plain URLs (inherit the pipeline-wide defaults) with
``Stream`` objects that override ``skip``/``decode`` for just that camera::

    streams = pycamtrt.Streams(["rtsp://entrance",
                                 pycamtrt.Stream("rtsp://parking", skip=4)])

Endpoint sinks (Phase A2) - tee detections/text out as NDJSON, or the
original compressed video back out as a live RTSP relay, without any of that
touching the per-frame path (see ``Sink``)::

    events = pycamtrt.Sink(det, kind="events", target="tcp://0.0.0.0:9000")
    relay = pycamtrt.Sink(streams, kind="stream", stream=0,
                           target="rtsp://localhost:8554/cam1_relay")
    pipe = pycamtrt.Pipeline(streams, layers=[L1], sinks=[events, relay])

Validation note: beyond mapping the ``family=`` string (see
``_FAMILIES``), this module does NOT re-validate the graph shape. The v1
C++ executor's ``Validate()`` (pipeline.cpp) is the single source of truth
for which step graphs are actually runnable; its ``std::runtime_error``
messages surface here as ``RuntimeError``.

Three-tier frame-memory access: every ``Result`` carries its underlying
full-res NV12 frame at three escalating levels of commitment (see
``core/result.h``/``core/pipeline.h`` for the C++ side)::

    Tier 1 - always on, informational only.  ``r.frame_addr``,
        ``r.frame_pitch``, ``r.frame_width``, ``r.frame_height`` are
        populated for every result, print/log fine, but the memory they
        describe may be recycled (overwritten by another frame) at any
        moment - never trust them as a live pointer.
    Tier 2 - ``r.fetch_frame()``, a plain D2H copy into a fresh numpy
        array. Safe from any Python thread/context; costs a copy.
    Tier 3 - ``pycamtrt.Pipeline(..., hold_frames=True)`` keeps this
        result's ring slot alive (see ``ring_depth`` below) until every
        copy of the result is garbage collected or ``r.release_frame()``
        is called. With that, ``r.frame_cuda()`` hands back a zero-copy
        CUDA view (``__cuda_array_interface__``) usable directly with
        ``torch.as_tensor(r.frame_cuda(), device="cuda")`` or
        ``cupy.asarray(r.frame_cuda())`` - no copy, no format conversion.

``hold_frames``/``ring_depth`` contract: holding a frame (tier 3) means
its ring slot cannot be reused until you let go of it, so
``Pipeline(ring_depth=N)`` must be sized to how many frames of one stream
you plan to hold concurrently - a consumer holding ~``ring_depth - 1``
frames of a stream stalls THAT stream's producer (intended backpressure,
not a bug). ``ring_depth`` defaults to 4 (fine when not holding); raise it
when ``hold_frames=True``.

SAHI (tiled inference for small/far objects), M2 - a per-layer mechanism::

    L1 = pycamtrt.Layer("detect", sahi=dict(tile=640, overlap=0.2))
    eng = L1.add(pycamtrt.Engine(streams, "detector.engine"))
    L1.add(pycamtrt.Postprocess(eng, family="yolo"))
    pipe = pycamtrt.Pipeline(streams, layers=[L1])

What it is: instead of one whole-frame inference pass, the frame is cut
into overlapping ``tile``-pixel tiles (classic SAHI - see ``sahi_tiles.h``),
each run through the SAME engine, then merged back into one detection list
(NMS across tile boundaries, ``merge_iou``) - a full-frame pass is included
by default (``full_frame=True``) so objects larger than one tile are still
caught. This trades GPU cost for RECALL on small/far objects the whole-
frame 640x640 letterbox would shrink past visibility (e.g. a distant
license plate).

``Layer(sahi=...)`` (preferred) attaches SAHI to THAT layer specifically -
the C++ v1 executor only accepts it on the yolo detection layer (today,
always layer 0); putting it on a cascade (ctc/argmax) layer raises a named
``RuntimeError`` from ``Validate()``. ``Pipeline(sahi=...)`` still works -
a legacy, position-based form that applies to layer 0 only; prefer
``Layer(sahi=...)``. Giving BOTH with genuinely different dicts raises
``ValueError`` (ambiguous - which one wins?); giving both the same dict is
redundant but fine.

The dict's keys: ``tile`` (pixel tile size, default 640), ``overlap``
(fraction shared between neighbor tiles, default 0.2), ``merge_iou``
(cross-tile NMS threshold, default 0.5), ``full_frame`` (default True),
``serial`` (default False - see below). Presence of the dict (even
``{}``) turns SAHI on for that layer.

**Pooled is the default; ``serial=True`` is an A/B escape hatch.** Pooling
(``pipeline.cpp``'s non-serial SAHI path) batches tile jobs from every
in-flight slot into shared engine calls (capped at the engine's max
batch); ``serial`` keeps each frame's tiles in their own chunk, never
mixed across slots/frames - never faster than pooled, occasionally useful
as a control when isolating a regression (``tools/sahi_parity.sh``).

**Measured guidance (do NOT change the 640 default silently - it is a
recall/cost trade, not a pure speed knob):** tile 640 is the recall-first
default. Tile 1280 (fewer, bigger tiles) trades small-object recall for
throughput: at 4K, pooled tile 640 tops out around a 60 fps whole-GPU
fleet ceiling versus ~208 fps at tile 1280 (+~3.5x) - see
``Reports/Atlas/SAHI Refresh/SAHI_REFRESH_SUMMARY.md`` §5 for the full
per-tile ceiling table and the recall caveat. Use ``pycamtrt.recommend()``
(below) to check a fleet size/resolution/tile combination against these
numbers before deploying it, rather than guessing::

    >>> import pycamtrt
    >>> r = pycamtrt.recommend(streams=16, resolution="4k", fps=30,
    ...                        sahi={"tile": 640})
    >>> r.holds_realtime, r.suggestion
    (False, 'over capacity: raise skip to 9 / use decode="key" / ...')

See ``pycamtrt._capacity``'s module docstring for exactly how
``recommend()``'s predictions are derived (and their limits - all
constants are measured on ONE GPU, RTX 3060 Ti / sm_86).

Custom postprocessing - two tiers (M3b), split by TENSOR SIZE, not by
preference (a settled CORDERO design rule):

  1. **Python on COMPACT results (this tier, legal and encouraged).** A
     ``Result``'s ``detections``/``texts``/``labels`` are already tiny
     (tens of survivors, ~KB) by the time your ``for r in pipe:`` loop
     sees them - filtering, zone logic, alerting, temporal smoothing, etc.
     on THAT data can never touch the per-frame GPU path; it runs entirely
     on your consumer thread. Worked example: ``examples/zone_filter.py``
     (per-stream polygon zones, occupancy counts, enter/leave prints - see
     its module docstring for the full contract, including the
     ``backpressure="drop_oldest"`` / ``dropped_results()`` escape hatch
     for a consumer that falls behind).
  2. **Compiled GPU sibling-launcher families, for the RAW OUTPUT TENSOR.**
     Anything reading a model's raw head - per-anchor/per-pixel decode,
     e.g. a new detection or segmentation layout - is GPU-only by
     construction (``postprocess.h``'s own design comment: "a plain launch
     function, caller owns every device buffer, everything async on the
     caller's stream"). ``yolo``/``yolo-e2e``/``ctc``/``argmax`` (see
     ``_FAMILIES``) are the built-in examples; adding your own is a
     recompile, not a plugin -
     see ``docs/ADDING_A_FAMILY.md`` for the full walkthrough (using
     ``argmax`` as the worked, already-shipped template).

The litmus test is always: does your code read *survivors* (short,
compact, post-decode) or the *raw tensor* (megabytes, every anchor, every
frame)? Survivors -> tier 1, Python, here. Raw tensor -> tier 2, compiled,
``docs/ADDING_A_FAMILY.md``.
"""

from __future__ import annotations

import weakref
from typing import (Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Union)

try:
    # Installed layout (pip wheel/`pip install .`): the compiled module lives
    # inside the package (CMake install() rule, see pyproject.toml).
    from . import _pycamtrt as _c  # type: ignore[attr-defined]
except ImportError:
    # In-tree dev layout: PYTHONPATH=/workspace/build:/workspace/python puts
    # the freshly built module on the top-level path (BUILD.md workflow).
    import _pycamtrt as _c

from ._capacity import Recommendation, recommend

__version__ = "0.3.0"

__all__ = [
    "Streams",
    "Stream",
    "Process",
    "Engine",
    "Postprocess",
    "Select",
    "Layer",
    "Sink",
    "Pipeline",
    "Result",
    "PollStatus",
    "Detection",
    "StreamInfo",
    "recommend",
    "Recommendation",
]

Detection = _c.Detection
StreamInfo = _c.StreamInfo
PollStatus = _c.Pipeline.PollStatus

# Family strings accepted by Postprocess(family=...), mapped to
# cordero::Family (graph.h). Keep in sync with that enum. "argmax" (M1a) is
# the classifier family - plain max-logit, no softmax (see
# LaunchArgmaxBatched in postprocess.h) - usable as any SIBLING cascade
# child's postprocess (M4b: layers 1..K, a depth-2 tree) alongside "ctc".
# "yolo-e2e" (M4a) is the NMS-free end-to-end
# detector family (yolo26-style heads - see LaunchYoloE2EBatched in
# postprocess.h) - a LAYER-0 detector alongside "yolo", not a cascade
# family; `score` is honored, `iou` is ignored (no NMS stage), and SAHI is
# rejected on a layer using it (named RuntimeError at Pipeline
# construction - see pipeline.cpp's Validate()).
# "embedding" (T3.1/report 11) is the raw pass-through cascade-child
# family for penultimate-feature / re-ID heads: the child engine's 2D
# [N,D] output rows are handed back UNDECODED, one [D] float list per
# detection, via Result.outputs[<layer>] / ChildOutput.vectors - there is
# no kernel, the pass-through IS the decode. `score`/`iou` are ignored.
# "rtdetr" (T3.2/report 11) is yolo-e2e's twin for ultralytics RT-DETR
# exports: the SAME NMS-free [N,300,6] head contract, but box columns are
# normalized cx,cy,w,h (their pixel scaling lives in ultralytics' python
# postprocess, so it lives in our kernel instead) - a LAYER-0 detector
# family; like yolo-e2e, `score` honored, `iou` ignored, SAHI rejected.
_FAMILIES = {
    "yolo": _c.Family.YoloDetect,
    "ctc": _c.Family.Ctc,
    "argmax": _c.Family.Argmax,
    "yolo-e2e": _c.Family.YoloE2E,
    "embedding": _c.Family.Embedding,
    "rtdetr": _c.Family.RtDetr,
}

_DECODE_MODES = ("all", "key")

# Backpressure strings accepted by Pipeline(backpressure=...), mapped to
# cordero::Backpressure (graph.h). Keep in sync with that enum.
_BACKPRESSURE = {
    "block": _c.Backpressure.Block,
    "drop_oldest": _c.Backpressure.DropOldest,
}

# Sink kind strings accepted by Sink(kind=...), mapped to cordero::SinkKind
# (graph.h). Keep in sync with that enum.
_SINK_KINDS = {
    "events": _c.SinkKind.Events,
    "stream": _c.SinkKind.StreamRelay,
}

# Keys accepted by Layer(sahi=...)/Pipeline(sahi=...)'s dict - kept in sync
# with LayerDesc's sahi_* fields (graph.h). Shared by both entry points
# (see _apply_sahi) so the two forms compile to identical LayerDesc state.
_SAHI_KEYS = {"tile", "overlap", "merge_iou", "full_frame", "serial"}


def _apply_sahi(c_layer: "_c.LayerDesc", sahi: dict) -> None:
    """Applies a Layer(sahi=...)/Pipeline(sahi=...) dict onto a compiled
    ``_c.LayerDesc``, leaving any key not present at its LayerDesc default.
    Shared by Pipeline.__init__'s per-layer compile loop and its legacy
    Pipeline(sahi=...) alias path (see M2's Layer-vs-Pipeline sahi= WHY-
    comment there) so both forms behave identically.
    """
    unknown = set(sahi) - _SAHI_KEYS
    if unknown:
        raise ValueError(
            f"unknown sahi key(s) {sorted(unknown)}; valid: "
            f"{sorted(_SAHI_KEYS)}"
        )
    c_layer.sahi = True
    if "tile" in sahi:
        c_layer.sahi_tile = sahi["tile"]
    if "overlap" in sahi:
        c_layer.sahi_overlap = sahi["overlap"]
    if "merge_iou" in sahi:
        c_layer.sahi_merge_iou = sahi["merge_iou"]
    if "full_frame" in sahi:
        c_layer.sahi_full_frame = sahi["full_frame"]
    if "serial" in sahi:
        c_layer.sahi_serial = sahi["serial"]

StepInput = Union["Streams", "Process", "Engine", "Postprocess", "Select"]


class Stream:
    """A per-camera override of the two load dials (``skip``, ``decode``) -
    see ``PipelineConfig``/``StreamDesc`` (graph.h). Capacity planning is
    per-camera (stream count x decode rate x inference rate): a parking
    camera can run ``skip=4, decode="key"`` next to an entrance camera left
    at the pipeline default. Pass one of these (instead of a plain URL
    string) inside ``Streams([...])`` wherever a stream needs to differ
    from the pipeline-wide default.

    Args:
        url: the RTSP URL (or other demuxer input).
        skip: infer every Kth decoded frame for this stream; ``None``
            (default) inherits ``Pipeline(skip=...)``.
        decode: ``"all"`` or ``"key"`` for this stream; ``None`` (default)
            inherits ``Pipeline(decode=...)``.
    """

    def __init__(self, url: str, skip: Optional[int] = None,
                 decode: Optional[str] = None):
        if skip is not None and skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip!r}")
        if decode is not None and decode not in _DECODE_MODES:
            raise ValueError(
                f"decode must be one of {_DECODE_MODES} or None, got {decode!r}"
            )
        self.url = url
        self.skip = skip
        self.decode = decode


class Streams:
    """The raw stream source: a list of RTSP URLs (or other inputs the
    demuxer accepts), each optionally a ``Stream`` object carrying
    per-camera ``skip``/``decode`` overrides (a plain ``str`` means no
    overrides - inherit the pipeline-wide defaults). Every step graph
    starts here - a step whose input is a ``Streams`` object compiles to
    ``StepDesc.input = -1`` (the C++ convention for "raw stream source",
    see graph.h).

    Held by reference: steps refer back to whichever ``Streams``/step
    object they were built from, so this object (and every step) must stay
    alive until ``Pipeline`` compiles the graph.
    """

    def __init__(self, urls: Sequence[Union[str, Stream]]):
        self.entries: List[Stream] = [
            u if isinstance(u, Stream) else Stream(u) for u in urls
        ]


class _Step:
    """Base for the three generic stage kinds. A step holds its INPUT
    reference (the dataflow edge) - that reference, not any name or
    position, is the single source of truth ``Pipeline`` uses to compile
    the index graph.
    """

    kind: _c.StepKind

    def __init__(self, input: StepInput):
        self.input = input


class Process(_Step):
    """An explicit preprocess stage. Optional: an ``Engine`` step may take
    ``Streams`` (or another step) directly and the implicit preprocess is
    assumed - include ``Process`` only when you want to name/reuse that
    stage explicitly.
    """

    kind = _c.StepKind.Process


class Select(_Step):
    """A declarative DETECTION FILTER - the routing node (R1, v0.2.0).

    Sits between the detector layer's ``Postprocess`` and a cascade
    child's ``Engine``: only detections passing EVERY active criterion
    have their crops sent to that child. Model-agnostic by construction:
    criteria are comparisons over the detection struct (integer class id,
    score, crop size) - the library routes on ints, never on class NAMES
    ("person" is a model's opinion, not the library's; cls 0 means person
    only if YOUR detector says so).

    ::

        person = pycamtrt.Layer("person")
        sel = person.add(pycamtrt.Select(dets, classes={0}, min_size=32))
        eng = person.add(pycamtrt.Engine(sel, "models/resnet18emb...", ...))
        person.add(pycamtrt.Postprocess(eng, family="embedding"))

    Contracts:

    - A child layer is ``[Engine, Postprocess]`` (unrouted, unchanged) or
      ``[Select, Engine, Postprocess]``. The Select's input must be the
      DETECTOR layer's Postprocess step; a Select fed from another
      child's output is the depth-3 case and raises the named chained-
      cascade error.
    - Results stay ALIGNED: ``outputs[layer][i]`` still describes
      ``detections[i]`` - a routed-away detection simply gets that
      child's empty entry ("" / no label pair / empty vector). Consumers
      keep one indexing rule whether or not routing is on.
    - Sharing is legal: another child layer's ``Engine`` may take this
      same Select handle as its input (that layer is then the 2-step
      shape, its filter defined elsewhere).
    - Routing strictly REDUCES work: fewer crops per child, smaller child
      batches.
    - Criteria must stay compilable (host-side comparisons in the crop
      path). Anything needing pixels or arbitrary Python belongs to the
      Python-on-results tier (see ``examples/zone_filter.py``), not here.

    Args:
        input: the detector layer's ``Postprocess`` step.
        classes: iterable of ints - detection class ids to pass; ``None``
            (default) = any class.
        min_score: pass only detections with ``score >= min_score``;
            ``None`` = any score.
        min_size: pass only detections whose crop is at least this many
            SOURCE pixels on its smaller side (min(w, h), measured on the
            clamped crop rectangle the child would actually see);
            ``None`` = any size. A ``Select()`` with no criteria passes
            everything (useful as an A/B control).
    """

    kind = _c.StepKind.Select

    def __init__(self, input: StepInput,
                 classes: Optional[Iterable[int]] = None,
                 min_score: Optional[float] = None,
                 min_size: Optional[float] = None):
        super().__init__(input)
        if classes is not None:
            cls_list = sorted(set(classes))
            bad = [c for c in cls_list
                   if not isinstance(c, int) or isinstance(c, bool)]
            if bad or not cls_list:
                raise ValueError(
                    f"classes must be a non-empty iterable of ints (class "
                    f"ids as YOUR detector emits them), got {classes!r}")
            self.classes: Optional[List[int]] = cls_list
        else:
            self.classes = None
        for name, v in (("min_score", min_score), ("min_size", min_size)):
            if v is not None and (not isinstance(v, (int, float))
                                  or isinstance(v, bool) or v < 0):
                raise ValueError(f"{name} must be a non-negative number or "
                                 f"None, got {v!r}")
        self.min_score = min_score
        self.min_size = min_size


class Engine(_Step):
    """Runs a TensorRT engine on its input.

    Args:
        input: a ``Streams``, ``Process``, or (for a cross-layer/cascade
            edge) a ``Postprocess`` step from an earlier layer - the C++
            side turns that edge into an automatic per-detection crop. M4b:
            more than one ``Engine`` (across different ``Layer``s) may take
            the SAME detector ``Postprocess`` step as `input` - that is
            exactly how sibling recognition children are built (a depth-2
            tree; see the module docstring's "Depth-2 tree" section).
            Feeding one child's OWN ``Postprocess`` step into another
            ``Engine`` (chaining a child off a child) is accepted here at
            the Python level but rejected by the C++ executor's
            ``Validate()`` with a named ``RuntimeError`` - v1 does not run
            cascades deeper than 2.
        engine_path: path to the .engine file.
        norm: ``(offset, scale)`` applied PER CHANNEL as
            ``(pixel[c] + offset[c]) * scale[c]`` before this engine sees it
            (see ``StepDesc`` in graph.h) - M3a upgrade of M1a's single
            scalar pair. Two shapes are accepted:
              - SCALAR: ``(offset, scale)`` as two numbers, broadcast to
                all 3 channels (M1a back-compat shape; equivalent to the
                per-channel form with all three channels equal).
              - PER-CHANNEL: ``((o0, o1, o2), (s0, s1, s2))``, one
                offset/scale pair per channel.
            Any other shape raises ``ValueError`` naming both accepted
            forms. CONVENTION: index 0/1/2 are the FIRST/SECOND/THIRD
            channel of THIS ENGINE's channel order - i.e. the same order
            ``color`` selects below (``color="rgb"`` -> index 0 = R, 1 = G,
            2 = B; ``color="bgr"`` -> index 0 = B, 1 = G, 2 = R). This
            matches how users normally state a model's trained mean/std
            (e.g. torchvision's ImageNet mean ``[0.485, 0.456, 0.406]`` /
            std ``[0.229, 0.224, 0.225]`` are RGB-ordered - convert to this
            knob's ``(pixel + offset) * scale`` form on a 0..255 pixel as
            ``offset[c] = -mean[c]*255``, ``scale[c] = 1/(std[c]*255)``).
            ``None`` (default) INHERITS the family default for this
            engine's POSITION in the v1 executor: layer 0 (whole-frame +
            SAHI tiles, YOLO-family input) = ``(0, 1/255)`` all channels;
            layer 1 (cascade crop input, today's LPRNet/OCR convention) =
            ``(-127.5, 1/128)`` all channels. Override this when onboarding
            a model trained with different (optionally per-channel) input
            normalization (e.g. a true-ImageNet-normalized classifier -
            see M1b/M3a and python/export_classifier.py).
        color: ``"rgb"`` or ``"bgr"`` - the channel order this engine
            expects; ``None`` (default) inherits the same per-position
            family default (layer 0 = RGB, layer 1 = BGR, matching today's
            hardcoded behavior). Raises ``ValueError`` for any other value.
    """

    kind = _c.StepKind.Engine

    def __init__(self, input: StepInput, engine_path: str,
                 norm: Optional[tuple] = None,
                 color: Optional[str] = None,
                 max_batch: int = 16,
                 fp16: bool = True,
                 shape: Optional[tuple] = None):
        super().__init__(input)
        self.engine_path = engine_path
        self.norm = self._parse_norm(norm)
        if color is not None and color not in ("rgb", "bgr"):
            raise ValueError(
                f"color must be 'rgb', 'bgr', or None, got {color!r}")
        self.color = color
        # R3 auto-build knobs - meaningful only when engine_path is a
        # .onnx: dynamic-batch profile max, precision, and explicit (H, W)
        # for exports whose spatial dims are symbolic (ultralytics
        # dynamic=True); a plain .engine path ignores all three.
        if not isinstance(max_batch, int) or isinstance(max_batch, bool) \
                or max_batch < 1:
            raise ValueError(f"max_batch must be a positive int, got "
                             f"{max_batch!r}")
        if shape is not None:
            try:
                h, w = shape
            except (TypeError, ValueError):
                raise ValueError(
                    f"shape must be an (H, W) tuple or None, got "
                    f"{shape!r}") from None
            if not all(isinstance(v, int) and not isinstance(v, bool)
                       and v >= 32 for v in (h, w)):
                raise ValueError(
                    f"shape must be an (H, W) tuple of ints >= 32, got "
                    f"{shape!r}")
        self.max_batch = max_batch
        self.fp16 = bool(fp16)
        self.shape = tuple(shape) if shape is not None else None

    @staticmethod
    def _parse_norm(norm: Optional[tuple]) -> Optional[Tuple[Tuple[float, float, float],
                                                              Tuple[float, float, float]]]:
        """Validates/normalizes the `norm=` argument (see the docstring
        above for the two accepted shapes) down to a canonical
        ((o0,o1,o2), (s0,s1,s2)) per-channel form, or None. Raises
        ValueError naming BOTH accepted shapes on anything else (M3a) -
        including a MIXED shape (one side scalar, the other per-channel),
        which is neither of the two named forms."""
        if norm is None:
            return None
        shape_err = (
            "norm must be a scalar (offset, scale) tuple (broadcast to all "
            "3 channels) or a per-channel ((o0,o1,o2), (s0,s1,s2)) tuple, "
            f"got {norm!r}"
        )
        try:
            offset, scale = norm
        except (TypeError, ValueError):
            raise ValueError(shape_err) from None

        def _is_num(v) -> bool:
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        def _as_triple(v) -> Optional[Tuple[float, float, float]]:
            if _is_num(v):
                return None  # scalar, not a triple - caller distinguishes
            try:
                v3 = tuple(v)
            except TypeError:
                return None
            if len(v3) != 3 or not all(_is_num(x) for x in v3):
                return None
            return (float(v3[0]), float(v3[1]), float(v3[2]))

        if _is_num(offset) and _is_num(scale):
            return ((float(offset),) * 3, (float(scale),) * 3)
        offset3, scale3 = _as_triple(offset), _as_triple(scale)
        if offset3 is not None and scale3 is not None:
            return (offset3, scale3)
        # Anything else - including a MIXED shape (e.g. one scalar, one
        # 3-tuple) - is neither accepted form.
        raise ValueError(shape_err)


class Postprocess(_Step):
    """Decodes an ``Engine``'s raw tensor output into detections (family
    ``"yolo"``, anchor decode + NMS, or ``"yolo-e2e"``, M4a - NMS-free
    end-to-end detector heads like yolo26, see ``LaunchYoloE2EBatched`` in
    postprocess.h), text (family ``"ctc"``), or a classifier label (family
    ``"argmax"``, M1a - plain max-logit, no softmax: see
    ``LaunchArgmaxBatched`` in postprocess.h) - the latter two aligned with
    the DETECTOR layer's detections in a cascade (M4b: any number of
    sibling ``"ctc"``/``"argmax"`` children may crop the SAME root - see
    the module docstring's "Depth-2 tree" section). ``"yolo"``/``"yolo-e2e"``
    are both LAYER-0 (detection) families; ``"ctc"``/``"argmax"`` are both
    cascade-CHILD families - see ``Family`` in graph.h.

    Args:
        input: the ``Engine`` step whose output this decodes.
        family: ``"yolo"``, ``"yolo-e2e"``, ``"ctc"``, or ``"argmax"`` (see
            ``_FAMILIES``).
        score: score threshold (yolo, yolo-e2e).
        iou: NMS IoU threshold (yolo only - ignored by yolo-e2e, which has
            no NMS stage; see ``_FAMILIES``' yolo-e2e comment).
    """

    kind = _c.StepKind.Postprocess

    def __init__(
        self,
        input: StepInput,
        family: str,
        score: float = 0.4,
        iou: float = 0.45,
    ):
        super().__init__(input)
        if family not in _FAMILIES:
            valid = ", ".join(sorted(_FAMILIES))
            raise ValueError(f"unknown family {family!r}; valid: {valid}")
        self.family = family
        self.score = score
        self.iou = iou


class Layer:
    """A named group of steps compiled into one ``LayerDesc``.

    Args:
        name: this layer's name (keys ``Result.outputs``).
        sahi: ``None`` (default), or a dict turning SAHI (tiled inference -
            see the module docstring's SAHI section) on for THIS layer:
            any of ``tile``, ``overlap``, ``merge_iou``, ``full_frame``,
            ``serial`` (all optional; unset ones keep the ``LayerDesc``
            default - presence of the dict, even ``{}``, is what turns
            SAHI on). The v1 C++ executor (``Validate()``, pipeline.cpp)
            only accepts SAHI on the yolo detection layer (today, always
            layer 0) - putting it on a cascade (ctc/argmax) layer raises a
            named ``RuntimeError`` at ``Pipeline`` construction. This is
            the PREFERRED way to set SAHI; see ``Pipeline(sahi=...)`` for
            the legacy position-based alias it can conflict with.

    ``layer.steps`` is a read-only view of the steps added, in insertion
    order - the same order they land in ``LayerDesc.steps``.
    """

    def __init__(self, name: str, sahi: Optional[dict] = None):
        self.name = name
        self.sahi = sahi
        self._steps: List[_Step] = []

    def add(self, step: _Step) -> _Step:
        """Appends ``step`` to this layer and returns it (a handle you
        pass as another step's ``input=``, or index into results by).
        """
        self._steps.append(step)
        return step

    @property
    def steps(self) -> Sequence[_Step]:
        return tuple(self._steps)


class Sink:
    """A downstream tee off the running pipeline (Phase A1/A2 endpoint
    sinks - see ``SinkKind``/``SinkDesc`` in graph.h). A sink is a strictly
    downstream tap: it NEVER backpressures the pipeline itself - each sink
    drains through its own bounded, drop-oldest queue on a dedicated
    thread, so a slow/stalled consumer of a sink loses its OWN oldest data
    (see ``Pipeline.sink_dropped()``), not frames from the pipeline proper.

    Two kinds:

    ``kind="events"``
        Per-frame detection/text metadata, serialized as one compact NDJSON
        line per ``FrameResult``, in this exact shape::

            {"stream": <int>, "frame": <int>, "pts_us": <int>,
             "batch": <int>,
             "dets": [{"x": <float>, "y": <float>, "w": <float>,
                       "h": <float>, "score": <float>, "cls": <int>}, ...],
             "texts": ["...", ...],
             "ms": {"pre": <float>, "queue": <float>, "gpu": <float>},
             "frame_addr": "0x.."}

        ``ms`` mirrors the ``Result`` timing split (``pre`` = pop-to-ready,
        ``queue`` = ready-to-take, ``gpu`` = take-to-done); ``frame_addr``
        is the same tier-1 informational address as ``Result.frame_addr``
        (printable/loggable only - never dereference it from the sink's
        receiving end). ``input`` must be one of this pipeline's
        ``Postprocess`` steps (the detector layer, or any sibling cascade
        child's - M4b - all work; results are per-frame, not per-layer);
        ``target`` is one of
        ``"tcp://host:port"``, ``"file:///abs/path"``, or ``"stdout"``.
        ``stream`` is an OPTIONAL filter: ``None`` (default) emits every
        stream's results, or pass a stream index to report on just that
        camera.

    ``kind="stream"``
        The ORIGINAL compressed video for one stream, republished as a live
        RTSP stream (``-c copy`` - no re-encode). ``input`` must be the
        pipeline's ``Streams`` object (it forwards the raw source, upstream
        of any inference step - not a postprocess output); ``target`` is
        ``"rtsp://host:port/name"``. ``stream`` is REQUIRED here: v1
        restricts one relay sink to one concrete stream (one relay target
        is one RTSP path, and multiplexing "all streams" onto one path has
        no defined behavior - a deliberate v1 restriction, not an
        oversight). A relay's first connection replays that stream's
        ring history (up to ``Pipeline(ring_seconds=...)`` of it) at
        realtime pace, then goes live - so a viewer that connects mid-run
        still gets a continuous stream starting from up to ``ring_seconds``
        in the past, not a jump straight to "now".

    Args:
        input: a ``Streams`` object (``kind="stream"``) or a ``Postprocess``
            step (``kind="events"``) - the dataflow edge this sink taps.
        kind: ``"events"`` or ``"stream"``.
        target: the sink's destination - see the per-kind grammar above.
            Checked at ``Pipeline`` construction time (fails fast).
        stream: stream index. Required for ``kind="stream"``; an optional
            per-stream filter for ``kind="events"`` (``None`` = all
            streams).
    """

    def __init__(self, input: StepInput, kind: str, target: str,
                 stream: Optional[int] = None):
        if kind not in _SINK_KINDS:
            valid = ", ".join(sorted(_SINK_KINDS))
            raise ValueError(f"unknown sink kind {kind!r}; valid: {valid}")
        if kind == "stream":
            if not isinstance(input, Streams):
                raise ValueError(
                    'Sink(kind="stream") requires input=<the pipeline\'s '
                    'Streams object> - it forwards the raw/original source, '
                    "upstream of any inference step, not a postprocess "
                    "output"
                )
            if stream is None:
                raise ValueError(
                    'Sink(kind="stream") requires stream=<int> - v1 '
                    "restricts one relay sink to one concrete stream (one "
                    "relay target is one RTSP path; \"all streams\" onto "
                    "one path has no defined multiplexing - see SinkDesc "
                    "in graph.h)"
                )
        self.input = input
        self.kind = kind
        self.target = target
        self.stream = stream


class _CudaFrameView:
    """Tiny holder exposing ``__cuda_array_interface__`` (v3) for a
    ``hold_frames`` result's full-res NV12 buffer, so ``torch.as_tensor()``
    / ``cupy.asarray()`` accept it directly as a zero-copy GPU view (both
    libraries accept any object exposing that attribute). Holds a
    reference to the owning ``Result`` - which holds the ``_c.FrameResult``
    whose ``frame_hold`` keeps the ring slot claimed - for exactly as long
    as this view itself is alive: the keep-alive chain a consumer's
    ``t = torch.as_tensor(r.frame_cuda(), device="cuda")`` relies on (`t`
    keeps no reference back to `r` on its own - see CAVEAT below).

    CAVEAT (inherent to the CUDA Array Interface protocol, not specific to
    this wrapper): unlike ``__dlpack__``, CAI carries no deleter/refcount of
    its own - a consumer that keeps `t` around after releasing the hold
    (``r.release_frame()``, or letting every copy of `r` die) holds a
    dangling view. Read everything you need out of `t` before releasing.
    """

    __slots__ = ("_result", "_iface")

    def __init__(self, result: "Result"):
        # Eager, not lazy: validates hold_frames (and captures the dict)
        # immediately, so frame_cuda() raises RuntimeError right here when
        # there is no hold - matching the underlying FrameResult.frame_cuda()
        # binding's own eager check - rather than deferring the failure to
        # whenever something (e.g. torch.as_tensor) happens to read
        # __cuda_array_interface__.
        self._iface = result._r.frame_cuda()
        self._result = result

    @property
    def __cuda_array_interface__(self) -> dict:
        return self._iface


class Result:
    """Wraps one ``_pycamtrt.FrameResult`` with a couple of conveniences:

    - ``outputs``: dict keyed by layer name -> that layer's payload - the
      detections list for the detector (layer 0), and for EVERY sibling
      recognition child (M4b - a depth-2 tree, layers 1..K, see
      ``Pipeline``'s docstring): the texts list for a ``family="ctc"``
      child, or a list of ``(label, score)`` tuples (zip of that child's
      ``ChildOutput.labels``/``label_scores``) for a ``family="argmax"``
      one.
    - every raw ``FrameResult`` field is still exposed directly (stream_id,
      pts_us, frame_no, batch_size, batch_seq, detections, texts, labels,
      label_scores, children, the three ms_* timing fields, verified/
      verify_ok/verify_cpu_dets, and the three-tier frame fields
      frame_addr/frame_pitch/frame_width/frame_height - see the module
      docstring). ``texts``/``labels``/``label_scores`` are the pre-M4b
      BACK-COMPAT fields (first Ctc / first Argmax child only - see
      ``core/result.h``'s WHY-comment); ``children`` is the full per-child
      list (``_pycamtrt.ChildOutput``, one per sibling, in layer order) -
      ``outputs`` above is simply that list already unpacked by name.
    - ``frame_cuda()``/``fetch_frame()``/``release_frame()``: the tier
      2/3 frame-memory accessors (see the module docstring for the full
      three-tier contract). ``frame_cuda()`` and the raw ``frame_addr``
      tier-1 fields are informational/require ``hold_frames=True`` unless
      noted; without ``hold_frames``, only ``fetch_frame()`` and the
      tier-1 fields are meaningful.
    """

    __slots__ = ("_r", "outputs", "_pipeline")

    def __init__(self, r: _c.FrameResult, layer_names: Sequence[str],
                 pipeline: "Pipeline", child_families: Sequence[str] = ()):
        self._r = r
        outputs: Dict[str, object] = {}
        if layer_names:
            outputs[layer_names[0]] = r.detections
            # M4b: EVERY layer past the detector is a sibling recognition
            # child (a depth-2 tree - see Pipeline's docstring), in the SAME
            # order as `r.children` (both mirror cfg.layers[1:] order - see
            # Pipeline.__init__'s compile loop). `child_families[i]` is
            # resolved once at Pipeline construction (from each child
            # layer's own Postprocess step), not re-derived per result -
            # replaces the single-child `l1_argmax` bool this generalizes.
            for i, name in enumerate(layer_names[1:]):
                child = r.children[i] if i < len(r.children) else None
                fam = child_families[i] if i < len(child_families) else "ctc"
                if fam == "argmax":
                    outputs[name] = (
                        list(zip(child.labels, child.label_scores))
                        if child is not None else [])
                elif fam == "embedding":
                    # T3.1: one [D] float list per detection, undecoded
                    # (ChildOutput.vectors) - consumers cosine/L2/cluster
                    # these however they like; np.asarray(v) is zero-fuss.
                    outputs[name] = (list(child.vectors)
                                     if child is not None else [])
                else:
                    outputs[name] = list(child.texts) if child is not None else []
        self.outputs = outputs
        # weakref, not a strong ref: a Result should not be the thing
        # keeping the owning Pipeline (and its threads/CUDA state) alive -
        # it only needs the Pipeline for as long as fetch_frame() might
        # still be called on it.
        self._pipeline = weakref.ref(pipeline)

    def __getattr__(self, name):
        return getattr(self._r, name)

    def __repr__(self):
        r = self._r
        return (
            f"<Result stream_id={r.stream_id} frame_no={r.frame_no} "
            f"batch_size={r.batch_size} dets={len(r.detections)} "
            f"texts={len(r.texts)}>"
        )

    def frame_cuda(self) -> _CudaFrameView:
        """Tier 3 (``hold_frames=True`` only): a zero-copy CUDA view of
        this frame's full-res NV12 buffer (luma + interleaved UV stacked
        as one ``(height*3/2, width)`` uint8 plane) - pass the return
        value directly to ``torch.as_tensor(..., device="cuda")`` or
        ``cupy.asarray(...)``. Raises ``RuntimeError`` (from the
        underlying ``FrameResult.frame_cuda()``) if this pipeline was not
        constructed with ``hold_frames=True``.
        """
        return _CudaFrameView(self)

    def fetch_frame(self):
        """Tier 2: D2H copy of this frame's full-res NV12 buffer into a
        freshly allocated numpy array, shape ``(height*3/2, width)``
        uint8. Requires ``hold_frames=True`` (raises ``RuntimeError``
        otherwise - same requirement as ``frame_cuda()``, since both need
        the ring slot to still be alive when they run).
        """
        pipeline = self._pipeline()
        if pipeline is None:
            raise RuntimeError(
                "fetch_frame(): owning Pipeline has been garbage collected")
        return pipeline._pipe.fetch_frame(self._r)

    def release_frame(self) -> None:
        """Tier 3: drop this result's hold on its ring slot immediately,
        rather than waiting for this ``Result`` (and every other copy of
        its underlying ``FrameResult``) to be garbage collected. A no-op
        if this pipeline was not constructed with ``hold_frames=True``.
        """
        self._r.release_frame()


class Pipeline:
    """Compiles a ``Streams`` + ``Layer`` object graph into a
    ``cordero::PipelineConfig`` and drives the compiled ``_pycamtrt.Pipeline``.

    Args:
        streams: the ``Streams`` object every step graph is rooted at. Any
            entry built with ``Stream(url, skip=..., decode=...)`` overrides
            the pipeline-wide ``skip``/``decode`` below for just that
            camera; plain URL strings inherit them.
        layers: ``Layer`` objects, in execution order. Layer 0's family
            must be ``"yolo"`` for ``sahi=`` to apply (matches
            ``LayerDesc.sahi`` being valid only alongside a YoloDetect
            postprocess - see graph.h).
        skip: infer every Kth decoded frame (``PipelineConfig.skip``);
            pipeline-wide default, overridable per stream (see ``streams``).
        max_frames: decoded frames per stream; 0 = run until ``stop()``.
        verify: per-frame CPU-reference check (``PipelineConfig.verify``).
        decode: ``"all"`` (default, decode every frame) or ``"key"``
            (keyframes only); pipeline-wide default, overridable per stream
            (see ``streams``).
        sahi: ``None``, or a dict of overrides applied to layer 0's
            ``LayerDesc`` - any of ``tile``, ``overlap``, ``merge_iou``,
            ``full_frame``, ``serial`` (all optional; unset ones keep the
            ``LayerDesc`` default). Presence of the dict (even ``{}``)
            turns SAHI on. This is the LEGACY, position-based form (it
            always means "layer 0", however many layers there are) -
            prefer ``Layer(sahi=...)`` on the layer itself, which also
            works on any layer (though v1's ``Validate()`` still only
            accepts it on the yolo detection layer). Giving both this AND
            ``layers[0]``'s own ``sahi=`` with DIFFERENT dicts raises
            ``ValueError`` (ambiguous which one wins); giving both the
            same dict is redundant but fine - Layer-level always wins
            (it is applied first; this legacy path is then a no-op).
        log: a callable taking one ``str``, invoked from C++ worker
            threads - keep it cheap (it runs on the hot path and holds the
            GIL while it executes). ``None`` (default) uses the C++ side's
            default stderr logging.
        queue_capacity: result-queue depth (``PipelineConfig.queue_capacity``).
        backpressure: ``"block"`` (default - the GPU thread waits when the
            result queue is full, current/v0 behavior) or ``"drop_oldest"``
            (the GPU thread never blocks: a full queue evicts its oldest
            result to make room, so producers keep near-real-time pace
            against a slow consumer). See ``dropped_results()`` to detect
            evictions in the latter mode.
        hold_frames: tier 3 of the three-tier frame-memory access (see the
            module docstring) - keeps each result's full-res NV12 ring
            slot alive until the result (every copy of it) is garbage
            collected or ``Result.release_frame()`` is called, enabling
            ``Result.frame_cuda()`` (zero-copy torch/CuPy view) in
            addition to the always-available ``Result.fetch_frame()``
            (D2H copy). Off by default (``False``) - the tier-1 fields
            (``frame_addr`` etc.) stay informational-only either way.
        ring_depth: frames-in-flight slack per producer/stream (see the
            module docstring's hold_frames/ring_depth contract) - default
            4 is fine when not holding frames; raise it when
            ``hold_frames=True`` to size for how many frames of one
            stream you plan to hold concurrently (a smaller value stalls
            that stream's producer sooner - intended backpressure).
        sinks: ``Sink`` objects (Phase A1/A2 endpoint sinks - see ``Sink``
            for the full NDJSON schema, target grammar, and the
            replay-then-live relay contract). Empty by default: no sink
            threads are spawned and there is no behavior change versus a
            pipeline that never sets this. Every sink is a strictly
            downstream tap - a slow/stalled sink consumer never
            backpressures the pipeline itself (see ``sink_dropped()``).
        ring_seconds: fixed-duration ring of the ORIGINAL compressed
            packets kept per stream, feeding ``kind="stream"`` sinks and
            ``clip()`` (``PipelineConfig.ring_seconds`` - see graph.h).
            Always allocated (cheap when empty) but only pushed into when
            ``> 0``, at a small memcpy-per-packet cost paid for every
            demuxed packet regardless of skip/decode filtering - that's
            the point: sinks/clips see everything demuxed, not just what
            gets decoded/inferred. Default 10 (seconds); ``0`` opts a
            pipeline out of the ring entirely (no relay/clip support, no
            per-packet copy cost).
        cascade_serial: ``PipelineConfig.cascade_serial`` (CP1) - ``False``
            (default) runs every cascade child's crop/infer/decode on its
            OWN CUDA stream, concurrently, with no synchronization between
            siblings. ``True`` is the A/B escape hatch: today's pre-CP1
            sequential path (every child fully processed, in turn, on the
            single shared GPU stream) - unchanged, kept for regression
            isolation and side-by-side timing comparison (see each child's
            ``ChildOutput.ms_gpu`` in ``outputs``/``children``). No effect
            on a pipeline with no cascade children.

    Usage: iterate the pipeline directly (``for r in pipe:``) - iteration
    auto-starts if ``start()`` wasn't called yet - or use it as a context
    manager (``with pycamtrt.Pipeline(...) as pipe:``), which calls
    ``stop()`` on exit.
    """

    def __init__(
        self,
        streams: Streams,
        layers: Sequence[Layer],
        skip: int = 1,
        max_frames: int = 0,
        verify: bool = False,
        decode: str = "all",
        sahi: Optional[dict] = None,
        log: Optional[Callable[[str], None]] = None,
        queue_capacity: int = 256,
        backpressure: str = "block",
        hold_frames: bool = False,
        ring_depth: int = 4,
        sinks: Sequence[Sink] = (),
        ring_seconds: int = 10,
        cascade_serial: bool = False,
    ):
        if decode not in _DECODE_MODES:
            raise ValueError(f"decode must be one of {_DECODE_MODES}, got {decode!r}")
        if backpressure not in _BACKPRESSURE:
            valid = ", ".join(sorted(_BACKPRESSURE))
            raise ValueError(
                f"backpressure must be one of {valid}, got {backpressure!r}"
            )

        self._layer_names = [l.name for l in layers]
        # M4b: per-CHILD family map (one entry per layer past the
        # detector, in order) - replaces M1a's single-child `_l1_argmax`
        # bool now that v1 runs a depth-2 TREE (N sibling Ctc/Argmax
        # children, not just one). Resolved once here (from the
        # python-level Layer/Postprocess objects, before compiling to
        # StepDesc indices) so Result() never has to re-derive it per
        # poll()ed frame - see Result.__init__ and __next__ below. A
        # layer's payload is (label, score) pairs iff its LAST step is a
        # Postprocess(family="argmax"); any other/malformed shape defaults
        # to "ctc" here and is left for the C++ Validate() below to reject
        # with the authoritative named error (this map only needs to be
        # correct for graphs that DO validate).
        self._child_families: List[str] = []
        for layer in layers[1:]:
            fam = "ctc"
            if layer.steps and isinstance(layer.steps[-1], Postprocess) and \
                    layer.steps[-1].family in ("argmax", "embedding"):
                fam = layer.steps[-1].family
            self._child_families.append(fam)
        cfg = _c.PipelineConfig()
        # skip None -> 0 (inherit cfg.skip), else the override as-is;
        # decode None -> -1 (inherit cfg.key_only), "all" -> 0, "key" -> 1 -
        # the StreamDesc sentinels graph.h documents.
        c_streams: List[_c.StreamDesc] = []
        for entry in streams.entries:
            c_sd = _c.StreamDesc()
            c_sd.url = entry.url
            c_sd.skip = 0 if entry.skip is None else entry.skip
            c_sd.decode = -1 if entry.decode is None else (
                0 if entry.decode == "all" else 1)
            c_streams.append(c_sd)
        cfg.streams = c_streams
        cfg.skip = skip
        cfg.key_only = decode == "key"
        cfg.verify = verify
        cfg.max_frames = max_frames
        cfg.queue_capacity = queue_capacity
        cfg.backpressure = _BACKPRESSURE[backpressure]
        cfg.hold_frames = hold_frames
        cfg.ring_depth = ring_depth
        cfg.ring_seconds = ring_seconds
        cfg.cascade_serial = cascade_serial
        cfg.log = log

        # object identity -> compiled step index (Streams maps to -1, the
        # "raw stream source" sentinel StepDesc.input expects).
        index_of: Dict[int, int] = {id(streams): -1}
        c_steps: List[_c.StepDesc] = []
        c_layers: List[_c.LayerDesc] = []

        def compile_input(step_input: StepInput) -> int:
            idx = index_of.get(id(step_input))
            if idx is None:
                raise ValueError(
                    "step input refers to an object not present earlier in "
                    "the graph (steps must be added to a Layer in "
                    "dependency order)"
                )
            return idx

        for layer in layers:
            c_layer = _c.LayerDesc()
            c_layer.name = layer.name
            layer_step_indices: List[int] = []
            for step in layer.steps:
                c_step = _c.StepDesc()
                c_step.kind = step.kind
                c_step.input = compile_input(step.input)
                if isinstance(step, Engine):
                    c_step.engine_path = step.engine_path
                    # M1a/M3a: leave norm_offset/norm_scale/color at
                    # StepDesc's own NAN/-1 "inherit" defaults when not
                    # given (see Engine's docstring) - only overwrite what
                    # the caller actually set. step.norm is already
                    # canonicalized to ((o0,o1,o2), (s0,s1,s2)) by
                    # Engine._parse_norm regardless of which shape the
                    # caller passed.
                    if step.norm is not None:
                        c_step.norm_offset, c_step.norm_scale = step.norm
                    if step.color is not None:
                        c_step.color = 1 if step.color == "rgb" else 0
                    c_step.build_max_batch = step.max_batch
                    c_step.build_fp16 = step.fp16
                    if step.shape is not None:
                        c_step.build_h, c_step.build_w = step.shape
                elif isinstance(step, Postprocess):
                    c_step.family = _FAMILIES[step.family]
                    c_step.score_thresh = step.score
                    c_step.iou_thresh = step.iou
                elif isinstance(step, Select):
                    # R1 routing: empty/0 = criterion off (see graph.h's
                    # StepDesc sel_* WHY-comment).
                    if step.classes is not None:
                        c_step.sel_classes = step.classes
                    if step.min_score is not None:
                        c_step.sel_min_score = float(step.min_score)
                    if step.min_size is not None:
                        c_step.sel_min_size = float(step.min_size)
                new_idx = len(c_steps)
                c_steps.append(c_step)
                index_of[id(step)] = new_idx
                layer_step_indices.append(new_idx)
            c_layer.steps = layer_step_indices
            if layer.sahi is not None:
                _apply_sahi(c_layer, layer.sahi)
            c_layers.append(c_layer)

        # Legacy Pipeline(sahi=...) alias: applies to layer 0 only (see
        # this param's docstring above). Layer-level wins on conflict -
        # if layers[0] ALREADY got its own sahi= applied above and this
        # legacy dict differs, that's ambiguous (which one wins?) and we
        # say so loudly rather than silently picking one; an identical
        # dict is redundant-but-harmless (the loop above already applied
        # it, so this is a no-op).
        if sahi is not None:
            if not c_layers:
                raise ValueError("sahi given but no layers")
            if layers[0].sahi is not None:
                if layers[0].sahi != sahi:
                    raise ValueError(
                        "Pipeline(sahi=...) and layers[0]'s own "
                        "Layer(sahi=...) were both given, with DIFFERENT "
                        "dicts - ambiguous (which one wins?). Pass the "
                        "same dict to both (redundant but allowed), or "
                        "set it in only one place - Layer(sahi=...) is "
                        "the preferred, unambiguous form."
                    )
                # Identical dict: already applied via the per-layer path
                # above - nothing left to do.
            else:
                _apply_sahi(c_layers[0], sahi)

        # Sinks compile last: kind="events" sinks reference a Postprocess
        # step's compiled index, so index_of must already hold every step
        # added above (kind="stream" sinks only ever reference `streams`
        # itself, already seeded into index_of as -1).
        c_sinks: List[_c.SinkDesc] = []
        for sink in sinks:
            c_sink = _c.SinkDesc()
            c_sink.kind = _SINK_KINDS[sink.kind]
            c_sink.input = compile_input(sink.input)
            c_sink.stream_id = -1 if sink.stream is None else sink.stream
            c_sink.target = sink.target
            c_sinks.append(c_sink)
        cfg.sinks = c_sinks

        cfg.steps = c_steps
        cfg.layers = c_layers

        # Construction (engine load + warmup) happens here, under the GIL
        # released on the C++ side. Raises RuntimeError (translated from
        # std::runtime_error) on any bad-graph-shape/load failure - see
        # pipeline.cpp Validate().
        self._pipe = _c.Pipeline(cfg)
        self._started = False

    def start(self) -> None:
        """Spawns producer + GPU threads. Single-shot - calling twice
        raises (mirrors ``cordero::Pipeline::Start``).
        """
        self._pipe.start()
        self._started = True

    def stop(self) -> None:
        """Idempotent; safe from any thread; blocks until fully torn
        down. Also called by ``__exit__``.
        """
        self._pipe.stop()

    def get_stream_info(self, i: int) -> _c.StreamInfo:
        return self._pipe.get_stream_info(i)

    def dropped_results(self) -> int:
        """Count of results evicted by ``backpressure="drop_oldest"``
        (always 0 under the default ``"block"`` mode).
        """
        return self._pipe.dropped_results()

    def sink_dropped(self) -> int:
        """Total NDJSON lines dropped across every ``kind="events"`` sink's
        own bounded drop-oldest queue (see ``Sink``) - a sink never
        backpressures the pipeline, so a receiver that falls behind loses
        its oldest lines instead. Always 0 if no events sink is configured,
        or none has ever fallen behind its target.
        """
        return self._pipe.sink_dropped()

    def clip(self, stream_id: int, seconds_back: float, path: str) -> bool:
        """Write a raw Annex-B ``.h264`` elementary stream of ``stream_id``
        covering roughly the last ``seconds_back`` seconds (from the newest
        keyframe at-or-before that point, to now) to ``path`` - a snapshot
        pulled from the same packet ring ``kind="stream"`` sinks replay
        from (``Pipeline(ring_seconds=...)``), independent of whether any
        sink is configured. Returns ``False`` (and logs why - see
        ``Pipeline(log=...)``) if ``stream_id`` is invalid, the ring is
        disabled (``ring_seconds=0``) or still empty, or no keyframe is
        available yet. Safe to call from any thread while the pipeline is
        running.
        """
        return self._pipe.extract_clip(stream_id, seconds_back, path)

    def max_batch(self) -> int:
        return self._pipe.max_batch()

    def classes(self) -> int:
        return self._pipe.classes()

    def anchors(self) -> int:
        return self._pipe.anchors()

    def __iter__(self) -> "Pipeline":
        if not self._started:
            self.start()
        return self

    def __next__(self) -> Result:
        while True:
            # timeout_ms=500: Poll releases the GIL for the wait, so
            # Python (Ctrl-C, other threads) stays responsive; a Timeout
            # just loops back around rather than propagating.
            status, r = self._pipe.poll(500)
            if status == PollStatus.Timeout:
                continue
            if status == PollStatus.Finished:
                raise StopIteration
            return Result(r, self._layer_names, self, self._child_families)

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def __del__(self) -> None:
        # Deadlock guard, not just tidiness: when Python garbage-collects
        # this object, pybind11 runs the C++ Pipeline destructor WHILE
        # HOLDING THE GIL. If a ``log=`` Python callback is set, a
        # producer/GPU thread may at that moment be blocked acquiring the
        # GIL inside the callback trampoline - and the destructor's thread
        # joins would then wait on it forever. Calling stop() here goes
        # through the binding that RELEASES the GIL for the wait, so every
        # thread is joined (and no callback can be pending) before the C++
        # destructor runs.
        try:
            self.stop()
        except Exception:
            pass  # interpreter shutdown: _pipe/module may be half-gone
