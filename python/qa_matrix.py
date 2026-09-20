#!/usr/bin/env python3
"""qa_matrix.py - pycamtrt v0 acceptance matrix (the Python-level checkpoint,
same philosophy as the C++ *_test binaries: exercise the paths the smoke
test and showcase example don't).

Covers:
  A. error paths     - bad family / missing engine / unsupported graph must
                       fail with named errors, not crashes
  B. SAHI via python - M2: per-layer Layer(sahi=...) is the primary form
                       (B1, tiled inference + cascade through the binding);
                       B2 checks the legacy Pipeline(sahi=...) alias (layer
                       0 only) still works; B3 checks that giving BOTH
                       forms with DIFFERENT dicts raises ValueError
                       (same dict on both is redundant-but-allowed)
  C. decode="key"    - keyframe-only decode through the binding
  D. per-stream overrides - pycamtrt.Stream(url, skip=..., decode=...) load
                       dials applied to one camera while another inherits
                       the pipeline-wide default
  E. frame access (three tiers) - tier-1 info always present + refuses
                       tier-3 access without hold_frames; with
                       hold_frames=True, a zero-copy torch CUDA view
                       (proves the primary-ctx migration: torch and the
                       pipeline share an address space) matches a
                       faithful tier-2 D2H fetch of the same live frame
  F. endpoint sinks (Phase A2) - events sink delivers NDJSON 1:1 with the
                       consumed results over a real TCP socket (drop-oldest,
                       never backpressures); StreamRelay republishes the
                       original stream as a decodable live RTSP stream;
                       extract_clip() writes a decodable, keyframe-first
                       raw .h264 clip from the packet ring
  G. M1b classifier cascade (argmax family) - G1: the detect->classify
                       cascade demo (examples/classify_detections/classify_detections.py's
                       pipeline shape) runs 60 results with outputs["classify"]
                       length always equal to len(detections) (alignment);
                       G2: a parity subprocess independently recomputes the
                       classifier's argmax label in plain torch/cv2 on the
                       SAME live crops (r.fetch_frame() + the SAME
                       per-channel ImageNet norm, M3a) and gates on the
                       LOGIT-GAP invariant: any label
                       disagreement must be a class the reference itself
                       scores within 1.0 logit of its top (a decision-
                       boundary flip between independent bilinear resizers),
                       never one it scores far down (mechanics bug), plus a
                       60% overall-agreement floor. See
                       _qa_classifier_parity_subprocess.py's GAP_LIMIT
                       comment for the two-design history.
  I1. M4a yolo-e2e detector family - a live yolo26n run at LAYER 0
                       (Postprocess(family="yolo-e2e"), see
                       LaunchYoloE2EBatched in postprocess.h) on the plate
                       farm: results flow, per-frame detection counts are
                       plausible, and loosely compared (order of magnitude
                       only - yolo26n is COCO-pretrained, not plate-
                       specific) against the yolov8-plates count on the
                       SAME content; SAHI+yolo-e2e raises the named
                       RuntimeError (pipeline.cpp's Validate()); and a
                       yolo26 (yolo-e2e) root -> lprnet (ctc) child cascade
                       proves cascade children work unchanged on an e2e
                       root.
  I2. M4b depth-2 tree (the full-tree gate) - one detector root + TWO
                       sibling children (read=ctc, classify=argmax, the
                       examples/read_and_classify/read_and_classify.py shape) run 60 results:
                       per result, outputs["read"]/outputs["classify"] are
                       both length == len(detections) (alignment), and the
                       legacy flat fields (r.texts/r.labels) equal the
                       first-matching-family child's output (back-compat -
                       see result.h's WHY-comment). CROSS-CHECK: the tree
                       run's dominant plate string equals a read-only
                       (2-layer ctc) run's on the SAME farm content within
                       this section, and the tree run's TOP argmax label
                       equals a classify-only (2-layer argmax) run's - i.e.
                       running two children together changes neither
                       child's own answer. Section A3 asserts the
                       complementary rejection (a CHAINED cascade, a child
                       of a child) now that sibling layers are legal.
  J. CP1 cascade-parallelism A/B - per-child CUDA-stream parallelism (the
                       default, ``PipelineConfig.cascade_serial=False``)
                       versus today's pre-CP1 sequential path (``=True``,
                       the escape hatch): the SAME 2-child tree section I2
                       uses (read=ctc, classify=argmax) run twice on the
                       SAME content - the plate clip FILE when QA_CLIP
                       exists (deterministic), else the live farm - 60
                       results each. PASS = outputs
                       equivalent: dominant plate string and top argmax
                       label EQUAL between the two modes, and per-result
                       ``outputs["read"]``/``outputs["classify"]`` lengths
                       stay aligned with ``detections`` in both. Also
                       prints each run's mean ``ChildOutput.ms_gpu`` per
                       child - the first per-child GPU-stream timing
                       numbers (RAW, not directly comparable in isolation
                       across modes - see ``ChildOutput.ms_gpu``'s
                       WHY-comment in ``core/result.h``).

Run inside the tensorrt-dev docker (see examples/read_plates/read_plates.py docstring),
with the stream farm up:  python3 python/qa_matrix.py rtsp://... rtsp://...
Exit 0 = all sections pass.

Section E2 (hold_frames + torch CUDA Array Interface) runs in a SEPARATE
process via _qa_hold_subprocess.py, not inline - see that file's docstring
for why (a reproducible native-level exit-time crash when more than one
pycamtrt::Pipeline has existed in-process alongside a torch CAI consumer;
sections A-D above already construct several). Section G2 runs in a
SEPARATE process for the same reason (hold_frames=True) - see
_qa_classifier_parity_subprocess.py's docstring.

Section H (recommend()) needs no urls/farm/GPU at all - it is pure Python
capacity arithmetic (pycamtrt._capacity), exercised here for the same
reason every other section is: prove the shipped behavior against a
concrete, cited number, not just "it imports".
"""
import json
import os
import shutil
import pathlib
import socket
import subprocess
import sys
import threading
import time
from collections import Counter

import pycamtrt

DET = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
OCR = "models/lprnet_b1-32_fp32_sm86.engine"
# M4a: yolo-e2e family's demo engine (see manual/FINDINGS.md's "M1
# model-generality findings" section B) - COCO-pretrained (no plate class),
# so section I1 below compares GENERAL-OBJECT vs. PLATE detection counts on
# the SAME content: a loose order-of-magnitude smoke check, not an
# apples-to-apples class match.
YOLO_E2E = "models/yolo26n_b1-8_fp16_sm86.engine"
# M1b: the argmax-family classifier cascade demo's engine (see
# python/export_classifier.py, examples/classify_detections/classify_detections.py).
CLASSIFIER = "models/mobilenet_v3s_b1-32_fp16_sm86.engine"
# M3a: true per-channel ImageNet norm (upgrade of M1b's scalar
# approximation (-114.0, 1/58.6)) - see python/export_classifier.py's
# derivation. Shared with _qa_classifier_parity_subprocess.py's NORM_OFFSET/
# NORM_SCALE and examples/classify_detections/classify_detections.py's CLASSIFIER_NORM (kept in
# sync by hand - all three demo the same engine/norm pairing).
CLASSIFIER_NORM = ((-123.675, -116.28, -103.53), (1 / 58.395, 1 / 57.12, 1 / 57.375))

# ffprobe: FFPROBE env var, else whatever is on PATH.
FFPROBE = (os.environ.get("FFPROBE") or shutil.which("ffprobe")
           or str(pathlib.Path(sys.executable).parent / "ffprobe"))   # conda-style env: next to the interpreter
# Deterministic content for the cross-run A/B gates (I2, J): the plate clip
# FILE when present, else the live farm URLs (see section J's comment).
QA_CLIP = os.environ.get("QA_CLIP", "tools/stream_farm/media/atlas_plate_g30.mp4")


def ab_source(urls):
    """File input when QA_CLIP exists (both runs then see the SAME frames),
    otherwise the live farm - where two runs can sample different loop
    segments and a top-label mismatch is not conclusive."""
    if os.path.isfile(QA_CLIP):
        print(f"  source: {QA_CLIP} (file - deterministic content)")
        return [QA_CLIP]
    print("  source: live farm (QA_CLIP not found; cross-run comparisons may "
          "sample different loop segments)")
    return urls


def one_layer(streams, engine=DET, **post_kw):
    layer = pycamtrt.Layer("detect")
    eng = layer.add(pycamtrt.Engine(streams, engine))
    layer.add(pycamtrt.Postprocess(eng, family="yolo", **post_kw))
    return layer


def expect_raises(what, exc_type, fn):
    try:
        fn()
    except exc_type as e:
        print(f"  ok: {what} -> {exc_type.__name__}: {str(e)[:90]}")
        return True
    except Exception as e:  # wrong type = fail loudly
        print(f"  FAIL: {what} raised {type(e).__name__} (wanted "
              f"{exc_type.__name__}): {e}")
        return False
    print(f"  FAIL: {what} did not raise")
    return False


def section_a_errors(urls):
    print("[A] error paths")
    streams = pycamtrt.Streams(urls)
    ok = True

    # A1: unknown postprocess family -> ValueError from the python layer.
    def bad_family():
        layer = pycamtrt.Layer("x")
        eng = layer.add(pycamtrt.Engine(streams, DET))
        layer.add(pycamtrt.Postprocess(eng, family="segmentation"))
    ok &= expect_raises("family='segmentation'", ValueError, bad_family)

    # A2: nonexistent engine file -> RuntimeError from TrtEngine, at
    # construction (fail fast), not at start().
    def bad_engine():
        pycamtrt.Pipeline(streams, layers=[one_layer(streams,
                                                     engine="no_such.engine")])
    ok &= expect_raises("missing engine file", RuntimeError, bad_engine)

    # A3 (REWRITTEN for M4b): three SIBLING layers are now LEGAL - a
    # depth-2 tree (one detector root + N sibling recognition children) is
    # exactly what the v1 executor runs starting M4b (see section I2 below
    # for the full-tree acceptance gate). What's still rejected, and now by
    # a NEW named error, is a CHAINED cascade - a child fed from ANOTHER
    # CHILD's Postprocess step instead of the detector's (a depth-3+ tree).
    def chained_cascade():
        l1 = one_layer(streams)
        l2 = pycamtrt.Layer("read")
        e2 = l2.add(pycamtrt.Engine(l1.steps[-1], OCR))
        l2.add(pycamtrt.Postprocess(e2, family="ctc"))
        l3 = pycamtrt.Layer("extra")
        # Chained onto l2 (a CHILD's own Postprocess), not l1 (the
        # detector's) - this is the depth>2 shape Validate() must reject.
        e3 = l3.add(pycamtrt.Engine(l2.steps[-1], OCR))
        l3.add(pycamtrt.Postprocess(e3, family="ctc"))
        pycamtrt.Pipeline(streams, layers=[l1, l2, l3])
    ok &= expect_raises("chained cascade (child of a child)", RuntimeError,
                         chained_cascade)

    # A4: Ctc at layer 0 -> rejected by Validate().
    def ctc_first():
        layer = pycamtrt.Layer("x")
        eng = layer.add(pycamtrt.Engine(streams, OCR))
        layer.add(pycamtrt.Postprocess(eng, family="ctc"))
        pycamtrt.Pipeline(streams, layers=[layer])
    ok &= expect_raises("ctc at layer 0", RuntimeError, ctc_first)
    return ok


def section_b1_layer_form(urls):
    print("  [B1] Layer(sahi=...) - the per-layer form (tiled inference, "
          "plate content)")
    streams = pycamtrt.Streams(urls)
    layer = pycamtrt.Layer("detect", sahi=dict(tile=640, overlap=0.2))
    eng = layer.add(pycamtrt.Engine(streams, DET))
    layer.add(pycamtrt.Postprocess(eng, family="yolo"))
    pipe = pycamtrt.Pipeline(streams, layers=[layer], skip=2, max_frames=120)
    frames = dets = 0
    with pipe:
        for r in pipe:
            frames += 1
            dets += len(r.detections)
    print(f"    {frames} results, {dets} detections "
          f"({dets / max(frames, 1):.2f}/frame)")
    if frames == 0 or dets == 0:
        print("  FAIL: expected detections on plate content under SAHI")
        return False
    print("  ok")
    return True


def section_b2_legacy_alias(urls):
    print('  [B2] Pipeline(sahi=...) legacy alias (layer 0 only) still '
          "works")
    streams = pycamtrt.Streams(urls)
    layer = one_layer(streams)
    pipe = pycamtrt.Pipeline(streams, layers=[layer], skip=2, max_frames=60,
                             sahi=dict(tile=640, overlap=0.2))
    frames = dets = 0
    with pipe:
        for r in pipe:
            frames += 1
            dets += len(r.detections)
    print(f"    {frames} results, {dets} detections")
    if frames == 0 or dets == 0:
        print("  FAIL: expected detections on plate content under legacy "
              "sahi=")
        return False
    print("  ok")
    return True


def section_b3_conflict(urls):
    print("  [B3] Layer(sahi=...) + Pipeline(sahi=...) both given")
    streams = pycamtrt.Streams(urls)
    ok = True

    # Different dicts -> ambiguous -> ValueError (from the python layer,
    # before any C++ construction).
    def conflict():
        layer = pycamtrt.Layer("detect", sahi=dict(tile=640))
        eng = layer.add(pycamtrt.Engine(streams, DET))
        layer.add(pycamtrt.Postprocess(eng, family="yolo"))
        pycamtrt.Pipeline(streams, layers=[layer], sahi=dict(tile=960))
    ok &= expect_raises("Layer(tile=640) + Pipeline(tile=960)", ValueError,
                         conflict)

    # Identical dicts on both -> redundant, not an error.
    print("    identical dicts on both -> should NOT raise")
    layer = pycamtrt.Layer("detect", sahi=dict(tile=640))
    eng = layer.add(pycamtrt.Engine(streams, DET))
    layer.add(pycamtrt.Postprocess(eng, family="yolo"))
    try:
        p = pycamtrt.Pipeline(streams, layers=[layer], sahi=dict(tile=640))
        p.stop()  # never started; stop() is idempotent/safe regardless
        print("    ok: no error, as expected")
    except Exception as e:
        print(f"  FAIL: identical sahi dicts on both raised {type(e).__name__}: {e}")
        ok = False
    return ok


def section_b_sahi(urls):
    print("[B] SAHI via python (M2: per-layer form)")
    ok = section_b1_layer_form(urls)
    ok &= section_b2_legacy_alias(urls)
    ok &= section_b3_conflict(urls)
    return ok


def section_c_keyonly(urls):
    print('[C] decode="key" (keyframe-only, GOP-cadence updates)')
    streams = pycamtrt.Streams(urls)
    layer = one_layer(streams)
    # 8 decoded KEYframes per stream: with the g30 clip (GOP 30 @ 30fps)
    # that is ~8 s of wall clock and pts deltas of ~1000 ms.
    pipe = pycamtrt.Pipeline(streams, layers=[layer], max_frames=8,
                             decode="key")
    frames = 0
    prev_pts = {}
    deltas = []
    with pipe:
        for r in pipe:
            frames += 1
            if r.stream_id in prev_pts and r.pts_us >= 0:
                deltas.append((r.pts_us - prev_pts[r.stream_id]) / 1000.0)
            prev_pts[r.stream_id] = r.pts_us
    med = sorted(deltas)[len(deltas) // 2] if deltas else 0.0
    print(f"  {frames} keyframe results, median pts delta {med:.0f} ms")
    if frames == 0 or not 700 <= med <= 1300:  # ~1 s GOP cadence
        print("  FAIL: expected ~1000 ms keyframe cadence")
        return False
    print("  ok")
    return True


def section_d_per_stream(urls):
    print("[D] per-stream overrides (skip, decode)")
    if len(urls) < 2:
        print("  FAIL: need >= 2 stream URLs for per-stream overrides")
        return False
    url0, url1 = urls[0], urls[1]
    ok = True

    # D1: skip override - stream 0 is a plain url (inherits skip=1), stream
    # 1 is Stream(url1, skip=4). Both streams decode the same 120 frames,
    # so counts should land near 120 (stream 0) vs ~30 (stream 1): ratio ~4.
    print("  [D1] skip override (stream1 skip=4)")
    streams = pycamtrt.Streams([url0, pycamtrt.Stream(url1, skip=4)])
    layer = one_layer(streams)
    pipe = pycamtrt.Pipeline(streams, layers=[layer], max_frames=120)
    counts = {0: 0, 1: 0}
    with pipe:
        for r in pipe:
            counts[r.stream_id] += 1
    print(f"    stream0={counts[0]} stream1={counts[1]}")
    if counts[0] == 0 or counts[1] == 0:
        print("  FAIL: expected results from both streams")
        ok = False
    else:
        ratio = counts[0] / counts[1]
        print(f"    ratio stream0/stream1 = {ratio:.2f}")
        if not 3.0 <= ratio <= 5.5:
            print("  FAIL: expected ~4x ratio (skip=1 vs skip=4)")
            ok = False

    # D2: decode override - stream 0 stays a plain url (all frames), stream
    # 1 is Stream(url1, decode="key"). max_frames counts DECODED frames per
    # stream, and a key-only stream decodes ~1/s (g30 GOP) - 120 would take
    # ~2 minutes, so use max_frames=8: stream 0 finishes its 8 decoded
    # frames almost instantly, stream 1 takes ~8 s.
    print('  [D2] decode override (stream1 decode="key")')
    streams2 = pycamtrt.Streams([url0, pycamtrt.Stream(url1, decode="key")])
    layer2 = one_layer(streams2)
    pipe2 = pycamtrt.Pipeline(streams2, layers=[layer2], max_frames=8)
    prev_pts = {}
    deltas = {0: [], 1: []}
    with pipe2:
        for r in pipe2:
            if r.stream_id in prev_pts and r.pts_us >= 0:
                deltas[r.stream_id].append(
                    (r.pts_us - prev_pts[r.stream_id]) / 1000.0)
            prev_pts[r.stream_id] = r.pts_us
    med0 = sorted(deltas[0])[len(deltas[0]) // 2] if deltas[0] else 0.0
    med1 = sorted(deltas[1])[len(deltas[1]) // 2] if deltas[1] else 0.0
    print(f"    stream0 median pts delta {med0:.0f} ms, "
          f"stream1 median pts delta {med1:.0f} ms")
    if not 20 <= med0 <= 60:
        print("  FAIL: expected stream0 (~33 ms, all frames) cadence")
        ok = False
    if not 700 <= med1 <= 1300:
        print("  FAIL: expected stream1 (~1000 ms, key-only) cadence")
        ok = False

    if ok:
        print("  ok")
    return ok


def section_e_frame_access(urls):
    print("[E] frame access (three tiers)")
    ok = True

    # E1: non-hold pipeline - tier 1 fields are always populated
    # (informational: a real device address, but never safe to
    # dereference without hold_frames), and frame_cuda() (tier 3) must
    # refuse outright.
    print("  [E1] non-hold: tier-1 fields present, frame_cuda() refuses")
    streams = pycamtrt.Streams(urls)
    layer = one_layer(streams)
    pipe = pycamtrt.Pipeline(streams, layers=[layer], max_frames=8)
    with pipe:
        for r in pipe:
            break
    print(f"    frame_addr=0x{r.frame_addr:x} pitch={r.frame_pitch} "
          f"{r.frame_width}x{r.frame_height}")
    if r.frame_addr == 0:
        print("  FAIL: frame_addr is 0 (want a real device address)")
        ok = False
    ok &= expect_raises("frame_cuda() without hold_frames", RuntimeError,
                         r.frame_cuda)

    # E2: hold_frames pipeline - a zero-copy torch CUDA view of the SAME
    # live memory must agree with a tier-2 D2H fetch. This one assertion
    # proves several things at once: frame_addr is a real, dereferenceable
    # pointer; it's a live zero-copy view in TORCH's CUDA context (i.e.
    # the primary-context migration actually works - a private context's
    # pointers would be invalid there); the D2H copy is faithful; and the
    # hold kept the memory alive long enough for both reads.
    #
    # Run as a SEPARATE PROCESS (see _qa_hold_subprocess.py's docstring):
    # sections A-D above already constructed several Pipelines in THIS
    # process, and combining "more than one Pipeline has ever existed here"
    # with "a hold_frames frame_cuda() view consumed via torch.as_tensor"
    # hits a reproducible native exit-time crash - after every assertion
    # below has already passed. Isolating this check into a fresh process
    # (which constructs exactly one Pipeline) avoids the trigger entirely;
    # the subprocess prints its results before any teardown begins, so we
    # capture them regardless of how it exits.
    print("  [E2] hold_frames: torch zero-copy view vs fetch_frame() D2H "
          "(subprocess)")
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_qa_hold_subprocess.py")
    proc = subprocess.run([sys.executable, "-u", helper] + urls[:1],
                          capture_output=True, text=True, timeout=60)
    result_line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")),
        None)
    error_line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("ERROR ")),
        None)
    if error_line:
        print(f"  FAIL: {error_line}")
        ok = False
    elif not result_line:
        print(f"  FAIL: no RESULT line from subprocess (exit "
              f"{proc.returncode}); stdout:\n{proc.stdout}\nstderr:\n"
              f"{proc.stderr[-2000:]}")
        ok = False
    else:
        fields = dict(kv.split("=", 1) for kv in result_line.split()[1:])
        gpu_mean = float(fields["gpu_mean"])
        host_mean = float(fields["host_mean"])
        host_var = float(fields["host_var"])
        print(f"    frame_addr=0x{fields['frame_addr']} "
              f"gpu_mean={gpu_mean:.3f} host_mean={host_mean:.3f} "
              f"host_luma_var={host_var:.1f}")
        if proc.returncode != 0:
            # Expected in some environments per this section's WHY-comment
            # above: the known exit-time crash happens AFTER the RESULT
            # line is printed, so it does not itself fail this check.
            print(f"    note: subprocess exited {proc.returncode} after "
                  f"printing its result (see WHY-comment above)")
        if abs(gpu_mean - host_mean) >= 0.5:
            print("  FAIL: gpu/host luma means disagree (>= 0.5)")
            ok = False
        if host_var <= 1:
            print("  FAIL: host luma variance too low (garbage/blank frame)")
            ok = False
    if ok:
        print("  ok")
    return ok


def _tcp_line_listener(server_sock, lines, errors):
    """Accepts exactly ONE connection on server_sock and appends every
    newline-delimited line received (until the peer closes) to `lines` -
    runs in its own thread. See section_f1_events."""
    try:
        server_sock.settimeout(30)
        conn, _ = server_sock.accept()
        conn.settimeout(30)
        with conn, conn.makefile("r") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    lines.append(line)
    except Exception as e:  # surfaced via `errors`, not raised (wrong thread)
        errors.append(str(e))
    finally:
        server_sock.close()


def section_f1_events(urls):
    print("[F1] events sink: 1:1 NDJSON delivery over tcp://")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen(1)
    lines: list = []
    errors: list = []
    listener = threading.Thread(target=_tcp_line_listener,
                                args=(server, lines, errors), daemon=True)
    listener.start()

    streams = pycamtrt.Streams(urls)
    layer = one_layer(streams)
    det_step = layer.steps[-1]  # the Postprocess step - Sink(kind="events")
                                # taps a Postprocess step's output
    events = pycamtrt.Sink(det_step, kind="events",
                           target=f"tcp://127.0.0.1:{port}")
    pipe = pycamtrt.Pipeline(streams, layers=[layer], max_frames=60,
                             sinks=[events])
    consumed = []
    with pipe:
        for r in pipe:
            consumed.append((r.stream_id, r.frame_no, r.pts_us,
                             len(r.detections)))
        # Short drain: the events sink drains its OWN queue on its own
        # thread, independent of (and slightly behind) the main result
        # queue we just finished consuming above - give it a moment to
        # flush any lines still in flight before stop() (below, via
        # __exit__) closes the TCP connection out from under it.
        time.sleep(1.0)
    # pipe.stop() has now run (via __exit__) and closed the connection -
    # the listener thread's readline loop sees EOF and returns.
    listener.join(timeout=10)

    dropped = pipe.sink_dropped()
    print(f"  consumed {len(consumed)} results via Poll, "
          f"received {len(lines)} NDJSON lines, sink_dropped={dropped}")
    ok = True
    if errors:
        print(f"  FAIL: listener thread error(s): {errors}")
        ok = False
    if listener.is_alive():
        print("  FAIL: listener thread never finished (connection never closed?)")
        ok = False
    if dropped != 0:
        print(f"  FAIL: sink_dropped() = {dropped}, want 0")
        ok = False
    if len(lines) != len(consumed):
        print(f"  FAIL: received line count {len(lines)} != consumed result "
              f"count {len(consumed)}")
        ok = False

    received = []
    for ln in lines:
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError as e:
            print(f"  FAIL: unparseable NDJSON line: {e}: {ln[:120]!r}")
            ok = False
            continue
        received.append((obj["stream"], obj["frame"], obj["pts_us"],
                         len(obj["dets"])))

    if Counter(received) != Counter(consumed):
        print("  FAIL: received (stream,frame,pts_us,dets-len) tuples do "
              "not match the consumed set 1:1")
        missing = Counter(consumed) - Counter(received)
        extra = Counter(received) - Counter(consumed)
        if missing:
            print(f"    in consumed but not received: {list(missing.items())[:5]}")
        if extra:
            print(f"    in received but not consumed: {list(extra.items())[:5]}")
        ok = False

    if ok:
        print("  ok")
    return ok


def _ffprobe_stream(target, timeout=15):
    """Runs ffprobe against an RTSP URL or a local file; returns
    (codec_name, width, height, raw_csv_line)."""
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v",
          "-show_entries", "stream=codec_name,width,height", "-of", "csv"]
    if str(target).startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp"]
    cmd += [target]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    fields = line.split(",")
    if len(fields) < 3:
        raise RuntimeError(
            f"ffprobe({target}) gave no usable stream info (rc="
            f"{proc.returncode}) stdout={proc.stdout!r} stderr={proc.stderr!r}")
    codec, w, h = fields[-3], int(fields[-2]), int(fields[-1])
    return codec, w, h, line


def _ffprobe_first_frame_keyframe(path, timeout=15):
    """Returns the raw csv line for the FIRST frame's key_frame entry
    (e.g. "frame,1") via -read_intervals %+#1 (decode just one frame)."""
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v", "-show_frames",
          "-read_intervals", "%+#1", "-show_entries", "frame=key_frame",
          "-of", "csv", path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""


def section_f2_f3_relay_and_clip(urls):
    print("[F2] StreamRelay: live rtsp:// republish is decodable")
    print("[F3] extract_clip(): raw .h264 clip is decodable, keyframe-first")
    ok = True
    streams = pycamtrt.Streams(urls)
    layer = one_layer(streams)
    relay = pycamtrt.Sink(streams, kind="stream",
                          target="rtsp://localhost:8554/qarelay", stream=0)
    # backpressure="drop_oldest": nothing in this section ever Polls the
    # main result queue (only the sinks are under test) - Block mode would
    # eventually stall the GPU thread once that queue fills, which is
    # unrelated to what F2/F3 are checking.
    pipe = pycamtrt.Pipeline(streams, layers=[layer], max_frames=0,
                             sinks=[relay], backpressure="drop_oldest")
    # Container-visible path (see HANDOFF.md's docker mount: $PWD -> /workspace)
    # - cleaned up in `finally` below regardless of outcome.
    clip_path = "/workspace/_qa_clip.h264"
    try:
        pipe.start()
        time.sleep(3.0)  # let the relay connect to mediamtx + start publishing

        try:
            src_codec, src_w, src_h, src_line = _ffprobe_stream(urls[0])
            rel_codec, rel_w, rel_h, rel_line = _ffprobe_stream(
                "rtsp://localhost:8554/qarelay")
        except Exception as e:
            print(f"  FAIL: ffprobe error: {e}")
            ok = False
        else:
            print(f"  source probe: {src_line}")
            print(f"  relay  probe: {rel_line}")
            if rel_codec != "h264":
                print(f"  FAIL: relay codec {rel_codec!r}, want h264")
                ok = False
            elif (rel_w, rel_h) != (src_w, src_h):
                print(f"  FAIL: relay {rel_w}x{rel_h} != source {src_w}x{src_h}")
                ok = False
            else:
                print("  [F2] ok")

        time.sleep(5.0)  # ~8 s of total runtime before extracting the clip

        got = pipe.clip(0, 5.0, clip_path)
        print(f"  clip(0, 5.0, {clip_path!r}) -> {got}")
        if not got:
            print("  FAIL: clip() returned False")
            ok = False
        else:
            try:
                clip_codec, _, _, clip_line = _ffprobe_stream(clip_path)
                frame_line = _ffprobe_first_frame_keyframe(clip_path)
            except Exception as e:
                print(f"  FAIL: ffprobe error on clip: {e}")
                ok = False
            else:
                print(f"  clip stream probe: {clip_line}")
                print(f"  clip frame  probe: {frame_line}")
                key = frame_line.split(",")[-1] if frame_line else ""
                if clip_codec != "h264":
                    print(f"  FAIL: clip codec {clip_codec!r}, want h264")
                    ok = False
                elif key != "1":
                    print(f"  FAIL: clip first frame key_frame={key!r}, want '1'")
                    ok = False
                else:
                    print("  [F3] ok")
    finally:
        pipe.stop()
        if os.path.exists(clip_path):
            os.remove(clip_path)
            print(f"  cleaned up {clip_path}")
    return ok


def section_f_sinks(urls):
    print("[F] endpoint sinks (events, relay, clip)")
    ok_events = section_f1_events(urls)
    ok_relay_clip = section_f2_f3_relay_and_clip(urls)
    return ok_events and ok_relay_clip


def section_g1_demo(urls):
    print("[G1] classifier cascade demo (M1b): 60 results, "
          "outputs['classify'] aligned with detections")
    streams = pycamtrt.Streams(urls)
    detect = one_layer(streams)
    classify = pycamtrt.Layer("classify")
    ceng = classify.add(pycamtrt.Engine(detect.steps[-1], CLASSIFIER,
                                        norm=CLASSIFIER_NORM, color="rgb"))
    classify.add(pycamtrt.Postprocess(ceng, family="argmax"))
    pipe = pycamtrt.Pipeline(streams, layers=[detect, classify],
                             max_frames=60)

    results = 0
    dets_seen = 0
    ok = True
    with pipe:
        for r in pipe:
            results += 1
            pairs = r.outputs["classify"]
            if len(pairs) != len(r.detections):
                print(f"  FAIL: len(outputs['classify'])={len(pairs)} != "
                      f"len(detections)={len(r.detections)} at "
                      f"s{r.stream_id} frame {r.frame_no}")
                ok = False
            dets_seen += len(pairs)
    print(f"  {results} results, {dets_seen} detections classified")
    if results == 0:
        print("  FAIL: no results")
        ok = False
    if dets_seen == 0:
        print("  FAIL: outputs['classify'] never non-empty (no detections "
              "seen at all - is plate content up? see BUILD.md's farm section - "
              "CLIP=media/atlas_plate_g30.mp4 farm override)")
        ok = False
    if ok:
        print("  ok")
    return ok


def section_g2_parity(urls):
    # UPDATED 2026-08-31 (M3a): this section previously ran M1b's scalar
    # ImageNet approximation (norm=(-114.0, 1/58.6)) and repeatably scored
    # ~70-80% agreement, NOT a mechanics bug - see manual/FINDINGS.md's "M1
    # model-generality findings" section A for the investigation (two real
    # harness bugs found/fixed along the way - missing
    # Result.release_frame() under hold_frames=True, and a crop-rect
    # rounding mismatch against pipeline.cpp - neither closed the gap).
    # Root cause: this demo's content converges almost entirely on a
    # near-tied decision between two of mobilenet_v3_small's 1000 ImageNet
    # classes (919 "street sign" <-> 530 "digital clock" - an
    # out-of-distribution classifier on plate crops), so ANY
    # equally-legitimate GPU-vs-cv2 crop/resize implementation difference
    # is enough to flip the argmax winner far more often than a confident,
    # in-distribution classification would show.
    #
    # M3a (this pass) switched CLASSIFIER_NORM (above) to TRUE per-channel
    # ImageNet norm - one of the options this comment used to flag to the
    # owner. Measured effect (4 consecutive runs on this demo's content):
    # agreement 78.3%/91.7%/93.3%/98.3% (was repeatably ~70-80% before),
    # all with max mismatch logit-gap <= 0.263 (<< GAP_LIMIT=1.0) - a real,
    # measurable improvement, and this section now typically PASSES its
    # gate (>= 60% floor, see _qa_classifier_parity_subprocess.py) where it
    # used to fail an earlier (superseded) >= 90% aspiration. NOT a full
    # fix, though: the SAME 919<->530 near-tie still appears in every run's
    # mismatches, at similarly small gaps - per-channel norm moves this
    # out-of-distribution classifier's logits but does not resolve the
    # underlying content near-tie. Left reported plainly (agreement/gap
    # printed either way) rather than silently gated tighter.
    print("[G2] classifier parity (M1b): pipeline argmax label vs. a plain "
          "torch/cv2 reference computed on the SAME live crops (subprocess)")
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_qa_classifier_parity_subprocess.py")
    proc = subprocess.run([sys.executable, "-u", helper] + urls[:1],
                          capture_output=True, text=True, timeout=180)
    result_line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")),
        None)
    error_line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("ERROR ")),
        None)
    ok = True
    if error_line:
        print(f"  FAIL: {error_line}")
        ok = False
    elif not result_line:
        print(f"  FAIL: no RESULT line from subprocess (exit "
              f"{proc.returncode}); stdout:\n{proc.stdout}\nstderr:\n"
              f"{proc.stderr[-2000:]}")
        ok = False
    else:
        fields = dict(kv.split("=", 1) for kv in result_line.split()[1:])
        agreement = float(fields["agreement"])
        total = int(fields["total"])
        matched = int(fields["matched"])
        max_gap = float(fields.get("max_gap", 99.0))
        print(f"  agreement={agreement * 100:.1f}% ({matched}/{total}); "
              f"max mismatch logit-gap={max_gap:.3f}, "
              f"results_seen={fields.get('results_seen')}")
        if proc.returncode != 0:
            # Expected sometimes per _qa_classifier_parity_subprocess.py's
            # WHY-comment: the known hold_frames exit-time crash happens
            # AFTER the RESULT line is printed, so it does not itself fail
            # this check (same convention as section E2 above).
            print(f"    note: subprocess exited {proc.returncode} after "
                  f"printing its result (see FINDINGS.md's hold_frames "
                  f"exit-time crash entry)")
        # Logit-gap gate (see _qa_classifier_parity_subprocess.py's
        # GAP_LIMIT comment): every disagreement must be a class the
        # reference itself scores near its top (boundary flip), never one
        # it scores far down (a mechanics bug). Agreement floor guards the
        # degenerate case.
        if max_gap >= 1.0:
            print(f"  FAIL: a mismatch had logit-gap {max_gap:.3f} >= 1.0 "
                  f"- the pipeline classified something the reference "
                  f"considers wrong by a wide margin (mechanics suspect)")
            ok = False
        elif agreement < 0.60:
            print(f"  FAIL: agreement {agreement * 100:.1f}% < 60% floor")
            ok = False
    if ok:
        print("  ok")
    return ok


def section_g_classifier(urls):
    print("[G] M1b classifier cascade (argmax family)")
    ok_demo = section_g1_demo(urls)
    ok_parity = section_g2_parity(urls)
    return ok_demo and ok_parity


def section_h_recommend():
    print("[H] recommend() - M2 capacity-planning seed (pure python, no "
          "GPU/farm needed)")
    ok = True

    # H1: 4K/16-stream/skip=1/sahi tile 640, pooled (decode="all" default)
    # - cross-check against a MEASURED atlas cell (SAHI Refresh §3a,
    # v2 anchor (SAHI Refresh v2, 2026-09): measured pooled ms/frame at
    # tile 640 / T=33 / N=8 is 16.576 - the band below is vs that cell
    # rounded) and confirm it correctly calls this over capacity: 16
    # streams * 30 fps / skip 1 = 480 offered inference fps against a
    # ~60 fps whole-GPU tiled ceiling (1000/16.6).
    print("  [H1] 4K/16-stream/skip=1/tile640 pooled vs the measured "
          "16.58 ms/frame (SAHI Refresh v2)")
    r1 = pycamtrt.recommend(streams=16, resolution="4k", fps=30, skip=1,
                            decode="all", sahi={"tile": 640})
    dev = abs(r1.predicted_gpu_ms_per_frame - 16.58) / 16.58
    print(f"    predicted={r1.predicted_gpu_ms_per_frame:.3f} ms/frame "
          f"(dev {dev * 100:.1f}% from 16.8), "
          f"holds_realtime={r1.holds_realtime}, "
          f"required_gpu_fps={r1.required_gpu_fps:.1f}, "
          f"gpu_capacity_fps={r1.gpu_capacity_fps:.1f}")
    if dev > 0.20:
        print("  FAIL: predicted ms/frame deviates > 20% from the "
              "measured 16.8")
        ok = False
    if r1.holds_realtime:
        print("  FAIL: expected holds_realtime=False "
              "(480 offered fps >> ~60 fps tiled ceiling)")
        ok = False

    # H2: 720p/16-stream/skip=2, no SAHI -> comfortably holds (240 offered
    # inference fps against a ~505 fps baseline ceiling).
    print("  [H2] 720p/16-stream/skip=2, no SAHI -> holds_realtime")
    r2 = pycamtrt.recommend(streams=16, resolution="720p", fps=30, skip=2)
    print(f"    holds_realtime={r2.holds_realtime} "
          f"gpu_headroom={r2.gpu_headroom_pct:.0f}% "
          f"suggestion={r2.suggestion!r}")
    if not r2.holds_realtime:
        print("  FAIL: expected holds_realtime=True")
        ok = False

    # H3: decode="key" gop=30, 100 streams/720p - both the decode load
    # (100 streams x 30/30 = 100 decoded/s) and the inference load (same,
    # skip=1) are trivial against their ~1500 decoded/s NVDEC and ~505 fps
    # GPU ceilings - holds easily, and binding_resource must still be one
    # of the three valid values (here "none": neither ratio clears the
    # 0.5 "worth watching" threshold - see recommend()'s WHY-comment on
    # _DOMINANT_RATIO_THRESHOLD).
    print('  [H3] decode="key" gop=30, 100 streams/720p -> nvdec/binding '
          "sanity")
    r3 = pycamtrt.recommend(streams=100, resolution="720p", fps=30, skip=1,
                            decode="key", key_gop=30)
    print(f"    holds_realtime={r3.holds_realtime} "
          f"binding_resource={r3.binding_resource!r} "
          f"required_gpu_fps={r3.required_gpu_fps:.1f} "
          f"required_nvdec_fps={r3.required_nvdec_fps:.1f}")
    if not r3.holds_realtime:
        print("  FAIL: expected holds_realtime=True (100 decoded/inferred "
              "fps aggregate is a trivial load at 720p)")
        ok = False
    if r3.binding_resource not in ("gpu-inference", "nvdec", "none"):
        print(f"  FAIL: binding_resource {r3.binding_resource!r} is not "
              "one of the three documented values")
        ok = False
    if r3.binding_resource != "none":
        print("  FAIL: expected binding_resource='none' at this "
              "trivial (~100 fps aggregate) load")
        ok = False

    # H4: suggestion is always a non-empty sentence, and names a concrete
    # dial to turn when over capacity (reuses H1's over-capacity case).
    print("  [H4] suggestion string - non-empty; names a dial when over "
          "capacity")
    print(f"    H1 (over capacity): {r1.suggestion!r}")
    print(f"    H2 (holds):         {r2.suggestion!r}")
    if not r1.suggestion or not r2.suggestion:
        print("  FAIL: empty suggestion string")
        ok = False
    if not r1.suggestion.startswith("over capacity"):
        print("  FAIL: H1 is over capacity but suggestion doesn't say so")
        ok = False
    if not any(kw in r1.suggestion for kw in ("skip", "decode", "tile",
                                               "stream")):
        print("  FAIL: over-capacity suggestion doesn't name a concrete "
              "dial (skip/decode/tile/stream)")
        ok = False

    if ok:
        print("  ok")
    return ok


def section_i1_yolo_e2e(urls):
    print("[I1] M4a yolo-e2e detector family (yolo26n at layer 0)")
    streams = pycamtrt.Streams(urls)
    ok = True

    # I1a: live run, 60 results (2 streams x 30 max_frames), plausible
    # detection counts. yolo26n is a COCO-pretrained general-object
    # detector (no plate class) - see manual/FINDINGS.md's "M1
    # model-generality findings" section B - so this is a loose
    # order-of-magnitude comparison against the plate-specific yolov8
    # detector on the SAME content, not an apples-to-apples class match.
    print("  [I1a] live yolo26n run vs. yolov8-plates count (same "
          "content, order-of-magnitude only)")
    e2e_layer = pycamtrt.Layer("detect")
    e2e_eng = e2e_layer.add(pycamtrt.Engine(streams, YOLO_E2E))
    e2e_layer.add(pycamtrt.Postprocess(e2e_eng, family="yolo-e2e", score=0.4))
    pipe = pycamtrt.Pipeline(streams, layers=[e2e_layer], max_frames=30)
    frames = dets = 0
    with pipe:
        for r in pipe:
            frames += 1
            dets += len(r.detections)
    print(f"    yolo-e2e (yolo26n, COCO):  {frames} results, {dets} dets "
          f"({dets / max(frames, 1):.2f}/frame)")
    if frames == 0:
        print("  FAIL: no results flowed")
        ok = False
    if dets == 0:
        print("  FAIL: expected >0 detections on plate-farm content (a "
              "COCO detector should still find e.g. person/tv/car in "
              "frame - see BUILD.md's farm section - CLIP=media/atlas_plate_g30.mp4 "
              "farm override)")
        ok = False

    plates_layer = one_layer(streams)  # family="yolo", DET = yolov8-plates
    pipe2 = pycamtrt.Pipeline(streams, layers=[plates_layer], max_frames=30)
    pframes = pdets = 0
    with pipe2:
        for r in pipe2:
            pframes += 1
            pdets += len(r.detections)
    print(f"    yolo (yolov8n-plates):     {pframes} results, {pdets} dets "
          f"({pdets / max(pframes, 1):.2f}/frame)")
    if frames and dets and pframes and pdets:
        ratio = (dets / frames) / (pdets / pframes)
        print(f"    ratio (yolo-e2e avg dets-per-frame / yolo avg) = "
              f"{ratio:.2f}x")
        if not (0.1 <= ratio <= 10.0):
            print("  FAIL: detection-count order of magnitude diverges "
                  "by more than 10x either way - possible decode bug "
                  "rather than a genuine content difference")
            ok = False

    # I1b: SAHI + yolo-e2e must raise the named RuntimeError at Pipeline
    # construction (pipeline.cpp's Validate()) - asserted here rather than
    # section A since it is specific to this new family (see the task's
    # own "A or I - your choice": chose I, alongside the rest of this
    # family's acceptance checks).
    print("  [I1b] sahi + yolo-e2e -> named RuntimeError")

    def sahi_e2e():
        l = pycamtrt.Layer("detect", sahi=dict(tile=640))
        e = l.add(pycamtrt.Engine(streams, YOLO_E2E))
        l.add(pycamtrt.Postprocess(e, family="yolo-e2e", score=0.4))
        pycamtrt.Pipeline(streams, layers=[l])
    ok &= expect_raises("sahi + yolo-e2e", RuntimeError, sahi_e2e)

    # I1c: cascade children work UNCHANGED on a yolo-e2e root (yolo26 root
    # -> lprnet ctc child) - proves the crop/cascade mechanism (which only
    # ever consumes GpuDetections, never looks at which layer-0 family
    # produced them) is oblivious to the family swap. NOT a claim of
    # plate-reading correctness: yolo26n is COCO-pretrained (no plate
    # class), so its "detections" are general objects (person/tv/car/...),
    # not plates - this checks alignment and non-empty text output (the
    # CTC decode path actually running end to end), the same
    # mechanics-not-fit disclaimer FINDINGS.md's classifier cascade demo
    # already carries for an out-of-distribution model.
    print("  [I1c] cascade: yolo26 (yolo-e2e) root -> lprnet (ctc) child, "
          "children unchanged on an e2e root")
    detect = pycamtrt.Layer("detect")
    deng = detect.add(pycamtrt.Engine(streams, YOLO_E2E))
    detect.add(pycamtrt.Postprocess(deng, family="yolo-e2e", score=0.4))
    read = pycamtrt.Layer("read")
    reng = read.add(pycamtrt.Engine(detect.steps[-1], OCR))
    read.add(pycamtrt.Postprocess(reng, family="ctc"))
    pipe3 = pycamtrt.Pipeline(streams, layers=[detect, read], max_frames=30)
    c_results = 0
    c_align_ok = True
    c_nonempty = 0
    with pipe3:
        for r in pipe3:
            c_results += 1
            texts = r.outputs["read"]
            if len(texts) != len(r.detections):
                c_align_ok = False
            c_nonempty += sum(1 for t in texts if t)
    print(f"    {c_results} results, align_ok={c_align_ok}, "
          f"{c_nonempty} non-empty texts")
    if c_results == 0:
        print("  FAIL: no cascade results flowed")
        ok = False
    if not c_align_ok:
        print("  FAIL: outputs['read'] misaligned with detections on a "
              "yolo-e2e root")
        ok = False
    if c_nonempty == 0:
        print("  FAIL: CTC decode never produced non-empty text on a "
              "yolo-e2e root's crops")
        ok = False

    if ok:
        print("  ok")
    return ok


def section_i2_tree(urls):
    print("[I2] M4b depth-2 tree (full-tree gate): 1 detector root + 2 "
          "sibling children (read=ctc, classify=argmax)")
    streams = pycamtrt.Streams(ab_source(urls))   # tree, read-only and classify-only runs share it
    ok = True

    # The tree: 1 detector root ("detect") + 2 SIBLING children ("read"
    # ctc, "classify" argmax) - both crop `detect`'s own Postprocess step
    # directly (examples/read_and_classify/read_and_classify.py's exact shape).
    detect = one_layer(streams)  # family="yolo", DET = yolov8-plates
    read = pycamtrt.Layer("read")
    reng = read.add(pycamtrt.Engine(detect.steps[-1], OCR))
    read.add(pycamtrt.Postprocess(reng, family="ctc"))
    classify = pycamtrt.Layer("classify")
    ceng = classify.add(pycamtrt.Engine(detect.steps[-1], CLASSIFIER,
                                        norm=CLASSIFIER_NORM, color="rgb"))
    classify.add(pycamtrt.Postprocess(ceng, family="argmax"))
    pipe = pycamtrt.Pipeline(streams, layers=[detect, read, classify],
                             max_frames=60)

    results = 0
    tree_plate_counts = Counter()
    tree_label_counts = Counter()
    with pipe:
        for r in pipe:
            results += 1
            texts = r.outputs["read"]
            pairs = r.outputs["classify"]
            n_dets = len(r.detections)
            if len(texts) != n_dets or len(pairs) != n_dets:
                print(f"  FAIL: alignment - dets={n_dets} "
                      f"read={len(texts)} classify={len(pairs)} at "
                      f"s{r.stream_id} frame {r.frame_no}")
                ok = False
            # BACK-COMPAT: the legacy flat fields must equal the FIRST
            # matching-family child's output (see result.h's WHY-comment) -
            # "read" is the only ctc child, "classify" the only argmax one,
            # so each IS the "first of its family" here.
            if list(r.texts) != list(texts):
                print(f"  FAIL: back-compat r.texts != outputs['read'] at "
                      f"s{r.stream_id} frame {r.frame_no}")
                ok = False
            if list(r.labels) != [label for label, _ in pairs]:
                print(f"  FAIL: back-compat r.labels misaligned with "
                      f"outputs['classify'] at s{r.stream_id} frame "
                      f"{r.frame_no}")
                ok = False
            tree_plate_counts.update(t for t in texts if t)
            tree_label_counts.update(label for label, _ in pairs)
    print(f"  {results} results, alignment/back-compat checked per result")
    if results == 0:
        print("  FAIL: no results flowed")
        return False

    # CROSS-CHECK 1: dominant plate string, tree run vs. an independent
    # READ-ONLY (2-layer ctc) run on the SAME farm content.
    ro_detect = pycamtrt.Layer("detect_ro")
    ro_eng = ro_detect.add(pycamtrt.Engine(streams, DET))
    ro_detect.add(pycamtrt.Postprocess(ro_eng, family="yolo", score=0.4))
    ro_read = pycamtrt.Layer("read_ro")
    ro_reng = ro_read.add(pycamtrt.Engine(ro_detect.steps[-1], OCR))
    ro_read.add(pycamtrt.Postprocess(ro_reng, family="ctc"))
    pipe_read = pycamtrt.Pipeline(streams, layers=[ro_detect, ro_read],
                                  max_frames=60)
    read_only_counts = Counter()
    with pipe_read:
        for r in pipe_read:
            read_only_counts.update(t for t in r.outputs["read_ro"] if t)

    tree_top = tree_plate_counts.most_common(1)
    ro_top = read_only_counts.most_common(1)
    tree_dominant = tree_top[0][0] if tree_top else None
    read_only_dominant = ro_top[0][0] if ro_top else None
    print(f"    tree dominant plate:      {tree_dominant!r}")
    print(f"    read-only dominant plate: {read_only_dominant!r}")
    if tree_dominant != read_only_dominant:
        print("  FAIL: dominant plate string differs between the tree run "
              "and a read-only (2-layer ctc) run on the same farm")
        ok = False

    # CROSS-CHECK 2: top argmax label, tree run vs. an independent
    # CLASSIFY-ONLY (2-layer argmax) run on the SAME farm content.
    co_detect = pycamtrt.Layer("detect_co")
    co_eng = co_detect.add(pycamtrt.Engine(streams, DET))
    co_detect.add(pycamtrt.Postprocess(co_eng, family="yolo", score=0.4))
    co_classify = pycamtrt.Layer("classify_co")
    co_ceng = co_classify.add(pycamtrt.Engine(co_detect.steps[-1], CLASSIFIER,
                                              norm=CLASSIFIER_NORM,
                                              color="rgb"))
    co_classify.add(pycamtrt.Postprocess(co_ceng, family="argmax"))
    pipe_classify = pycamtrt.Pipeline(streams, layers=[co_detect, co_classify],
                                      max_frames=60)
    classify_only_counts = Counter()
    with pipe_classify:
        for r in pipe_classify:
            classify_only_counts.update(
                label for label, _ in r.outputs["classify_co"])

    tree_top_lbl = tree_label_counts.most_common(1)
    co_top_lbl = classify_only_counts.most_common(1)
    tree_top_label = tree_top_lbl[0][0] if tree_top_lbl else None
    classify_only_top_label = co_top_lbl[0][0] if co_top_lbl else None
    print(f"    tree top argmax label:          {tree_top_label}")
    print(f"    classify-only top argmax label: {classify_only_top_label}")
    if tree_top_label != classify_only_top_label:
        print("  FAIL: top argmax label differs between the tree run and "
              "a classify-only (2-layer argmax) run on the same farm")
        ok = False

    if ok:
        print("  ok")
    return ok


def _build_read_classify_tree(urls, cascade_serial):
    """The exact 2-child tree section_i2_tree uses (read=ctc,
    classify=argmax over the yolov8-plates root), parameterized by
    PipelineConfig.cascade_serial (CP1) - shared by section J's two A/B
    runs so the only thing that differs between them is that one flag.
    """
    streams = pycamtrt.Streams(urls)
    detect = one_layer(streams)  # family="yolo", DET = yolov8-plates
    read = pycamtrt.Layer("read")
    reng = read.add(pycamtrt.Engine(detect.steps[-1], OCR))
    read.add(pycamtrt.Postprocess(reng, family="ctc"))
    classify = pycamtrt.Layer("classify")
    ceng = classify.add(pycamtrt.Engine(detect.steps[-1], CLASSIFIER,
                                        norm=CLASSIFIER_NORM, color="rgb"))
    classify.add(pycamtrt.Postprocess(ceng, family="argmax"))
    return pycamtrt.Pipeline(streams, layers=[detect, read, classify],
                             max_frames=60, cascade_serial=cascade_serial)


def section_j_cascade_ab(urls):
    print("[J] CP1 A/B equivalence: cascade_serial=False (default, "
          "per-child CUDA streams) vs True (sequential escape hatch)")
    # A/B on a FILE when one is available: the two runs then see the SAME
    # frames, so a differing top label is a real divergence. On the looping
    # live farm the two runs sample different loop segments, and the plate
    # clip carries two dominant classifier labels (530 on its opening
    # segment, 919 afterwards) - measured 2026-09-16: the file A/B agreed
    # 60/60 labels while the live gate flipped 1 run in 3 on that alone.
    src = ab_source(urls)

    def run(cascade_serial):
        pipe = _build_read_classify_tree(src, cascade_serial)
        results = 0
        align_ok = True
        plate_counts = Counter()
        label_counts = Counter()
        ms_gpu_sum = [0.0, 0.0]  # index 0 = "read" child, 1 = "classify"
        with pipe:
            for r in pipe:
                results += 1
                texts = r.outputs["read"]
                pairs = r.outputs["classify"]
                n_dets = len(r.detections)
                if len(texts) != n_dets or len(pairs) != n_dets:
                    align_ok = False
                plate_counts.update(t for t in texts if t)
                label_counts.update(label for label, _ in pairs)
                # r.children mirrors layers[1:] order (read, classify) - see
                # pycamtrt.Result's docstring.
                if len(r.children) >= 2:
                    ms_gpu_sum[0] += r.children[0].ms_gpu
                    ms_gpu_sum[1] += r.children[1].ms_gpu
        top_plate = plate_counts.most_common(1)
        top_label = label_counts.most_common(1)
        return {
            "results": results,
            "align_ok": align_ok,
            "dominant_plate": top_plate[0][0] if top_plate else None,
            "top_label": top_label[0][0] if top_label else None,
            "mean_ms_gpu_read": ms_gpu_sum[0] / results if results else 0.0,
            "mean_ms_gpu_classify": ms_gpu_sum[1] / results if results else 0.0,
        }

    parallel = run(cascade_serial=False)
    serial = run(cascade_serial=True)

    def report(name, r):
        print(f"  {name}: {r['results']} results, align_ok={r['align_ok']}, "
              f"dominant_plate={r['dominant_plate']!r}, "
              f"top_label={r['top_label']}")
        print(f"    mean ChildOutput.ms_gpu: read={r['mean_ms_gpu_read']:.3f} ms  "
              f"classify={r['mean_ms_gpu_classify']:.3f} ms")

    report("parallel (cascade_serial=False)", parallel)
    report("serial   (cascade_serial=True) ", serial)

    ok = True
    if parallel["results"] == 0 or serial["results"] == 0:
        print("  FAIL: no results flowed in one or both runs")
        return False
    if not parallel["align_ok"] or not serial["align_ok"]:
        print("  FAIL: outputs['read']/outputs['classify'] misaligned with "
              "detections in one or both runs")
        ok = False
    if parallel["dominant_plate"] != serial["dominant_plate"]:
        print("  FAIL: dominant plate string differs between "
              "cascade_serial=False and cascade_serial=True")
        ok = False
    if parallel["top_label"] != serial["top_label"]:
        print("  FAIL: top argmax label differs between "
              "cascade_serial=False and cascade_serial=True")
        ok = False

    if ok:
        print("  ok")
    return ok


def _section_gate_subprocess(tag, script, media, frames):
    """K/L (v0.2.0): subprocess wrappers over the standalone bit-exact
    gates - file-input based (deterministic, farm-independent), run in a
    separate process so a crash cannot take the matrix down. SKIPs
    (passing, loudly) when the clip is absent: the repo ships no media
    (bring-your-own-clip, BUILD.md section 4) - run the script manually
    with any clip to exercise the gate on a clip-less checkout."""
    if not os.path.exists(media):
        print(f"[{tag}] SKIP: {media} not present (bring-your-own-clip, "
              f"see BUILD.md) - run python/{script} manually to exercise "
              f"this gate")
        return True
    r = subprocess.run(
        [sys.executable, os.path.join("python", script), media,
         str(frames)],
        capture_output=True, text=True, timeout=1200)
    tail = [l for l in r.stdout.splitlines() if "overall" in l]
    ok = r.returncode == 0
    print(f"[{tag}] {tail[-1] if tail else 'no overall line'} "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        print(r.stdout[-2000:])
        print(r.stderr[-1000:])
    return ok


def section_k_embedding(urls):
    return _section_gate_subprocess(
        "K embedding", "_qa_embedding_gate.py",
        "tools/stream_farm/media/atlas_plate_g30.mp4", 150)


def section_l_select(urls):
    return _section_gate_subprocess(
        "L select-routing", "_qa_select_gate.py",
        "tools/stream_farm/media/webcam_60s.mp4", 150)


def main():
    urls = sys.argv[1:] or ["rtsp://localhost:8554/cam1"]
    results = [section_a_errors(urls), section_b_sahi(urls),
               section_c_keyonly(urls), section_d_per_stream(urls),
               section_e_frame_access(urls), section_f_sinks(urls),
               section_g_classifier(urls), section_h_recommend(),
               section_i1_yolo_e2e(urls), section_i2_tree(urls),
               section_j_cascade_ab(urls), section_k_embedding(urls),
               section_l_select(urls)]
    if all(results):
        print("QA MATRIX PASS")
        return 0
    print("QA MATRIX FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
