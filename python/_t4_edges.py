"""T4 (6h test run): cascade & edge-case matrix. One subcommand per cell,
run in separate processes so a crash in one never hides the others.

  zerodet  <url|file>..  plate detector on plate-free content, 2 children,
                         events sink: empty-dets path end to end.
  flood    <url>..       score=0.01 detection flood: behavior at the
                         kMaxNmsCandidates=1024 clamp, stability.
  trees    <url>..       4 sibling children (ctc + 3 argmax, 2 of them the
                         SAME engine file) under a SAHI root, events+relay
                         sinks attached: results/sinks for every child.
  reschange <url>..      long-run; writes _t4_res_phaseA.marker after 200
                         results, then waits for the host to swap the farm
                         to a different-resolution clip; passes when 100
                         results arrive at the new dimensions.
  filedet  <path>        determinism: same file through the pipeline twice,
                         strict (full-float) and tolerant (3-decimal)
                         per-frame compares.
"""
import os
import sys
import threading
import time
from collections import Counter

import pycamtrt

DET = os.environ.get("T4_DET", "models/yolov8n_plates_b1-16_fp16_sm86.engine")
OCR = "models/lprnet_b1-32_fp32_sm86.engine"
CLS = "models/mobilenet_v3s_b1-32_fp16_sm86.engine"
IMAGENET_NORM = ((-123.675, -116.28, -103.53),
                 (1 / 58.395, 1 / 57.12, 1 / 57.375))


def tree(streams, n_children, score=0.4, sahi=None):
    det = pycamtrt.Layer("detect", sahi=sahi)
    e = det.add(pycamtrt.Engine(streams, DET))
    d = det.add(pycamtrt.Postprocess(e, family="yolo", score=score))
    layers = [det]
    if n_children >= 1:
        read = pycamtrt.Layer("read")
        oe = read.add(pycamtrt.Engine(d, OCR))
        read.add(pycamtrt.Postprocess(oe, family="ctc"))
        layers.append(read)
    for k in range(2, n_children + 1):
        cl = pycamtrt.Layer(f"classify{k - 1}")
        ce = cl.add(pycamtrt.Engine(d, CLS, norm=IMAGENET_NORM, color="rgb"))
        cl.add(pycamtrt.Postprocess(ce, family="argmax"))
        layers.append(cl)
    return layers, d


def cell_zerodet(urls):
    streams = pycamtrt.Streams(urls)
    layers, d = tree(streams, 2)
    sink = pycamtrt.Sink(d, kind="events",
                         target="file:///workspace/python/_t4_zerodet.ndjson")
    n = dets = texts = 0
    with pycamtrt.Pipeline(streams, layers=layers, skip=1, max_frames=300,
                           sinks=[sink]) as pipe:
        for r in pipe:
            n += 1
            dets += len(r.detections)
            texts += len(r.outputs["read"])
        sd = pipe.sink_dropped()
    lines = sum(1 for _ in open("/workspace/python/_t4_zerodet.ndjson"))
    print(f"ZERODET results={n} dets_total={dets} texts_total={texts} "
          f"sink_lines={lines} sink_dropped={sd}", flush=True)
    ok = n > 0 and dets == 0 and texts == 0 and lines > 0
    print("ZERODET", "PASS" if ok else "CHECK", flush=True)


def cell_flood(urls):
    logs = Counter()

    def log(msg):
        for key in ("clamp", "cand", "warn", "error", "drop"):
            if key in msg.lower():
                logs[key] += 1

    streams = pycamtrt.Streams(urls)
    layers, d = tree(streams, 1, score=0.01)
    per_frame = []
    with pycamtrt.Pipeline(streams, layers=layers, skip=1, max_frames=300,
                           log=log) as pipe:
        for r in pipe:
            per_frame.append(len(r.detections))
    c = Counter(per_frame)
    print(f"FLOOD results={len(per_frame)} max_dets={max(per_frame)} "
          f"mean_dets={sum(per_frame)/max(len(per_frame),1):.1f} "
          f"top_counts={c.most_common(5)} logs={dict(logs)}", flush=True)
    print("FLOOD", "PASS" if per_frame else "CHECK", flush=True)


def cell_trees(urls):
    streams = pycamtrt.Streams(urls)
    layers, d = tree(streams, 4, sahi={"tile": 640})
    child_pp = layers[1].steps[-1]  # the ctc child's Postprocess
    sinks = [
        pycamtrt.Sink(d, kind="events",
                      target="file:///workspace/python/_t4_trees_det.ndjson"),
        pycamtrt.Sink(child_pp, kind="events",
                      target="file:///workspace/python/_t4_trees_child.ndjson"),
        pycamtrt.Sink(streams, kind="stream", stream=0,
                      target="rtsp://localhost:8554/t4relay"),
    ]
    n = 0
    have = Counter()
    kid_ms = Counter()
    with pycamtrt.Pipeline(streams, layers=layers, skip=1, max_frames=200,
                           sinks=sinks) as pipe:
        for r in pipe:
            n += 1
            for name in ("read", "classify1", "classify2", "classify3"):
                if name in r.outputs:
                    have[name] += 1
            for i, c in enumerate(r.children):
                kid_ms[i] += c.ms_gpu
        sd, dr = pipe.sink_dropped(), pipe.dropped_results()
    l1 = sum(1 for _ in open("/workspace/python/_t4_trees_det.ndjson"))
    l2 = sum(1 for _ in open("/workspace/python/_t4_trees_child.ndjson"))
    print(f"TREES results={n} outputs_present={dict(have)} "
          f"det_sink_lines={l1} child_sink_lines={l2} sink_dropped={sd} "
          f"dropped_results={dr}", flush=True)
    ok = n > 0 and all(have[k] == n for k in
                       ("read", "classify1", "classify2", "classify3"))
    print("TREES", "PASS" if ok else "CHECK", flush=True)


def cell_reschange(urls):
    logs = Counter()

    def log(msg):
        m = msg.lower()
        if "reconnect" in m:
            logs["reconnect"] += 1
        if "error" in m:
            logs["error"] += 1

    streams = pycamtrt.Streams(urls)
    layers, d = tree(streams, 1)
    pipe = pycamtrt.Pipeline(streams, layers=layers, skip=1, max_frames=0,
                             log=log)
    deadline = threading.Timer(420.0, pipe.stop)
    dims_a = dims_b = None
    n_a = n_b = 0
    try:
        for r in pipe:
            wh = (r.frame_width, r.frame_height)
            if dims_a is None:
                dims_a = wh
            if wh == dims_a and dims_b is None:
                n_a += 1
                if n_a == 200:
                    open("/workspace/python/_t4_res_phaseA.marker", "w").close()
                    print(f"PHASE_A dims={dims_a} results={n_a}", flush=True)
                    deadline.start()
            elif wh != dims_a:
                if dims_b is None:
                    dims_b = wh
                    print(f"PHASE_B first new-res result dims={dims_b}",
                          flush=True)
                if wh == dims_b:
                    n_b += 1
                    if n_b >= 100:
                        break
    finally:
        deadline.cancel()
        pipe.stop()
    print(f"RESCHANGE dims_a={dims_a} n_a={n_a} dims_b={dims_b} n_b={n_b} "
          f"logs={dict(logs)}", flush=True)
    ok = dims_b is not None and dims_b != dims_a and n_b >= 100
    print("RESCHANGE", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 3)


def _file_run(path):
    streams = pycamtrt.Streams([path])
    layers, d = tree(streams, 2)
    strict, tol = [], []
    with pycamtrt.Pipeline(streams, layers=layers, skip=1,
                           max_frames=300) as pipe:
        for r in pipe:
            ds = sorted((d.x, d.y, d.w, d.h, d.score, d.cls)
                        for d in r.detections)
            key = (r.stream_id, r.frame_no)
            strict.append((key, repr(ds), tuple(r.outputs["read"]),
                           tuple(r.outputs["classify1"])))
            tol.append((key,
                        repr([tuple(round(v, 3) for v in t) for t in ds]),
                        tuple(r.outputs["read"]),
                        tuple(l for l, s in r.outputs["classify1"])))
    return sorted(strict), sorted(tol)


def cell_filedet(path):
    s1, t1 = _file_run(path)
    s2, t2 = _file_run(path)
    frames_match = [a[0] for a in s1] == [b[0] for b in s2]
    strict_eq = s1 == s2
    tol_eq = t1 == t2
    diffs = sum(1 for a, b in zip(t1, t2) if a != b)
    print(f"FILEDET run1={len(s1)} run2={len(s2)} frames_match={frames_match} "
          f"strict_identical={strict_eq} tolerant_identical={tol_eq} "
          f"tolerant_diff_frames={diffs}", flush=True)
    print("FILEDET", "PASS" if frames_match and tol_eq else
          ("PARTIAL" if frames_match else "FAIL"), flush=True)


def main():
    cell, args = sys.argv[1], sys.argv[2:]
    if cell == "filedet":
        cell_filedet(args[0])
    else:
        {"zerodet": cell_zerodet, "flood": cell_flood, "trees": cell_trees,
         "reschange": cell_reschange}[cell](args)


if __name__ == "__main__":
    main()
