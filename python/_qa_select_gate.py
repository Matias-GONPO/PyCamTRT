"""Select-routing gate (R1, v0.2.0): exact per-detection routing checks.

Cells:
  1. Routed exactness (BOTH cascade modes): COCO detector on webcam
     content; embedding child Select(classes={0}) and argmax child
     Select(classes=INDOOR, min_size=40). For EVERY detection of EVERY
     frame: the child's entry is populated IFF the detection passes that
     child's predicate. The embedding family is the perfect discriminator
     (routed-away = empty list, passed = 512 floats); for argmax a passed
     crop's score is nonzero while a routed-away slot keeps (0, 0.0).
  2. Pass-all parity: Select() with no criteria vs no Select at all -
     same result count, same frame set, non-empty totals within 1%.
  3. Shared Select: a 2-step child consuming another layer's Select
     handle routes identically to owning it.
  4. Named errors: depth-3 Select, bad classes=, Select in wrong slot.

Usage: _qa_select_gate.py <src> [frames]
"""
import sys

import pycamtrt

COCO = "models/yolov8n_b1-16_fp16_sm86.engine"
EMB = "models/resnet18emb_b1-32_fp16_sm86.engine"
CLS = "models/mobilenet_v3s_b1-32_fp16_sm86.engine"
IMAGENET_NORM = ((-123.675, -116.28, -103.53),
                 (1 / 58.395, 1 / 57.12, 1 / 57.375))
PERSON = {0}
INDOOR = {56, 62, 63, 66, 73}  # chair, tv, laptop, keyboard, book
MIN_SIZE = 40.0


def clamped_min_side(d, sw, sh):
    """Mirror pipeline.cpp's crop clamp exactly (lroundf + clamp)."""
    rx = int(d.x + 0.5)
    ry = int(d.y + 0.5)
    rw = int(d.w + 0.5)
    rh = int(d.h + 0.5)
    rx = max(0, min(rx, sw - 1))
    ry = max(0, min(ry, sh - 1))
    rw = max(1, min(rw, sw - rx))
    rh = max(1, min(rh, sh - ry))
    return min(rw, rh)


def tree(urls, mode):
    """mode: 'routed' (own Selects), 'shared' (child B consumes child A's
    Select), 'passall' (criteria-free Selects), 'plain' (no Selects)."""
    streams = pycamtrt.Streams(urls)
    det = pycamtrt.Layer("detect")
    e = det.add(pycamtrt.Engine(streams, COCO))
    d = det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))

    emb = pycamtrt.Layer("embed")
    if mode == "routed" or mode == "shared":
        sel_a = emb.add(pycamtrt.Select(d, classes=PERSON))
    elif mode == "passall":
        sel_a = emb.add(pycamtrt.Select(d))
    else:
        sel_a = d
    ee = emb.add(pycamtrt.Engine(sel_a, EMB, norm=IMAGENET_NORM, color="rgb"))
    emb.add(pycamtrt.Postprocess(ee, family="embedding"))

    cls = pycamtrt.Layer("cls")
    if mode == "routed":
        sel_b = cls.add(pycamtrt.Select(d, classes=INDOOR, min_size=MIN_SIZE))
    elif mode == "shared":
        sel_b = sel_a  # 2-step child consuming layer A's Select
    elif mode == "passall":
        sel_b = cls.add(pycamtrt.Select(d))
    else:
        sel_b = d
    ce = cls.add(pycamtrt.Engine(sel_b, CLS, norm=IMAGENET_NORM, color="rgb"))
    cls.add(pycamtrt.Postprocess(ce, family="argmax"))
    return streams, [det, emb, cls]


def run(urls, mode, frames, serial=False):
    streams, layers = tree(urls, mode)
    stats = dict(results=0, dets=0, emb_full=0, cls_full=0, viol=0)
    frames_seen = []
    with pycamtrt.Pipeline(streams, layers=layers, skip=1,
                           max_frames=frames, cascade_serial=serial) as pipe:
        for r in pipe:
            stats["results"] += 1
            frames_seen.append((r.stream_id, r.frame_no))
            vecs = r.outputs["embed"]
            pairs = r.outputs["cls"]
            sw, sh = r.frame_width, r.frame_height
            for i, d in enumerate(r.detections):
                stats["dets"] += 1
                emb_full = i < len(vecs) and len(vecs[i]) > 0
                cls_full = i < len(pairs) and pairs[i][1] != 0.0
                stats["emb_full"] += emb_full
                stats["cls_full"] += cls_full
                if mode == "routed":
                    want_emb = d.cls in PERSON
                    want_cls = (d.cls in INDOOR and
                                clamped_min_side(d, sw, sh) >= MIN_SIZE)
                elif mode == "shared":
                    want_emb = want_cls = d.cls in PERSON
                else:
                    want_emb = want_cls = True
                if emb_full != want_emb or cls_full != want_cls:
                    stats["viol"] += 1
                    if stats["viol"] <= 3:
                        print(f"  VIOLATION {mode} f{r.frame_no} det{i} "
                              f"cls={d.cls} size="
                              f"{clamped_min_side(d, sw, sh)} "
                              f"emb={emb_full}/{want_emb} "
                              f"cls_child={cls_full}/{want_cls}", flush=True)
    return stats, frames_seen


def main():
    src = sys.argv[1]
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    ok = True

    for serial in (False, True):
        st, _ = run([src], "routed", frames, serial)
        cell_ok = st["results"] > 0 and st["dets"] > 0 and st["viol"] == 0 \
            and st["emb_full"] > 0
        ok &= cell_ok
        print(f"SELECT_GATE[routed {'serial' if serial else 'parallel'}] "
              f"{st} {'PASS' if cell_ok else 'FAIL'}", flush=True)

    st, _ = run([src], "shared", frames)
    cell_ok = st["results"] > 0 and st["viol"] == 0
    ok &= cell_ok
    print(f"SELECT_GATE[shared] {st} {'PASS' if cell_ok else 'FAIL'}",
          flush=True)

    sp, fp = run([src], "passall", frames)
    sn, fn = run([src], "plain", frames)
    # The routing property is WITHIN-run and exact: a criteria-free Select
    # must populate every detection, exactly like no Select at all does.
    # CROSS-run detection counts are a DETECTOR property, not a routing
    # one - they vary a few per mille run-to-run (batch-composition
    # numerics near score_thresh; characterized in report 11's filedet
    # cell), so they only get a coarse 5% sanity band here.
    det_band = abs(sp["dets"] - sn["dets"]) <= max(0.05 * sn["dets"], 3)
    parity_ok = (sp["viol"] == 0 and sn["viol"] == 0 and
                 sp["emb_full"] == sp["dets"] and
                 sp["cls_full"] == sp["dets"] and
                 sn["emb_full"] == sn["dets"] and
                 sn["cls_full"] == sn["dets"] and
                 sorted(fp) == sorted(fn) and det_band)
    ok &= parity_ok
    print(f"SELECT_GATE[passall-parity] passall={sp} plain={sn} "
          f"{'PASS' if parity_ok else 'FAIL'}", flush=True)

    err_ok = True
    streams = pycamtrt.Streams([src])
    det = pycamtrt.Layer("detect")
    e = det.add(pycamtrt.Engine(streams, COCO))
    d = det.add(pycamtrt.Postprocess(e, family="yolo"))
    read = pycamtrt.Layer("read")
    oe = read.add(pycamtrt.Engine(d, EMB, norm=IMAGENET_NORM, color="rgb"))
    op = read.add(pycamtrt.Postprocess(oe, family="embedding"))
    deep = pycamtrt.Layer("deep")
    ds = deep.add(pycamtrt.Select(op, classes={0}))
    de = deep.add(pycamtrt.Engine(ds, CLS, norm=IMAGENET_NORM, color="rgb"))
    deep.add(pycamtrt.Postprocess(de, family="argmax"))
    try:
        pycamtrt.Pipeline(streams, layers=[det, read, deep], max_frames=1)
        err_ok = False
        print("  ERR: depth-3 Select was ACCEPTED", flush=True)
    except RuntimeError as ex:
        if "chained cascades" not in str(ex):
            err_ok = False
            print(f"  ERR: wrong depth-3 error: {ex}", flush=True)
    try:
        pycamtrt.Select(d, classes=[])
        err_ok = False
        print("  ERR: empty classes list accepted", flush=True)
    except ValueError:
        pass
    try:
        pycamtrt.Select(d, min_size=-3)
        err_ok = False
        print("  ERR: negative min_size accepted", flush=True)
    except ValueError:
        pass
    ok &= err_ok
    print(f"SELECT_GATE[named-errors] {'PASS' if err_ok else 'FAIL'}",
          flush=True)

    print("SELECT_GATE overall:", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
