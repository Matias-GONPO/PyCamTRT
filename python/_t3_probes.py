"""T3 (6h test run): model-generality gauntlet probes. Each probe is a
FINDING — success or precise rejection — never a hack. Run one probe per
process; every exception is printed VERBATIM (the named error is the data).

  embed   <engine> <src>..   resnet18 512-d embedding engine as a cascade
                             child: family="argmax" mechanical acceptance
                             (raw vector unreachable), family="ctc" rejection.
  rtdetr  <engine> <src>..   RT-DETR engine as layer 0 under family
                             "yolo-e2e" then "yolo": named error or output.
  non640  <engine> <src>..   non-640 detector as layer 0 vs kNetW/kNetH=640.
  minb    <engine> <src>..   static-batch / min_batch>1 profile engine
                             through the Python API.
  hevc    <src>..            H.265 end-to-end (COCO yolov8n layer 0).
"""
import sys
import traceback

import pycamtrt

DET = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
COCO = "models/yolov8n_b1-16_fp16_sm86.engine"
IMAGENET_NORM = ((-123.675, -116.28, -103.53),
                 (1 / 58.395, 1 / 57.12, 1 / 57.375))


def attempt(tag, fn):
    print(f"--- {tag}", flush=True)
    try:
        fn()
    except Exception:
        print(f"{tag} EXCEPTION (verbatim):", flush=True)
        traceback.print_exc()
        sys.stdout.flush()


def run(streams, layers, frames=150, sample=3):
    n = dets = 0
    samples = []
    with pycamtrt.Pipeline(streams, layers=layers, skip=1,
                           max_frames=frames) as pipe:
        for r in pipe:
            n += 1
            dets += len(r.detections)
            if r.detections and len(samples) < sample:
                d = r.detections[0]
                samples.append((round(d.x, 1), round(d.y, 1), round(d.w, 1),
                                round(d.h, 1), round(d.score, 3), d.cls))
            extra = {k: v[:2] for k, v in r.outputs.items()
                     if k != "detect" and v} if n <= 5 else {}
            if extra:
                print("  child sample:", extra, flush=True)
    print(f"  results={n} dets_total={dets} first_boxes={samples}",
          flush=True)
    return n, dets


def probe_embed(engine, srcs):
    def as_argmax():
        streams = pycamtrt.Streams(srcs)
        det = pycamtrt.Layer("detect")
        e = det.add(pycamtrt.Engine(streams, DET))
        d = det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
        emb = pycamtrt.Layer("embed")
        ee = emb.add(pycamtrt.Engine(d, engine, norm=IMAGENET_NORM,
                                     color="rgb"))
        emb.add(pycamtrt.Postprocess(ee, family="argmax"))
        run(streams, [det, emb])

    def as_ctc():
        streams = pycamtrt.Streams(srcs)
        det = pycamtrt.Layer("detect")
        e = det.add(pycamtrt.Engine(streams, DET))
        d = det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
        emb = pycamtrt.Layer("embed")
        ee = emb.add(pycamtrt.Engine(d, engine, norm=IMAGENET_NORM,
                                     color="rgb"))
        emb.add(pycamtrt.Postprocess(ee, family="ctc"))
        run(streams, [det, emb])

    attempt("EMBED/argmax (512-d head as classifier - mechanical only)",
            as_argmax)
    attempt("EMBED/ctc (should be rejected or nonsense)", as_ctc)


def probe_rtdetr(engine, srcs):
    for fam in ("rtdetr", "yolo-e2e", "yolo"):
        def go(fam=fam):
            streams = pycamtrt.Streams(srcs)
            det = pycamtrt.Layer("detect")
            e = det.add(pycamtrt.Engine(streams, engine))
            det.add(pycamtrt.Postprocess(e, family=fam, score=0.4))
            run(streams, [det])
        attempt(f"RTDETR/{fam}", go)


def probe_non640(engine, srcs):
    def go():
        streams = pycamtrt.Streams(srcs)
        det = pycamtrt.Layer("detect")
        e = det.add(pycamtrt.Engine(streams, engine))
        det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
        run(streams, [det])

    def baseline():
        streams = pycamtrt.Streams(srcs)
        det = pycamtrt.Layer("detect")
        e = det.add(pycamtrt.Engine(streams, COCO))
        det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
        run(streams, [det])

    attempt(f"NON640/{engine}", go)
    attempt("NON640/baseline-COCO-640 (same content, for comparison)",
            baseline)


def probe_minb(engine, srcs):
    def go():
        streams = pycamtrt.Streams(srcs)
        det = pycamtrt.Layer("detect")
        e = det.add(pycamtrt.Engine(streams, engine))
        det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
        run(streams, [det])
    attempt(f"MINB/{engine}", go)


def probe_hevc(srcs):
    def go():
        streams = pycamtrt.Streams(srcs)
        det = pycamtrt.Layer("detect")
        e = det.add(pycamtrt.Engine(streams, COCO))
        det.add(pycamtrt.Postprocess(e, family="yolo", score=0.4))
        n, dets = run(streams, [det], frames=300)
        print(f"HEVC verdict: results={n} dets={dets} "
              f"{'PASS' if n > 0 and dets > 0 else 'CHECK'}", flush=True)
    attempt("HEVC/e2e (cudaVideoCodec_HEVC first-ever exercise)", go)


def main():
    probe, args = sys.argv[1], sys.argv[2:]
    if probe == "hevc":
        probe_hevc(args)
    else:
        {"embed": probe_embed, "rtdetr": probe_rtdetr,
         "non640": probe_non640, "minb": probe_minb}[probe](args[0], args[1:])


if __name__ == "__main__":
    main()
