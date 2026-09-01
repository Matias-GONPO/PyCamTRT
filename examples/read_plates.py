#!/usr/bin/env python3
"""read_plates.py - a two-stage license-plate reader built on pycamtrt.

Stage 1 ("detect") runs a YOLOv8 plate detector over each stream; stage 2
("read") crops every plate detection and runs an LPRNet-style CTC OCR
engine over the crop. This is the showcase cascade example for the
pycamtrt v0 API - see python/pycamtrt/__init__.py for the full API this
is built from.

Note on output text: the bundled LPRNet engine (models/lprnet_b1-32_fp32_sm86.engine)
was trained on a Chinese license-plate charset, so decoded strings on
non-Chinese plates will look garbled (e.g. "<su>A123..."). That's expected -
this example exercises the detect -> crop -> OCR *mechanics*, not charset
fit for any particular region.

Run it (from the repo root, inside the tensorrt-dev docker container, with
the Python-dev conda env providing the interpreter):

    docker run --rm --gpus all --network host \\
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \\
        -v $(pwd):/workspace \\
        -v ~/anaconda3:/home/<user>/anaconda3 \\
        -w /workspace tensorrt-dev bash -c '
            ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
            PYTHONPATH=/workspace/build:/workspace/python \\
            ~/anaconda3/envs/Python-dev/bin/python3 \\
            examples/read_plates.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
        '

The RTSP sources come from tools/stream_farm/farm.sh. To serve plate
footage instead of the default webcam loop, override CLIP before bringing
the farm up:

    cd tools/stream_farm && CLIP=media/atlas_plate_g30.mp4 ./farm.sh up 2

Pass --verify to also run the C++ side's per-frame CPU-reference check on
the detector (pycamtrt.Pipeline(verify=True)); every consumed result's
verify_ok is then asserted True and a "PYTHON VERIFY N/N" line is printed
at the end instead of the plate summary.
"""
import sys
from collections import Counter

import pycamtrt

DETECT_ENGINE = "models/yolov8n_plates_b1-16_fp16_sm86.engine"
READ_ENGINE = "models/lprnet_b1-32_fp32_sm86.engine"

# skip=2 (infer every other decoded frame) + max_frames=600 (decoded frames
# per stream) -> ~300 inferred frames/stream, ~20s wall at 30fps source.
SKIP = 2
MAX_FRAMES = 600


def build_pipeline(urls, verify):
    streams = pycamtrt.Streams(urls)

    detect = pycamtrt.Layer("detect")
    eng = detect.add(pycamtrt.Engine(streams, DETECT_ENGINE))
    detect.add(pycamtrt.Postprocess(eng, family="yolo", score=0.4))

    read = pycamtrt.Layer("read")
    # Cross-layer edge: an Engine fed a Postprocess step auto-crops each
    # detection from the earlier layer before running the OCR engine on it.
    oeng = read.add(pycamtrt.Engine(detect.steps[-1], READ_ENGINE))
    read.add(pycamtrt.Postprocess(oeng, family="ctc"))

    return pycamtrt.Pipeline(
        streams, layers=[detect, read], skip=SKIP, max_frames=MAX_FRAMES,
        verify=verify,
    )


def main():
    args = sys.argv[1:]
    verify = "--verify" in args
    urls = [a for a in args if a != "--verify"] or ["rtsp://localhost:8554/cam1"]

    frames_seen = 0
    plates_seen = 0
    plate_counts = Counter()
    verify_ok = 0
    verify_total = 0

    with build_pipeline(urls, verify) as pipe:
        for r in pipe:
            frames_seen += 1
            texts = r.outputs["read"]
            if texts:
                plates_seen += len(texts)
                plate_counts.update(texts)
                for t in texts:
                    print(f"s{r.stream_id} frame {r.frame_no}: {t}")
            if verify:
                verify_total += 1
                if r.verify_ok:
                    verify_ok += 1

        n_streams = len(urls)
        per_stream = []
        for i in range(n_streams):
            si = pipe.get_stream_info(i)
            per_stream.append(si)

    print("\n---- summary ----")
    print(f"frames seen: {frames_seen}  plates read: {plates_seen}")
    print("distinct plate strings:")
    for text, count in plate_counts.most_common():
        print(f"  {count:5d}  {text!r}")
    for i, si in enumerate(per_stream):
        print(f"s{i}: decoded={si.decoded} reconnects={si.reconnects}")

    if verify:
        print(f"PYTHON VERIFY {verify_ok}/{verify_total}")
        if verify_ok != verify_total:
            sys.exit(1)


if __name__ == "__main__":
    main()
