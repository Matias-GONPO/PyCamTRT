#!/usr/bin/env python3
"""route_and_reid.py - class-routed cascade + cross-camera re-ID built on
pycamtrt. THE showcase for detection routing (``Select``) and the
``embedding`` family together.

Stage 1 ("detect") runs a COCO YOLOv8 detector over every stream. Its
detections are then ROUTED by class to two specialist siblings:

  - ``Select(classes={0})`` (person) -> stage "person": a resnet18
    penultimate-feature engine (``family="embedding"``) turns each person
    crop into a 512-d vector, and this script cosine-matches vectors
    ACROSS streams - "the person on cam1 is the person on cam2".
  - ``Select(classes={2,3,5,7}, min_size=48)`` (car/motorcycle/bus/truck,
    big enough to classify) -> stage "vehicle": a mobilenet ImageNet
    classifier (``family="argmax"``) labels the vehicle type.

Routing is exact and strictly reduces work: each sibling only ever
infers the crops its Select passes; every other detection keeps that
sibling's empty entry (``outputs[layer][i]`` stays aligned with
``detections[i]`` - one indexing rule, routed or not). See the Select
docstring in python/pycamtrt/__init__.py for the full contract.

Honest caveats: resnet18-on-ImageNet is a MECHANICS demo for re-ID - it
was never trained for person re-identification, so treat matches as
illustrative (a real deployment would export a re-ID checkpoint, e.g.
OSNet, through the same ``embedding`` family). Likewise the vehicle
labels are ImageNet classes (pickup, minivan, sports car, ...), a loose
stand-in for a purpose-trained vehicle classifier. And the demo farm clip
is an indoor webcam scene: expect person matches and no vehicles there -
point it at real street streams to see both branches light up.

Run it (from the repo root; full canonical docker incantation in
examples/read_plates/read_plates.py's docstring):

    PYTHONPATH=/workspace/build:/workspace/python \\
    python3 \\
    examples/route_and_reid/route_and_reid.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2

No cameras? tools/stream_farm/farm.sh serves a looping clip as N RTSP
streams (cd tools/stream_farm && ./farm.sh up 2).
"""
import sys
from collections import Counter, deque

import numpy as np

import pycamtrt

DETECT_ENGINE = "models/yolov8n_b1-16_fp16_sm86.engine"      # COCO, 80 cls
EMBED_ENGINE = "models/resnet18emb_b1-32_fp16_sm86.engine"   # [N,512] head
VEHICLE_ENGINE = "models/mobilenet_v3s_b1-32_fp16_sm86.engine"

IMAGENET_NORM = ((-123.675, -116.28, -103.53),
                 (1 / 58.395, 1 / 57.12, 1 / 57.375))

PERSON_CLASSES = {0}            # COCO ids - YOUR detector's ids, not ours
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
VEHICLE_MIN_SIZE = 48           # skip crops too small to classify

# Cosine similarity above which two person vectors count as "same person"
# across cameras. Tune per model/content; 0.85 is a demo value.
MATCH_THRESHOLD = 0.85
GALLERY_PER_STREAM = 50         # recent vectors kept per stream

# A few well-known ImageNet-1k vehicle classes, for readable labels; any
# other label id prints as "imagenet_<id>".
VEHICLE_LABELS = {407: "ambulance", 468: "cab", 511: "convertible",
                  555: "fire_engine", 569: "garbage_truck", 654: "minibus",
                  656: "minivan", 675: "moving_van", 717: "pickup",
                  734: "police_van", 751: "racer", 779: "school_bus",
                  817: "sports_car", 864: "tow_truck", 867: "trailer_truck",
                  874: "trolleybus"}

SKIP = 2
MAX_FRAMES = 600


def build_pipeline(urls):
    streams = pycamtrt.Streams(urls)

    detect = pycamtrt.Layer("detect")
    eng = detect.add(pycamtrt.Engine(streams, DETECT_ENGINE))
    det = detect.add(pycamtrt.Postprocess(eng, family="yolo", score=0.4))

    person = pycamtrt.Layer("person")
    # The routing node: only class-0 (person) crops reach this engine.
    psel = person.add(pycamtrt.Select(det, classes=PERSON_CLASSES))
    peng = person.add(pycamtrt.Engine(psel, EMBED_ENGINE,
                                      norm=IMAGENET_NORM, color="rgb"))
    person.add(pycamtrt.Postprocess(peng, family="embedding"))

    vehicle = pycamtrt.Layer("vehicle")
    vsel = vehicle.add(pycamtrt.Select(det, classes=VEHICLE_CLASSES,
                                       min_size=VEHICLE_MIN_SIZE))
    veng = vehicle.add(pycamtrt.Engine(vsel, VEHICLE_ENGINE,
                                       norm=IMAGENET_NORM, color="rgb"))
    vehicle.add(pycamtrt.Postprocess(veng, family="argmax"))

    return pycamtrt.Pipeline(streams, layers=[detect, person, vehicle],
                             skip=SKIP, max_frames=MAX_FRAMES)


def main():
    urls = sys.argv[1:] or ["rtsp://localhost:8554/cam1"]

    galleries = {i: deque(maxlen=GALLERY_PER_STREAM)
                 for i in range(len(urls))}
    frames = persons = vehicles = matches = 0
    vehicle_counts = Counter()
    best_match = (0.0, None, None)

    with build_pipeline(urls) as pipe:
        for r in pipe:
            frames += 1
            for i, det in enumerate(r.detections):
                vec = r.outputs["person"][i] if i < len(
                    r.outputs["person"]) else []
                if vec:  # a routed person crop - 512 floats
                    persons += 1
                    v = np.asarray(vec, dtype=np.float32)
                    v /= np.linalg.norm(v) + 1e-9
                    # Cosine-match against OTHER streams' recent persons.
                    for sid, gal in galleries.items():
                        if sid == r.stream_id or not gal:
                            continue
                        sims = np.stack(gal) @ v
                        top = float(sims.max())
                        if top >= MATCH_THRESHOLD:
                            matches += 1
                            if top > best_match[0]:
                                best_match = (top, r.stream_id, sid)
                            if matches <= 10:
                                print(f"MATCH person s{r.stream_id} frame "
                                      f"{r.frame_no} ~ s{sid} "
                                      f"(cos {top:.3f})")
                    galleries[r.stream_id].append(v)
                pair = r.outputs["vehicle"][i] if i < len(
                    r.outputs["vehicle"]) else None
                if pair is not None and pair[1] != 0.0:  # routed vehicle
                    vehicles += 1
                    label = VEHICLE_LABELS.get(pair[0],
                                               f"imagenet_{pair[0]}")
                    vehicle_counts[label] += 1

        per_stream = [pipe.get_stream_info(i) for i in range(len(urls))]

    print("\n---- summary ----")
    print(f"frames: {frames}  person crops embedded: {persons}  "
          f"vehicle crops classified: {vehicles}")
    print(f"cross-camera person matches (cos >= {MATCH_THRESHOLD}): "
          f"{matches}")
    if best_match[1] is not None:
        print(f"best match: s{best_match[1]} ~ s{best_match[2]} "
              f"(cos {best_match[0]:.3f})")
    if vehicle_counts:
        print("vehicle types:")
        for label, count in vehicle_counts.most_common():
            print(f"  {count:5d}  {label}")
    for i, si in enumerate(per_stream):
        print(f"s{i}: decoded={si.decoded} reconnects={si.reconnects}")


if __name__ == "__main__":
    main()
