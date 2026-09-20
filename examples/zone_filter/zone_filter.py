#!/usr/bin/env python3
"""zone_filter.py - a worked example of the "PyPostprocess" consumer-side
pattern: per-stream polygon "zones" filtering an already-COMPACT pycamtrt
result, no raw tensor in sight.

CONTRACT (read this before copying the pattern elsewhere - see the package
docstring's "Custom postprocessing - two tiers" section, and
docs/ADDING_A_FAMILY.md for the OTHER tier):
  - This code runs on the CONSUMER THREAD, i.e. inside your `for r in
    pipe:` loop, on the COMPACT per-frame result the C++ data plane has
    already produced - tens of boxes at most, never the raw
    [4+classes,anchors] output tensor. It therefore can NEVER slow the GPU
    path: by the time this code runs, the GPU thread has already moved on
    to the next batch. Nothing here is on that critical path.
  - If YOUR consumer logic is slow enough to fall behind the GPU thread's
    result rate, don't let it silently pile up latency: build the
    ``Pipeline`` with ``backpressure="drop_oldest"`` and periodically read
    ``pipe.dropped_results()`` (see the package docstring) - a live-video
    consumer should skip stale frames rather than lag behind. This script
    does exactly that below.
  - When NOT to do this: anything that reads the RAW OUTPUT TENSOR (a
    custom decode of a model's raw head - per-anchor/per-pixel work, e.g.
    a segmentation mask or a bespoke box-decode) belongs in a COMPILED GPU
    sibling-launcher family instead (see docs/ADDING_A_FAMILY.md), never
    a Python callback on the per-frame path. The litmus test: does the
    code read *survivors* (a short compact list, this file) or the *raw
    tensor* (megabytes, every anchor, every frame)? Survivors -> Python is
    fine, here. Raw tensor -> compiled family, always - the API makes the
    wrong thing merely inconvenient (no raw-tensor hook exists to misuse).

What this script does: attaches one or more named polygon "zones" per
stream (pixel-space, SOURCE-frame coordinates - the same coordinate system
as ``Detection.x/y/w/h``, i.e. independent of whatever letterbox the model
used internally). Each polled ``Result``'s detections are filtered to
those whose BOX CENTER lies inside a zone (point-in-polygon, plain
even-odd ray casting, ~20 lines - shapely is NOT required, zero new
deps). Per-zone counts accumulate, and an ENTER/LEAVE line prints whenever
a zone's OCCUPANCY (any detection center inside it, this frame) changes.

Why occupancy and not per-object identity: pycamtrt has no persistent
track ID yet (the IOU tracker is a backlog item - see HANDOFF.md's open
items), so "entering/leaving" here is defined at the ZONE level (was
anything inside last frame vs. now), not per-tracked-object. Once a
tracker lands, a per-track-id version of this same filter is a small
extension of this file, not a redesign - the zone/point-in-polygon/
counting parts are unaffected.

Content note (measured 2026-08-31 against the live farm, 900 sampled
frames on both cams): tools/stream_farm's atlas_plate_g30.mp4 plate
detection is essentially STATIC - its box center jitters inside a
[645.6, 646.1]px band the entire clip (a stationary vehicle, not a moving
one). No single zone polygon is ever CROSSED by real motion in this clip.
Per this example's own design brief, the fallback is two zones split at
x=557 so their COUNTS DIFFER (the plate sits permanently right of the
split); the ENTER transition still fires honestly exactly once per
stream - at the first result carrying a detection, since a freshly
attached Zone's occupancy starts False - see ZONE_ENTER below. Point
``make_zones()`` at your own geometry for content that actually moves.

Run it (from the repo root, inside the tensorrt-dev docker container - see
examples/read_plates/read_plates.py's docstring for the full canonical incantation):

    docker run --rm --gpus all --network host \\
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \\
        -v $(pwd):/workspace \\
        -v ~/anaconda3:/home/<user>/anaconda3 \\
        -w /workspace tensorrt-dev bash -c '
            ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
            PYTHONPATH=/workspace/build:/workspace/python \\
            python3 \\
            examples/zone_filter/zone_filter.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
        '

The RTSP sources come from tools/stream_farm/farm.sh; plate-shaped content
(so there is anything to zone-filter) needs:

    cd tools/stream_farm && CLIP=media/atlas_plate_g30.mp4 ./farm.sh up 2

Env override ``ZONE_FILTER_RESULTS`` (default 60) caps how many results
this demo consumes before printing its summary and exiting.
"""
import os
import sys

import pycamtrt

DETECT_ENGINE = "models/yolov8n_plates_b1-16_fp16_sm86.engine"

# atlas_plate_g30.mp4 is served at 1280x720 (see tools/stream_farm) - zones
# below are plain pixel-space rectangles at that geometry, split at the
# clip's known static plate x-position (see the module docstring's Content
# note). A real deployment would size/shape zones (any polygon, not just
# rectangles - see point_in_polygon) to its own camera framing.
FRAME_W, FRAME_H = 1280, 720
SPLIT_X = 557


def point_in_polygon(x, y, polygon):
    """Even-odd ray-casting point-in-polygon test. `polygon` is a list of
    (x, y) vertices, any winding order, not necessarily convex. This is
    the entire geometry dependency this example needs - no shapely, no new
    package. O(len(polygon)) per point; at "tens of boxes x a handful of
    zones" per frame this is noise next to the GPU frame budget (the
    CONTRACT above's whole point).
    """
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


class Zone:
    """One named polygon zone, with a running detection count and an
    occupancy flag used to print ENTER/LEAVE transitions (see the module
    docstring for why occupancy, not per-object identity).
    """

    def __init__(self, name, polygon):
        self.name = name
        self.polygon = polygon
        self.count = 0       # cumulative detections ever counted here
        self.occupied = False

    def update(self, detections, stream_id, frame_no):
        """Filter `detections` (a Result.outputs["detect"]-shaped list) to
        those whose box center lands inside this zone, update the running
        count, and print an ENTER/LEAVE line on an occupancy edge.
        Returns the filtered (kept) list, in case a caller wants it.
        """
        kept = [d for d in detections
                if point_in_polygon(d.x + d.w / 2.0, d.y + d.h / 2.0,
                                     self.polygon)]
        self.count += len(kept)
        now_occupied = bool(kept)
        if now_occupied and not self.occupied:
            print(f"[s{stream_id} f{frame_no}] ZONE '{self.name}' ENTER "
                  f"({len(kept)} detection(s); running count={self.count})")
        elif not now_occupied and self.occupied:
            print(f"[s{stream_id} f{frame_no}] ZONE '{self.name}' LEAVE "
                  f"(running count={self.count})")
        self.occupied = now_occupied
        return kept


def make_zones(frame_w=FRAME_W, frame_h=FRAME_H, split_x=SPLIT_X):
    """One zone SET for a single stream: "left" and "right" halves split
    at `split_x`. Task A's pattern is "a zone per stream" - each stream
    below gets its OWN Zone instances (independent counts/occupancy), even
    though this demo hands every stream the same layout.
    """
    left = [(0, 0), (split_x, 0), (split_x, frame_h), (0, frame_h)]
    right = [(split_x, 0), (frame_w, 0), (frame_w, frame_h), (split_x, frame_h)]
    return [Zone("left", left), Zone("right", right)]


def build_pipeline(urls):
    streams = pycamtrt.Streams(urls)
    detect = pycamtrt.Layer("detect")
    eng = detect.add(pycamtrt.Engine(streams, DETECT_ENGINE))
    detect.add(pycamtrt.Postprocess(eng, family="yolo", score=0.4))
    # drop_oldest (see the module docstring's CONTRACT): this consumer's
    # own print-per-frame work is cheap, but the pattern is the point -
    # any consumer-side postprocess should say this out loud.
    return pycamtrt.Pipeline(streams, layers=[detect], skip=1, max_frames=0,
                             backpressure="drop_oldest")


def main():
    urls = (sys.argv[1:] or
            ["rtsp://localhost:8554/cam1", "rtsp://localhost:8554/cam2"])
    n_results = int(os.environ.get("ZONE_FILTER_RESULTS", "60"))

    zones_by_stream = {i: make_zones() for i in range(len(urls))}

    seen = 0
    with build_pipeline(urls) as pipe:
        for r in pipe:
            seen += 1
            dets = r.outputs["detect"]
            for zone in zones_by_stream[r.stream_id]:
                zone.update(dets, r.stream_id, r.frame_no)
            if seen >= n_results:
                break
        dropped = pipe.dropped_results()

    print("\n---- zone counts ----")
    for sid in sorted(zones_by_stream):
        for zone in zones_by_stream[sid]:
            print(f"s{sid} zone {zone.name!r}: {zone.count} detection(s) "
                  f"counted, occupied={zone.occupied}")
    print(f"results consumed: {seen}  dropped (backpressure): {dropped}")


if __name__ == "__main__":
    main()
