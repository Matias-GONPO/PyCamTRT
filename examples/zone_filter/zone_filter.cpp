// zone_filter.cpp - native C++ port of zone_filter.py: the "postprocess on
// COMPACT results" consumer-side pattern, worked out on pycamtrt::Pipeline
// directly. Per-stream polygon "zones" filter an already-compact
// FrameResult's detections - no raw tensor in sight, same litmus test as
// the Python module docstring: does this code read *survivors* (this
// file) or the *raw tensor* (a compiled family instead - see
// docs/ADDING_A_FAMILY.md)? Survivors here, so plain consumer-thread code
// is the right tier - it can never slow the GPU path.
//
// Attaches "left"/"right" polygon zones per stream (pixel-space, SOURCE-
// frame coordinates), counts detections whose box CENTER lands inside each
// zone, and prints an ENTER/LEAVE line whenever a zone's occupancy changes.
// Uses backpressure=DropOldest (a live-video consumer should skip stale
// frames rather than lag behind) and reports DroppedResults() at the end.
//
// Usage:
//   zone_filter URL [URL...]
// Env override ZONE_FILTER_RESULTS (default 60) caps how many results this
// demo consumes before printing its summary and exiting.

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/pipeline.h"

using pycamtrt::Backpressure;
using pycamtrt::Detection;
using pycamtrt::Family;
using pycamtrt::FrameResult;
using pycamtrt::LayerDesc;
using pycamtrt::Pipeline;
using pycamtrt::PipelineConfig;
using pycamtrt::StepDesc;
using pycamtrt::StepKind;
using pycamtrt::StreamDesc;

namespace {

const char* const kDetectEngine = "models/yolov8n_plates_b1-16_fp16_sm86.engine";

// atlas_plate_g30.mp4 is served at 1280x720 (see tools/stream_farm) - zones
// below are plain pixel-space rectangles at that geometry, split at the
// clip's known static plate x-position (see zone_filter.py's module
// docstring "Content note"). A real deployment would size/shape zones (any
// polygon, not just rectangles) to its own camera framing.
const int kFrameW = 1280, kFrameH = 720;
const int kSplitX = 557;

using Point = std::pair<double, double>;

// Even-odd ray-casting point-in-polygon test - the entire geometry
// dependency this example needs. O(len(polygon)) per point; at "tens of
// boxes x a handful of zones" per frame this is noise next to the GPU
// frame budget.
bool PointInPolygon(double x, double y, const std::vector<Point>& polygon) {
    bool inside = false;
    const size_t n = polygon.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const auto [xi, yi] = polygon[i];
        const auto [xj, yj] = polygon[j];
        if ((yi > y) != (yj > y)) {
            const double x_cross = xi + (y - yi) * (xj - xi) / (yj - yi);
            if (x < x_cross) inside = !inside;
        }
    }
    return inside;
}

// One named polygon zone, with a running detection count and an occupancy
// flag used to print ENTER/LEAVE transitions (occupancy, not per-object
// identity - pycamtrt has no persistent track id yet).
struct Zone {
    std::string name;
    std::vector<Point> polygon;
    int count = 0;
    bool occupied = false;

    std::vector<Detection> Update(const std::vector<Detection>& detections,
                                   int stream_id, int frame_no) {
        std::vector<Detection> kept;
        for (const Detection& d : detections) {
            if (PointInPolygon(d.x + d.w / 2.0, d.y + d.h / 2.0, polygon)) kept.push_back(d);
        }
        count += (int)kept.size();
        const bool now_occupied = !kept.empty();
        if (now_occupied && !occupied) {
            printf("[s%d f%d] ZONE '%s' ENTER (%zu detection(s); running count=%d)\n",
                   stream_id, frame_no, name.c_str(), kept.size(), count);
        } else if (!now_occupied && occupied) {
            printf("[s%d f%d] ZONE '%s' LEAVE (running count=%d)\n", stream_id, frame_no,
                   name.c_str(), count);
        }
        occupied = now_occupied;
        return kept;
    }
};

// One zone SET for a single stream: "left"/"right" halves split at
// split_x. Every stream gets its OWN Zone instances (independent
// counts/occupancy), even though this demo hands every stream the same
// layout.
std::vector<Zone> MakeZones(int frame_w = kFrameW, int frame_h = kFrameH,
                             int split_x = kSplitX) {
    std::vector<Point> left = {{0, 0}, {(double)split_x, 0},
                               {(double)split_x, (double)frame_h}, {0, (double)frame_h}};
    std::vector<Point> right = {{(double)split_x, 0}, {(double)frame_w, 0},
                                {(double)frame_w, (double)frame_h},
                                {(double)split_x, (double)frame_h}};
    return {Zone{"left", left}, Zone{"right", right}};
}

PipelineConfig BuildConfig(const std::vector<std::string>& urls) {
    PipelineConfig cfg;
    cfg.streams.reserve(urls.size());
    for (const std::string& u : urls) cfg.streams.push_back(StreamDesc{u});
    cfg.skip = 1;
    cfg.max_frames = 0;
    // drop_oldest (see the module comment's CONTRACT): this consumer's own
    // print-per-frame work is cheap, but the pattern is the point - any
    // consumer-side postprocess should say this out loud.
    cfg.backpressure = Backpressure::DropOldest;
    cfg.log = [](const std::string& s) { printf("%s\n", s.c_str()); };

    StepDesc engine_step;
    engine_step.kind = StepKind::Engine;
    engine_step.input = -1;
    engine_step.engine_path = kDetectEngine;
    const int engine_idx = (int)cfg.steps.size();
    cfg.steps.push_back(engine_step);

    StepDesc post_step;
    post_step.kind = StepKind::Postprocess;
    post_step.input = engine_idx;
    post_step.family = Family::YoloDetect;
    post_step.score_thresh = 0.4f;
    const int post_idx = (int)cfg.steps.size();
    cfg.steps.push_back(post_step);

    LayerDesc detect;
    detect.name = "detect";
    detect.steps = {engine_idx, post_idx};
    cfg.layers.push_back(detect);

    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> urls(argv + 1, argv + argc);
    if (urls.empty()) {
        urls = {"rtsp://localhost:8554/cam1", "rtsp://localhost:8554/cam2"};
    }
    int n_results = 60;
    if (const char* env = std::getenv("ZONE_FILTER_RESULTS")) n_results = std::atoi(env);

    std::map<int, std::vector<Zone>> zones_by_stream;
    for (int i = 0; i < (int)urls.size(); i++) zones_by_stream[i] = MakeZones();

    std::unique_ptr<Pipeline> pipeline;
    try {
        pipeline.reset(new Pipeline(BuildConfig(urls)));
    } catch (const std::exception& e) {
        fprintf(stderr, "pipeline init failed: %s\n", e.what());
        return -1;
    }
    pipeline->Start();

    int seen = 0;
    while (seen < n_results) {
        FrameResult r;
        const Pipeline::PollStatus st = pipeline->Poll(&r, 500);
        if (st == Pipeline::PollStatus::Timeout) continue;
        if (st == Pipeline::PollStatus::Finished) break;

        seen++;
        for (Zone& zone : zones_by_stream[r.stream_id]) {
            zone.Update(r.detections, r.stream_id, r.frame_no);
        }
    }
    const uint64_t dropped = pipeline->DroppedResults();
    pipeline.reset();

    printf("\n---- zone counts ----\n");
    for (const auto& [sid, zones] : zones_by_stream) {
        for (const Zone& zone : zones) {
            printf("s%d zone '%s': %d detection(s) counted, occupied=%s\n", sid,
                   zone.name.c_str(), zone.count, zone.occupied ? "True" : "False");
        }
    }
    printf("results consumed: %d  dropped (backpressure): %llu\n", seen,
           (unsigned long long)dropped);
    return 0;
}
