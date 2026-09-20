// route_and_reid.cpp - native C++ port of route_and_reid.py: class-routed
// cascade + cross-camera re-ID, built directly on pycamtrt::Pipeline. THE
// showcase for Select (detection routing) and the Embedding family.
//
// Stage 1 ("detect") runs a COCO YOLOv8 detector; detections are ROUTED by
// class to two specialist siblings, each a Select -> Engine -> Postprocess
// 3-step layer: Select(classes={0}) (person) -> "person" (a resnet18
// penultimate-feature engine, Family::Embedding, 512-d vectors this binary
// cosine-matches ACROSS streams), and Select(classes={2,3,5,7},
// min_size=48) (vehicles) -> "vehicle" (a mobilenet ImageNet classifier,
// Family::Argmax). Honest caveats (same as the Python example): resnet18-
// on-ImageNet was never trained for re-ID (matches are illustrative),
// vehicle labels are loose ImageNet stand-ins, and the demo farm clip is
// an indoor scene (expect person matches, no vehicles).
//
// Usage:
//   route_and_reid URL [URL...]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/pipeline.h"

using pycamtrt::ChildOutput;
using pycamtrt::Family;
using pycamtrt::FrameResult;
using pycamtrt::LayerDesc;
using pycamtrt::Pipeline;
using pycamtrt::PipelineConfig;
using pycamtrt::StepDesc;
using pycamtrt::StepKind;
using pycamtrt::StreamDesc;
using pycamtrt::StreamInfo;

namespace {

const char* const kDetectEngine = "models/yolov8n_b1-16_fp16_sm86.engine";  // COCO, 80 cls
const char* const kEmbedEngine = "models/resnet18emb_b1-32_fp16_sm86.engine";  // [N,512]
const char* const kVehicleEngine = "models/mobilenet_v3s_b1-32_fp16_sm86.engine";

const float kNormOffset[3] = {-123.675f, -116.28f, -103.53f};
const float kNormScale[3] = {1.f / 58.395f, 1.f / 57.12f, 1.f / 57.375f};

const std::vector<int> kPersonClasses = {0};
const std::vector<int> kVehicleClasses = {2, 3, 5, 7};  // car, motorcycle, bus, truck
const float kVehicleMinSize = 48.f;

// Cosine similarity above which two person vectors count as "same person"
// across cameras - tune per model/content; 0.85 is a demo value.
const float kMatchThreshold = 0.85f;
const size_t kGalleryPerStream = 50;  // recent vectors kept per stream

const int kSkip = 2;
const int kMaxFrames = 600;

// A few well-known ImageNet-1k vehicle classes, for readable labels; any
// other label id prints as "imagenet_<id>" (see VehicleLabel below).
const std::map<int, std::string> kVehicleLabels = {
    {407, "ambulance"},   {468, "cab"},         {511, "convertible"},
    {555, "fire_engine"}, {569, "garbage_truck"}, {654, "minibus"},
    {656, "minivan"},     {675, "moving_van"},  {717, "pickup"},
    {734, "police_van"},  {751, "racer"},       {779, "school_bus"},
    {817, "sports_car"},  {864, "tow_truck"},   {867, "trailer_truck"},
    {874, "trolleybus"},
};

std::string VehicleLabel(int id) {
    auto it = kVehicleLabels.find(id);
    return it != kVehicleLabels.end() ? it->second : "imagenet_" + std::to_string(id);
}

// L2-normalizes `v` in place (same +1e-9 epsilon as the Python example).
void Normalize(std::vector<float>* v) {
    double sumsq = 0.0;
    for (float x : *v) sumsq += (double)x * x;
    const float norm = (float)std::sqrt(sumsq) + 1e-9f;
    for (float& x : *v) x /= norm;
}

float Dot(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) s += a[i] * b[i];
    return s;
}

PipelineConfig BuildConfig(const std::vector<std::string>& urls) {
    PipelineConfig cfg;
    cfg.streams.reserve(urls.size());
    for (const std::string& u : urls) cfg.streams.push_back(StreamDesc{u});
    cfg.skip = kSkip;
    cfg.max_frames = kMaxFrames;
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
    const int det_idx = (int)cfg.steps.size();
    cfg.steps.push_back(post_step);
    cfg.layers.push_back(LayerDesc{"detect", {engine_idx, det_idx}});

    // "person": Select(classes={0}) -> Engine(embedding) -> Postprocess.
    StepDesc person_sel;
    person_sel.kind = StepKind::Select;
    person_sel.input = det_idx;
    person_sel.sel_classes = kPersonClasses;
    const int person_sel_idx = (int)cfg.steps.size();
    cfg.steps.push_back(person_sel);

    StepDesc person_eng;
    person_eng.kind = StepKind::Engine;
    person_eng.input = person_sel_idx;
    person_eng.engine_path = kEmbedEngine;
    person_eng.color = 1;  // RGB
    for (int c = 0; c < 3; c++) {
        person_eng.norm_offset[c] = kNormOffset[c];
        person_eng.norm_scale[c] = kNormScale[c];
    }
    const int person_eng_idx = (int)cfg.steps.size();
    cfg.steps.push_back(person_eng);

    StepDesc person_post;
    person_post.kind = StepKind::Postprocess;
    person_post.input = person_eng_idx;
    person_post.family = Family::Embedding;
    const int person_post_idx = (int)cfg.steps.size();
    cfg.steps.push_back(person_post);
    cfg.layers.push_back(
        LayerDesc{"person", {person_sel_idx, person_eng_idx, person_post_idx}});

    // "vehicle": Select(classes={2,3,5,7}, min_size=48) -> Engine(argmax).
    StepDesc vehicle_sel;
    vehicle_sel.kind = StepKind::Select;
    vehicle_sel.input = det_idx;
    vehicle_sel.sel_classes = kVehicleClasses;
    vehicle_sel.sel_min_size = kVehicleMinSize;
    const int vehicle_sel_idx = (int)cfg.steps.size();
    cfg.steps.push_back(vehicle_sel);

    StepDesc vehicle_eng;
    vehicle_eng.kind = StepKind::Engine;
    vehicle_eng.input = vehicle_sel_idx;
    vehicle_eng.engine_path = kVehicleEngine;
    vehicle_eng.color = 1;
    for (int c = 0; c < 3; c++) {
        vehicle_eng.norm_offset[c] = kNormOffset[c];
        vehicle_eng.norm_scale[c] = kNormScale[c];
    }
    const int vehicle_eng_idx = (int)cfg.steps.size();
    cfg.steps.push_back(vehicle_eng);

    StepDesc vehicle_post;
    vehicle_post.kind = StepKind::Postprocess;
    vehicle_post.input = vehicle_eng_idx;
    vehicle_post.family = Family::Argmax;
    const int vehicle_post_idx = (int)cfg.steps.size();
    cfg.steps.push_back(vehicle_post);
    cfg.layers.push_back(
        LayerDesc{"vehicle", {vehicle_sel_idx, vehicle_eng_idx, vehicle_post_idx}});

    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> urls(argv + 1, argv + argc);
    if (urls.empty()) urls.push_back("rtsp://localhost:8554/cam1");
    const int n_streams = (int)urls.size();

    std::unique_ptr<Pipeline> pipeline;
    try {
        pipeline.reset(new Pipeline(BuildConfig(urls)));
    } catch (const std::exception& e) {
        fprintf(stderr, "pipeline init failed: %s\n", e.what());
        return -1;
    }
    pipeline->Start();

    std::map<int, std::deque<std::vector<float>>> galleries;
    for (int i = 0; i < n_streams; i++) galleries[i];  // seed every stream, empty
    int frames = 0, persons = 0, vehicles = 0, matches = 0;
    std::map<std::string, int> vehicle_counts;
    float best_cos = 0.f;
    int best_s1 = -1, best_s2 = -1;

    for (;;) {
        FrameResult r;
        const Pipeline::PollStatus st = pipeline->Poll(&r, 500);
        if (st == Pipeline::PollStatus::Timeout) continue;
        if (st == Pipeline::PollStatus::Finished) break;

        frames++;
        // r.children[0] = "person" (Embedding: vectors), r.children[1] =
        // "vehicle" (Argmax: labels/label_scores) - same order as
        // cfg.layers[1:], mirroring Python's r.outputs["person"]/
        // r.outputs["vehicle"].
        const ChildOutput* person_out = r.children.size() > 0 ? &r.children[0] : nullptr;
        const ChildOutput* vehicle_out = r.children.size() > 1 ? &r.children[1] : nullptr;

        for (size_t i = 0; i < r.detections.size(); i++) {
            if (person_out && i < person_out->vectors.size() &&
                !person_out->vectors[i].empty()) {
                std::vector<float> v = person_out->vectors[i];
                Normalize(&v);
                persons++;
                for (auto& [sid, gal] : galleries) {
                    if (sid == r.stream_id || gal.empty()) continue;
                    float top = 0.f;
                    for (const auto& g : gal) top = std::max(top, Dot(g, v));
                    if (top >= kMatchThreshold) {
                        matches++;
                        if (top > best_cos) {
                            best_cos = top;
                            best_s1 = r.stream_id;
                            best_s2 = sid;
                        }
                        if (matches <= 10) {
                            printf("MATCH person s%d frame %d ~ s%d (cos %.3f)\n",
                                   r.stream_id, r.frame_no, sid, top);
                        }
                    }
                }
                auto& own = galleries[r.stream_id];
                own.push_back(v);
                if (own.size() > kGalleryPerStream) own.pop_front();
            }
            // "routed" == non-empty entry - a filtered-away detection
            // keeps this child's default (0, 0.0), same alignment
            // contract Select guarantees on the Python side.
            if (vehicle_out && i < vehicle_out->labels.size() &&
                vehicle_out->label_scores[i] != 0.0f) {
                vehicles++;
                vehicle_counts[VehicleLabel(vehicle_out->labels[i])]++;
            }
        }
    }

    std::vector<StreamInfo> per_stream;
    per_stream.reserve(n_streams);
    for (int i = 0; i < n_streams; i++) per_stream.push_back(pipeline->GetStreamInfo(i));
    pipeline.reset();

    printf("\n---- summary ----\n");
    printf("frames: %d  person crops embedded: %d  vehicle crops classified: %d\n",
           frames, persons, vehicles);
    printf("cross-camera person matches (cos >= %.2f): %d\n", kMatchThreshold, matches);
    if (best_s1 >= 0) printf("best match: s%d ~ s%d (cos %.3f)\n", best_s1, best_s2, best_cos);
    if (!vehicle_counts.empty()) {
        printf("vehicle types:\n");
        std::vector<std::pair<std::string, int>> sorted_counts(vehicle_counts.begin(),
                                                                 vehicle_counts.end());
        std::sort(sorted_counts.begin(), sorted_counts.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        for (const auto& [name, count] : sorted_counts) printf("  %5d  %s\n", count, name.c_str());
    }
    for (int i = 0; i < n_streams; i++) {
        printf("s%d: decoded=%d reconnects=%d\n", i, per_stream[i].decoded,
               per_stream[i].reconnects);
    }
    return 0;
}
