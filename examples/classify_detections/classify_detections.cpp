// classify_detections.cpp - native C++ port of classify_detections.py: a
// detect -> classify cascade exercising the "argmax" postprocess family,
// built directly on pycamtrt::Pipeline. Same step graph as read_plates.cpp
// with the OCR (family=Ctc) child swapped for a classifier (family=Argmax)
// one - see examples/classify_detections/README.md for run commands.
//
// Difference from the Python example: this binary has no torchvision to
// borrow ImageNet-1k category names from, so it prints bare numeric class
// ids (see the README's expected-output sample) - the mechanics under test
// (crop -> classify -> aligned labels) are identical either way.
//
// Usage:
//   classify_detections URL [URL...] [--verify]

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/pipeline.h"

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

const char* const kDetectEngine = "models/yolov8n_plates_b1-16_fp16_sm86.engine";
const char* const kClassifierEngine = "models/mobilenet_v3s_b1-32_fp16_sm86.engine";

// True per-channel ImageNet normalization (M3a) - see
// python/export_classifier.py's docstring for the mean/std -> offset/scale
// derivation. index 0/1/2 = R/G/B (this engine's color is set to RGB
// below), matching torchvision's mean=[0.485,0.456,0.406],
// std=[0.229,0.224,0.225].
const float kNormOffset[3] = {-123.675f, -116.28f, -103.53f};
const float kNormScale[3] = {1.f / 58.395f, 1.f / 57.12f, 1.f / 57.375f};

const int kSkip = 2;
const int kMaxFrames = 600;

PipelineConfig BuildConfig(const std::vector<std::string>& urls, bool verify) {
    PipelineConfig cfg;
    cfg.streams.reserve(urls.size());
    for (const std::string& u : urls) cfg.streams.push_back(StreamDesc{u});
    cfg.skip = kSkip;
    cfg.max_frames = kMaxFrames;
    cfg.verify = verify;
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

    // Cross-layer edge (same crop mechanism as read_plates.cpp), with an
    // explicit per-channel norm/color override - the classifier's own
    // trained input convention, not the layer-1 (OCR) default.
    StepDesc classify_engine_step;
    classify_engine_step.kind = StepKind::Engine;
    classify_engine_step.input = post_idx;
    classify_engine_step.engine_path = kClassifierEngine;
    for (int c = 0; c < 3; c++) {
        classify_engine_step.norm_offset[c] = kNormOffset[c];
        classify_engine_step.norm_scale[c] = kNormScale[c];
    }
    classify_engine_step.color = 1;  // RGB
    const int classify_engine_idx = (int)cfg.steps.size();
    cfg.steps.push_back(classify_engine_step);

    StepDesc classify_post_step;
    classify_post_step.kind = StepKind::Postprocess;
    classify_post_step.input = classify_engine_idx;
    classify_post_step.family = Family::Argmax;
    const int classify_post_idx = (int)cfg.steps.size();
    cfg.steps.push_back(classify_post_step);

    LayerDesc classify;
    classify.name = "classify";
    classify.steps = {classify_engine_idx, classify_post_idx};
    cfg.layers.push_back(classify);

    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> urls;
    bool verify = false;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--verify")) verify = true;
        else urls.push_back(argv[i]);
    }
    if (urls.empty()) urls.push_back("rtsp://localhost:8554/cam1");

    std::unique_ptr<Pipeline> pipeline;
    try {
        pipeline.reset(new Pipeline(BuildConfig(urls, verify)));
    } catch (const std::exception& e) {
        fprintf(stderr, "pipeline init failed: %s\n", e.what());
        return -1;
    }
    pipeline->Start();

    int frames_seen = 0, dets_seen = 0;
    int verify_ok = 0, verify_total = 0;
    std::map<int, int> label_counts;

    for (;;) {
        FrameResult r;
        const Pipeline::PollStatus st = pipeline->Poll(&r, 500);
        if (st == Pipeline::PollStatus::Timeout) continue;
        if (st == Pipeline::PollStatus::Finished) break;

        frames_seen++;
        // r.labels/r.label_scores: the "classify" layer's back-compat
        // fields (first Argmax child), aligned with r.detections - what
        // Python's r.outputs["classify"] (a list of (label, score) pairs)
        // resolves to for this single-child cascade.
        if (!r.labels.empty()) {
            dets_seen += (int)r.labels.size();
            for (size_t i = 0; i < r.labels.size(); i++) {
                label_counts[r.labels[i]]++;
                printf("s%d frame %d: label=%d logit=%.2f\n", r.stream_id,
                       r.frame_no, r.labels[i], r.label_scores[i]);
            }
        }
        if (verify) {
            verify_total++;
            if (r.verify_ok) verify_ok++;
        }
    }

    const int n_streams = (int)urls.size();
    std::vector<StreamInfo> per_stream;
    per_stream.reserve(n_streams);
    for (int i = 0; i < n_streams; i++) per_stream.push_back(pipeline->GetStreamInfo(i));
    pipeline.reset();

    printf("\n---- summary ----\n");
    printf("frames seen: %d  detections classified: %d\n", frames_seen, dets_seen);
    printf("label counts:\n");
    std::vector<std::pair<int, int>> sorted_counts(label_counts.begin(), label_counts.end());
    std::sort(sorted_counts.begin(), sorted_counts.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    const size_t top_n = std::min<size_t>(15, sorted_counts.size());
    for (size_t i = 0; i < top_n; i++) {
        printf("  %5d  label=%d\n", sorted_counts[i].second, sorted_counts[i].first);
    }
    for (int i = 0; i < n_streams; i++) {
        printf("s%d: decoded=%d reconnects=%d\n", i, per_stream[i].decoded,
               per_stream[i].reconnects);
    }

    if (verify) {
        printf("VERIFY %d/%d\n", verify_ok, verify_total);
        if (verify_ok != verify_total) return 1;
    }
    return 0;
}
