// read_and_classify.cpp - native C++ port of read_and_classify.py: a
// depth-2 TREE, one detector root feeding TWO SIBLING recognition
// children (read=ctc, classify=argmax), built directly on
// pycamtrt::Pipeline. Both children's Engine steps take the SAME root
// Postprocess step index as `input` - neither depends on the other's
// output (that is the whole point of a depth-2 tree, as opposed to a
// 3-deep chain, which the executor rejects by name - see pipeline.cpp's
// Validate()). This reuses read_plates.cpp's and classify_detections.cpp's
// engines/norm unchanged, just wired as two siblings under one root.
//
// Usage:
//   read_and_classify URL [URL...] [--verify]

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
const char* const kReadEngine = "models/lprnet_b1-32_fp32_sm86.engine";
const char* const kClassifierEngine = "models/mobilenet_v3s_b1-32_fp16_sm86.engine";

// True per-channel ImageNet normalization (M3a) - copied verbatim from
// classify_detections.cpp (kept in sync by hand, same convention the
// Python examples use).
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

    // Root: the plate detector. `root_idx` is the SHARED input both
    // sibling children below crop from - never each other's output.
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
    const int root_idx = (int)cfg.steps.size();
    cfg.steps.push_back(post_step);

    LayerDesc detect;
    detect.name = "detect";
    detect.steps = {engine_idx, root_idx};
    cfg.layers.push_back(detect);

    // Sibling 1: OCR ("read", ctc) - crops `root_idx`'s detections.
    StepDesc read_engine_step;
    read_engine_step.kind = StepKind::Engine;
    read_engine_step.input = root_idx;
    read_engine_step.engine_path = kReadEngine;
    const int read_engine_idx = (int)cfg.steps.size();
    cfg.steps.push_back(read_engine_step);

    StepDesc read_post_step;
    read_post_step.kind = StepKind::Postprocess;
    read_post_step.input = read_engine_idx;
    read_post_step.family = Family::Ctc;
    const int read_post_idx = (int)cfg.steps.size();
    cfg.steps.push_back(read_post_step);

    LayerDesc read;
    read.name = "read";
    read.steps = {read_engine_idx, read_post_idx};
    cfg.layers.push_back(read);

    // Sibling 2: classifier ("classify", argmax) - ALSO crops `root_idx`
    // directly (not `read`'s output).
    StepDesc classify_engine_step;
    classify_engine_step.kind = StepKind::Engine;
    classify_engine_step.input = root_idx;
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

    int frames_seen = 0, dets_seen = 0, read_nonempty = 0;
    int verify_ok = 0, verify_total = 0;
    std::map<int, int> classify_counts;

    for (;;) {
        FrameResult r;
        const Pipeline::PollStatus st = pipeline->Poll(&r, 500);
        if (st == Pipeline::PollStatus::Timeout) continue;
        if (st == Pipeline::PollStatus::Finished) break;

        frames_seen++;
        dets_seen += (int)r.detections.size();
        // r.texts/r.labels/r.label_scores are the back-compat fields (the
        // FIRST Ctc child / FIRST Argmax child - see core/result.h's
        // WHY-comment), which for this exact two-sibling tree are simply
        // "read"'s and "classify"'s own outputs - same values Python's
        // r.outputs["read"]/r.outputs["classify"] carry.
        for (size_t i = 0; i < r.detections.size(); i++) {
            const std::string text = i < r.texts.size() ? r.texts[i] : "";
            const int label = i < r.labels.size() ? r.labels[i] : -1;
            const float score = i < r.label_scores.size() ? r.label_scores[i] : 0.f;
            if (!text.empty()) read_nonempty++;
            classify_counts[label]++;
            printf("s%d frame %d det%zu: '%s' | %d(%.2f)\n", r.stream_id,
                   r.frame_no, i, text.c_str(), label, score);
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
    printf("frames seen: %d  detections: %d  non-empty reads: %d\n", frames_seen,
           dets_seen, read_nonempty);
    printf("classify label counts (top 10):\n");
    std::vector<std::pair<int, int>> sorted_counts(classify_counts.begin(),
                                                     classify_counts.end());
    std::sort(sorted_counts.begin(), sorted_counts.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    const size_t top_n = std::min<size_t>(10, sorted_counts.size());
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
