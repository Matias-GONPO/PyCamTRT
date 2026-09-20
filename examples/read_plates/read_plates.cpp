// read_plates.cpp - native C++ port of read_plates.py, built directly on
// pycamtrt::Pipeline (src/core/graph.h/pipeline.h/result.h) instead of the
// pycamtrt Python compiler layer. Same two-stage step graph (detect ->
// crop -> OCR), same per-result plate prints, same summary block - see
// examples/read_plates/README.md for run commands and expected output.
//
// Note on output text (same caveat as the Python side): the bundled
// LPRNet engine was trained on a Chinese license-plate charset, so
// decoded strings on non-Chinese plates look garbled - expected, this
// exercises the detect -> crop -> OCR mechanics, not charset fit.
//
// Usage:
//   read_plates URL [URL...] [--verify]

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

// skip=2 (infer every other decoded frame) + max_frames=600 (decoded
// frames per stream) -> ~300 inferred frames/stream, ~20s wall at 30fps
// source - same cadence as the Python example.
const int kSkip = 2;
const int kMaxFrames = 600;

// Builds the same detect(yolo) -> read(ctc) step graph
// pycamtrt.Layer/Engine/Postprocess compiles to in read_plates.py's
// build_pipeline().
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
    engine_step.input = -1;  // raw stream source
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

    // Cross-layer edge: an Engine fed a Postprocess step from an earlier
    // layer auto-crops each detection before running the OCR engine on it
    // (same mechanism the Python Engine(detect.steps[-1], READ_ENGINE) call
    // compiles down to - see pipeline.cpp's cascade-crop path).
    StepDesc read_engine_step;
    read_engine_step.kind = StepKind::Engine;
    read_engine_step.input = post_idx;
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

    int frames_seen = 0, plates_seen = 0;
    int verify_ok = 0, verify_total = 0;
    std::map<std::string, int> plate_counts;

    for (;;) {
        FrameResult r;
        const Pipeline::PollStatus st = pipeline->Poll(&r, 500);
        if (st == Pipeline::PollStatus::Timeout) continue;
        if (st == Pipeline::PollStatus::Finished) break;

        frames_seen++;
        // r.texts is the "read" layer's back-compat field (first Ctc
        // child), aligned with r.detections - exactly what the Python
        // side's r.outputs["read"] resolves to for this single-child
        // cascade (see core/result.h's WHY-comment).
        if (!r.texts.empty()) {
            plates_seen += (int)r.texts.size();
            for (const std::string& t : r.texts) {
                plate_counts[t]++;
                printf("s%d frame %d: %s\n", r.stream_id, r.frame_no, t.c_str());
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
    pipeline.reset();  // Stop() + teardown before printing (same order as
                       // the Python example's `with` block exit)

    printf("\n---- summary ----\n");
    printf("frames seen: %d  plates read: %d\n", frames_seen, plates_seen);
    printf("distinct plate strings:\n");
    std::vector<std::pair<std::string, int>> sorted_counts(plate_counts.begin(),
                                                            plate_counts.end());
    std::sort(sorted_counts.begin(), sorted_counts.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& [text, count] : sorted_counts) {
        printf("  %5d  '%s'\n", count, text.c_str());
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
