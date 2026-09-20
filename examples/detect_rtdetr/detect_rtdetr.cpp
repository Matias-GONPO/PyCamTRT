// detect_rtdetr.cpp - native C++ port of detect_rtdetr.py: RT-DETR as the
// detection layer, built directly on pycamtrt::Pipeline. One stage
// ("detect"): an RT-DETR-L engine under Family::RtDetr, the NMS-free
// transformer detector family - same [N,300,6] head contract as YoloE2E,
// but RT-DETR's box columns are normalized cx,cy,w,h (see graph.h's
// Family comment); score_thresh is honored, iou_thresh is ignored (no NMS
// stage exists), and SAHI is rejected on this family at construction.
//
// Model note: rtdetr_l is ultralytics-licensed (AGPL-3.0), so its .onnx
// export is not distributed in this repo - see models/README.md and
// python/export_ultralytics.py.
//
// Usage:
//   detect_rtdetr URL [URL...]

#include <algorithm>
#include <cstdio>
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

const char* const kEngine = "models/rtdetr_l_b1-8_fp16_sm86.engine";
const int kSkip = 2;
const int kMaxFrames = 600;

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
    engine_step.engine_path = kEngine;
    const int engine_idx = (int)cfg.steps.size();
    cfg.steps.push_back(engine_step);

    StepDesc post_step;
    post_step.kind = StepKind::Postprocess;
    post_step.input = engine_idx;
    post_step.family = Family::RtDetr;
    post_step.score_thresh = 0.5f;
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
    if (urls.empty()) urls.push_back("rtsp://localhost:8554/cam1");

    std::unique_ptr<Pipeline> pipeline;
    try {
        pipeline.reset(new Pipeline(BuildConfig(urls)));
    } catch (const std::exception& e) {
        fprintf(stderr, "pipeline init failed: %s\n", e.what());
        return -1;
    }
    pipeline->Start();

    int frames = 0, dets = 0;
    std::map<int, int> class_counts;

    for (;;) {
        FrameResult r;
        const Pipeline::PollStatus st = pipeline->Poll(&r, 500);
        if (st == Pipeline::PollStatus::Timeout) continue;
        if (st == Pipeline::PollStatus::Finished) break;

        frames++;
        dets += (int)r.detections.size();
        for (const auto& d : r.detections) class_counts[d.cls]++;
        if (frames <= 5 && !r.detections.empty()) {
            const auto& d = r.detections[0];
            printf("s%d frame %d: cls=%d score=%.2f box=(%.0f,%.0f,%.0fx%.0f)\n",
                   r.stream_id, r.frame_no, d.cls, d.score, d.x, d.y, d.w, d.h);
        }
    }

    const int n_streams = (int)urls.size();
    std::vector<StreamInfo> per_stream;
    per_stream.reserve(n_streams);
    for (int i = 0; i < n_streams; i++) per_stream.push_back(pipeline->GetStreamInfo(i));
    pipeline.reset();

    printf("\n---- summary ----\n");
    printf("frames: %d  detections: %d\n", frames, dets);
    printf("top COCO class ids:\n");
    std::vector<std::pair<int, int>> sorted_counts(class_counts.begin(), class_counts.end());
    std::sort(sorted_counts.begin(), sorted_counts.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    const size_t top_n = std::min<size_t>(8, sorted_counts.size());
    for (size_t i = 0; i < top_n; i++) {
        printf("  %5d  cls %d\n", sorted_counts[i].second, sorted_counts[i].first);
    }
    for (int i = 0; i < n_streams; i++) {
        printf("s%d: decoded=%d reconnects=%d\n", i, per_stream[i].decoded,
               per_stream[i].reconnects);
    }
    return 0;
}
