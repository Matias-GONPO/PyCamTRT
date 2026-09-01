// Step 5 CLI: thin consumer of cordero::Pipeline (see src/core/). All the
// multi-stream orchestration (producer threads, greedy batcher, GPU thread,
// SAHI, OCR cascade, --verify) now lives in libcordero; this binary just
// builds a PipelineConfig step graph from flags, drives Start()/Poll(), and
// reproduces the same per-frame lines and summary block the monolithic
// binary used to print directly.
//
// Usage:
//   rtsp_infer_multi URL [URL...] [--engine PATH] [--frames N] [--skip K]
//                    [--verify]
// N counts DECODED frames per stream (run length is comparable across skip
// factors). --skip K infers every K-th decoded frame, phase-offset per
// stream; every frame is still decoded and mapped (H.264 inter-frame deps),
// so K only sheds preprocess/inference/postprocess load. --verify re-runs
// each INFERRED frame's postprocess on the CPU from the raw tensor and
// diffs (detection-level regression check; composes with --skip).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/pipeline.h"

using cordero::Detection;
using cordero::Family;
using cordero::FrameResult;
using cordero::LayerDesc;
using cordero::Pipeline;
using cordero::PipelineConfig;
using cordero::StepDesc;
using cordero::StepKind;
using cordero::StreamDesc;
using cordero::StreamInfo;

namespace {

// ---- Per-stream aggregates the CLI recomputes from FrameResults ---------
// (the core only tracks decoded/reconnects/failed - see StreamInfo).
struct StreamAgg {
    int frames = 0;
    double latency_ms_sum = 0.0;
    int64_t prev_pts = -1;
    int pts_nonmono = 0;
    std::vector<int64_t> pts_deltas;
};

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> urls;
    std::string engine_path = "models/yolov8n_b1-16_fp32_sm86.engine";
    int max_frames = 300;
    int skip = 1;  // 1 = infer every decoded frame (no skipping)
    bool verify = false;
    bool key_only = false;  // --decode key: decode only keyframes (IDR)
    std::string ocr_path;   // empty = cascade off
    // SAHI (tiled inference), opt-in. Off => the exact pre-SAHI path.
    bool sahi = false;
    int sahi_tile = 640;          // tile side, source px (= engine input)
    float sahi_overlap = 0.2f;    // fraction shared between neighbors
    float sahi_merge_iou = 0.5f;  // cross-tile de-dup threshold
    bool sahi_full_frame = true;  // also merge the whole-frame pass
    bool sahi_serial = false;     // A/B: keep the old per-slot tile loop
    std::string dump_dets_path;   // --dump-dets PATH: parity-test dump, see
                                   // tools/sahi_parity.sh
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--engine") && i + 1 < argc) engine_path = argv[++i];
        else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) max_frames = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--skip") && i + 1 < argc) skip = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--ocr") && i + 1 < argc) ocr_path = argv[++i];
        else if (!std::strcmp(argv[i], "--verify")) verify = true;
        else if (!std::strcmp(argv[i], "--sahi")) sahi = true;
        else if (!std::strcmp(argv[i], "--sahi-tile") && i + 1 < argc) sahi_tile = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--sahi-overlap") && i + 1 < argc) sahi_overlap = (float)std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--sahi-merge-iou") && i + 1 < argc) sahi_merge_iou = (float)std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--sahi-no-full-frame")) sahi_full_frame = false;
        else if (!std::strcmp(argv[i], "--sahi-serial")) sahi_serial = true;
        else if (!std::strcmp(argv[i], "--dump-dets") && i + 1 < argc) dump_dets_path = argv[++i];
        else if (!std::strcmp(argv[i], "--decode") && i + 1 < argc) {
            const char* m = argv[++i];
            if (!std::strcmp(m, "key")) key_only = true;
            else if (std::strcmp(m, "all")) {
                fprintf(stderr, "--decode must be 'all' or 'key'\n");
                return 1;
            }
        }
        else urls.push_back(argv[i]);
    }
    if (sahi && verify) {
        fprintf(stderr,
                "--verify checks the base (untiled) path against its CPU "
                "reference; run it with --sahi off. SAHI has its own "
                "checkpoint (sahi_tiles_test).\n");
        return 1;
    }
    if (urls.empty() || skip < 1) {
        fprintf(stderr,
                "Usage: %s URL [URL...] [--engine PATH] [--frames N] "
                "[--skip K] [--decode all|key] [--ocr PATH] [--verify]\n"
                "  --frames N  decoded frames per stream (default 300)\n"
                "  --skip K    infer every K-th decoded frame (default 1 = all).\n"
                "              Detection rate becomes ~30/K per stream; K>=2\n"
                "              trades density for GPU headroom and latency\n"
                "              (measured 16 streams: K=1 ~45 ms at the capacity\n"
                "              knee, K=2 ~6 ms with half the GPU free). Frames\n"
                "              between detections need a tracker downstream.\n"
                "  --decode M  'all' (default) decodes every frame; 'key'\n"
                "              decodes ONLY keyframes (IDR), dropping other\n"
                "              packets at the demuxer. Cuts NVDEC load to one\n"
                "              frame per GOP per stream (update rate = the\n"
                "              camera's keyframe interval) — the dial for\n"
                "              100+ streams at ~1 update/s. --skip then\n"
                "              applies per DECODED (key)frame.\n"
                "  --ocr PATH  stage-2 cascade: crop every stage-1 detection\n"
                "              from the full-res frame and read it with an\n"
                "              LPRNet-style engine (meant for a plate detector\n"
                "              as stage 1). Adds its cost to the batch cycle\n"
                "              (sequential mode).\n"
                "  --sahi      tiled inference (classic SAHI): slice each\n"
                "              inferred frame's FULL-RES ring copy into\n"
                "              overlapping tiles, infer every tile, merge\n"
                "              with the normal whole-frame pass. Recovers\n"
                "              small objects the 640-letterbox discards, at\n"
                "              ~T x the inference cost (T printed at start).\n"
                "              [--sahi-tile 640] [--sahi-overlap 0.2]\n"
                "              [--sahi-merge-iou 0.5] [--sahi-no-full-frame]\n"
                "              [--sahi-serial] (per-slot tile loop, for A/B\n"
                "              benchmarking)\n"
                "  --dump-dets PATH  append one line per detection ('stream\n"
                "              frame_no cls score x y w h', %%.9g precision)\n"
                "              plus one 'F stream frame_no ndets' line per\n"
                "              frame (empty frames included) to PATH - for\n"
                "              offline serial-vs-pooled SAHI parity diffing\n"
                "              (tools/sahi_parity.sh), not for normal use.\n",
                argv[0]);
        return -1;
    }
    FILE* dump_dets_fp = nullptr;
    if (!dump_dets_path.empty()) {
        dump_dets_fp = fopen(dump_dets_path.c_str(), "w");
        if (!dump_dets_fp) {
            fprintf(stderr, "--dump-dets: could not open '%s' for writing\n",
                    dump_dets_path.c_str());
            return -1;
        }
    }

    const int n_streams = (int)urls.size();

    printf("=== Step 5: multi-stream batched pipeline (%d streams, skip %d%s) ===\n",
           n_streams, skip, key_only ? ", decode=key" : "");
    if (skip > 1) {
        printf("note: skip %d -> ~%.1f detections/s per 30 fps stream; frames "
               "between detections carry no boxes (tracker not yet built)\n",
               skip, 30.0 / skip);
    }
    if (key_only) {
        printf("note: decode=key -> only keyframes are decoded; effective "
               "rate = stream GOP cadence (e.g. GOP 30 @ 30 fps = 1 frame/s "
               "per stream); --frames counts DECODED (key)frames\n");
    }

    // ---- Build the step graph. This is what a Python user would do too:
    // push StepDescs, wire a Process(implicit)->Engine->Postprocess layer
    // 0, optionally an Engine->Postprocess(Ctc) layer 1 for the OCR
    // cascade.
    PipelineConfig cfg;
    // CLI keeps global-only flags (no per-stream overrides here): every
    // StreamDesc is just a bare url, so all streams inherit cfg.skip/
    // cfg.key_only below - see graph.h's StreamDesc for the per-stream API.
    cfg.streams.reserve(urls.size());
    for (const std::string& u : urls) cfg.streams.push_back(StreamDesc{u});
    cfg.skip = skip;
    cfg.key_only = key_only;
    cfg.verify = verify;
    cfg.max_frames = max_frames;
    // Core log lines go to stdout: Setup() emits the same "✓ Engine loaded"
    // / "✓ Head" / "✓ OCR engine" banners the monolithic binary printed, so
    // the CLI must not print its own copies (they'd duplicate).
    cfg.log = [](const std::string& s) { printf("%s\n", s.c_str()); };

    StepDesc engine_step;
    engine_step.kind = StepKind::Engine;
    engine_step.input = -1;  // Process step implied - straight off the raw
                              // stream source
    engine_step.engine_path = engine_path;
    const int engine_idx = (int)cfg.steps.size();
    cfg.steps.push_back(engine_step);

    StepDesc post_step;
    post_step.kind = StepKind::Postprocess;
    post_step.input = engine_idx;
    post_step.family = Family::YoloDetect;
    post_step.score_thresh = 0.4f;
    post_step.iou_thresh = 0.45f;
    const int post_idx = (int)cfg.steps.size();
    cfg.steps.push_back(post_step);

    LayerDesc layer0;
    layer0.name = "detect";
    layer0.steps = {engine_idx, post_idx};
    layer0.sahi = sahi;
    layer0.sahi_tile = sahi_tile;
    layer0.sahi_overlap = sahi_overlap;
    layer0.sahi_merge_iou = sahi_merge_iou;
    layer0.sahi_full_frame = sahi_full_frame;
    layer0.sahi_serial = sahi_serial;
    cfg.layers.push_back(layer0);

    if (!ocr_path.empty()) {
        StepDesc ocr_engine_step;
        ocr_engine_step.kind = StepKind::Engine;
        ocr_engine_step.input = post_idx;
        ocr_engine_step.engine_path = ocr_path;
        const int ocr_engine_idx = (int)cfg.steps.size();
        cfg.steps.push_back(ocr_engine_step);

        StepDesc ocr_post_step;
        ocr_post_step.kind = StepKind::Postprocess;
        ocr_post_step.input = ocr_engine_idx;
        ocr_post_step.family = Family::Ctc;
        const int ocr_post_idx = (int)cfg.steps.size();
        cfg.steps.push_back(ocr_post_step);

        LayerDesc layer1;
        layer1.name = "ocr";
        layer1.steps = {ocr_engine_idx, ocr_post_idx};
        cfg.layers.push_back(layer1);
    }

    std::unique_ptr<Pipeline> pipeline;
    try {
        pipeline.reset(new Pipeline(std::move(cfg)));
    } catch (const std::exception& e) {
        fprintf(stderr, "pipeline init failed: %s\n", e.what());
        return -1;
    }

    pipeline->Start();

    std::vector<StreamAgg> agg(n_streams);
    std::vector<int> batch_hist(pipeline->MaxBatch() + 1, 0);
    int batches = 0;          // true GPU batches, counted via batch_seq
    int last_batch_seq = -1;  // (every frame of one batch carries the same
                              // seq, so count/histogram once per change)
    int verify_ok = 0, verify_fail = 0;
    int plates_total = 0;
    double pre_gpu_ms = 0.0, queue_ms = 0.0, gpu_ms = 0.0;

    const auto t_start = std::chrono::steady_clock::now();
    for (;;) {
        FrameResult r;
        const Pipeline::PollStatus st = pipeline->Poll(&r, 500);
        if (st == Pipeline::PollStatus::Timeout) continue;
        if (st == Pipeline::PollStatus::Finished) break;

        if (r.batch_seq != last_batch_seq) {  // once per GPU batch
            last_batch_seq = r.batch_seq;
            batches++;
            batch_hist[std::min<int>(r.batch_size, (int)batch_hist.size() - 1)]++;
            gpu_ms += r.ms_take_to_done;  // whole-batch cycle, avg per batch
        }
        pre_gpu_ms += r.ms_pop_to_ready;
        queue_ms += r.ms_ready_to_take;

        StreamAgg& sa = agg[r.stream_id];
        sa.frames++;
        sa.latency_ms_sum += r.ms_pop_to_ready + r.ms_ready_to_take + r.ms_take_to_done;
        if (r.pts_us >= 0) {
            if (sa.prev_pts >= 0) {
                const int64_t d = r.pts_us - sa.prev_pts;
                if (d <= 0) sa.pts_nonmono++;
                else sa.pts_deltas.push_back(d);
            }
            sa.prev_pts = r.pts_us;
        }

        if (r.verified) {
            if (r.verify_ok) verify_ok++;
            else {
                verify_fail++;
                printf("\xe2\x9c\x97 verify mismatch: stream %d frame %d "
                       "(cpu %d, gpu %zu dets)\n",
                       r.stream_id, r.frame_no, r.verify_cpu_dets,
                       r.detections.size());
            }
        }

        if (!r.texts.empty()) plates_total += (int)r.texts.size();

        if (dump_dets_fp) {
            fprintf(dump_dets_fp, "F %d %d %zu\n", r.stream_id, r.frame_no,
                    r.detections.size());
            for (const Detection& d : r.detections) {
                fprintf(dump_dets_fp,
                        "%d %d %d %.9g %.9g %.9g %.9g %.9g\n", r.stream_id,
                        r.frame_no, d.cls, d.score, d.x, d.y, d.w, d.h);
            }
        }

        if (r.frame_no <= 2 || r.frame_no % 100 == 0) {
            std::string plates;
            if (!r.texts.empty()) {
                plates = " plates:";
                for (const std::string& t : r.texts) plates += " \"" + t + "\"";
            }
            printf("s%d frame[%d] batch=%d %zu dets%s\n", r.stream_id,
                   r.frame_no, r.batch_size, r.detections.size(),
                   plates.c_str());
        }
    }
    const double wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
            .count();

    // ---- Summary (same structure as the monolithic binary) ------------
    printf("\n---- per-stream ----\n");
    for (int i = 0; i < n_streams; i++) {
        const StreamInfo& si = pipeline->GetStreamInfo(i);
        StreamAgg& sa = agg[i];
        if (si.failed) {
            printf("s%d: FAILED to stream\n", i);
            continue;
        }
        double med = 0.0;
        if (!sa.pts_deltas.empty()) {
            auto mid = sa.pts_deltas.begin() + sa.pts_deltas.size() / 2;
            std::nth_element(sa.pts_deltas.begin(), mid, sa.pts_deltas.end());
            med = *mid / 1000.0;
        }
        char churn[48] = "";
        if (si.reconnects > 0)
            snprintf(churn, sizeof(churn), " | %d reconnects", si.reconnects);
        printf("s%d: %d inferred of %d decoded, avg latency %.2f ms "
               "(pop -> detections on host) | pts: %d non-monotonic, "
               "median delta %.1f ms %s%s\n",
               i, sa.frames, si.decoded,
               sa.frames ? sa.latency_ms_sum / sa.frames : 0.0,
               sa.pts_nonmono, med, sa.pts_nonmono == 0 ? "\xe2\x9c\x93" : "\xe2\x9c\x97",
               churn);
    }

    int grand_total_frames = 0;
    for (const auto& sa : agg) grand_total_frames += sa.frames;

    printf("---- batching ----\n");
    printf("latency split: %.2f ms pop->ready (preproc+sync), "
           "%.2f ms ready->take (queue+handoff), "
           "%.2f ms take->done (GPU batch) avg\n",
           grand_total_frames ? pre_gpu_ms / grand_total_frames : 0.0,
           grand_total_frames ? queue_ms / grand_total_frames : 0.0,
           batches ? gpu_ms / batches : 0.0);
    printf("%d batches over %d frames | batch-size histogram:", batches,
           grand_total_frames);
    for (int sz = 1; sz < (int)batch_hist.size(); sz++) {
        if (batch_hist[sz]) printf(" %d:%d", sz, batch_hist[sz]);
    }
    printf("\n");
    if (verify) {
        printf("verify: %d/%d frames matched CPU reference %s\n", verify_ok,
               verify_ok + verify_fail,
               verify_fail == 0 ? "\xe2\x9c\x93" : "\xe2\x9c\x97 MISMATCHES PRESENT");
    }
    if (!ocr_path.empty()) {
        printf("ocr: %d plates read over %d frames (%.2f per frame); "
               "stage-2 cost is included in the batch cycle above\n",
               plates_total, grand_total_frames,
               grand_total_frames ? (double)plates_total / grand_total_frames
                                   : 0.0);
    }
    int total_decoded = 0;
    bool any_failed = false;
    for (int i = 0; i < n_streams; i++) {
        const StreamInfo& si = pipeline->GetStreamInfo(i);
        total_decoded += si.decoded;
        any_failed = any_failed || si.failed;
    }
    printf("---- throughput ----\n");
    printf("%d inferred of %d decoded / %.1f s wall = %.1f inferred/s, "
           "%.1f decoded/s (%d streams x ~30 fps = %.0f offered, skip %d)\n",
           grand_total_frames, total_decoded, wall_s,
           grand_total_frames / wall_s, total_decoded / wall_s, n_streams,
           n_streams * 30.0, skip);

    pipeline.reset();  // Stop() + teardown (engines/decoder/ctx_lock before
                       // cuCtxDestroy - see Pipeline::~Pipeline)

    if (dump_dets_fp) fclose(dump_dets_fp);

    if (grand_total_frames > 0 && !any_failed) {
        printf("\xe2\x9c\x93 Success: multi-stream batched pipeline complete.\n");
        return 0;
    }
    return -1;
}
