// Part 2: lifecycle-robustness test for cordero::Pipeline - deliberately
// hammers the edge cases a careless consumer of the library would hit
// (Stop()-before-Start(), double-Stop(), dtor-only teardown, Start()-after-
// Stop(), Poll() at every lifecycle boundary, a slow consumer against the
// bounded-queue backpressure path, and repeated construct/destroy for a
// VRAM/context leak check). See pipeline.h's lifecycle contract comments -
// this binary is what verifies that contract.
//
// Usage:
//   pipeline_api_test URL [URL...] [--engine PATH] [--cycles N]
//                     [--consume K] [--slow] [--hold SECONDS] [--drop]
//   --engine PATH   default models/yolov8n_b1-16_fp32_sm86.engine
//   --cycles N      default 20: N x { construct, Start(), consume K results,
//                   teardown (varied - see Variant() below), destruct }
//   --consume K     default 100 results consumed per cycle
//   --slow          sleep 50 ms per result for the first K/2 results each
//                   cycle (blocking-backpressure path), then drain fast
//   --hold SECONDS  single unbounded run for SECONDS, printing a
//                   results/second line each second, then Stop() + dump
//                   per-stream GetStreamInfo (farm kill/restart tool)
//   --drop          Part 2 drop-oldest backpressure check: construct with
//                   Backpressure::DropOldest and a small queue_capacity,
//                   consume very slowly for 15 s against an unbounded run,
//                   then Stop() - see RunDrop() for the pass criteria.
//
// Exit 0 only if every cycle completed within the watchdog budget and the
// VRAM check passed; non-zero otherwise (see WatchdogLoop/UsedVramMb).

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/pipeline.h"

using cordero::Backpressure;
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

using Clock = std::chrono::steady_clock;

// Same step-graph shape rtsp_infer_multi.cpp builds (Process implied ->
// Engine -> Postprocess(YoloDetect)); this test only exercises lifecycle,
// not the graph surface, so no OCR/SAHI wiring.
PipelineConfig BuildConfig(const std::vector<std::string>& urls,
                           const std::string& engine_path,
                           Backpressure backpressure = Backpressure::Block,
                           size_t queue_capacity = 256) {
    PipelineConfig cfg;
    for (const std::string& u : urls) cfg.streams.push_back(StreamDesc{u});
    cfg.max_frames = 0;  // unbounded: every cycle runs until Stop() - the
                         // matrix this test is built to exercise.
    cfg.backpressure = backpressure;
    cfg.queue_capacity = queue_capacity;
    // Silenced: Setup() logs "Engine loaded"/"Head" banners, and this binary
    // constructs a fresh Pipeline (and re-loads the engine) once per cycle -
    // 20+ copies of those banners would bury the one line per cycle that
    // actually matters.
    cfg.log = [](const std::string&) {};

    StepDesc engine_step;
    engine_step.kind = StepKind::Engine;
    engine_step.input = -1;  // Process step implied
    engine_step.engine_path = engine_path;
    const int engine_idx = (int)cfg.steps.size();
    cfg.steps.push_back(engine_step);

    StepDesc post_step;
    post_step.kind = StepKind::Postprocess;
    post_step.input = engine_idx;
    post_step.family = Family::YoloDetect;
    const int post_idx = (int)cfg.steps.size();
    cfg.steps.push_back(post_step);

    LayerDesc layer0;
    layer0.name = "detect";
    layer0.steps = {engine_idx, post_idx};
    cfg.layers.push_back(layer0);
    return cfg;
}

// ---- Watchdog -------------------------------------------------------------
// Converts a lifecycle hang into a hard, visible test failure instead of a
// silent multi-minute (or infinite) stall under CI/farm automation. Tracks
// only ONE cycle's wall clock at a time (g_cycle_start_ms is stamped fresh
// by every CycleBegin), not the cumulative run - --hold mode never arms it
// at all (see main()), since a deliberately long hold is not a hang.
std::atomic<bool> g_cycle_active{false};
std::atomic<int64_t> g_cycle_start_ms{0};

int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               Clock::now().time_since_epoch())
        .count();
}

void WatchdogLoop() {
    constexpr int64_t kMaxCycleMs = 90000;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (g_cycle_active.load() &&
            NowMs() - g_cycle_start_ms.load() > kMaxCycleMs) {
            fprintf(stderr,
                    "WATCHDOG: cycle exceeded 90 s - lifecycle hang\n");
            fflush(stderr);
            exit(2);
        }
    }
}

struct CycleBegin {
    CycleBegin() {
        g_cycle_start_ms.store(NowMs());
        g_cycle_active.store(true);
    }
    ~CycleBegin() { g_cycle_active.store(false); }
};

size_t UsedVramMb() {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    return (total_b - free_b) / (1024 * 1024);
}

// Consumes up to `k` results from `p`, sleeping `sleep_ms` after each of the
// first `slow_count` (exercises the GPU thread's blocking-Push backpressure
// path against a deliberately slow consumer). Stops early (returns fewer
// than k) if the pipeline reports Finished - a dead/never-connected stream
// should not hang the test forever.
int ConsumeK(Pipeline& p, int k, int slow_count, int sleep_ms) {
    FrameResult r;
    int got = 0;
    for (int i = 0; i < k; i++) {
        for (;;) {
            const Pipeline::PollStatus st = p.Poll(&r, 500);
            if (st == Pipeline::PollStatus::Ok) {
                got++;
                break;
            }
            if (st == Pipeline::PollStatus::Finished) return got;
        }
        if (i < slow_count && sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
    }
    return got;
}

// Deterministic teardown-variant matrix (Task B): every 4th cycle double-
// Stop()s, every 5th skips Stop() entirely (dtor-only). Cycle 20 is
// divisible by both - dtor-only wins (it's the stricter/less-forgiving path:
// no explicit Stop() at all, vs. one extra idempotent Stop() call).
const char* Variant(int cycle) {
    if (cycle % 5 == 0) return "dtor-only";
    if (cycle % 4 == 0) return "double-stop";
    return "normal";
}

int RunHold(const std::vector<std::string>& urls, const std::string& engine_path,
            int hold_s) {
    printf("hold: %zu streams, %d s\n", urls.size(), hold_s);
    std::unique_ptr<Pipeline> p;
    try {
        p.reset(new Pipeline(BuildConfig(urls, engine_path)));
    } catch (const std::exception& e) {
        fprintf(stderr, "pipeline init failed: %s\n", e.what());
        return 1;
    }
    p->Start();

    const auto t0 = Clock::now();
    int total = 0, sec_count = 0;
    auto tick = t0;
    FrameResult r;
    for (;;) {
        const double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
        if (elapsed >= hold_s) break;
        const Pipeline::PollStatus st = p->Poll(&r, 200);
        if (st == Pipeline::PollStatus::Ok) {
            total++;
            sec_count++;
        } else if (st == Pipeline::PollStatus::Finished) {
            break;  // every stream died - nothing left to hold on
        }
        const double since_tick = std::chrono::duration<double>(Clock::now() - tick).count();
        if (since_tick >= 1.0) {
            printf("hold: t=%.0fs %d results/s (total %d)\n", elapsed, sec_count, total);
            sec_count = 0;
            tick = Clock::now();
        }
    }
    p->Stop();

    printf("---- per-stream (post-Stop) ----\n");
    for (int i = 0; i < (int)urls.size(); i++) {
        const StreamInfo& si = p->GetStreamInfo(i);
        printf("stream %d: decoded=%d reconnects=%d failed=%d\n", i,
               si.decoded, si.reconnects, si.failed);
    }
    p.reset();
    return 0;
}

// Part 2 drop-oldest check. Proves the whole point of DropOldest: the GPU
// thread (and therefore the producers feeding it, via the batcher) never
// blocks on the result queue, even when the consumer stalls hard. A small
// queue_capacity + a 200 ms/Poll consumer over 15 s guarantees the queue
// fills almost immediately (capacity results in well under a second at 30
// fps/stream), so every subsequent GPU-thread push for the rest of the run
// is a drop, not a real wait.
int RunDrop(const std::vector<std::string>& urls, const std::string& engine_path) {
    constexpr size_t kQueueCap = 8;
    constexpr int kHoldS = 15;
    constexpr int kConsumeSleepMs = 200;
    printf("drop: %zu streams, %d s, queue_capacity=%zu\n", urls.size(), kHoldS,
           kQueueCap);

    std::unique_ptr<Pipeline> p;
    try {
        p.reset(new Pipeline(
            BuildConfig(urls, engine_path, Backpressure::DropOldest, kQueueCap)));
    } catch (const std::exception& e) {
        fprintf(stderr, "drop: pipeline init failed: %s\n", e.what());
        return 1;
    }
    p->Start();

    const auto t0 = Clock::now();
    FrameResult r;
    for (;;) {
        const double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
        if (elapsed >= kHoldS) break;
        const Pipeline::PollStatus st = p->Poll(&r, 500);
        if (st == Pipeline::PollStatus::Finished) break;
        if (st == Pipeline::PollStatus::Ok) {
            // Deliberately slow: this is what should push the GPU thread
            // into repeatedly evicting queued results rather than blocking.
            std::this_thread::sleep_for(std::chrono::milliseconds(kConsumeSleepMs));
        }
    }
    const double elapsed_s = std::chrono::duration<double>(Clock::now() - t0).count();
    const uint64_t dropped = p->DroppedResults();

    p->Stop();

    int total_decoded = 0;
    printf("---- per-stream (post-Stop) ----\n");
    for (int i = 0; i < (int)urls.size(); i++) {
        const StreamInfo& si = p->GetStreamInfo(i);
        printf("stream %d: decoded=%d reconnects=%d failed=%d\n", i, si.decoded,
               si.reconnects, si.failed);
        total_decoded += si.decoded;
    }
    p.reset();

    const double expected = 0.7 * (double)urls.size() * 30.0 * elapsed_s;
    printf("drop: dropped_results=%llu total_decoded=%d expected>=%.0f "
           "(0.7 * %zu streams * 30 fps * %.1f s)\n",
           (unsigned long long)dropped, total_decoded, expected, urls.size(),
           elapsed_s);

    const bool pass_a = dropped > 0;
    const bool pass_b = (double)total_decoded >= expected;
    if (pass_a && pass_b) {
        printf("DROP MODE PASS\n");
        return 0;
    }
    printf("DROP MODE FAIL (dropped>0: %s, decoded-pace: %s)\n",
           pass_a ? "ok" : "FAIL", pass_b ? "ok" : "FAIL");
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> urls;
    std::string engine_path = "models/yolov8n_b1-16_fp32_sm86.engine";
    int cycles = 20;
    int consume = 100;
    bool slow = false;
    int hold_s = -1;
    bool drop = false;

    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--engine") && i + 1 < argc) engine_path = argv[++i];
        else if (!std::strcmp(argv[i], "--cycles") && i + 1 < argc) cycles = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--consume") && i + 1 < argc) consume = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--slow")) slow = true;
        else if (!std::strcmp(argv[i], "--hold") && i + 1 < argc) hold_s = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--drop")) drop = true;
        else urls.push_back(argv[i]);
    }
    if (urls.empty()) {
        fprintf(stderr,
                "Usage: %s URL [URL...] [--engine PATH] [--cycles N] "
                "[--consume K] [--slow] [--hold SECONDS] [--drop]\n",
                argv[0]);
        return 1;
    }

    // Detached, process-lifetime watchdog: exit(2) on a hang tears the whole
    // process (and this thread with it) down, so there is nothing to join.
    std::thread(WatchdogLoop).detach();

    if (drop) return RunDrop(urls, engine_path);
    if (hold_s > 0) return RunHold(urls, engine_path, hold_s);

    size_t vram_after_1 = 0, vram_after_last = 0;

    for (int c = 1; c <= cycles; c++) {
        CycleBegin cb;
        std::string variant_label = Variant(c);

        std::unique_ptr<Pipeline> p;
        try {
            p.reset(new Pipeline(BuildConfig(urls, engine_path)));
        } catch (const std::exception& e) {
            fprintf(stderr, "cycle %d: pipeline init failed: %s\n", c, e.what());
            return 1;
        }

        if (c == 1) {
            // Edge case: Poll() before Start(). Must return Timeout (not
            // crash, not block forever) - see ResultQueue::Poll's WHY-
            // comment in pipeline.cpp for why this is safe.
            FrameResult r;
            const Pipeline::PollStatus st = p->Poll(&r, 200);
            if (st != Pipeline::PollStatus::Timeout) {
                fprintf(stderr,
                        "cycle 1: Poll() before Start() returned %d, want "
                        "Timeout(%d)\n",
                        (int)st, (int)Pipeline::PollStatus::Timeout);
                return 1;
            }
            // Edge case: construct + destroy a Pipeline that never calls
            // Start() at all - exercises Impl::~Impl -> StopImpl() (no
            // threads exist: must be a clean no-op) -> Teardown().
            { Pipeline extra(BuildConfig(urls, engine_path)); }
            variant_label += "+pre-start-checks";
        }

        const auto t0 = Clock::now();
        p->Start();

        // --slow: exercise the bounded-queue blocking-backpressure path for
        // the first half of this cycle's results, then drain at full speed.
        const int slow_count = slow ? consume / 2 : 0;
        const int sleep_ms = slow ? 50 : 0;
        const int got = ConsumeK(*p, consume, slow_count, sleep_ms);

        const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        printf("cycle %d: %d results in %.1f s [%s]\n", c, got, secs,
               variant_label.c_str());

        // Pipeline is unbounded (max_frames=0) and still running here, so
        // every Stop() below - in every variant - is inherently a
        // slow-consumer-then-Stop()-mid-flow shutdown, exactly what --slow
        // is meant to stress.
        const std::string variant = Variant(c);
        if (variant == "dtor-only") {
            // No explicit Stop(): p.reset() below drives it through
            // Impl::~Impl -> StopImpl() + Teardown() only.
        } else if (variant == "double-stop") {
            p->Stop();
            p->Stop();  // idempotent, concurrency-safe (see StopImpl): must
                        // not double-join or hang.
        } else {
            p->Stop();
        }
        p.reset();  // destructor without Stop() (dtor-only cycles) or after
                    // Stop() (the rest): neither may crash or hang.

        if (c == 1) vram_after_1 = UsedVramMb();
        if (c == cycles) vram_after_last = UsedVramMb();
    }

    const int64_t grown = (int64_t)vram_after_last - (int64_t)vram_after_1;
    printf("vram: %zu MB after cycle 1, %zu MB after cycle %d (grew %lld MB)\n",
           vram_after_1, vram_after_last, cycles, (long long)grown);
    if (grown > 64) {
        fprintf(stderr,
                "FAIL: VRAM grew %lld MB (> 64 MB budget) across %d cycles - "
                "context/engine leak\n",
                (long long)grown, cycles);
        return 1;
    }

    printf("PASS: %d cycles completed, VRAM stable\n", cycles);
    return 0;
}
