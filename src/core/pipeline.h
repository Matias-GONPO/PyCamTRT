#pragma once

// pycamtrt::Pipeline - the multi-stream batched-inference orchestrator,
// extracted from rtsp_infer_multi.cpp's main() (Step 5) so it can be driven
// by more than one thin CLI/binding. Owns everything: CUDA context,
// engine(s), the greedy batcher, the NV12 rings, all device scratch, the
// producer threads and the dedicated GPU thread.
//
// Pimpl on purpose: consumers of this header (a future Python binding
// included) shouldn't need CUDA/TensorRT/FFmpeg headers on their include
// path just to hold a Pipeline and Poll() it.

#include <cstdint>
#include <memory>

#include "core/graph.h"
#include "core/result.h"

namespace pycamtrt {

class Pipeline {
public:
    // Validates the graph, loads engine(s), allocates every buffer and
    // warms up. Throws std::runtime_error with a clear message on any
    // failure (bad graph shape, missing engine file, unexpected engine
    // I/O shape, CUDA/allocation failure, ...).
    explicit Pipeline(PipelineConfig cfg);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Spawns the producer threads (one per stream) and the dedicated GPU
    // thread. Single-shot: throws if called twice, and throws if called
    // after Stop() (Stop() before Start() is itself fine - see Stop()).
    void Start();

    enum class PollStatus { Ok, Timeout, Finished };

    // Pops the next result from the bounded result queue (capacity
    // cfg.queue_capacity), waiting up to timeout_ms. Finished means the run
    // is complete and drained (naturally, or after Stop()) - no more
    // results will ever arrive.
    PollStatus Poll(FrameResult* out, int timeout_ms);

    // Safe to call from any thread, any number of times, concurrently, and
    // even before Start() (a clean no-op that still latches the queue as
    // finished, so a subsequent Poll() returns Finished rather than timing
    // out forever); also called by the destructor. Every caller - including
    // ones that lose the idempotency race - blocks until the shutdown
    // sequence has fully completed. See pipeline.cpp for the exact shutdown
    // sequence - it is designed to avoid the wait-cycles between
    // producers/batcher/GPU thread that a naive stop would hit.
    void Stop();

    const StreamInfo& GetStreamInfo(int i) const;

    // WHY: with cfg.backpressure == DropOldest the GPU thread never blocks
    // on a full result queue, so a consumer that falls behind gets no
    // signal from Poll() alone (it just silently stops seeing the oldest
    // results) - this lets it detect that it fell behind. Always 0 in Block
    // mode (nothing is ever dropped there).
    uint64_t DroppedResults() const;

    // Phase A1 (endpoint sinks): total lines ever dropped across every
    // Events sink (each has its own bounded drop-oldest queue - see
    // pipeline.cpp's EventsSink/LineQueue) - same telemetry role as
    // DroppedResults() above, one layer further downstream. Always 0 if no
    // Events sink is configured.
    uint64_t SinkDropped() const;

    // Phase A1: snapshot the given stream's PacketRing (cfg.ring_seconds >
    // 0 required - see PipelineConfig::ring_seconds), find the newest
    // keyframe at-or-before (newest_pts - seconds_back), and write from
    // there to the ring's end as a raw Annex-B .h264 elementary stream to
    // `path` (plain fwrite - RTSP/RTP H.264 depacketization already yields
    // Annex-B; falls back to the h264_mp4toannexb bitstream filter if the
    // retained codec parameters look AVCC instead, see pipeline.cpp). Safe
    // to call from any thread, any number of times, while the pipeline is
    // running (it reads a lock-guarded snapshot, never the live ring
    // directly). Returns false (logging why) if the stream_id is invalid,
    // the ring is disabled/empty, or no keyframe is available at all.
    bool ExtractClip(int stream_id, double seconds_back,
                      const std::string& path) const;

    int MaxBatch() const;
    int Classes() const;
    int Anchors() const;

    // Tier 2 (fetch): D2H copy of a hold_frames result's full-res NV12
    // frame into a caller-provided host buffer. `dst_host` must have room
    // for fr.frame_width * (fr.frame_height * 3 / 2) bytes, densely packed
    // (row stride == frame_width, no padding - unlike the device-side
    // pitch). Throws std::runtime_error if fr.frame_hold is null
    // (construct this Pipeline with cfg.hold_frames = true). Sets the
    // frame's owning CUDA context current on the calling thread first -
    // safe to call from any thread, including one that has never touched
    // CUDA before.
    void FetchFrame(const FrameResult& fr, uint8_t* dst_host) const;

    // Tier-1 info: raw device pointers (as integers) of the engine's
    // input/output bindings (see TrtEngine::InputPtr/OutputPtr) - stable
    // for the pipeline's lifetime (allocated once in Setup(), never
    // reallocated). Informational only, same caveat as FrameResult's
    // frame_addr: printable/loggable, not meaningful to dereference
    // outside this pipeline's CUDA context.
    uintptr_t InputBindingAddr() const;
    uintptr_t OutputBindingAddr() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pycamtrt
