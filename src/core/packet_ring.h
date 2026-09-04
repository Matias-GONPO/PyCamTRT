#pragma once

// cordero::PacketRing - per-stream, host-only ring of ORIGINAL compressed
// video packets (Phase A1: endpoint capability, host-side tee sinks).
//
// WHY THIS EXISTS: RTSP packets already traverse host RAM (FFmpegDemuxer)
// before ever reaching NVDEC - see pipeline.cpp's ProducerLoop, which demuxes
// into a host-side AVPacket before Decode(). Forwarding the ORIGINAL video
// out a side channel (StreamRelay sink, ExtractClip) therefore needs NO
// re-encode and NO device-to-host copy: just retain the compressed packets
// that were already sitting in host memory for a moment on their way to the
// decoder. This type owns that retention - nothing here touches CUDA.
//
// Zero-CUDA on purpose (see pipeline.h's Pimpl WHY-comment for the same
// principle applied one layer up): a consumer of this ring (a relay thread,
// a clip-extraction call) never needs a CUDA context current to read it.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <chrono>
#include <mutex>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace cordero {

// One retained packet. `data` is an owned copy (see PacketRing::Push's
// WHY-comment on why the copy must happen immediately) - never a view into
// the demuxer's own packet storage.
struct RingPacket {
    std::vector<uint8_t> data;
    int64_t pts_us = -1;
    // Wall-clock arrival stamp (steady_clock us at Push) - the ring's
    // retention window trims on THIS, not pts (B3.5, campaign night 5):
    // B-frame sources deliver packets in DECODE order, so pts oscillates
    // per GOP (measured: 108 regressions in 299 packets on a real 4K
    // phone recording), making a pts-window trim fuzzy on such sources
    // and stall-prone on true PTS resets (reconnects, NVR relays).
    // Arrival time is monotonic by construction, immune to B-frames, PTS
    // resets, and NOPTS packets alike. pts_us stays for clip/relay
    // anchoring semantics. (Honesty note: the RSS growth that triggered
    // this hunt turned out to be glibc ARENA RETENTION under 4K-packet
    // churn, not ring content - instrumented counters showed the ring
    // bounded at ~40 MB throughout, and MALLOC_ARENA_MAX=1 flattened RSS.
    // The arrival-trim ships because it is strictly more robust, not
    // because the old trim was the measured leak.)
    int64_t arrival_us = -1;
    bool keyframe = false;
};

// Finds the index of the last keyframe at-or-before `cutoff_us`, so a caller
// (ExtractClip) can start a replay from a decodable point. Falls back to the
// EARLIEST keyframe in the snapshot if none qualifies (best-effort: the
// caller asked for more history than the ring currently holds - correctness
// over precision, same stance as TrimLocked below). Returns -1 only if the
// snapshot has no keyframe at all (nothing decodable can be produced from
// it - e.g. called before the stream's first IDR ever arrived).
inline int FindAnchor(const std::vector<RingPacket>& snap, int64_t cutoff_us) {
    int anchor = -1;
    int earliest_key = -1;
    for (size_t i = 0; i < snap.size(); i++) {
        if (!snap[i].keyframe) continue;
        if (earliest_key < 0) earliest_key = (int)i;
        if (snap[i].pts_us >= 0 && snap[i].pts_us <= cutoff_us) anchor = (int)i;
    }
    return anchor >= 0 ? anchor : earliest_key;
}

class PacketRing {
public:
    // ring_seconds <= 0 means "opt out" (see PipelineConfig::ring_seconds):
    // Push() becomes a no-op so a pipeline that wants neither relay nor clip
    // support pays no per-packet memcpy at all.
    explicit PacketRing(int ring_seconds) : ring_seconds_(ring_seconds) {}
    ~PacketRing() {
        if (codecpar_) avcodec_parameters_free(&codecpar_);
    }
    // Mutex member -> neither copyable nor movable (same reasoning as
    // Nv12Ring in pipeline.cpp: callers hold these in a std::deque, built
    // with emplace_back, one per stream).
    PacketRing(const PacketRing&) = delete;
    PacketRing& operator=(const PacketRing&) = delete;

    // Needed downstream for muxing (StreamRelay) / bitstream-format
    // detection (ExtractClip) without re-probing the stream. Called once per
    // demuxer (re)open - "(re)configure" includes every reconnect, so a
    // resolution/codec change on a churny camera is picked up automatically.
    // avcodec_parameters_copy deep-copies (extradata included), so this ring
    // owns an independent copy safe to read from another thread without any
    // lifetime tie to the demuxer that produced it.
    void SetCodecParameters(const AVCodecParameters* src) {
        std::lock_guard<std::mutex> lk(m_);
        if (codecpar_) avcodec_parameters_free(&codecpar_);
        codecpar_ = avcodec_parameters_alloc();
        avcodec_parameters_copy(codecpar_, src);
    }

    // Returns a fresh deep copy (caller-owned, avcodec_parameters_free it)
    // so a reader never has to hold the ring's lock while using it. nullptr
    // if SetCodecParameters has never been called (stream never opened).
    AVCodecParameters* CopyCodecParameters() const {
        std::lock_guard<std::mutex> lk(m_);
        if (!codecpar_) return nullptr;
        AVCodecParameters* out = avcodec_parameters_alloc();
        avcodec_parameters_copy(out, codecpar_);
        return out;
    }

    // Push one demuxed packet. `data`/`size` point into FFmpegDemuxer's own
    // internal AVPacket storage and are ONLY VALID UNTIL THE DEMUXER'S NEXT
    // Demux() CALL (see FFmpegDemuxer::Demux's doc comment) - this copies
    // immediately (before returning) so the ring owns bytes that stay valid
    // indefinitely, independent of the producer thread's next loop
    // iteration.
    void Push(const uint8_t* data, int size, int64_t pts_us, bool keyframe) {
        if (ring_seconds_ <= 0) return;  // opted out - see ctor WHY-comment
        std::lock_guard<std::mutex> lk(m_);
        RingPacket p;
        p.data.assign(data, data + size);
        p.pts_us = pts_us;
        p.arrival_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
        p.keyframe = keyframe;
        pkts_.push_back(std::move(p));
        TrimLocked();
    }

    // Full snapshot: deep-copies every retained packet out under the lock,
    // then returns - so a reader (relay/clip) does its actual work (muxing,
    // file IO, pacing sleeps) OUTSIDE any lock, never holding it long enough
    // to block the producer's next Push(). Cost is bounded by ring_seconds
    // of one stream's bitrate (a few MB at most) and is paid on the
    // READER's thread, never the producer's.
    std::vector<RingPacket> Snapshot() const {
        std::lock_guard<std::mutex> lk(m_);
        return std::vector<RingPacket>(pkts_.begin(), pkts_.end());
    }

    // Same snapshot contract, but bounded to packets newer than
    // `after_pts_us` (exclusive) instead of the whole ring - lets a live
    // follower (StreamRelay) re-copy only what it hasn't sent yet instead of
    // repeatedly re-snapshotting the full ~ring_seconds of history on every
    // poll cycle. Pass -1 for "everything" (equivalent to Snapshot()).
    std::vector<RingPacket> SnapshotSince(int64_t after_pts_us) const {
        std::lock_guard<std::mutex> lk(m_);
        if (after_pts_us < 0)
            return std::vector<RingPacket>(pkts_.begin(), pkts_.end());
        std::vector<RingPacket> out;
        for (const RingPacket& p : pkts_)
            if (p.pts_us > after_pts_us) out.push_back(p);
        return out;
    }

private:
    // Eviction, called with m_ held: trims packets older than ring_seconds_
    // (relative to the newest retained pts), but ALWAYS trims to the next
    // keyframe at-or-before the cutoff rather than the cutoff itself - so
    // the ring is always ANCHORED AT AN IDR and a consumer replaying from
    // ring-start is always decodable from its very first packet (a P-frame
    // is an edit of prior frames and cannot stand alone - see
    // FFmpegDemuxer's keyframe doc comment). If the current GOP (since the
    // last keyframe) already exceeds ring_seconds_, this keeps the WHOLE
    // GOP rather than cut mid-GOP: correctness (decodability) over precise
    // duration - a long-GOP stream simply retains a bit more than
    // ring_seconds implies. That's why this only ever finds the LATEST
    // keyframe at-or-before the cutoff and erases everything before it, and
    // never touches anything past it regardless of how long the resulting
    // ring ends up being.
    // Retention window on ARRIVAL time (see RingPacket::arrival_us's
    // WHY-comment - pts is decode-order-hostile). arrival_us is stamped by
    // Push itself, so it is always valid and strictly non-decreasing; the
    // keyframe rule is unchanged: keep from the last keyframe at-or-before
    // the cutoff so a snapshot always starts decodable.
    void TrimLocked() {
        if (pkts_.empty()) return;
        const int64_t cutoff =
            pkts_.back().arrival_us - (int64_t)ring_seconds_ * 1000000;
        size_t keep_from = 0;
        for (size_t i = 0; i < pkts_.size(); i++) {
            if (pkts_[i].arrival_us > cutoff) break;
            if (pkts_[i].keyframe) keep_from = i;
        }
        if (keep_from > 0) pkts_.erase(pkts_.begin(), pkts_.begin() + keep_from);
    }

    int ring_seconds_;
    mutable std::mutex m_;
    std::deque<RingPacket> pkts_;
    AVCodecParameters* codecpar_ = nullptr;
};

}  // namespace cordero
