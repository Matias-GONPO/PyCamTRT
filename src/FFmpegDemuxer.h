#pragma once

// Pulls encoded packets (e.g. H.264 NAL units) out of an RTSP stream or video
// file using libavformat. This does NOT decode any pixels — FFmpeg is used
// here purely as a demuxer/network client. Actual pixel decode happens on
// GPU via NVDEC (see NvDecoder.h), keeping to the project's "no CPU decode"
// architecture decision.

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
}

#include <stdexcept>
#include <string>

class FFmpegDemuxer {
public:
    explicit FFmpegDemuxer(const std::string& url) {
        avformat_network_init();

        const bool is_rtsp = url.rfind("rtsp://", 0) == 0;

        AVDictionary* opts = nullptr;
        if (is_rtsp) {
            // Force TCP transport for RTSP: avoids UDP packet loss/
            // reordering, which otherwise corrupts NAL units before they
            // ever reach NVDEC. These options are RTSP-specific (libavformat
            // ignores unknown-for-this-demuxer keys, but there's no reason
            // to hand file inputs noise that doesn't apply to them).
            av_dict_set(&opts, "rtsp_transport", "tcp", 0);
            av_dict_set(&opts, "stimeout", "5000000", 0); // 5s open/read timeout (µs)
        }

        if (avformat_open_input(&fmt_ctx_, url.c_str(), nullptr, &opts) != 0) {
            av_dict_free(&opts);
            throw std::runtime_error("Failed to open input: " + url);
        }
        av_dict_free(&opts);

        if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
            throw std::runtime_error("Failed to find stream info for: " + url);
        }

        video_stream_index_ = av_find_best_stream(
            fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_index_ < 0) {
            throw std::runtime_error("No video stream found in: " + url);
        }

        AVCodecParameters* params = fmt_ctx_->streams[video_stream_index_]->codecpar;
        codec_id_ = params->codec_id; // e.g. AV_CODEC_ID_H264, AV_CODEC_ID_HEVC
        width_ = params->width;
        height_ = params->height;
        time_base_ = fmt_ctx_->streams[video_stream_index_]->time_base;

        packet_ = av_packet_alloc();

        // MP4/MOV (and other AVCC-framed containers) hand out H.264/HEVC
        // packets as 4-byte-length-prefixed NALs with SPS/PPS carried only
        // in codecpar->extradata (an avcC/hvcC box) - never inline in the
        // packet stream. cuvid's parser requires Annex-B (start-code-
        // delimited NALs, SPS/PPS discoverable via a start code): fed AVCC
        // bytes unmodified, it silently parses nothing and NvDecoder never
        // decodes a single frame (see the research project's FINDINGS ledger's "MP4
        // file input fails" entry - root-caused there). RTSP's RTP/H.264
        // depacketizer already reconstructs Annex-B (start codes + SPS/PPS
        // from the SDP or inline), so it never hits this and is left alone
        // entirely (no bsf attached, regardless of what its extradata looks
        // like) - only non-rtsp:// sources are even considered here.
        if (!is_rtsp && (codec_id_ == AV_CODEC_ID_H264 ||
                         codec_id_ == AV_CODEC_ID_HEVC) &&
            IsAvccExtradata(params)) {
            const char* bsf_name = codec_id_ == AV_CODEC_ID_H264
                                        ? "h264_mp4toannexb"
                                        : "hevc_mp4toannexb";
            const AVBitStreamFilter* bsf = av_bsf_get_by_name(bsf_name);
            if (!bsf) {
                throw std::runtime_error(
                    std::string("bitstream filter not found: ") + bsf_name);
            }
            if (av_bsf_alloc(bsf, &bsf_ctx_) < 0) {
                throw std::runtime_error(
                    std::string("av_bsf_alloc failed for ") + bsf_name);
            }
            if (avcodec_parameters_copy(bsf_ctx_->par_in, params) < 0) {
                throw std::runtime_error(
                    "avcodec_parameters_copy failed (bsf input params)");
            }
            bsf_ctx_->time_base_in = time_base_;
            if (av_bsf_init(bsf_ctx_) < 0) {
                throw std::runtime_error(
                    std::string("av_bsf_init failed for ") + bsf_name);
            }
            bsf_packet_ = av_packet_alloc();
        }
    }

    ~FFmpegDemuxer() {
        if (bsf_packet_) av_packet_free(&bsf_packet_);
        if (bsf_ctx_) av_bsf_free(&bsf_ctx_);
        if (packet_) av_packet_free(&packet_);
        if (fmt_ctx_) avformat_close_input(&fmt_ctx_);
    }

    FFmpegDemuxer(const FFmpegDemuxer&) = delete;
    FFmpegDemuxer& operator=(const FFmpegDemuxer&) = delete;

    AVCodecID GetCodecID() const { return codec_id_; }
    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }

    // Phase A1 (endpoint sinks): the video stream's codec parameters
    // (profile/level, extradata/SPS-PPS, dimensions, ...), needed downstream
    // for muxing without re-probing (see core/packet_ring.h's PacketRing,
    // which deep-copies this via avcodec_parameters_copy at (re)configure -
    // this pointer itself is only valid for this demuxer's lifetime, same
    // as fmt_ctx_).
    //
    // NOTE: this always reflects the ORIGINAL container's codecpar, even
    // when the bsf below is active - for an AVCC file source that means
    // this still reports avcC-style extradata even though Demux() now
    // yields Annex-B packets. PacketRing/StreamRelay/ExtractClip currently
    // assume Annex-B end-to-end (true for RTSP, and now true for the
    // packets themselves on a converted file source too), but a consumer
    // that inspects THIS extradata directly and expects it to match the
    // packet framing would see a mismatch on file sources. Not exercised by
    // any current sink; flagged for awareness, not fixed here.
    const AVCodecParameters* GetCodecParameters() const {
        return fmt_ctx_->streams[video_stream_index_]->codecpar;
    }

    // Reads the next encoded packet belonging to the video stream, skipping
    // any other streams (audio, etc). Returns false at end of stream / on
    // read error. `data`/`size` point into internal packet storage and are
    // only valid until the next call to Demux().
    //
    // `pts_us` (optional) receives the packet's presentation timestamp in
    // microseconds (stream time_base rescaled), or -1 if the container
    // provides none. RTSP timestamps start at an arbitrary offset — only
    // deltas between frames are meaningful.
    //
    // `keyframe` (optional) receives whether this packet is a keyframe
    // (IDR): a self-contained picture decodable with no reference frames.
    // Keyframe-only decode mode drops every other packet BEFORE the
    // decoder, cutting NVDEC load to one frame per GOP per stream.
    bool Demux(uint8_t** data, int* size, int64_t* pts_us = nullptr,
               bool* keyframe = nullptr) {
        if (bsf_ctx_) return DemuxThroughBsf(data, size, pts_us, keyframe);

        av_packet_unref(packet_);

        int ret;
        do {
            ret = av_read_frame(fmt_ctx_, packet_);
        } while (ret >= 0 && packet_->stream_index != video_stream_index_);

        if (ret < 0) {
            *data = nullptr;
            *size = 0;
            return false;
        }

        FillOut(packet_, data, size, pts_us, keyframe);
        return true;
    }

private:
    // Heuristic per the recorded root-cause recipe: AVCC extradata is an
    // avcC/hvcC box (starts with a version byte, e.g. 0x01 for avcC), never
    // an Annex-B start code (0x00 0x00 0x01 or 0x00 0x00 0x00 0x01). No
    // extradata at all (e.g. a raw .h264 elementary stream with inline
    // SPS/PPS and no container-level extradata) is left alone - nothing to
    // convert, and Demux() already works for that case today.
    static bool IsAvccExtradata(const AVCodecParameters* params) {
        if (!params->extradata || params->extradata_size < 4) return false;
        const uint8_t* e = params->extradata;
        const bool annexb_short = e[0] == 0x00 && e[1] == 0x00 && e[2] == 0x01;
        const bool annexb_long =
            e[0] == 0x00 && e[1] == 0x00 && e[2] == 0x00 && e[3] == 0x01;
        return !annexb_short && !annexb_long;
    }

    void FillOut(const AVPacket* pkt, uint8_t** data, int* size,
                 int64_t* pts_us, bool* keyframe) const {
        *data = pkt->data;
        *size = pkt->size;
        if (keyframe) *keyframe = (pkt->flags & AV_PKT_FLAG_KEY) != 0;
        if (pts_us) {
            const int64_t ts =
                pkt->pts != AV_NOPTS_VALUE ? pkt->pts : pkt->dts;
            *pts_us = ts == AV_NOPTS_VALUE
                          ? -1
                          : av_rescale_q(ts, time_base_, AVRational{1, 1000000});
        }
    }

    // Drives the standard bsf send/receive loop. h264_mp4toannexb/
    // hevc_mp4toannexb are 1-in-1-out in practice (they just rewrite NAL
    // framing + inline the extradata's SPS/PPS ahead of the first packet),
    // but the API doesn't document that as a guarantee, so this doesn't
    // assume it: it loops on EAGAIN, feeding fresh raw packets in, until
    // the bsf actually yields one or the raw stream ends. `packet_` is
    // reused as the raw-read buffer exactly as in the non-bsf path;
    // `bsf_packet_` is the one whose lifetime is exposed to the caller,
    // unref'd only at the TOP of this function (i.e. right before a new
    // one replaces it) so its data stays valid until the next Demux() call,
    // matching the documented contract.
    bool DemuxThroughBsf(uint8_t** data, int* size, int64_t* pts_us,
                          bool* keyframe) {
        av_packet_unref(bsf_packet_);

        int ret = av_bsf_receive_packet(bsf_ctx_, bsf_packet_);
        while (ret == AVERROR(EAGAIN)) {
            av_packet_unref(packet_);
            int rret;
            do {
                rret = av_read_frame(fmt_ctx_, packet_);
            } while (rret >= 0 && packet_->stream_index != video_stream_index_);

            if (rret < 0) {
                // Raw EOF: signal it to the bsf (NULL send is the documented
                // drain/flush trigger) and take whatever it still has
                // buffered, if anything.
                av_bsf_send_packet(bsf_ctx_, nullptr);
                ret = av_bsf_receive_packet(bsf_ctx_, bsf_packet_);
                break;
            }
            if (av_bsf_send_packet(bsf_ctx_, packet_) < 0) {
                *data = nullptr;
                *size = 0;
                return false;
            }
            ret = av_bsf_receive_packet(bsf_ctx_, bsf_packet_);
        }

        if (ret < 0) {  // AVERROR_EOF, or a real error - both are "no packet"
            *data = nullptr;
            *size = 0;
            return false;
        }

        // h264_mp4toannexb/hevc_mp4toannexb only rewrite NAL framing and
        // where SPS/PPS live - they pass pts/dts/flags (including the
        // keyframe flag) through from the input packet unchanged, so the
        // same extraction used for the non-bsf path applies here unchanged.
        FillOut(bsf_packet_, data, size, pts_us, keyframe);
        return true;
    }

    AVFormatContext* fmt_ctx_ = nullptr;
    AVPacket* packet_ = nullptr;
    int video_stream_index_ = -1;
    AVRational time_base_{0, 1};
    AVCodecID codec_id_ = AV_CODEC_ID_NONE;
    int width_ = 0;
    int height_ = 0;

    // Only allocated for AVCC-framed non-rtsp:// sources (see ctor); null
    // otherwise, which is what Demux() branches on.
    AVBSFContext* bsf_ctx_ = nullptr;
    AVPacket* bsf_packet_ = nullptr;
};
