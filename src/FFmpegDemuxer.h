#pragma once

// Pulls encoded packets (e.g. H.264 NAL units) out of an RTSP stream or video
// file using libavformat. This does NOT decode any pixels — FFmpeg is used
// here purely as a demuxer/network client. Actual pixel decode happens on
// GPU via NVDEC (see NvDecoder.h), keeping to the project's "no CPU decode"
// architecture decision.

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

#include <stdexcept>
#include <string>

class FFmpegDemuxer {
public:
    explicit FFmpegDemuxer(const std::string& url) {
        avformat_network_init();

        AVDictionary* opts = nullptr;
        // Force TCP transport for RTSP: avoids UDP packet loss/reordering,
        // which otherwise corrupts NAL units before they ever reach NVDEC.
        av_dict_set(&opts, "rtsp_transport", "tcp", 0);
        av_dict_set(&opts, "stimeout", "5000000", 0); // 5s open/read timeout (µs)

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
    }

    ~FFmpegDemuxer() {
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

        *data = packet_->data;
        *size = packet_->size;
        if (keyframe) *keyframe = (packet_->flags & AV_PKT_FLAG_KEY) != 0;
        if (pts_us) {
            const int64_t ts =
                packet_->pts != AV_NOPTS_VALUE ? packet_->pts : packet_->dts;
            *pts_us = ts == AV_NOPTS_VALUE
                          ? -1
                          : av_rescale_q(ts, time_base_, AVRational{1, 1000000});
        }
        return true;
    }

private:
    AVFormatContext* fmt_ctx_ = nullptr;
    AVPacket* packet_ = nullptr;
    int video_stream_index_ = -1;
    AVRational time_base_{0, 1};
    AVCodecID codec_id_ = AV_CODEC_ID_NONE;
    int width_ = 0;
    int height_ = 0;
};
