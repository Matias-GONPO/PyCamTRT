#include "NvDecoder.h"

#include <cstdlib>

#include <cstring>
#include <iostream>

namespace {
void CheckCu(CUresult result, const char* what) {
    if (result != CUDA_SUCCESS) {
        const char* err_name = nullptr;
        cuGetErrorName(result, &err_name);
        throw std::runtime_error(std::string("CUDA driver call failed: ") + what +
                                  " (" + (err_name ? err_name : "unknown") + ")");
    }
}
}  // namespace

CUvideoctxlock NvDecoder::CreateContextLock(CUcontext cu_context) {
    CUvideoctxlock ctx_lock = nullptr;
    CheckCu(cuvidCtxLockCreate(&ctx_lock, cu_context), "cuvidCtxLockCreate");
    return ctx_lock;
}

void NvDecoder::DestroyContextLock(CUvideoctxlock ctx_lock) {
    if (ctx_lock) cuvidCtxLockDestroy(ctx_lock);
}

NvDecoder::NvDecoder(CUvideoctxlock ctx_lock, cudaVideoCodec codec,
                     CUstream output_stream)
    : ctx_lock_(ctx_lock), codec_type_(codec), output_stream_(output_stream) {
    CUVIDPARSERPARAMS parser_params = {};
    parser_params.CodecType = codec_type_;
    parser_params.ulMaxNumDecodeSurfaces = 8;   // needs tuning once real GOP/latency reqs known
    // Display delay 0 = emit each picture as soon as it is decoded (lowest
    // latency). Measured 2026-09-19: letting the parser run one picture
    // ahead per stream buys no throughput at the NVDEC wall and costs one
    // frame period of latency, so 0 stays.
    parser_params.ulClockRate = 0;              // use packet timestamps directly
    parser_params.pUserData = this;
    parser_params.pfnSequenceCallback = HandleVideoSequenceProc;
    parser_params.pfnDecodePicture = HandlePictureDecodeProc;
    parser_params.pfnDisplayPicture = HandlePictureDisplayProc;

    CheckCu(cuvidCreateVideoParser(&parser_, &parser_params), "cuvidCreateVideoParser");
}

NvDecoder::~NvDecoder() {
    if (parser_) cuvidDestroyVideoParser(parser_);
    if (decoder_) cuvidDestroyDecoder(decoder_);
    // ctx_lock_ is not owned by this instance - may be shared with other
    // NvDecoders on the same CUDA context. Caller destroys it via
    // DestroyContextLock() once all decoders using it are gone.
}

void NvDecoder::Decode(const uint8_t* data, int size, int64_t timestamp) {
    CUVIDSOURCEDATAPACKET packet = {};
    packet.payload = data;
    packet.payload_size = static_cast<unsigned long>(size);
    packet.timestamp = timestamp;
    packet.flags = CUVID_PKT_TIMESTAMP;
    if (data == nullptr || size == 0) {
        packet.flags |= CUVID_PKT_ENDOFSTREAM;
    } else {
        // Every packet the demuxer hands us is one whole access unit
        // (av_read_frame contract for video), so tell the parser the
        // picture is complete. Without this flag the parser only learns
        // that when the NEXT packet's start code arrives, i.e. one frame
        // period later at a live camera: measured 2026-09-19, demuxer-to-
        // pop fell from 53 to 19 ms at 1080p/24 cameras and from 46 to
        // 10 ms at 720p/48 with no change in throughput or detections.
        packet.flags |= CUVID_PKT_ENDOFPICTURE;
    }

    CheckCu(cuvidParseVideoData(parser_, &packet), "cuvidParseVideoData");
}

void NvDecoder::Flush() {
    Decode(nullptr, 0);
}

bool NvDecoder::PopFrame(DecodedFrame* out) {
    std::lock_guard<std::mutex> lock(frame_queue_mutex_);
    if (ready_frames_.empty()) return false;
    *out = ready_frames_.front();
    ready_frames_.pop_front();
    return true;
}

void NvDecoder::ReleaseFrame(const DecodedFrame& frame) {
    cuvidCtxLock(ctx_lock_, 0);
    cuvidUnmapVideoFrame64(decoder_, frame.device_ptr);
    cuvidCtxUnlock(ctx_lock_, 0);
}

// --- static trampolines ---

int CUDAAPI NvDecoder::HandleVideoSequenceProc(void* user_data, CUVIDEOFORMAT* format) {
    return reinterpret_cast<NvDecoder*>(user_data)->HandleVideoSequence(format);
}

int CUDAAPI NvDecoder::HandlePictureDecodeProc(void* user_data, CUVIDPICPARAMS* pic_params) {
    return reinterpret_cast<NvDecoder*>(user_data)->HandlePictureDecode(pic_params);
}

int CUDAAPI NvDecoder::HandlePictureDisplayProc(void* user_data, CUVIDPARSERDISPINFO* disp_info) {
    return reinterpret_cast<NvDecoder*>(user_data)->HandlePictureDisplay(disp_info);
}

// --- instance methods ---

int NvDecoder::HandleVideoSequence(CUVIDEOFORMAT* format) {
    width_ = format->display_area.right - format->display_area.left;
    height_ = format->display_area.bottom - format->display_area.top;
    num_decode_surfaces_ = format->min_num_decode_surfaces;

    if (decoder_) {
        // Stream parameters changed mid-session (rare for a static test
        // camera) - not handled yet, would need cuvidReconfigureDecoder.
        return num_decode_surfaces_;
    }

    CUVIDDECODECREATEINFO create_info = {};
    create_info.CodecType = codec_type_;
    create_info.ChromaFormat = format->chroma_format;
    create_info.OutputFormat = cudaVideoSurfaceFormat_NV12;
    create_info.bitDepthMinus8 = format->bit_depth_luma_minus8;
    create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    create_info.ulNumDecodeSurfaces = num_decode_surfaces_;
    create_info.ulNumOutputSurfaces = 2;
    create_info.ulWidth = format->coded_width;
    create_info.ulHeight = format->coded_height;
    create_info.ulTargetWidth = width_;
    create_info.ulTargetHeight = height_;
    create_info.display_area.left = format->display_area.left;
    create_info.display_area.top = format->display_area.top;
    create_info.display_area.right = format->display_area.right;
    create_info.display_area.bottom = format->display_area.bottom;
    create_info.target_rect.left = 0;
    create_info.target_rect.top = 0;
    create_info.target_rect.right = width_;
    create_info.target_rect.bottom = height_;
    create_info.vidLock = ctx_lock_;

    CheckCu(cuvidCreateDecoder(&decoder_, &create_info), "cuvidCreateDecoder");
    return num_decode_surfaces_;
}

int NvDecoder::HandlePictureDecode(CUVIDPICPARAMS* pic_params) {
    cuvidCtxLock(ctx_lock_, 0);
    CUresult result = cuvidDecodePicture(decoder_, pic_params);
    cuvidCtxUnlock(ctx_lock_, 0);
    return result == CUDA_SUCCESS ? 1 : 0;
}

int NvDecoder::HandlePictureDisplay(CUVIDPARSERDISPINFO* disp_info) {
    CUVIDPROCPARAMS proc_params = {};
    proc_params.progressive_frame = disp_info->progressive_frame;
    proc_params.top_field_first = disp_info->top_field_first;
    proc_params.unpaired_field = disp_info->repeat_first_field < 0;
    // Run the map's post-processing on the caller's stream instead of the
    // legacy default stream (0), which synchronizes with every blocking
    // stream in the context - i.e. with the TensorRT batch in flight.
    // NVIDIA's NvDecoder sample sets this too. Before this line every map
    // could wait for the running inference batch, and because the letterbox
    // kernel on a NON-blocking stream never waited for stream 0, it could
    // read a surface still being written (non-reproducible detections).
    proc_params.output_stream = output_stream_;

    CUdeviceptr device_ptr = 0;
    unsigned int pitch = 0;

    cuvidCtxLock(ctx_lock_, 0);
    CUresult result = cuvidMapVideoFrame64(
        decoder_, disp_info->picture_index, &device_ptr, &pitch, &proc_params);
    cuvidCtxUnlock(ctx_lock_, 0);

    if (result != CUDA_SUCCESS) {
        return 0;
    }

    DecodedFrame frame;
    frame.device_ptr = device_ptr;
    frame.pitch = pitch;
    frame.width = width_;
    frame.height = height_;
    frame.timestamp = disp_info->timestamp;
    frame.picture_index = disp_info->picture_index;

    // Held mapped until the caller calls ReleaseFrame() - keeps this a
    // zero-copy handoff, but means ulNumDecodeSurfaces must stay ahead of
    // however many frames are in flight downstream at once.
    std::lock_guard<std::mutex> lock(frame_queue_mutex_);
    ready_frames_.push_back(frame);
    return 1;
}
