#pragma once

// Minimal GPU decoder wrapper around the NVIDIA Video Codec SDK's low-level
// CUVID driver API (cuviddec.h / nvcuvid.h). Written directly against the
// stable driver-level API rather than the SDK's own sample C++ helper
// classes, since that low-level API has been stable across SDK versions.
//
// NOTE: this file needs Video Codec SDK headers (cuviddec.h, nvcuvid.h) on
// the include path — see CMakeLists.txt's VIDEO_CODEC_SDK_DIR handling.
// It has not been compiled yet (SDK not downloaded at time of writing) —
// treat struct/field names as "best known, needs a compile-fix pass."
//
// Decoded output is NV12 (semi-planar 4:2:0), left on the GPU as a device
// pointer — matches the Step 2 assumption of GPU-resident NV12 surfaces.

#include <cuviddec.h>
#include <nvcuvid.h>
#include <cuda.h>

#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>

// One decoded frame, ready for the preprocessing stage. `device_ptr` points
// into an internal pool owned by NvDecoder — valid until the frame is
// released via ReleaseFrame(), not owned by the caller.
struct DecodedFrame {
    CUdeviceptr device_ptr = 0;
    unsigned int pitch = 0;
    int width = 0;
    int height = 0;
    int64_t timestamp = 0;
    // Which slot of the decoder's internal surface pool this picture was
    // decoded into (CUVIDPARSERDISPINFO::picture_index). Diagnostic only -
    // shows pool-slot reuse when tracing frames through the pipeline.
    int picture_index = -1;
};

class NvDecoder {
public:
    // Creates a lock guarding `cu_context`. Multiple NvDecoder instances can
    // share one ctx_lock when decoding concurrently against the same CUDA
    // context (Step 5: multi-stream) - cuvidCtxLock/cuvidCtxUnlock serialize
    // all of them against this one lock rather than each other. Caller owns
    // the returned handle's lifetime; destroy it with DestroyContextLock()
    // only after every NvDecoder using it has been destroyed.
    static CUvideoctxlock CreateContextLock(CUcontext cu_context);
    static void DestroyContextLock(CUvideoctxlock ctx_lock);

    // `ctx_lock` must have been created (via CreateContextLock) against the
    // CUDA context this decoder will run on. Give each decoder its OWN lock,
    // exactly like NVIDIA's NvDecoder sample does per instance: one lock
    // shared by N decoders (the design until 2026-09) serialized every
    // submit, map and unmap across cameras, and because the map could wait
    // on the inference stream (see output_stream below) all cameras waited
    // with it - the full-inference ceiling was half the NVDEC wall.
    //
    // `output_stream`: the CUDA stream cuvidMapVideoFrame queues its
    // post-processing on (CUVIDPROCPARAMS::output_stream). Left at 0 it is
    // the legacy default stream, which synchronizes with every blocking
    // stream in the context - i.e. with the TensorRT batch in flight. Pass
    // the producer's own non-blocking stream; the caller must then only
    // read the mapped pointer from work queued on that stream (or after
    // synchronizing it), and must synchronize it before ReleaseFrame().
    NvDecoder(CUvideoctxlock ctx_lock, cudaVideoCodec codec,
              CUstream output_stream = nullptr);
    ~NvDecoder();

    NvDecoder(const NvDecoder&) = delete;
    NvDecoder& operator=(const NvDecoder&) = delete;

    // Feeds one demuxed packet (e.g. from FFmpegDemuxer::Demux) to the
    // parser. Decoded frames surface asynchronously via the display
    // callback and become available through PopFrame().
    void Decode(const uint8_t* data, int size, int64_t timestamp = 0);

    // Signals end of stream so the parser flushes any buffered frames.
    void Flush();

    // Pops the oldest ready frame, if any. Returns false if none available.
    bool PopFrame(DecodedFrame* out);

    // Must be called once the caller is done reading a frame's pixels, so
    // its slot in the internal surface pool can be reused.
    void ReleaseFrame(const DecodedFrame& frame);

    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }

private:
    // CUVID callbacks (static trampolines -> instance methods)
    static int CUDAAPI HandleVideoSequenceProc(void* user_data, CUVIDEOFORMAT* format);
    static int CUDAAPI HandlePictureDecodeProc(void* user_data, CUVIDPICPARAMS* pic_params);
    static int CUDAAPI HandlePictureDisplayProc(void* user_data, CUVIDPARSERDISPINFO* disp_info);

    int HandleVideoSequence(CUVIDEOFORMAT* format);
    int HandlePictureDecode(CUVIDPICPARAMS* pic_params);
    int HandlePictureDisplay(CUVIDPARSERDISPINFO* disp_info);

    CUvideoctxlock ctx_lock_ = nullptr;  // not owned - see CreateContextLock/DestroyContextLock
    CUstream output_stream_ = nullptr;  // see ctor comment
    CUvideoparser parser_ = nullptr;
    CUvideodecoder decoder_ = nullptr;

    cudaVideoCodec codec_type_;
    int width_ = 0;
    int height_ = 0;
    unsigned int num_decode_surfaces_ = 0;

    std::mutex frame_queue_mutex_;
    std::deque<DecodedFrame> ready_frames_;
};
