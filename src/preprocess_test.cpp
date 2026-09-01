// Step 2 checkpoint: decode one real RTSP frame on the GPU, run the fused
// NV12 -> letterboxed RGB float tensor kernel on it, and dump the result as a
// PNG. Same philosophy as demuxer_test / rtsp_decode --dump: prove this stage
// alone works, visually, before wiring it into inference (Step 4).
//
// Also times the kernel with CUDA events — this is the "mandatory overhead"
// between decode and inference, so we want the real number on record.
//
// Usage: ./preprocess_test rtsp://<ip>:8554/stream

#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>

#include "FFmpegDemuxer.h"
#include "NvDecoder.h"
#include "preprocess.h"

namespace {
void CheckCu(CUresult result, const char* what) {
    if (result != CUDA_SUCCESS) {
        const char* err_name = nullptr;
        cuGetErrorName(result, &err_name);
        std::cerr << "CUDA driver call failed: " << what << " ("
                  << (err_name ? err_name : "unknown") << ")" << std::endl;
        std::exit(-1);
    }
}

void CheckCuda(cudaError_t result, const char* what) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA runtime call failed: " << what << " ("
                  << cudaGetErrorString(result) << ")" << std::endl;
        std::exit(-1);
    }
}

constexpr int kTensorW = 640;
constexpr int kTensorH = 640;
// Let auto-exposure settle and get past initial reference frames.
constexpr int kTargetFrame = 30;
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <rtsp-url>" << std::endl;
        return -1;
    }

    std::cout << "=== Step 2 checkpoint: NV12 -> tensor preprocessing ===" << std::endl;

    FFmpegDemuxer demuxer(argv[1]);
    std::cout << "✓ Stream opened: " << demuxer.GetWidth() << "x"
              << demuxer.GetHeight() << std::endl;

    CheckCu(cuInit(0), "cuInit");
    CUdevice cu_device;
    CheckCu(cuDeviceGet(&cu_device, 0), "cuDeviceGet");
    CUcontext cu_context;
    CheckCu(cuCtxCreate(&cu_context, 0, cu_device), "cuCtxCreate");

    // All runtime-API calls below (cudaMalloc, kernel launch) bind to the
    // driver context that is current on this thread — i.e. the one we just
    // created, the same one NVDEC decodes into. This is the interop pattern
    // Step 4 relies on for TensorRT: one context, one address space.
    float* d_tensor = nullptr;
    CheckCuda(cudaMalloc(&d_tensor, sizeof(float) * 3 * kTensorW * kTensorH),
              "cudaMalloc tensor");

    CUvideoctxlock ctx_lock = NvDecoder::CreateContextLock(cu_context);
    bool done = false;

    {
        NvDecoder decoder(ctx_lock, demuxer.GetCodecID() == AV_CODEC_ID_HEVC
                                        ? cudaVideoCodec_HEVC
                                        : cudaVideoCodec_H264);
        std::cout << "✓ NVDEC decoder ready" << std::endl;

        int frames_decoded = 0;
        uint8_t* data;
        int size;
        while (!done && demuxer.Demux(&data, &size)) {
            decoder.Decode(data, size);

            DecodedFrame frame;
            while (decoder.PopFrame(&frame)) {
                frames_decoded++;
                if (frames_decoded < kTargetFrame) {
                    decoder.ReleaseFrame(frame);
                    continue;
                }

                std::cout << "Processing frame " << frames_decoded << " ("
                          << frame.width << "x" << frame.height
                          << ", pitch " << frame.pitch << ")" << std::endl;

                cudaEvent_t start, stop;
                CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
                CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");

                CheckCuda(cudaEventRecord(start), "cudaEventRecord");
                LetterboxInfo lb = LaunchNV12ToTensor(
                    frame.device_ptr, frame.pitch, frame.width, frame.height,
                    d_tensor, kTensorW, kTensorH, /*stream=*/0);
                CheckCuda(cudaEventRecord(stop), "cudaEventRecord");
                CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");

                float ms = 0.f;
                cudaEventElapsedTime(&ms, start, stop);
                std::printf("✓ Fused kernel: %.3f ms (scale=%.3f pad_x=%d pad_y=%d)\n",
                            ms, lb.scale, lb.pad_x, lb.pad_y);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                // Kernel has read the surface - decoder can have it back.
                decoder.ReleaseFrame(frame);

                // D2H + PNG below is checkpoint verification only, not part
                // of the pipeline being measured.
                std::vector<float> h_tensor(3 * kTensorW * kTensorH);
                CheckCuda(cudaMemcpy(h_tensor.data(), d_tensor,
                                     h_tensor.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy tensor D2H");

                cv::Mat img(kTensorH, kTensorW, CV_8UC3);
                const int plane = kTensorW * kTensorH;
                for (int y = 0; y < kTensorH; y++) {
                    for (int x = 0; x < kTensorW; x++) {
                        const int i = y * kTensorW + x;
                        img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                            static_cast<uint8_t>(h_tensor[2 * plane + i] * 255.f),
                            static_cast<uint8_t>(h_tensor[plane + i] * 255.f),
                            static_cast<uint8_t>(h_tensor[i] * 255.f));
                    }
                }
                mkdir("preprocess_dump", 0755);
                cv::imwrite("preprocess_dump/tensor_as_image.png", img);
                std::cout << "✓ Wrote preprocess_dump/tensor_as_image.png "
                             "(letterboxed 640x640, converted back from the "
                             "float tensor)" << std::endl;

                done = true;
                break;
            }
        }
    }

    NvDecoder::DestroyContextLock(ctx_lock);
    cudaFree(d_tensor);
    cuCtxDestroy(cu_context);

    std::cout << (done ? "✓ Success" : "✗ Failure")
              << ": preprocessing checkpoint." << std::endl;
    return done ? 0 : -1;
}
