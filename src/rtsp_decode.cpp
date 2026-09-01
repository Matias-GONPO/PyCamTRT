// Step 3 checkpoint: RTSP -> demux (CPU, FFmpeg) -> decode (GPU, NVDEC).
// Deliberately stripped down like simple_inference.cpp was for Step 1: no
// preprocessing, no inference, just prove frames make it into GPU memory as
// NV12 surfaces. That's the handoff point into Step 2's preprocessing work.
//
// Usage: ./rtsp_decode rtsp://<ip>:8554/stream [--dump N] [--dump-dir DIR]
//   --dump N        also write N randomly chosen decoded frames to disk as
//                   PNGs (visual proof the GPU frames hold real pixels)
//   --dump-dir DIR  where to write them (default: ./frames_dump)

#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <cuda.h>

#include "FFmpegDemuxer.h"
#include "FrameDumper.h"
#include "NvDecoder.h"

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

cudaVideoCodec ToCuvidCodec(AVCodecID id) {
    switch (id) {
        case AV_CODEC_ID_H264: return cudaVideoCodec_H264;
        case AV_CODEC_ID_HEVC: return cudaVideoCodec_HEVC;
        default:
            throw std::runtime_error("Unsupported codec for NVDEC in this test");
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <rtsp-url> [--dump N] [--dump-dir DIR]" << std::endl;
        return -1;
    }

    int num_dump = 0;
    std::string dump_dir = "frames_dump";
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) {
            num_dump = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--dump-dir") && i + 1 < argc) {
            dump_dir = argv[++i];
        }
    }

    std::cout << "=== RTSP -> NVDEC GPU Decode ===" << std::endl;

    // 1. Demuxer: pulls encoded packets out of the RTSP stream (CPU-side,
    //    network + container parsing only - no pixel decode happens here).
    std::cout << "Opening stream..." << std::endl;
    FFmpegDemuxer demuxer(argv[1]);
    std::cout << "✓ Stream opened: " << demuxer.GetWidth() << "x" << demuxer.GetHeight()
               << std::endl;

    // 2. CUDA driver context for NVDEC.
    CheckCu(cuInit(0), "cuInit");
    CUdevice cu_device;
    CheckCu(cuDeviceGet(&cu_device, 0), "cuDeviceGet");
    CUcontext cu_context;
    CheckCu(cuCtxCreate(&cu_context, 0, cu_device), "cuCtxCreate");
    std::cout << "✓ CUDA context created" << std::endl;

    // 3. GPU decoder. The ctx_lock is created once per shared CUDA context,
    //    not once per decoder - when Step 5 adds more streams on this same
    //    context, they all get handed this same ctx_lock.
    CUvideoctxlock ctx_lock = NvDecoder::CreateContextLock(cu_context);
    int frames_decoded = 0;
    int packets_read = 0;

    {
        // Scoped so NvDecoder's destructor (cuvidDestroyDecoder/Parser) runs
        // before the ctx_lock and CUDA context it depends on are torn down.
        NvDecoder decoder(ctx_lock, ToCuvidCodec(demuxer.GetCodecID()));
        std::cout << "✓ NVDEC decoder ready" << std::endl;

        // 4. Feed packets in, drain decoded NV12 frames out.
        const int kNumPacketsToRead = 150;  // a few seconds at typical fps

        // Optional --dump: pick N random frame indices up front. Capped at
        // 100 since ~125 frames come out of 150 packets (decoder holds a few
        // back for reorder) and we want every picked index to actually land.
        FrameDumper dumper(dump_dir);
        std::set<int> dump_indices;
        if (num_dump > 0) {
            std::mt19937 rng(std::random_device{}());
            std::uniform_int_distribution<int> dist(1, 100);
            while ((int)dump_indices.size() < std::min(num_dump, 100))
                dump_indices.insert(dist(rng));
            std::cout << "Dumping " << dump_indices.size()
                      << " random frames to " << dump_dir << "/" << std::endl;
        }

        uint8_t* data;
        int size;
        int64_t pkt_pts_us = -1;
        while (packets_read < kNumPacketsToRead && demuxer.Demux(&data, &size, &pkt_pts_us)) {
            decoder.Decode(data, size, pkt_pts_us);
            packets_read++;

            DecodedFrame frame;
            while (decoder.PopFrame(&frame)) {
                frames_decoded++;
                if (frames_decoded <= 5 || frames_decoded % 30 == 0) {
                    std::cout << "frame[" << frames_decoded << "] "
                              << frame.width << "x" << frame.height
                              << " pitch=" << frame.pitch
                              << " device_ptr=0x" << std::hex << frame.device_ptr << std::dec
                              << " ts=" << frame.timestamp << std::endl;
                }
                if (dump_indices.count(frames_decoded))
                    dumper.WritePNG(frame, frames_decoded);
                decoder.ReleaseFrame(frame);
            }
        }

        decoder.Flush();
        DecodedFrame frame;
        while (decoder.PopFrame(&frame)) {
            frames_decoded++;
            decoder.ReleaseFrame(frame);
        }
    }

    NvDecoder::DestroyContextLock(ctx_lock);

    std::cout << "\nRead " << packets_read << " packets, decoded " << frames_decoded
               << " frames." << std::endl;
    std::cout << (frames_decoded > 0 ? "✓ Success" : "✗ Failure")
               << ": NV12 frames landed in GPU memory via NVDEC." << std::endl;

    cuCtxDestroy(cu_context);
    return frames_decoded > 0 ? 0 : -1;
}
