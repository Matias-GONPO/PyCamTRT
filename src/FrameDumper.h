#pragma once

// Debug/verification utility: copies a decoded NV12 frame out of GPU memory
// and writes it to disk as a viewable PNG. Not part of the pipeline hot path
// — the D2H copy is deliberate, it's the whole point (proving the device
// pointer holds real decoded pixels, same spirit as demuxer_test proving the
// demuxer alone works).
//
// Layout assumption (matches NvDecoder: ulTargetWidth/Height = display size):
// mapped surface is height rows of Y followed by height/2 rows of interleaved
// UV, all at the same pitch.

#include <cuda.h>
#include <opencv2/opencv.hpp>

#include <cstdio>
#include <string>
#include <sys/stat.h>

#include "NvDecoder.h"

class FrameDumper {
public:
    explicit FrameDumper(const std::string& out_dir) : out_dir_(out_dir) {
        mkdir(out_dir_.c_str(), 0755);  // ok if it already exists
    }

    // D2H copy + NV12 -> BGR conversion, no file written. Must be called while
    // `frame` is still mapped (i.e. before ReleaseFrame()). Also useful on
    // its own when the caller wants to annotate before saving (rtsp_infer).
    static bool ToBGR(const DecodedFrame& frame, cv::Mat* out) {
        const int h_nv12 = frame.height * 3 / 2;
        cv::Mat nv12(h_nv12, frame.width, CV_8UC1);

        CUDA_MEMCPY2D copy = {};
        copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.srcDevice = frame.device_ptr;
        copy.srcPitch = frame.pitch;
        copy.dstMemoryType = CU_MEMORYTYPE_HOST;
        copy.dstHost = nv12.data;
        copy.dstPitch = static_cast<size_t>(frame.width);
        copy.WidthInBytes = static_cast<size_t>(frame.width);
        copy.Height = static_cast<size_t>(h_nv12);
        if (cuMemcpy2D(&copy) != CUDA_SUCCESS) {
            std::fprintf(stderr, "FrameDumper: cuMemcpy2D failed\n");
            return false;
        }

        cv::cvtColor(nv12, *out, cv::COLOR_YUV2BGR_NV12);
        return true;
    }

    // Returns true if the PNG was written. Must be called while `frame` is
    // still mapped (i.e. before ReleaseFrame()).
    bool WritePNG(const DecodedFrame& frame, int frame_index) {
        cv::Mat bgr;
        if (!ToBGR(frame, &bgr)) return false;

        char name[64];
        std::snprintf(name, sizeof(name), "frame_%04d.png", frame_index);
        const std::string path = out_dir_ + "/" + name;
        if (!cv::imwrite(path, bgr)) {
            std::fprintf(stderr, "FrameDumper: failed to write %s\n", path.c_str());
            return false;
        }
        std::printf("  dumped %s\n", path.c_str());
        return true;
    }

private:
    std::string out_dir_;
};
