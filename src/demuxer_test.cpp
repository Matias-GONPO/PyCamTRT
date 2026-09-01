// Standalone checkpoint: proves we can open an RTSP URL and pull encoded
// packets out of it, independent of NVDEC/Video Codec SDK. Run this against
// the test RTSP stream before wiring up GPU decode.
//
// Usage: ./demuxer_test rtsp://<ip>:8554/stream

#include <iostream>
#include "FFmpegDemuxer.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <rtsp-url>" << std::endl;
        return -1;
    }

    try {
        FFmpegDemuxer demuxer(argv[1]);

        std::cout << "Opened stream: " << argv[1] << std::endl;
        std::cout << "Resolution: " << demuxer.GetWidth() << "x" << demuxer.GetHeight() << std::endl;
        std::cout << "Codec ID: " << demuxer.GetCodecID()
                   << (demuxer.GetCodecID() == AV_CODEC_ID_H264 ? " (H.264)" :
                       demuxer.GetCodecID() == AV_CODEC_ID_HEVC ? " (HEVC)" : " (other)")
                   << std::endl;

        const int kNumPackets = 100;
        int count = 0;
        uint8_t* data;
        int size;

        while (count < kNumPackets && demuxer.Demux(&data, &size)) {
            std::cout << "packet[" << count << "] size=" << size << " bytes" << std::endl;
            count++;
        }

        std::cout << "\n" << (count > 0 ? "Success" : "Failure")
                   << ": read " << count << " packets." << std::endl;
        return count > 0 ? 0 : -1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
