// Step 4+: the single-stream pipeline, now with GPU postprocessing. RTSP ->
// demux (CPU, compressed) -> NVDEC decode (GPU) -> fused preprocess kernel
// writing directly into the TensorRT input binding (GPU) -> yolov8n
// inference (GPU) -> box decode + NMS kernels (GPU, in VRAM) -> compact
// kept-detections D2H (~24 KB) -> struct convert (CPU, trivial).
//
// Pixels never touch the CPU, and neither does the raw prediction tensor
// anymore. Per frame, host<->device traffic is: compressed slice up (~KB),
// final detections down (~24 KB). Annotated PNG dumps are the deliberate
// debug exception. --verify additionally D2Hs the raw tensor each frame,
// re-runs the old CPU postprocess on it and diffs the two paths live (its
// timing is not representative - that copy is exactly what the GPU path
// removed).
//
// Usage: ./rtsp_infer <rtsp-url> [--engine PATH] [--frames N] [--dump-every N]
//                     [--trace] [--verify]

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>

#include "FFmpegDemuxer.h"
#include "FrameDumper.h"
#include "NvDecoder.h"
#include "TrtEngine.h"
#include "postprocess.h"
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

// VRAM + host-RSS snapshot for --memstats. cudaMemGetInfo reports GPU-wide
// usage (X server etc. included), so absolute numbers include neighbors -
// the *deltas* between labeled snapshots isolate this process's allocations.
void PrintMemStats(const char* label) {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    long rss_kb = 0;
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::sscanf(line.c_str(), "VmRSS: %ld", &rss_kb);
            break;
        }
    }
    std::printf("[mem] %-24s VRAM used %8.1f MB (GPU-wide) | host RSS %7.1f MB\n",
                label, (total_b - free_b) / 1048576.0, rss_kb / 1024.0);
}

const char* kCocoNames[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

struct Detection {
    cv::Rect2f box;  // source-frame coordinates
    int cls = 0;
    float score = 0.f;
};

// Production thresholds - the GPU kernels and the CPU reference below must
// use the same values for --verify to be meaningful.
constexpr float kScoreThresh = 0.4f;
constexpr float kIouThresh = 0.45f;
constexpr int kClasses = 80;

// CPU reference path, kept verbatim for --verify (this was the production
// postprocess until the GPU kernels replaced it).
// yolov8 raw output layout: [84, 8400] transposed - out[c * 8400 + i] for
// anchor i. Rows 0..3 = cx,cy,w,h in 640-space, rows 4..83 = class scores.
std::vector<Detection> Postprocess(const float* out, const LetterboxInfo& lb,
                                   int src_w, int src_h,
                                   float score_thresh = kScoreThresh,
                                   float iou_thresh = kIouThresh) {
    constexpr int kAnchors = 8400;
    std::vector<Detection> cands;
    for (int i = 0; i < kAnchors; i++) {
        int best_cls = 0;
        float best = 0.f;
        for (int c = 0; c < 80; c++) {
            const float s = out[(4 + c) * kAnchors + i];
            if (s > best) { best = s; best_cls = c; }
        }
        if (best < score_thresh) continue;

        const float cx = out[0 * kAnchors + i];
        const float cy = out[1 * kAnchors + i];
        const float w = out[2 * kAnchors + i];
        const float h = out[3 * kAnchors + i];

        // Un-letterbox: 640-tensor space -> source frame space.
        Detection d;
        d.box.x = (cx - w / 2.f - lb.pad_x) / lb.scale;
        d.box.y = (cy - h / 2.f - lb.pad_y) / lb.scale;
        d.box.width = w / lb.scale;
        d.box.height = h / lb.scale;
        d.box &= cv::Rect2f(0, 0, (float)src_w, (float)src_h);
        d.cls = best_cls;
        d.score = best;
        cands.push_back(d);
    }

    // Greedy per-class NMS.
    std::sort(cands.begin(), cands.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    for (const auto& d : cands) {
        bool suppressed = false;
        for (const auto& k : kept) {
            if (k.cls != d.cls) continue;
            const float inter = (k.box & d.box).area();
            const float uni = k.box.area() + d.box.area() - inter;
            if (uni > 0 && inter / uni > iou_thresh) { suppressed = true; break; }
        }
        if (!suppressed) kept.push_back(d);
    }
    return kept;
}

void DrawAndSave(const cv::Mat& bgr, const std::vector<Detection>& dets,
                 int frame_index) {
    cv::Mat img = bgr.clone();
    for (const auto& d : dets) {
        cv::rectangle(img, d.box, cv::Scalar(0, 220, 0), 2);
        char label[64];
        std::snprintf(label, sizeof(label), "%s %.2f", kCocoNames[d.cls], d.score);
        cv::putText(img, label, {(int)d.box.x, (int)d.box.y - 6},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 220, 0), 2);
    }
    mkdir("detections_dump", 0755);
    char path[64];
    std::snprintf(path, sizeof(path), "detections_dump/frame_%04d.png", frame_index);
    cv::imwrite(path, img);
    std::cout << "  dumped " << path << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <rtsp-url> [--engine PATH] [--frames N] [--dump-every N]"
                  << std::endl;
        return -1;
    }
    std::string engine_path = "yolov8n.engine";
    int max_frames = 150;
    int dump_every = 50;
    bool trace = false;
    bool verify = false;
    bool memstats = false;
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--engine") && i + 1 < argc) engine_path = argv[++i];
        else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) max_frames = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--dump-every") && i + 1 < argc) dump_every = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--trace")) trace = true;
        else if (!std::strcmp(argv[i], "--verify")) verify = true;
        else if (!std::strcmp(argv[i], "--memstats")) memstats = true;
    }
    // 5 spaced frames to follow through every stage of the pipeline.
    const std::set<int> kTraceFrames = {10, 35, 60, 85, 110};

    std::cout << "=== Step 4: RTSP -> NVDEC -> preprocess -> TensorRT ===" << std::endl;

    FFmpegDemuxer demuxer(argv[1]);
    std::cout << "✓ Stream opened: " << demuxer.GetWidth() << "x"
              << demuxer.GetHeight() << std::endl;

    CheckCu(cuInit(0), "cuInit");
    CUdevice cu_device;
    CheckCu(cuDeviceGet(&cu_device, 0), "cuDeviceGet");
    CUcontext cu_context;
    CheckCu(cuCtxCreate(&cu_context, 0, cu_device), "cuCtxCreate");
    if (memstats) PrintMemStats("after CUDA context");

    int frames = 0;
    double total_ms = 0.0;
    int verify_ok = 0, verify_fail = 0;
    int pts_valid = 0, pts_nonmono = 0;
    std::vector<int64_t> pts_deltas_us;

    // Everything owning CUDA resources lives in this scope so destructors
    // (TrtEngine's cudaFree, NvDecoder's cuvidDestroyDecoder) run before
    // cuCtxDestroy below - same ordering rule as rtsp_decode.
    {
    // TensorRT + all runtime calls bind to the context we just made current -
    // same address space as the decoder's surfaces (verified by
    // preprocess_test). Engine bindings are allocated once, here.
    TrtEngine engine(engine_path);
    std::cout << "✓ Engine loaded: " << engine_path
              << " (input " << engine.InputCount() << " floats, output "
              << engine.OutputCount() << " floats)" << std::endl;
    if (memstats) PrintMemStats("after TRT engine load");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t preprocess_done;
    cudaEventCreate(&preprocess_done);
    // Extra events for --trace per-stage timing.
    cudaEvent_t ev_start, ev_infer, ev_post, ev_copy;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_infer);
    cudaEventCreate(&ev_post);
    cudaEventCreate(&ev_copy);

    // GPU postprocess buffers - allocated once, like the engine bindings.
    const int anchors = (int)engine.OutputCount() / (4 + kClasses);
    GpuDetection* d_cands = nullptr;
    int* d_cand_count = nullptr;
    GpuDetection* d_kept = nullptr;
    int* d_kept_count = nullptr;
    cudaMalloc(&d_cands, sizeof(GpuDetection) * anchors);
    cudaMalloc(&d_cand_count, sizeof(int));
    cudaMalloc(&d_kept, sizeof(GpuDetection) * kMaxNmsCandidates);
    cudaMalloc(&d_kept_count, sizeof(int));
    std::vector<GpuDetection> h_kept(kMaxNmsCandidates);
    int h_kept_count = 0;
    if (memstats) PrintMemStats("after postproc buffers");

    // Full raw tensor on host - only --verify pays for this copy now.
    std::vector<float> h_output(engine.OutputCount());

    CUvideoctxlock ctx_lock = NvDecoder::CreateContextLock(cu_context);

    {
        NvDecoder decoder(ctx_lock, demuxer.GetCodecID() == AV_CODEC_ID_HEVC
                                        ? cudaVideoCodec_HEVC
                                        : cudaVideoCodec_H264);
        std::cout << "✓ NVDEC decoder ready\n" << std::endl;
        if (memstats) PrintMemStats("after NVDEC decoder init");

        uint8_t* data;
        int size;
        int packets_read = 0;
        const uint8_t* last_pkt_addr = nullptr;
        int last_pkt_size = 0;
        int64_t pkt_pts_us = -1;
        // PTS sanity: display-order timestamps must be monotonic and spaced
        // ~one frame interval apart (batch bookkeeping depends on it).
        int64_t prev_pts_us = -1;
        while (frames < max_frames && demuxer.Demux(&data, &size, &pkt_pts_us)) {
            decoder.Decode(data, size, pkt_pts_us);
            packets_read++;
            last_pkt_addr = data;
            last_pkt_size = size;

            DecodedFrame frame;
            while (decoder.PopFrame(&frame) && frames < max_frames) {
                frames++;
                if (frame.timestamp >= 0) {
                    pts_valid++;
                    if (prev_pts_us >= 0) {
                        const int64_t d = frame.timestamp - prev_pts_us;
                        if (d <= 0) pts_nonmono++;
                        else pts_deltas_us.push_back(d);
                    }
                    prev_pts_us = frame.timestamp;
                }
                const bool trace_this = trace && kTraceFrames.count(frames);
                const auto t0 = std::chrono::steady_clock::now();

                // GPU pipeline, all on one stream: preprocess -> infer.
                if (trace_this) cudaEventRecord(ev_start, stream);
                const LetterboxInfo lb = LaunchNV12ToTensor(
                    frame.device_ptr, frame.pitch, frame.width, frame.height,
                    engine.InputPtr(), 640, 640, stream);
                cudaEventRecord(preprocess_done, stream);
                engine.Infer(stream);
                if (trace_this) cudaEventRecord(ev_infer, stream);
                // GPU postprocess, chained on the same stream: box decode +
                // NMS run in VRAM; only the compact kept-list crosses PCIe.
                LaunchBoxDecode(engine.OutputPtr(), anchors, kClasses, lb,
                                frame.width, frame.height, kScoreThresh,
                                d_cands, d_cand_count, stream);
                LaunchNms(d_cands, d_cand_count, kIouThresh, d_kept,
                          d_kept_count, stream);
                if (trace_this) cudaEventRecord(ev_post, stream);
                cudaMemcpyAsync(&h_kept_count, d_kept_count, sizeof(int),
                                cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(h_kept.data(), d_kept,
                                sizeof(GpuDetection) * kMaxNmsCandidates,
                                cudaMemcpyDeviceToHost, stream);
                if (trace_this) cudaEventRecord(ev_copy, stream);

                // Keep an annotated dump? Grab pixels while still mapped.
                const bool dump = dump_every > 0 && frames % dump_every == 0;
                cv::Mat bgr;
                if (dump) FrameDumper::ToBGR(frame, &bgr);

                // Surface is only needed until the kernel has *read* it -
                // release as soon as preprocessing completes, while inference
                // may still be running. Keeps the 2-surface output pool free.
                cudaEventSynchronize(preprocess_done);
                decoder.ReleaseFrame(frame);

                cudaStreamSynchronize(stream);  // kept-list now on host
                const auto t_pp0 = std::chrono::steady_clock::now();
                std::vector<Detection> dets;
                dets.reserve(h_kept_count);
                for (int k = 0; k < h_kept_count; k++) {
                    const GpuDetection& g = h_kept[k];
                    Detection d;
                    d.box = cv::Rect2f(g.x, g.y, g.w, g.h);
                    d.cls = g.cls;
                    d.score = g.score;
                    dets.push_back(d);
                }
                const auto t1 = std::chrono::steady_clock::now();
                const double ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                total_ms += ms;

                if (trace_this) {
                    float ms_pre = 0.f, ms_inf = 0.f, ms_post = 0.f, ms_copy = 0.f;
                    cudaEventElapsedTime(&ms_pre, ev_start, preprocess_done);
                    cudaEventElapsedTime(&ms_inf, preprocess_done, ev_infer);
                    cudaEventElapsedTime(&ms_post, ev_infer, ev_post);
                    cudaEventElapsedTime(&ms_copy, ev_post, ev_copy);
                    const double ms_pp =
                        std::chrono::duration<double, std::milli>(t1 - t_pp0).count();
                    const size_t mapped_bytes =
                        (size_t)frame.pitch * frame.height * 3 / 2;

                    std::printf("\n──── TRACE frame #%d ─────────────────────────────\n", frames);
                    std::printf("[1 demux    ] packet #%d, %d bytes compressed H.264, host addr %p (CPU RAM)\n",
                                packets_read, last_pkt_size, (const void*)last_pkt_addr);
                    std::printf("[2 decode   ] NVDEC wrote picture into decode-surface pool slot %d (of 8)\n",
                                frame.picture_index);
                    std::printf("              pts %lld µs (display order, from RTP via demuxer)\n",
                                (long long)frame.timestamp);
                    std::printf("[3 map      ] output surface VRAM addr 0x%llx, NV12 %dx%d, pitch %u\n",
                                (unsigned long long)frame.device_ptr,
                                frame.width, frame.height, frame.pitch);
                    std::printf("              Y plane @ +0x0, UV plane @ +0x%zx, %zu bytes mapped\n",
                                (size_t)frame.pitch * frame.height, mapped_bytes);
                    std::printf("[4 preprocess] fused kernel read NV12 -> wrote TRT input binding\n");
                    std::printf("              dst VRAM addr %p, 1x3x640x640 f32 (%zu bytes)\n",
                                (void*)engine.InputPtr(), engine.InputCount() * sizeof(float));
                    std::printf("              letterbox scale=%.3f pad=(%d,%d) | %.3f ms\n",
                                lb.scale, lb.pad_x, lb.pad_y, ms_pre);
                    std::printf("[5 inference ] TensorRT enqueueV3 -> output binding\n");
                    std::printf("              dst VRAM addr %p, 1x84x8400 f32 (%zu bytes) | %.3f ms\n",
                                (void*)engine.OutputPtr(), engine.OutputCount() * sizeof(float), ms_inf);
                    std::printf("[6 postproc  ] GPU box decode + NMS kernels, in VRAM: %zu detections | %.3f ms\n",
                                dets.size(), ms_post);
                    std::printf("              cands buf %p (%d max), kept buf %p (%d max)\n",
                                (void*)d_cands, anchors, (void*)d_kept, kMaxNmsCandidates);
                    std::printf("[7 D2H copy  ] kept-list -> host addr %p (%zu bytes over PCIe) | %.3f ms\n",
                                (void*)h_kept.data(),
                                sizeof(int) + sizeof(GpuDetection) * kMaxNmsCandidates, ms_copy);
                    std::printf("              CPU struct convert | %.3f ms\n", ms_pp);
                    for (const auto& d : dets)
                        std::printf("              %s %.2f @ (%.0f,%.0f %.0fx%.0f) src coords\n",
                                    kCocoNames[d.cls], d.score, d.box.x, d.box.y,
                                    d.box.width, d.box.height);
                    std::printf("──────────────────────────────────────────────────\n\n");
                }

                if (verify) {
                    // Old path on the same tensor: full D2H + CPU postprocess.
                    cudaMemcpy(h_output.data(), engine.OutputPtr(),
                               h_output.size() * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    auto cpu_dets = Postprocess(h_output.data(), lb,
                                                frame.width, frame.height);
                    // Canonical order on both sides (kernel output already is).
                    std::sort(cpu_dets.begin(), cpu_dets.end(),
                              [](const Detection& a, const Detection& b) {
                                  if (a.score != b.score) return a.score > b.score;
                                  if (a.cls != b.cls) return a.cls < b.cls;
                                  return a.box.x < b.box.x;
                              });
                    bool ok = cpu_dets.size() == dets.size();
                    for (size_t k = 0; ok && k < dets.size(); k++) {
                        ok = cpu_dets[k].cls == dets[k].cls &&
                             cpu_dets[k].score == dets[k].score &&
                             std::abs(cpu_dets[k].box.x - dets[k].box.x) < 0.1f &&
                             std::abs(cpu_dets[k].box.y - dets[k].box.y) < 0.1f &&
                             std::abs(cpu_dets[k].box.width - dets[k].box.width) < 0.1f &&
                             std::abs(cpu_dets[k].box.height - dets[k].box.height) < 0.1f;
                    }
                    if (ok) {
                        verify_ok++;
                    } else {
                        verify_fail++;
                        std::printf("✗ verify mismatch frame %d: cpu %zu dets, gpu %zu dets\n",
                                    frames, cpu_dets.size(), dets.size());
                    }
                }

                if (frames <= 5 || frames % 25 == 0 || dump) {
                    std::cout << "frame[" << frames << "] " << ms << " ms, "
                              << dets.size() << " detections:";
                    for (const auto& d : dets)
                        std::cout << " " << kCocoNames[d.cls] << "(" << d.score << ")";
                    std::cout << std::endl;
                }
                if (dump) DrawAndSave(bgr, dets, frames);
                if (memstats && (frames == 1 || frames == 30))
                    PrintMemStats(frames == 1 ? "after first frame"
                                              : "steady state (frame 30)");
            }
        }
    }

    NvDecoder::DestroyContextLock(ctx_lock);
    cudaFree(d_cands);
    cudaFree(d_cand_count);
    cudaFree(d_kept);
    cudaFree(d_kept_count);
    cudaEventDestroy(preprocess_done);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_infer);
    cudaEventDestroy(ev_post);
    cudaEventDestroy(ev_copy);
    cudaStreamDestroy(stream);
    }  // engine destroyed here, before cuCtxDestroy

    if (frames > 0) {
        std::cout << "\n" << frames << " frames, avg "
                  << total_ms / frames << " ms/frame (decode-to-detections, "
                  << 1000.0 / (total_ms / frames) << " fps capability)"
                  << std::endl;
        if (verify) {
            std::cout << "verify: " << verify_ok << "/" << (verify_ok + verify_fail)
                      << " frames matched CPU reference"
                      << (verify_fail ? " ✗ MISMATCHES PRESENT" : " ✓") << std::endl;
        }
        if (pts_valid > 0 && !pts_deltas_us.empty()) {
            auto mid = pts_deltas_us.begin() + pts_deltas_us.size() / 2;
            std::nth_element(pts_deltas_us.begin(), mid, pts_deltas_us.end());
            std::printf("pts: %d/%d frames timestamped, %d non-monotonic, "
                        "median delta %.1f ms%s\n",
                        pts_valid, frames, pts_nonmono, *mid / 1000.0,
                        pts_nonmono == 0 ? " ✓" : " ✗");
        } else {
            std::printf("pts: no timestamps from source ✗\n");
        }
        std::cout << "✓ Success: live RTSP inference pipeline complete." << std::endl;
    }

    cuCtxDestroy(cu_context);
    return frames > 0 ? 0 : -1;
}
