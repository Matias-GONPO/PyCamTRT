// GPU-postprocess checkpoint, parts 1+2: prove the box-decode kernel and the
// NMS kernel produce the same results as the CPU path, on a real engine
// output. Part 1 diffs decode candidates; part 2 chains the NMS kernel onto
// the *GPU* candidates (exactly as the pipeline will) and diffs the kept set
// against a CPU greedy reference run on the CPU candidates.
// Self-contained on purpose - runs inference once on a static image, no RTSP
// source needed (argmax, threshold, un-letterbox, clamp and IoU don't care
// where the tensor came from). Default image is Yolo_test.png because it is
// non-square (802x543), so the letterbox scale/pad arithmetic actually gets
// exercised, and it is busy enough to yield dozens of candidates.
//
// Usage: ./postprocess_test [--engine PATH] [--image PATH] [--thresh S] [--iou I]
// Run from the project root (relative default paths, like the other tools).

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "TrtEngine.h"
#include "postprocess.h"
#include "preprocess.h"

namespace {

void CheckCuda(cudaError_t result, const char* what) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA runtime call failed: " << what << " ("
                  << cudaGetErrorString(result) << ")" << std::endl;
        std::exit(-1);
    }
}

// CPU reference: the candidate loop from rtsp_infer's Postprocess, box decode
// only (NMS is part 2). Kept arithmetically identical to both that loop and
// the kernel. One knowing deviation from cv::Rect2f: a box entirely outside
// the frame clamps to a zero-size box at the frame edge instead of cv's
// all-zero empty rect - it cannot score above threshold in practice and has
// zero area either way.
std::vector<GpuDetection> CpuBoxDecode(const float* out, int anchors,
                                       int classes, const LetterboxInfo& lb,
                                       int src_w, int src_h, float thresh) {
    std::vector<GpuDetection> cands;
    for (int i = 0; i < anchors; i++) {
        int best_cls = 0;
        float best = 0.f;
        for (int c = 0; c < classes; c++) {
            const float s = out[(4 + c) * anchors + i];
            if (s > best) { best = s; best_cls = c; }
        }
        if (best < thresh) continue;

        const float cx = out[0 * anchors + i];
        const float cy = out[1 * anchors + i];
        const float w = out[2 * anchors + i];
        const float h = out[3 * anchors + i];

        float x0 = (cx - w / 2.f - lb.pad_x) / lb.scale;
        float y0 = (cy - h / 2.f - lb.pad_y) / lb.scale;
        float x1 = x0 + w / lb.scale;
        float y1 = y0 + h / lb.scale;
        x0 = std::max(x0, 0.f);
        y0 = std::max(y0, 0.f);
        x1 = std::min(x1, (float)src_w);
        y1 = std::min(y1, (float)src_h);

        GpuDetection d;
        d.x = x0;
        d.y = y0;
        d.w = std::max(x1 - x0, 0.f);
        d.h = std::max(y1 - y0, 0.f);
        d.score = best;
        d.cls = best_cls;
        cands.push_back(d);
    }
    return cands;
}

// Candidate order differs between the paths (CPU: anchor order, GPU: atomic
// compaction order), so both sides get the same canonical sort before the
// diff. Scores are read (not computed) from the same tensor, so they pair up
// bit-exactly. Same composite order the NMS kernel sorts by.
void CanonicalSort(std::vector<GpuDetection>* v) {
    std::sort(v->begin(), v->end(),
              [](const GpuDetection& a, const GpuDetection& b) {
                  if (a.score != b.score) return a.score > b.score;
                  if (a.cls != b.cls) return a.cls < b.cls;
                  return a.x < b.x;
              });
}

// CPU reference NMS: rtsp_infer's greedy loop, run over the canonical order
// (production sorts by score only, leaving ties unspecified; equal-score
// candidates are duplicate-anchor near-copies of the same box, so the kept
// set is the same - this just makes the diff against the kernel exact).
// IoU arithmetic matches cv::Rect2f intersection and the kernel's Iou().
std::vector<GpuDetection> CpuNms(std::vector<GpuDetection> cands,
                                 float iou_thresh) {
    CanonicalSort(&cands);
    std::vector<GpuDetection> kept;
    for (const auto& d : cands) {
        bool suppressed = false;
        for (const auto& k : kept) {
            if (k.cls != d.cls) continue;
            const float ix0 = std::max(k.x, d.x);
            const float iy0 = std::max(k.y, d.y);
            const float ix1 = std::min(k.x + k.w, d.x + d.w);
            const float iy1 = std::min(k.y + k.h, d.y + d.h);
            const float inter =
                std::max(ix1 - ix0, 0.f) * std::max(iy1 - iy0, 0.f);
            const float uni = k.w * k.h + d.w * d.h - inter;
            if (uni > 0.f && inter / uni > iou_thresh) { suppressed = true; break; }
        }
        if (!suppressed) kept.push_back(d);
    }
    return kept;
}

}  // namespace

int main(int argc, char** argv) {
    std::string engine_path = "yolov8n.engine";
    std::string image_path = "Yolo_test.png";
    float thresh = 0.4f;
    float iou_thresh = 0.45f;  // production value in rtsp_infer's Postprocess
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--engine") && i + 1 < argc) engine_path = argv[++i];
        else if (!std::strcmp(argv[i], "--image") && i + 1 < argc) image_path = argv[++i];
        else if (!std::strcmp(argv[i], "--thresh") && i + 1 < argc) thresh = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--iou") && i + 1 < argc) iou_thresh = std::atof(argv[++i]);
    }

    std::cout << "=== GPU postprocess part 1 checkpoint: box decode ===" << std::endl;

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Cannot read image: " << image_path << std::endl;
        return -1;
    }
    std::cout << "✓ Image: " << image_path << " (" << img.cols << "x"
              << img.rows << ")" << std::endl;

    TrtEngine engine(engine_path);
    std::cout << "✓ Engine loaded: " << engine_path << std::endl;

    // CPU letterbox to feed the engine - test scaffolding only, not pipeline
    // code (the real pipeline feeds the input binding from the NV12 kernel).
    // Same scale/pad formulas as LaunchNV12ToTensor so lb matches production.
    constexpr int kW = 640, kH = 640;
    LetterboxInfo lb;
    lb.scale = std::min((float)kW / img.cols, (float)kH / img.rows);
    lb.pad_x = (kW - (int)(img.cols * lb.scale)) / 2;
    lb.pad_y = (kH - (int)(img.rows * lb.scale)) / 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size((int)(img.cols * lb.scale),
                                      (int)(img.rows * lb.scale)));
    cv::Mat canvas(kH, kW, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(lb.pad_x, lb.pad_y, resized.cols, resized.rows)));

    std::vector<float> h_input(3 * kW * kH);
    const int plane = kW * kH;
    for (int y = 0; y < kH; y++) {
        for (int x = 0; x < kW; x++) {
            const cv::Vec3b px = canvas.at<cv::Vec3b>(y, x);  // BGR
            const int i = y * kW + x;
            h_input[i] = px[2] / 255.f;              // R
            h_input[plane + i] = px[1] / 255.f;      // G
            h_input[2 * plane + i] = px[0] / 255.f;  // B
        }
    }
    CheckCuda(cudaMemcpy(engine.InputPtr(), h_input.data(),
                         h_input.size() * sizeof(float), cudaMemcpyHostToDevice),
              "H2D input");
    engine.Infer(0);
    CheckCuda(cudaStreamSynchronize(0), "infer sync");
    std::cout << "✓ Inference done, output stays in VRAM for the GPU path"
              << std::endl;

    constexpr int kClasses = 80;
    const int anchors = (int)engine.OutputCount() / (4 + kClasses);
    if ((size_t)anchors * (4 + kClasses) != engine.OutputCount()) {
        std::cerr << "Unexpected output size " << engine.OutputCount() << std::endl;
        return -1;
    }

    // CPU reference path (what rtsp_infer does today).
    std::vector<float> h_raw(engine.OutputCount());
    CheckCuda(cudaMemcpy(h_raw.data(), engine.OutputPtr(),
                         h_raw.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "D2H raw");
    std::vector<GpuDetection> cpu_dets =
        CpuBoxDecode(h_raw.data(), anchors, kClasses, lb, img.cols, img.rows, thresh);

    // GPU path: kernel reads the output binding in place.
    GpuDetection* d_out = nullptr;
    int* d_count = nullptr;
    CheckCuda(cudaMalloc(&d_out, sizeof(GpuDetection) * anchors), "malloc d_out");
    CheckCuda(cudaMalloc(&d_count, sizeof(int)), "malloc d_count");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    LaunchBoxDecode(engine.OutputPtr(), anchors, kClasses, lb,
                    img.cols, img.rows, thresh, d_out, d_count, /*stream=*/0);
    cudaEventRecord(stop);
    CheckCuda(cudaEventSynchronize(stop), "kernel sync");
    float kernel_ms = 0.f;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int gpu_count = 0;
    CheckCuda(cudaMemcpy(&gpu_count, d_count, sizeof(int), cudaMemcpyDeviceToHost),
              "D2H count");
    std::vector<GpuDetection> gpu_dets(std::max(gpu_count, 0));
    if (gpu_count > 0) {
        CheckCuda(cudaMemcpy(gpu_dets.data(), d_out,
                             sizeof(GpuDetection) * gpu_count,
                             cudaMemcpyDeviceToHost),
                  "D2H dets");
    }

    std::printf("✓ Box-decode kernel: %.3f ms (incl. count memset)\n", kernel_ms);
    std::printf("  CPU candidates: %zu | GPU candidates: %d\n",
                cpu_dets.size(), gpu_count);

    // Diff. Scores must pair bit-exactly (read, not computed); coordinates
    // get a small tolerance for possible FMA contraction differences.
    bool pass = cpu_dets.size() == (size_t)gpu_count;
    CanonicalSort(&cpu_dets);
    CanonicalSort(&gpu_dets);
    float max_coord_diff = 0.f;
    if (pass) {
        for (size_t i = 0; i < cpu_dets.size(); i++) {
            const GpuDetection& a = cpu_dets[i];
            const GpuDetection& b = gpu_dets[i];
            if (a.cls != b.cls || a.score != b.score) { pass = false; break; }
            max_coord_diff = std::max({max_coord_diff,
                                       std::abs(a.x - b.x), std::abs(a.y - b.y),
                                       std::abs(a.w - b.w), std::abs(a.h - b.h)});
        }
        if (max_coord_diff > 0.05f) pass = false;
    }
    std::printf("  max coordinate diff: %.6f px\n", max_coord_diff);

    const size_t show = std::min<size_t>(5, cpu_dets.size());
    for (size_t i = 0; i < show; i++) {
        const GpuDetection& a = cpu_dets[i];
        const GpuDetection& b = gpu_dets[i];
        std::printf("  [%zu] cpu: cls=%2d %.4f (%.1f,%.1f %.1fx%.1f) | "
                    "gpu: cls=%2d %.4f (%.1f,%.1f %.1fx%.1f)\n",
                    i, a.cls, a.score, a.x, a.y, a.w, a.h,
                    b.cls, b.score, b.x, b.y, b.w, b.h);
    }
    std::cout << (pass ? "✓ PASS part 1" : "✗ FAIL part 1")
              << ": GPU box decode vs CPU reference." << std::endl;

    // ---- Part 2: NMS kernel, chained onto the GPU candidates in VRAM ----
    // (exactly the production flow: decode output feeds NMS with no host
    // round-trip; the count stays on the device).
    GpuDetection* d_kept = nullptr;
    int* d_kept_count = nullptr;
    CheckCuda(cudaMalloc(&d_kept, sizeof(GpuDetection) * kMaxNmsCandidates),
              "malloc d_kept");
    CheckCuda(cudaMalloc(&d_kept_count, sizeof(int)), "malloc d_kept_count");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    LaunchNms(d_out, d_count, iou_thresh, d_kept, d_kept_count, /*stream=*/0);
    cudaEventRecord(stop);
    CheckCuda(cudaEventSynchronize(stop), "nms sync");
    float nms_ms = 0.f;
    cudaEventElapsedTime(&nms_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int gpu_kept_count = 0;
    CheckCuda(cudaMemcpy(&gpu_kept_count, d_kept_count, sizeof(int),
                         cudaMemcpyDeviceToHost),
              "D2H kept count");
    std::vector<GpuDetection> gpu_kept(std::max(gpu_kept_count, 0));
    if (gpu_kept_count > 0) {
        CheckCuda(cudaMemcpy(gpu_kept.data(), d_kept,
                             sizeof(GpuDetection) * gpu_kept_count,
                             cudaMemcpyDeviceToHost),
                  "D2H kept");
    }

    std::vector<GpuDetection> cpu_kept = CpuNms(cpu_dets, iou_thresh);

    std::printf("✓ NMS kernel: %.3f ms\n", nms_ms);
    std::printf("  CPU kept: %zu | GPU kept: %d (iou %.2f)\n",
                cpu_kept.size(), gpu_kept_count, iou_thresh);

    // Kernel writes survivors already in canonical order; cpu_kept was
    // produced in that order too - compare elementwise.
    bool pass2 = cpu_kept.size() == (size_t)gpu_kept_count;
    float max_kept_diff = 0.f;
    if (pass2) {
        for (size_t i = 0; i < cpu_kept.size(); i++) {
            const GpuDetection& a = cpu_kept[i];
            const GpuDetection& b = gpu_kept[i];
            if (a.cls != b.cls || a.score != b.score) { pass2 = false; break; }
            max_kept_diff = std::max({max_kept_diff,
                                      std::abs(a.x - b.x), std::abs(a.y - b.y),
                                      std::abs(a.w - b.w), std::abs(a.h - b.h)});
        }
        if (max_kept_diff > 0.05f) pass2 = false;
    }
    std::printf("  max kept coordinate diff: %.6f px\n", max_kept_diff);

    const size_t show2 = std::min<size_t>(8, cpu_kept.size());
    for (size_t i = 0; i < show2; i++) {
        const GpuDetection& a = cpu_kept[i];
        const GpuDetection& b = gpu_kept[i];
        std::printf("  [%zu] cpu: cls=%2d %.4f (%.1f,%.1f %.1fx%.1f) | "
                    "gpu: cls=%2d %.4f (%.1f,%.1f %.1fx%.1f)\n",
                    i, a.cls, a.score, a.x, a.y, a.w, a.h,
                    b.cls, b.score, b.x, b.y, b.w, b.h);
    }
    std::cout << (pass2 ? "✓ PASS part 2" : "✗ FAIL part 2")
              << ": GPU NMS vs CPU greedy reference." << std::endl;

    cudaFree(d_out);
    cudaFree(d_count);
    cudaFree(d_kept);
    cudaFree(d_kept_count);

    std::cout << (pass && pass2 ? "✓ PASS" : "✗ FAIL")
              << ": GPU postprocess checkpoint." << std::endl;
    return pass && pass2 ? 0 : -1;
}
