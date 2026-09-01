// fp16-vs-fp32 stage-1 detection agreement checkpoint.
//
// Verifies that the fp16 plate engine preserves DETECTIONS relative to
// the fp32 engine (accuracy preservation). The --verify path in the
// pipeline is engine-agnostic (GPU postprocess vs CPU on the SAME raw
// tensor) and stays bit-exact regardless of precision; it does NOT
// answer "did fp16 change what we detect?" This test does.
//
// Both engines receive the IDENTICAL preprocessed input tensor
// (guaranteed same input is stronger than two live runs), then run the
// same GPU box-decode + NMS. Agreement contract: same detection COUNT,
// same CLASS per detection, box coords + score within tolerance.
// Tolerances are the meaningful quantity reported — fp16 is expected to
// drift slightly, so we measure HOW MUCH, not demand bit-exactness.
//
// Usage: compare_fp16 <fp32_engine> <fp16_engine> <image> [more images...]

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "TrtEngine.h"
#include "postprocess.h"
#include "preprocess.h"

static constexpr float kScoreThresh = 0.4f;  // pipeline defaults
static constexpr float kIouThresh = 0.45f;
// fp16 agreement: two detections are "the same" if same class and their
// boxes overlap at IOU >= kMatchIou. IOU is the standard detection-
// equivalence measure — absolute-pixel tolerance wrongly penalizes a few
// px of fp16 rounding on large boxes (IOU ~0.97 there). Score drift is
// reported informationally, not gated (a plate at 0.399 vs 0.401 flipping
// the count is a threshold artifact, not an fp16 defect).
static constexpr float kMatchIou = 0.90f;

// Letterbox a BGR image into a 640x640 RGB planar float tensor (/255),
// pad gray 114 — the pipeline's convention, so boxes land in real
// source-pixel coordinates. Fills `lb` for postprocess un-mapping.
static std::vector<float> Letterbox(const cv::Mat& bgr, int dst, LetterboxInfo* lb) {
    const float scale = std::min((float)dst / bgr.cols, (float)dst / bgr.rows);
    const int new_w = (int)std::round(bgr.cols * scale);
    const int new_h = (int)std::round(bgr.rows * scale);
    const int pad_x = (dst - new_w) / 2, pad_y = (dst - new_h) / 2;
    lb->scale = scale;
    lb->pad_x = pad_x;
    lb->pad_y = pad_y;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat canvas(dst, dst, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));

    std::vector<float> t((size_t)3 * dst * dst);
    for (int y = 0; y < dst; y++) {
        for (int x = 0; x < dst; x++) {
            const cv::Vec3b px = canvas.at<cv::Vec3b>(y, x);  // BGR
            // RGB planar: channel 0 = R = px[2]
            t[(size_t)0 * dst * dst + y * dst + x] = px[2] / 255.0f;
            t[(size_t)1 * dst * dst + y * dst + x] = px[1] / 255.0f;
            t[(size_t)2 * dst * dst + y * dst + x] = px[0] / 255.0f;
        }
    }
    return t;
}

// Runs stage-1 + GPU postprocess for one image on one engine; returns
// kept detections on the host.
static std::vector<GpuDetection> RunOne(TrtEngine& eng, const std::vector<float>& in,
                                        const LetterboxInfo& lb, int src_w, int src_h,
                                        int anchors, int classes, cudaStream_t stream,
                                        GpuDetection* d_cands, int* d_count,
                                        GpuDetection* d_kept, int* d_kept_count) {
    eng.SetBatch(1);
    cudaMemcpyAsync(eng.InputPtr(), in.data(), in.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    eng.Infer(stream);
    LaunchBoxDecode(eng.OutputPtr(), anchors, classes, lb, src_w, src_h,
                    kScoreThresh, d_cands, d_count, stream);
    LaunchNms(d_cands, d_count, kIouThresh, d_kept, d_kept_count, stream);
    int n = 0;
    cudaMemcpyAsync(&n, d_kept_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::vector<GpuDetection> out(n);
    if (n > 0)
        cudaMemcpy(out.data(), d_kept, n * sizeof(GpuDetection),
                   cudaMemcpyDeviceToHost);
    return out;
}

// IOU of two boxes given as center (x,y) + size (w,h) in source pixels.
static float Iou(const GpuDetection& a, const GpuDetection& b) {
    const float ax0 = a.x - a.w * 0.5f, ay0 = a.y - a.h * 0.5f;
    const float ax1 = a.x + a.w * 0.5f, ay1 = a.y + a.h * 0.5f;
    const float bx0 = b.x - b.w * 0.5f, by0 = b.y - b.h * 0.5f;
    const float bx1 = b.x + b.w * 0.5f, by1 = b.y + b.h * 0.5f;
    const float ix = std::max(0.f, std::min(ax1, bx1) - std::max(ax0, bx0));
    const float iy = std::max(0.f, std::min(ay1, by1) - std::max(ay0, by0));
    const float inter = ix * iy;
    const float uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("usage: %s <fp32_engine> <fp16_engine> <image> [images...]\n",
               argv[0]);
        return 2;
    }
    TrtEngine e32(argv[1]), e16(argv[2]);
    const auto d32 = e32.OutputDims(), d16 = e16.OutputDims();
    if (d32.nbDims != 3 || d16.nbDims != 3 ||
        d32.d[1] != d16.d[1] || d32.d[2] != d16.d[2]) {
        printf("✗ engines have mismatched head geometry\n");
        return 1;
    }
    const int classes = d32.d[1] - 4, anchors = d32.d[2];
    printf("=== fp16 vs fp32 stage-1 detection agreement ===\n");
    printf("fp32: %s | fp16: %s | head %d classes x %d anchors\n",
           argv[1], argv[2], classes, anchors);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    GpuDetection *d_cands, *d_kept;
    int *d_count, *d_kept_count;
    cudaMalloc(&d_cands, (size_t)anchors * sizeof(GpuDetection));
    cudaMalloc(&d_kept, (size_t)kMaxNmsCandidates * sizeof(GpuDetection));
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_kept_count, sizeof(int));

    int images = 0, matched = 0;
    float worst_iou = 1.f, worst_score = 0.f;
    for (int a = 3; a < argc; a++) {
        cv::Mat bgr = cv::imread(argv[a]);
        if (bgr.empty()) {
            printf("  ! skip unreadable %s\n", argv[a]);
            continue;
        }
        images++;
        LetterboxInfo lb;
        const std::vector<float> in = Letterbox(bgr, 640, &lb);
        auto r32 = RunOne(e32, in, lb, bgr.cols, bgr.rows, anchors, classes,
                          stream, d_cands, d_count, d_kept, d_kept_count);
        auto r16 = RunOne(e16, in, lb, bgr.cols, bgr.rows, anchors, classes,
                          stream, d_cands, d_count, d_kept, d_kept_count);

        bool ok = r32.size() == r16.size();
        float min_iou = 1.f, mscore = 0.f;
        for (size_t k = 0; ok && k < r32.size(); k++) {
            // detections come out in the same sorted order (score desc)
            const auto& A = r32[k];
            const auto& B = r16[k];
            const float iou = Iou(A, B);
            const float ds = std::fabs(A.score - B.score);
            min_iou = std::min(min_iou, iou);
            mscore = std::max(mscore, ds);
            if (A.cls != B.cls || iou < kMatchIou) ok = false;
        }
        // track worst (lowest) IOU and worst score across all images
        if (!r32.empty()) worst_iou = std::min(worst_iou, min_iou);
        worst_score = std::max(worst_score, mscore);
        if (ok) matched++;
        printf("  %-28s fp32 %zu dets, fp16 %zu dets | min IOU %.4f, "
               "max score %.4f %s\n",
               argv[a], r32.size(), r16.size(), r32.empty() ? 1.f : min_iou,
               mscore, ok ? "✓" : "✗ MISMATCH");
    }

    printf("---- agreement: %d/%d images | worst IOU %.4f, "
           "worst score drift %.4f ----\n",
           matched, images, worst_iou, worst_score);
    printf("%s\n", matched == images && images > 0
                       ? "✓ fp16 preserves fp32 detections within tolerance"
                       : "✗ fp16 diverges — inspect above");
    return matched == images && images > 0 ? 0 : 1;
}
