// Cascade checkpoint: batched NV12 crop+resize kernel vs a CPU reference.
//
// Same philosophy as postprocess_batch_test: synthetic inputs (LCG-filled
// pitched NV12 frames - no decoder, no network), a plain-float host mirror
// of the kernel math, and a bit-exact comparison. Covers the cases stage 2
// will hit: plate-like wide boxes, different source frames per crop in one
// batch, rects touching frame corners, degenerate 1x1 rects, full-frame
// rects, and odd (unaligned) coordinates for the 4:2:0 chroma path.
//
// M3a: the CPU reference and kernel call both take PER-CHANNEL
// norm_offset[3]/norm_scale[3] (upgrade of M1a's single scalar pair - see
// graph.h's StepDesc). Two norm passes run over the same case list: the
// EQUAL-per-channel case (offset/scale identical across all 3 channels,
// same values as before M3a) - this must stay bit-exact, proving the
// per-channel kernel reproduces the old scalar arithmetic exactly - and a
// genuinely PER-CHANNEL case (distinct offset/scale per channel, modeled
// on true ImageNet norm) which must also be bit-exact against the
// per-channel CPU mirror.
//
// Usage: ./crop_test   (no arguments, exits nonzero on mismatch)

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "preprocess.h"

namespace {

void CheckCuda(cudaError_t r, const char* what) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA: %s failed: %s\n", what, cudaGetErrorString(r));
        std::exit(-1);
    }
}

constexpr int kOutW = 94, kOutH = 24;  // LPRNet input geometry

// M3a: two norm regimes exercised across every crop case below.
// EQUAL-per-channel: today's scalar LPRNet norm, replicated across all 3
// channels - proves the per-channel kernel/CPU-mirror reproduce the
// pre-M3a scalar arithmetic bit-for-bit (back-compat).
constexpr float kEqualOffset[3] = {-127.5f, -127.5f, -127.5f};
constexpr float kEqualScale[3] = {0.0078125f, 0.0078125f, 0.0078125f};  // 1/128
// PER-CHANNEL: distinct offset/scale per channel, modeled on true ImageNet
// norm (offset = -mean*255, scale = 1/(std*255), RGB order - see
// python/export_classifier.py) - proves the new per-channel path itself.
constexpr float kPerChanOffset[3] = {-123.675f, -116.28f, -103.53f};
constexpr float kPerChanScale[3] = {1.f / 58.395f, 1.f / 57.12f,
                                    1.f / 57.375f};

float3 F3(const float a[3]) { return make_float3(a[0], a[1], a[2]); }

// Deterministic byte pattern, same generator family as engine_batch_test.
uint32_t g_lcg = 0x2026'0707u;
uint8_t NextByte() {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return static_cast<uint8_t>(g_lcg >> 24);
}

struct HostFrame {
    int w, h, pitch;
    std::vector<uint8_t> data;  // pitch * h * 3/2 rows, luma then chroma
};

HostFrame MakeFrame(int w, int h, int pitch) {
    HostFrame f{w, h, pitch, {}};
    f.data.resize(static_cast<size_t>(pitch) * (h + h / 2));
    for (auto& b : f.data) b = NextByte();
    return f;
}

// Host mirror of Nv12CropResizeKernel - same float ops, same order. The
// kernel pins its mul-add chains to unfused round-to-nearest (__fmul_rn/
// __fadd_rn) precisely so this plain-float mirror is bit-exact; plain
// x86 float ops round the same way.
float Lerp(float a, float b, float t) {
    return a * (1.f - t) + b * t;
}

void CpuCropResize(const HostFrame& f, int cx, int cy, int cw, int ch,
                   float* out, const float norm_offset[3],
                   const float norm_scale[3], bool rgb = false) {
    const int plane = kOutW * kOutH;
    for (int y = 0; y < kOutH; y++) {
        for (int x = 0; x < kOutW; x++) {
            float sx = (x + 0.5f) * (float)cw / kOutW - 0.5f;
            float sy = (y + 0.5f) * (float)ch / kOutH - 0.5f;
            sx = std::fmin(std::fmax(sx, 0.f), cw - 1.f);
            sy = std::fmin(std::fmax(sy, 0.f), ch - 1.f);

            const int x0l = static_cast<int>(sx);
            const int y0l = static_cast<int>(sy);
            const float fx = sx - x0l;
            const float fy = sy - y0l;
            const int x0 = cx + x0l;
            const int y0 = cy + y0l;
            const int x1 = cx + std::min(x0l + 1, cw - 1);
            const int y1 = cy + std::min(y0l + 1, ch - 1);

            const uint8_t* yp = f.data.data();
            const float Y =
                Lerp(Lerp(yp[y0 * f.pitch + x0], yp[y0 * f.pitch + x1], fx),
                     Lerp(yp[y1 * f.pitch + x0], yp[y1 * f.pitch + x1], fx),
                     fy);

            const uint8_t* uvp = f.data.data() + (size_t)f.pitch * f.h;
            const int uv_row = y0 >> 1;
            const int uv_col = (x0 >> 1) << 1;
            const float U = uvp[uv_row * f.pitch + uv_col] - 128.f;
            const float V = uvp[uv_row * f.pitch + uv_col + 1] - 128.f;

            const float Yl = 1.164f * (Y - 16.f);
            const float r = std::fmin(std::fmax(Yl + 1.596f * V, 0.f), 255.f);
            const float g =
                std::fmin(std::fmax(Yl - 0.392f * U - 0.813f * V, 0.f), 255.f);
            const float b = std::fmin(std::fmax(Yl + 2.017f * U, 0.f), 255.f);

            const int idx = y * kOutW + x;
            const float c0 = rgb ? r : b;
            const float c2 = rgb ? b : r;
            out[idx] = (c0 + norm_offset[0]) * norm_scale[0];
            out[plane + idx] = (g + norm_offset[1]) * norm_scale[1];
            out[2 * plane + idx] = (c2 + norm_offset[2]) * norm_scale[2];
        }
    }
}

}  // namespace

int main() {
    printf("=== Cascade checkpoint: NV12 crop+resize (batched) ===\n");

    // Two frames with distinct geometries/pitches: 720p-like (the camera)
    // and the odd-sized 802x543 case the postprocess tests also use.
    HostFrame f0 = MakeFrame(1280, 720, 1536);
    HostFrame f1 = MakeFrame(802, 543, 1024);

    uint8_t *d_f0, *d_f1;
    CheckCuda(cudaMalloc(&d_f0, f0.data.size()), "malloc f0");
    CheckCuda(cudaMalloc(&d_f1, f1.data.size()), "malloc f1");
    CheckCuda(cudaMemcpy(d_f0, f0.data.data(), f0.data.size(),
                         cudaMemcpyHostToDevice), "upload f0");
    CheckCuda(cudaMemcpy(d_f1, f1.data.data(), f1.data.size(),
                         cudaMemcpyHostToDevice), "upload f1");

    // One stage-2 "batch" mixing both frames, incl. every edge case.
    struct Case { const HostFrame* f; const uint8_t* d; int x, y, w, h;
                  const char* name; };
    const Case cases[] = {
        {&f0, d_f0, 478, 341, 502, 195, "plate-like wide box"},
        {&f0, d_f0, 0, 0, 1280, 720, "full frame"},
        {&f0, d_f0, 1277, 717, 3, 3, "bottom-right corner, 3x3"},
        {&f0, d_f0, 640, 360, 1, 1, "degenerate 1x1"},
        {&f1, d_f1, 33, 77, 251, 63, "odd coords, odd frame"},
        {&f1, d_f1, 0, 540, 802, 3, "bottom edge strip"},
    };
    const int n = sizeof(cases) / sizeof(cases[0]);

    std::vector<CropParams> params(n);
    for (int i = 0; i < n; i++) {
        params[i] = {cases[i].d, cases[i].f->pitch, cases[i].f->w,
                     cases[i].f->h, cases[i].x, cases[i].y, cases[i].w,
                     cases[i].h};
    }
    CropParams* d_params;
    CheckCuda(cudaMalloc(&d_params, n * sizeof(CropParams)), "malloc params");
    CheckCuda(cudaMemcpy(d_params, params.data(), n * sizeof(CropParams),
                         cudaMemcpyHostToDevice), "upload params");

    const int per_crop = 3 * kOutW * kOutH;
    float* d_out;
    CheckCuda(cudaMalloc(&d_out, (size_t)n * per_crop * sizeof(float)),
              "malloc out");

    std::vector<float> gpu((size_t)n * per_crop);
    std::vector<float> cpu(per_crop);
    int failures = 0;

    // M3a matrix: EQUAL-per-channel (scalar back-compat) and genuinely
    // PER-CHANNEL norm, each run in both BGR (LPRNet/cascade default) and
    // RGB (SAHI tile) plane order - 4 passes over the same 6 crop cases,
    // every one bit-exact against the CPU mirror.
    struct NormCase { const char* name; const float* offset; const float* scale; };
    const NormCase norm_cases[] = {
        {"equal/scalar-back-compat", kEqualOffset, kEqualScale},
        {"per-channel/ImageNet-like", kPerChanOffset, kPerChanScale},
    };

    for (const NormCase& nc : norm_cases) {
        for (int rgb_mode = 0; rgb_mode < 2; rgb_mode++) {
            const bool rgb = rgb_mode != 0;
            LaunchNv12CropResizeBatched(d_params, n, d_out, kOutW, kOutH,
                                        F3(nc.offset), F3(nc.scale), nullptr,
                                        rgb);
            CheckCuda(cudaDeviceSynchronize(), "kernel");
            CheckCuda(cudaMemcpy(gpu.data(), d_out, gpu.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost), "download");
            for (int i = 0; i < n; i++) {
                CpuCropResize(*cases[i].f, cases[i].x, cases[i].y,
                              cases[i].w, cases[i].h, cpu.data(), nc.offset,
                              nc.scale, rgb);
                int bad = 0;
                for (int e = 0; e < per_crop; e++) {
                    if (gpu[(size_t)i * per_crop + e] != cpu[e]) bad++;
                }
                printf("%s crop %d %s%s (%s): %s\n", bad ? "✗" : "✓", i,
                       nc.name, rgb ? " RGB" : "", cases[i].name,
                       bad ? "MISMATCH" : "bit-exact");
                if (bad) {
                    printf("   %d/%d elements differ\n", bad, per_crop);
                    failures++;
                }
            }
        }
    }

    // Timing: a plausible stage-2 batch (16 plate crops) end to end.
    const int tn = 16;
    std::vector<CropParams> tparams(tn, params[0]);
    CropParams* d_tparams;
    CheckCuda(cudaMalloc(&d_tparams, tn * sizeof(CropParams)), "malloc tp");
    CheckCuda(cudaMemcpy(d_tparams, tparams.data(), tn * sizeof(CropParams),
                         cudaMemcpyHostToDevice), "upload tp");
    float* d_tout;
    CheckCuda(cudaMalloc(&d_tout, (size_t)tn * per_crop * sizeof(float)),
              "malloc tout");
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    LaunchNv12CropResizeBatched(d_tparams, tn, d_tout, kOutW, kOutH,
                                F3(kEqualOffset), F3(kEqualScale),
                                nullptr);  // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int r = 0; r < 100; r++) {
        LaunchNv12CropResizeBatched(d_tparams, tn, d_tout, kOutW, kOutH,
                                    F3(kEqualOffset), F3(kEqualScale),
                                    nullptr);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("timing: %d crops -> %dx%d in %.4f ms/batch (100-run avg)\n", tn,
           kOutW, kOutH, ms / 100.f);

    cudaFree(d_f0); cudaFree(d_f1); cudaFree(d_params); cudaFree(d_out);
    cudaFree(d_tparams); cudaFree(d_tout);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    if (failures) {
        printf("✗ CHECKPOINT FAILED (%d crops mismatched)\n", failures);
        return 1;
    }
    printf("✓ Checkpoint passed: GPU crop+resize bit-exact vs CPU reference\n");
    return 0;
}
