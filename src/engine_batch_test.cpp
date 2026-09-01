// Step 5 checkpoint (part B): dynamic-batch engine vs the static reference.
//
// Part 1  batch-1 parity: the same deterministic input through the static
//         engine and the dynamic engine (batch=1) must produce the same
//         output tensor (fp tolerance: different builder tactics may
//         reorder float math, so equality is near-, not bit-).
// Part 2  batch consistency: a batch of 4 distinct inputs through the
//         dynamic engine must match the same 4 inputs run one-by-one at
//         batch=1 on the same engine. Slot isolation + shape switching.
// Part 3  (informational) throughput: batched inference vs sequential
//         batch-1 calls, the number Step 5 exists to win.
//
// Inputs are pseudo-random tensors (fixed LCG seed) — this diffs engine
// math, not detection quality, so no image/preprocess dependency.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "TrtEngine.h"

namespace {

// Deterministic filler: distinct pattern per seed, values in [0,1).
void FillPattern(std::vector<float>* v, uint32_t seed) {
    uint32_t s = seed * 2654435761u + 1u;
    for (auto& x : *v) {
        s = s * 1664525u + 1013904223u;
        x = (s >> 8) * (1.0f / 16777216.0f);
    }
}

struct Diff {
    float max_abs = 0.f;
    float max_rel = 0.f;
};

Diff Compare(const std::vector<float>& a, const std::vector<float>& b) {
    Diff d;
    for (size_t i = 0; i < a.size(); i++) {
        const float abs = std::fabs(a[i] - b[i]);
        const float rel = abs / std::max(1.0f, std::fabs(a[i]));
        if (abs > d.max_abs) d.max_abs = abs;
        if (rel > d.max_rel) d.max_rel = rel;
    }
    return d;
}

// Same-engine comparisons (part 2) are deterministic in practice; keep tight.
constexpr float kRelTol = 1e-3f;
// Part 1 compares two DIFFERENT builds of the same network. TensorRT picks
// different tactics per build and enables TF32 on Ampere by default, so
// element-wise drift of ~1% over yolov8n's depth is expected and benign —
// measured 0.0100 max rel on this hardware. Detection-level equivalence is
// verified separately against the CPU reference (rtsp_infer --verify).
constexpr float kCrossBuildRelTol = 2e-2f;

void RunBatch1(TrtEngine& eng, const std::vector<float>& in,
               std::vector<float>* out, cudaStream_t stream) {
    eng.SetBatch(1);
    cudaMemcpyAsync(eng.InputPtr(), in.data(), in.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    eng.Infer(stream);
    cudaMemcpyAsync(out->data(), eng.OutputPtr(), out->size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string static_path = argc > 1 ? argv[1] : "models/yolov8n.engine";
    const std::string dynamic_path =
        argc > 2 ? argv[2] : "models/yolov8n_b1-16_fp32_sm86.engine";

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    printf("=== Dynamic-batch engine checkpoint ===\n");
    TrtEngine stat(static_path);
    TrtEngine dyn(dynamic_path);
    printf("static:  %s (max batch %d)\n", static_path.c_str(), stat.MaxBatch());
    printf("dynamic: %s (max batch %d)\n", dynamic_path.c_str(), dyn.MaxBatch());
    if (stat.InputCount() != dyn.InputCount() ||
        stat.OutputCount() != dyn.OutputCount()) {
        printf("✗ FAIL: per-image element counts differ between engines\n");
        return 1;
    }
    const size_t in_n = dyn.InputCount(), out_n = dyn.OutputCount();

    // ---- Part 1: batch-1 parity, static vs dynamic --------------------
    std::vector<float> input(in_n), out_static(out_n), out_dyn(out_n);
    FillPattern(&input, 1);
    RunBatch1(stat, input, &out_static, stream);
    RunBatch1(dyn, input, &out_dyn, stream);
    const Diff p1 = Compare(out_static, out_dyn);
    const bool ok1 = p1.max_rel < kCrossBuildRelTol;
    printf("[part 1] batch-1 parity static vs dynamic (cross-build, TF32): "
           "max abs %.6f, max rel %.6f %s\n",
           p1.max_abs, p1.max_rel, ok1 ? "✓" : "✗ FAIL");

    // ---- Part 2: batch-4 slices vs sequential batch-1 ------------------
    const int kB = 4;
    std::vector<std::vector<float>> inputs(kB, std::vector<float>(in_n));
    for (int i = 0; i < kB; i++) FillPattern(&inputs[i], 100 + i);

    dyn.SetBatch(kB);
    for (int i = 0; i < kB; i++) {
        cudaMemcpyAsync(dyn.InputPtr(i), inputs[i].data(),
                        in_n * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    dyn.Infer(stream);
    std::vector<std::vector<float>> batched(kB, std::vector<float>(out_n));
    for (int i = 0; i < kB; i++) {
        cudaMemcpyAsync(batched[i].data(), dyn.OutputPtr(i),
                        out_n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    bool ok2 = true;
    for (int i = 0; i < kB; i++) {
        std::vector<float> solo(out_n);
        RunBatch1(dyn, inputs[i], &solo, stream);
        const Diff d = Compare(batched[i], solo);
        const bool ok = d.max_rel < kRelTol;
        ok2 = ok2 && ok;
        printf("[part 2] slot %d batched vs solo: max abs %.6f, max rel %.6f %s\n",
               i, d.max_abs, d.max_rel, ok ? "✓" : "✗ FAIL");
    }

    // ---- Part 3: throughput, batched vs sequential (informational) -----
    printf("[part 3] throughput (fp32, 100 timed iterations each):\n");
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (int b : {1, 4, 8, 16}) {
        if (b > dyn.MaxBatch()) break;
        dyn.SetBatch(b);
        for (int w = 0; w < 20; w++) dyn.Infer(stream);  // warmup this shape
        cudaStreamSynchronize(stream);
        cudaEventRecord(t0, stream);
        for (int it = 0; it < 100; it++) dyn.Infer(stream);
        cudaEventRecord(t1, stream);
        cudaStreamSynchronize(stream);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, t0, t1);
        printf("         batch %2d: %7.3f ms/batch  %6.3f ms/image  %7.1f img/s\n",
               b, ms / 100.0, ms / 100.0 / b, 100.0 * b * 1000.0 / ms);
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaStreamDestroy(stream);

    if (ok1 && ok2) {
        printf("✓ PASS: dynamic-batch engine checkpoint.\n");
        return 0;
    }
    printf("✗ FAIL: dynamic-batch engine checkpoint.\n");
    return 1;
}
