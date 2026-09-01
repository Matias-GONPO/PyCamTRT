// Cascade checkpoint: LPRNet TensorRT engine vs the PyTorch reference.
//
// export_lprnet.py dumps a deterministic batch-5 input and the export
// model's logits for it (test_data/lprnet_ref_*.bin). This test runs the
// same input through the built engine and requires:
//   1. logits parity within TF32 tolerance (same bar as engine_batch_test:
//      TRT tactics + Ampere TF32 vs CPU torch give ~1% element drift), and
//   2. IDENTICAL greedy-CTC decodes - the semantic result the cascade
//      actually consumes must not depend on that numeric drift.
//
// Usage: ./lprnet_test [engine] (default models/lprnet_b1-32_fp32_sm86.engine,
//        run from the repo root so test_data/ resolves)

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "TrtEngine.h"
#include "lprnet_ctc.h"

namespace {

constexpr int kBatch = 5;
constexpr float kRelTol = 2e-2f;

std::vector<float> ReadBin(const char* path, size_t count) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s (run from the repo root; generate "
                        "with export_lprnet.py)\n", path);
        std::exit(-1);
    }
    std::vector<float> v(count);
    if (fread(v.data(), sizeof(float), count, f) != count) {
        fprintf(stderr, "%s: short read\n", path);
        std::exit(-1);
    }
    fclose(f);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    const char* engine_path =
        argc > 1 ? argv[1] : "models/lprnet_b1-32_fp32_sm86.engine";
    printf("=== Cascade checkpoint: LPRNet engine vs PyTorch reference ===\n");

    TrtEngine engine(engine_path);
    printf("✓ Engine: %s (max batch %d, in %zu, out %zu floats/img)\n",
           engine_path, engine.MaxBatch(), engine.InputCount(),
           engine.OutputCount());
    const size_t in_count = engine.InputCount();     // 3*24*94
    const size_t out_count = engine.OutputCount();   // 68*18

    const auto input =
        ReadBin("test_data/lprnet_ref_input_5x3x24x94.bin", kBatch * in_count);
    const auto ref =
        ReadBin("test_data/lprnet_ref_output_5x68x18.bin", kBatch * out_count);

    engine.SetBatch(kBatch);
    cudaMemcpy(engine.InputPtr(), input.data(),
               input.size() * sizeof(float), cudaMemcpyHostToDevice);
    engine.Infer(nullptr);
    cudaDeviceSynchronize();
    std::vector<float> got(kBatch * out_count);
    cudaMemcpy(got.data(), engine.OutputPtr(), got.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 1. Numeric parity.
    float max_rel = 0.f, max_abs = 0.f;
    for (size_t i = 0; i < got.size(); i++) {
        const float a = std::fabs(got[i] - ref[i]);
        max_abs = std::fmax(max_abs, a);
        max_rel = std::fmax(max_rel, a / std::fmax(std::fabs(ref[i]), 1.f));
    }
    printf("logits parity: max abs %.4f, max rel %.4f (tol %.3f) %s\n",
           max_abs, max_rel, kRelTol, max_rel < kRelTol ? "✓" : "✗");

    // 2. Semantic parity: greedy CTC decode must match exactly.
    int decode_fail = 0;
    for (int b = 0; b < kBatch; b++) {
        const auto d_ref = CtcGreedyDecode(ref.data() + b * out_count);
        const auto d_got = CtcGreedyDecode(got.data() + b * out_count);
        const bool ok = d_ref == d_got;
        if (!ok) decode_fail++;
        printf("%s crop %d: engine \"%s\" vs reference \"%s\"\n",
               ok ? "✓" : "✗", b, LprLabelString(d_got).c_str(),
               LprLabelString(d_ref).c_str());
    }

    // Timing at the opt batch.
    const int tb = 8;
    engine.SetBatch(tb);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    engine.Infer(nullptr);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++) engine.Infer(nullptr);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("timing: batch %d in %.3f ms (100-run avg)\n", tb, ms / 100.f);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    if (max_rel >= kRelTol || decode_fail) {
        printf("✗ CHECKPOINT FAILED\n");
        return 1;
    }
    printf("✓ Checkpoint passed: engine matches PyTorch reference "
           "(numerically within TF32 tolerance, semantically exact)\n");
    return 0;
}
