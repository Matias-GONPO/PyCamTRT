// Step 5 checkpoint (part C): batched postprocess kernels vs the verified
// single-image kernels, bit-exact.
//
// Builds 4 synthetic raw tensors [84, 8400] with *planted* detections
// (uniform noise would put every anchor above threshold and overflow the
// 1024-candidate cap, where nondeterministic compaction order would make
// the comparison meaningless). Per image:
//   img 0: typical scene, ~60 boxes, 1280x720 letterbox geometry
//   img 1: crowded overlapping boxes (NMS suppression exercised), same geo
//   img 2: EMPTY (no anchor above threshold) - the zero path
//   img 3: different geometry (802x543, non-square) - per-image params
// Reference: the single-image LaunchBoxDecode + LaunchNms run per image on
// the same buffers. Batched output must match bit-for-bit: the kept list is
// deterministically ordered by the NMS sort, so memcmp-level equality holds.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "postprocess.h"

namespace {

constexpr int kAnchors = 8400;
constexpr int kClasses = 80;
constexpr int kBatch = 5;  // 4 original + 1 SAHI-offset (tile remap) case
constexpr float kScoreThresh = 0.4f;
constexpr float kIouThresh = 0.45f;

uint32_t Lcg(uint32_t* s) { return *s = *s * 1664525u + 1013904223u; }
float Unit(uint32_t* s) { return (Lcg(s) >> 8) * (1.0f / 16777216.0f); }

// Fill one [4+classes, anchors] tensor: sub-threshold noise everywhere,
// then `planted` anchors with a class score in [0.45, 0.95) and a box
// around a pseudo-random center. `spread` controls overlap (small spread =
// crowded scene = heavy suppression).
void BuildTensor(std::vector<float>* t, uint32_t seed, int planted,
                 float spread) {
    uint32_t s = seed;
    for (int c = 0; c < kClasses; c++)
        for (int i = 0; i < kAnchors; i++)
            (*t)[(4 + c) * kAnchors + i] = Unit(&s) * 0.3f;
    for (int i = 0; i < kAnchors; i++) {
        (*t)[0 * kAnchors + i] = Unit(&s) * 640.f;         // cx
        (*t)[1 * kAnchors + i] = Unit(&s) * 640.f;         // cy
        (*t)[2 * kAnchors + i] = 8.f + Unit(&s) * 100.f;   // w
        (*t)[3 * kAnchors + i] = 8.f + Unit(&s) * 100.f;   // h
    }
    for (int p = 0; p < planted; p++) {
        const int anchor = Lcg(&s) % kAnchors;
        const int cls = Lcg(&s) % kClasses;
        (*t)[(4 + cls) * kAnchors + anchor] = 0.45f + Unit(&s) * 0.5f;
        // Cluster centers so IoU suppression actually triggers.
        (*t)[0 * kAnchors + anchor] = 320.f + Unit(&s) * spread;
        (*t)[1 * kAnchors + anchor] = 320.f + Unit(&s) * spread;
    }
}

bool SameDetections(const std::vector<GpuDetection>& a, int na,
                    const std::vector<GpuDetection>& b, int nb) {
    if (na != nb) return false;
    return std::memcmp(a.data(), b.data(), na * sizeof(GpuDetection)) == 0;
}

// ---- M1a: argmax classifier family checkpoint --------------------------
// Same house style as the box-decode/NMS checkpoint above: synthetic
// [batch, classes] logits with a *planted* maximum per row (strictly
// greater than every other value in that row, so the winner is
// unambiguous - no tie-breaking rule needs to match to get a bit-exact
// comparison) plus LCG noise elsewhere. Reference: a plain CPU linear scan
// per row - the same argmax-over-classes loop ArgmaxKernel runs on-device.
constexpr int kArgmaxBatch = 37;    // odd, not block-aligned (block = 256)
constexpr int kArgmaxClasses = 1000;  // ImageNet-scale, small enough for a
                                      // per-thread sequential scan

struct ArgmaxRef {
    int label;
    float score;
};

ArgmaxRef CpuArgmax(const float* row, int classes) {
    int best_cls = 0;
    float best = row[0];
    for (int c = 1; c < classes; c++) {
        if (row[c] > best) {
            best = row[c];
            best_cls = c;
        }
    }
    return {best_cls, best};
}

// ---- M4a: yolo-e2e detector family checkpoint ---------------------------
// Same house style again: synthetic [batch, max_dets, 6] row-major tensors
// (see postprocess.h's LaunchYoloE2EBatched WHY-comment for the verified
// (x1,y1,x2,y2,score,cls) row layout) with sub-threshold LCG noise in every
// row, then specific PLANTED rows overwritten with known values covering:
// interior boxes needing no clamp, all four clamp directions (left/top/
// right/bottom - some via deliberately out-of-[0,640] letterbox coords, a
// robustness check on the clamp math itself, not a claim a real model
// emits those), an off_x/off_y tile-translation case (SAHI-shaped, even
// though SAHI+yolo-e2e is rejected one layer up in pipeline.cpp - the
// KERNEL still honors off_x/off_y per its WHY-comment), a non-square
// geometry, and one EMPTY image (all rows below threshold - the zero
// path, mirroring the box-decode/NMS checkpoint above).
//
// UNLIKE that checkpoint (whose kept list is deterministically ORDERED by
// the NMS bitonic sort) this family has NO NMS/sort stage - survivors land
// in d_out via a raw atomicAdd race, so entry ORDER is nondeterministic
// across launches even though the SET is deterministic (the same
// nondeterminism this file's own top comment flags for uniform-noise
// candidate counts, here inherent to the family rather than caused by
// picking bad test data). Fix: every planted row in an image gets a
// STRICTLY DISTINCT score, so sorting both the kernel's output and the CPU
// reference by score-descending (the same tie-break-first field
// postprocess.cu's device Before() and pipeline.cpp's CpuReference() both
// sort by) yields one canonical order for a true bit-exact memcmp - no
// tolerance, since sorting reorders structs without touching their bits.
constexpr int kE2EBatch = 4;
constexpr int kE2EMaxDets = 300;  // matches yolo26n's real head exactly
constexpr float kE2EScoreThresh = 0.4f;

struct PlantedRow {
    int row;  // which of the kE2EMaxDets rows to overwrite
    float x1, y1, x2, y2, score;
    int cls;
};

// img 0: typical 1280x720 letterbox geometry, every planted box interior
// (no clamp exercised here - that's img 1's job).
const PlantedRow kPlant0[] = {
    {0, 5.f, 141.f, 15.f, 150.f, 0.43f, 42},
    {10, 100.f, 200.f, 300.f, 400.f, 0.41f, 0},
    {50, 400.f, 180.f, 600.f, 500.f, 0.55f, 5},
    {120, 50.f, 150.f, 150.f, 250.f, 0.90f, 17},
    {200, 300.f, 160.f, 340.f, 200.f, 0.42f, 79},
    {299, 10.f, 145.f, 630.f, 495.f, 0.99f, 1},
};
// img 1: 802x543 non-square geometry (same numbers as the box-decode/NMS
// checkpoint's own img 3, for a realistic non-square case) - every planted
// row exercises a DIFFERENT clamp direction: left (x1 < 0 pre-clamp),
// top (y1 < pad_y), right (x2 past src_w), bottom (y2 past src_h), and one
// more bottom-clamp on a near-full-frame box.
const PlantedRow kPlant1[] = {
    {5, -10.f, 200.f, 100.f, 300.f, 0.44f, 3},     // left clamp
    {60, 300.f, 50.f, 400.f, 150.f, 0.47f, 9},     // top clamp
    {150, 500.f, 300.f, 655.f, 400.f, 0.51f, 11},  // right clamp
    {250, 200.f, 300.f, 300.f, 650.f, 0.63f, 15},  // bottom clamp
    {299, 0.f, 103.f, 640.f, 640.f, 0.77f, 20},    // full-frame, bottom clamp
};
// img 2: EMPTY - no planted rows at all, every row stays sub-threshold
// noise (the zero-survivor path).
// img 3: identity letterbox (SAHI-tile-shaped), off_x/off_y translation -
// mirrors the box-decode/NMS checkpoint's own img 4 offset (1920, 1080).
const PlantedRow kPlant3[] = {
    {30, 10.f, 10.f, 100.f, 100.f, 0.50f, 2},
    {100, 300.f, 300.f, 639.f, 639.f, 0.65f, 7},
    {299, 0.f, 0.f, 640.f, 640.f, 0.85f, 3},
};

void BuildE2ETensor(std::vector<float>* t, uint32_t seed,
                    const PlantedRow* planted, int n_planted) {
    uint32_t s = seed;
    for (int i = 0; i < kE2EMaxDets; i++) {
        float* row = t->data() + (size_t)i * 6;
        row[0] = Unit(&s) * 640.f;
        row[1] = Unit(&s) * 640.f;
        row[2] = row[0] + Unit(&s) * 100.f;
        row[3] = row[1] + Unit(&s) * 100.f;
        row[4] = Unit(&s) * 0.3f;  // strictly < kE2EScoreThresh (0.4)
        row[5] = (float)(Lcg(&s) % 80);
    }
    for (int p = 0; p < n_planted; p++) {
        float* row = t->data() + (size_t)planted[p].row * 6;
        row[0] = planted[p].x1;
        row[1] = planted[p].y1;
        row[2] = planted[p].x2;
        row[3] = planted[p].y2;
        row[4] = planted[p].score;
        row[5] = (float)planted[p].cls;
    }
}

// Independent from-scratch reference: the SAME un-letterbox affine
// DecodeAnchor/YoloE2EBatchedKernel use (`(coord - pad) / scale`, applied
// to all four xyxy columns directly - see postprocess.h's WHY-comment),
// written with std::max/std::min/std::lround instead of the device
// fmaxf/fminf/roundf intrinsics they mirror - both are plain IEEE-754
// single-op float arithmetic, so results are bit-identical (same
// assumption pipeline.cpp's own CpuReference() already relies on for the
// yolo family checkpoint below).
std::vector<GpuDetection> CpuYoloE2E(const float* raw, int max_dets,
                                     const PostprocImageParams& p,
                                     float thresh) {
    std::vector<GpuDetection> out;
    for (int i = 0; i < max_dets; i++) {
        const float* row = raw + (size_t)i * 6;
        const float score = row[4];
        if (score < thresh) continue;
        float x0 = (row[0] - (float)p.lb.pad_x) / p.lb.scale;
        float y0 = (row[1] - (float)p.lb.pad_y) / p.lb.scale;
        float x1 = (row[2] - (float)p.lb.pad_x) / p.lb.scale;
        float y1 = (row[3] - (float)p.lb.pad_y) / p.lb.scale;
        x0 = std::max(x0, 0.f);
        y0 = std::max(y0, 0.f);
        x1 = std::min(x1, (float)p.src_w);
        y1 = std::min(y1, (float)p.src_h);
        GpuDetection d;
        d.x = x0 + (float)p.off_x;
        d.y = y0 + (float)p.off_y;
        d.w = std::max(x1 - x0, 0.f);
        d.h = std::max(y1 - y0, 0.f);
        d.score = score;
        d.cls = (int)std::lround(row[5]);
        out.push_back(d);
    }
    return out;
}

// Canonical order for comparing two nondeterministically-ordered kept
// lists: every planted score in this test is unique WITHIN an image (see
// the kPlant* tables above), so a plain score-descending sort is already
// unambiguous - no cls/x tie-break needed.
void SortByScoreDesc(std::vector<GpuDetection>* v) {
    std::sort(v->begin(), v->end(), [](const GpuDetection& a,
                                       const GpuDetection& b) {
        return a.score > b.score;
    });
}

}  // namespace

int main() {
    printf("=== Batched postprocess checkpoint ===\n");

    // Per-image geometry: 720p letterbox for 0-2, non-square source for 3,
    // and a SAHI tile for 4. Zero-init: off_x/off_y must be 0 for the
    // whole-frame images (field-by-field assignment below would otherwise
    // leave them as stack garbage).
    PostprocImageParams params[kBatch] = {};
    for (int i = 0; i < 3; i++) {
        params[i].lb = LetterboxInfo{0.5f, 0, 140};
        params[i].src_w = 1280;
        params[i].src_h = 720;
    }
    params[3].lb = LetterboxInfo{0.7980f, 0, 103};
    params[3].src_w = 802;
    params[3].src_h = 543;
    // img 4: a 640x640 SAHI tile whose origin sits at (1920, 1080) in a 4K
    // frame - identity letterbox (tile == engine input), boxes must come
    // out translated by the offset.
    params[4].lb = LetterboxInfo{1.0f, 0, 0};
    params[4].src_w = 640;
    params[4].src_h = 640;
    params[4].off_x = 1920;
    params[4].off_y = 1080;

    const int planted[kBatch] = {60, 200, 0, 40, 50};
    const float spread[kBatch] = {600.f, 60.f, 0.f, 300.f, 400.f};

    const size_t img_elems = (size_t)(4 + kClasses) * kAnchors;
    std::vector<float> host((size_t)kBatch * img_elems);
    for (int i = 0; i < kBatch; i++) {
        std::vector<float> t(img_elems);
        BuildTensor(&t, 7u * (i + 1), planted[i], spread[i]);
        std::memcpy(host.data() + i * img_elems, t.data(),
                    img_elems * sizeof(float));
    }

    float* d_raw;
    GpuDetection *d_cands, *d_kept;
    int *d_counts, *d_kept_counts;
    PostprocImageParams* d_params;
    cudaMalloc(&d_raw, host.size() * sizeof(float));
    cudaMalloc(&d_cands, (size_t)kBatch * kAnchors * sizeof(GpuDetection));
    cudaMalloc(&d_kept, (size_t)kBatch * kMaxNmsCandidates * sizeof(GpuDetection));
    cudaMalloc(&d_counts, kBatch * sizeof(int));
    cudaMalloc(&d_kept_counts, kBatch * sizeof(int));
    cudaMalloc(&d_params, sizeof(params));
    cudaMemcpy(d_raw, host.data(), host.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, params, sizeof(params), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Reference: verified single-image kernels, sequentially per image.
    std::vector<std::vector<GpuDetection>> ref_kept(
        kBatch, std::vector<GpuDetection>(kMaxNmsCandidates));
    int ref_counts[kBatch];
    for (int i = 0; i < kBatch; i++) {
        LaunchBoxDecode(d_raw + i * img_elems, kAnchors, kClasses,
                        params[i].lb, params[i].src_w, params[i].src_h,
                        kScoreThresh, d_cands + (size_t)i * kAnchors,
                        d_counts + i, stream);
        LaunchNms(d_cands + (size_t)i * kAnchors, d_counts + i, kIouThresh,
                  d_kept + (size_t)i * kMaxNmsCandidates, d_kept_counts + i,
                  stream);
        cudaMemcpyAsync(ref_kept[i].data(),
                        d_kept + (size_t)i * kMaxNmsCandidates,
                        kMaxNmsCandidates * sizeof(GpuDetection),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&ref_counts[i], d_kept_counts + i, sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    // Batched path on the same inputs (buffers reused; decode rewrites them).
    LaunchBoxDecodeBatched(d_raw, kBatch, kAnchors, kClasses, d_params,
                           kScoreThresh, d_cands, d_counts, stream);
    LaunchNmsBatched(d_cands, d_counts, kBatch, kAnchors, kIouThresh, d_kept,
                     d_kept_counts, stream);
    std::vector<std::vector<GpuDetection>> got_kept(
        kBatch, std::vector<GpuDetection>(kMaxNmsCandidates));
    int got_counts[kBatch];
    for (int i = 0; i < kBatch; i++) {
        cudaMemcpyAsync(got_kept[i].data(),
                        d_kept + (size_t)i * kMaxNmsCandidates,
                        kMaxNmsCandidates * sizeof(GpuDetection),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&got_counts[i], d_kept_counts + i, sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    bool ok = true;
    const char* label[kBatch] = {"typical", "crowded", "empty", "802x543 geo",
                                 "SAHI offset"};
    // The single-image reference path has no offset parameter (always 0),
    // so for the SAHI tile (img 4) its kept boxes are in TILE coords: add
    // the tile origin on the host. Same floats, same order of adds as the
    // kernel (x0 + off as float), so equality stays bit-exact.
    for (int k = 0; k < ref_counts[4]; k++) {
        ref_kept[4][k].x += (float)params[4].off_x;
        ref_kept[4][k].y += (float)params[4].off_y;
    }
    for (int i = 0; i < kBatch; i++) {
        const bool same =
            SameDetections(ref_kept[i], ref_counts[i], got_kept[i], got_counts[i]);
        ok = ok && same;
        printf("img %d (%-11s): single-image kept %3d | batched kept %3d | %s\n",
               i, label[i], ref_counts[i], got_counts[i],
               same ? "bit-exact ✓" : "MISMATCH ✗");
    }

    // Timing (informational): 4 sequential launch pairs vs 1 batched pair.
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    const int iters = 200;
    cudaEventRecord(t0, stream);
    for (int it = 0; it < iters; it++)
        for (int i = 0; i < kBatch; i++) {
            LaunchBoxDecode(d_raw + i * img_elems, kAnchors, kClasses,
                            params[i].lb, params[i].src_w, params[i].src_h,
                            kScoreThresh, d_cands + (size_t)i * kAnchors,
                            d_counts + i, stream);
            LaunchNms(d_cands + (size_t)i * kAnchors, d_counts + i, kIouThresh,
                      d_kept + (size_t)i * kMaxNmsCandidates,
                      d_kept_counts + i, stream);
        }
    cudaEventRecord(t1, stream);
    cudaStreamSynchronize(stream);
    float ms_seq;
    cudaEventElapsedTime(&ms_seq, t0, t1);
    cudaEventRecord(t0, stream);
    for (int it = 0; it < iters; it++) {
        LaunchBoxDecodeBatched(d_raw, kBatch, kAnchors, kClasses, d_params,
                               kScoreThresh, d_cands, d_counts, stream);
        LaunchNmsBatched(d_cands, d_counts, kBatch, kAnchors, kIouThresh,
                         d_kept, d_kept_counts, stream);
    }
    cudaEventRecord(t1, stream);
    cudaStreamSynchronize(stream);
    float ms_bat;
    cudaEventElapsedTime(&ms_bat, t0, t1);
    printf("timing (batch %d): sequential %.3f ms, batched %.3f ms per batch "
           "(%d kernel launches -> 2)\n",
           kBatch, ms_seq / iters, ms_bat / iters, 2 * kBatch);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaStreamDestroy(stream);
    cudaFree(d_raw);
    cudaFree(d_cands);
    cudaFree(d_kept);
    cudaFree(d_counts);
    cudaFree(d_kept_counts);
    cudaFree(d_params);

    printf(ok ? "✓ PASS: batched postprocess checkpoint.\n"
              : "✗ FAIL: batched postprocess checkpoint.\n");

    // ---- M1a: argmax classifier family checkpoint -----------------------
    printf("\n=== Argmax classifier checkpoint ===\n");
    std::vector<float> ax_host((size_t)kArgmaxBatch * kArgmaxClasses);
    std::vector<int> ax_planted(kArgmaxBatch);
    {
        uint32_t s = 424242u;
        for (int i = 0; i < kArgmaxBatch; i++) {
            for (int c = 0; c < kArgmaxClasses; c++)
                ax_host[(size_t)i * kArgmaxClasses + c] = Unit(&s) * 10.f - 5.f;
            const int p = Lcg(&s) % kArgmaxClasses;
            ax_host[(size_t)i * kArgmaxClasses + p] = 100.f + Unit(&s);  // planted max
            ax_planted[i] = p;
        }
        // Boundary cases: force the max to land at class index 0 (row 0)
        // and at the last class index (row 1) - exercises the loop's
        // initial-best and final-iteration edges, not just interior hits.
        for (int c = 0; c < kArgmaxClasses; c++)
            ax_host[0 * kArgmaxClasses + c] = (c == 0) ? 200.f : -5.f;
        ax_planted[0] = 0;
        for (int c = 0; c < kArgmaxClasses; c++)
            ax_host[1 * kArgmaxClasses + c] =
                (c == kArgmaxClasses - 1) ? 200.f : -5.f;
        ax_planted[1] = kArgmaxClasses - 1;
    }

    std::vector<ArgmaxRef> ax_ref(kArgmaxBatch);
    for (int i = 0; i < kArgmaxBatch; i++) {
        ax_ref[i] =
            CpuArgmax(ax_host.data() + (size_t)i * kArgmaxClasses, kArgmaxClasses);
    }
    // Sanity on the test's OWN construction (not the kernel): the CPU
    // reference must agree with what was planted, or the synthetic data
    // itself is broken and the bit-exact check below is meaningless.
    bool ax_construction_ok = true;
    for (int i = 0; i < kArgmaxBatch; i++)
        ax_construction_ok = ax_construction_ok && (ax_ref[i].label == ax_planted[i]);

    float* d_ax_logits;
    int* d_ax_labels;
    float* d_ax_scores;
    cudaMalloc(&d_ax_logits, ax_host.size() * sizeof(float));
    cudaMalloc(&d_ax_labels, kArgmaxBatch * sizeof(int));
    cudaMalloc(&d_ax_scores, kArgmaxBatch * sizeof(float));
    cudaMemcpy(d_ax_logits, ax_host.data(), ax_host.size() * sizeof(float),
              cudaMemcpyHostToDevice);

    cudaStream_t ax_stream;
    cudaStreamCreate(&ax_stream);
    LaunchArgmaxBatched(d_ax_logits, kArgmaxBatch, kArgmaxClasses, d_ax_labels,
                        d_ax_scores, ax_stream);
    std::vector<int> ax_got_labels(kArgmaxBatch);
    std::vector<float> ax_got_scores(kArgmaxBatch);
    cudaMemcpyAsync(ax_got_labels.data(), d_ax_labels,
                    kArgmaxBatch * sizeof(int), cudaMemcpyDeviceToHost,
                    ax_stream);
    cudaMemcpyAsync(ax_got_scores.data(), d_ax_scores,
                    kArgmaxBatch * sizeof(float), cudaMemcpyDeviceToHost,
                    ax_stream);
    cudaStreamSynchronize(ax_stream);

    // Bit-exact: both labels (int ==) and scores (float ==, no tolerance -
    // the GPU and CPU read the identical planted float value, no
    // arithmetic is performed on it by either side).
    int ax_mismatches = 0;
    for (int i = 0; i < kArgmaxBatch; i++) {
        const bool same = ax_got_labels[i] == ax_ref[i].label &&
                          ax_got_scores[i] == ax_ref[i].score;
        if (!same) ax_mismatches++;
    }
    const bool ax_ok = ax_construction_ok && ax_mismatches == 0;
    printf("construction check (CPU ref matches planted maxima): %s\n",
           ax_construction_ok ? "✓" : "✗");
    printf("batch %d, %d classes: %d/%d (label,score) pairs bit-exact vs CPU "
           "reference: %s\n",
           kArgmaxBatch, kArgmaxClasses, kArgmaxBatch - ax_mismatches,
           kArgmaxBatch, ax_ok ? "✓" : "✗");

    cudaStreamDestroy(ax_stream);
    cudaFree(d_ax_logits);
    cudaFree(d_ax_labels);
    cudaFree(d_ax_scores);

    printf(ax_ok ? "✓ PASS: argmax checkpoint.\n"
                : "✗ FAIL: argmax checkpoint.\n");

    // ---- M4a: yolo-e2e detector family checkpoint ------------------------
    printf("\n=== yolo-e2e detector checkpoint ===\n");
    PostprocImageParams e2e_params[kE2EBatch] = {};
    e2e_params[0].lb = LetterboxInfo{0.5f, 0, 140};
    e2e_params[0].src_w = 1280;
    e2e_params[0].src_h = 720;
    e2e_params[1].lb = LetterboxInfo{0.7980f, 0, 103};
    e2e_params[1].src_w = 802;
    e2e_params[1].src_h = 543;
    e2e_params[2].lb = LetterboxInfo{0.5f, 0, 140};  // empty image, geo irrelevant
    e2e_params[2].src_w = 1280;
    e2e_params[2].src_h = 720;
    e2e_params[3].lb = LetterboxInfo{1.0f, 0, 0};
    e2e_params[3].src_w = 640;
    e2e_params[3].src_h = 640;
    e2e_params[3].off_x = 1920;
    e2e_params[3].off_y = 1080;

    std::vector<float> e2e_host((size_t)kE2EBatch * kE2EMaxDets * 6);
    {
        std::vector<float> t0(kE2EMaxDets * 6), t1(kE2EMaxDets * 6),
            t2(kE2EMaxDets * 6), t3(kE2EMaxDets * 6);
        BuildE2ETensor(&t0, 101u, kPlant0, sizeof(kPlant0) / sizeof(kPlant0[0]));
        BuildE2ETensor(&t1, 202u, kPlant1, sizeof(kPlant1) / sizeof(kPlant1[0]));
        BuildE2ETensor(&t2, 303u, nullptr, 0);  // empty: no planted rows
        BuildE2ETensor(&t3, 404u, kPlant3, sizeof(kPlant3) / sizeof(kPlant3[0]));
        std::memcpy(e2e_host.data() + 0 * t0.size(), t0.data(), t0.size() * sizeof(float));
        std::memcpy(e2e_host.data() + 1 * t1.size(), t1.data(), t1.size() * sizeof(float));
        std::memcpy(e2e_host.data() + 2 * t2.size(), t2.data(), t2.size() * sizeof(float));
        std::memcpy(e2e_host.data() + 3 * t3.size(), t3.data(), t3.size() * sizeof(float));
    }

    // CPU reference, per image, BEFORE the GPU touches the buffer (same
    // ordering discipline as the box-decode/NMS checkpoint above).
    std::vector<std::vector<GpuDetection>> e2e_ref(kE2EBatch);
    for (int i = 0; i < kE2EBatch; i++) {
        e2e_ref[i] = CpuYoloE2E(e2e_host.data() + (size_t)i * kE2EMaxDets * 6,
                                kE2EMaxDets, e2e_params[i], kE2EScoreThresh);
        SortByScoreDesc(&e2e_ref[i]);
    }

    float* d_e2e_raw;
    GpuDetection* d_e2e_kept;
    int* d_e2e_kept_counts;
    PostprocImageParams* d_e2e_params;
    cudaMalloc(&d_e2e_raw, e2e_host.size() * sizeof(float));
    cudaMalloc(&d_e2e_kept, (size_t)kE2EBatch * kMaxNmsCandidates * sizeof(GpuDetection));
    cudaMalloc(&d_e2e_kept_counts, kE2EBatch * sizeof(int));
    cudaMalloc(&d_e2e_params, sizeof(e2e_params));
    cudaMemcpy(d_e2e_raw, e2e_host.data(), e2e_host.size() * sizeof(float),
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_e2e_params, e2e_params, sizeof(e2e_params),
              cudaMemcpyHostToDevice);

    cudaStream_t e2e_stream;
    cudaStreamCreate(&e2e_stream);
    LaunchYoloE2EBatched(d_e2e_raw, kE2EBatch, kE2EMaxDets, d_e2e_params,
                        kE2EScoreThresh, d_e2e_kept, d_e2e_kept_counts,
                        e2e_stream);
    std::vector<std::vector<GpuDetection>> e2e_got(
        kE2EBatch, std::vector<GpuDetection>(kMaxNmsCandidates));
    int e2e_got_counts[kE2EBatch];
    for (int i = 0; i < kE2EBatch; i++) {
        cudaMemcpyAsync(e2e_got[i].data(),
                        d_e2e_kept + (size_t)i * kMaxNmsCandidates,
                        kMaxNmsCandidates * sizeof(GpuDetection),
                        cudaMemcpyDeviceToHost, e2e_stream);
        cudaMemcpyAsync(&e2e_got_counts[i], d_e2e_kept_counts + i, sizeof(int),
                        cudaMemcpyDeviceToHost, e2e_stream);
    }
    cudaStreamSynchronize(e2e_stream);

    bool e2e_ok = true;
    const char* e2e_label[kE2EBatch] = {"typical", "802x543 clamps", "empty",
                                        "offset (1920,1080)"};
    for (int i = 0; i < kE2EBatch; i++) {
        e2e_got[i].resize(e2e_got_counts[i]);
        SortByScoreDesc(&e2e_got[i]);
        const bool same = SameDetections(e2e_ref[i], (int)e2e_ref[i].size(),
                                         e2e_got[i], e2e_got_counts[i]);
        e2e_ok = e2e_ok && same;
        printf("img %d (%-19s): CPU ref kept %3d | GPU kept %3d | %s\n", i,
               e2e_label[i], (int)e2e_ref[i].size(), e2e_got_counts[i],
               same ? "bit-exact ✓" : "MISMATCH ✗");
    }
    // Construction sanity (mirrors the argmax section's own check): image 2
    // must be genuinely empty, and every other image's kept count must
    // equal exactly its planted-row count (no noise row leaked through
    // score_thresh, no planted row got dropped) - otherwise the test data
    // itself is broken and the bit-exact comparison above is meaningless.
    const int n_planted[kE2EBatch] = {
        (int)(sizeof(kPlant0) / sizeof(kPlant0[0])),
        (int)(sizeof(kPlant1) / sizeof(kPlant1[0])), 0,
        (int)(sizeof(kPlant3) / sizeof(kPlant3[0]))};
    bool e2e_construction_ok = true;
    for (int i = 0; i < kE2EBatch; i++)
        e2e_construction_ok =
            e2e_construction_ok && (int)e2e_ref[i].size() == n_planted[i];
    printf("construction check (CPU ref kept count == planted count, every "
           "image): %s\n",
           e2e_construction_ok ? "✓" : "✗");
    e2e_ok = e2e_ok && e2e_construction_ok;

    cudaStreamDestroy(e2e_stream);
    cudaFree(d_e2e_raw);
    cudaFree(d_e2e_kept);
    cudaFree(d_e2e_kept_counts);
    cudaFree(d_e2e_params);

    printf(e2e_ok ? "✓ PASS: yolo-e2e checkpoint.\n"
                 : "✗ FAIL: yolo-e2e checkpoint.\n");

    ok = ok && ax_ok && e2e_ok;
    printf(ok ? "\n✓ PASS: all postprocess checkpoints.\n"
              : "\n✗ FAIL: at least one postprocess checkpoint failed.\n");
    return ok ? 0 : 1;
}
