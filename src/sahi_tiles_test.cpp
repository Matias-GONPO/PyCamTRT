// SAHI checkpoint, two parts.
//
// Part A - tile geometry (host): every frame pixel is covered by >=1
// tile, neighbors overlap by >= the requested fraction, no tile exceeds
// the frame, last tiles sit flush with the far edges. Several frame
// shapes incl. 4K, odd dims, and frame-smaller-than-tile.
//
// Part B - the SAHI merge machinery (GPU vs CPU reference): synthetic
// per-tile raw tensors with PLANTED detections for known frame-space
// objects (inference itself is the engine's job and already verified;
// what SAHI adds - offset remap, per-tile NMS, cross-tile merge - is
// what this exercises):
//   - object A strictly inside one tile           -> survives as-is
//   - object B on a tile seam, planted in BOTH
//     overlapping tiles at slightly different
//     scores                                      -> merge keeps ONE
//   - object C larger than a tile, planted only
//     in the full-frame pass slot                 -> only the full-frame
//                                                    pass can supply it
// GPU: LaunchBoxDecodeBatched (per-tile params w/ offsets) ->
// LaunchNmsBatched -> host gather -> LaunchNms merge. CPU mirror: same
// plain-float decode + greedy NMS + merge. Bit-exact comparison.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "postprocess.h"
#include "sahi_tiles.h"

namespace {

constexpr float kScoreThresh = 0.4f;
constexpr float kIouThresh = 0.45f;
constexpr float kMergeIou = 0.5f;
constexpr int kAnchors = 8400;
constexpr int kClasses = 1;  // plate-detector-like head
constexpr int kTile = 640;

// ---- Part A ---------------------------------------------------------

bool CheckGrid(int fw, int fh, int tile, float overlap, const char* name) {
    const auto tiles = MakeTileGrid(fw, fh, tile, overlap);
    bool ok = true;
    // bounds + flush edges
    int max_x1 = 0, max_y1 = 0;
    for (const auto& t : tiles) {
        if (t.x < 0 || t.y < 0 || t.x + t.w > fw || t.y + t.h > fh) ok = false;
        max_x1 = std::max(max_x1, t.x + t.w);
        max_y1 = std::max(max_y1, t.y + t.h);
    }
    if (max_x1 != fw || max_y1 != fh) ok = false;
    // coverage: sample a grid of points, every one inside some tile
    for (int y = 0; y < fh && ok; y += 13) {
        for (int x = 0; x < fw; x += 17) {
            bool in = false;
            for (const auto& t : tiles)
                if (x >= t.x && x < t.x + t.w && y >= t.y && y < t.y + t.h) {
                    in = true;
                    break;
                }
            if (!in) { ok = false; break; }
        }
    }
    // neighbor overlap along x for same row (when >1 column)
    for (size_t i = 0; i + 1 < tiles.size() && ok; i++) {
        const auto &a = tiles[i], &b = tiles[i + 1];
        if (a.y == b.y && b.x > a.x) {  // adjacent in the same row
            const int ov = a.x + a.w - b.x;
            if (ov < (int)(tile * overlap) - 1) ok = false;
        }
    }
    printf("%s grid %-22s %dx%d tile %d ov %.1f -> %zu tiles\n",
           ok ? "✓" : "✗", name, fw, fh, tile, overlap, tiles.size());
    return ok;
}

// ---- Part B helpers --------------------------------------------------

struct Planted {
    // frame-space ground truth box (cx, cy, w, h) + score, and which
    // batch slots (tiles / full-frame) it is planted into
    float cx, cy, w, h;
    std::vector<std::pair<int, float>> slots;  // (slot index, score)
};

// CPU mirror of DecodeAnchor + greedy NMS (same math as the kernels and
// as rtsp_infer_multi's CpuReference), operating per slot then merged.
struct RefDet {
    float x, y, w, h, score;
    int cls;
};

float RefIou(const RefDet& a, const RefDet& b) {
    const float ix0 = std::max(a.x, b.x);
    const float iy0 = std::max(a.y, b.y);
    const float ix1 = std::min(a.x + a.w, b.x + b.w);
    const float iy1 = std::min(a.y + a.h, b.y + b.h);
    const float inter =
        std::max(ix1 - ix0, 0.f) * std::max(iy1 - iy0, 0.f);
    const float uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

// Greedy NMS with the kernels' deterministic (score desc, cls, x) order.
std::vector<RefDet> RefNms(std::vector<RefDet> c, float iou) {
    std::sort(c.begin(), c.end(), [](const RefDet& a, const RefDet& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.cls != b.cls) return a.cls < b.cls;
        return a.x < b.x;
    });
    std::vector<bool> alive(c.size(), true);
    std::vector<RefDet> kept;
    for (size_t i = 0; i < c.size(); i++) {
        if (!alive[i]) continue;
        kept.push_back(c[i]);
        for (size_t j = i + 1; j < c.size(); j++)
            if (alive[j] && c[j].cls == c[i].cls &&
                RefIou(c[i], c[j]) > iou)
                alive[j] = false;
    }
    return kept;
}

}  // namespace

int main() {
    printf("=== SAHI checkpoint ===\n[part A] tile geometry\n");
    bool ok = true;
    ok &= CheckGrid(3840, 2160, kTile, 0.2f, "4K");
    ok &= CheckGrid(1920, 1080, kTile, 0.2f, "1080p");
    ok &= CheckGrid(1280, 720, kTile, 0.2f, "720p");
    ok &= CheckGrid(802, 543, kTile, 0.2f, "odd dims");
    ok &= CheckGrid(640, 640, kTile, 0.2f, "frame == tile");
    ok &= CheckGrid(320, 240, kTile, 0.2f, "frame < tile");
    ok &= CheckGrid(3840, 2160, kTile, 0.0f, "no overlap");
    if (!ok) { printf("✗ FAIL part A\n"); return 1; }

    printf("[part B] remap + cross-tile merge (4K frame)\n");
    const int fw = 3840, fh = 2160;
    const auto tiles = MakeTileGrid(fw, fh, kTile, 0.2f);
    const int n_tiles = (int)tiles.size();
    const int slots = n_tiles + 1;  // + full-frame pass (last slot)

    // Per-slot params: tiles = identity letterbox at their offset;
    // full-frame = 4K letterboxed into 640 (scale 1/6, pad y).
    std::vector<PostprocImageParams> params(slots);
    for (int i = 0; i < n_tiles; i++) {
        params[i] = {};
        params[i].lb = LetterboxInfo{1.0f, 0, 0};
        params[i].src_w = tiles[i].w;
        params[i].src_h = tiles[i].h;
        params[i].off_x = tiles[i].x;
        params[i].off_y = tiles[i].y;
    }
    params[n_tiles] = {};
    const float ff_scale = 640.f / fw;                       // 0.1667
    const int ff_pad_y = (640 - (int)(fh * ff_scale)) / 2;   // letterbox
    params[n_tiles].lb = LetterboxInfo{ff_scale, 0, ff_pad_y};
    params[n_tiles].src_w = fw;
    params[n_tiles].src_h = fh;

    // Planted objects (frame space).
    // A: strictly inside tile 0 (0,0,640,640).
    // B: on the seam between tile 0 and tile 1 (x ~ 560) - planted in
    //    both at different scores; merge must keep exactly one (0.9).
    // C: 900px wide (bigger than a tile) - full-frame slot only.
    std::vector<Planted> objs = {
        {200.f, 200.f, 80.f, 60.f, {{0, 0.85f}}},
        {560.f, 300.f, 120.f, 90.f, {{0, 0.90f}, {1, 0.88f}}},
        {2000.f, 1200.f, 900.f, 500.f, {{n_tiles, 0.95f}}},
    };

    // Build per-slot raw tensors: sub-threshold noise + planted anchors.
    const size_t img_elems = (size_t)(4 + kClasses) * kAnchors;
    std::vector<float> raw((size_t)slots * img_elems);
    uint32_t seed = 1234u;
    auto lcg = [&]() { return seed = seed * 1664525u + 1013904223u; };
    auto unit = [&]() { return (lcg() >> 8) * (1.0f / 16777216.0f); };
    for (int s = 0; s < slots; s++) {
        float* t = raw.data() + (size_t)s * img_elems;
        for (int c = 0; c < kClasses; c++)
            for (int i = 0; i < kAnchors; i++)
                t[(4 + c) * kAnchors + i] = unit() * 0.3f;
        for (int i = 0; i < kAnchors; i++) {
            t[0 * kAnchors + i] = unit() * 640.f;
            t[1 * kAnchors + i] = unit() * 640.f;
            t[2 * kAnchors + i] = 4.f + unit() * 40.f;
            t[3 * kAnchors + i] = 4.f + unit() * 40.f;
        }
    }
    int next_anchor = 100;  // distinct anchors for planted entries
    for (const auto& o : objs) {
        for (auto [slot, score] : o.slots) {
            const PostprocImageParams& p = params[slot];
            // frame -> slot-local -> letterboxed model coords
            const float lx = (o.cx - p.off_x) * p.lb.scale + p.lb.pad_x;
            const float ly = (o.cy - p.off_y) * p.lb.scale + p.lb.pad_y;
            float* t = raw.data() + (size_t)slot * img_elems;
            const int a = next_anchor;
            next_anchor += 37;
            t[0 * kAnchors + a] = lx;
            t[1 * kAnchors + a] = ly;
            t[2 * kAnchors + a] = o.w * p.lb.scale;
            t[3 * kAnchors + a] = o.h * p.lb.scale;
            t[4 * kAnchors + a] = score;  // class 0
        }
    }

    // ---- GPU chain ----
    float* d_raw;
    GpuDetection *d_cands, *d_kept, *d_merge_c, *d_merge_k;
    int *d_counts, *d_kept_counts, *d_merge_count, *d_merge_kept;
    PostprocImageParams* d_params;
    cudaMalloc(&d_raw, raw.size() * sizeof(float));
    cudaMalloc(&d_cands, (size_t)slots * kAnchors * sizeof(GpuDetection));
    cudaMalloc(&d_kept,
               (size_t)slots * kMaxNmsCandidates * sizeof(GpuDetection));
    cudaMalloc(&d_counts, slots * sizeof(int));
    cudaMalloc(&d_kept_counts, slots * sizeof(int));
    cudaMalloc(&d_params, slots * sizeof(PostprocImageParams));
    cudaMalloc(&d_merge_c, kMaxNmsCandidates * sizeof(GpuDetection));
    cudaMalloc(&d_merge_k, kMaxNmsCandidates * sizeof(GpuDetection));
    cudaMalloc(&d_merge_count, sizeof(int));
    cudaMalloc(&d_merge_kept, sizeof(int));
    cudaMemcpy(d_raw, raw.data(), raw.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, params.data(),
               slots * sizeof(PostprocImageParams), cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    LaunchBoxDecodeBatched(d_raw, slots, kAnchors, kClasses, d_params,
                           kScoreThresh, d_cands, d_counts, stream);
    LaunchNmsBatched(d_cands, d_counts, slots, kAnchors, kIouThresh, d_kept,
                     d_kept_counts, stream);

    // Host gather of per-slot survivors -> one merge candidate list
    // (exactly what the pipeline will do).
    std::vector<GpuDetection> kept((size_t)slots * kMaxNmsCandidates);
    std::vector<int> kept_counts(slots);
    cudaMemcpyAsync(kept.data(), d_kept, kept.size() * sizeof(GpuDetection),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(kept_counts.data(), d_kept_counts, slots * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<GpuDetection> merge_in;
    for (int s = 0; s < slots; s++)
        for (int k = 0; k < kept_counts[s]; k++)
            merge_in.push_back(kept[(size_t)s * kMaxNmsCandidates + k]);
    const int mc = (int)merge_in.size();
    cudaMemcpy(d_merge_c, merge_in.data(), mc * sizeof(GpuDetection),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_merge_count, &mc, sizeof(int), cudaMemcpyHostToDevice);
    LaunchNms(d_merge_c, d_merge_count, kMergeIou, d_merge_k, d_merge_kept,
              stream);
    int final_n = 0;
    cudaMemcpyAsync(&final_n, d_merge_kept, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::vector<GpuDetection> final_dets(final_n);
    cudaMemcpy(final_dets.data(), d_merge_k,
               final_n * sizeof(GpuDetection), cudaMemcpyDeviceToHost);

    // ---- CPU reference: same math end to end ----
    std::vector<RefDet> ref_merge_in;
    for (int s = 0; s < slots; s++) {
        const PostprocImageParams& p = params[s];
        const float* t = raw.data() + (size_t)s * img_elems;
        std::vector<RefDet> cands;  // this slot's candidates only
        for (int i = 0; i < kAnchors; i++) {
            int bc = 0;
            float best = 0.f;
            for (int c = 0; c < kClasses; c++) {
                const float sc = t[(4 + c) * kAnchors + i];
                if (sc > best) { best = sc; bc = c; }
            }
            if (best < kScoreThresh) continue;
            const float cx = t[0 * kAnchors + i], cy = t[1 * kAnchors + i];
            const float w = t[2 * kAnchors + i], h = t[3 * kAnchors + i];
            float x0 = (cx - w / 2.f - p.lb.pad_x) / p.lb.scale;
            float y0 = (cy - h / 2.f - p.lb.pad_y) / p.lb.scale;
            float x1 = x0 + w / p.lb.scale;
            float y1 = y0 + h / p.lb.scale;
            x0 = std::max(x0, 0.f);
            y0 = std::max(y0, 0.f);
            x1 = std::min(x1, (float)p.src_w);
            y1 = std::min(y1, (float)p.src_h);
            RefDet d;
            d.x = x0 + (float)p.off_x;
            d.y = y0 + (float)p.off_y;
            d.w = std::max(x1 - x0, 0.f);
            d.h = std::max(y1 - y0, 0.f);
            d.score = best;
            d.cls = bc;
            cands.push_back(d);
        }
        // per-slot NMS, then the survivors feed the merge - mirroring the
        // GPU chain's per-slot LaunchNmsBatched -> gather.
        for (const auto& d : RefNms(cands, kIouThresh))
            ref_merge_in.push_back(d);
    }
    auto ref_final = RefNms(ref_merge_in, kMergeIou);

    // ---- Compare ----
    bool same = ((int)ref_final.size() == final_n);
    for (int i = 0; same && i < final_n; i++) {
        const auto& g = final_dets[i];
        const auto& r = ref_final[i];
        same = g.cls == r.cls && g.score == r.score && g.x == r.x &&
               g.y == r.y && g.w == r.w && g.h == r.h;
    }
    printf("%s merged detections: gpu %d | cpu %zu | %s\n",
           same ? "✓" : "✗", final_n, ref_final.size(),
           same ? "bit-exact" : "MISMATCH");

    // Semantic assertions on the planted objects:
    auto count_near = [&](float cx, float cy) {
        int c = 0;
        for (const auto& d : final_dets) {
            const float dx = d.x + d.w / 2.f, dy = d.y + d.h / 2.f;
            if (std::fabs(dx - cx) < 30.f && std::fabs(dy - cy) < 30.f) c++;
        }
        return c;
    };
    const int nA = count_near(200.f, 200.f);
    const int nB = count_near(560.f, 300.f);
    const int nC = count_near(2000.f, 1200.f);
    printf("%s object A (inside tile):   %d detection(s), want 1\n",
           nA == 1 ? "✓" : "✗", nA);
    printf("%s object B (seam, planted twice): %d detection(s), want 1 "
           "(merge de-dup)\n",
           nB == 1 ? "✓" : "✗", nB);
    printf("%s object C (bigger than a tile): %d detection(s), want 1 "
           "(full-frame pass)\n",
           nC == 1 ? "✓" : "✗", nC);

    const bool pass = same && nA == 1 && nB == 1 && nC == 1;
    printf("%s SAHI checkpoint\n", pass ? "✓ PASS:" : "✗ FAIL:");
    return pass ? 0 : 1;
}
