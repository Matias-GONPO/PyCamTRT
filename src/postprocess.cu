#include "postprocess.h"

namespace {

// One thread per anchor. raw[(4+c)*anchors + i] means consecutive threads
// read consecutive addresses - coalesced across the warp for every class row.
// The math mirrors rtsp_infer's CPU Postprocess candidate loop exactly (same
// strict-< threshold, same argmax tie-breaking toward the lower class index,
// same un-letterbox expressions, same clamp semantics as the cv::Rect2f
// intersection) so the checkpoint can diff the two paths field by field.
// Body shared verbatim by the single-image and batched kernels.
__device__ inline void DecodeAnchor(const float* __restrict__ raw, int i,
                                    int anchors, int classes, float scale,
                                    int pad_x, int pad_y, int src_w, int src_h,
                                    int off_x, int off_y, float thresh,
                                    GpuDetection* __restrict__ out,
                                    int* count) {
    int best_cls = 0;
    float best = 0.f;
    for (int c = 0; c < classes; c++) {
        const float s = raw[(4 + c) * anchors + i];
        if (s > best) { best = s; best_cls = c; }
    }
    if (best < thresh) return;

    const float cx = raw[0 * anchors + i];
    const float cy = raw[1 * anchors + i];
    const float w = raw[2 * anchors + i];
    const float h = raw[3 * anchors + i];

    // Un-letterbox to source coords, then clamp to the frame.
    float x0 = (cx - w / 2.f - pad_x) / scale;
    float y0 = (cy - h / 2.f - pad_y) / scale;
    float x1 = x0 + w / scale;
    float y1 = y0 + h / scale;
    x0 = fmaxf(x0, 0.f);
    y0 = fmaxf(y0, 0.f);
    x1 = fminf(x1, (float)src_w);
    y1 = fminf(y1, (float)src_h);

    GpuDetection d;
    // SAHI tiles: the "source" above is the TILE (src_w/h = tile dims,
    // clamping to it is correct - a tile's detection cannot exceed the
    // tile); off_x/off_y then translate into full-frame coordinates.
    // The normal path passes 0,0 - byte-identical behavior.
    d.x = x0 + (float)off_x;
    d.y = y0 + (float)off_y;
    d.w = fmaxf(x1 - x0, 0.f);
    d.h = fmaxf(y1 - y0, 0.f);
    d.score = best;
    d.cls = best_cls;
    out[atomicAdd(count, 1)] = d;
}

__global__ void BoxDecodeKernel(const float* __restrict__ raw, int anchors,
                                int classes, float scale, int pad_x, int pad_y,
                                int src_w, int src_h, float thresh,
                                GpuDetection* __restrict__ out, int* count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= anchors) return;
    DecodeAnchor(raw, i, anchors, classes, scale, pad_x, pad_y, src_w, src_h,
                 /*off_x=*/0, /*off_y=*/0, thresh, out, count);
}

// Batched: grid.y = image index. Each image reads its slice of the
// contiguous [batch, 4+classes, anchors] tensor with its own letterbox
// geometry, and compacts into its own slot-strided candidate buffer.
__global__ void BoxDecodeBatchedKernel(
    const float* __restrict__ raw, int anchors, int classes,
    const PostprocImageParams* __restrict__ params, float thresh,
    GpuDetection* __restrict__ out, int* counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= anchors) return;
    const int img = blockIdx.y;
    const PostprocImageParams p = params[img];
    DecodeAnchor(raw + (size_t)img * (4 + classes) * anchors, i, anchors,
                 classes, p.lb.scale, p.lb.pad_x, p.lb.pad_y, p.src_w, p.src_h,
                 p.off_x, p.off_y, thresh, out + (size_t)img * anchors,
                 counts + img);
}

// Same intersection-over-union arithmetic as the CPU path's cv::Rect2f
// intersection: overlap from max/min of edges, union from the two areas.
__device__ inline float Iou(const GpuDetection& a, const GpuDetection& b) {
    const float ix0 = fmaxf(a.x, b.x);
    const float iy0 = fmaxf(a.y, b.y);
    const float ix1 = fminf(a.x + a.w, b.x + b.w);
    const float iy1 = fminf(a.y + a.h, b.y + b.h);
    const float inter = fmaxf(ix1 - ix0, 0.f) * fmaxf(iy1 - iy0, 0.f);
    const float uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

// Sort comparator: should candidate a come before candidate b? Score first;
// the cls/x tie-breaks make the order deterministic where the CPU path's
// score-only std::sort leaves ties unspecified (equal-score candidates are
// duplicate-anchor near-copies of the same box, so the kept *set* comes out
// the same either way - the checkpoint's CPU reference uses this same
// composite order so the diff is exact).
__device__ inline bool Before(const GpuDetection& a, const GpuDetection& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.cls != b.cls) return a.cls < b.cls;
    return a.x < b.x;
}

// One block per image. Load candidates to shared memory, bitonic-sort an
// index array, then greedy suppression: walk survivors in score order
// (sequential outer loop - greedy NMS's chain dependency is irreducible),
// suppress everything they overlap in parallel (threads split the j range).
// O(n) syncs, tens of candidates in practice - microseconds.
// Body shared verbatim by the single-image and batched kernels.
__device__ void NmsBlock(const GpuDetection* __restrict__ cands,
                         const int* __restrict__ count_ptr, float iou_thresh,
                         GpuDetection* __restrict__ kept, int* kept_count) {
    __shared__ GpuDetection s_dets[kMaxNmsCandidates];
    __shared__ short s_order[kMaxNmsCandidates];
    __shared__ unsigned char s_alive[kMaxNmsCandidates];

    const int tid = threadIdx.x;
    const int n = min(*count_ptr, kMaxNmsCandidates);
    if (n == 0) {
        if (tid == 0) *kept_count = 0;
        return;
    }

    int npow2 = 1;
    while (npow2 < n) npow2 <<= 1;

    for (int i = tid; i < npow2; i += blockDim.x) {
        if (i < n) s_dets[i] = cands[i];
        s_order[i] = i < n ? i : -1;  // -1 pads sort to a power of two
        s_alive[i] = 1;
    }
    __syncthreads();

    // Bitonic sort of s_order by Before(); -1 padding sorts to the end.
    for (int k = 2; k <= npow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < npow2; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj <= i) continue;
                const short a = s_order[i];
                const short b = s_order[ixj];
                const bool ascending = (i & k) == 0;
                // "a after b" under Before(), padding always last.
                const bool out_of_order =
                    a == -1 ? b != -1
                            : (b != -1 && Before(s_dets[b], s_dets[a]));
                if (out_of_order == ascending) {
                    s_order[i] = b;
                    s_order[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    // Greedy suppression in sorted order - identical semantics to the CPU
    // loop: a candidate is kept iff no higher-ranked *kept* same-class box
    // overlaps it above iou_thresh.
    for (int i = 0; i < n; i++) {
        __syncthreads();
        if (!s_alive[i]) continue;  // uniform: same shared value in all threads
        const GpuDetection a = s_dets[s_order[i]];
        for (int j = i + 1 + tid; j < n; j += blockDim.x) {
            if (!s_alive[j]) continue;
            const GpuDetection b = s_dets[s_order[j]];
            if (b.cls != a.cls) continue;
            if (Iou(a, b) > iou_thresh) s_alive[j] = 0;
        }
    }
    __syncthreads();

    if (tid == 0) {
        int k = 0;
        for (int i = 0; i < n; i++) {
            if (s_alive[i]) kept[k++] = s_dets[s_order[i]];
        }
        *kept_count = k;
    }
}

__global__ void NmsKernel(const GpuDetection* __restrict__ cands,
                          const int* __restrict__ count_ptr, float iou_thresh,
                          GpuDetection* __restrict__ kept, int* kept_count) {
    NmsBlock(cands, count_ptr, iou_thresh, kept, kept_count);
}

// Batched: block b == image b. N independent NMS problems run concurrently
// on different SMs; each block reads its own count on-device, so the whole
// batched decode -> NMS chain still needs no host synchronization.
__global__ void NmsBatchedKernel(const GpuDetection* __restrict__ cands,
                                 const int* __restrict__ counts, int anchors,
                                 float iou_thresh,
                                 GpuDetection* __restrict__ kept,
                                 int* kept_counts) {
    const int img = blockIdx.x;
    NmsBlock(cands + (size_t)img * anchors, counts + img, iou_thresh,
             kept + (size_t)img * kMaxNmsCandidates, kept_counts + img);
}

// M1a classifier family: one thread per batch item, plain sequential
// argmax over `classes` logits - see postprocess.h's LaunchArgmaxBatched
// WHY-comment for why no softmax and why one-thread-per-item (mirrors
// DecodeAnchor's per-anchor argmax-over-classes loop above).
__global__ void ArgmaxKernel(const float* __restrict__ logits, int batch,
                             int classes, int* __restrict__ labels,
                             float* __restrict__ scores) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;
    const float* row = logits + (size_t)i * classes;
    int best_cls = 0;
    float best = row[0];
    for (int c = 1; c < classes; c++) {
        const float v = row[c];
        if (v > best) {
            best = v;
            best_cls = c;
        }
    }
    labels[i] = best_cls;
    scores[i] = best;
}

// M4a: yolo-e2e family - one thread per (image, row) candidate, same
// grid.y=batch / per-image PostprocImageParams shape as
// BoxDecodeBatchedKernel above. See postprocess.h's LaunchYoloE2EBatched
// WHY-comment for the verified [x1,y1,x2,y2,score,cls] row layout and the
// un-letterbox derivation (DecodeAnchor's affine above, applied directly
// to both corners instead of derived from cx,cy,w,h). No NMS stage for
// this family: writes straight into the kept-list stride
// (kMaxNmsCandidates), the same buffers LaunchNmsBatched's output lands in.
__global__ void YoloE2EBatchedKernel(
    const float* __restrict__ raw, int max_dets,
    const PostprocImageParams* __restrict__ params, float thresh,
    bool norm_cxcywh, float net_w, float net_h,
    GpuDetection* __restrict__ out, int* counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_dets) return;
    const int img = blockIdx.y;
    const PostprocImageParams p = params[img];
    const float* row = raw + ((size_t)img * max_dets + i) * 6;
    const float score = row[4];
    if (score < thresh) return;

    // rtdetr variant (report 11 T3.2): same [dets,6] row contract, but the
    // box columns are NORMALIZED (0-1) cx,cy,w,h (ultralytics RT-DETR
    // export - pixel scaling lives in ITS python postprocess, so it must
    // live here for us) - scale to letterbox pixels and convert to the
    // corner form the shared affine below expects. yolo-e2e rows
    // (norm_cxcywh=false) pass through untouched - same values, same ops,
    // bit-exact with the pre-variant kernel.
    float bx0 = row[0], by0 = row[1], bx1 = row[2], by1 = row[3];
    if (norm_cxcywh) {
        // Intrinsics, not plain ops: `cx - 0.5f*bw` is a mul-add shape
        // nvcc contracts into fma (the existing kernels' `(a - b) / c`
        // shapes never were - why this file needed no care before), and a
        // contracted fma differs from the CPU reference's separate
        // round-per-op in the last ulp (caught by this family's own
        // checkpoint on the non-square-geometry image). __fmul_rn/
        // __fadd_rn/__fsub_rn are guaranteed never fused - pinning the
        // exact op sequence CpuRtDetr (postprocess_batch_test.cpp) mirrors.
        const float cx = __fmul_rn(row[0], net_w);
        const float cy = __fmul_rn(row[1], net_h);
        const float bw = __fmul_rn(row[2], net_w);
        const float bh = __fmul_rn(row[3], net_h);
        const float hw = __fmul_rn(0.5f, bw);
        const float hh = __fmul_rn(0.5f, bh);
        bx0 = __fsub_rn(cx, hw);
        by0 = __fsub_rn(cy, hh);
        bx1 = __fadd_rn(cx, hw);
        by1 = __fadd_rn(cy, hh);
    }

    // Un-letterbox: DecodeAnchor's `(coord - pad) / scale` affine, applied
    // directly to both corners - row is already x1,y1,x2,y2 in letterbox
    // space, unlike DecodeAnchor's cx,cy,w,h input (see postprocess.h's
    // WHY-comment for the full derivation).
    float x0 = (bx0 - p.lb.pad_x) / p.lb.scale;
    float y0 = (by0 - p.lb.pad_y) / p.lb.scale;
    float x1 = (bx1 - p.lb.pad_x) / p.lb.scale;
    float y1 = (by1 - p.lb.pad_y) / p.lb.scale;
    x0 = fmaxf(x0, 0.f);
    y0 = fmaxf(y0, 0.f);
    x1 = fminf(x1, (float)p.src_w);
    y1 = fminf(y1, (float)p.src_h);

    GpuDetection d;
    // Same off_x/off_y frame-translation as DecodeAnchor uses for SAHI
    // tiles; always 0,0 today (yolo-e2e+SAHI is rejected at Validate()),
    // kept for shape parity - see postprocess.h's WHY-comment.
    d.x = x0 + (float)p.off_x;
    d.y = y0 + (float)p.off_y;
    d.w = fmaxf(x1 - x0, 0.f);
    d.h = fmaxf(y1 - y0, 0.f);
    d.score = score;
    d.cls = (int)roundf(row[5]);
    out[(size_t)img * kMaxNmsCandidates + atomicAdd(&counts[img], 1)] = d;
}

}  // namespace

void LaunchNms(const GpuDetection* d_cands, const int* d_count,
               float iou_thresh, GpuDetection* d_kept, int* d_kept_count,
               cudaStream_t stream) {
    NmsKernel<<<1, 256, 0, stream>>>(d_cands, d_count, iou_thresh, d_kept,
                                     d_kept_count);
}

void LaunchBoxDecode(const float* d_raw, int num_anchors, int num_classes,
                     const LetterboxInfo& lb, int src_w, int src_h,
                     float score_thresh, GpuDetection* d_out, int* d_count,
                     cudaStream_t stream) {
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);
    const int block = 256;
    const int grid = (num_anchors + block - 1) / block;
    BoxDecodeKernel<<<grid, block, 0, stream>>>(
        d_raw, num_anchors, num_classes, lb.scale, lb.pad_x, lb.pad_y,
        src_w, src_h, score_thresh, d_out, d_count);
}

void LaunchBoxDecodeBatched(const float* d_raw, int batch, int num_anchors,
                            int num_classes,
                            const PostprocImageParams* d_params,
                            float score_thresh, GpuDetection* d_out,
                            int* d_counts, cudaStream_t stream) {
    cudaMemsetAsync(d_counts, 0, batch * sizeof(int), stream);
    const int block = 256;
    const dim3 grid((num_anchors + block - 1) / block, batch);
    BoxDecodeBatchedKernel<<<grid, block, 0, stream>>>(
        d_raw, num_anchors, num_classes, d_params, score_thresh, d_out,
        d_counts);
}

void LaunchNmsBatched(const GpuDetection* d_cands, const int* d_counts,
                      int batch, int num_anchors, float iou_thresh,
                      GpuDetection* d_kept, int* d_kept_counts,
                      cudaStream_t stream) {
    NmsBatchedKernel<<<batch, 256, 0, stream>>>(
        d_cands, d_counts, num_anchors, iou_thresh, d_kept, d_kept_counts);
}

void LaunchArgmaxBatched(const float* d_logits, int batch, int classes,
                         int* d_labels, float* d_scores, cudaStream_t stream) {
    if (batch <= 0) return;
    const int block = 256;
    const int grid = (batch + block - 1) / block;
    ArgmaxKernel<<<grid, block, 0, stream>>>(d_logits, batch, classes,
                                             d_labels, d_scores);
}

void LaunchYoloE2EBatched(const float* d_raw, int batch, int max_dets,
                          const PostprocImageParams* d_params,
                          float score_thresh, bool norm_cxcywh, float net_w,
                          float net_h, GpuDetection* d_out, int* d_counts,
                          cudaStream_t stream) {
    if (batch <= 0) return;
    cudaMemsetAsync(d_counts, 0, batch * sizeof(int), stream);
    const int block = 256;
    const dim3 grid((max_dets + block - 1) / block, batch);
    YoloE2EBatchedKernel<<<grid, block, 0, stream>>>(
        d_raw, max_dets, d_params, score_thresh, norm_cxcywh, net_w, net_h,
        d_out, d_counts);
}
