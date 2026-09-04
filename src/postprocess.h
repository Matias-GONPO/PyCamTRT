#pragma once

// GPU postprocess, part 1: box decode. First stage of moving postprocessing
// (currently CPU box decode + NMS in rtsp_infer) onto the GPU, so the
// per-frame D2H copy shrinks from the raw [84,8400] tensor (2.8 MB) to a
// compact detection list (~KB) and the ~1.9 ms CPU postprocess leaves the
// critical path.
//
// Contract kept deliberately minimal so other/experimental models can plug in
// their own decode without architectural change: a plain launch function,
// caller owns every device buffer, everything async on the caller's stream.
// This file is yolov8-family specific in *layout* only (transposed
// [4+classes, anchors]: rows 0..3 = cx,cy,w,h in tensor space, rows 4.. =
// class scores). A different model family gets a sibling launcher next to
// this one, not a plugin framework.

#include <cuda_runtime.h>

#include "preprocess.h"  // LetterboxInfo

// One confidence-filtered candidate in source-frame pixel coordinates,
// clamped to the frame. Same information the CPU path produces; the NMS
// kernel (part 2) will filter these in place in VRAM.
struct GpuDetection {
    float x, y, w, h;
    float score;
    int cls;
};

// Decodes the raw yolov8 output into confidence-filtered candidates.
//   d_raw:   the TensorRT output binding, [4+num_classes, num_anchors].
//   d_out:   caller-allocated, must hold num_anchors entries (worst case).
//   d_count: caller-allocated single int; zeroed here, candidate count after.
// Entry order in d_out is nondeterministic (atomic compaction) but the *set*
// is deterministic; part 2's score sort imposes order anyway. Async on
// `stream` - caller synchronizes before reading d_out/d_count.
void LaunchBoxDecode(const float* d_raw, int num_anchors, int num_classes,
                     const LetterboxInfo& lb, int src_w, int src_h,
                     float score_thresh, GpuDetection* d_out, int* d_count,
                     cudaStream_t stream);

// Part 2: greedy per-class NMS on LaunchBoxDecode's candidates, entirely in
// VRAM. Single-block kernel by design: it reads the candidate count from
// d_count on the device, so the decode->NMS chain needs no host sync at all -
// the CPU first touches detection data when the final compact list arrives.
// Sorts by (score desc, cls asc, x asc) in shared memory, then runs the same
// greedy suppression loop as the CPU path. Survivors are written to d_kept in
// that sorted order, count to d_kept_count.
//
// Candidate count is clamped to kMaxNmsCandidates (shared-memory budget; the
// 0.4 score threshold yields tens in practice). One block = one SM is the
// right shape at this size; multi-stream batching later runs one block per
// stream through the same API.
constexpr int kMaxNmsCandidates = 1024;

void LaunchNms(const GpuDetection* d_cands, const int* d_count,
               float iou_thresh, GpuDetection* d_kept, int* d_kept_count,
               cudaStream_t stream);

// ---- Batched variants (Step 5) ---------------------------------------
// Same kernels' math applied to N images in one launch each. Layouts are
// slot-strided views of the single-image contract:
//   raw tensors: contiguous [batch, 4+classes, anchors] (the TRT output
//                binding of a dynamic-batch engine)
//   candidates:  image i at d_out + i*num_anchors, count in d_counts[i]
//   kept lists:  image i at d_kept + i*kMaxNmsCandidates, d_kept_counts[i]
// Per-image letterbox/frame geometry rides in a device array (streams may
// differ in resolution); caller owns and uploads it.

struct PostprocImageParams {
    LetterboxInfo lb;
    int src_w, src_h;
    // SAHI tile origin in the full frame: decoded boxes are translated by
    // this after un-letterboxing (and after clamping against src_w/src_h,
    // which for a tile ARE the tile dims), so tile detections land in
    // FRAME coordinates. Whole-frame images use 0,0 - existing callers'
    // shorter aggregate inits ({lb, w, h}) value-initialize these to 0,
    // so all pre-SAHI behavior is unchanged.
    int off_x, off_y;
};

// One launch, grid.y = batch. d_params: batch entries, device memory.
void LaunchBoxDecodeBatched(const float* d_raw, int batch, int num_anchors,
                            int num_classes,
                            const PostprocImageParams* d_params,
                            float score_thresh, GpuDetection* d_out,
                            int* d_counts, cudaStream_t stream);

// One launch, one block per image - N independent NMS problems running
// concurrently on different SMs; counts still read on-device (no host sync).
void LaunchNmsBatched(const GpuDetection* d_cands, const int* d_counts,
                      int batch, int num_anchors, float iou_thresh,
                      GpuDetection* d_kept, int* d_kept_counts,
                      cudaStream_t stream);

// ---- Classifier family: argmax (M1a) -----------------------------------
// A sibling launcher next to LaunchBoxDecode*/LaunchNms* above, same house
// rule (this file's top WHY-comment): a different model family gets its
// own launcher, not a plugin framework. Plain max-logit argmax, NO softmax
// - argmax is order-preserving under any monotonic transform (softmax
// included), so the predicted LABEL is bit-identical either way; only the
// reported CONFIDENCE differs (raw logit here vs. a softmax probability).
// Documented as "raw logit" confidence rather than paying softmax's extra
// exp/sum pass for a number this family's caller doesn't need to be a
// probability - trivial to softmax the winning row client-side if wanted.
//
// One thread per batch item, sequential scan over `classes` - the same
// per-thread argmax-over-classes idiom LaunchBoxDecode's DecodeAnchor
// already uses per anchor (classes here is the tens-to-low-thousands a
// classifier head typically has; no block-per-item reduction needed at
// that size).
//   d_logits: [batch, classes] contiguous (an engine's raw output binding).
//   d_labels: caller-allocated, batch ints - winning class index per item.
//   d_scores: caller-allocated, batch floats - that class's raw logit.
// Async on `stream` - caller synchronizes before reading d_labels/d_scores.
void LaunchArgmaxBatched(const float* d_logits, int batch, int classes,
                         int* d_labels, float* d_scores, cudaStream_t stream);

// ---- Detector family: yolo-e2e (M4a) -----------------------------------
// A sibling launcher next to LaunchBoxDecode*/LaunchNms*/LaunchArgmaxBatched
// above (this file's top WHY-comment) - but LAYER 0 (detection), not a
// layer-1 cascade like Ctc/Argmax: the family Validate() accepts as the
// layer-0 detector alongside YoloDetect (see graph.h's Family enum,
// pipeline.cpp's Validate()/GpuLoop).
//
// VERIFIED HEAD LAYOUT (yolo26n, M1b finding + M4a re-verification,
// 2026-08-31): output0 = [batch, 300, 6], row = (x1, y1, x2, y2, score,
// cls) in the model's OWN 640x640 LETTERBOX pixel space (NOT normalized
// 0..1, NOT cx/cy/w/h). Confirmed independently this session (no
// ultralytics decode involved - onnxruntime CPU inference on the raw ONNX
// graph only) on a real letterboxed frame from
// tools/stream_farm/media/atlas_plate_g30.mp4:
//   - onnx.load(...).graph.output: name='output0' dims=['batch',300,6];
//     op histogram has 2 TopK nodes and ZERO NMS nodes - NMS-free,
//     already-deduplicated by the model's own one-to-one matching.
//   - Column 4 (hypothesized score) is monotonically non-increasing across
//     all 300 rows of the test image - pre-sorted by descending score.
//   - Columns 0-3 range ~0-640 (not 0..1): pixel space, not normalized.
//     Column 0 < column 2 and column 1 < column 3 for every row above
//     threshold: (x1,y1,x2,y2) corner order, not (cx,cy,w,h).
//   - Column 5 takes small integer values (COCO class indices, 0-79 seen),
//     not a per-class score vector - already class-RESOLVED.
//   - VISUAL confirmation, not just plausible ranges: the two rows above a
//     0.4 threshold on the test frame were class 0 (COCO "person", score
//     0.82) and class 62 (COCO "tv", score 0.40); overlaying both boxes on
//     the letterboxed canvas lands them exactly on a person's head/
//     shoulders and a television screen respectively.
//
// No anchor decode (no cx,cy,w,h -> x0,y0,x1,y1 transform, no per-anchor
// argmax-over-classes) and NO NMS kernel at all - genuinely SIMPLER than
// LaunchBoxDecode*/LaunchNms* (manual/FINDINGS.md's "M1 model-generality
// findings" section B anticipated exactly this shape): per row, threshold
// the score, UN-LETTERBOX the four box columns, clamp to frame,
// atomic-compact survivors. The un-letterbox step reuses DecodeAnchor's own
// affine (postprocess.cu) verbatim: that kernel starts from cx,cy,w,h and
// derives one corner as `x0 = (cx - w/2.f - pad_x) / scale` before getting
// the other via `x1 = x0 + w/scale`; yolo-e2e already has BOTH corners in
// letterbox space, so the identical `(coord - pad) / scale` affine applies
// directly to all four of x1,y1,x2,y2 - same scale/pad_x/pad_y, same clamp
// order (max 0, min src_w/src_h), same off_x/off_y frame-translation
// DecodeAnchor applies for SAHI tiles. SAHI itself is rejected for this
// family in v1 (Validate()'s named error - cross-tile merge NMS is
// undefined for an already-NMS-free head), so off_x/off_y is always 0,0 in
// practice today - kept anyway because it costs nothing and keeps this
// launcher's shape parallel to DecodeAnchor's for a future tiled use.
//   d_raw:     [batch, max_dets, 6] contiguous, ROW-MAJOR (an engine's raw
//              output binding - row r of image i is
//              d_raw + (i*max_dets + r)*6, unlike LaunchBoxDecodeBatched's
//              channel-major [batch,4+classes,anchors] layout).
//   max_dets:  rows per image (300 for yolo26n; pipeline.cpp's Setup()
//              reads this from the engine's own OutputDims).
//   d_out/d_counts: SAME shape/stride contract as LaunchNmsBatched's
//              d_kept/d_kept_counts (image i's survivors at
//              d_out + i*kMaxNmsCandidates, count at d_counts[i]) - this
//              family has no NMS stage, so its launcher writes directly
//              into the kept-list buffers GpuLoop already owns (max_dets
//              300 <= kMaxNmsCandidates 1024 - see pipeline.cpp's Setup()
//              shape check, which enforces this).
//   norm_cxcywh/net_w/net_h: rtdetr variant (report 11 T3.2) - when
//              norm_cxcywh is true the row's box columns are NORMALIZED
//              (0-1) cx,cy,w,h (ultralytics RT-DETR export layout) and are
//              scaled by net_w/net_h + converted to corners before the
//              shared un-letterbox affine; false = yolo-e2e's native
//              pixel-space x1,y1,x2,y2, passed through bit-identically
//              (pass 640,640 - ignored).
// Async on `stream` - caller synchronizes before reading d_out/d_counts.
void LaunchYoloE2EBatched(const float* d_raw, int batch, int max_dets,
                          const PostprocImageParams* d_params,
                          float score_thresh, bool norm_cxcywh, float net_w,
                          float net_h, GpuDetection* d_out, int* d_counts,
                          cudaStream_t stream);
