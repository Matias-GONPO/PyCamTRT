#pragma once

// Step 2: fused GPU preprocessing. One kernel pass takes the decoder's mapped
// NV12 surface (pitched, VRAM) and writes a letterboxed, normalized, planar
// RGB float tensor (CHW) — the exact layout yolov8n.engine expects — directly
// into a caller-provided device buffer (in Step 4: the TensorRT input
// binding). No intermediate buffers, no CPU involvement: this is the single
// "mandatory" transform pass between decode and inference.

#include <cuda.h>
#include <cuda_runtime.h>

// How the source image was placed inside the destination tensor. Needed later
// to map detection boxes back to source-frame coordinates:
//   src_x = (dst_x - pad_x) / scale, src_y = (dst_y - pad_y) / scale
struct LetterboxInfo {
    float scale;
    int pad_x;
    int pad_y;
};

// Launches the fused NV12 -> letterboxed float CHW kernel on `stream`.
// `nv12`/`pitch` come straight from DecodedFrame (must still be mapped).
// `dst` must hold dst_w * dst_h * 3 floats. Output plane i (i=0,1,2) gets
// value = (pixel_i + norm_offset[i]) * norm_scale[i] - PER-CHANNEL (M3a,
// upgrade of M1a's single scalar pair), where plane i is already the i-th
// channel of the model's expected order because the rgb/bgr swap below
// picks which source channel (R or B) lands in plane 0 vs. plane 2 (see
// the kernel's c0/c2 selection in preprocess.cu) - same convention and
// arithmetic as LaunchNv12CropResizeBatched below (StepDesc's
// norm_offset[3]/norm_scale[3]/color, see graph.h). Defaults (all-0,
// all-1/255, RGB) preserve the letterboxed-YOLO-input behavior this kernel
// has always had, and equal-per-channel values reproduce the M1a scalar
// arithmetic bit-for-bit (same (p+o)*s form per channel). NOTE: the
// default arithmetic changed from `pixel / 255.f` to
// `(pixel + 0.f) * (1.f / 255.f)` to share one code path with the
// caller-chosen case - the two differ only in the last bit of float
// rounding. Harmless: the --verify gate diffs GPU-vs-CPU POSTPROCESS on the
// SAME already-produced tensor (this kernel's output is common to both
// sides, never itself the reference), so it is unaffected; preprocess_test
// is a visual/statistical check, not bit-exact against this kernel either.
// Async — caller synchronizes.
LetterboxInfo LaunchNV12ToTensor(CUdeviceptr nv12, unsigned int pitch,
                                 int src_w, int src_h,
                                 float* dst, int dst_w, int dst_h,
                                 cudaStream_t stream,
                                 float3 norm_offset = {0.f, 0.f, 0.f},
                                 float3 norm_scale = {1.f / 255.f,
                                                       1.f / 255.f,
                                                       1.f / 255.f},
                                 bool rgb = true);

// ---- Stage-2 crop preprocessing (cascade) --------------------------------

// One crop's source: which NV12 frame, and which pixel rect within it.
// The rect must lie inside the frame (caller clamps/rounds detections);
// w/h >= 1. Lives in a device array - crops of one stage-2 batch may come
// from different frames (different streams' ring copies).
struct CropParams {
    const uint8_t* nv12;  // pitched NV12; chroma rows follow src_h luma rows
    int pitch;
    int src_w, src_h;     // full frame dims (locate the chroma plane)
    int x, y, w, h;       // crop rect, source pixels
};

// Crops n rects (possibly from n different frames) and resizes each to
// out_w x out_h - equivalent to "crop the sub-image, then cv2.resize
// INTER_LINEAR" that recognition models are trained on (bilinear luma,
// nearest chroma, edge-replicated at the rect border). Output: n contiguous
// planar CHW tensors, output plane i (i=0,1,2) gets value =
// (pixel_i + norm_offset[i]) * norm_scale[i] - PER-CHANNEL (M3a; see
// LaunchNV12ToTensor above for the full convention). Channel order:
// rgb=false -> BGR (LPRNet: offset [-127.5]*3, scale [1/128]*3, BGR
// because its training pipeline read images with cv2 - the default, all
// pre-SAHI callers); rgb=true -> RGB (YOLO-family tiles for SAHI, offset
// [0]*3, scale [1/255]*3). Same arithmetic either way; equal-per-channel
// values reproduce the M1a scalar arithmetic bit-for-bit. Async - caller
// synchronizes.
void LaunchNv12CropResizeBatched(const CropParams* d_params, int n,
                                 float* d_out, int out_w, int out_h,
                                 float3 norm_offset, float3 norm_scale,
                                 cudaStream_t stream, bool rgb = false);
