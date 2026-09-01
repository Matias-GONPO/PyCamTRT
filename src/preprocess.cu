#include "preprocess.h"

#include <cstdint>

namespace {

// One thread per destination pixel. Letterbox padding is filled with YOLO's
// conventional gray (114/255). Luma is sampled bilinearly; chroma nearest —
// at 4:2:0 chroma resolution the visual difference is nil and it halves the
// sampling work. BT.601 limited-range YUV -> RGB.
__global__ void Nv12ToTensorKernel(const uint8_t* __restrict__ nv12, int pitch,
                                   int src_w, int src_h,
                                   float* __restrict__ dst, int dst_w, int dst_h,
                                   float scale, int pad_x, int pad_y,
                                   float3 norm_offset, float3 norm_scale, int rgb) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const int scaled_w = static_cast<int>(src_w * scale);
    const int scaled_h = static_cast<int>(src_h * scale);

    float r = 114.f, g = 114.f, b = 114.f;

    const int lx = x - pad_x;  // position inside the scaled image
    const int ly = y - pad_y;
    if (lx >= 0 && ly >= 0 && lx < scaled_w && ly < scaled_h) {
        // Back-project destination pixel center into source coordinates.
        float sx = (lx + 0.5f) / scale - 0.5f;
        float sy = (ly + 0.5f) / scale - 0.5f;
        sx = fminf(fmaxf(sx, 0.f), src_w - 1.f);
        sy = fminf(fmaxf(sy, 0.f), src_h - 1.f);

        const int x0 = static_cast<int>(sx);
        const int y0 = static_cast<int>(sy);
        const int x1 = min(x0 + 1, src_w - 1);
        const int y1 = min(y0 + 1, src_h - 1);
        const float fx = sx - x0;
        const float fy = sy - y0;

        const uint8_t* yp = nv12;
        const float Y =
            (yp[y0 * pitch + x0] * (1.f - fx) + yp[y0 * pitch + x1] * fx) * (1.f - fy) +
            (yp[y1 * pitch + x0] * (1.f - fx) + yp[y1 * pitch + x1] * fx) * fy;

        // UV plane starts right after src_h rows of luma (ulTargetHeight ==
        // display height in NvDecoder, verified by FrameDumper output).
        const uint8_t* uvp = nv12 + static_cast<size_t>(pitch) * src_h;
        const int uv_row = y0 >> 1;
        const int uv_col = (x0 >> 1) << 1;
        const float U = uvp[uv_row * pitch + uv_col] - 128.f;
        const float V = uvp[uv_row * pitch + uv_col + 1] - 128.f;

        const float Yl = 1.164f * (Y - 16.f);
        r = fminf(fmaxf(Yl + 1.596f * V, 0.f), 255.f);
        g = fminf(fmaxf(Yl - 0.392f * U - 0.813f * V, 0.f), 255.f);
        b = fminf(fmaxf(Yl + 2.017f * U, 0.f), 255.f);
    }

    const int plane = dst_w * dst_h;
    const int idx = y * dst_w + x;
    // Channel order is a caller choice (M1a - see header): rgb=1 -> RGB
    // planes (YOLO-family default, this kernel's historical/only
    // behavior), rgb=0 -> BGR. c0/c2 already select WHICH source channel
    // (R or B) lands in output plane 0 vs. plane 2, so plane i is already
    // the model's i-th channel - norm_offset/norm_scale (M3a: per-channel,
    // .x/.y/.z = plane 0/1/2) apply directly with no further reordering.
    // Same (pixel + norm_offset[i]) * norm_scale[i] arithmetic either way,
    // matching Nv12CropResizeKernel below.
    const float c0 = rgb ? r : b;
    const float c2 = rgb ? b : r;
    dst[idx] = (c0 + norm_offset.x) * norm_scale.x;
    dst[plane + idx] = (g + norm_offset.y) * norm_scale.y;
    dst[2 * plane + idx] = (c2 + norm_offset.z) * norm_scale.z;
}

// Explicitly unfused lerp: nvcc contracts a*b+c into FMA (different
// rounding), which would break the bit-exact CPU-reference checkpoint.
// The _rn intrinsics pin each op to plain round-to-nearest, matching the
// reference's separate float ops. Cost is noise (~us kernel).
__device__ inline float LerpExact(float a, float b, float t) {
    return __fadd_rn(__fmul_rn(a, __fsub_rn(1.f, t)), __fmul_rn(b, t));
}

// One thread per destination pixel, blockIdx.z = crop index. Same sampling
// scheme as Nv12ToTensorKernel (bilinear luma, nearest chroma, BT.601), but
// over a sub-rect with edge replication instead of letterbox padding, and
// BGR plane order with caller-chosen normalization (see header).
__global__ void Nv12CropResizeKernel(const CropParams* __restrict__ params,
                                     float* __restrict__ dst,
                                     int dst_w, int dst_h,
                                     float3 norm_offset, float3 norm_scale,
                                     int rgb) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;
    const CropParams p = params[blockIdx.z];

    // Back-project destination pixel center into crop-local coordinates,
    // clamped to the rect so border samples replicate its edge pixels -
    // exactly what resizing the cropped sub-image would do.
    float sx = (x + 0.5f) * (float)p.w / dst_w - 0.5f;
    float sy = (y + 0.5f) * (float)p.h / dst_h - 0.5f;
    sx = fminf(fmaxf(sx, 0.f), p.w - 1.f);
    sy = fminf(fmaxf(sy, 0.f), p.h - 1.f);

    const int x0l = static_cast<int>(sx);
    const int y0l = static_cast<int>(sy);
    const float fx = sx - x0l;
    const float fy = sy - y0l;
    const int x0 = p.x + x0l;
    const int y0 = p.y + y0l;
    const int x1 = p.x + min(x0l + 1, p.w - 1);
    const int y1 = p.y + min(y0l + 1, p.h - 1);

    const uint8_t* yp = p.nv12;
    const float Y = LerpExact(
        LerpExact(yp[y0 * p.pitch + x0], yp[y0 * p.pitch + x1], fx),
        LerpExact(yp[y1 * p.pitch + x0], yp[y1 * p.pitch + x1], fx), fy);

    const uint8_t* uvp = p.nv12 + static_cast<size_t>(p.pitch) * p.src_h;
    const int uv_row = y0 >> 1;
    const int uv_col = (x0 >> 1) << 1;
    const float U = uvp[uv_row * p.pitch + uv_col] - 128.f;
    const float V = uvp[uv_row * p.pitch + uv_col + 1] - 128.f;

    // Same anti-FMA treatment as LerpExact for every mul-add chain.
    const float Yl = __fmul_rn(1.164f, __fsub_rn(Y, 16.f));
    const float r =
        fminf(fmaxf(__fadd_rn(Yl, __fmul_rn(1.596f, V)), 0.f), 255.f);
    const float g =
        fminf(fmaxf(__fsub_rn(__fsub_rn(Yl, __fmul_rn(0.392f, U)),
                              __fmul_rn(0.813f, V)), 0.f), 255.f);
    const float b =
        fminf(fmaxf(__fadd_rn(Yl, __fmul_rn(2.017f, U)), 0.f), 255.f);

    const int plane = dst_w * dst_h;
    const int idx = y * dst_w + x;
    float* out = dst + static_cast<size_t>(blockIdx.z) * 3 * plane;
    // Channel order is a caller choice: BGR (rgb=0, LPRNet/cv2 training
    // convention) or RGB (rgb=1, YOLO-family tiles for SAHI). Only the
    // plane order changes - plane i is already the model's i-th channel
    // (see Nv12ToTensorKernel's comment above), so norm_offset/norm_scale
    // (M3a: per-channel, .x/.y/.z = plane 0/1/2) apply directly. The
    // arithmetic is identical per channel either way.
    const float c0 = rgb ? r : b;
    const float c2 = rgb ? b : r;
    out[idx] = (c0 + norm_offset.x) * norm_scale.x;
    out[plane + idx] = (g + norm_offset.y) * norm_scale.y;
    out[2 * plane + idx] = (c2 + norm_offset.z) * norm_scale.z;
}

}  // namespace

void LaunchNv12CropResizeBatched(const CropParams* d_params, int n,
                                 float* d_out, int out_w, int out_h,
                                 float3 norm_offset, float3 norm_scale,
                                 cudaStream_t stream, bool rgb) {
    if (n <= 0) return;
    const dim3 block(32, 8);
    const dim3 grid((out_w + block.x - 1) / block.x,
                    (out_h + block.y - 1) / block.y, n);
    Nv12CropResizeKernel<<<grid, block, 0, stream>>>(
        d_params, d_out, out_w, out_h, norm_offset, norm_scale,
        rgb ? 1 : 0);
}

LetterboxInfo LaunchNV12ToTensor(CUdeviceptr nv12, unsigned int pitch,
                                 int src_w, int src_h,
                                 float* dst, int dst_w, int dst_h,
                                 cudaStream_t stream,
                                 float3 norm_offset, float3 norm_scale,
                                 bool rgb) {
    LetterboxInfo info;
    info.scale = fminf(static_cast<float>(dst_w) / src_w,
                       static_cast<float>(dst_h) / src_h);
    info.pad_x = (dst_w - static_cast<int>(src_w * info.scale)) / 2;
    info.pad_y = (dst_h - static_cast<int>(src_h * info.scale)) / 2;

    const dim3 block(32, 8);
    const dim3 grid((dst_w + block.x - 1) / block.x,
                    (dst_h + block.y - 1) / block.y);
    Nv12ToTensorKernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(nv12), static_cast<int>(pitch),
        src_w, src_h, dst, dst_w, dst_h, info.scale, info.pad_x, info.pad_y,
        norm_offset, norm_scale, rgb ? 1 : 0);
    return info;
}
