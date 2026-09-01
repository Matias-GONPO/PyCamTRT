#pragma once

// SAHI tile geometry: split a frame into overlapping tile rects, the SAHI
// convention (tile pixel size + overlap ratio; the grid is DERIVED from
// the frame dims, so higher resolutions automatically get more tiles).
//
// Header-only host code, shared by the pipeline and its checkpoint
// (sahi_test) so the verified code is the shipped code - same pattern as
// lprnet_ctc.h.
//
// Placement rule (classic SAHI): tiles advance by stride = tile*(1-overlap)
// from the frame's origin; the LAST tile in each axis is pulled back flush
// with the frame edge (never letting a tile hang past the frame, never
// leaving uncovered pixels). Frames not larger than the tile in an axis
// get a single full-span tile there.

#include <algorithm>
#include <vector>

struct TileRect {
    int x, y, w, h;
};

inline std::vector<int> TileOffsets1D(int frame, int tile, int stride) {
    std::vector<int> offs;
    if (frame <= tile) {
        offs.push_back(0);
        return offs;
    }
    for (int o = 0;; o += stride) {
        if (o + tile >= frame) {
            offs.push_back(frame - tile);  // flush with the far edge
            break;
        }
        offs.push_back(o);
    }
    return offs;
}

// tile_px: tile side in source pixels (normally the engine input, 640).
// overlap: fraction of the tile shared between neighbors, [0, 0.9].
inline std::vector<TileRect> MakeTileGrid(int frame_w, int frame_h,
                                          int tile_px, float overlap) {
    overlap = std::min(std::max(overlap, 0.f), 0.9f);
    const int stride = std::max(1, (int)(tile_px * (1.f - overlap)));
    std::vector<TileRect> tiles;
    for (int oy : TileOffsets1D(frame_h, tile_px, stride)) {
        for (int ox : TileOffsets1D(frame_w, tile_px, stride)) {
            tiles.push_back({ox, oy, std::min(tile_px, frame_w),
                             std::min(tile_px, frame_h)});
        }
    }
    return tiles;
}
