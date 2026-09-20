// Custom nvinfer bbox parser for YOLOv8-family detect heads.
//
// Our own parser (benchmark decision, 2026-07-07): the [4+nc, anchors]
// decode below is the same math the PyCamTRT pipeline ships in
// postprocess.cu and verifies bit-exact against its CPU reference —
// third-party parser code in the benchmark's hot path would be an
// unverified variable.
//
// Layout: output binding [4+nc, anchors] per image (batch dim handled by
// nvinfer — this function is called once per image with that image's
// buffer). Rows 0..3 = cx,cy,w,h in network-input pixels, rows 4.. =
// per-class scores (post-sigmoid in the ultralytics export).
//
// The parser only decodes + confidence-filters, exactly like
// LaunchBoxDecode; NMS is nvinfer's job (cluster-mode=2,
// nms-iou-threshold=0.45 in the config, matching kIouThresh).
// Coordinates are reported in network-input space (640x640); nvinfer
// undoes the letterbox (maintain-aspect-ratio=1, symmetric-padding=1).

#include <algorithm>
#include <cstdint>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList) {
    if (outputLayersInfo.empty()) return false;

    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    // Dims as seen per image: [4+nc, anchors] (batch dim already stripped).
    if (layer.inferDims.numDims != 2) return false;
    const int rows = layer.inferDims.d[0];
    const int anchors = layer.inferDims.d[1];
    const int num_classes = rows - 4;
    if (num_classes < 1) return false;

    const float* out = static_cast<const float*>(layer.buffer);
    const float net_w = static_cast<float>(networkInfo.width);
    const float net_h = static_cast<float>(networkInfo.height);

    for (int a = 0; a < anchors; a++) {
        int best = 0;
        float best_s = out[4 * anchors + a];
        for (int c = 1; c < num_classes; c++) {
            const float s = out[(4 + c) * anchors + a];
            if (s > best_s) {
                best_s = s;
                best = c;
            }
        }
        const float thresh =
            detectionParams.perClassPreclusterThreshold[best];
        if (best_s < thresh) continue;

        const float cx = out[0 * anchors + a];
        const float cy = out[1 * anchors + a];
        const float w = out[2 * anchors + a];
        const float h = out[3 * anchors + a];

        // Clamp all four edges to the network frame, THEN derive extent —
        // clamping left/top alone would leave width/height overhanging on
        // edge-clipped boxes.
        const float left = std::max(0.f, std::min(cx - 0.5f * w, net_w - 1.f));
        const float top = std::max(0.f, std::min(cy - 0.5f * h, net_h - 1.f));
        const float right = std::max(0.f, std::min(cx + 0.5f * w, net_w - 1.f));
        const float bottom = std::max(0.f, std::min(cy + 0.5f * h, net_h - 1.f));

        NvDsInferParseObjectInfo obj{};
        obj.classId = best;
        obj.detectionConfidence = best_s;
        obj.left = left;
        obj.top = top;
        obj.width = right - left;
        obj.height = bottom - top;
        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV8);
