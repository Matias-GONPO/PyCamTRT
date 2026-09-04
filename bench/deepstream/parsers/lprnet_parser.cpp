// Custom nvinfer classifier parser for LPRNet: greedy CTC decode.
//
// Wraps src/lprnet_ctc.h — the exact decoder the CORDERO checkpoint
// (lprnet_test) verified against the PyTorch reference — instead of
// adapting NVIDIA's lpr parser, whose dictionary/charset targets their
// US/CN LPRNet variants, not the sirius-ai 68-class head we deploy.
//
// Output binding per image: [68, 18] class-major logits — identical to
// what CtcGreedyDecode expects, no reshuffle needed.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "lprnet_ctc.h"
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferClassiferParseCtcLpr(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute>& attrList, std::string& descString) {
    if (outputLayersInfo.empty()) return false;

    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    if (layer.inferDims.numDims != 2) return false;
    const int classes = layer.inferDims.d[0];
    const int timesteps = layer.inferDims.d[1];
    const float* logits = static_cast<const float*>(layer.buffer);

    const std::vector<int> label = CtcGreedyDecode(logits, classes, timesteps);
    if (label.empty()) return true;  // no readable plate; no attribute
    const std::string plate = LprLabelString(label);

    // Smoke-test aid: DS_BENCH_LOG_PLATES=1 prints each decoded plate —
    // one line proves the full cascade (detect -> crop -> OCR -> CTC).
    static const bool kLog = std::getenv("DS_BENCH_LOG_PLATES") != nullptr;
    if (kLog) fprintf(stderr, "PLATE %s\n", plate.c_str());

    NvDsInferAttribute attr{};
    attr.attributeIndex = 0;
    attr.attributeValue = 0;
    attr.attributeConfidence = 1.0f;  // greedy CTC emits no sequence score
    attr.attributeLabel = strdup(plate.c_str());  // freed by nvinfer
    attrList.push_back(attr);
    descString = plate;
    return true;
}

CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassiferParseCtcLpr);
