#pragma once

// R3 (v0.2.0): in-process ONNX -> TensorRT engine builder + cache.
//
// WHY this exists: engines are per-GPU, per-TRT-version artifacts, so the
// repo ships .onnx exports and every user used to hand-type a trtexec
// incantation per model before anything ran. With this, an Engine step
// (or the CLI's --engine/--ocr) can point straight at the .onnx: TrtEngine
// resolves the house-named cache next to it, builds it ONCE if missing
// (logging that it's a one-time, minutes-long step), and loads it - the
// second run is exactly as fast as a hand-built engine, because it IS one.
//
// Deliberate contract choices:
//   - In-core C++ (nvonnxparser), not a trtexec subprocess: works for the
//     Python surface AND the CLI, no runtime dependency on trtexec's
//     location, and the input tensor's NAME is discovered from the parsed
//     network (exports disagree: ultralytics uses "images", torchvision
//     exports here use "input" - a wrapper would have to guess).
//   - Same 1-input + 1-output contract TrtEngine enforces at load,
//     enforced HERE too, before minutes of build time are spent.
//   - Dynamic batch profile min=1 / opt=max(1, max_batch/2) / max=
//     max_batch, matching the house engine naming (<stem>_b1-<max>_...).
//     min is pinned at 1 because the pipeline's warm-up/first batch is
//     batch-1 (see TrtEngine's MinBatch() WHY-comment).
//   - Exports with SYMBOLIC spatial dims (ultralytics dynamic=True
//     parametrizes H/W too, not just batch) need concrete build dims:
//     prefs.net_h/net_w when given, else 640x640 with a printed notice -
//     pass shape= from Python to override.
//   - Atomic cache write (.tmp + rename): a crashed/killed build never
//     leaves a truncated .engine for the next run to trip over.

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

namespace enginebuilder {

struct BuildPrefs {
    int max_batch = 16;
    bool fp16 = true;
    int net_h = 0;  // 0 = take from the onnx; used only if it's symbolic
    int net_w = 0;
};

inline std::string ArchTag() {
    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    return "sm" + std::to_string(prop.major * 10 + prop.minor);
}

// models/foo_dynamic.onnx + {16, fp16} -> models/foo_dynamic_b1-16_fp16_sm86.engine
inline std::string CachePathFor(const std::string& onnx_path,
                                const BuildPrefs& p) {
    const std::string stem = onnx_path.substr(0, onnx_path.size() - 5);
    return stem + "_b1-" + std::to_string(p.max_batch) +
           (p.fp16 ? "_fp16_" : "_fp32_") + ArchTag() + ".engine";
}

inline void BuildEngineFromOnnx(const std::string& onnx_path,
                                const BuildPrefs& prefs,
                                const std::string& out_path,
                                nvinfer1::ILogger& logger) {
    std::unique_ptr<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("createInferBuilder failed");
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(
            onnx_path.c_str(),
            (int)nvinfer1::ILogger::Severity::kWARNING)) {
        std::string errs;
        for (int i = 0; i < parser->getNbErrors(); i++) {
            errs += std::string("\n  ") + parser->getError(i)->desc();
        }
        throw std::runtime_error("ONNX parse failed for " + onnx_path + ":" +
                                 errs);
    }
    if (network->getNbInputs() != 1 || network->getNbOutputs() != 1) {
        throw std::runtime_error(
            "auto-build supports exactly 1 input + 1 output tensor; " +
            onnx_path + " has " + std::to_string(network->getNbInputs()) +
            " inputs / " + std::to_string(network->getNbOutputs()) +
            " outputs (same contract TrtEngine enforces at load)");
    }

    nvinfer1::ITensor* in = network->getInput(0);
    nvinfer1::Dims d = in->getDimensions();
    if (d.nbDims != 4) {
        throw std::runtime_error(
            "auto-build expects a 4D [N,C,H,W] input; " + onnx_path +
            " declares " + std::to_string(d.nbDims) + "D");
    }
    int h = d.d[2], w = d.d[3];
    if (h < 0 || w < 0) {  // symbolic spatial dims (ultralytics dynamic=True)
        h = prefs.net_h > 0 ? prefs.net_h : 640;
        w = prefs.net_w > 0 ? prefs.net_w : 640;
        if (prefs.net_h <= 0 || prefs.net_w <= 0) {
            fprintf(stderr,
                    "[EngineBuilder] %s has symbolic H/W - building at "
                    "%dx%d (pass shape= to override)\n",
                    onnx_path.c_str(), h, w);
        }
    }
    const int c = d.d[1] < 0 ? 3 : (int)d.d[1];

    nvinfer1::IOptimizationProfile* profile =
        builder->createOptimizationProfile();
    const char* in_name = in->getName();
    const int opt = prefs.max_batch > 1 ? prefs.max_batch / 2 : 1;
    profile->setDimensions(in_name, nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4{1, c, h, w});
    profile->setDimensions(in_name, nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4{opt, c, h, w});
    profile->setDimensions(in_name, nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4{prefs.max_batch, c, h, w});

    std::unique_ptr<nvinfer1::IBuilderConfig> config(
        builder->createBuilderConfig());
    config->addOptimizationProfile(profile);
    if (prefs.fp16) config->setFlag(nvinfer1::BuilderFlag::kFP16);

    fprintf(stderr,
            "[EngineBuilder] building %s -> %s (batch 1-%d, %s, %s) - "
            "one-time, can take minutes...\n",
            onnx_path.c_str(), out_path.c_str(), prefs.max_batch,
            prefs.fp16 ? "fp16" : "fp32", in_name);
    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        throw std::runtime_error("engine build failed for " + onnx_path +
                                 " (see TensorRT log above)");
    }

    const std::string tmp = out_path + ".tmp";
    {
        FILE* f = fopen(tmp.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + tmp);
        const size_t n =
            fwrite(serialized->data(), 1, serialized->size(), f);
        fclose(f);
        if (n != serialized->size()) {
            std::remove(tmp.c_str());
            throw std::runtime_error("short write to " + tmp);
        }
    }
    if (std::rename(tmp.c_str(), out_path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("cannot move " + tmp + " -> " + out_path);
    }
    fprintf(stderr, "[EngineBuilder] cached %s (%.1f MB)\n",
            out_path.c_str(), serialized->size() / 1048576.0);
}

}  // namespace enginebuilder
