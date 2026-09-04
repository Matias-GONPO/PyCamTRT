#pragma once

// Thin TensorRT wrapper extracted from Step 1's simple_inference.cpp, adapted
// for the pipeline: bindings are allocated once at load and their device
// pointers exposed, so upstream stages (the NV12 preprocessing kernel) write
// *directly into the input binding* — no staging buffer, no H2D copy of
// pixels. Inference is enqueued async on a caller-provided stream
// (enqueueV3), unlike Step 1's blocking executeV2.

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "EngineBuilder.h"

class TrtEngine {
public:
    explicit TrtEngine(const std::string& engine_path,
                       const enginebuilder::BuildPrefs& build_prefs = {}) {
        // R3 auto-build: a .onnx path resolves to its house-named cache
        // next to the file (<stem>_b1-<max>_fp16_sm<XY>.engine), built
        // ONCE if absent. A cache that later fails to deserialize (TRT
        // version drift after an upgrade) is rebuilt once and retried -
        // never silently used, never rebuilt in a loop. Plain .engine
        // paths behave exactly as before this feature existed.
        std::string path = engine_path;
        const bool from_onnx =
            engine_path.size() > 5 &&
            engine_path.compare(engine_path.size() - 5, 5, ".onnx") == 0;
        if (from_onnx) {
            path = enginebuilder::CachePathFor(engine_path, build_prefs);
            std::ifstream probe(path, std::ios::binary);
            if (!probe.good()) {
                enginebuilder::BuildEngineFromOnnx(engine_path, build_prefs,
                                                   path, logger_);
            }
        }

        runtime_ = nvinfer1::createInferRuntime(logger_);
        for (int attempt = 0;; attempt++) {
            std::ifstream file(path, std::ios::binary);
            if (!file.good()) {
                throw std::runtime_error("Cannot open engine file: " + path);
            }
            file.seekg(0, std::ios::end);
            const size_t size = file.tellg();
            file.seekg(0, std::ios::beg);
            std::vector<char> data(size);
            file.read(data.data(), size);
            engine_ = runtime_->deserializeCudaEngine(data.data(), size);
            if (engine_) break;
            if (from_onnx && attempt == 0) {
                fprintf(stderr,
                        "[TrtEngine] cached engine %s failed to "
                        "deserialize (TensorRT version drift?) - "
                        "rebuilding once from %s\n",
                        path.c_str(), engine_path.c_str());
                enginebuilder::BuildEngineFromOnnx(engine_path, build_prefs,
                                                   path, logger_);
                continue;
            }
            throw std::runtime_error("Failed to deserialize engine: " + path);
        }
        context_ = engine_->createExecutionContext();

        // One input, one output (yolov8n: images[N,3,640,640] ->
        // output0[N,84,8400]), discovered by IO mode rather than hardcoding
        // names. A static engine has N fixed (1); a dynamic-batch engine has
        // N == -1 and its usable range comes from the optimization profile.
        // Bindings are allocated once at the profile's max batch; per-image
        // counts stay the unit everywhere so batch-1 callers are unaffected.
        //
        // Counting pass: fail fast on any engine that isn't exactly 1 input +
        // 1 output *before* any cudaMalloc, so a bad engine can't leak a
        // binding or silently rebind over an earlier one (each allocation
        // pass below overwrites d_input_/d_output_ on every iteration - fine
        // for exactly one match, a leak-then-clobber for more than one).
        {
            std::vector<std::string> in_names, out_names;
            for (int i = 0; i < engine_->getNbIOTensors(); i++) {
                const char* name = engine_->getIOTensorName(i);
                if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                    in_names.push_back(name);
                } else if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
                    out_names.push_back(name);
                }
            }
            if (in_names.size() != 1 || out_names.size() != 1) {
                auto join = [](const std::vector<std::string>& v) {
                    std::string s;
                    for (size_t i = 0; i < v.size(); i++) {
                        if (i) s += ", ";
                        s += v[i];
                    }
                    return s;
                };
                throw std::runtime_error(
                    "TrtEngine supports exactly 1 input + 1 output tensor; "
                    "engine has " + std::to_string(in_names.size()) +
                    " inputs (" + join(in_names) + ") and " +
                    std::to_string(out_names.size()) + " outputs (" +
                    join(out_names) + ")");
            }
        }

        // Pass 1: input. If the batch dim is dynamic, pin the shape to the
        // profile's min batch (with H/W at the profile's single pinned
        // value) so every downstream shape query is concrete - pinning to 1
        // would fail outright if the profile's kMIN is itself > 1.
        for (int i = 0; i < engine_->getNbIOTensors(); i++) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
                continue;
            input_name_ = name;
            nvinfer1::Dims shape = engine_->getTensorShape(name);
            if (shape.d[0] == -1) {
                const nvinfer1::Dims min_shape = engine_->getProfileShape(
                    name, 0, nvinfer1::OptProfileSelector::kMIN);
                min_batch_ = min_shape.d[0];
                shape = engine_->getProfileShape(
                    name, 0, nvinfer1::OptProfileSelector::kMAX);
                max_batch_ = shape.d[0];
                shape.d[0] = min_batch_;
                if (!context_->setInputShape(name, shape)) {
                    throw std::runtime_error("setInputShape failed for " +
                                             std::string(name));
                }
            }
            input_dims_ = shape;  // pinned batch, all dims concrete
            batch_ = shape.d[0];  // whatever batch was actually pinned above
            input_count_ = 1;
            for (int d = 1; d < shape.nbDims; d++) input_count_ *= shape.d[d];

            void* ptr = nullptr;
            if (cudaMalloc(&ptr, (size_t)max_batch_ * input_count_ *
                                     sizeof(float)) != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed for input binding");
            }
            d_input_ = static_cast<float*>(ptr);
            context_->setTensorAddress(name, ptr);
        }

        // Pass 2: output. With the input shape set, the context reports a
        // fully concrete output shape even for dynamic engines.
        for (int i = 0; i < engine_->getNbIOTensors(); i++) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kOUTPUT)
                continue;
            const nvinfer1::Dims shape = context_->getTensorShape(name);
            output_dims_ = shape;
            output_count_ = 1;
            for (int d = 1; d < shape.nbDims; d++) output_count_ *= shape.d[d];

            void* ptr = nullptr;
            if (cudaMalloc(&ptr, (size_t)max_batch_ * output_count_ *
                                     sizeof(float)) != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed for output binding");
            }
            d_output_ = static_cast<float*>(ptr);
            context_->setTensorAddress(name, ptr);
        }
        if (!d_input_ || !d_output_) {
            throw std::runtime_error("Engine does not have 1 input + 1 output");
        }
    }

    ~TrtEngine() {
        cudaFree(d_input_);
        cudaFree(d_output_);
        delete context_;
        delete engine_;
        delete runtime_;
    }

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Device pointer of the input binding - the preprocessing kernel's dst.
    // Counts are PER IMAGE; slot i of a batch lives at InputPtr(i).
    float* InputPtr(int slot = 0) const { return d_input_ + slot * input_count_; }
    size_t InputCount() const { return input_count_; }
    // Concrete input shape as resolved at init (d[0] = pinned batch, i.e.
    // MinBatch() for a dynamic engine, 1 for a static one).
    const nvinfer1::Dims& InputDims() const { return input_dims_; }

    float* OutputPtr(int slot = 0) const { return d_output_ + slot * output_count_; }
    size_t OutputCount() const { return output_count_; }
    // Concrete per-image output shape (d[0] = batch as resolved at init).
    // YOLO detect heads are [batch, 4+classes, anchors].
    const nvinfer1::Dims& OutputDims() const { return output_dims_; }

    // Usable batch range [MinBatch(), MaxBatch()]. [1,1] for static engines.
    int MaxBatch() const { return max_batch_; }
    int MinBatch() const { return min_batch_; }
    int Batch() const { return batch_; }

    // Sets the batch for subsequent Infer() calls. No-op if unchanged.
    // Throws on a static engine for n != 1 and on sizes outside
    // [MinBatch(), MaxBatch()], with a clear wrapper-level message instead of
    // failing inside setInputShape.
    void SetBatch(int n) {
        if (n == batch_) return;
        if (n < min_batch_ || n > max_batch_) {
            throw std::runtime_error("batch " + std::to_string(n) +
                                     " outside engine range [" +
                                     std::to_string(min_batch_) + "," +
                                     std::to_string(max_batch_) + "]");
        }
        nvinfer1::Dims shape = input_dims_;
        shape.d[0] = n;
        if (!context_->setInputShape(input_name_.c_str(), shape)) {
            throw std::runtime_error("setInputShape failed for batch " +
                                     std::to_string(n));
        }
        batch_ = n;
    }

    // Retargets the input binding (ping-pong batching: fill buffer B while
    // inference reads buffer A). `ptr` must hold MaxBatch()*InputCount()
    // floats. The engine-owned buffer from construction stays valid;
    // passing InputPtr() restores it.
    void SetInputAddress(float* ptr) {
        if (!context_->setTensorAddress(input_name_.c_str(), ptr)) {
            throw std::runtime_error("setTensorAddress failed");
        }
    }

    // Async - work is queued on `stream`, caller synchronizes. Processes
    // the current Batch() images.
    bool Infer(cudaStream_t stream) { return context_->enqueueV3(stream); }

private:
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                fprintf(stderr, "[TensorRT] %s\n", msg);
            }
        }
    };

    Logger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    float* d_input_ = nullptr;
    float* d_output_ = nullptr;
    size_t input_count_ = 0;   // elements per image
    size_t output_count_ = 0;  // elements per image
    std::string input_name_;
    nvinfer1::Dims input_dims_{};   // pinned-batch reference shape (see InputDims())
    nvinfer1::Dims output_dims_{};  // as resolved at init (pinned-batch shape)
    int max_batch_ = 1;
    int min_batch_ = 1;
    int batch_ = 1;
};
