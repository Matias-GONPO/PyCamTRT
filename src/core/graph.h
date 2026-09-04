#pragma once

// Pipeline DESCRIPTION types: a small step graph a caller (CLI today,
// Python later - see MEMORY's "long-term = multi-stream + cascade + Python
// edge lib") builds to describe what to run. This is a settled API shape:
// generic on paper (steps + layers), but v1's executor (Pipeline, in
// pipeline.h/.cpp) only knows how to run ONE concrete resolution of it -
// see pipeline.cpp's Validate()/ExecPlan for exactly which shapes are
// accepted today.

#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace cordero {

// Select (R1, v0.2.0): a declarative DETECTION FILTER between a detector's
// Postprocess and a cascade child's Engine - the routing node ("send only
// class-0 crops to this child"). Model-agnostic by construction: predicates
// are comparisons over the GpuDetection struct (cls int, score, crop size),
// never class NAMES - "person" is a model's opinion, not the library's.
// Predicates must stay compilable (host-side comparisons in the crop-build
// path); anything needing pixels or Python belongs to the
// Python-on-results tier instead.
enum class StepKind { Process, Engine, Postprocess, Select };
// M1a: Argmax is the classifier family (plain max-logit, no softmax - see
// postprocess.h's LaunchArgmaxBatched) alongside YoloDetect (layer 0) and
// Ctc (today's LPRNet/OCR cascade family).
// M4a: YoloE2E is the "yolo-e2e" family - NMS-free end-to-end detector
// heads (yolo26-style, [N,dets,6]=(x1,y1,x2,y2,score,cls) already
// TopK'd/deduplicated in-graph - see postprocess.h's LaunchYoloE2EBatched
// WHY-comment and manual/FINDINGS.md's "M1 model-generality findings"
// section B). LAYER 0 (detection) only, alongside YoloDetect.
// M4b: Ctc/Argmax remain CASCADE-only (never layer 0), but v1 now runs a
// depth-2 TREE - one detector root feeding N SIBLING Ctc/Argmax children
// (layers 1..K, any mix of the two families), not just the single cascade
// layer M1a-M4a supported. Every child must crop the ROOT's detections
// directly; chaining a child off ANOTHER child's Postprocess (a child of a
// child) is a named, deliberate rejection - see pipeline.cpp's Validate().
// RtDetr (report 11 T3.2): yolo-e2e's twin - same [N,300,6] NMS-free head
// contract, but box columns are NORMALIZED cx,cy,w,h (ultralytics RT-DETR
// export) instead of pixel x1,y1,x2,y2; decoded by the same kernel with a
// layout flag. Embedding (T3.1): raw pass-through cascade-child family
// for 2D [N,D] feature heads - no decode kernel, rows land in
// ChildOutput::vectors.
enum class Family { YoloDetect, Ctc, Argmax, YoloE2E, Embedding, RtDetr };

struct StepDesc {
    StepKind kind;
    int input = -1;             // producing step index in PipelineConfig::steps; -1 = raw stream source
    std::string engine_path;    // Engine steps
    Family family = Family::YoloDetect;  // Postprocess steps
    float score_thresh = 0.4f;  // yolo / yolo-e2e postprocess
    // yolo postprocess only - IGNORED by yolo-e2e (M4a): that family's
    // head is already NMS-free/deduplicated in-graph, so there is no NMS
    // stage to threshold (see postprocess.h's LaunchYoloE2EBatched
    // WHY-comment). Left at its default rather than erroring so existing
    // StepDesc-building code need not special-case the family.
    float iou_thresh = 0.45f;

    // WHY (M1a): input normalization was the first concrete blocker to
    // onboarding a non-YOLO model (a plain classifier, M1b's target) - the
    // GPU kernels were already parameterized for it on the crop/cascade
    // path (LaunchNv12CropResizeBatched already takes norm_offset/
    // norm_scale/rgb - see preprocess.h) or trivially parameterizable on
    // the whole-frame path (Nv12ToTensorKernel hardcoded /255 RGB). These
    // fields turn that into a per-Engine-step DATA choice instead of a
    // recompile. NAN (each array element's default) / -1 (color) mean
    // "inherit the family default for this engine's position in the
    // graph" - see pipeline.cpp's Validate()/ExecPlan (ResolveNormColor)
    // for the concrete resolved values, which default to exactly today's
    // hardcoded constants (byte-compatible when left unset). Only
    // meaningful on Engine steps.
    //
    // M3a: PER-CHANNEL (upgrade of M1a's single scalar pair) - each engine
    // sees value[c] = (pixel[c] + norm_offset[c]) * norm_scale[c]. NAN is
    // resolved PER ELEMENT (a channel can inherit while its siblings
    // override), so an all-NAN array (the default) inherits the whole
    // family default and a scalar caller value (M1a shape, still accepted
    // at the Python surface) simply broadcasts to all three before it gets
    // here. CONVENTION (matches how users state a model's trained mean/
    // std, e.g. ImageNet mean [0.485,0.456,0.406]/std [0.229,0.224,0.225],
    // always given in RGB): index 0/1/2 are the FIRST/SECOND/THIRD channel
    // of THIS ENGINE'S channel order, i.e. the same order `color` selects -
    // color="rgb" -> index 0 = R, 1 = G, 2 = B; color="bgr" -> index 0 = B,
    // 1 = G, 2 = R. The GPU kernels apply norm_offset[i]/norm_scale[i] to
    // OUTPUT PLANE i, and plane i is already the model's i-th channel by
    // construction (the rgb/bgr swap picks WHICH source channel (R or B)
    // lands in plane 0 and plane 2 - see preprocess.cu's c0/c2 selection),
    // so no further reordering is needed once the array is in this
    // convention.
    // Select steps (R1): the routing predicate. A detection passes when it
    // matches ALL active criteria; a zero/empty value means "criterion
    // off" (an empty Select passes everything - the pass-all parity case).
    // sel_min_size is min(crop w, crop h) in SOURCE pixels, measured on
    // the post-clamp crop rectangle (what the child would actually see).
    std::vector<int> sel_classes;   // empty = any class
    float sel_min_score = 0.f;      // 0 = any score
    float sel_min_size = 0.f;       // 0 = any size
    // Engine steps, R3 auto-build preferences - used ONLY when engine_path
    // is a .onnx (see EngineBuilder.h's BuildPrefs): dynamic-batch profile
    // max, fp16 toggle, and explicit build H/W for exports whose spatial
    // dims are symbolic (0 = take from the onnx, or 640x640 fallback).
    int build_max_batch = 16;
    bool build_fp16 = true;
    int build_h = 0;
    int build_w = 0;
    float norm_offset[3] = {NAN, NAN, NAN};
    float norm_scale[3] = {NAN, NAN, NAN};
    int color = -1;  // -1 = inherit, 0 = BGR, 1 = RGB
};

struct LayerDesc {
    std::string name;
    std::vector<int> steps;     // indices into PipelineConfig::steps, insertion order

    // SAHI options, valid on a layer whose postprocess family is
    // YoloDetect. NOT valid for YoloE2E (M4a) - Validate() raises a named
    // RuntimeError: cross-tile merge NMS is undefined for an already
    // NMS-free/deduplicated e2e head (see postprocess.h's
    // LaunchYoloE2EBatched WHY-comment).
    bool sahi = false;
    int sahi_tile = 640;
    float sahi_overlap = 0.2f;
    float sahi_merge_iou = 0.5f;
    bool sahi_full_frame = true;
    bool sahi_serial = false;
};

// Result-queue-full policy (Part 2 design doc). Block is the v0/settled
// default (unchanged behavior): the GPU thread waits for a consumer to make
// room. DropOldest instead pops the oldest queued result to make room and
// never blocks the GPU thread - for a consumer that would rather see the
// newest results promptly (e.g. live monitoring) than every result.
enum class Backpressure { Block, DropOldest };

// ---- Endpoint sinks (Phase A1: host-side compressed-packet ring + tee
// sinks) ----------------------------------------------------------------
// WHY: RTSP packets already traverse host RAM (FFmpegDemuxer) before NVDEC
// decode (see pipeline.cpp's ProducerLoop). Forwarding the ORIGINAL video
// therefore needs NO re-encode and NO device-to-host copy - just retaining
// the compressed packets that were already in host RAM for a moment (see
// core/packet_ring.h's PacketRing) and re-serializing them out a side
// channel. Sinks are strictly downstream taps: see Pipeline's sink-thread
// WHY-comments for the "never backpressure the pipeline" contract every
// sink implementation must honor (bounded drop-oldest queues, no locks held
// across I/O).
enum class SinkKind {
    Events,       // per-frame detection/text metadata, as NDJSON lines
    StreamRelay,  // original compressed video, republished (RTSP, -c copy)
};

struct SinkDesc {
    SinkKind kind;
    // Events: index into PipelineConfig::steps of the Postprocess step this
    // sink reports on. v1 accepts the executed graph's layer-0
    // (YoloDetect|YoloE2E) Postprocess step OR any sibling cascade child's
    // (Ctc|Argmax) Postprocess step (M4b: a depth-2 tree can have several)
    // - all deliver the same FrameResult shape since results are per-frame,
    // not per-layer (see Validate() in pipeline.cpp for the exact check).
    // StreamRelay: must be -1 (raw source) - it forwards the ORIGINAL
    // packets, upstream of any inference step, not a postprocess output.
    int input = -1;
    // -1 = all streams (Events only). StreamRelay requires a CONCRETE
    // stream_id in v1: one relay target is one RTSP path, and "all streams"
    // onto one path has no defined multiplexing here - a deliberate v1
    // restriction (flagged, not an oversight - see the A1 report),
    // Validate() rejects -1 for a StreamRelay sink by name.
    int stream_id = -1;
    // Target grammar, checked at CONSTRUCTION (Pipeline's ctor -> Validate())
    // so a bad target fails fast rather than mid-run:
    //   Events:      "tcp://host:port" | "file:///abs/path" | "stdout"
    //   StreamRelay: "rtsp://host:port/name" (grammar for a
    //                "tcpraw://host:port" length-prefixed-Annex-B fallback
    //                is reserved but only implemented if the RTSP output
    //                muxer path isn't in use - see Pipeline's StreamRelay
    //                sink WHY-comment for which this build ships)
    std::string target;
};

// WHY: capacity planning is per-camera (stream count x decode rate x
// inference rate) - a parking camera can run skip=4/decode=key next to an
// entrance camera at skip=1, so the two load dials need a per-stream
// override on top of the pipeline-wide default. 0/-1 mean "inherit the
// PipelineConfig-level value" so the common case (every stream the same)
// needs no per-stream setup at all.
struct StreamDesc {
    std::string url;
    int skip = 0;    // 0 = inherit PipelineConfig::skip
    int decode = -1; // -1 = inherit key_only, 0 = all frames, 1 = keyframes-only
};

struct PipelineConfig {
    std::vector<StreamDesc> streams;
    std::vector<StepDesc> steps;
    std::vector<LayerDesc> layers;
    int skip = 1;
    bool key_only = false;      // --decode key
    bool verify = false;        // per-frame CPU-reference check
    int max_frames = 0;         // decoded frames per stream; 0 = run until Stop()
    size_t queue_capacity = 256;
    Backpressure backpressure = Backpressure::Block;
    // WHY: extends each inferred frame's full-res NV12 ring slot lifetime
    // (see result.h's FrameResult::frame_hold) from "released the instant
    // the GPU thread is done with it" to "released when the CONSUMER is
    // done with it" (every FrameResult copy destroyed, or ReleaseFrame()
    // called). Off by default: holding onto ring memory past the GPU
    // thread's own use of it is an opt-in cost - see ring_depth below for
    // the backpressure this implies.
    bool hold_frames = false;
    // WHY: frames-in-flight slack PER PRODUCER before Nv12Ring::AcquireSlot
    // blocks it - raise this when hold_frames is on. A consumer holding
    // roughly ring_depth-1 frames of one stream at once stalls that
    // stream's producer (every slot is in_use, AcquireSlot has nowhere to
    // wrap to) - that's intended backpressure, not a bug, but it means
    // ring_depth needs to be sized to how many frames of a stream a
    // consumer actually plans to hold concurrently.
    int ring_depth = 4;

    // Phase A1 endpoint sinks (see SinkKind/SinkDesc above) - empty by
    // default: no sink threads, no behavior change for an existing caller
    // that never sets this.
    std::vector<SinkDesc> sinks;
    // WHY: fixed-DURATION ring of ORIGINAL compressed packets per stream
    // (core/packet_ring.h's PacketRing), feeding StreamRelay sinks and
    // Pipeline::ExtractClip(). Always ALLOCATED (cheap, empty at 0 packets)
    // but only PUSHED INTO by the producer when > 0 - a small memcpy per
    // demuxed packet, paid even for packets that skip inference entirely
    // (that's the point: sinks see everything demuxed, not just what gets
    // decoded/inferred) - see ProducerLoop in pipeline.cpp. 0 opts a
    // pipeline that wants neither relay nor clip support out of that cost
    // entirely.
    int ring_seconds = 10;

    // CP1: A/B escape hatch + regression isolation for per-child CUDA-stream
    // cascade parallelism (same pattern as LayerDesc::sahi_serial above,
    // just PIPELINE-wide rather than per-layer - unlike SAHI, which is tied
    // to one specific yolo detection layer, the cascade is every sibling
    // child 1..K acting together, so there is no single LayerDesc it
    // belongs to). false (the default) = PARALLEL: every cascade child's
    // entire pipeline (crop/infer/decode) is enqueued on its OWN
    // cudaStream_t with no synchronization between children - concurrent
    // enqueueV3() across distinct TrtEngine execution contexts is legal
    // (see pipeline.cpp's ChildScratch/GpuLoop). true = today's SEQUENTIAL
    // path, unchanged, kept for A/B measurement and to isolate a regression
    // to "is it the parallelism" before anything else.
    bool cascade_serial = false;

    std::function<void(const std::string&)> log;  // nullptr => fprintf(stderr,...)
};

}  // namespace cordero
