// Part 1: cordero::Pipeline - the multi-stream pipeline orchestration moved
// out of rtsp_infer_multi.cpp's main()/Producer()/GPU-loop into a reusable
// library core. See pipeline.h for the public contract; this file keeps
// every WHY-comment from the original code intact, adjusted only for
// member-variable plumbing, the stop flag, result-queue emission, and the
// log-callback conversion (no printf/std::cout below this line - see Log()
// /Logf()).

#include "core/pipeline.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
}

// Phase A1 (endpoint sinks): raw BSD sockets for the Events sink's tcp://
// target only - the StreamRelay sink's networking goes entirely through
// libavformat's RTSP output muxer (see RelaySink below), which manages its
// own connection.
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include "FFmpegDemuxer.h"
#include "NvDecoder.h"
#include "TrtEngine.h"
#include "core/packet_ring.h"
#include "lprnet_ctc.h"
#include "sahi_tiles.h"
#include "postprocess.h"
#include "preprocess.h"

namespace cordero {
namespace {

using Clock = std::chrono::steady_clock;

// Net input side is fixed at 640x640 for the v1 executor (YoloDetect
// family) - not yet part of the graph API; a future Family/step option
// would generalize this.
constexpr int kNetW = 640, kNetH = 640;

void CheckCu(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(r, &s);
        throw std::runtime_error(std::string(what) + " failed: " +
                                 (s ? s : "?"));
    }
}

// M3a: ExecPlan/StepDesc carry per-channel norm as plain float[3] (matching
// graph.h's StepDesc); the preprocess.cu kernels take float3 by value.
inline float3 ToFloat3(const float a[3]) {
    return make_float3(a[0], a[1], a[2]);
}

// ---- v1 executor resolution --------------------------------------------
// The graph surface (graph.h) is generic; v1's EXECUTOR is exactly today's
// fixed conductor (decode -> preprocess -> engine -> yolo postprocess
// [+SAHI] [-> engine -> ctc postprocess]). Validate() resolves a
// PipelineConfig onto that conductor or throws, naming both what v1
// supports and what was given.

// M4b: one resolved SIBLING recognition child (depth-2 tree) - each crops
// layer 0's detections independently with its OWN engine/family/norm/
// color. Replaces the single l1_* scalar fields ExecPlan carried through
// M1a-M4a (exactly one cascade child was all v1 supported until now) - see
// the vector<ChildPlan> `children` member below and Validate()'s new
// per-layer (index >= 1) loop for how this is populated.
struct ChildPlan {
    std::string name;
    std::string engine_path;
    Family family;  // Ctc or Argmax only - see Validate()'s check.
    // M1a/M3a per-position defaults still apply (see ResolveNormColor's
    // call site below): today's hardcoded LPRNet/cv2 convention
    // (-127.5, 1/128, BGR) unless this child's Engine step overrides it.
    float norm_offset[3];
    float norm_scale[3];
    int rgb;  // 0 = BGR, 1 = RGB (already resolved - not -1/inherit here)
};

struct ExecPlan {
    std::string engine_path;
    // M4a: which Postprocess family layer 0's DETECTOR decodes to
    // (YoloDetect or YoloE2E - see Family in graph.h). Selects both
    // Setup()'s engine I/O shape check and GpuLoop's layer-0 decode path
    // (BoxDecodeBatched+NmsBatched vs. the single YoloE2EBatched launcher).
    Family l0_family = Family::YoloDetect;
    float score_thresh = 0.4f;
    float iou_thresh = 0.45f;  // yolo-e2e: unused (see graph.h's StepDesc)
    // M1a/M3a: layer-0 engine's resolved PER-CHANNEL normalization/color
    // (see graph.h's StepDesc norm_offset[3]/norm_scale[3]/color
    // WHY-comment and ResolveNormColor below) - feeds the whole-frame
    // LaunchNV12ToTensor call AND the SAHI tile LaunchNv12CropResizeBatched
    // calls (same engine, same expected input either way). Default =
    // today's hardcoded YOLO-family values (0, 1/255, RGB) broadcast to
    // all 3 channels, now data.
    float l0_norm_offset[3] = {0.f, 0.f, 0.f};
    float l0_norm_scale[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    bool l0_rgb = true;
    bool sahi = false;
    int sahi_tile = 640;
    float sahi_overlap = 0.2f;
    float sahi_merge_iou = 0.5f;
    bool sahi_full_frame = true;
    bool sahi_serial = false;
    // M4b: zero or more SIBLING recognition children (a depth-2 tree: one
    // detector root, N children that EACH crop the root's detections
    // independently - see ChildPlan above). Empty = detector-only
    // pipeline. Replaces the single ocr_engine_path/l1_family/l1_norm_*/
    // l1_rgb fields M1a-M4a carried (exactly one child was all v1
    // supported until now).
    std::vector<ChildPlan> children;
};

constexpr const char* kV1Supported =
    "v1 executor supports Process->Engine->Postprocess(YoloDetect|YoloE2E) "
    "[+SAHI, YoloDetect only] optionally feeding one or more SIBLING "
    "Engine->Postprocess(Ctc|Argmax) children (a depth-2 tree: every child "
    "crops the DETECTOR layer's own Postprocess step, never another "
    "child's)";

[[noreturn]] void FailGraph(const std::string& got) {
    throw std::runtime_error(std::string(kV1Supported) + "; got " + got);
}

// M1a/M3a: resolves one Engine StepDesc's inherit-by-default
// norm_offset[3]/norm_scale[3]/color (graph.h) onto concrete PER-CHANNEL
// values, given the family default for this engine's POSITION in the v1
// executor (layer 0 vs. layer 1 - see ExecPlan's l0_*/l1_* WHY-comments).
// NAN is resolved PER ELEMENT (a channel can inherit while its siblings
// override) and is exact to test here since StepDesc's field default IS
// NAN (a caller-set NaN is nonsensical input anyway, so no legitimate
// value is misread as "inherit").
void ResolveNormColor(const StepDesc& e, const float def_offset[3],
                      const float def_scale[3], bool def_rgb,
                      float* out_offset, float* out_scale, bool* out_rgb) {
    for (int i = 0; i < 3; i++) {
        out_offset[i] = std::isnan(e.norm_offset[i]) ? def_offset[i]
                                                      : e.norm_offset[i];
        out_scale[i] = std::isnan(e.norm_scale[i]) ? def_scale[i]
                                                    : e.norm_scale[i];
    }
    if (e.color == -1) {
        *out_rgb = def_rgb;
    } else if (e.color == 0) {
        *out_rgb = false;
    } else if (e.color == 1) {
        *out_rgb = true;
    } else {
        std::ostringstream os;
        os << "engine step color " << e.color
           << " (want -1 = inherit, 0 = BGR, 1 = RGB)";
        throw std::runtime_error(os.str());
    }
}

// M3a: family-default norm arrays, broadcast scalars from the pre-M3a
// hardcoded constants (see ExecPlan's l0_*/l1_* WHY-comments) - passed to
// ResolveNormColor at each call site below.
constexpr float kL0DefOffset[3] = {0.f, 0.f, 0.f};
constexpr float kL0DefScale[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
constexpr float kL1DefOffset[3] = {-127.5f, -127.5f, -127.5f};
constexpr float kL1DefScale[3] = {1.f / 128.f, 1.f / 128.f, 1.f / 128.f};

ExecPlan Validate(const PipelineConfig& cfg) {
    // Per-stream overrides: skip 0 means "inherit", so any legal override
    // value is >= 0 (a real skip must be >= 1 - see the resolution below);
    // decode's sentinel is -1 (inherit), with 0/1 the two real modes.
    for (size_t i = 0; i < cfg.streams.size(); i++) {
        const StreamDesc& sd = cfg.streams[i];
        if (sd.skip < 0) {
            std::ostringstream os;
            os << "stream " << i << " skip " << sd.skip << " (want >= 0, 0 = inherit)";
            FailGraph(os.str());
        }
        if (sd.decode < -1 || sd.decode > 1) {
            std::ostringstream os;
            os << "stream " << i << " decode " << sd.decode
               << " (want -1 = inherit, 0 = all, 1 = key)";
            FailGraph(os.str());
        }
    }
    // Every step's `input`, if set, must name a real step - checked before
    // anything shape-specific so a bad wire always gets this message
    // regardless of which layer it's in.
    for (size_t i = 0; i < cfg.steps.size(); i++) {
        const int in = cfg.steps[i].input;
        if (in >= 0 && (size_t)in >= cfg.steps.size()) {
            std::ostringstream os;
            os << "step " << i << " input " << in << " does not exist";
            FailGraph(os.str());
        }
    }
    if (cfg.layers.empty()) FailGraph("no layers");
    // M4b: the layer-count cap (>2 used to be rejected outright - exactly
    // one cascade layer was all v1 supported) is GONE - layers[1..] are now
    // any number of SIBLING recognition children (a depth-2 tree). What's
    // still rejected, and now by a NAMED error rather than this blanket
    // count check, is a CHAIN (a child fed from another child's Postprocess
    // instead of the detector's) - see the ctc_argmax_post_steps scan and
    // the per-child loop below.

    // ring_depth is a real correctness floor, not just a graph-shape check
    // (see Nv12Ring's WHY-comment): depth 1 would make AcquireSlot(seq)
    // always map to slot 0 and immediately deadlock against whatever
    // frame is still occupying it, depth 0 is nonsensical (mod by zero).
    if (cfg.ring_depth < 2) {
        std::ostringstream os;
        os << "ring_depth " << cfg.ring_depth << " (want >= 2)";
        throw std::runtime_error(os.str());
    }
    // Not fatal, just a likely footgun: hold_frames means a consumer can
    // legitimately be sitting on one or more ring slots at once, and the
    // default depth (4) is sized for the *unheld* case (GPU-thread-paced
    // release only). A smaller depth under hold_frames stalls that
    // stream's producer very quickly - warn rather than refuse, since a
    // caller doing deliberately tight backpressure may want exactly this.
    if (cfg.hold_frames && cfg.ring_depth < 4) {
        const std::string msg =
            "warning: hold_frames with ring_depth " +
            std::to_string(cfg.ring_depth) +
            " < 4 - a consumer holding frames will stall producers "
            "quickly; consider ring_depth >= 4";
        if (cfg.log) cfg.log(msg);
        else fprintf(stderr, "%s\n", msg.c_str());
    }

    auto StepAt = [&](int idx) -> const StepDesc& { return cfg.steps.at(idx); };

    const LayerDesc& l0 = cfg.layers[0];
    ExecPlan plan;
    int engine_step_idx = -1, post_step_idx = -1;
    // M4b: every child's postprocess step index (Postprocess(Ctc|Argmax) at
    // a layer >= 1) - sink validation (below) needs the full set, alongside
    // post_step_idx, to know what an Events sink may legally reference.
    // Filled by the per-child loop below, one entry per accepted child (in
    // cfg.layers[1..] order).
    std::vector<int> child_post_indices;
    if (l0.steps.size() == 3) {
        const StepDesc& p = StepAt(l0.steps[0]);
        const StepDesc& e = StepAt(l0.steps[1]);
        const StepDesc& q = StepAt(l0.steps[2]);
        if (p.kind != StepKind::Process || p.input != -1)
            FailGraph("layer 0 step 0 (want Process, input -1)");
        if (e.kind != StepKind::Engine || e.input != l0.steps[0])
            FailGraph("layer 0 step 1 (want Engine fed from the Process step)");
        if (q.kind != StepKind::Postprocess || q.input != l0.steps[1])
            FailGraph("layer 0 step 2 (want Postprocess fed from the Engine step)");
        if (q.family != Family::YoloDetect && q.family != Family::YoloE2E)
            FailGraph("Postprocess(Ctc|Argmax) at layer 0");
        engine_step_idx = l0.steps[1];
        post_step_idx = l0.steps[2];
        plan.engine_path = e.engine_path;
        plan.l0_family = q.family;
        plan.score_thresh = q.score_thresh;
        plan.iou_thresh = q.iou_thresh;
        ResolveNormColor(e, kL0DefOffset, kL0DefScale,
                         /*def_rgb=*/true, plan.l0_norm_offset,
                         plan.l0_norm_scale, &plan.l0_rgb);
    } else if (l0.steps.size() == 2) {
        // Process step implied: the Engine reads straight from the raw
        // stream source (input -1) and the core inserts preprocessing.
        const StepDesc& e = StepAt(l0.steps[0]);
        const StepDesc& q = StepAt(l0.steps[1]);
        if (e.kind != StepKind::Engine || e.input != -1)
            FailGraph("layer 0 step 0 (want Engine, input -1 - Process implied)");
        if (q.kind != StepKind::Postprocess || q.input != l0.steps[0])
            FailGraph("layer 0 step 1 (want Postprocess fed from the Engine step)");
        if (q.family != Family::YoloDetect && q.family != Family::YoloE2E)
            FailGraph("Postprocess(Ctc|Argmax) at layer 0");
        engine_step_idx = l0.steps[0];
        post_step_idx = l0.steps[1];
        plan.engine_path = e.engine_path;
        plan.l0_family = q.family;
        plan.score_thresh = q.score_thresh;
        plan.iou_thresh = q.iou_thresh;
        ResolveNormColor(e, kL0DefOffset, kL0DefScale,
                         /*def_rgb=*/true, plan.l0_norm_offset,
                         plan.l0_norm_scale, &plan.l0_rgb);
    } else {
        std::ostringstream os;
        os << l0.steps.size() << " steps at layer 0";
        FailGraph(os.str());
    }
    plan.sahi = l0.sahi;
    plan.sahi_tile = l0.sahi_tile;
    plan.sahi_overlap = l0.sahi_overlap;
    plan.sahi_merge_iou = l0.sahi_merge_iou;
    plan.sahi_full_frame = l0.sahi_full_frame;
    plan.sahi_serial = l0.sahi_serial;
    // M4a: SAHI's cross-tile merge is a merge NMS pass (see GpuLoop's SAHI
    // blocks below, LaunchNms on the pooled tile+whole-frame candidates) -
    // undefined for a family whose whole point is "already NMS-free,
    // one-to-one matched in-graph" (see postprocess.h's LaunchYoloE2EBatched
    // WHY-comment). Named and fatal, matching this file's house style for
    // scope violations (the layer-1-sahi check just above is the same
    // shape) rather than silently ignoring sahi= or producing tile-boundary
    // duplicate detections.
    if (plan.sahi && plan.l0_family == Family::YoloE2E) {
        throw std::runtime_error(
            "SAHI requires the yolo family; cross-tile merge NMS is not "
            "defined for e2e heads in v1");
    }

    // M4b: which steps are a legal cascade-child ANCHOR (a
    // Postprocess(Ctc|Argmax), at any layer >= 1) - computed once, over ALL
    // of cfg.steps, BEFORE validating any individual child layer below. This
    // is what makes the "chained cascade" check order-independent: it does
    // not matter whether the layer a chain incorrectly targets was declared
    // earlier or later in cfg.layers, only whether the targeted step really
    // is some child's Postprocess (as opposed to layer 0's, which is never
    // Ctc/Argmax - see the family check above).
    std::vector<int> ctc_argmax_post_steps;
    for (size_t i = 0; i < cfg.steps.size(); i++) {
        const StepDesc& s = cfg.steps[i];
        if (s.kind == StepKind::Postprocess &&
            (s.family == Family::Ctc || s.family == Family::Argmax)) {
            ctc_argmax_post_steps.push_back((int)i);
        }
    }

    // M4b: layers[1..] are zero or more SIBLING recognition children - each
    // must independently be [Engine(input = layer 0's Postprocess step)] ->
    // [Postprocess(Ctc|Argmax)]. "Sibling" is enforced by the Engine.input
    // check below: every child's Engine must be fed from post_step_idx
    // (the DETECTOR's own Postprocess step), never from another child's -
    // that would be a depth-3+ chain (a child of a child), which this
    // executor does not run (see kV1Supported and the named error below).
    for (size_t li = 1; li < cfg.layers.size(); li++) {
        const LayerDesc& l = cfg.layers[li];
        // M2: SAHI (graph.h's LayerDesc::sahi) is only meaningful on the
        // yolo detection layer - a cascade child has no tile grid to speak
        // of. Before this check, a non-layer-0 layer's sahi fields were
        // simply never read - silently ignoring a caller's sahi= there
        // rather than telling them it can't apply. Named and fatal instead,
        // matching this file's house style for scope violations.
        if (l.sahi) {
            std::ostringstream os;
            os << "sahi on layer " << li << " ('" << l.name << "') - SAHI "
                  "tiling is only valid on the yolo detection layer (today, "
                  "always layer 0); layer " << li << " is a Ctc|Argmax "
                  "cascade child, which has no tile grid - set sahi on "
                  "layer 0 instead (Python: Layer(\"detect\", sahi=...), "
                  "not a cascade child layer)";
            throw std::runtime_error(os.str());
        }
        if (l.steps.size() != 2) {
            std::ostringstream os;
            os << l.steps.size() << " steps at layer " << li;
            FailGraph(os.str());
        }
        const StepDesc& e = StepAt(l.steps[0]);
        const StepDesc& q = StepAt(l.steps[1]);
        if (e.kind != StepKind::Engine || e.input != post_step_idx) {
            // Distinguish "fed from some OTHER child's Postprocess" (a
            // named, deliberate scope rejection - deep chains are a
            // documented future generalization, not an oversight) from any
            // other malformed wiring (the generic FailGraph below, same
            // house style as every other shape check in this function).
            const bool via_child =
                e.kind == StepKind::Engine &&
                std::find(ctc_argmax_post_steps.begin(),
                         ctc_argmax_post_steps.end(),
                         e.input) != ctc_argmax_post_steps.end();
            if (via_child) {
                throw std::runtime_error(
                    "v1 executor supports a depth-2 tree: one detector "
                    "layer feeding sibling recognition layers; chained "
                    "cascades (a child of a child) are not yet executable");
            }
            std::ostringstream os;
            os << "layer " << li << " step 0 (want Engine fed from layer "
                  "0's postprocess step)";
            FailGraph(os.str());
        }
        if (q.kind != StepKind::Postprocess || q.input != l.steps[0] ||
            (q.family != Family::Ctc && q.family != Family::Argmax)) {
            std::ostringstream os;
            os << "layer " << li << " step 1 (want Postprocess(Ctc|Argmax) "
                  "fed from the Engine step)";
            FailGraph(os.str());
        }
        ChildPlan cp;
        cp.name = l.name;
        cp.engine_path = e.engine_path;
        cp.family = q.family;
        // WHY default: the cascade position is today's only crop-edge
        // target (fed by a cross-layer crop edge - see the Engine input
        // check above); LPRNet/OCR (Ctc) is the original concrete example,
        // so an Argmax classifier (or any additional sibling) inherits the
        // SAME position default unless it overrides norm/color explicitly
        // (e.g. per-channel ImageNet norm - see examples/
        // classify_detections.py).
        bool rgb = false;
        ResolveNormColor(e, kL1DefOffset, kL1DefScale, /*def_rgb=*/false,
                         cp.norm_offset, cp.norm_scale, &rgb);
        cp.rgb = rgb ? 1 : 0;
        plan.children.push_back(cp);
        child_post_indices.push_back(l.steps[1]);
    }

    // ---- Sink validation (Phase A1: endpoint sinks) --------------------
    // Unknown targets are rejected HERE, at construction (Validate() runs
    // inside Impl's ctor, before Setup() spawns anything) - see graph.h's
    // SinkDesc WHY-comment. This checks GRAMMAR only (scheme + wiring
    // shape); a scheme-valid but unreachable/malformed target (e.g.
    // "tcp://noport") is still caught at construction time, just one level
    // deeper - see pipeline.cpp's EventsSink/RelaySink Setup() code.
    for (size_t i = 0; i < cfg.sinks.size(); i++) {
        const SinkDesc& sd = cfg.sinks[i];
        std::ostringstream tag;
        tag << "sink " << i;
        if (sd.stream_id != -1 &&
            (sd.stream_id < 0 || (size_t)sd.stream_id >= cfg.streams.size())) {
            std::ostringstream os;
            os << tag.str() << " stream_id " << sd.stream_id << " does not exist";
            throw std::runtime_error(os.str());
        }
        if (sd.kind == SinkKind::Events) {
            // v1/M4b: accept either the executed graph's layer-0
            // (YoloDetect|YoloE2E) Postprocess step, or ANY sibling child's
            // (Ctc|Argmax) Postprocess step - all deliver the same
            // FrameResult shape (results are per-frame, not per-layer; a
            // child's own texts/labels ride in FrameResult::children, plus
            // the back-compat flat fields - see result.h).
            const bool is_child_post =
                std::find(child_post_indices.begin(), child_post_indices.end(),
                         sd.input) != child_post_indices.end();
            if (sd.input != post_step_idx && !is_child_post) {
                std::ostringstream os;
                os << tag.str() << " (Events) input " << sd.input
                   << " must be the executed graph's Postprocess step "
                      "(detector = step " << post_step_idx;
                for (int idx : child_post_indices)
                    os << ", or a cascade child = step " << idx;
                os << ")";
                throw std::runtime_error(os.str());
            }
            if (sd.target != "stdout" && sd.target.rfind("tcp://", 0) != 0 &&
                sd.target.rfind("file://", 0) != 0) {
                std::ostringstream os;
                os << tag.str() << " (Events) target '" << sd.target
                   << "' - want \"tcp://host:port\", \"file:///abs/path\", "
                      "or \"stdout\"";
                throw std::runtime_error(os.str());
            }
        } else {  // SinkKind::StreamRelay
            if (sd.input != -1) {
                std::ostringstream os;
                os << tag.str() << " (StreamRelay) input " << sd.input
                   << " (want -1 - it forwards the raw/original source, "
                      "not a postprocess output)";
                throw std::runtime_error(os.str());
            }
            if (sd.stream_id == -1) {
                std::ostringstream os;
                os << tag.str()
                   << " (StreamRelay) stream_id -1 (\"all streams\") is not "
                      "supported in v1 - one relay target is one concrete "
                      "stream (see graph.h's SinkDesc WHY-comment)";
                throw std::runtime_error(os.str());
            }
            if (sd.target.rfind("rtsp://", 0) != 0 &&
                sd.target.rfind("tcpraw://", 0) != 0) {
                std::ostringstream os;
                os << tag.str() << " (StreamRelay) target '" << sd.target
                   << "' - want \"rtsp://host:port/name\" (or the "
                      "\"tcpraw://host:port\" fallback)";
                throw std::runtime_error(os.str());
            }
        }
    }

    return plan;
}

// ---- Batching state (moved from rtsp_infer_multi.cpp, logic unchanged) --

struct Nv12Ring;  // forward decl - SlotMeta needs the type, not the body yet

struct SlotMeta {
    int stream_id = -1;
    int frame_no = 0;        // per-stream frame index
    int64_t pts_us = -1;
    Clock::time_point t_pop;   // when the producer popped the decoded frame
    Clock::time_point t_ready; // preprocess synced, slot marked ready
    // Full-res NV12 copy of this frame (owned by the producer's ring, held
    // in_use - see Nv12Ring - until the GPU thread releases ring_slot right
    // before Recycle()). Stage 2 crops OCR inputs from here at source
    // resolution.
    const uint8_t* nv12 = nullptr;
    size_t nv12_pitch = 0;
    // Which ring, and which of its slots, backs `nv12` above - so the GPU
    // thread can ReleaseSlot() it once this SlotMeta's buffer has been
    // fully consumed (SAHI/OCR crops done, whole-frame pass synced).
    Nv12Ring* ring = nullptr;
    int ring_slot = -1;
};

// Ring of full-res NV12 frame copies, one per producer (cascade frame
// ownership: copy at preprocess time, keep releasing the decoder surface
// early).
//
// Correctness here is by REFCOUNT, not by depth: AcquireSlot(seq) maps a
// producer's monotonic frame counter to a ring slot (seq % depth) but then
// blocks until that physical slot's `in_use` flag is clear - i.e. until the
// GPU thread has released the OLD frame that lived there (see
// SlotMeta::ring/ring_slot and the GPU thread's ReleaseSlot() call right
// before Recycle()). A prior version of this comment argued depth 4 > the
// 3-buffer batcher ring was itself sufficient; that's wrong - a producer can
// acquire many batch slots (across kNumBufs buffers x MaxBatch) before any
// of them is taken and processed by the GPU thread, so a purely counter-based
// `inferred % depth` index could wrap and overwrite a slot the GPU thread
// hadn't read yet (silent wrong-frame crops in SAHI/OCR). With the refcount
// wait, depth is a latency/VRAM knob - how many frames of slack a producer
// gets before it has to wait on the GPU thread (or, with cfg.hold_frames, on
// a slow CONSUMER - see PipelineConfig::ring_depth in graph.h) - not a
// correctness bound. It's plumbed from PipelineConfig::ring_depth (Validate()
// enforces >= 2) rather than fixed at compile time, so a caller doing tier-3
// frame holding can size it to how many frames it plans to hold at once.
struct Nv12Ring {
    uint8_t* buf = nullptr;
    size_t pitch = 0;
    int w = 0, h = 0;
    size_t slot_bytes = 0;
    int depth = 4;

    // Per-slot refcount (single-writer, single-reader-set per slot in
    // practice, but guarded properly since producer and GPU threads touch
    // it from different threads). vector<char>, not vector<bool>: avoids
    // the bit-packed specialization (no real memory win here, and it can't
    // hand out a normal bool& anyway) for no benefit given every access is
    // already mutex-guarded.
    std::mutex m;
    std::condition_variable cv;
    std::vector<char> in_use;

    explicit Nv12Ring(int d = 4) : depth(d), in_use((size_t)d, 0) {}
    Nv12Ring(const Nv12Ring&) = delete;
    Nv12Ring& operator=(const Nv12Ring&) = delete;

    // Lazy: dims are only known at the first decoded frame. Also handles a
    // mid-stream resolution change (HandleVideoSequence renegotiation) by
    // reallocating at the new size.
    void Ensure(int width, int height) {
        std::unique_lock<std::mutex> lk(m);
        if (buf && width == w && height == h) return;  // fast path: no realloc
        // About to free/realloc `buf`: a queued batch's SlotMeta::nv12 may
        // still point into it (SAHI/OCR read it later in the GPU thread's
        // cycle), so freeing here without waiting would be a second,
        // latent use-after-free on top of Bug 1. Wait for every slot to be
        // released first - same in_use bookkeeping AcquireSlot/ReleaseSlot
        // use, so this can't race a producer's own wait.
        cv.wait(lk, [&] {
            for (char u : in_use) if (u) return false;
            return true;
        });
        cudaFree(buf);
        w = width;
        h = height;
        const int rows = h * 3 / 2;  // luma + interleaved UV, same pitch
        if (cudaMallocPitch((void**)&buf, &pitch, (size_t)w,
                            (size_t)rows * depth) != cudaSuccess) {
            throw std::runtime_error("cudaMallocPitch failed for NV12 ring");
        }
        slot_bytes = pitch * rows;
    }
    uint8_t* Slot(int i) const { return buf + (size_t)i * slot_bytes; }

    // Blocks until ring slot (seq % depth) is free (i.e. the GPU thread -
    // or, under hold_frames, a consumer - released whatever frame
    // previously lived there), then claims it. `seq` is the producer's
    // monotonic inferred-frame counter, so slots are handed out in the same
    // round-robin order as before - the refcount just makes reuse WAIT
    // instead of blindly overwriting.
    int AcquireSlot(int seq) {
        const int s = seq % depth;
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !in_use[s]; });
        in_use[s] = true;
        return s;
    }
    void ReleaseSlot(int s) {
        std::lock_guard<std::mutex> lk(m);
        in_use[s] = false;
        cv.notify_all();
    }

    ~Nv12Ring() { cudaFree(buf); }
};

// Tier 3 (hold_frames) holder: keeps one Nv12Ring slot's refcount claimed
// via a type-erased shared_ptr<void> (see FrameResult::frame_hold in
// result.h - kept CUDA/pipeline-internals-free by design, Pimpl-style). The
// deleter installed alongside this struct only touches a mutex + cv
// (Nv12Ring::ReleaseSlot), so it's safe to run from ANY thread - including
// Python's garbage collector, which in practice is where the last reference
// to a hold_frames FrameResult often actually dies. `ctx` rides along for
// Pipeline::FetchFrame (tier 2): a consumer thread has no CUDA context of
// its own current, so FetchFrame needs to know which one to make current
// before its cuMemcpy2D.
struct Nv12RingHold {
    Nv12Ring* ring;
    int slot;
    CUcontext ctx;
};

struct BatchBuffer {
    float* base = nullptr;                     // max_batch input slots
    std::vector<PostprocImageParams> params;   // host, per slot
    std::vector<SlotMeta> meta;
    int allocated = 0;   // slots handed to producers
    int ready = 0;       // slots whose preprocess completed
};

// Ring of kNumBufs batch buffers: one being filled by producers, up to one
// being processed by the GPU thread, the rest free or queued full. Strict
// ping-pong measured badly at 16 streams: a burst fills the single fill
// buffer, then every producer blocks for the whole ~17 ms of the in-flight
// batch (19 ms pop->ready). With a third buffer producers rotate instead.
constexpr int kNumBufs = 3;

struct Batcher {
    std::mutex m;
    std::condition_variable cv_ready;  // GPU waits for frames
    std::condition_variable cv_space;  // producers wait for a free buffer
    BatchBuffer buf[kNumBufs];
    BatchBuffer* filling = nullptr;          // producers' current target
    std::vector<BatchBuffer*> free_bufs;     // recycled, empty
    std::vector<BatchBuffer*> pending;       // full, FIFO for the GPU
    int active_producers = 0;
    int max_batch = 0;
    // Part 1 (Pipeline::Stop): when set, Acquire() stops handing out slots
    // and returns nullptr instead - the producer treats that exactly like
    // completion (see ProducerLoop). Take() is untouched: the GPU thread
    // keeps draining pending/filling buffers normally so ring slots still
    // get released and no producer can be left stuck in AcquireSlot.
    bool stopping = false;

    void Init(int producers, int mb) {
        active_producers = producers;
        max_batch = mb;
        filling = &buf[0];
        for (int i = 1; i < kNumBufs; i++) free_bufs.push_back(&buf[i]);
    }

    void SetStopping() {
        std::lock_guard<std::mutex> lk(m);
        stopping = true;
        cv_space.notify_all();
        cv_ready.notify_all();
    }

    // Producer: claim a slot in the filling buffer; on full, queue it for
    // the GPU and rotate to a free buffer. Blocks only when the ring is
    // exhausted (true backpressure: kNumBufs*max_batch frames in flight).
    BatchBuffer* Acquire(int* slot) {
        std::unique_lock<std::mutex> lk(m);
        for (;;) {
            if (stopping) return nullptr;
            if (filling && filling->allocated < max_batch) {
                *slot = filling->allocated++;
                return filling;
            }
            if (filling) {  // full: hand to GPU, rotate
                pending.push_back(filling);
                filling = nullptr;
                cv_ready.notify_one();
            }
            if (!free_bufs.empty()) {
                filling = free_bufs.back();
                free_bufs.pop_back();
                continue;
            }
            cv_space.wait(lk);
        }
    }

    void MarkReady(BatchBuffer* b) {
        std::lock_guard<std::mutex> lk(m);
        b->ready++;
        // First ready slot wakes an idle GPU thread (greedy close); a
        // completed buffer releases a Take waiting out stragglers.
        if (b->ready == 1 || b->ready == b->allocated) cv_ready.notify_one();
    }

    // GPU thread, greedy: oldest full buffer first; otherwise close the
    // filling buffer as soon as anything in it is ready (waiting only for
    // sub-ms in-flight stragglers) - under saturation the batch size then
    // self-regulates to arrivals-per-GPU-cycle instead of locking to
    // full-batch mode.
    BatchBuffer* Take() {
        std::unique_lock<std::mutex> lk(m);
        for (;;) {
            if (!pending.empty()) {
                BatchBuffer* b = pending.front();
                if (b->ready == b->allocated) {
                    pending.erase(pending.begin());
                    return b;
                }
            } else if (filling && filling->ready > 0) {
                BatchBuffer* b = filling;
                filling = nullptr;  // close: producers rotate to a free buf
                cv_space.notify_all();
                cv_ready.wait(lk, [&] { return b->ready == b->allocated; });
                return b;
            } else if (active_producers == 0 &&
                       (!filling || filling->allocated == 0)) {
                return nullptr;  // shutdown: everything drained
            }
            cv_ready.wait(lk);
        }
    }

    void Recycle(BatchBuffer* b) {
        std::lock_guard<std::mutex> lk(m);
        b->allocated = 0;
        b->ready = 0;
        free_bufs.push_back(b);
        cv_space.notify_all();
    }

    void ProducerDone() {
        std::lock_guard<std::mutex> lk(m);
        if (--active_producers == 0) cv_ready.notify_all();
    }
};

// ---- Bounded result queue -----------------------------------------------
// GPU thread pushes with BLOCKING semantics when full by default (settled
// v0 backpressure decision). Stop() flips `stopping`: blocked/future pushes
// turn into silent drops and return immediately (so the GPU thread can go
// on to drain and exit), and Poll() reports Finished once the queue is both
// stopping/finished AND empty.
//
// Part 2 design doc: DropOldest mode (SetBackpressure) never blocks Push -
// a full queue instead pops its front (oldest result) to make room, so the
// GPU thread stays at the producers' pace even against an arbitrarily slow
// consumer. `dropped_` counts how many results DropOldest has ever evicted
// (see Pipeline::DroppedResults' WHY-comment for who reads this and why).
//
// Part 2 lifecycle audit, all satisfied by the flags already here:
//  - Poll() before Start(): neither stopping_ nor finished_ is set yet and
//    the queue is empty, so wait_for's predicate is false and it blocks for
//    the full timeout_ms, then returns Timeout (never crashes/hangs
//    forever). If Stop() ran first, stopping_ is already true -> Finished.
//  - Poll() after Finished: finished_ latches true forever (only ever set,
//    never cleared), so every subsequent Poll() sees the predicate
//    immediately true and returns Finished without waiting - not just the
//    first time.
//  - Slow consumer + Stop() (Block mode): Stop()'s step (a) is exactly
//    `queue.Stop()` below, which sets stopping_ under the same mutex Push()
//    waits on and notifies cv_push_ - a Push() blocked because the queue is
//    full wakes, re-checks the predicate (now true via stopping_), and
//    drops instead of pushing. This must happen BEFORE Stop() touches the
//    batcher/producer flags (see Impl::StopImpl's ordering) so the GPU
//    thread is never stuck holding a full queue while producers/batcher are
//    simultaneously being torn down under it. In DropOldest mode Push()
//    never waits at all, so this wake-up step is simply a no-op for it.

class ResultQueue {
public:
    void SetCapacity(size_t cap) { capacity_ = cap; }
    void SetBackpressure(Backpressure bp) { backpressure_ = bp; }

    void Push(FrameResult r) {
        std::unique_lock<std::mutex> lk(m_);
        if (backpressure_ == Backpressure::DropOldest) {
            if (!stopping_ && q_.size() >= capacity_) {
                // RAII bonus: pop_front() destroys the evicted FrameResult
                // right here. If it was a tier-3 (hold_frames) result,
                // that destructor drops its frame_hold shared_ptr, and -
                // assuming no other copy of this FrameResult is alive
                // elsewhere - the refcount hits zero and the ring slot is
                // released via Nv12RingHold's deleter. No special-casing
                // needed for DropOldest vs. a consumer's own Poll(): both
                // are just "a FrameResult got destroyed".
                q_.pop_front();
                dropped_.fetch_add(1, std::memory_order_relaxed);
            }
            if (stopping_) return;  // dropped: shutting down
            q_.push_back(std::move(r));
            cv_pop_.notify_one();
            return;
        }
        cv_push_.wait(lk, [&] { return stopping_ || q_.size() < capacity_; });
        if (stopping_) return;  // dropped: shutting down
        q_.push_back(std::move(r));
        cv_pop_.notify_one();
    }

    uint64_t Dropped() const { return dropped_.load(std::memory_order_relaxed); }

    // Marks the run complete (all producers + the GPU thread are done).
    // Together with `stopping_` this lets Poll() report Finished once the
    // queue drains.
    void MarkFinished() {
        std::lock_guard<std::mutex> lk(m_);
        finished_ = true;
        cv_pop_.notify_all();
    }

    void Stop() {
        std::lock_guard<std::mutex> lk(m_);
        stopping_ = true;
        cv_push_.notify_all();
        cv_pop_.notify_all();
    }

    Pipeline::PollStatus Poll(FrameResult* out, int timeout_ms) {
        std::unique_lock<std::mutex> lk(m_);
        const bool got = cv_pop_.wait_for(
            lk, std::chrono::milliseconds(timeout_ms),
            [&] { return !q_.empty() || finished_ || stopping_; });
        if (!got) return Pipeline::PollStatus::Timeout;
        if (!q_.empty()) {
            *out = std::move(q_.front());
            q_.pop_front();
            cv_push_.notify_one();
            return Pipeline::PollStatus::Ok;
        }
        // Empty AND (finished or stopping): run over, nothing left to drain.
        return Pipeline::PollStatus::Finished;
    }

private:
    std::mutex m_;
    std::condition_variable cv_push_, cv_pop_;
    // RAII bonus, shutdown path: Stop() (see StopImpl) never explicitly
    // drains q_ - it just latches stopping_/lets the GPU thread finish, and
    // the deque itself is torn down for free when this ResultQueue (a
    // member of Impl) is destroyed. Destroying a deque<FrameResult>
    // destroys every element still queued at that point, which for any
    // never-Polled tier-3 result runs its frame_hold destructor exactly
    // like a normal Poll()'d-then-discarded one would - so a slow consumer
    // that never drained the queue before Stop() still can't leak a ring
    // slot. No special-casing needed here either.
    std::deque<FrameResult> q_;
    size_t capacity_ = 256;
    Backpressure backpressure_ = Backpressure::Block;
    std::atomic<uint64_t> dropped_{0};
    bool stopping_ = false;
    bool finished_ = false;
};

// ---- Endpoint sinks (Phase A1) -------------------------------------------
// WHY (see graph.h's SinkKind/SinkDesc for the top-level rationale): a sink
// must NEVER backpressure GpuLoop or a producer. Every sink here follows the
// same shape as ResultQueue's DropOldest branch above: a small bounded
// queue that a dedicated thread drains, where overflow evicts the OLDEST
// entry rather than ever blocking the pusher.

// Sleeps up to `dur`, waking immediately if `stop_flag` goes true - used by
// every sink's backoff/pacing waits so Stop() never has to wait out a full
// sleep before a sink thread notices shutdown (mirrors the producer's own
// reconnect-backoff sleep, just interruptible).
template <class Rep, class Period>
void InterruptibleSleep(std::atomic<bool>& stop_flag, std::mutex& m,
                        std::condition_variable& cv,
                        std::chrono::duration<Rep, Period> dur) {
    std::unique_lock<std::mutex> lk(m);
    cv.wait_for(lk, dur, [&] { return stop_flag.load(std::memory_order_relaxed); });
}

// Bounded (1024) drop-oldest queue of formatted lines for one Events sink -
// same DropOldest contract as ResultQueue's (see its WHY-comment), just
// specialized to std::string and always drop-oldest (an Events sink has no
// Block mode: it must never be the thing that slows GpuLoop down). Overflow
// AND "stuck reconnecting" both funnel through the same eviction path here:
// while a TCP sink is down, the writer thread simply isn't popping, so the
// queue fills to capacity and starts evicting its own oldest entries -
// "drop lines meanwhile, count them" falls out of that for free, no
// separate bookkeeping needed.
class LineQueue {
public:
    void Push(std::string s) {
        std::lock_guard<std::mutex> lk(m_);
        if (stopping_) return;
        if (q_.size() >= kCapacity) {
            q_.pop_front();
            dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        q_.push_back(std::move(s));
        cv_.notify_one();
    }
    // Waits up to timeout_ms for a line. False on timeout, or once stopping
    // AND drained (still delivers whatever was queued before Stop(), so a
    // shutdown flushes rather than silently losing the tail).
    bool Pop(std::string* out, int timeout_ms) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                    [&] { return !q_.empty() || stopping_; });
        if (q_.empty()) return false;
        *out = std::move(q_.front());
        q_.pop_front();
        return true;
    }
    void Stop() {
        std::lock_guard<std::mutex> lk(m_);
        stopping_ = true;
        cv_.notify_all();
    }
    uint64_t Dropped() const { return dropped_.load(std::memory_order_relaxed); }

private:
    static constexpr size_t kCapacity = 1024;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::string> q_;
    std::atomic<uint64_t> dropped_{0};
    bool stopping_ = false;
};

// Blocking send() loop handling partial writes; MSG_NOSIGNAL so a peer that
// hung up doesn't SIGPIPE-kill the process (the sink thread just sees a
// failed write and reconnects, same as any other transient failure).
bool SendAll(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        const ssize_t n = send(fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += (size_t)n;
    }
    return true;
}

// One Events sink: a target (parsed once at construction - see Setup()),
// its own LineQueue, and the thread draining it (spawned in Start(), joined
// in StopImpl()). `wait_m`/`wait_cv` back InterruptibleSleep for this sink's
// reconnect backoff.
struct EventsSink {
    SinkDesc desc;
    LineQueue queue;
    std::thread thread;
    std::atomic<bool> stopping{false};
    std::mutex wait_m;
    std::condition_variable wait_cv;

    enum class Kind { Tcp, File, Stdout } kind = Kind::Stdout;
    std::string host;
    int port = 0;
    std::string path;
};

// One StreamRelay sink: republishes ONE stream's PacketRing onward via
// libavformat's RTSP output muxer, -c copy semantics (no re-encode, no
// D2H - see the file's top WHY-comment). See Pipeline::Impl::RelaySinkLoop
// for the connect/publish/pace/reconnect loop.
struct RelaySink {
    SinkDesc desc;
    std::thread thread;
    std::atomic<bool> stopping{false};
    std::mutex wait_m;
    std::condition_variable wait_cv;
};

// Compact NDJSON line for the Events sink - snprintf/append only (no
// iostreams), formatted on the GPU thread once per FrameResult (see
// GpuLoop): cheap because a frame carries at most tens of detections, but
// kept lean anyway since it still runs on the hot per-frame path.
void AppendJsonEscaped(std::string& out, const std::string& s) {
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
}

std::string FormatEventLine(const FrameResult& fr) {
    char buf[256];
    std::string line;
    line.reserve(160 + fr.detections.size() * 96 + fr.texts.size() * 32);
    line += "{\"stream\":";
    snprintf(buf, sizeof(buf), "%d", fr.stream_id);
    line += buf;
    line += ",\"frame\":";
    snprintf(buf, sizeof(buf), "%d", fr.frame_no);
    line += buf;
    line += ",\"pts_us\":";
    snprintf(buf, sizeof(buf), "%lld", (long long)fr.pts_us);
    line += buf;
    line += ",\"batch\":";
    snprintf(buf, sizeof(buf), "%d", fr.batch_seq);
    line += buf;
    line += ",\"dets\":[";
    for (size_t i = 0; i < fr.detections.size(); i++) {
        const Detection& d = fr.detections[i];
        if (i) line += ",";
        snprintf(buf, sizeof(buf),
                 "{\"x\":%.2f,\"y\":%.2f,\"w\":%.2f,\"h\":%.2f,\"score\":%.4f,"
                 "\"cls\":%d}",
                 d.x, d.y, d.w, d.h, d.score, d.cls);
        line += buf;
    }
    line += "],\"texts\":[";
    for (size_t i = 0; i < fr.texts.size(); i++) {
        if (i) line += ",";
        line += "\"";
        AppendJsonEscaped(line, fr.texts[i]);
        line += "\"";
    }
    line += "],\"ms\":{";
    snprintf(buf, sizeof(buf), "\"pre\":%.3f,\"queue\":%.3f,\"gpu\":%.3f",
             fr.ms_pop_to_ready, fr.ms_ready_to_take, fr.ms_take_to_done);
    line += buf;
    line += "},\"frame_addr\":\"0x";
    snprintf(buf, sizeof(buf), "%llx", (unsigned long long)fr.frame_addr);
    line += buf;
    line += "\"}";
    return line;
}

// ---- CPU verify reference ------------------------------------------------
// Independent host-side mirror of the postprocess math (same role as the
// checkpoint references): raw tensor -> candidates -> greedy NMS with the
// composite (score, cls, x) order the kernels use.

struct RefDet {
    float x, y, w, h, score;
    int cls;
};

std::vector<RefDet> CpuReference(const float* raw, int anchors, int classes,
                                 const PostprocImageParams& p,
                                 float score_thresh, float iou_thresh) {
    std::vector<RefDet> cands;
    for (int i = 0; i < anchors; i++) {
        int best_cls = 0;
        float best = 0.f;
        for (int c = 0; c < classes; c++) {
            const float s = raw[(4 + c) * anchors + i];
            if (s > best) { best = s; best_cls = c; }
        }
        if (best < score_thresh) continue;
        const float cx = raw[0 * anchors + i], cy = raw[1 * anchors + i];
        const float w = raw[2 * anchors + i], h = raw[3 * anchors + i];
        float x0 = (cx - w / 2.f - p.lb.pad_x) / p.lb.scale;
        float y0 = (cy - h / 2.f - p.lb.pad_y) / p.lb.scale;
        float x1 = x0 + w / p.lb.scale, y1 = y0 + h / p.lb.scale;
        x0 = std::max(x0, 0.f);
        y0 = std::max(y0, 0.f);
        x1 = std::min(x1, (float)p.src_w);
        y1 = std::min(y1, (float)p.src_h);
        cands.push_back({x0, y0, std::max(x1 - x0, 0.f),
                         std::max(y1 - y0, 0.f), best, best_cls});
    }
    std::sort(cands.begin(), cands.end(), [](const RefDet& a, const RefDet& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.cls != b.cls) return a.cls < b.cls;
        return a.x < b.x;
    });
    std::vector<RefDet> kept;
    std::vector<bool> alive(cands.size(), true);
    for (size_t i = 0; i < cands.size(); i++) {
        if (!alive[i]) continue;
        kept.push_back(cands[i]);
        for (size_t j = i + 1; j < cands.size(); j++) {
            if (!alive[j] || cands[j].cls != cands[i].cls) continue;
            const RefDet &a = cands[i], &b = cands[j];
            const float ix0 = std::max(a.x, b.x), iy0 = std::max(a.y, b.y);
            const float ix1 = std::min(a.x + a.w, b.x + b.w);
            const float iy1 = std::min(a.y + a.h, b.y + b.h);
            const float inter =
                std::max(ix1 - ix0, 0.f) * std::max(iy1 - iy0, 0.f);
            const float uni = a.w * a.h + b.w * b.h - inter;
            if (uni > 0.f && inter / uni > iou_thresh) alive[j] = false;
        }
    }
    return kept;
}

}  // namespace

// ---------------------------------------------------------------------
// Pipeline::Impl
// ---------------------------------------------------------------------

struct Pipeline::Impl {
    PipelineConfig cfg;
    ExecPlan plan;
    int n_streams = 0;
    size_t in_stride = 0;

    CUcontext ctx = nullptr;
    CUdevice dev = 0;  // valid iff ctx != nullptr - Teardown() needs it for
                       // cuDevicePrimaryCtxRelease
    CUvideoctxlock ctx_lock = nullptr;
    std::unique_ptr<TrtEngine> engine;
    int classes = 0, anchors = 0, mb = 0;
    // M4a: yolo-e2e's row count per image (300 for yolo26n) - the
    // YoloE2E-family counterpart of classes/anchors above, read from the
    // engine's own OutputDims in Setup() (see its I/O-shape check). Unused
    // (0) when plan.l0_family == YoloDetect.
    int max_dets = 0;

    Batcher batcher;
    std::vector<float*> extra_inputs;
    // deque, not vector: Nv12Ring holds a mutex + condition_variable, so
    // it's neither copyable nor movable - a vector would need to move
    // existing elements on any growth-triggered reallocation (even just
    // to go from empty to n_streams via emplace_back, since ring_depth is
    // now a runtime constructor arg rather than the old fixed-size
    // vector(count) fill-construct). deque never moves already-inserted
    // elements when it grows, so emplace_back works with a non-movable
    // element type; random-access indexing (rings[id]) is unaffected.
    std::deque<Nv12Ring> rings;

    // Phase A1 (endpoint sinks): one PacketRing per stream, same deque-for-
    // non-movable-member reasoning as `rings` above (PacketRing holds a
    // mutex). Always allocated; only pushed into when cfg.ring_seconds > 0
    // (see PacketRing::Push and PipelineConfig::ring_seconds). Owned here,
    // not by the producers, for the same reason as `rings`: a StreamRelay/
    // Events... (actually Events doesn't read it, only StreamRelay and
    // ExtractClip do) sink thread can be reading a snapshot of it
    // independent of any one producer's lifetime.
    std::deque<PacketRing> packet_rings;
    // unique_ptr, not a bare vector<EventsSink>/vector<RelaySink>: each sink
    // holds a mutex/condition_variable/std::thread (non-movable), so the
    // container needs stable element ADDRESSES (Start() hands each thread a
    // raw pointer to its own sink) - unique_ptr in a vector gives that
    // without the deque-vs-vector reasoning `rings`/`packet_rings` need,
    // since these are built once in Setup() and never reallocated/resized
    // after Start() takes their addresses.
    std::vector<std::unique_ptr<EventsSink>> events_sinks;
    std::vector<std::unique_ptr<RelaySink>> relay_sinks;

    // Per-stream skip/decode overrides resolved ONCE here (inheritance from
    // cfg.skip/cfg.key_only applied), so ProducerLoop just reads its own
    // stream's resolved pair instead of re-deriving it (and cfg.skip/
    // cfg.key_only stay meaningful as "the pipeline-level default" for
    // whichever streams don't override).
    struct ResolvedStream {
        int skip;
        bool key_only;
    };
    std::vector<ResolvedStream> resolved_streams;

    PostprocImageParams* d_params = nullptr;
    GpuDetection *d_cands = nullptr, *d_kept = nullptr;
    int *d_counts = nullptr, *d_kept_counts = nullptr;
    std::vector<GpuDetection> h_kept;
    std::vector<int> h_kept_counts;
    std::vector<float> h_raw;  // --verify only

    // ---- Stage-2 cascade scratch (M4b: N sibling children) ----
    // Per-child engine + scratch, parallel to plan.children (same index).
    // A Ctc child uses h_logits (CPU CtcGreedyDecode); an Argmax child uses
    // d_argmax_labels/d_argmax_scores (GPU decode, see LaunchArgmaxBatched
    // in postprocess.h) - the other's fields are simply left at their
    // default (nullptr/empty), same "only one branch is live" contract M1a
    // established for the single-child case.
    struct ChildScratch {
        std::unique_ptr<TrtEngine> engine;
        int h = 0, w = 0, classes = 0, steps = 0;  // steps = CTC time axis
                                                   // (1, unused, for Argmax)
        CropParams* d_crops = nullptr;
        std::vector<float> h_logits;
        int* d_argmax_labels = nullptr;
        float* d_argmax_scores = nullptr;
        std::vector<int> h_argmax_labels;
        std::vector<float> h_argmax_scores;
    };
    std::vector<ChildScratch> children;  // index-parallel to plan.children
    // Crop-RECT scratch (source-frame coordinates only - see CropParams):
    // IDENTICAL across every child for a given batch (the crop kernel's
    // destination size/norm/color come from the CHILD, not from this list),
    // so GpuLoop builds it ONCE per batch and every child's chunked
    // crop-resize call reuses it - see GpuLoop's cascade block.
    std::vector<CropParams> h_crops;
    std::vector<std::pair<int, int>> crop_owner;  // (slot, det index)
    // M4b back-compat (see result.h's FrameResult::texts/labels/
    // label_scores WHY-comment and GpuLoop's compat WHY-comment): index (in
    // `children`/plan.children) of the FIRST Ctc child and FIRST Argmax
    // child, or -1 if none - resolved once in Setup(), not re-derived per
    // frame.
    int first_ctc_child = -1, first_argmax_child = -1;

    // ---- SAHI scratch ----
    float* d_sahi_in = nullptr;
    CropParams* d_sahi_crops = nullptr;
    GpuDetection* d_merge_c = nullptr;
    GpuDetection* d_merge_k = nullptr;
    int *d_merge_n = nullptr, *d_merge_kn = nullptr;
    std::vector<GpuDetection> h_tile_kept;
    std::vector<int> h_tile_kept_counts;
    std::map<std::pair<int, int>, std::vector<TileRect>> sahi_grids;

    cudaStream_t gpu_stream = nullptr;

    // Per-stream state producers write and GetStreamInfo reads; atomics so
    // both sides can touch it lock-free while the run is live.
    struct StreamAtomic {
        std::atomic<int> decoded{0};
        std::atomic<int> reconnects{0};
        std::atomic<bool> failed{false};
    };
    std::vector<std::unique_ptr<StreamAtomic>> stream_atomics;
    std::vector<StreamInfo> stream_info_cache;  // snapshot filled by GetStreamInfo
    // Part 2: GetStreamInfo() writes 3 fields into stream_info_cache[i] one
    // at a time, then returns a reference into it. The individual fields are
    // sourced from atomics (safe to read concurrently with producers), but
    // the WRITE into the plain StreamInfo struct is not itself atomic - two
    // threads calling GetStreamInfo(i) for the same i concurrently would be
    // an unsynchronized concurrent write to the same memory (UB), and could
    // hand back a torn (half-old/half-new) struct. One mutex per stream
    // (rather than a single global one) keeps concurrent calls for
    // DIFFERENT streams from serializing on each other. std::mutex is
    // neither copyable nor movable, so this is unique_ptr<mutex> (same
    // pattern as stream_atomics above), not vector<mutex> directly.
    mutable std::vector<std::unique_ptr<std::mutex>> stream_info_m;

    ResultQueue queue;
    std::atomic<bool> stop{false};       // producer loops' shutdown flag
    std::atomic<bool> started{false};
    // Part 2: Stop() idempotency/lifecycle guards. `stop_seq_started` picks
    // exactly one caller to run the actual shutdown sequence (exchange(), not
    // load+store, so two concurrent Stop() calls can't both see "I'm first").
    // The loser must NOT just return immediately though (see StopImpl): a
    // concurrent Stop() from another thread racing the destructor could
    // otherwise let Teardown() run while the winner is still mid-join,
    // freeing CUDA state out from under a live gpu_thread. So losers block on
    // stop_seq_cv until the winner sets stop_seq_complete under stop_seq_m.
    std::atomic<bool> stop_seq_started{false};
    std::mutex stop_seq_m;
    std::condition_variable stop_seq_cv;
    bool stop_seq_complete = false;

    // Part 2: also read from Start() to reject Start()-after-Stop() (see
    // Pipeline::Start). Kept distinct from stop_seq_started/complete so
    // Start() only needs a relaxed peek, not the join-wait machinery above.
    std::atomic<bool> stop_requested{false};

    std::vector<std::thread> producers;
    std::thread gpu_thread;

    explicit Impl(PipelineConfig c) : cfg(std::move(c)) {
        plan = Validate(cfg);
        n_streams = (int)cfg.streams.size();
        resolved_streams.reserve(n_streams);
        for (const StreamDesc& sd : cfg.streams) {
            resolved_streams.push_back(
                {sd.skip == 0 ? cfg.skip : sd.skip,
                 sd.decode == -1 ? cfg.key_only : (bool)sd.decode});
        }
        stream_atomics.resize(n_streams);
        for (auto& p : stream_atomics) p = std::make_unique<StreamAtomic>();
        stream_info_cache.resize(n_streams);
        stream_info_m.resize(n_streams);
        for (auto& m : stream_info_m) m = std::make_unique<std::mutex>();
        queue.SetCapacity(cfg.queue_capacity);
        queue.SetBackpressure(cfg.backpressure);
        // Part 2: if Setup() throws partway (e.g. CUDA context created, then
        // an alloc fails), the Impl constructor throws too - which means
        // ~Impl() NEVER RUNS (a throwing constructor leaves the object
        // never-fully-constructed, so its destructor is skipped by the
        // language, not just "returns early"). Teardown()'s `if (!ctx)
        // return;` guard only protects a Teardown() call that actually
        // happens; without this try/catch that call never happens at all, so
        // anything Setup() allocated before throwing (CUDA context, engine,
        // device buffers) leaked silently. Catching here and calling
        // Teardown() explicitly is what makes that guard meaningful.
        try {
            Setup();
        } catch (...) {
            Teardown();
            throw;
        }
    }

    ~Impl() {
        StopImpl();
        Teardown();
    }

    void Log(const std::string& s) const {
        if (cfg.log) cfg.log(s);
        else fprintf(stderr, "%s\n", s.c_str());
    }
    void Logf(const char* fmt, ...) const {
        char buf[1024];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        Log(buf);
    }

    void Setup();
    void Teardown();
    void StopImpl();
    void ProducerLoop(int id);
    void GpuLoop();
    // Phase A1 (endpoint sinks) - see the EventsSink/RelaySink struct
    // WHY-comments above.
    void EventsSinkLoop(EventsSink& sink);
    void RelaySinkLoop(RelaySink& sink);
};

void Pipeline::Impl::Setup() {
    CheckCu(cuInit(0), "cuInit");
    CheckCu(cuDeviceGet(&dev, 0), "cuDeviceGet");
    // BLOCKING_SYNC: with N producer threads + a GPU thread the default
    // SCHED_AUTO degrades to yield-loops (threads > cores), turning every
    // per-frame cudaEventSynchronize into a multi-ms wait. Interrupt-driven
    // sync wakes in ~10 us regardless of thread count.
    //
    // PRIMARY context, not a private one (this used to be a plain
    // cuCtxCreate): driver-API allocations are CONTEXT-SCOPED, and
    // torch/CuPy - tier 3's zero-copy interop target (FrameResult::
    // frame_hold / frame_cuda(), see result.h) - always run against the
    // device's PRIMARY context. A device pointer minted in a private
    // context here would simply be invalid in torch's context, making
    // zero-copy interop impossible; retaining the primary context instead
    // means our pointers and torch's are the same address space.
    // SetFlags must run BEFORE Retain - the primary context's sync mode is
    // fixed at first retain/use and can't be changed after. If something
    // in-process already initialized it (another retain, or an earlier
    // CUDA-runtime-API call that lazily created it with default flags),
    // SetFlags fails with CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE; that's not
    // fatal - we just inherit whatever sync mode is already active
    // (degrades to yield-loop sync under thread oversubscription at
    // worst) - log a warning and continue rather than abort.
    const CUresult flag_r =
        cuDevicePrimaryCtxSetFlags(dev, CU_CTX_SCHED_BLOCKING_SYNC);
    if (flag_r != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(flag_r, &s);
        Logf("warning: cuDevicePrimaryCtxSetFlags(BLOCKING_SYNC) failed: %s "
             "- primary context already active under another sync mode; "
             "continuing without it (degrades gracefully, does not abort)",
             s ? s : "?");
    }
    CheckCu(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    // Unlike cuCtxCreate (which pushes/makes its new context current on the
    // calling thread automatically), a bare retain does not - make it
    // current explicitly so every CUDA call below (and the rest of this
    // constructor's thread) targets it, same as every other thread in this
    // file already does via its own cuCtxSetCurrent(ctx) call.
    CheckCu(cuCtxSetCurrent(ctx), "cuCtxSetCurrent (Setup)");

    engine.reset(new TrtEngine(plan.engine_path));
    Logf("\xe2\x9c\x93 Engine loaded: %s (max batch %d)", plan.engine_path.c_str(),
         engine->MaxBatch());
    if (n_streams > engine->MaxBatch()) {
        // Not fatal: a batch simply never exceeds MaxBatch slots, so more
        // streams than that just means no single batch can hold one frame
        // from every stream at once. Fine at reduced rates (skip>1 or
        // --decode key, where batches stay small); at every-frame full
        // rate it caps throughput - warn either way.
        Logf("warning: %d streams > engine max batch %d - batches chunk at "
             "%d; fine at reduced inference rates, caps throughput at "
             "every-frame full rate",
             n_streams, engine->MaxBatch(), engine->MaxBatch());
    }
    const nvinfer1::Dims& od = engine->OutputDims();
    // M4a: layer-0 engine I/O-shape check branches on plan.l0_family - the
    // yolo-e2e head is UNTRANSPOSED ([N,dets,6], row-major per detection)
    // vs. yolo's transposed [N,4+classes,anchors] (channel-major); see
    // postprocess.h's LaunchYoloE2EBatched WHY-comment for the verified
    // layout + evidence.
    if (plan.l0_family == Family::YoloE2E) {
        if (od.nbDims != 3 || od.d[2] != 6) {
            throw std::runtime_error(
                "yolo-e2e engine: want 3D output [N,dets,6] "
                "(x1,y1,x2,y2,score,cls)");
        }
        max_dets = od.d[1];
        // GpuLoop writes yolo-e2e survivors directly into d_kept/d_kept_
        // counts at the kMaxNmsCandidates stride (no NMS stage - see
        // LaunchYoloE2EBatched) - that only fits if every row can survive
        // without overflowing the slot, i.e. max_dets <= kMaxNmsCandidates.
        // yolo26n's 300 comfortably clears the 1024 budget; named error
        // rather than a silent overflow/truncation if some future e2e head
        // ships with more rows.
        if (max_dets > kMaxNmsCandidates) {
            throw std::runtime_error(
                "yolo-e2e engine: " + std::to_string(max_dets) +
                " dets/image > kMaxNmsCandidates (" +
                std::to_string(kMaxNmsCandidates) +
                ") - the no-NMS kept-list reuse (see postprocess.h) "
                "assumes max_dets fits the existing NMS candidate budget");
        }
        Logf("\xe2\x9c\x93 Head (yolo-e2e): %d dets/image", max_dets);
    } else {
        if (od.nbDims != 3 || od.d[1] <= 4) {
            throw std::runtime_error(
                "unexpected output shape (want [N,4+nc,anchors])");
        }
        classes = od.d[1] - 4;
        anchors = od.d[2];
        Logf("\xe2\x9c\x93 Head: %d classes, %d anchors", classes, anchors);
    }
    in_stride = engine->InputCount();
    mb = engine->MaxBatch();

    // Ring input buffers: slot 0 = the engine's own binding, the rest are
    // ours; SetInputAddress retargets the engine per batch.
    batcher.Init(n_streams, mb);
    batcher.buf[0].base = engine->InputPtr();
    for (int i = 1; i < kNumBufs; i++) {
        float* p = nullptr;
        cudaMalloc(&p, (size_t)mb * in_stride * sizeof(float));
        extra_inputs.push_back(p);
        batcher.buf[i].base = p;
    }
    for (auto& b : batcher.buf) {
        b.params.resize(mb);
        b.meta.resize(mb);
    }

    // Postprocess buffers, slot-strided at max batch.
    cudaMalloc(&d_params, mb * sizeof(PostprocImageParams));
    cudaMalloc(&d_cands, (size_t)mb * anchors * sizeof(GpuDetection));
    cudaMalloc(&d_kept, (size_t)mb * kMaxNmsCandidates * sizeof(GpuDetection));
    cudaMalloc(&d_counts, mb * sizeof(int));
    cudaMalloc(&d_kept_counts, mb * sizeof(int));
    h_kept.resize((size_t)mb * kMaxNmsCandidates);
    h_kept_counts.resize(mb);
    if (cfg.verify) h_raw.resize((size_t)mb * engine->OutputCount());

    // ---- Stage-2 cascade: N sibling children, each its own engine + crop
    // scratch (M4b - see ChildScratch/ChildPlan above). ----
    children.clear();
    for (const ChildPlan& cp : plan.children) {
        ChildScratch cs;
        cs.engine.reset(new TrtEngine(cp.engine_path));
        const nvinfer1::Dims& id = cs.engine->InputDims();
        const nvinfer1::Dims& ood = cs.engine->OutputDims();
        if (id.nbDims != 4) {
            throw std::runtime_error("cascade child '" + cp.name +
                                     "' engine: want 4D input [N,C,H,W]");
        }
        cs.h = id.d[2];
        cs.w = id.d[3];
        if (cp.family == Family::Argmax) {
            // M1a: a plain classifier head - 2D [N,classes], or
            // [N,classes,1,1,...] with every trailing dim squeezed to 1
            // (some exporters keep singleton spatial dims on a classifier
            // head) - no CTC time axis.
            if (ood.nbDims < 2) {
                throw std::runtime_error(
                    "cascade child '" + cp.name + "' (argmax) engine: want "
                    "2D output [N,classes] (or [N,classes,1,1] with "
                    "trailing dims == 1)");
            }
            for (int i = 2; i < ood.nbDims; i++) {
                if (ood.d[i] != 1) {
                    throw std::runtime_error(
                        "cascade child '" + cp.name + "' (argmax) engine: "
                        "want 2D output [N,classes] - trailing dim " +
                        std::to_string(i) + " is " + std::to_string(ood.d[i]) +
                        ", want 1 (squeezable)");
                }
            }
            cs.classes = ood.d[1];
            cs.steps = 1;  // no CTC time axis; h_logits/d_crops sizing
                           // below is generic (per-item OutputCount()),
                           // unaffected by this.
            Logf("\xe2\x9c\x93 Classifier (argmax) child '%s': %s (max "
                 "batch %d, in %dx%d, %d classes)",
                 cp.name.c_str(), cp.engine_path.c_str(),
                 cs.engine->MaxBatch(), cs.w, cs.h, cs.classes);
            cudaMalloc(&cs.d_argmax_labels, cs.engine->MaxBatch() * sizeof(int));
            cudaMalloc(&cs.d_argmax_scores, cs.engine->MaxBatch() * sizeof(float));
            cs.h_argmax_labels.resize(cs.engine->MaxBatch());
            cs.h_argmax_scores.resize(cs.engine->MaxBatch());
        } else {
            if (ood.nbDims != 3) {
                throw std::runtime_error(
                    "cascade child '" + cp.name + "' (ctc) engine: want 3D "
                    "output [N,classes,steps]");
            }
            cs.classes = ood.d[1];
            cs.steps = ood.d[2];
            Logf("\xe2\x9c\x93 OCR (ctc) child '%s': %s (max batch %d, in "
                 "%dx%d, %d classes x %d steps)",
                 cp.name.c_str(), cp.engine_path.c_str(),
                 cs.engine->MaxBatch(), cs.w, cs.h, cs.classes, cs.steps);
        }
        cudaMalloc(&cs.d_crops, cs.engine->MaxBatch() * sizeof(CropParams));
        cs.h_logits.resize((size_t)cs.engine->MaxBatch() * cs.engine->OutputCount());
        children.push_back(std::move(cs));
    }
    // M4b back-compat (see result.h's WHY-comment): resolve the FIRST
    // Ctc/Argmax child once, not per frame.
    for (size_t i = 0; i < plan.children.size(); i++) {
        if (plan.children[i].family == Family::Ctc && first_ctc_child < 0)
            first_ctc_child = (int)i;
        if (plan.children[i].family == Family::Argmax && first_argmax_child < 0)
            first_argmax_child = (int)i;
    }

    // ---- SAHI: tile scratch. Tiles are cropped from the ring frames into
    // their own input buffer (chunked by the engine's max batch) and their
    // detections merged with the whole-frame pass.
    if (plan.sahi) {
        cudaMalloc(&d_sahi_in, (size_t)mb * in_stride * sizeof(float));
        cudaMalloc(&d_sahi_crops, mb * sizeof(CropParams));
        cudaMalloc(&d_merge_c, kMaxNmsCandidates * sizeof(GpuDetection));
        cudaMalloc(&d_merge_k, kMaxNmsCandidates * sizeof(GpuDetection));
        cudaMalloc(&d_merge_n, sizeof(int));
        cudaMalloc(&d_merge_kn, sizeof(int));
        h_tile_kept.resize((size_t)mb * kMaxNmsCandidates);
        h_tile_kept_counts.resize(mb);
    }

    cudaStreamCreate(&gpu_stream);

    // Warm the engine at EVERY batch size in its range before any frame
    // exists: the first inference at a cold context pays lazy module load
    // / autotune (hundreds of ms) and would otherwise queue up the opening
    // seconds of every stream. Originally just the two ends (1 and mb) -
    // widened to the full range for the primary-context migration: a
    // dynamic-shape engine's first-ever inference AT A GIVEN BATCH SIZE
    // pays a one-time, roughly tens-of-MB VRAM cost (observed empirically;
    // exact cause not confirmed - TensorRT/cuBLAS/cuDNN algorithm-search
    // or workspace caching are the likely candidates) that used to vanish
    // for free every cycle when Teardown() fully destroyed a PRIVATE
    // context. On the now-shared PRIMARY context that cost instead sticks
    // for the rest of the PROCESS's life the first time any Pipeline ever
    // uses that shape - which real-time batch arrival jitter could
    // otherwise defer to an unpredictable LATER cycle, making that cycle's
    // measured VRAM growth look like a leak. Paying every shape's cost
    // here, inside Setup() (i.e. before Start(), before this cycle's own
    // "after" VRAM snapshot), makes it deterministic instead: exercised
    // once per Pipeline construction rather than left to chance.
    for (int wb = 1; wb <= mb; wb++) {
        engine->SetBatch(wb);
        engine->Infer(gpu_stream);
    }
    for (ChildScratch& cs : children) {
        for (int wb = 1; wb <= cs.engine->MaxBatch(); wb++) {
            cs.engine->SetBatch(wb);
            cs.engine->Infer(gpu_stream);
        }
    }
    cudaStreamSynchronize(gpu_stream);
    engine->SetBatch(1);

    ctx_lock = NvDecoder::CreateContextLock(ctx);

    // Owned here, not by the producers: destroyed only after the GPU
    // thread joins, since queued batches reference ring memory. Built with
    // emplace_back (one at a time, see rings' deque WHY-comment above),
    // not a vector-fill constructor - each ring gets cfg.ring_depth slots
    // (Validate() has already enforced >= 2).
    rings.clear();
    for (int i = 0; i < n_streams; i++) rings.emplace_back(cfg.ring_depth);

    // Phase A1: one PacketRing per stream (always built - see its own
    // member WHY-comment for why it's cheap when unused), then the sink
    // objects cfg.sinks describes. Target grammar was already checked in
    // Validate(); this is where a scheme-valid-but-malformed target (e.g.
    // "tcp://noport") is caught - still at construction time (Setup() runs
    // inside Impl's ctor), just one level deeper than the broad grammar
    // check.
    packet_rings.clear();
    for (int i = 0; i < n_streams; i++) packet_rings.emplace_back(cfg.ring_seconds);

    events_sinks.clear();
    relay_sinks.clear();
    for (const SinkDesc& sd : cfg.sinks) {
        if (sd.kind == SinkKind::Events) {
            auto sink = std::make_unique<EventsSink>();
            sink->desc = sd;
            if (sd.target == "stdout") {
                sink->kind = EventsSink::Kind::Stdout;
            } else if (sd.target.rfind("file://", 0) == 0) {
                sink->kind = EventsSink::Kind::File;
                sink->path = sd.target.substr(7);  // "file://" + "/abs/path"
            } else {  // "tcp://host:port" - Validate() already enforced the
                      // scheme; this parses host/port and fails fast on a
                      // missing/unparsable port.
                sink->kind = EventsSink::Kind::Tcp;
                const std::string hp = sd.target.substr(6);  // strip "tcp://"
                const size_t colon = hp.rfind(':');
                if (colon == std::string::npos || colon + 1 >= hp.size()) {
                    throw std::runtime_error("sink target '" + sd.target +
                                             "' - want tcp://host:port");
                }
                sink->host = hp.substr(0, colon);
                sink->port = std::atoi(hp.substr(colon + 1).c_str());
                if (sink->port <= 0) {
                    throw std::runtime_error("sink target '" + sd.target +
                                             "' - bad port");
                }
            }
            events_sinks.push_back(std::move(sink));
        } else {  // SinkKind::StreamRelay
            auto sink = std::make_unique<RelaySink>();
            sink->desc = sd;
            relay_sinks.push_back(std::move(sink));
        }
    }
}

void Pipeline::Impl::Teardown() {
    if (!ctx) return;  // Setup() never completed (ctor threw)
    cuCtxSetCurrent(ctx);
    if (ctx_lock) NvDecoder::DestroyContextLock(ctx_lock);
    for (float* p : extra_inputs) cudaFree(p);
    cudaFree(d_params);
    cudaFree(d_cands);
    cudaFree(d_kept);
    cudaFree(d_counts);
    cudaFree(d_kept_counts);
    // M4b: per-child scratch (see ChildScratch) - each child owns its own
    // d_crops/d_argmax_labels/d_argmax_scores device buffers and TrtEngine.
    for (ChildScratch& cs : children) {
        cudaFree(cs.d_crops);
        cudaFree(cs.d_argmax_labels);
        cudaFree(cs.d_argmax_scores);
        cs.engine.reset();
    }
    children.clear();
    cudaFree(d_sahi_in);
    cudaFree(d_sahi_crops);
    cudaFree(d_merge_c);
    cudaFree(d_merge_k);
    cudaFree(d_merge_n);
    cudaFree(d_merge_kn);
    if (gpu_stream) cudaStreamDestroy(gpu_stream);
    // Phase A1: sink threads are joined inside StopImpl() (called before
    // Teardown() ever runs - see Impl::~Impl), so nothing is reading
    // packet_rings by this point - host-only memory, no CUDA/ctx ordering
    // constraint like `rings` below, just cleared for symmetry/cleanliness.
    packet_rings.clear();
    rings.clear();   // Nv12Ring dtors: cudaFree the ring buffers
    engine.reset();  // main TrtEngine dtor - LAST device user before the
                     // context goes away
    // Explicit full-device sync before letting go of the context: a
    // PRIVATE context's cuCtxDestroy (the old code) tore down and reclaimed
    // literally everything at once, so nothing outstanding could ever
    // survive it. A retained PRIMARY context has no equivalent moment - it
    // is refcounted and, in every build/test observed, some other retainer
    // (the CUDA runtime API itself lazily retains a device's primary
    // context and does not let go of it mid-process) keeps the underlying
    // context alive straight through our Release below, so nothing here
    // actually gets "destroyed" the way it used to. Measured effect
    // without this line: repeated construct/destroy cycles of a Pipeline
    // in one process (pipeline_api_test) grew VRAM by ~15-20 MB/cycle -
    // consistent with NVDEC decoder-session teardown (cuvidDestroyDecoder,
    // invoked once per stream per cycle, more often across reconnects)
    // reclaiming its surface-pool memory back to the SHARED arena
    // asynchronously rather than immediately, something a private
    // context's teardown used to force implicitly. Synchronizing here
    // drains that before Teardown() is considered complete; confirmed to
    // fully eliminate the growth (see the ctx-migration report).
    cuCtxSynchronize();
    // Release, not destroy: this is the PRIMARY context (see Setup()'s
    // WHY-comment), which is refcounted process-wide rather than owned
    // outright by this Pipeline. cuDevicePrimaryCtxRelease only tears the
    // context down once every retainer (including any this process's
    // CUDA-runtime-API use, torch, etc. holds) has released it - a plain
    // cuCtxDestroy would be wrong here (that call is for contexts created
    // via cuCtxCreate, not retained primary ones).
    cuDevicePrimaryCtxRelease(dev);
    ctx = nullptr;
}

// ---- Producer -----------------------------------------------------------
// Moved from rtsp_infer_multi.cpp's free function `Producer`, adjusted for
// member-variable plumbing and the stop flag. Every WHY-comment is kept.

void Pipeline::Impl::ProducerLoop(int id) {
    cuCtxSetCurrent(ctx);
    const std::string& url = cfg.streams[id].url;
    const int max_frames = cfg.max_frames;
    // Resolved once in the ctor (inheriting cfg.skip/cfg.key_only where the
    // stream doesn't override) - see resolved_streams' WHY-comment.
    const int skip = resolved_streams[id].skip;
    const bool key_only = resolved_streams[id].key_only;
    StreamAtomic& atomics = *stream_atomics[id];
    Nv12Ring& ring = rings[id];  // owned by Impl: must outlive this
                                 // producer (last frames can still sit in
                                 // unprocessed batch buffers at exit)

    // Highest-priority stream: the 0.024 ms preprocess kernel must not queue
    // behind a ~17 ms inference batch, or producers stall and the batcher
    // starves (measured 19 ms pop->ready at 16 streams without this).
    int prio_lo, prio_hi;
    cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, prio_hi);
    cudaEvent_t preprocess_done;
    cudaEventCreate(&preprocess_done);

    // Churn handling: a live camera drops and comes back. On any stream
    // error (open failure, mid-run disconnect, decode error) the demuxer
    // and decoder are torn down and re-opened with exponential backoff.
    // The producer gives up only after kGiveUpMs of CONSECUTIVE failure
    // (no successfully decoded frame), so a flapping stream keeps its
    // slot but a permanently dead one ends the run cleanly.
    constexpr int kGiveUpMs = 30000;
    constexpr int kBackoffMaxMs = 5000;
    int backoff_ms = 250;
    auto last_success = Clock::now();

    int inferred = 0;
    int decoded = 0;  // every frame is decoded (H.264 inter-frame deps);
                      // max_frames counts these so run length is
                      // comparable across skip factors
    bool opened_once = false;

    // Part 1 (Pipeline::Stop): max_frames == 0 means "run until Stop()"
    // (the CLI's --frames N still works exactly as before: this is `false
    // || decoded < max_frames` when N > 0). `stop` is the same atomic
    // Pipeline::Stop() sets right before it flips Batcher::stopping, so a
    // producer parked in batcher.Acquire() and one still spinning through
    // this predicate observe shutdown together.
    //
    // Part 2 audit: all three call sites below (the outer reconnect loop,
    // the demux loop, and the PopFrame loop) already gate on this single
    // `wanted()` predicate rather than each re-testing `decoded <
    // max_frames` separately (that was the pre-Part-1 bug shape - three
    // places to keep in sync, one of which could miss the `max_frames == 0`
    // case and spin/exit wrong). Unbounded runs (max_frames == 0) fall
    // through the `||` short-circuit at every one of them.
    auto wanted = [&] {
        return !stop.load(std::memory_order_relaxed) &&
               (max_frames == 0 || decoded < max_frames);
    };

    while (wanted()) {
    try {
        FFmpegDemuxer demuxer(url);
        NvDecoder decoder(ctx_lock,
                          demuxer.GetCodecID() == AV_CODEC_ID_HEVC
                              ? cudaVideoCodec_HEVC
                              : cudaVideoCodec_H264);
        if (opened_once) {
            atomics.reconnects++;
            Logf("stream %d: reconnected (%s)", id, url.c_str());
        }
        opened_once = true;
        // Phase A1: (re)configure this stream's PacketRing with the fresh
        // demuxer's codec parameters - every reconnect counts as a
        // "(re)configure" (see PacketRing::SetCodecParameters), so a churny
        // camera that comes back at a different resolution/profile is
        // picked up automatically rather than serving StreamRelay/
        // ExtractClip stale params.
        if (cfg.ring_seconds > 0)
            packet_rings[id].SetCodecParameters(demuxer.GetCodecParameters());
        uint8_t* data;
        int size;
        int64_t pts_us = -1;
        bool is_key = false;
        bool stopped_mid = false;
        while (wanted() && demuxer.Demux(&data, &size, &pts_us, &is_key)) {
            // Phase A1: retain a copy of EVERY demuxed video packet, BEFORE
            // any skip/key_only filtering below - sinks (StreamRelay,
            // ExtractClip) see the ORIGINAL stream regardless of whatever
            // inference-side decode/skip policy this pipeline is running,
            // that's the whole point (see the file's top WHY-comment: the
            // packet is already in host RAM here, on its way to NVDEC, so
            // retaining it costs one memcpy and no re-encode/D2H).
            // `data`/`size` are only valid until Demux()'s NEXT call -
            // PacketRing::Push copies immediately, before this loop can ever
            // advance past this point.
            if (cfg.ring_seconds > 0) packet_rings[id].Push(data, size, pts_us, is_key);

            // Decode skipping (keyframe-only mode): drop every non-key
            // packet BEFORE the decoder. A P-frame is an edit of prior
            // frames and cannot be decoded alone, but an IDR keyframe is
            // self-contained — feeding only IDRs is a legal H.264 stream
            // and cuts NVDEC load to one frame per GOP per stream. The
            // update rate becomes the CAMERA's keyframe interval.
            if (key_only && !is_key) continue;
            decoder.Decode(data, size, pts_us);
            DecodedFrame frame;
            // Count check BEFORE PopFrame: popping first would exit the loop
            // holding a mapped surface that never gets released.
            while (wanted() && decoder.PopFrame(&frame)) {
                last_success = Clock::now();
                backoff_ms = 250;  // healthy again: reset the backoff
                decoded++;
                // Frame skipping: run inference on every skip-th decoded
                // frame, phase-offset by stream id so the streams don't all
                // detect on the same tick (which would spike batch sizes).
                // The surface is already mapped by the display callback, so
                // a skipped frame still pays map+unmap (ctx_lock) - only
                // preprocess/inference/postprocess are saved. That is
                // deliberate: it isolates the GPU-cycle hypothesis for the
                // 16-stream latency while holding ctx_lock load constant.
                if ((decoded + id) % skip != 0) {
                    decoder.ReleaseFrame(frame);
                    continue;
                }
                // Ring slot BEFORE batch slot - the deadlock-avoidance
                // ordering. A producer blocked inside AcquireSlot must not
                // be holding an un-ready batch slot: if it did, this could
                // happen — producer P1 waits on ring slot S (still in_use,
                // owned by a frame sitting in buffer X), while the GPU
                // thread's straggler wait in Take() blocks on buffer X's
                // last un-ready slot, which happens to be held by P1.
                // Nobody makes progress. Acquiring the ring first also
                // moves Ensure()'s cudaMallocPitch (the only throwing call
                // in this whole stretch) OUT of the Acquire->MarkReady
                // window, so a mid-run resolution change can't leak a
                // batch slot (see the catch below, which now only has to
                // handle the truly unexpected).
                ring.Ensure(frame.width, frame.height);
                const int rs = ring.AcquireSlot(inferred);
                uint8_t* ring_dst = ring.Slot(rs);
                // t_pop after the ring wait: queue-time accounting
                // (pop->ready) should reflect batcher queueing, not time
                // spent waiting for a ring slot to free up.
                const auto t_pop = Clock::now();
                int slot;
                BatchBuffer* b = batcher.Acquire(&slot);
                if (!b) {
                    // Pipeline::Stop() mid-flight: Batcher::stopping is
                    // set. Treat exactly like completion - release the
                    // ring slot we just took (unused) and the still-mapped
                    // decoder surface, then fall out through `wanted()`
                    // (already false, since `stop` flips before
                    // Batcher::stopping does) at every loop level below.
                    ring.ReleaseSlot(rs);
                    decoder.ReleaseFrame(frame);
                    stopped_mid = true;
                    break;
                }
                try {
                    // M1a/M3a: layer-0 engine's resolved per-channel
                    // norm/color (see ExecPlan::l0_norm_offset[3]/
                    // l0_norm_scale[3]/l0_rgb) - defaults preserve today's
                    // hardcoded YOLO behavior.
                    const LetterboxInfo lb = LaunchNV12ToTensor(
                        frame.device_ptr, frame.pitch, frame.width,
                        frame.height, b->base + slot * in_stride, kNetW,
                        kNetH, stream, ToFloat3(plan.l0_norm_offset),
                        ToFloat3(plan.l0_norm_scale), plan.l0_rgb);
                    // Full-res copy for stage-2 crops, on the same stream:
                    // the event below then covers kernel + copy before
                    // release.
                    cudaMemcpy2DAsync(ring_dst, ring.pitch,
                                      (const void*)frame.device_ptr,
                                      frame.pitch, frame.width,
                                      (size_t)frame.height * 3 / 2,
                                      cudaMemcpyDeviceToDevice, stream);
                    inferred++;
                    cudaEventRecord(preprocess_done, stream);
                    // Surface needed only until the kernel has read it -
                    // the single-stream early-release property, kept per
                    // producer. The ring copy above reads frame.device_ptr
                    // too, so this event (kernel + copy, same stream) must
                    // still be the release gate - do not move it earlier.
                    cudaEventSynchronize(preprocess_done);
                    decoder.ReleaseFrame(frame);

                    b->params[slot] = {lb, frame.width, frame.height};
                    b->meta[slot] = {id,       decoded, frame.timestamp,
                                     t_pop,    Clock::now(), ring_dst,
                                     ring.pitch, &ring, rs};
                    batcher.MarkReady(b);
                } catch (...) {
                    // Belt-and-braces beyond the reorder above: anything
                    // thrown here would otherwise leave this slot
                    // allocated-but-never-ready, and the GPU thread's
                    // straggler wait in Take() (ready == allocated) would
                    // hang forever. Abandon the slot instead of leaking
                    // it: reset to defaults (stream_id=-1, nv12=nullptr) so
                    // the GPU thread's stats loop and the SAHI/OCR paths
                    // skip it, release the ring slot since the copy above
                    // may not have happened, then MarkReady so the buffer
                    // still completes - the engine will infer this slot's
                    // stale input tensor, which is accepted (rare error
                    // path, result is discarded). Rethrow so the existing
                    // churn/retry path handles the actual failure.
                    b->params[slot] = {};
                    b->meta[slot] = SlotMeta{};
                    ring.ReleaseSlot(rs);
                    batcher.MarkReady(b);
                    throw;
                }
            }
            if (stopped_mid) break;
        }
        // Demux returned false with frames still wanted: on a live RTSP
        // source that is a disconnect (farm streams never end cleanly).
        // Route it through the same retry path as an exception. Skipped
        // entirely on a deliberate Stop() (wanted() already false).
        if (wanted())
            throw std::runtime_error("demux ended (stream disconnected)");
    } catch (const std::exception& e) {
        const int silent_ms =
            (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - last_success)
                .count();
        if (silent_ms > kGiveUpMs) {
            Logf("stream %d (%s): %s - no frames for %d ms, giving up", id,
                 url.c_str(), e.what(), silent_ms);
            atomics.failed = true;
            break;
        }
        Logf("stream %d (%s): %s - retrying in %d ms", id, url.c_str(),
             e.what(), backoff_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        backoff_ms = std::min(backoff_ms * 2, kBackoffMaxMs);
        continue;  // tear down demuxer+decoder (scope) and re-open
    }
    break;  // decoded >= max_frames, or Stop(): clean completion
    }      // reconnect loop
    atomics.decoded = decoded;
    cudaEventDestroy(preprocess_done);
    cudaStreamDestroy(stream);
    batcher.ProducerDone();
}

// ---- GPU thread -----------------------------------------------------------
// Moved from rtsp_infer_multi.cpp's main(): greedy batch -> infer -> batched
// GPU postprocess -> optional SAHI merge -> optional OCR cascade -> per-slot
// FrameResult, pushed to the result queue. Per-frame prints and run-level
// aggregation are GONE from here (a caller rebuilds them from FrameResults);
// the informational prints (sahi grid line) still go through Log/Logf.

void Pipeline::Impl::GpuLoop() {
    cuCtxSetCurrent(ctx);
    int batch_seq = 0;  // one per Take(): stamps every FrameResult of a
                        // batch so consumers can count TRUE batches and
                        // build a per-batch (not per-frame) histogram.
    while (BatchBuffer* b = batcher.Take()) {
        const auto t_take = Clock::now();
        const int n = b->allocated;
        engine->SetInputAddress(b->base);
        engine->SetBatch(n);
        cudaMemcpyAsync(d_params, b->params.data(),
                        n * sizeof(PostprocImageParams),
                        cudaMemcpyHostToDevice, gpu_stream);
        engine->Infer(gpu_stream);
        // M4a: layer-0 decode branches on plan.l0_family - yolo-e2e's
        // single launcher writes directly into d_kept/d_kept_counts (no
        // NMS stage; see postprocess.h's LaunchYoloE2EBatched WHY-comment),
        // replacing the BoxDecodeBatched+NmsBatched pair below. Everything
        // downstream (h_kept/h_kept_counts layout, cascade, sinks, results)
        // is unchanged either way.
        if (plan.l0_family == Family::YoloE2E) {
            LaunchYoloE2EBatched(engine->OutputPtr(), n, max_dets, d_params,
                                plan.score_thresh, d_kept, d_kept_counts,
                                gpu_stream);
        } else {
            LaunchBoxDecodeBatched(engine->OutputPtr(), n, anchors, classes,
                                   d_params, plan.score_thresh, d_cands,
                                   d_counts, gpu_stream);
            LaunchNmsBatched(d_cands, d_counts, n, anchors, plan.iou_thresh,
                             d_kept, d_kept_counts, gpu_stream);
        }
        cudaMemcpyAsync(h_kept_counts.data(), d_kept_counts,
                        n * sizeof(int), cudaMemcpyDeviceToHost, gpu_stream);
        cudaMemcpyAsync(h_kept.data(), d_kept,
                        (size_t)n * kMaxNmsCandidates * sizeof(GpuDetection),
                        cudaMemcpyDeviceToHost, gpu_stream);
        if (cfg.verify) {
            cudaMemcpyAsync(h_raw.data(), engine->OutputPtr(),
                            (size_t)n * engine->OutputCount() * sizeof(float),
                            cudaMemcpyDeviceToHost, gpu_stream);
        }
        cudaStreamSynchronize(gpu_stream);
        // ---- SAHI: per slot, slice the full-res ring frame into
        // overlapping tiles, infer them through the same engine, and merge
        // tile detections (already in FRAME coords via the params offsets)
        // with the slot's whole-frame results above. h_kept/h_kept_counts
        // are updated in place, so stage 2 and everything downstream are
        // oblivious to SAHI.
        if (plan.sahi && plan.sahi_serial) {
            for (int s = 0; s < n; s++) {
                const SlotMeta& m = b->meta[s];
                if (!m.nv12) continue;
                const int sw = b->params[s].src_w;
                const int sh = b->params[s].src_h;
                auto key = std::make_pair(sw, sh);
                auto it = sahi_grids.find(key);
                if (it == sahi_grids.end()) {
                    it = sahi_grids
                             .emplace(key, MakeTileGrid(sw, sh, plan.sahi_tile,
                                                        plan.sahi_overlap))
                             .first;
                    Logf("sahi: %dx%d -> %zu tiles (tile %d, overlap "
                         "%.2f)%s => T=%zu x inference",
                         sw, sh, it->second.size(), plan.sahi_tile,
                         plan.sahi_overlap,
                         plan.sahi_full_frame ? " + full-frame pass" : "",
                         it->second.size() + (plan.sahi_full_frame ? 1 : 0));
                }
                const std::vector<TileRect>& grid = it->second;

                // Merge candidates start with the whole-frame pass (the
                // slot's existing detections) unless disabled.
                std::vector<GpuDetection> merge_in;
                if (plan.sahi_full_frame) {
                    const GpuDetection* base =
                        h_kept.data() + (size_t)s * kMaxNmsCandidates;
                    merge_in.assign(base, base + h_kept_counts[s]);
                }

                // Tile passes, chunked by the engine's max batch.
                for (size_t off = 0; off < grid.size(); off += mb) {
                    const int nt = (int)std::min(grid.size() - off, (size_t)mb);
                    std::vector<CropParams> tcrops(nt);
                    std::vector<PostprocImageParams> tparams(nt);
                    for (int t = 0; t < nt; t++) {
                        const TileRect& r = grid[off + t];
                        tcrops[t] = {m.nv12, (int)m.nv12_pitch, sw, sh, r.x,
                                     r.y, r.w, r.h};
                        tparams[t] = {};
                        // tile -> engine input letterbox geometry
                        const float sc = std::min((float)kNetW / r.w,
                                                  (float)kNetH / r.h);
                        tparams[t].lb = LetterboxInfo{
                            sc, (int)((kNetW - r.w * sc) / 2),
                            (int)((kNetH - r.h * sc) / 2)};
                        tparams[t].src_w = r.w;
                        tparams[t].src_h = r.h;
                        tparams[t].off_x = r.x;
                        tparams[t].off_y = r.y;
                    }
                    cudaMemcpyAsync(d_sahi_crops, tcrops.data(),
                                    nt * sizeof(CropParams),
                                    cudaMemcpyHostToDevice, gpu_stream);
                    // NOTE: tiles are square (== engine input) except at
                    // frame edges, where the crop kernel's resize to
                    // kNetW x kNetH matches the letterbox above only if the
                    // tile is square; edge tiles from MakeTileGrid are
                    // always tile_px x tile_px (pulled back flush), or
                    // full-span when the frame is smaller - both
                    // square-or-frame cases the lb math above covers.
                    // M1a/M3a: tiles feed the SAME layer-0 engine as the
                    // whole-frame pass, so they inherit its resolved
                    // per-channel norm/color too (ExecPlan::l0_*) -
                    // defaults preserve today's hardcoded YOLO-tile
                    // behavior.
                    LaunchNv12CropResizeBatched(d_sahi_crops, nt, d_sahi_in,
                                                kNetW, kNetH,
                                                ToFloat3(plan.l0_norm_offset),
                                                ToFloat3(plan.l0_norm_scale),
                                                gpu_stream,
                                                /*rgb=*/plan.l0_rgb);
                    engine->SetInputAddress(d_sahi_in);
                    engine->SetBatch(nt);
                    engine->Infer(gpu_stream);
                    cudaMemcpyAsync(d_params, tparams.data(),
                                    nt * sizeof(PostprocImageParams),
                                    cudaMemcpyHostToDevice, gpu_stream);
                    LaunchBoxDecodeBatched(engine->OutputPtr(), nt, anchors,
                                          classes, d_params, plan.score_thresh,
                                          d_cands, d_counts, gpu_stream);
                    LaunchNmsBatched(d_cands, d_counts, nt, anchors,
                                     plan.iou_thresh, d_kept, d_kept_counts,
                                     gpu_stream);
                    cudaMemcpyAsync(h_tile_kept_counts.data(), d_kept_counts,
                                    nt * sizeof(int), cudaMemcpyDeviceToHost,
                                    gpu_stream);
                    cudaMemcpyAsync(
                        h_tile_kept.data(), d_kept,
                        (size_t)nt * kMaxNmsCandidates * sizeof(GpuDetection),
                        cudaMemcpyDeviceToHost, gpu_stream);
                    cudaStreamSynchronize(gpu_stream);
                    for (int t = 0; t < nt; t++)
                        for (int k = 0; k < h_tile_kept_counts[t]; k++)
                            merge_in.push_back(
                                h_tile_kept[(size_t)t * kMaxNmsCandidates + k]);
                }

                // Cross-tile merge NMS -> this slot's final detections.
                int mc = (int)std::min(merge_in.size(), (size_t)kMaxNmsCandidates);
                cudaMemcpyAsync(d_merge_c, merge_in.data(),
                                mc * sizeof(GpuDetection),
                                cudaMemcpyHostToDevice, gpu_stream);
                cudaMemcpyAsync(d_merge_n, &mc, sizeof(int),
                                cudaMemcpyHostToDevice, gpu_stream);
                LaunchNms(d_merge_c, d_merge_n, plan.sahi_merge_iou, d_merge_k,
                          d_merge_kn, gpu_stream);
                int fn = 0;
                cudaMemcpyAsync(&fn, d_merge_kn, sizeof(int),
                                cudaMemcpyDeviceToHost, gpu_stream);
                cudaStreamSynchronize(gpu_stream);
                cudaMemcpy(h_kept.data() + (size_t)s * kMaxNmsCandidates,
                           d_merge_k, fn * sizeof(GpuDetection),
                           cudaMemcpyDeviceToHost);
                h_kept_counts[s] = fn;
            }
            // restore the engine to the batch buffer for the next take
            engine->SetInputAddress(b->base);
        } else if (plan.sahi) {
            // Pooled path: gather tile jobs from ALL slots in this batch
            // into one list, then run the engine on FULL mb-sized chunks
            // regardless of which slot each tile came from. Each tile
            // carries its own CropParams/PostprocImageParams so its
            // detections still map back to the right slot/frame. Final
            // per-slot detections are identical to the serial path
            // (decode/NMS/merge are per-tile / per-slot independent); this
            // only changes how tiles are grouped into engine calls.
            struct TileJob {
                int slot;
                CropParams crop;
                PostprocImageParams params;
            };
            std::vector<TileJob> jobs;
            std::vector<std::vector<GpuDetection>> merge(n);

            // Phase 1: COLLECT. Seed whole-frame detections and build the
            // pooled tile job list, one pass over all slots.
            for (int s = 0; s < n; s++) {
                const SlotMeta& m = b->meta[s];
                if (!m.nv12) continue;
                const int sw = b->params[s].src_w;
                const int sh = b->params[s].src_h;
                auto key = std::make_pair(sw, sh);
                auto it = sahi_grids.find(key);
                if (it == sahi_grids.end()) {
                    it = sahi_grids
                             .emplace(key, MakeTileGrid(sw, sh, plan.sahi_tile,
                                                        plan.sahi_overlap))
                             .first;
                    Logf("sahi: %dx%d -> %zu tiles (tile %d, overlap "
                         "%.2f)%s => T=%zu x inference",
                         sw, sh, it->second.size(), plan.sahi_tile,
                         plan.sahi_overlap,
                         plan.sahi_full_frame ? " + full-frame pass" : "",
                         it->second.size() + (plan.sahi_full_frame ? 1 : 0));
                }
                const std::vector<TileRect>& grid = it->second;

                if (plan.sahi_full_frame) {
                    const GpuDetection* base =
                        h_kept.data() + (size_t)s * kMaxNmsCandidates;
                    merge[s].assign(base, base + h_kept_counts[s]);
                }

                for (const TileRect& r : grid) {
                    TileJob job;
                    job.slot = s;
                    job.crop = {m.nv12, (int)m.nv12_pitch, sw, sh, r.x, r.y,
                                r.w, r.h};
                    job.params = {};
                    // tile -> engine input letterbox geometry
                    const float sc =
                        std::min((float)kNetW / r.w, (float)kNetH / r.h);
                    job.params.lb = LetterboxInfo{
                        sc, (int)((kNetW - r.w * sc) / 2),
                        (int)((kNetH - r.h * sc) / 2)};
                    job.params.src_w = r.w;
                    job.params.src_h = r.h;
                    job.params.off_x = r.x;
                    job.params.off_y = r.y;
                    jobs.push_back(job);
                }
            }

            // Phase 2: POOL & INFER. Chunk the pooled jobs by the engine's
            // max batch, so cross-slot tiles share full calls.
            for (size_t off = 0; off < jobs.size(); off += mb) {
                const int nt = (int)std::min(jobs.size() - off, (size_t)mb);
                std::vector<CropParams> tcrops(nt);
                std::vector<PostprocImageParams> tparams(nt);
                for (int t = 0; t < nt; t++) {
                    tcrops[t] = jobs[off + t].crop;
                    tparams[t] = jobs[off + t].params;
                }
                cudaMemcpyAsync(d_sahi_crops, tcrops.data(),
                                nt * sizeof(CropParams),
                                cudaMemcpyHostToDevice, gpu_stream);
                // NOTE: tiles are square (== engine input) except at frame
                // edges, where the crop kernel's resize to kNetW x kNetH
                // matches the letterbox above only if the tile is square;
                // edge tiles from MakeTileGrid are always tile_px x tile_px
                // (pulled back flush), or full-span when the frame is
                // smaller - both square-or-frame cases the lb math above
                // covers.
                // M1a/M3a: same layer-0-engine per-channel norm/color as
                // the serial path above (ExecPlan::l0_*) - defaults
                // preserve today's hardcoded YOLO-tile behavior.
                LaunchNv12CropResizeBatched(d_sahi_crops, nt, d_sahi_in, kNetW,
                                            kNetH, ToFloat3(plan.l0_norm_offset),
                                            ToFloat3(plan.l0_norm_scale),
                                            gpu_stream,
                                            /*rgb=*/plan.l0_rgb);
                engine->SetInputAddress(d_sahi_in);
                engine->SetBatch(nt);
                engine->Infer(gpu_stream);
                cudaMemcpyAsync(d_params, tparams.data(),
                                nt * sizeof(PostprocImageParams),
                                cudaMemcpyHostToDevice, gpu_stream);
                LaunchBoxDecodeBatched(engine->OutputPtr(), nt, anchors,
                                      classes, d_params, plan.score_thresh,
                                      d_cands, d_counts, gpu_stream);
                LaunchNmsBatched(d_cands, d_counts, nt, anchors,
                                 plan.iou_thresh, d_kept, d_kept_counts,
                                 gpu_stream);
                cudaMemcpyAsync(h_tile_kept_counts.data(), d_kept_counts,
                                nt * sizeof(int), cudaMemcpyDeviceToHost,
                                gpu_stream);
                cudaMemcpyAsync(
                    h_tile_kept.data(), d_kept,
                    (size_t)nt * kMaxNmsCandidates * sizeof(GpuDetection),
                    cudaMemcpyDeviceToHost, gpu_stream);
                cudaStreamSynchronize(gpu_stream);
                for (int t = 0; t < nt; t++) {
                    const int slot = jobs[off + t].slot;
                    for (int k = 0; k < h_tile_kept_counts[t]; k++)
                        merge[slot].push_back(
                            h_tile_kept[(size_t)t * kMaxNmsCandidates + k]);
                }
            }

            // Phase 3: MERGE. Per slot, identical cross-tile merge NMS as
            // the serial path. Mirrors the Phase 1 skip set so a tile-less
            // slot's whole-frame result is never overwritten.
            for (int s = 0; s < n; s++) {
                const SlotMeta& m = b->meta[s];
                if (!m.nv12) continue;
                std::vector<GpuDetection>& merge_in = merge[s];
                int mc = (int)std::min(merge_in.size(), (size_t)kMaxNmsCandidates);
                cudaMemcpyAsync(d_merge_c, merge_in.data(),
                                mc * sizeof(GpuDetection),
                                cudaMemcpyHostToDevice, gpu_stream);
                cudaMemcpyAsync(d_merge_n, &mc, sizeof(int),
                                cudaMemcpyHostToDevice, gpu_stream);
                LaunchNms(d_merge_c, d_merge_n, plan.sahi_merge_iou, d_merge_k,
                          d_merge_kn, gpu_stream);
                int fn = 0;
                cudaMemcpyAsync(&fn, d_merge_kn, sizeof(int),
                                cudaMemcpyDeviceToHost, gpu_stream);
                cudaStreamSynchronize(gpu_stream);
                cudaMemcpy(h_kept.data() + (size_t)s * kMaxNmsCandidates,
                           d_merge_k, fn * sizeof(GpuDetection),
                           cudaMemcpyDeviceToHost);
                h_kept_counts[s] = fn;
            }
            // restore the engine to the batch buffer for the next take
            engine->SetInputAddress(b->base);
        }
        // ---- Stage 2 (M4b depth-2 tree): detections -> full-res crops
        // from the ring frames -> N SIBLING children, each its own
        // engine/norm/color -> plate strings (Ctc) or (label,score) pairs
        // (Argmax), aligned with each slot's detection list per child (see
        // result.h's FrameResult::children/ChildOutput). Deliberately
        // inside the timed cycle: this is the cost the capacity-knee
        // re-measurement must see.
        //
        // The crop RECT list (source-frame coordinates only - CropParams
        // carries no destination size) is IDENTICAL across every child, so
        // it is built exactly ONCE per batch here and reused by every
        // child's own chunked crop-resize call below - the only thing that
        // differs per child is the destination (that child's engine input
        // W/H), norm/color, and MaxBatch-driven chunk size.
        std::vector<std::vector<std::vector<std::string>>> child_texts(
            children.size());
        std::vector<std::vector<std::vector<int>>> child_labels(children.size());
        std::vector<std::vector<std::vector<float>>> child_label_scores(
            children.size());
        if (!children.empty()) {
            h_crops.clear();
            crop_owner.clear();
            for (int s = 0; s < n; s++) {
                const SlotMeta& m = b->meta[s];
                if (!m.nv12) continue;
                const int sw = b->params[s].src_w;
                const int sh = b->params[s].src_h;
                const GpuDetection* dets =
                    h_kept.data() + (size_t)s * kMaxNmsCandidates;
                for (int k = 0; k < h_kept_counts[s]; k++) {
                    int rx = (int)lroundf(dets[k].x);
                    int ry = (int)lroundf(dets[k].y);
                    int rw = (int)lroundf(dets[k].w);
                    int rh = (int)lroundf(dets[k].h);
                    rx = std::max(0, std::min(rx, sw - 1));
                    ry = std::max(0, std::min(ry, sh - 1));
                    rw = std::max(1, std::min(rw, sw - rx));
                    rh = std::max(1, std::min(rh, sh - ry));
                    h_crops.push_back({m.nv12, (int)m.nv12_pitch, sw, sh, rx,
                                       ry, rw, rh});
                    crop_owner.emplace_back(s, k);
                }
            }

            for (size_t ci = 0; ci < children.size(); ci++) {
                ChildScratch& cs = children[ci];
                const ChildPlan& cp = plan.children[ci];
                const bool is_argmax = cp.family == Family::Argmax;
                std::vector<std::vector<std::string>>& c_texts = child_texts[ci];
                std::vector<std::vector<int>>& c_labels = child_labels[ci];
                std::vector<std::vector<float>>& c_scores =
                    child_label_scores[ci];
                c_texts.assign(n, {});
                c_labels.assign(n, {});
                c_scores.assign(n, {});
                for (int s = 0; s < n; s++) {
                    if (is_argmax) {
                        c_labels[s].resize(h_kept_counts[s]);
                        c_scores[s].resize(h_kept_counts[s]);
                    } else {
                        c_texts[s].resize(h_kept_counts[s]);
                    }
                }
                // Chunked by THIS child's own engine profile; crops land
                // directly in its input binding (kernel slot stride ==
                // InputCount) - children can differ in MaxBatch/input size,
                // so each gets its own chunk loop over the SHARED h_crops.
                const int cmb = cs.engine->MaxBatch();
                for (size_t off = 0; off < h_crops.size(); off += cmb) {
                    const int nc =
                        (int)std::min(h_crops.size() - off, (size_t)cmb);
                    cudaMemcpyAsync(cs.d_crops, h_crops.data() + off,
                                    nc * sizeof(CropParams),
                                    cudaMemcpyHostToDevice, gpu_stream);
                    // M1a/M3a/M4b: this child's resolved per-channel
                    // norm/color (see ChildPlan::norm_offset[3]/
                    // norm_scale[3]/rgb) - defaults preserve today's
                    // hardcoded LPRNet/cv2 (-127.5, 1/128, BGR) behavior; a
                    // classifier child can instead carry true per-channel
                    // ImageNet norm, independently of its siblings.
                    LaunchNv12CropResizeBatched(cs.d_crops, nc,
                                                cs.engine->InputPtr(), cs.w,
                                                cs.h, ToFloat3(cp.norm_offset),
                                                ToFloat3(cp.norm_scale),
                                                gpu_stream,
                                                /*rgb=*/cp.rgb != 0);
                    cs.engine->SetBatch(nc);
                    cs.engine->Infer(gpu_stream);
                    if (is_argmax) {
                        // GPU argmax straight off the engine's output
                        // binding (no per-crop CPU decode loop, unlike Ctc
                        // below - see LaunchArgmaxBatched's WHY-comment in
                        // postprocess.h): D2H only the compact (label,score)
                        // pairs, not the full logits.
                        LaunchArgmaxBatched(cs.engine->OutputPtr(), nc,
                                            cs.classes, cs.d_argmax_labels,
                                            cs.d_argmax_scores, gpu_stream);
                        cudaMemcpyAsync(cs.h_argmax_labels.data(),
                                        cs.d_argmax_labels, nc * sizeof(int),
                                        cudaMemcpyDeviceToHost, gpu_stream);
                        cudaMemcpyAsync(cs.h_argmax_scores.data(),
                                        cs.d_argmax_scores, nc * sizeof(float),
                                        cudaMemcpyDeviceToHost, gpu_stream);
                        cudaStreamSynchronize(gpu_stream);
                        for (int c = 0; c < nc; c++) {
                            const auto& owner = crop_owner[off + c];
                            c_labels[owner.first][owner.second] =
                                cs.h_argmax_labels[c];
                            c_scores[owner.first][owner.second] =
                                cs.h_argmax_scores[c];
                        }
                    } else {
                        cudaMemcpyAsync(cs.h_logits.data(),
                                        cs.engine->OutputPtr(),
                                        (size_t)nc * cs.engine->OutputCount() *
                                            sizeof(float),
                                        cudaMemcpyDeviceToHost, gpu_stream);
                        cudaStreamSynchronize(gpu_stream);
                        for (int c = 0; c < nc; c++) {
                            const auto label = CtcGreedyDecode(
                                cs.h_logits.data() +
                                    (size_t)c * cs.engine->OutputCount(),
                                cs.classes, cs.steps);
                            const auto& owner = crop_owner[off + c];
                            c_texts[owner.first][owner.second] =
                                LprLabelString(label);
                        }
                    }
                }
            }
        }
        const auto t_done = Clock::now();
        const double batch_ms =
            std::chrono::duration<double, std::milli>(t_done - t_take).count();

        for (int s = 0; s < n; s++) {
            SlotMeta& m = b->meta[s];
            // Abandoned slot (producer threw between Acquire and MarkReady
            // - see the producer's catch block): stream_id is reset to -1.
            // The engine still inferred its stale tensor, but that result
            // is simply not attributed to any stream - no FrameResult.
            if (m.stream_id < 0) continue;

            FrameResult fr;
            fr.stream_id = m.stream_id;
            fr.frame_no = m.frame_no;
            fr.pts_us = m.pts_us;
            fr.batch_size = n;
            fr.batch_seq = batch_seq;
            fr.ms_pop_to_ready =
                std::chrono::duration<double, std::milli>(m.t_ready - m.t_pop)
                    .count();
            fr.ms_ready_to_take =
                std::chrono::duration<double, std::milli>(t_take - m.t_ready)
                    .count();
            fr.ms_take_to_done = batch_ms;  // same for every slot of this
                                             // batch - it's a whole-batch
                                             // GPU cycle time, same
                                             // quantity the old code
                                             // averaged per BATCH.

            // Tier 1 (informational, always populated - see result.h):
            // where this frame's full-res NV12 copy lives right now. True
            // for every attributed slot reaching this point (m.nv12 is
            // only ever null on an abandoned slot, already skipped above).
            fr.frame_addr = reinterpret_cast<uint64_t>(m.nv12);
            fr.frame_pitch = (int)m.nv12_pitch;
            fr.frame_width = b->params[s].src_w;
            fr.frame_height = b->params[s].src_h;

            if (cfg.hold_frames && m.ring && m.ring_slot >= 0) {
                // Tier 3: hand this slot's refcount to the FrameResult
                // itself instead of releasing it below. The deleter is
                // exactly Nv12Ring::ReleaseSlot - mutex+cv only, safe from
                // any thread (see Nv12RingHold's WHY-comment). Clearing
                // m.ring/m.ring_slot here (the same fields the pre-Recycle
                // loop below tests) makes that loop's existing
                // `if (m.ring && ...)` guard skip this slot automatically -
                // no separate "was it held" bookkeeping needed.
                Nv12Ring* ring = m.ring;
                const int slot_idx = m.ring_slot;
                fr.frame_hold = std::shared_ptr<void>(
                    new Nv12RingHold{ring, slot_idx, ctx},
                    [](void* p) {
                        Nv12RingHold* h = static_cast<Nv12RingHold*>(p);
                        h->ring->ReleaseSlot(h->slot);
                        delete h;
                    });
                m.ring = nullptr;
                m.ring_slot = -1;
            }

            const GpuDetection* dets =
                h_kept.data() + (size_t)s * kMaxNmsCandidates;
            fr.detections.reserve(h_kept_counts[s]);
            for (int k = 0; k < h_kept_counts[s]; k++) {
                fr.detections.push_back({dets[k].x, dets[k].y, dets[k].w,
                                         dets[k].h, dets[k].score,
                                         dets[k].cls});
            }
            // M4b: one ChildOutput per sibling child (see result.h),
            // populated from exactly one of {texts} / {labels,
            // label_scores} per child family - same "the other stays
            // empty" contract M1a established for the single-child case,
            // now per child instead of once on fr directly.
            //
            // WHY the back-compat copy below: fr.texts/fr.labels/
            // fr.label_scores (the pre-M4b flat fields) are NOT removed -
            // every existing consumer that predates the tree (the CLI's
            // --ocr prints in rtsp_infer_multi.cpp, the Events sink's
            // FormatEventLine NDJSON schema, and qa_matrix's G1/G2/I1c
            // gates) reads them directly and must keep working unchanged.
            // A single-child pipeline (still the common case) has exactly
            // one Ctc-or-Argmax child, so "first child of the matching
            // family" reproduces the old single-child fields byte-for-byte;
            // a multi-child (M4b) pipeline gets a defined, documented
            // choice (first-of-family) instead of an ambiguous one.
            if (!children.empty()) {
                fr.children.reserve(children.size());
                for (size_t ci = 0; ci < children.size(); ci++) {
                    ChildOutput co;
                    co.layer = (int)ci + 1;  // cfg.layers index (0 = detector)
                    const bool is_argmax =
                        plan.children[ci].family == Family::Argmax;
                    if (is_argmax) {
                        co.labels = child_labels[ci][s];
                        co.label_scores = child_label_scores[ci][s];
                        if ((int)ci == first_argmax_child) {
                            fr.labels = co.labels;
                            fr.label_scores = co.label_scores;
                        }
                    } else {
                        co.texts = child_texts[ci][s];
                        if ((int)ci == first_ctc_child) fr.texts = co.texts;
                    }
                    fr.children.push_back(std::move(co));
                }
            }

            // M4a: CpuReference() assumes yolo's channel-major
            // [4+classes,anchors] layout (see its own definition above) -
            // meaningless for yolo-e2e's row-major [dets,6] layout, so
            // --verify is a no-op (fr.verified stays false) rather than
            // comparing against garbage for that family. Not one of this
            // family's required gates (postprocess_batch_test's synthetic
            // checkpoint + qa_matrix's I1 cover it instead) - a from-scratch
            // yolo-e2e CPU reference for --verify is future work if wanted.
            if (cfg.verify && plan.l0_family == Family::YoloDetect) {
                const auto ref = CpuReference(
                    h_raw.data() + (size_t)s * engine->OutputCount(), anchors,
                    classes, b->params[s], plan.score_thresh, plan.iou_thresh);
                bool ok = (int)ref.size() == h_kept_counts[s];
                for (size_t k = 0; ok && k < ref.size(); k++) {
                    ok = ref[k].cls == dets[k].cls &&
                         ref[k].score == dets[k].score &&
                         std::fabs(ref[k].x - dets[k].x) < 0.1f &&
                         std::fabs(ref[k].y - dets[k].y) < 0.1f &&
                         std::fabs(ref[k].w - dets[k].w) < 0.1f &&
                         std::fabs(ref[k].h - dets[k].h) < 0.1f;
                }
                fr.verified = true;
                fr.verify_ok = ok;
                fr.verify_cpu_dets = (int)ref.size();
            }
            // Phase A1: format+dispatch to Events sinks BEFORE the
            // queue.Push(std::move(fr)) below - fr's vectors (detections/
            // texts) are moved-from and unspecified afterward, so this must
            // run first. Formatting cost is small (tens of dets) but this
            // still runs on the GPU thread, so it's skipped entirely (no
            // string built at all) when no Events sink is configured.
            if (!events_sinks.empty()) {
                const std::string line = FormatEventLine(fr);
                for (auto& sink : events_sinks) {
                    if (sink->desc.stream_id != -1 &&
                        sink->desc.stream_id != fr.stream_id)
                        continue;
                    sink->queue.Push(line);
                }
            }
            queue.Push(std::move(fr));
        }
        // Every ring-memory consumer above (whole-frame pass needed no
        // ring copy; SAHI's serial/pooled paths and the OCR cascade each
        // end in a cudaStreamSynchronize before this point) is done
        // reading b->meta[s].nv12 by now - safe to release the ring slots
        // back to their producers before this buffer is recycled. Slots
        // emitted with a tier-3 holder above already had m.ring cleared,
        // so this loop naturally releases ONLY the rest: non-hold slots,
        // and abandoned slots (never reached the fr-building loop at all,
        // so m.ring/m.ring_slot are still whatever the producer set).
        for (int s = 0; s < n; s++) {
            SlotMeta& m = b->meta[s];
            if (m.ring && m.ring_slot >= 0) m.ring->ReleaseSlot(m.ring_slot);
            m.ring = nullptr;  // Recycle() only clears allocated/ready
        }
        batch_seq++;
        batcher.Recycle(b);
    }
    // Drained: join producers here (they're all guaranteed done - Take()
    // only returns nullptr once active_producers reaches 0), then signal
    // Finished so Poll() stops blocking once the queue empties.
    for (auto& t : producers)
        if (t.joinable()) t.join();
    queue.MarkFinished();
}

// ---- Events sink thread ---------------------------------------------------
// Phase A1. Drains one EventsSink's LineQueue to its target (tcp/file/
// stdout), reconnecting with exponential backoff on failure - exactly the
// producer's own churn-handling shape (see ProducerLoop's WHY-comment),
// just for an outbound connection instead of an inbound RTSP one. While
// disconnected, this thread simply isn't popping the queue, so it fills to
// LineQueue's capacity and starts evicting its own oldest entries - "drop
// lines meanwhile, count them" falls out of that for free (see LineQueue's
// WHY-comment), no separate bookkeeping needed here.
void Pipeline::Impl::EventsSinkLoop(EventsSink& sink) {
    int fd = -1;         // valid iff sink.kind == Tcp and connected
    FILE* fp = nullptr;  // valid iff sink.kind == File/Stdout and connected
    int backoff_ms = 250;
    constexpr int kBackoffMaxMs = 5000;

    auto is_connected = [&] {
        return sink.kind == EventsSink::Kind::Tcp ? fd >= 0 : fp != nullptr;
    };
    auto close_conn = [&] {
        if (fd >= 0) { close(fd); fd = -1; }
        if (fp && fp != stdout) { fclose(fp); fp = nullptr; }
    };
    auto try_connect = [&]() -> bool {
        if (sink.kind == EventsSink::Kind::Stdout) { fp = stdout; return true; }
        if (sink.kind == EventsSink::Kind::File) {
            fp = fopen(sink.path.c_str(), "a");
            return fp != nullptr;
        }
        // Tcp: resolve host (dotted IPv4 or hostname) via getaddrinfo, then
        // connect. rtsp_transport-style options don't apply here - this is
        // a plain newline-delimited NDJSON stream, not RTSP.
        char portbuf[16];
        snprintf(portbuf, sizeof(portbuf), "%d", sink.port);
        addrinfo hints{};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        addrinfo* res = nullptr;
        if (getaddrinfo(sink.host.c_str(), portbuf, &hints, &res) != 0 || !res)
            return false;
        fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        const bool ok = fd >= 0 && connect(fd, res->ai_addr, res->ai_addrlen) == 0;
        freeaddrinfo(res);
        if (!ok) { if (fd >= 0) close(fd); fd = -1; return false; }
        return true;
    };

    while (!sink.stopping.load(std::memory_order_relaxed)) {
        if (!is_connected()) {
            if (!try_connect()) {
                InterruptibleSleep(sink.stopping, sink.wait_m, sink.wait_cv,
                                   std::chrono::milliseconds(backoff_ms));
                backoff_ms = std::min(backoff_ms * 2, kBackoffMaxMs);
                continue;
            }
            backoff_ms = 250;
        }
        std::string line;
        if (!sink.queue.Pop(&line, 200)) continue;  // timeout: recheck stopping
        line.push_back('\n');
        const bool ok = sink.kind == EventsSink::Kind::Tcp
                            ? SendAll(fd, line.data(), line.size())
                            : (fwrite(line.data(), 1, line.size(), fp) ==
                                   line.size() &&
                               fflush(fp) == 0);
        if (!ok) {
            Logf("events sink '%s': write failed, reconnecting",
                 sink.desc.target.c_str());
            close_conn();
        }
    }
    close_conn();
}

// ---- StreamRelay sink thread -----------------------------------------------
// Phase A1. Republishes one stream's PacketRing via libavformat's RTSP
// output muxer (-c copy semantics: codec params copied verbatim from the
// ring, no re-encode). Mirrors farm.sh's own publishing incantation
// (`ffmpeg -c copy -f rtsp -rtsp_transport tcp rtsp://...`, confirmed
// working against the mediamtx this repo already uses for its stream farm)
// through the equivalent libavformat C API calls. Reconnect-with-backoff on
// any failure, same shape as ProducerLoop/EventsSinkLoop.
void Pipeline::Impl::RelaySinkLoop(RelaySink& sink) {
    const int sid = sink.desc.stream_id;
    int backoff_ms = 250;
    constexpr int kBackoffMaxMs = 5000;

    while (!sink.stopping.load(std::memory_order_relaxed)) {
        AVFormatContext* oc = nullptr;
        AVStream* st = nullptr;
        try {
            AVCodecParameters* cp = packet_rings[sid].CopyCodecParameters();
            if (!cp)
                throw std::runtime_error(
                    "no codec parameters yet (stream hasn't opened)");
            if (avformat_alloc_output_context2(&oc, nullptr, "rtsp",
                                               sink.desc.target.c_str()) < 0 ||
                !oc) {
                avcodec_parameters_free(&cp);
                throw std::runtime_error("avformat_alloc_output_context2 failed for '" +
                                         sink.desc.target + "'");
            }
            st = avformat_new_stream(oc, nullptr);
            avcodec_parameters_copy(st->codecpar, cp);
            st->codecpar->codec_tag = 0;  // avoid a container-tag mismatch
                                          // between the RTSP source and the
                                          // RTSP-muxer sink
            avcodec_parameters_free(&cp);

            AVDictionary* opts = nullptr;
            av_dict_set(&opts, "rtsp_transport", "tcp", 0);
            const int hret = avformat_write_header(oc, &opts);
            av_dict_free(&opts);
            if (hret < 0) {
                char eb[128];
                av_strerror(hret, eb, sizeof(eb));
                throw std::runtime_error(std::string("avformat_write_header: ") + eb);
            }
            Logf("relay: stream %d -> %s connected", sid,
                 sink.desc.target.c_str());
            backoff_ms = 250;

            // Live-follow loop: SnapshotSince(last_sent_pts) returns only
            // packets not yet sent (see PacketRing::SnapshotSince's
            // WHY-comment) - the FIRST call (last_sent_pts == -1) returns
            // the WHOLE ring, which is always IDR-anchored by construction
            // (see PacketRing::TrimLocked), so "loop: pull next packets from
            // ring (starting at anchor IDR)" falls out with no separate
            // anchor search here. Playback is paced to the SOURCE's real
            // pts deltas (see the sleep below), so a fresh relay connection
            // replays the ring's ~ring_seconds of retained history in real
            // time before catching up to live and continuing there.
            int64_t last_sent_pts = -1;
            int64_t pts_anchor = -1;
            Clock::time_point t_anchor{};
            while (!sink.stopping.load(std::memory_order_relaxed)) {
                std::vector<RingPacket> batch =
                    packet_rings[sid].SnapshotSince(last_sent_pts);
                for (const RingPacket& p : batch) {
                    if (sink.stopping.load(std::memory_order_relaxed)) break;
                    if (pts_anchor < 0) {
                        pts_anchor = p.pts_us;
                        t_anchor = Clock::now();
                    } else if (p.pts_us > pts_anchor) {
                        const auto due =
                            t_anchor + std::chrono::microseconds(p.pts_us - pts_anchor);
                        const auto now = Clock::now();
                        if (due > now)
                            InterruptibleSleep(sink.stopping, sink.wait_m,
                                               sink.wait_cv, due - now);
                    }
                    AVPacket* pkt = av_packet_alloc();
                    av_new_packet(pkt, (int)p.data.size());
                    memcpy(pkt->data, p.data.data(), p.data.size());
                    pkt->stream_index = st->index;
                    pkt->pts = pkt->dts =
                        av_rescale_q(p.pts_us, AVRational{1, 1000000}, st->time_base);
                    if (p.keyframe) pkt->flags |= AV_PKT_FLAG_KEY;
                    const int wret = av_interleaved_write_frame(oc, pkt);
                    av_packet_free(&pkt);
                    if (wret < 0)
                        throw std::runtime_error("av_interleaved_write_frame failed");
                    last_sent_pts = p.pts_us;
                }
                // Poll interval: new packets arrive at roughly the source's
                // frame period (tens of ms) - 20 ms keeps relay latency low
                // without spinning PacketRing's lock.
                InterruptibleSleep(sink.stopping, sink.wait_m, sink.wait_cv,
                                   std::chrono::milliseconds(20));
            }
            av_write_trailer(oc);
        } catch (const std::exception& e) {
            Logf("relay stream %d -> %s: %s%s", sid, sink.desc.target.c_str(),
                 e.what(),
                 sink.stopping.load(std::memory_order_relaxed) ? "" : " - reconnecting");
        }
        if (oc) {
            if (oc->pb && oc->oformat && !(oc->oformat->flags & AVFMT_NOFILE))
                avio_closep(&oc->pb);
            avformat_free_context(oc);
        }
        if (!sink.stopping.load(std::memory_order_relaxed)) {
            InterruptibleSleep(sink.stopping, sink.wait_m, sink.wait_cv,
                               std::chrono::milliseconds(backoff_ms));
            backoff_ms = std::min(backoff_ms * 2, kBackoffMaxMs);
        }
    }
}

void Pipeline::Impl::StopImpl() {
    stop_requested.store(true, std::memory_order_relaxed);  // Start() checks
                                                             // this even if
                                                             // we're the loser
                                                             // below.
    if (stop_seq_started.exchange(true)) {
        // Idempotent AND concurrency-safe: we lost the race to be the one
        // running the sequence below. Block until the winner is done rather
        // than returning immediately - a caller that gets Stop() back is
        // entitled to assume the pipeline is fully quiesced (e.g. the
        // destructor calling StopImpl() then immediately Teardown()'ing
        // CUDA state). Returning early here would let Teardown() free
        // context/engine/buffers while the winner's gpu_thread.join() (and
        // the thread it's joining) are still running - a real
        // use-after-free, not just a benign double-call.
        std::unique_lock<std::mutex> lk(stop_seq_m);
        stop_seq_cv.wait(lk, [&] { return stop_seq_complete; });
        return;
    }
    // (a) Mark the result queue stopping: blocked/future pushes become
    // drops and return immediately, waking any blocked push.
    queue.Stop();
    // (a2) Phase A1: signal every sink to stop too. Sinks are independent of
    // the producer/batcher/GPU pipeline below (they read PacketRing/
    // FrameResult-derived data, not the batcher), so they can be told to
    // stop right away rather than waiting for step (d) - but they're only
    // JOINED in (d), alongside gpu_thread, so Teardown() (which frees
    // packet_rings) never runs while one might still be reading.
    for (auto& s : events_sinks) {
        s->stopping.store(true, std::memory_order_relaxed);
        s->queue.Stop();
        s->wait_cv.notify_all();
    }
    for (auto& s : relay_sinks) {
        s->stopping.store(true, std::memory_order_relaxed);
        s->wait_cv.notify_all();
    }
    // (b) Producer loops' `wanted()` predicate and Batcher::Acquire() both
    // start failing; a producer parked in Acquire() gets nullptr back and
    // exits cleanly (ProducerDone(), same as natural completion).
    stop.store(true, std::memory_order_relaxed);
    batcher.SetStopping();
    // (c) The GPU thread keeps draining pending/filling buffers via its
    // existing Take()-returns-nullptr path, releasing ring slots as it
    // goes, so no producer stays blocked in Nv12Ring::AcquireSlot. NOTE: a
    // producer blocked inside FFmpeg's network read can take up to its 5 s
    // stimeout to notice `wanted()` went false - accepted for v1, hardened
    // in Part 2.
    // (d) Join everything. GpuLoop() itself joins `producers` right before
    // it returns (see above) - by the time we get here that has either
    // already happened or is about to; joining gpu_thread here waits for
    // both.
    if (started.load() && gpu_thread.joinable()) gpu_thread.join();
    // (d2) Phase A1: join every sink thread too - see (a2) above for why
    // this must happen before Teardown() ever runs (it does: Impl::~Impl
    // calls StopImpl() to completion, then Teardown()).
    for (auto& s : events_sinks)
        if (s->thread.joinable()) s->thread.join();
    for (auto& s : relay_sinks)
        if (s->thread.joinable()) s->thread.join();
    // Wake any concurrent Stop() callers that lost the race above.
    {
        std::lock_guard<std::mutex> lk(stop_seq_m);
        stop_seq_complete = true;
    }
    stop_seq_cv.notify_all();
}

// ---------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------

Pipeline::Pipeline(PipelineConfig cfg) : impl_(new Impl(std::move(cfg))) {}

Pipeline::~Pipeline() = default;  // Impl::~Impl() does Stop() + Teardown()

void Pipeline::Start() {
    // Part 2: single-shot lifecycle, both directions. Without this check,
    // Start() called after Stop() (e.g. a careless consumer, or the
    // dtor-only test-matrix cycle racing a stray Stop()) would spawn a fresh
    // producers/gpu_thread generation into an Impl whose stop/queue/batcher
    // flags are already latched "stopping" - those threads exit almost
    // immediately, which sounds harmless, but StopImpl() has ALREADY run
    // (stop_seq_started latched) and will refuse to run its join sequence
    // again. The destructor's plain StopImpl() call would then return
    // without joining this new gpu_thread/producers, and ~thread() on a
    // still-joinable thread calls std::terminate(). Must throw before any
    // thread is spawned.
    if (impl_->stop_requested.load(std::memory_order_relaxed)) {
        throw std::runtime_error(
            "Pipeline::Start called after Stop (single-shot lifecycle)");
    }
    if (impl_->started.exchange(true)) {
        throw std::runtime_error("Pipeline::Start called twice");
    }
    for (int i = 0; i < impl_->n_streams; i++) {
        impl_->producers.emplace_back([this, i] { impl_->ProducerLoop(i); });
    }
    impl_->gpu_thread = std::thread([this] { impl_->GpuLoop(); });
    // Phase A1: sink threads, one per configured EventsSink/RelaySink (built
    // in Setup(), stable addresses - see their container's WHY-comment).
    // Joined in StopImpl().
    for (auto& s : impl_->events_sinks) {
        EventsSink* sp = s.get();
        s->thread = std::thread([this, sp] { impl_->EventsSinkLoop(*sp); });
    }
    for (auto& s : impl_->relay_sinks) {
        RelaySink* sp = s.get();
        s->thread = std::thread([this, sp] { impl_->RelaySinkLoop(*sp); });
    }
}

Pipeline::PollStatus Pipeline::Poll(FrameResult* out, int timeout_ms) {
    return impl_->queue.Poll(out, timeout_ms);
}

void Pipeline::Stop() { impl_->StopImpl(); }

const StreamInfo& Pipeline::GetStreamInfo(int stream_id) const {
    // .at() (not operator[]) on both vectors below: out-of-range stream_id
    // throws std::out_of_range rather than reading past the end.
    Impl::StreamAtomic& sa = *impl_->stream_atomics.at(stream_id);
    StreamInfo& cache = impl_->stream_info_cache.at(stream_id);
    // Part 2: serialize the multi-field write into `cache` - see the
    // stream_info_m WHY-comment in Impl. The atomics themselves are safe to
    // read lock-free; the lock is purely to make the following three-field
    // write (and the caller's read of the returned reference) not race a
    // concurrent GetStreamInfo(stream_id) call from another thread.
    std::lock_guard<std::mutex> lk(*impl_->stream_info_m.at(stream_id));
    cache.decoded = sa.decoded.load();
    cache.reconnects = sa.reconnects.load();
    cache.failed = sa.failed.load();
    return cache;
}

uint64_t Pipeline::DroppedResults() const { return impl_->queue.Dropped(); }

uint64_t Pipeline::SinkDropped() const {
    uint64_t total = 0;
    for (const auto& s : impl_->events_sinks) total += s->queue.Dropped();
    return total;
}

bool Pipeline::ExtractClip(int stream_id, double seconds_back,
                           const std::string& path) const {
    if (stream_id < 0 || stream_id >= impl_->n_streams) {
        impl_->Logf("ExtractClip: stream_id %d out of range", stream_id);
        return false;
    }
    if (impl_->cfg.ring_seconds <= 0) {
        impl_->Log("ExtractClip: ring disabled (PipelineConfig::ring_seconds "
                   "== 0)");
        return false;
    }
    std::vector<RingPacket> snap = impl_->packet_rings[stream_id].Snapshot();
    if (snap.empty()) {
        impl_->Logf("ExtractClip: stream %d's ring is empty", stream_id);
        return false;
    }
    // Anchor: the newest keyframe at-or-before (newest_pts - seconds_back) -
    // see FindAnchor's WHY-comment (packet_ring.h) for the best-effort
    // fallback when seconds_back exceeds what the ring currently holds.
    const int64_t newest = snap.back().pts_us;
    const int64_t cutoff = newest - (int64_t)(seconds_back * 1e6);
    const int anchor = FindAnchor(snap, cutoff);
    if (anchor < 0) {
        impl_->Logf("ExtractClip: stream %d has no keyframe in its ring - "
                    "cannot produce a decodable clip",
                    stream_id);
        return false;
    }

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        impl_->Logf("ExtractClip: could not open '%s' for writing",
                    path.c_str());
        return false;
    }

    // Annex-B vs AVCC: RTSP/RTP H.264 depacketization in FFmpeg already
    // yields Annex-B (start-code-prefixed) access units - VERIFIED for this
    // pipeline's actual sources with ffprobe on an extracted clip (see the
    // A1 self-check report; a plain fwrite of packet bytes was sufficient,
    // no bitstream filter needed). This still guards defensively for a
    // future non-RTSP source whose retained codecpar looks AVCC (a
    // length-prefixed avcC extradata record, recognizable by its first byte
    // being configurationVersion == 1): route packets through the
    // h264_mp4toannexb bitstream filter instead of a raw fwrite in that
    // case.
    AVCodecParameters* cp = impl_->packet_rings[stream_id].CopyCodecParameters();
    AVBSFContext* bsf = nullptr;
    if (cp && cp->extradata && cp->extradata_size >= 1 && cp->extradata[0] == 1) {
        const AVBitStreamFilter* f = av_bsf_get_by_name("h264_mp4toannexb");
        if (f && av_bsf_alloc(f, &bsf) == 0) {
            avcodec_parameters_copy(bsf->par_in, cp);
            av_bsf_init(bsf);
        }
    }
    if (cp) avcodec_parameters_free(&cp);

    bool ok = true;
    for (size_t i = (size_t)anchor; i < snap.size() && ok; i++) {
        const RingPacket& p = snap[i];
        if (bsf) {
            AVPacket* in = av_packet_alloc();
            av_new_packet(in, (int)p.data.size());
            memcpy(in->data, p.data.data(), p.data.size());
            ok = av_bsf_send_packet(bsf, in) >= 0;
            av_packet_free(&in);
            AVPacket* out = av_packet_alloc();
            while (ok && av_bsf_receive_packet(bsf, out) >= 0) {
                if (fwrite(out->data, 1, (size_t)out->size, fp) !=
                    (size_t)out->size)
                    ok = false;
                av_packet_unref(out);
            }
            av_packet_free(&out);
        } else {
            if (fwrite(p.data.data(), 1, p.data.size(), fp) != p.data.size())
                ok = false;
        }
    }
    if (bsf) av_bsf_free(&bsf);
    fclose(fp);
    if (!ok) impl_->Logf("ExtractClip: write error to '%s'", path.c_str());
    return ok;
}

int Pipeline::MaxBatch() const { return impl_->mb; }
int Pipeline::Classes() const { return impl_->classes; }
int Pipeline::Anchors() const { return impl_->anchors; }

void Pipeline::FetchFrame(const FrameResult& fr, uint8_t* dst_host) const {
    if (!fr.frame_hold) {
        throw std::runtime_error(
            "FetchFrame: result has no frame_hold - construct Pipeline "
            "with hold_frames=True");
    }
    // Type-erased back to its real type (see FrameResult::frame_hold in
    // result.h and Nv12RingHold above) - valid: a shared_ptr<void> built
    // from a Nv12RingHold* always points at one.
    const Nv12RingHold* h =
        static_cast<const Nv12RingHold*>(fr.frame_hold.get());
    // The calling thread has no CUDA context of its own current (it's a
    // consumer thread, not a producer/GPU thread) - make the frame's
    // owning context current before touching its device memory. Post-
    // migration this is the retained primary context (see Setup()), so in
    // practice this is a no-op if the thread already has some context
    // current from having called torch/CuPy first, but keeping it explicit
    // costs nothing and doesn't assume that.
    CheckCu(cuCtxSetCurrent(h->ctx), "cuCtxSetCurrent (FetchFrame)");
    CUDA_MEMCPY2D copy = {};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = (CUdeviceptr)fr.frame_addr;
    copy.srcPitch = fr.frame_pitch;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = dst_host;
    copy.dstPitch = fr.frame_width;  // dst is densely packed (no padding)
    copy.WidthInBytes = fr.frame_width;
    copy.Height = fr.frame_height * 3 / 2;  // luma + interleaved UV
    CheckCu(cuMemcpy2D(&copy), "cuMemcpy2D (FetchFrame)");
}

uintptr_t Pipeline::InputBindingAddr() const {
    return reinterpret_cast<uintptr_t>(impl_->engine->InputPtr());
}
uintptr_t Pipeline::OutputBindingAddr() const {
    return reinterpret_cast<uintptr_t>(impl_->engine->OutputPtr());
}

}  // namespace cordero
