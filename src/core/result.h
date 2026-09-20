#pragma once

// Per-frame RESULT types the Pipeline emits via Poll(). Everything a caller
// (CLI today) needs to reproduce today's per-frame prints and aggregates,
// without the core doing any printing/aggregation itself.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pycamtrt {

struct Detection {
    float x, y, w, h, score;
    int cls;
};

// M4b: one SIBLING recognition child's per-detection output (a depth-2
// tree: one detector root feeding N children, each cropping the root's
// detections independently with its own engine/family/norm/color - see
// pipeline.cpp's ChildPlan/Validate()). Aligned with FrameResult::detections
// the same way the pre-M4b single-child texts/labels/label_scores fields
// were: element i describes detections[i]'s crop through THIS child.
// Exactly one of {texts} / {labels, label_scores} / {vectors} is
// populated, matching this child's family (Ctc -> texts, Argmax ->
// labels+label_scores, Embedding -> vectors) - the others stay empty.
// R1 routing note: under a Select filter the ALIGNMENT DOES NOT CHANGE -
// element i still describes detections[i]; a detection routed away from
// this child simply keeps that child's empty/default entry ("" / label 0
// with score 0 / empty vector). One indexing rule, routed or not.
struct ChildOutput {
    int layer = -1;  // index into the compiled graph's layers (0 = the
                     // detector root; siblings are 1..K in declaration
                     // order - same order as FrameResult::children below)
    std::vector<std::string> texts;
    std::vector<int> labels;
    std::vector<float> label_scores;
    // T3.1 embedding family (report 11): vectors[i] is detections[i]'s
    // crop's RAW child-engine output row ([D] floats, e.g. a 512-d re-ID
    // feature) - no decode kernel exists for this family by design, the
    // "decode" IS the pass-through. Populated only for a
    // Family::Embedding child (the other fields stay empty, same
    // exactly-one-populated contract as texts/labels above). Size note:
    // ~D*4 bytes per detection (2 KB at D=512) - tens of detections keeps
    // this inside the compact-results contract.
    std::vector<std::vector<float>> vectors;
    // CP1: this child's own GPU-stream time for the batch this result rode
    // in - cudaEventElapsedTime(ev_start, ev_done) on the child's stream
    // (see pipeline.cpp's ChildScratch), same value on every result of that
    // batch (mirrors FrameResult::ms_take_to_done's whole-batch semantics,
    // just per-child instead of per-batch). RAW meaning, read carefully:
    // enqueue-to-done on WHATEVER stream this child actually ran on. Under
    // the default (PipelineConfig::cascade_serial == false), that is the
    // child's OWN dedicated stream, so this is a genuinely isolated
    // per-child GPU time (can overlap with siblings - the whole point).
    // Under cascade_serial == true, every child (and the detector's own
    // pass, and SAHI when on) shares ONE stream, so "the child's stream" IS
    // that shared main gpu_stream, carrying every sibling's traffic in
    // strict FIFO order rather than this child's own dedicated queue - the
    // measured window still brackets only this child's own enqueued work,
    // but on a stream whose scheduling is now entangled with everyone
    // else's, rather than isolated. That asymmetry between the two modes is
    // deliberate, not a bug to normalize away - it IS the A/B measurement
    // (see qa_matrix.py section J: compare this field's mean, per child,
    // cascade_serial=True vs. False, on the same content).
    double ms_gpu = 0;
};

struct FrameResult {
    int stream_id = -1;
    int frame_no = 0;
    int64_t pts_us = -1;
    int batch_size = 0;
    // Monotonically increasing GPU-batch index (same value on every frame
    // that rode in one batch): lets a consumer count true batches and build
    // a per-batch histogram instead of a frames-weighted one.
    int batch_seq = -1;

    // Three-way latency split, same instants as the old StreamStats/summary
    // math: pop (producer popped the decoded frame) -> ready (preprocess
    // synced) -> take (GPU thread claimed the batch) -> done (batch fully
    // postprocessed, including SAHI/OCR when enabled).
    double ms_pop_to_ready = 0;
    double ms_ready_to_take = 0;
    double ms_take_to_done = 0;

    std::vector<Detection> detections;   // final (post-SAHI-merge) layer-0 detections
    // M4b BACK-COMPAT (see ChildOutput above and pipeline.cpp's GpuLoop
    // assembly-loop WHY-comment): a depth-2 tree can have MANY sibling
    // Ctc/Argmax children now (`children` below carries every one of
    // them), but these three flat fields predate the tree and every
    // existing consumer (the CLI's --ocr prints, the Events sink's NDJSON
    // schema, qa_matrix's pre-M4b gates) reads them directly - so they are
    // KEPT, unconditionally filled from the FIRST child of the matching
    // family (first Ctc child -> texts; first Argmax child -> labels/
    // label_scores), never re-derived per caller. A single-child pipeline
    // (still the common case) sees byte-identical values to before M4b.
    std::vector<std::string> texts;      // aligned with detections when ANY Ctc child exists, else empty
    // WHY (M1a, classifier/argmax family): the Argmax counterpart of
    // `texts` above - aligned with `detections` the same way (labels[i]/
    // label_scores[i] describe detections[i]'s crop through the FIRST
    // Argmax child), but only populated when at least one Argmax cascade
    // child exists - empty otherwise (see pipeline.cpp's GpuLoop cascade
    // block and graph.h's Family::Argmax). `labels[i]` is the winning class
    // index; `label_scores[i]` is that class's raw logit - no softmax (see
    // LaunchArgmaxBatched's WHY-comment in postprocess.h for why raw-logit
    // confidence is the documented contract, not a probability).
    std::vector<int> labels;
    std::vector<float> label_scores;
    // M4b: EVERY sibling recognition child's own per-detection output (a
    // depth-2 tree - see ChildOutput above), in the same order as the
    // compiled graph's non-root layers. Empty iff the pipeline has no
    // cascade children at all. A single-child pipeline has exactly one
    // entry here, redundant with (but independent storage from) the
    // texts/labels/label_scores back-compat fields above.
    std::vector<ChildOutput> children;

    bool verified = false;
    bool verify_ok = false;
    int verify_cpu_dets = -1;

    // ---- Three-tier frame-memory access -----------------------------
    // Tier 1 (informational, ALWAYS populated for an attributed/inferred
    // frame): raw device pointer/pitch/dims of this frame's full-res NV12
    // ring slot (see pipeline.cpp's Nv12Ring/SlotMeta), as of emission -
    // printable/loggable proof of where the frame lives, e.g. for an
    // external tool correlating addresses. WITHOUT hold_frames the ring
    // slot is released back to its producer the moment the GPU thread is
    // done with it and may be OVERWRITTEN BY ANOTHER FRAME at any point
    // after this FrameResult is emitted - never dereference frame_addr
    // unless frame_hold (below) is non-null.
    uint64_t frame_addr = 0;
    int frame_pitch = 0;
    int frame_width = 0;
    int frame_height = 0;

    // Tier 3 (hold_frames only): non-null iff PipelineConfig::hold_frames
    // was set AND this result rode with a ring-backed slot. Holding this
    // (via a copy of the FrameResult, or of just this shared_ptr) keeps
    // the owning Nv12Ring slot's `in_use` flag set - so frame_addr stays
    // live, readable, and safe to hand to another CUDA context (see
    // Pipeline::FetchFrame and the CUDA Array Interface binding) - until
    // EVERY copy is destroyed or ReleaseFrame() is called on one of them.
    // Typed as an opaque void* deliberately (Pimpl, matching pipeline.h):
    // this header stays free of CUDA/pipeline-internals types; the real
    // payload is a small ring-slot-releasing holder constructed only in
    // pipeline.cpp.
    std::shared_ptr<void> frame_hold;

    // Explicit early release: drops this FrameResult's reference to the
    // hold immediately, rather than waiting for the FrameResult (and every
    // other copy of it) to be destroyed. The deleter behind frame_hold
    // only touches a mutex + condition_variable (no CUDA calls), so this
    // is safe to call from any thread - including a garbage collector
    // (e.g. Python's) - and safe to call more than once or on an
    // already-empty hold.
    void ReleaseFrame() { frame_hold.reset(); }
};

struct StreamInfo {
    int decoded = 0;
    int reconnects = 0;
    bool failed = false;
};

}  // namespace pycamtrt
