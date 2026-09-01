// _pycamtrt: thin pybind11 binding over cordero::Pipeline (see
// src/core/pipeline.h, graph.h, result.h). This module exposes the C++
// types as directly as reasonable - no policy, no defaults beyond what the
// C++ structs already default to. The declarative, task-shaped API (Layer/
// Engine/Postprocess/Pipeline as the user writes it) lives in the pure
// python package python/pycamtrt/__init__.py, which compiles down to the
// StepDesc/LayerDesc/PipelineConfig graph bound here.
//
// Two GIL-sensitive spots, per the settled design:
//   - Pipeline ctor / Start() release the GIL: engine load + warmup can
//     take seconds and must not block other Python threads.
//   - Pipeline::Poll releases the GIL around the blocking C++ wait so
//     Python (signal handling, other threads) stays responsive; Ctrl-C
//     during a long Poll() is delivered promptly because CPython runs the
//     signal handler on GIL reacquire, and Poll returns Timeout at least
//     every timeout_ms.
//   - cfg.log, when set to a Python callable, is invoked from arbitrary
//     C++ threads (producer/GPU thread) - the wrapper acquires the GIL
//     before calling into Python. Keep that callback cheap: it runs on the
//     hot path and holds the GIL while it runs.

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <stdexcept>

#include "core/graph.h"
#include "core/pipeline.h"
#include "core/result.h"

namespace py = pybind11;

using cordero::Backpressure;
using cordero::ChildOutput;
using cordero::Detection;
using cordero::Family;
using cordero::FrameResult;
using cordero::LayerDesc;
using cordero::Pipeline;
using cordero::PipelineConfig;
using cordero::SinkDesc;
using cordero::SinkKind;
using cordero::StepDesc;
using cordero::StepKind;
using cordero::StreamDesc;
using cordero::StreamInfo;

PYBIND11_MODULE(_pycamtrt, m) {
    m.doc() =
        "Thin pybind11 binding over cordero::Pipeline. Not meant to be used "
        "directly - see the pycamtrt package for the user-facing API.";

    py::enum_<StepKind>(m, "StepKind")
        .value("Process", StepKind::Process)
        .value("Engine", StepKind::Engine)
        .value("Postprocess", StepKind::Postprocess);

    py::enum_<Family>(m, "Family")
        .value("YoloDetect", Family::YoloDetect)
        .value("Ctc", Family::Ctc)
        .value("Argmax", Family::Argmax)
        .value("YoloE2E", Family::YoloE2E);

    py::enum_<Backpressure>(m, "Backpressure")
        .value("Block", Backpressure::Block)
        .value("DropOldest", Backpressure::DropOldest);

    // Phase A1/A2 (endpoint sinks) - see graph.h's SinkKind/SinkDesc
    // WHY-comments for the exact grammar/semantics each kind expects.
    py::enum_<SinkKind>(m, "SinkKind")
        .value("Events", SinkKind::Events)
        .value("StreamRelay", SinkKind::StreamRelay);

    py::class_<StepDesc>(m, "StepDesc")
        .def(py::init<>())
        .def_readwrite("kind", &StepDesc::kind)
        .def_readwrite("input", &StepDesc::input)
        .def_readwrite("engine_path", &StepDesc::engine_path)
        .def_readwrite("family", &StepDesc::family)
        .def_readwrite("score_thresh", &StepDesc::score_thresh)
        .def_readwrite("iou_thresh", &StepDesc::iou_thresh)
        // M1a/M3a: Engine-step input normalization/color (see graph.h's
        // WHY-comment) - NAN per element (norm_offset/norm_scale, each a
        // 3-element PER-CHANNEL array - StepDesc's own field default) / -1
        // (color) mean "inherit the family default for this engine's
        // position in the graph" (pycamtrt.Engine(norm=None, color=None)
        // compiles to exactly these C++ defaults - see
        // python/pycamtrt/__init__.py). Bound as std::array<float,3>
        // (<->Python 3-tuple) rather than def_readwrite because pybind11
        // has no built-in caster for a raw C array member.
        .def_property(
            "norm_offset",
            [](const StepDesc& s) {
                return std::array<float, 3>{s.norm_offset[0], s.norm_offset[1],
                                            s.norm_offset[2]};
            },
            [](StepDesc& s, const std::array<float, 3>& v) {
                for (int i = 0; i < 3; i++) s.norm_offset[i] = v[i];
            })
        .def_property(
            "norm_scale",
            [](const StepDesc& s) {
                return std::array<float, 3>{s.norm_scale[0], s.norm_scale[1],
                                            s.norm_scale[2]};
            },
            [](StepDesc& s, const std::array<float, 3>& v) {
                for (int i = 0; i < 3; i++) s.norm_scale[i] = v[i];
            })
        .def_readwrite("color", &StepDesc::color);

    py::class_<LayerDesc>(m, "LayerDesc")
        .def(py::init<>())
        .def_readwrite("name", &LayerDesc::name)
        .def_readwrite("steps", &LayerDesc::steps)
        .def_readwrite("sahi", &LayerDesc::sahi)
        .def_readwrite("sahi_tile", &LayerDesc::sahi_tile)
        .def_readwrite("sahi_overlap", &LayerDesc::sahi_overlap)
        .def_readwrite("sahi_merge_iou", &LayerDesc::sahi_merge_iou)
        .def_readwrite("sahi_full_frame", &LayerDesc::sahi_full_frame)
        .def_readwrite("sahi_serial", &LayerDesc::sahi_serial);

    py::class_<StreamDesc>(m, "StreamDesc")
        .def(py::init<>())
        .def_readwrite("url", &StreamDesc::url)
        .def_readwrite("skip", &StreamDesc::skip)
        .def_readwrite("decode", &StreamDesc::decode);

    py::class_<SinkDesc>(m, "SinkDesc")
        .def(py::init<>())
        .def_readwrite("kind", &SinkDesc::kind)
        .def_readwrite("input", &SinkDesc::input)
        .def_readwrite("stream_id", &SinkDesc::stream_id)
        .def_readwrite("target", &SinkDesc::target);

    py::class_<PipelineConfig>(m, "PipelineConfig")
        .def(py::init<>())
        .def_readwrite("streams", &PipelineConfig::streams)
        .def_readwrite("steps", &PipelineConfig::steps)
        .def_readwrite("layers", &PipelineConfig::layers)
        .def_readwrite("skip", &PipelineConfig::skip)
        .def_readwrite("key_only", &PipelineConfig::key_only)
        .def_readwrite("verify", &PipelineConfig::verify)
        .def_readwrite("max_frames", &PipelineConfig::max_frames)
        .def_readwrite("queue_capacity", &PipelineConfig::queue_capacity)
        .def_readwrite("backpressure", &PipelineConfig::backpressure)
        // Three-tier frame-memory access (see result.h/pipeline.h):
        // hold_frames extends a ring slot's lifetime to the consumer;
        // ring_depth is the frames-in-flight slack per producer - raise it
        // when holding frames (see graph.h's WHY-comments for both).
        .def_readwrite("hold_frames", &PipelineConfig::hold_frames)
        .def_readwrite("ring_depth", &PipelineConfig::ring_depth)
        // Phase A1/A2 endpoint sinks (see graph.h): sinks is empty by
        // default (no sink threads spawned); ring_seconds is the packet
        // ring's fixed duration feeding StreamRelay sinks and
        // extract_clip() - 0 opts a pipeline out of that ring entirely.
        .def_readwrite("sinks", &PipelineConfig::sinks)
        .def_readwrite("ring_seconds", &PipelineConfig::ring_seconds)
        // cfg.log: None (default, unset) => C++ side falls back to
        // fprintf(stderr, ...) since PipelineConfig::log is a nullptr
        // std::function by default. A Python callable is wrapped so the
        // GIL is (re)acquired before entering Python - it will be called
        // from producer/GPU threads that do not otherwise hold it.
        .def_property(
            "log",
            [](const PipelineConfig&) -> py::object {
                // Not readable back meaningfully (std::function isn't
                // introspectable); expose None here rather than lie about
                // holding the last-set callable.
                return py::none();
            },
            [](PipelineConfig& cfg, py::object fn) {
                if (fn.is_none()) {
                    cfg.log = nullptr;
                    return;
                }
                if (!py::isinstance<py::function>(fn) && !PyCallable_Check(fn.ptr())) {
                    throw py::type_error("cfg.log must be callable or None");
                }
                cfg.log = [fn](const std::string& s) {
                    py::gil_scoped_acquire gil;
                    try {
                        fn(s);
                    } catch (const py::error_already_set&) {
                        // Swallow: a log callback throwing must not tear
                        // down a producer/GPU thread. The exception is
                        // still visible via sys.unraisablehook.
                        PyErr_WriteUnraisable(fn.ptr());
                    }
                };
            });

    py::class_<Detection>(m, "Detection")
        .def(py::init<>())
        .def_readonly("x", &Detection::x)
        .def_readonly("y", &Detection::y)
        .def_readonly("w", &Detection::w)
        .def_readonly("h", &Detection::h)
        .def_readonly("score", &Detection::score)
        .def_readonly("cls", &Detection::cls)
        .def("__repr__", [](const Detection& d) {
            return "<Detection x=" + std::to_string(d.x) +
                   " y=" + std::to_string(d.y) + " w=" + std::to_string(d.w) +
                   " h=" + std::to_string(d.h) +
                   " score=" + std::to_string(d.score) +
                   " cls=" + std::to_string(d.cls) + ">";
        });

    // M4b: one sibling recognition child's per-detection output (a depth-2
    // tree - see core/result.h's ChildOutput WHY-comment). Bound readonly,
    // same shape as Detection above - pycamtrt's Result.outputs (Python)
    // wraps this into the per-layer-name dict a caller actually iterates.
    py::class_<ChildOutput>(m, "ChildOutput")
        .def(py::init<>())
        .def_readonly("layer", &ChildOutput::layer)
        .def_readonly("texts", &ChildOutput::texts)
        .def_readonly("labels", &ChildOutput::labels)
        .def_readonly("label_scores", &ChildOutput::label_scores)
        .def("__repr__", [](const ChildOutput& c) {
            return "<ChildOutput layer=" + std::to_string(c.layer) +
                   " texts=" + std::to_string(c.texts.size()) +
                   " labels=" + std::to_string(c.labels.size()) + ">";
        });

    py::class_<FrameResult>(m, "FrameResult")
        .def(py::init<>())
        .def_readonly("stream_id", &FrameResult::stream_id)
        .def_readonly("frame_no", &FrameResult::frame_no)
        .def_readonly("pts_us", &FrameResult::pts_us)
        .def_readonly("batch_size", &FrameResult::batch_size)
        .def_readonly("batch_seq", &FrameResult::batch_seq)
        .def_readonly("ms_pop_to_ready", &FrameResult::ms_pop_to_ready)
        .def_readonly("ms_ready_to_take", &FrameResult::ms_ready_to_take)
        .def_readonly("ms_take_to_done", &FrameResult::ms_take_to_done)
        .def_readonly("detections", &FrameResult::detections)
        .def_readonly("texts", &FrameResult::texts)
        // M1a: the Argmax-family counterpart of `texts` above - aligned
        // with `detections` the same way, populated only when a layer-1
        // Postprocess(Argmax) cascade exists (empty otherwise - see
        // result.h's WHY-comment).
        .def_readonly("labels", &FrameResult::labels)
        .def_readonly("label_scores", &FrameResult::label_scores)
        // M4b: every sibling recognition child's own per-detection output
        // (a depth-2 tree), in compiled-graph layer order - see
        // core/result.h's FrameResult::children WHY-comment. `texts`/
        // `labels`/`label_scores` above remain populated too, from the
        // FIRST child of the matching family (back-compat).
        .def_readonly("children", &FrameResult::children)
        .def_readonly("verified", &FrameResult::verified)
        .def_readonly("verify_ok", &FrameResult::verify_ok)
        .def_readonly("verify_cpu_dets", &FrameResult::verify_cpu_dets)
        // Tier 1 (informational, always populated for an inferred frame -
        // see result.h): printable/loggable proof of where the frame's
        // full-res NV12 copy lives. WITHOUT hold_frames the memory may be
        // recycled at any time - never dereference frame_addr yourself;
        // use frame_cuda()/fetch_frame() (which require hold_frames) for
        // actual access.
        .def_readonly("frame_addr", &FrameResult::frame_addr)
        .def_readonly("frame_pitch", &FrameResult::frame_pitch)
        .def_readonly("frame_width", &FrameResult::frame_width)
        .def_readonly("frame_height", &FrameResult::frame_height)
        .def("release_frame", &FrameResult::ReleaseFrame,
             "Tier 3: drop this result's hold on its ring slot immediately "
             "rather than waiting for it (and every copy of it) to be "
             "garbage collected. Safe from any thread, any number of "
             "times.")
        .def(
            "frame_cuda",
            [](const FrameResult& fr) -> py::dict {
                if (!fr.frame_hold) {
                    throw std::runtime_error(
                        "frame_cuda(): result has no frame_hold - "
                        "construct Pipeline with hold_frames=True");
                }
                // CUDA Array Interface v3 - see
                // https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html.
                // One (height*3/2, width) uint8 plane: NV12's luma rows
                // followed by its interleaved-UV rows, matching
                // FetchFrame's host layout.
                py::dict d;
                d["shape"] = py::make_tuple(fr.frame_height * 3 / 2,
                                            fr.frame_width);
                d["typestr"] = "|u1";
                d["data"] =
                    py::make_tuple((uintptr_t)fr.frame_addr, /*readonly=*/false);
                d["strides"] = py::make_tuple(fr.frame_pitch, 1);
                d["version"] = 3;
                return d;
            },
            "Tier 3: CUDA Array Interface v3 dict for this frame's full-res "
            "NV12 buffer - requires hold_frames=True (raises RuntimeError "
            "otherwise). Prefer pycamtrt.Result.frame_cuda(), which wraps "
            "this in an object torch.as_tensor()/cupy.asarray() accept "
            "directly and keeps this FrameResult (and its hold) alive.");

    py::class_<StreamInfo>(m, "StreamInfo")
        .def(py::init<>())
        .def_readonly("decoded", &StreamInfo::decoded)
        .def_readonly("reconnects", &StreamInfo::reconnects)
        .def_readonly("failed", &StreamInfo::failed);

    py::class_<Pipeline> pipeline(m, "Pipeline");

    py::enum_<Pipeline::PollStatus>(pipeline, "PollStatus")
        .value("Ok", Pipeline::PollStatus::Ok)
        .value("Timeout", Pipeline::PollStatus::Timeout)
        .value("Finished", Pipeline::PollStatus::Finished);

    pipeline
        .def(py::init([](PipelineConfig cfg) {
                 // Engine load + warmup: seconds, and must not hold the
                 // GIL (also lets cfg.log callbacks - which reacquire the
                 // GIL themselves - run without deadlocking).
                 py::gil_scoped_release release;
                 return std::make_unique<Pipeline>(std::move(cfg));
             }),
             py::arg("cfg"))
        .def(
            "start",
            [](Pipeline& self) {
                py::gil_scoped_release release;
                self.Start();
            },
            "Spawn producer threads + the GPU thread. Single-shot.")
        .def(
            "poll",
            [](Pipeline& self, int timeout_ms) {
                FrameResult r;
                Pipeline::PollStatus st;
                {
                    py::gil_scoped_release release;
                    st = self.Poll(&r, timeout_ms);
                }
                return py::make_tuple(st, std::move(r));
            },
            py::arg("timeout_ms"),
            "Returns (PollStatus, FrameResult). FrameResult is only "
            "meaningful when status is Ok.")
        .def(
            "stop",
            [](Pipeline& self) {
                py::gil_scoped_release release;
                self.Stop();
            },
            "Idempotent, safe from any thread, blocks until torn down.")
        .def("get_stream_info", &Pipeline::GetStreamInfo, py::arg("i"),
             py::return_value_policy::reference_internal)
        .def("dropped_results", &Pipeline::DroppedResults,
             "Count of results evicted by DropOldest backpressure (always 0 "
             "in Block mode).")
        .def("sink_dropped", &Pipeline::SinkDropped,
             "Phase A1: total NDJSON lines dropped across every Events "
             "sink's own bounded drop-oldest queue. Always 0 if no Events "
             "sink is configured.")
        .def(
            "extract_clip",
            [](Pipeline& self, int stream_id, double seconds_back,
               const std::string& path) {
                // Snapshots the ring (lock-guarded copy) and writes the
                // file - both take real wall-clock time proportional to
                // seconds_back, and neither touches Python state, so this
                // releases the GIL like every other blocking call here.
                py::gil_scoped_release release;
                return self.ExtractClip(stream_id, seconds_back, path);
            },
            py::arg("stream_id"), py::arg("seconds_back"), py::arg("path"),
            "Phase A1: snapshot stream_id's packet ring and write a raw "
            "Annex-B .h264 elementary stream (from the newest keyframe "
            "at-or-before now-seconds_back to the ring's end) to `path`. "
            "Returns False (logging why) if stream_id is invalid, the ring "
            "is disabled/empty (see cfg.ring_seconds), or no keyframe is "
            "available yet. Safe to call from any thread while running.")
        .def("max_batch", &Pipeline::MaxBatch)
        .def("classes", &Pipeline::Classes)
        .def("anchors", &Pipeline::Anchors)
        .def(
            "fetch_frame",
            [](Pipeline& self, const FrameResult& fr) {
                py::array_t<uint8_t> out(
                    {(py::ssize_t)(fr.frame_height * 3 / 2),
                     (py::ssize_t)fr.frame_width});
                {
                    // The D2H copy itself can take a little while at full
                    // frame resolution - release the GIL around it like
                    // every other blocking C++ call in this module.
                    py::gil_scoped_release release;
                    self.FetchFrame(fr, out.mutable_data());
                }
                return out;
            },
            py::arg("result"),
            "Tier 2: D2H copy of a hold_frames result's full-res NV12 "
            "frame into a freshly allocated (height*3/2, width) uint8 "
            "numpy array. Raises RuntimeError if the result has no "
            "frame_hold (construct Pipeline with hold_frames=True).")
        .def("input_binding_addr", &Pipeline::InputBindingAddr,
             "Tier-1 info: device pointer (as int) of the engine's input "
             "binding.")
        .def("output_binding_addr", &Pipeline::OutputBindingAddr,
             "Tier-1 info: device pointer (as int) of the engine's output "
             "binding.");
}
