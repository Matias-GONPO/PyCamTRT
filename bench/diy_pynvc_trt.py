"""bench/diy_pynvc_trt.py - the "build it yourself with NVIDIA's Python
pieces" baseline for the comparison ladder.

What a competent engineer would write WITHOUT PyCamTRT, using NVIDIA's own
Python building blocks: PyNvVideoCodec for zero-copy NVDEC decode of an
RTSP stream, torch CUDA ops for letterbox preprocessing, the TensorRT
Python API for detector inference, torchvision.ops.batched_nms for
postprocess. One thread per stream, each with its own decoder + TRT
execution context (the natural DIY shape - there is no cross-stream
batcher, cascade crop machinery, or backpressure design; building those IS
the library).

Measured per stream: per-frame processing latency split
(preprocess / infer / postprocess, CUDA-event fenced) and aggregate
results/s. CSV rows in the house grid convention:
    content,N,frames,n_results,wall_s,pre_ms,infer_ms,post_ms,e2e_ms

Deps (resolved at bench time, NEVER installed into the shared Python-dev
env - see the campaign plan): python with `tensorrt`, `PyNvVideoCodec`,
`torch`, `torchvision`, `numpy`. The DS/TRT containers ship `tensorrt`
for their system python; `pip install pynvvideocodec` adds decode; torch
CUDA wheels as available - or run inside an isolated venv.

Usage: diy_pynvc_trt.py out.csv N1,N2 content frames engine_path
  e.g. diy_pynvc_trt.py results/diy.csv 1,16 cam 900 \
           models/yolov8n_plates_b1-48_fp16_sm86.engine
"""
import sys
import threading
import time

import numpy as np
import tensorrt as trt
import torch
import torchvision
import PyNvVideoCodec as nvc

SCORE_THRESH = 0.4
IOU_THRESH = 0.45
SIZE = 640  # same production values as the PyCamTRT side


class TrtRunner:
    """Minimal TRT python wrapper: deserialize, pin batch 1, expose a
    torch-tensor infer(). One instance per stream thread (the DIY shape -
    contexts are not shared across threads without extra machinery)."""

    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(
                f.read())
        self.ctx = self.engine.create_execution_context()
        self.in_name = None
        self.out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                self.out_name = name
        in_shape = list(self.engine.get_tensor_shape(self.in_name))
        in_shape[0] = 1
        self.ctx.set_input_shape(self.in_name, in_shape)
        out_shape = list(self.ctx.get_tensor_shape(self.out_name))
        self.out = torch.empty(out_shape, dtype=torch.float32,
                               device="cuda")

    def infer(self, inp, stream):
        self.ctx.set_tensor_address(self.in_name, inp.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.out.data_ptr())
        self.ctx.execute_async_v3(stream.cuda_stream)
        return self.out


def letterbox_gpu(rgb, size=SIZE):
    """torch letterbox: resize + pad to size x size, NCHW float [0,1]."""
    _, h, w = rgb.shape
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    img = torch.nn.functional.interpolate(
        rgb.unsqueeze(0).float() / 255.0, size=(nh, nw), mode="bilinear",
        align_corners=False)
    out = torch.full((1, 3, size, size), 114 / 255.0, device="cuda")
    px, py = (size - nw) // 2, (size - nh) // 2
    out[:, :, py:py + nh, px:px + nw] = img
    return out


def decode_postprocess(raw, num_classes):
    """yolov8 head decode + NMS via torchvision (the standard DIY path)."""
    pred = raw[0]  # [4+nc, anchors]
    boxes = pred[:4].T  # cxcywh
    scores, cls = pred[4:4 + num_classes].max(0)
    keep = scores > SCORE_THRESH
    if not keep.any():
        return 0
    b = boxes[keep]
    xyxy = torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                        b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    kept = torchvision.ops.batched_nms(xyxy, scores[keep], cls[keep],
                                       IOU_THRESH)
    return int(kept.numel())


def stream_worker(url, engine_path, frames, stats, idx):
    torch.cuda.init()
    stream = torch.cuda.Stream()
    runner = TrtRunner(engine_path)
    num_classes = runner.out.shape[1] - 4
    # LIVE-STREAM REALITY (bench finding, recorded in report 12):
    # PyNvVideoCodec's own demuxer handles FILES but segfaults on rtsp://
    # in the pip wheel - a live-camera DIY must bring a third-party
    # demuxer. This uses PyAV (the standard workaround): demux h264
    # packets, hand each to the NVDEC decoder via PacketData. RGBP planar
    # output straight from the decoder.
    import av
    container = av.open(url, options={"rtsp_transport": "tcp"}, timeout=10)
    vstream = container.streams.video[0]
    dec = nvc.CreateDecoder(gpuid=0, codec=nvc.cudaVideoCodec.H264,
                            cudacontext=0, cudastream=0,
                            usedevicememory=True,
                            outputColorType=nvc.OutputColorType.RGBP)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    n = dets = 0
    pre_ms = inf_ms = post_ms = 0.0
    t0 = time.time()

    def frame_iter():
        pd = nvc.PacketData()
        for av_pkt in container.demux(vstream):
            raw = bytes(av_pkt)
            if not raw:
                continue
            arr = np.frombuffer(raw, dtype=np.uint8)
            pd.bsl_data = arr.ctypes.data
            pd.bsl = arr.size
            for decoded in dec.Decode(pd):
                yield decoded

    for frame in frame_iter():
        with torch.cuda.stream(stream):
            rgb = torch.as_tensor(frame, device="cuda")  # CAI zero-copy
            ev[0].record(stream)
            inp = letterbox_gpu(rgb.reshape(3, rgb.shape[-2], rgb.shape[-1]))
            ev[1].record(stream)
            raw = runner.infer(inp, stream)
            ev[2].record(stream)
            dets += decode_postprocess(raw, num_classes)
            ev[3].record(stream)
        stream.synchronize()
        pre_ms += ev[0].elapsed_time(ev[1])
        inf_ms += ev[1].elapsed_time(ev[2])
        post_ms += ev[2].elapsed_time(ev[3])
        n += 1
        if n >= frames:
            break
    wall = time.time() - t0
    m = max(n, 1)
    stats[idx] = dict(n=n, dets=dets, wall=wall, pre=pre_ms / m,
                      inf=inf_ms / m, post=post_ms / m)


def main():
    out, ns, content, frames, engine = (sys.argv[1], sys.argv[2].split(","),
                                        sys.argv[3], int(sys.argv[4]),
                                        sys.argv[5])
    with open(out, "a") as f:
        for N in ns:
            N = int(N)
            urls = [f"rtsp://127.0.0.1:8554/{content}{i}"
                    for i in range(1, N + 1)]
            stats = [None] * N
            threads = [threading.Thread(target=stream_worker,
                                        args=(u, engine, frames, stats, i))
                       for i, u in enumerate(urls)]
            t0 = time.time()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.time() - t0
            ok = [s for s in stats if s]
            n_total = sum(s["n"] for s in ok)
            pre = sum(s["pre"] for s in ok) / max(len(ok), 1)
            inf = sum(s["inf"] for s in ok) / max(len(ok), 1)
            post = sum(s["post"] for s in ok) / max(len(ok), 1)
            row = (f"{content},{N},{frames},{n_total},{wall:.1f},"
                   f"{pre:.3f},{inf:.3f},{post:.3f},"
                   f"{pre + inf + post:.3f}")
            print("CELL", row, flush=True)
            f.write(row + "\n")
            f.flush()


if __name__ == "__main__":
    main()
