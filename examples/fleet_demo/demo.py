"""fleet_demo - the smallest PyCamTRT live app.

N RTSP cameras -> YOLOv8n (COCO) detection -> per-camera IoU tracking -> boxes with
track ids in a browser, with the library's three capacity dials as sliders:
N streams, decode rate (all frames / keyframes only) and inference rate (skip).

One process, three threads:
  main thread   uvicorn: serves index.html, SSE /events, GET/POST /config, GET /stats
  engine thread pycamtrt consumer loop (`for r in pipe`); rebuilds the pipeline
                whenever POST /config stops the current one
  stats thread  1 Hz: measured decoded/inferred fps, GPU ms per frame, drops,
                nvidia-smi utilization, and pycamtrt.recommend()'s prediction

No database, no zones/alerts, no reconnect logic: a slider change rebuilds the
pipeline (warm - the TensorRT engine is cached) and the page reloads itself.
The engine processes all N cameras; only the first SHOWN are displayed. The page
plays the publishers' own mediamtx streams (aic/<slug>) over WebRTC - not a
Sink(kind="stream") relay: measured 2026-09-12, a relay's first connection replays
its ring backlog at real-time pace, which left the relayed video a fixed ~7 s
behind the detections.
Runs inside the tensorrt-dev container - see run.sh.
"""
import asyncio
import json
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pycamtrt
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ONNX = str(REPO / "models" / "yolov8n_dynamic.onnx")  # auto-built once per GPU, cached next to it
MANIFEST = json.load(open(HERE / "data" / "manifest.json"))  # 65 AIC22 cameras, manifest order
RTSP = "rtsp://localhost:8554"
MAX_BATCH = 32        # pinned: every rebuild reuses the same cached engine; >32 streams chunk (logged)
SHOWN = 16            # tiles displayed = streams sent over SSE
VEHICLES = {2, 3, 5, 7}   # COCO car, motorcycle, bus, truck
SCORE = 0.4
SOURCE_FPS = 10.0     # the AIC22 clips are 10 fps; demand = N * SOURCE_FPS
DEFAULT_GOP = 30      # keyframe interval of the transcodes (tools/prep_aic22.py --gop), per manifest entry
PORT = 8000

# ---------------------------------------------------------------------------
# Tracker: greedy IoU association per camera (descending IoU, one-to-one).
# Unmatched detections open a tentative track (confirmed at MIN_HITS); a track
# dies after MAX_AGE_US of pts time without a match. No motion model.
# ---------------------------------------------------------------------------
IOU_THRESH = 0.3
MAX_AGE_US = 1_500_000
MIN_HITS = 2


def _iou(a, b) -> float:
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


@dataclass
class Track:
    id: int
    box: tuple
    last_pts_us: int
    hit_count: int = 1
    confirmed: bool = False


class Tracker:
    def __init__(self):
        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, dets, pts_us: int) -> List[Track]:
        boxes = [(float(d.x), float(d.y), float(d.w), float(d.h)) for d in dets]
        cands = [(iou, ti, di) for ti, tr in enumerate(self.tracks) for di, b in enumerate(boxes)
                 if (iou := _iou(tr.box, b)) >= IOU_THRESH]
        cands.sort(key=lambda c: c[0], reverse=True)
        used_t, used_d, match = set(), set(), {}
        for _, ti, di in cands:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            match[di] = self.tracks[ti]
        out = []
        for di, b in enumerate(boxes):
            tr = match.get(di)
            if tr is None:
                tr = Track(self.next_id, b, pts_us)
                self.next_id += 1
                self.tracks.append(tr)
            else:
                tr.box, tr.last_pts_us, tr.hit_count = b, pts_us, tr.hit_count + 1
            tr.confirmed = tr.hit_count >= MIN_HITS
            out.append(tr)
        self.tracks = [t for t in self.tracks if pts_us - t.last_pts_us <= MAX_AGE_US]
        return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_pipeline(n: int, skip: int, decode: str):
    cams = MANIFEST[:n]
    streams = pycamtrt.Streams([f"{RTSP}/{c['rtsp_path']}" for c in cams])
    detect = pycamtrt.Layer("detect")
    eng = detect.add(pycamtrt.Engine(streams, ONNX, max_batch=MAX_BATCH))
    detect.add(pycamtrt.Postprocess(eng, family="yolo", score=SCORE))
    # No sinks: the browser plays the publishers' streams directly (see the module
    # docstring), so ring_seconds=0 opts out of the packet ring and its per-packet copy.
    return pycamtrt.Pipeline(streams, layers=[detect], skip=skip, decode=decode,
                             backpressure="drop_oldest", ring_seconds=0)


@dataclass
class Gen:              # one pipeline generation, always swapped as ONE object
    pipe: object
    n: int
    skip: int
    decode: str
    num: int
    gop: int            # mean keyframe interval of the enabled cameras (recommend()'s key_gop)


def mean_gop(n: int) -> int:
    return round(sum(c.get("gop", DEFAULT_GOP) for c in MANIFEST[:n]) / max(1, n))


CFG = {"n": 16, "skip": 1, "decode": "all"}   # startup dials: the full displayed grid
GEN: Optional[Gen] = None
READY = threading.Event()   # set while a pipeline is running; cleared during a rebuild
STOP = threading.Event()
STATS: dict = {}
COUNT = {"results": 0, "gpu_ms": 0.0}   # results consumed; EMA of per-frame GPU ms
LOOP: Optional[asyncio.AbstractEventLoop] = None
CLIENTS: set = set()        # one asyncio.Queue per SSE client


# ---------------------------------------------------------------------------
# SSE fan-out (thread -> event loop -> per-client drop-oldest queues)
# ---------------------------------------------------------------------------
def publish(evt: dict) -> None:
    if LOOP is not None:
        LOOP.call_soon_threadsafe(_fanout, evt)


def _fanout(evt: dict) -> None:
    for q in list(CLIENTS):
        if q.full():
            q.get_nowait()
        q.put_nowait(evt)


async def sse_gen(q: asyncio.Queue):
    try:
        while True:
            try:
                evt = await asyncio.wait_for(q.get(), 15)
                yield f"event: {evt['type']}\ndata: {json.dumps(evt)}\n\n"
            except asyncio.TimeoutError:
                yield ": ping\n\n"
    finally:
        CLIENTS.discard(q)


# ---------------------------------------------------------------------------
# Engine thread
# ---------------------------------------------------------------------------
def consume(r, trackers: Dict[int, Tracker]) -> None:
    wall_us = time.time_ns() // 1000   # stamped first: the page pairs video frames to events by it
    dets = [d for d in r.detections if int(d.cls) in VEHICLES]
    tracks = trackers.setdefault(r.stream_id, Tracker()).update(dets, int(r.pts_us))
    COUNT["results"] += 1
    gpu = float(r.ms_take_to_done) / max(1, int(r.batch_size))   # whole-batch time -> per frame
    COUNT["gpu_ms"] = gpu if COUNT["gpu_ms"] == 0 else COUNT["gpu_ms"] + 0.05 * (gpu - COUNT["gpu_ms"])
    if r.stream_id < SHOWN:
        publish({
            "type": "frame", "stream_id": r.stream_id, "frame": r.frame_no, "pts_us": int(r.pts_us),
            "wall_us": wall_us,
            "ms": {"pre": float(r.ms_pop_to_ready), "queue": float(r.ms_ready_to_take),
                   "gpu": float(r.ms_take_to_done)},
            "dets": [{"x": float(d.x), "y": float(d.y), "w": float(d.w), "h": float(d.h),
                      "score": float(d.score), "cls": int(d.cls), "track_id": t.id}
                     for d, t in zip(dets, tracks)],
        })


def engine_loop() -> None:
    global GEN
    num = 0
    while not STOP.is_set():
        c = dict(CFG)
        num += 1
        print(f"building gen {num}: n={c['n']} skip={c['skip']} decode={c['decode']}", flush=True)
        pipe = build_pipeline(**c)
        GEN = Gen(pipe, c["n"], c["skip"], c["decode"], num, mean_gop(c["n"]))
        pipe.start()
        READY.set()
        print(f"ready gen {num}", flush=True)
        trackers: Dict[int, Tracker] = {}
        for r in pipe:          # ends only when pipe.stop() is called (POST /config or shutdown)
            consume(r, trackers)


# ---------------------------------------------------------------------------
# Stats thread
# ---------------------------------------------------------------------------
def gpu_util():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.decoder",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=2).stdout
        g, d = out.strip().split(",")
        return int(g), int(d)
    except Exception:
        return -1, -1


def stats_loop() -> None:
    global STATS
    prev = None   # (gen, decoded, results, t)
    while not STOP.wait(1.0):
        g = GEN
        if g is None or not READY.is_set():
            continue
        try:
            infos = [g.pipe.get_stream_info(i) for i in range(g.n)]
            dropped = g.pipe.dropped_results()
        except Exception:
            continue        # pipeline mid-stop
        now = time.monotonic()
        decoded, results = sum(i.decoded for i in infos), COUNT["results"]
        if prev is None or prev[0] != g.num:
            prev = (g.num, decoded, results, now)
            continue
        dt = max(1e-3, now - prev[3])
        rec = pycamtrt.recommend(streams=g.n, resolution="1080p", fps=SOURCE_FPS,
                                 skip=g.skip, decode=g.decode, key_gop=g.gop)
        gu, du = gpu_util()
        STATS = {
            "type": "stats", "n": g.n, "skip": g.skip, "decode": g.decode, "gen": g.num, "gop": g.gop,
            "decoded_fps": round((decoded - prev[1]) / dt, 1), "demand_fps": round(g.n * SOURCE_FPS, 1),
            "results_fps": round((results - prev[2]) / dt, 1), "gpu_ms": round(COUNT["gpu_ms"], 2),
            "dropped": dropped, "failed": sum(1 for i in infos if i.failed),
            "gpu_util": gu, "nvdec_util": du,
            "rec": {"predicted_gpu_ms_per_frame": round(rec.predicted_gpu_ms_per_frame, 2),
                    "gpu_headroom_pct": round(rec.gpu_headroom_pct, 1),
                    "nvdec_headroom_pct": round(rec.nvdec_headroom_pct, 1),
                    "binding_resource": rec.binding_resource,
                    "holds_realtime": rec.holds_realtime, "suggestion": rec.suggestion},
        }
        prev = (g.num, decoded, results, now)
        publish(STATS)


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global LOOP
    LOOP = asyncio.get_running_loop()
    threading.Thread(target=engine_loop, daemon=True).start()
    threading.Thread(target=stats_loop, daemon=True).start()
    yield
    STOP.set()
    if GEN is not None:
        await asyncio.to_thread(GEN.pipe.stop)


app = FastAPI(lifespan=lifespan)
LOCK = asyncio.Lock()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/config")
def get_config():
    return {**CFG, "ready": READY.is_set(), "fps": SOURCE_FPS, "gop": mean_gop(CFG["n"]),
            "slugs": [c["slug"] for c in MANIFEST[:min(CFG["n"], SHOWN)]]}


@app.post("/config")
async def set_config(body: dict):
    n = max(1, min(len(MANIFEST), int(body.get("n", CFG["n"]))))
    skip = max(1, min(10, int(body.get("skip", CFG["skip"]))))
    decode = body.get("decode", CFG["decode"])
    decode = decode if decode in ("all", "key") else "all"
    async with LOCK:
        await asyncio.to_thread(READY.wait)      # a build may still be in progress
        CFG.update(n=n, skip=skip, decode=decode)
        READY.clear()
        await asyncio.to_thread(GEN.pipe.stop)   # ends the engine thread's loop -> it rebuilds from CFG
        await asyncio.to_thread(READY.wait, 300)
    return get_config()


@app.get("/stats")
def get_stats():
    return STATS


@app.get("/events")
async def events():
    q: asyncio.Queue = asyncio.Queue(256)
    CLIENTS.add(q)
    return StreamingResponse(sse_gen(q), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
