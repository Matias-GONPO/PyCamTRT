#!/usr/bin/env python3
"""Smoke test for pycamtrt (Part 3): builds the one-layer detect graph
against the default COCO engine, drains 50 results, checks StreamInfo,
then repeats a shorter run through the `with` + auto-start iteration path.

Usage (run from the repo root, with PYTHONPATH set to build/ + python/):
    PYTHONPATH=build:python python3 python/smoke_test.py \
        rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
"""
import sys

import pycamtrt

DEFAULT_ENGINE = "models/yolov8n_b1-16_fp32_sm86.engine"


def log(msg: str) -> None:
    print(f"[cordero] {msg}", file=sys.stderr)


def build_pipeline(urls, max_frames):
    streams = pycamtrt.Streams(urls)
    L1 = pycamtrt.Layer("detect")
    eng = L1.add(pycamtrt.Engine(streams, DEFAULT_ENGINE))
    L1.add(pycamtrt.Postprocess(eng, family="yolo", score=0.4, iou=0.45))
    return pycamtrt.Pipeline(
        streams, layers=[L1], skip=2, max_frames=max_frames, verify=False,
        decode="all", sahi=None, log=log,
    )


def phase1(urls):
    print("=== phase 1: plain start()/iterate/stop() ===")
    pipe = build_pipeline(urls, max_frames=0)
    pipe.start()
    n = 0
    for r in pipe:
        n += 1
        if n % 10 == 0:
            print(f"  [{n}] stream_id={r.stream_id} frame_no={r.frame_no} "
                  f"detections={len(r.detections)}")
        if n >= 50:
            break
    pipe.stop()

    assert n == 50, f"expected 50 results, got {n}"
    any_decoded = False
    for i in range(len(urls)):
        si = pipe.get_stream_info(i)
        print(f"  stream {i}: decoded={si.decoded} reconnects={si.reconnects} "
              f"failed={si.failed}")
        if si.decoded > 0:
            any_decoded = True
    assert any_decoded, "GetStreamInfo reports 0 decoded frames on every stream"
    print("SMOKE PASS")


def phase2(urls):
    print("=== phase 2: `with` + auto-start iteration ===")
    n = 0
    with build_pipeline(urls, max_frames=0) as pipe:
        for r in pipe:
            n += 1
            if n >= 10:
                break
    assert n == 10, f"expected 10 results, got {n}"
    print("CTX PASS")


def main():
    urls = sys.argv[1:]
    if not urls:
        print("usage: smoke_test.py URL [URL...]", file=sys.stderr)
        return 1
    phase1(urls)
    phase2(urls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
