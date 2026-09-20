# PyCamTRT examples

Six worked pipelines, each in its own subdirectory with a Python script, a
native C++ port built directly on `pycamtrt::Pipeline`, a `README.md` (what
it shows, prerequisites, and the exact commands for all three run modes),
and an optional convenience `Dockerfile`. The `.py` and `.cpp` in a given
subdirectory build the **identical step graph** and print equivalent
per-result lines - the C++ port is there to show what the Python compiler
layer (`python/pycamtrt/__init__.py`) compiles down to, not a different
example.

Plus a seventh, [`fleet_demo/`](fleet_demo/) - a minimal live **app**
rather than a step-graph mechanics demo: one Python process (pycamtrt
consumer thread + a small FastAPI/SSE server) and one plain `index.html`
that put the library's three capacity dials - N streams, decode rate,
inference rate - on sliders over all 65 AIC22 cameras, with detection
boxes and track ids drawn in the browser and measured performance next to
`recommend()`'s prediction - **Python-only, no C++ port** (a deliberate
deviation from the pattern above - the point of this one is the live
service, not re-demonstrating graph construction in a second language).
It has its own `run.sh up|down|status` lifecycle script instead of a bare
`docker run` line, and its own `README.md` with prerequisites, the dials,
and what the stats line means.

All commands below assume: the repo root as your working directory, the
[stream farm](../tools/stream_farm) up (`cd tools/stream_farm && ./farm.sh
up 2`; plate-content examples need `CLIP=media/atlas_plate_g30.mp4` set
before `up`), and (for the Python/C++ rows) that you're already inside the
`tensorrt-dev` container - see [`examples/CLAUDE.md`](CLAUDE.md) or the
root [`BUILD.md`](../BUILD.md) for the full docker incantation. Each
example's own `README.md` spells out the complete copy-paste commands
(docker wrapper included) plus a captured expected-output sample.

| example | demonstrates | Python | C++ (after `cmake`/`make`) | Docker |
|---|---|---|---|---|
| [`read_plates/`](read_plates/) | two-stage detect→crop→OCR cascade; the cross-layer edge that auto-crops a detection into the next engine (`family="ctc"`) | `python3 examples/read_plates/read_plates.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `./build/examples/read_plates rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `docker build -t pycamtrt-read-plates examples/read_plates && docker run --rm --gpus all --network host -v $PWD:/workspace pycamtrt-read-plates` |
| [`classify_detections/`](classify_detections/) | the same cascade shape with the OCR child swapped for an ImageNet classifier (`family="argmax"`, true per-channel norm) | `python3 examples/classify_detections/classify_detections.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `./build/examples/classify_detections rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `docker build -t pycamtrt-classify examples/classify_detections && docker run --rm --gpus all --network host -v $PWD:/workspace pycamtrt-classify` |
| [`read_and_classify/`](read_and_classify/) | depth-2 TREE: one detector root feeding TWO sibling children (`ctc` + `argmax`) that crop the same root independently | `python3 examples/read_and_classify/read_and_classify.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `./build/examples/read_and_classify rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `docker build -t pycamtrt-read-and-classify examples/read_and_classify && docker run --rm --gpus all --network host -v $PWD:/workspace pycamtrt-read-and-classify` |
| [`detect_rtdetr/`](detect_rtdetr/) | RT-DETR as the layer-0 detector family (`family="rtdetr"`, NMS-free, normalized box decode) | `python3 examples/detect_rtdetr/detect_rtdetr.py rtsp://localhost:8554/cam1` | `./build/examples/detect_rtdetr rtsp://localhost:8554/cam1` | `docker build -t pycamtrt-rtdetr examples/detect_rtdetr && docker run --rm --gpus all --network host -v $PWD:/workspace pycamtrt-rtdetr` |
| [`route_and_reid/`](route_and_reid/) | `Select` detection routing (person→embedding, vehicle→classifier) + the `embedding` family for cross-camera re-ID | `python3 examples/route_and_reid/route_and_reid.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `./build/examples/route_and_reid rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `docker build -t pycamtrt-route-reid examples/route_and_reid && docker run --rm --gpus all --network host -v $PWD:/workspace pycamtrt-route-reid` |
| [`zone_filter/`](zone_filter/) | the "Python on compact results" consumer-side tier: per-stream polygon zones, occupancy ENTER/LEAVE, `backpressure="drop_oldest"` | `python3 examples/zone_filter/zone_filter.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `./build/examples/zone_filter rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2` | `docker build -t pycamtrt-zone-filter examples/zone_filter && docker run --rm --gpus all --network host -v $PWD:/workspace pycamtrt-zone-filter` |
| [`fleet_demo/`](fleet_demo/) | **app example, not a mechanics demo** - the smallest live app on PyCamTRT: one process (pycamtrt consumer thread + FastAPI/SSE), a plain-JS page with WebRTC tiles, detection boxes + track ids, and the three capacity dials (N streams, decode rate, inference rate) as sliders with measured vs `recommend()` stats over all 65 AIC22 cameras | `cd examples/fleet_demo && ./run.sh up` (page at `http://localhost:8000/`) | *(no C++ port - Python-only, see the note above)* | `run.sh` drives the bind-mounted `tensorrt-dev` image |

The Python and C++ commands above still need the container wrapper (the
`docker run ... tensorrt-dev bash -c '...'` incantation) around them
unless you're already inside one - see any per-example `README.md` for the
copy-paste version, or `examples/CLAUDE.md` for the full run matrix.

## Building the C++ ports

```bash
mkdir -p build && cd build
cmake -DPython3_EXECUTABLE=$(which python3) ..
make -j read_plates classify_detections read_and_classify detect_rtdetr route_and_reid zone_filter
# binaries land in build/examples/<name>
```

`-DPYCAMTRT_BUILD_EXAMPLES=OFF` skips all six (they're `ON` by default,
gated the same as `rtsp_infer_multi`/`pycamtrt_core` on the Video Codec SDK - see
`CMakeLists.txt` and `BUILD.md`).

## Which one should I look at first?

`read_plates/` is the canonical two-stage cascade and the one most of the
other examples build on (`classify_detections/` swaps its OCR child for a
classifier; `read_and_classify/` runs both as siblings at once). If you
only want the routing/re-ID story, go straight to `route_and_reid/`; if
you want the pure-Python "act on results" pattern with no cascade at all,
`zone_filter/` is the shortest path. If you want to see it wired into a
live application - a browser page, the three capacity dials on sliders,
measured vs predicted performance - go straight to `fleet_demo/`.
