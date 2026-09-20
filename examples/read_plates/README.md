# read_plates

The canonical two-stage cascade: stage 1 ("detect") runs a YOLOv8 plate
detector over each stream; stage 2 ("read") crops every plate detection
(auto-cropped by the cross-layer `Engine(detect_postprocess, READ_ENGINE)`
edge) and runs an LPRNet-style CTC OCR engine over it. `read_plates.py`
and `read_plates.cpp` build the identical `detect(yolo) -> read(ctc)` step
graph - the `.cpp` is a direct `pycamtrt::Pipeline` port, not a different
example.

Note on output text: the bundled LPRNet engine was trained on a Chinese
license-plate charset, so decoded strings on non-Chinese plates look
garbled (e.g. `<wan>Y2444`) - expected. This exercises the detect → crop →
OCR *mechanics*, not charset fit for any particular region.

## Prerequisites

- `models/yolov8n_plates_b1-16_fp16_sm86.engine` and
  `models/lprnet_b1-32_fp32_sm86.engine` on disk. Both already exist in a
  checked-out repo; if missing, point at the corresponding `.onnx` instead
  (`models/yolov8n_plates_dynamic.onnx` is **not distributed** - see
  `models/README.md` - bring your own plate detector export) and the
  engine builds+caches itself on first use (`Engine()` accepts `.onnx`
  directly - `models/README.md`).
- The stream farm serving plate content:
  `cd tools/stream_farm && CLIP=media/atlas_plate_g30.mp4 ./farm.sh up 2`.

## Run it

**Python** (inside the `tensorrt-dev` container):

```bash
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace \
    -v ~/anaconda3:/home/<user>/anaconda3 \
    -w /workspace tensorrt-dev bash -c '
        ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
        PYTHONPATH=/workspace/build:/workspace/python \
        python3 \
        examples/read_plates/read_plates.py rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
    '
```

Pass `--verify` to also run the C++ side's per-frame CPU-reference check
(`pycamtrt.Pipeline(verify=True)`) - a "PYTHON VERIFY N/N" line replaces
the plate summary at the end.

**C++** (after `cmake`/`make` - see `examples/README.md`):

```bash
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace -w /workspace tensorrt-dev bash -c '
        ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
        ./build/examples/read_plates rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2
    '
```

**Docker** (this directory's convenience `Dockerfile` - see its header for
what it does and doesn't bake in):

```bash
docker build -t pycamtrt-read-plates examples/read_plates
docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v $(pwd):/workspace pycamtrt-read-plates
```

## Expected output (captured against the 2-cam plate farm)

Both the Python and C++ runs print per-plate lines as they arrive, then a
summary. Truncated, actual run:

```
s0 frame 598: <wan>Y2444
s0 frame 600: <wan>Y2444

---- summary ----
frames seen: 600  plates read: 599
distinct plate strings:
    586  '<wan>Y2444'
     13  '<wan>Y24449'
s0: decoded=600 reconnects=0
s1: decoded=600 reconnects=0
```

The **dominant plate string always agrees between the Python and C++
runs** (`<wan>Y2444` in both, captured this session) - per-string *counts*
can differ slightly run-to-run (the farm's two ffmpeg loops are
independent live RTSP streams, decorrelated by their staggered starts),
but which string wins is deterministic for a given clip/detector/OCR
triple.
