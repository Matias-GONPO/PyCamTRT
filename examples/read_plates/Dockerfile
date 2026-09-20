# OPTIONAL convenience runner for the read_plates example ONLY - not the
# project's build image (see the root Dockerfile / BUILD.md for that).
# Same base + toolchain as the root Dockerfile; this just saves retyping
# the canonical docker incantation to run one example.
#
# Does NOT bake the repo or the NVIDIA Video Codec SDK into the image (the
# SDK's license forbids redistributing it - see BUILD.md) - bind-mount the
# repo at /workspace instead (see the `docker run` line below). The image
# builds only the CMake targets this example needs (`_pycamtrt`, or the
# example binary itself for the C++ CMD below) on first CMD run, cached
# under the bind-mounted .docker-build/ for subsequent runs - NOT
# `pip install .`, which builds the WHOLE project including legacy
# single-stream checkpoint binaries (rtsp_decode/preprocess_test/
# rtsp_infer) this example never touches, and which do not compile against
# every CUDA toolkit version this base image tag may ship (a pre-existing,
# unrelated incompatibility - see BUILD.md's known-traps table before
# "fixing" it here). A DEDICATED build directory (not BUILD.md's own
# `build/`) is deliberate: `build/` on a dev checkout is typically already
# configured against a mounted conda interpreter (BUILD.md's workflow) -
# reconfiguring it here against this image's OWN python3 would corrupt
# that cache for both workflows.
FROM nvcr.io/nvidia/tensorrt:25.08-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config \
        libopencv-dev \
        libavformat-dev libavcodec-dev libavutil-dev \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*
# python3-dev: the root Dockerfile omits this, so pybind11/_pycamtrt's
# Development.Module component is NOT found on a bare NGC TensorRT image
# (verified against nvcr.io/nvidia/tensorrt:25.08-py3 - configure silently
# skips the _pycamtrt target rather than failing loudly). Needed here since
# this image's Python CMD builds against the container's OWN python3, not
# a mounted conda env (contrast BUILD.md's dev workflow).

RUN python3 -m pip install --no-cache-dir numpy

WORKDIR /workspace

# Bring the farm up on the HOST first (it is not started by this
# container): cd tools/stream_farm && CLIP=media/atlas_plate_g30.mp4
# ./farm.sh up 2
#
# Run with (the farm listens on the host's localhost - this container
# needs --network host to reach it as rtsp://localhost:8554/camN):
#   docker run --rm --gpus all --network host \
#     -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
#     -v $(pwd):/workspace pycamtrt-read-plates
#
# libnvcuvid.so.1 is mounted in by the driver at CONTAINER run time (not
# present at image build time), so the unversioned symlink (BUILD.md trap
# #1) has to be (re-)created here, not in a RUN step above.
CMD ["bash", "-c", "\
    ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so || true; \
    mkdir -p .docker-build && cd .docker-build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python3) .. && \
    make -j$(nproc) _pycamtrt && cd .. && \
    PYTHONPATH=/workspace/.docker-build:/workspace/python python3 \
        examples/read_plates/read_plates.py \
        rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2 \
"]

# Alternative CMD: build + run the C++ port instead of the Python one
# (comment out the CMD above, uncomment this one):
# CMD ["bash", "-c", "\
#     ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so || true; \
#     mkdir -p .docker-build && cd .docker-build && \
#     cmake -DCMAKE_BUILD_TYPE=Release .. && \
#     make -j$(nproc) read_plates && cd .. && \
#     ./.docker-build/examples/read_plates \
#         rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2 \
# "]
