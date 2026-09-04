# PyCamTRT build/run environment.
#
# Any image with CUDA 12.x + TensorRT 10.x works (the project was developed
# on a nvidia/cuda:12.6.3-devel base with TensorRT 10.12 installed); NVIDIA's
# TensorRT containers are the zero-thought route. See BUILD.md for the full
# story, including the Video Codec SDK download this image can NOT include
# (NVIDIA's license forbids redistributing it).
FROM nvcr.io/nvidia/tensorrt:25.08-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config \
        libopencv-dev \
        libavformat-dev libavcodec-dev libavutil-dev \
    && rm -rf /var/lib/apt/lists/* \
    # The container ships libnvcuvid.so.1 (via the driver mount) but no
    # unversioned symlink for the linker - create it (BUILD.md trap #1).
    && ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so || true

RUN python3 -m pip install --no-cache-dir numpy

WORKDIR /workspace
