cmake_minimum_required(VERSION 3.18)

# BUILD TYPE DEFAULT (root-caused the hard way - see FINDINGS "the
# unoptimized-release incident", 2026-09-02): CMake's empty default build
# type compiles with NO optimization, and nothing in this project ever set
# one - the CORDERO dev checkout only got -O3 because a years-old manual
# -DCMAKE_BUILD_TYPE=Release was riding in its build cache. An empty-type
# build is functionally correct but up to ~45x slower on sparse-cadence
# cells. Default to Release; -DCMAKE_BUILD_TYPE=Debug still overrides.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# GPU arch default - must be set BEFORE enable_language(CUDA) probes and
# caches its own default, or the guard can never fire on a fresh configure.
# Override for your GPU: cmake -DCMAKE_CUDA_ARCHITECTURES=89 .. (BUILD.md).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(PyCamTRT VERSION 0.3.0)

enable_language(CUDA)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h 
    HINTS /usr/include/x86_64-linux-gnu 
          /usr/local/include
          /usr/include)

find_library(TENSORRT_LIB nvinfer 
    HINTS /usr/lib/x86_64-linux-gnu 
          /usr/local/lib
          /usr/lib)

# R3 engine auto-build: the ONNX parser ships with every TensorRT install
# alongside nvinfer - linked so TrtEngine can build+cache an .engine from
# a .onnx path directly (see src/EngineBuilder.h).
find_library(NVONNXPARSER_LIB nvonnxparser
    HINTS /usr/lib/x86_64-linux-gnu 
          /usr/local/lib
          /usr/lib)

# Print what we found for debugging
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "TENSORRT_INCLUDE_DIR: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TENSORRT_LIB: ${TENSORRT_LIB}")
message(STATUS "NVONNXPARSER_LIB: ${NVONNXPARSER_LIB}")

# (The CORDERO research repo's Step-1 `simple_inference` relic is not part
# of the library release - the earliest checkpoint here is rtsp_decode.)

# --- Step 3: RTSP decode ---

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libavutil)

add_executable(demuxer_test src/demuxer_test.cpp)
target_include_directories(demuxer_test PRIVATE src ${FFMPEG_INCLUDE_DIRS})
target_link_libraries(demuxer_test ${FFMPEG_LIBRARIES})
target_link_directories(demuxer_test PRIVATE ${FFMPEG_LIBRARY_DIRS})

# NVDEC GPU decode needs the Video Codec SDK headers/libs, which require a
# manual download (NVIDIA account) - see third_party/README and
# BUILD.md. Only builds rtsp_decode once it's in place, so its
# absence doesn't break the rest of the project.
set(VIDEO_CODEC_SDK_DIR "${CMAKE_SOURCE_DIR}/third_party/Video_Codec_SDK")

if(EXISTS "${VIDEO_CODEC_SDK_DIR}/Interface")
    message(STATUS "Video Codec SDK found at ${VIDEO_CODEC_SDK_DIR} - building rtsp_decode")

    find_library(NVCUVID_LIB nvcuvid HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib)
    find_library(CUDA_DRIVER_LIB cuda HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64/stubs /usr/lib)

    add_executable(rtsp_decode src/rtsp_decode.cpp src/NvDecoder.cpp)

    target_include_directories(rtsp_decode PRIVATE
        src
        ${VIDEO_CODEC_SDK_DIR}/Interface
        ${FFMPEG_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(rtsp_decode
        ${NVCUVID_LIB}
        ${CUDA_DRIVER_LIB}
        ${FFMPEG_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
    )
    target_link_directories(rtsp_decode PRIVATE ${FFMPEG_LIBRARY_DIRS})

    # Step 2 checkpoint: fused NV12 -> tensor preprocessing kernel, verified
    # against a real decoded frame (needs the decoder, hence gated with it).
    add_executable(preprocess_test
        src/preprocess_test.cpp
        src/NvDecoder.cpp
        src/preprocess.cu
    )

    target_include_directories(preprocess_test PRIVATE
        src
        ${VIDEO_CODEC_SDK_DIR}/Interface
        ${FFMPEG_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(preprocess_test
        ${NVCUVID_LIB}
        ${CUDA_DRIVER_LIB}
        ${FFMPEG_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
    )
    target_link_directories(preprocess_test PRIVATE ${FFMPEG_LIBRARY_DIRS})

    # Step 4: the full single-stream pipeline (decode -> preprocess -> infer
    # -> GPU postprocess).
    add_executable(rtsp_infer
        src/rtsp_infer.cpp
        src/NvDecoder.cpp
        src/preprocess.cu
        src/postprocess.cu
    )

    target_include_directories(rtsp_infer PRIVATE
        src
        ${VIDEO_CODEC_SDK_DIR}/Interface
        ${TENSORRT_INCLUDE_DIR}
        ${FFMPEG_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(rtsp_infer
        ${TENSORRT_LIB} ${NVONNXPARSER_LIB}
        ${NVCUVID_LIB}
        ${CUDA_DRIVER_LIB}
        ${FFMPEG_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
    )
    target_link_directories(rtsp_infer PRIVATE ${FFMPEG_LIBRARY_DIRS})

    # Part 1: libcordero - the multi-stream pipeline orchestration extracted
    # out of rtsp_infer_multi.cpp into a reusable library core (see
    # src/core/). Same gate as rtsp_infer_multi below since it needs the
    # same NVDEC/TensorRT/FFmpeg stack.
    add_library(cordero STATIC
        src/core/pipeline.cpp
        src/NvDecoder.cpp
        src/preprocess.cu
        src/postprocess.cu
    )

    # PIC even though cordero is a static lib linked into executables
    # (rtsp_infer_multi, pipeline_api_test): it's ALSO linked into the
    # _pycamtrt shared module (Part 3, below), which requires every
    # object it pulls in to be position-independent.
    set_target_properties(cordero PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(cordero PUBLIC
        src
        ${VIDEO_CODEC_SDK_DIR}/Interface
        ${TENSORRT_INCLUDE_DIR}
        ${FFMPEG_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
    )

    target_link_libraries(cordero PUBLIC
        ${TENSORRT_LIB} ${NVONNXPARSER_LIB}
        ${NVCUVID_LIB}
        ${CUDA_DRIVER_LIB}
        ${FFMPEG_LIBRARIES}
        ${CUDA_LIBRARIES}
        pthread
    )
    target_link_directories(cordero PUBLIC ${FFMPEG_LIBRARY_DIRS})

    # Step 5: multi-stream pipeline CLI - a thin consumer of libcordero
    # (build a PipelineConfig step graph from flags, Start()/Poll(), print).
    add_executable(rtsp_infer_multi src/rtsp_infer_multi.cpp)

    target_include_directories(rtsp_infer_multi PRIVATE src)

    target_link_libraries(rtsp_infer_multi cordero)

    # Part 2: lifecycle-robustness test for cordero::Pipeline (construct/
    # Start/Poll/Stop/destruct edge cases) - same gate as cordero/
    # rtsp_infer_multi above since it links the library directly.
    add_executable(pipeline_api_test src/pipeline_api_test.cpp)

    target_include_directories(pipeline_api_test PRIVATE src)

    target_link_libraries(pipeline_api_test cordero)

    # Part 3: pycamtrt - Python binding over libcordero (see
    # src/python/bindings.cpp). Same NVDEC/TensorRT/FFmpeg gate as cordero
    # above, ANDed with Python3 being found (a dev box without a Python
    # dev/interpreter still gets every other target). pybind11 is vendored
    # via FetchContent, pinned to v2.13.6 - it needs network access at
    # configure time (docker run --network host in this repo's build
    # command).
    find_package(Python3 COMPONENTS Interpreter Development)

    if(Python3_FOUND)
        message(STATUS "Python3 found (${Python3_VERSION} at ${Python3_EXECUTABLE}) - building _pycamtrt")

        # pybind11 defaults to its legacy FindPythonInterp/FindPythonLibs
        # discovery, which ignores Python3_EXECUTABLE and (in this repo's
        # docker image) picks up the container's own /usr/bin/python3
        # instead of the conda Python-dev interpreter the module actually
        # needs to import against. PYBIND11_FINDPYTHON routes it through
        # the modern FindPython3 module instead, which does respect the
        # hint cached above.
        set(PYBIND11_FINDPYTHON ON)

        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6
        )
        FetchContent_MakeAvailable(pybind11)

        pybind11_add_module(_pycamtrt src/python/bindings.cpp)

        target_include_directories(_pycamtrt PRIVATE src)

        target_link_libraries(_pycamtrt PRIVATE cordero)

        # pip packaging (scikit-build-core, see pyproject.toml): the compiled
        # module installs INSIDE the pycamtrt package. Harmless for the plain
        # in-container `make` workflow, which never runs the install step.
        install(TARGETS _pycamtrt LIBRARY DESTINATION pycamtrt)
    else()
        message(STATUS "Python3 not found - skipping _pycamtrt target")
    endif()
else()
    message(STATUS "Video Codec SDK not found at ${VIDEO_CODEC_SDK_DIR} - skipping rtsp_decode target")
endif()

# --- GPU postprocess (box decode + NMS on GPU) ---

# Part 1 checkpoint: box-decode kernel diffed against the CPU reference on the
# same engine output. Runs on a static image - needs TensorRT/OpenCV/CUDA
# only, not the Video Codec SDK or ffmpeg, so it lives outside the SDK gate.
add_executable(postprocess_test
    src/postprocess_test.cpp
    src/postprocess.cu
)

target_include_directories(postprocess_test PRIVATE
    src
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(postprocess_test
    ${TENSORRT_LIB} ${NVONNXPARSER_LIB}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)

# Step 5 checkpoint (part C): batched postprocess kernels vs the verified
# single-image kernels, bit-exact. Pure CUDA - synthetic tensors.
add_executable(postprocess_batch_test
    src/postprocess_batch_test.cpp
    src/postprocess.cu
)

target_include_directories(postprocess_batch_test PRIVATE
    src
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(postprocess_batch_test ${CUDA_LIBRARIES})

# Cascade checkpoint: batched NV12 crop+resize (stage-2 input prep) vs CPU
# reference, bit-exact. Pure CUDA - synthetic pitched NV12 frames.
add_executable(crop_test
    src/crop_test.cpp
    src/preprocess.cu
)

target_include_directories(crop_test PRIVATE
    src
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(crop_test ${CUDA_LIBRARIES})

# Step 5 checkpoint (part B): dynamic-batch engine vs static reference.
# Pure TensorRT + CUDA - synthetic tensors, no image/decode dependency.
add_executable(engine_batch_test src/engine_batch_test.cpp)

target_include_directories(engine_batch_test PRIVATE
    src
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(engine_batch_test
    ${TENSORRT_LIB} ${NVONNXPARSER_LIB}
    ${CUDA_LIBRARIES}
)

# Cascade checkpoint: LPRNet engine vs PyTorch reference dump + greedy CTC
# decode parity. TensorRT + CUDA only; reference bins exported alongside the LPRNet onnx (CORDERO repo).
add_executable(lprnet_test src/lprnet_test.cpp)

target_include_directories(lprnet_test PRIVATE
    src
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(lprnet_test
    ${TENSORRT_LIB} ${NVONNXPARSER_LIB}
    ${CUDA_LIBRARIES}
)

# fp16 checkpoint: fp16 vs fp32 stage-1 detection agreement on real
# images (accuracy preservation). TensorRT + CUDA + OpenCV; same GPU
# postprocess both engines, identical preprocessed input.
add_executable(compare_fp16
    src/compare_fp16.cpp
    src/postprocess.cu
)

target_include_directories(compare_fp16 PRIVATE
    src
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(compare_fp16
    ${TENSORRT_LIB} ${NVONNXPARSER_LIB}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)
# SAHI checkpoint: tile geometry + offset remap + cross-tile merge vs CPU
# reference. TensorRT-free - postprocess kernels + host geometry only.
add_executable(sahi_tiles_test
    src/sahi_tiles_test.cpp
    src/postprocess.cu
)

target_include_directories(sahi_tiles_test PRIVATE
    src
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(sahi_tiles_test ${CUDA_LIBRARIES})
