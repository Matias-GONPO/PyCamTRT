# Builds the two custom nvinfer parser libraries. Run INSIDE the
# DeepStream 8.0 container with the repo mounted at /workspace:
#   cd /workspace/bench/deepstream/parsers && make
DS_PATH ?= /opt/nvidia/deepstream/deepstream
CUDA_PATH ?= /usr/local/cuda

CXX := g++
CXXFLAGS := -Wall -std=c++17 -shared -fPIC \
    -I$(DS_PATH)/sources/includes -I$(CUDA_PATH)/include \
    -I../../../src
# No TRT/CUDA libs needed: parsers are pure CPU code over nvinfer's
# buffers; nvinfer dlopens the .so and resolves nothing else.
LDFLAGS :=

all: libyolov8_parser.so liblprnet_parser.so

libyolov8_parser.so: yolov8_parser.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

liblprnet_parser.so: lprnet_parser.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f libyolov8_parser.so liblprnet_parser.so
