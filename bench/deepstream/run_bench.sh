#!/usr/bin/env bash
# Runs deepstream-app on a generated config inside the DS 8.0 container.
# Usage: ./run_bench.sh <config-in-configs/> [duration_s] [--latency]
#   duration_s  stop (SIGINT, graceful) after this many seconds; 0 = run
#               until Ctrl-C (default 60)
#   --latency   enable per-buffer component latency measurement
# Run from anywhere on the HOST; mounts the repo at /workspace like
# tensorrt-dev. Farm must already be publishing the streams.
# stdbuf -oL: deepstream-app block-buffers stdout without a TTY and the
# PERF lines never reach `docker logs` — line-buffering fixes it.
set -euo pipefail
CFG=${1:?usage: run_bench.sh <config> [duration_s] [--latency]}
DURATION=${2:-60}
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_LAT=()
if [[ "${3:-}" == "--latency" || "${2:-}" == "--latency" ]]; then
    [[ "${2:-}" == "--latency" ]] && DURATION=60
    ENV_LAT=(-e NVDS_ENABLE_LATENCY_MEASUREMENT=1
             -e NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1)
fi

RUNNER="stdbuf -oL -eL deepstream-app -c '$CFG' -t"
if [[ "$DURATION" != "0" ]]; then
    RUNNER="timeout -s INT --preserve-status $DURATION $RUNNER"
fi

# --entrypoint bash: the stock entrypoint expands $@ unquoted and
# word-splits a `bash -c "..."` command string.
exec docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -e DS_BENCH_LOG_PLATES \
    "${ENV_LAT[@]}" \
    -v "$REPO":/workspace \
    -w /workspace/bench/deepstream/configs \
    --entrypoint bash \
    nvcr.io/nvidia/deepstream:8.0-triton-multiarch \
    -c "$RUNNER"
