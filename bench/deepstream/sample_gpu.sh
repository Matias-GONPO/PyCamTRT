#!/usr/bin/env bash
# Samples GPU/VRAM utilization at 1s resolution to a CSV. Run alongside
# run_bench.sh (backgrounded) so the sample window covers a live run.
# Usage: ./sample_gpu.sh OUTFILE DURATION_S
set -euo pipefail
OUT=${1:?usage: sample_gpu.sh OUTFILE DURATION_S}
DUR=${2:-60}
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
    -l 1 -f "$OUT" &
PID=$!
sleep "$DUR"
kill "$PID" 2>/dev/null || true
