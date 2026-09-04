#!/usr/bin/env bash
# Runs the full comparison sweep: N=1/4/8/16 x {plate, cam} content x
# {default, tuned} DeepStream configs, capturing throughput, latency
# splits, and GPU utilization for each cell.
#
# Usage: ./run_sweep.sh OUTDIR [duration_s] [default|tuned|both]
#
# Content contract (see ../METHODOLOGY.md): the farm must already be
# publishing - cam = webcam clip (farm.sh up 16), plate = the real
# temporal plate clip (CLIP=media/atlas_plate_g30.mp4 farm.sh up 16).
# The old looped-still plate_test.mp4 gave keyframe-burst decode
# artifacts and was discarded; do not resurrect it.
#
# Engine builds: DeepStream builds its TRT engines on FIRST use of each
# GIE config (written next to the .onnx under models/, root-owned). A
# short cell duration would SIGINT mid-build, so any missing engine gets
# a one-time WARM run (generous fixed duration) before its variant's
# cells - subsequent cells load the cache instantly.
set -euo pipefail
export LC_ALL=C
OUTDIR=${1:?usage: run_sweep.sh OUTDIR [duration_s] [default|tuned|both]}
DUR=${2:-60}
MODE=${3:-both}
WARM_DUR=420
cd "$(dirname "$0")"
mkdir -p "$OUTDIR"
MODELS="$(cd ../../models && pwd)"

warm_if_needed() {  # $1 = variant
    local pgie_engine
    if [[ "$1" == tuned ]]; then
        pgie_engine="$MODELS/yolov8n_plates_dynamic.onnx_b16_gpu0_fp16.engine"
    else
        pgie_engine="$MODELS/yolov8n_plates_dynamic.onnx_b16_gpu0_fp32.engine"
    fi
    if [[ ! -f "$pgie_engine" ]]; then
        echo ">>> [$1] engine cache missing - one-time warm run (${WARM_DUR}s)"
        local cfg="app_cam1.txt"
        [[ "$1" == tuned ]] && cfg="app_cam1_tuned.txt"
        ./run_bench.sh "$cfg" "$WARM_DUR" > "$OUTDIR/warm_$1.log" 2>&1 || true
    fi
}

for variant in default tuned; do
    [[ "$MODE" == both || "$MODE" == "$variant" ]] || continue
    for content in plate cam; do
        for n in 1 4 8 16; do
            if [[ "$variant" == tuned ]]; then
                cfg="app_${content}${n}_tuned.txt"
                [[ -f "configs/$cfg" ]] || ./gen_app_config_tuned.sh "$n" "$content"
            else
                cfg="app_${content}${n}.txt"
                [[ -f "configs/$cfg" ]] || ./gen_app_config.sh "$n" "$content"
            fi
            [[ -f "configs/$cfg" ]] || { echo "skip missing configs/$cfg"; continue; }
            warm_if_needed "$variant"
            tag="${content}${n}_${variant}"
            echo ">>> $content N=$n [$variant]"
            ./sample_gpu.sh "$OUTDIR/gpu_${tag}.csv" $((DUR + 5)) &
            gpu_pid=$!
            ./run_bench.sh "$cfg" "$DUR" --latency > "$OUTDIR/run_${tag}.log" 2>&1
            wait "$gpu_pid" 2>/dev/null || true

            frames=$(grep -c "Frame latency" "$OUTDIR/run_${tag}.log" || true)
            echo "    frames captured: $frames"
            python3 parse_latency.py --warmup 60 < "$OUTDIR/run_${tag}.log" \
                > "$OUTDIR/latency_${tag}.txt" 2>&1 || true
            tail -1 "$OUTDIR/latency_${tag}.txt"

            util=$(tail -n +11 "$OUTDIR/gpu_${tag}.csv" 2>/dev/null | \
                   awk -F', ' '{s+=$1;n++} END{if(n>0) printf "%.1f", s/n; else print "NA"}')
            echo "    gpu_util_mean=${util}%"
            echo "$util" > "$OUTDIR/gpu_util_${tag}.txt"

            grep "PERF:" "$OUTDIR/run_${tag}.log" | tail -3 \
                > "$OUTDIR/perf_tail_${tag}.txt" || true
        done
    done
done
echo "=== sweep complete: $OUTDIR ==="
