#!/usr/bin/env bash
# Ceiling campaign: how many cameras each system holds with full inference at
# 30 fps, per resolution. Plate cascade (yolov8n_plates fp16 -> LPRNet), the
# same real plate recording encoded at 720p / 1080p / 4K (see make_content()),
# same machine, same hour. Systems per cell: DeepStream 8.0 default, DeepStream
# 8.0 latency-tuned fp16, PyCamTRT (ceiling_pycamtrt.py), NVIDIA DIY
# (diy_pynvc_trt.py, detect only). "Holds" = sustains >=95% of the offered rate.
#
# Usage: ./run_ceiling_campaign.sh OUTDIR [720p,1080p,4k] [duration_s]
# Needs: mediamtx (tools/stream_farm/farm.sh), the tensorrt-dev image, the DS 8.0
# image, .diyenv (venv with tensorrt==10.12 + pynvvideocodec + av), and the 4K
# source recordings for make_content (SRC_DIR).
set -uo pipefail
export LC_ALL=C
OUTDIR=$(readlink -f "${1:?usage: run_ceiling_campaign.sh OUTDIR [res,res] [duration_s]}")
RESES=${2:-720p,1080p,4k}
DUR=${3:-60}
cd "$(dirname "$0")"; BENCH=$(pwd); REPO=$(cd .. && pwd)
FFMPEG="${FFMPEG:-ffmpeg}"; PY="${PYTHON3:-python3}"
MEDIA="$REPO/tools/stream_farm/media"
SRC_DIR="${SRC_DIR:-}"   # 4K source recordings for make_content(); only needed when plate_<res>_g30.mp4 is absent
SEG_START="${SEG_START:-0}"; SEG_LEN="${SEG_LEN:-60}"
declare -A DIMS=([720p]="1280:720" [1080p]="1920:1080" [4k]="3840:2160")
declare -A GRID=([720p]="${GRID_720P:-16 24 32 40 48}" [1080p]="${GRID_1080P:-8 16 20 24 28 32}" [4k]="${GRID_4K:-2 4 6 8}")
mkdir -p "$OUTDIR"; SUMMARY="$OUTDIR/cells.csv"
[ -f "$SUMMARY" ] || echo "res,system,N,offered_fps,achieved_fps,hold_pct,post_decode_ms,gpu_util_pct,note" > "$SUMMARY"

make_content() {  # $1 = res -> $MEDIA/plate_<res>_g30.mp4 (atlas recipe: concat sources, lanczos, x264 main, GOP 30, 2.5 Mbps)
    local res=$1 out="$MEDIA/plate_$1_g30.mp4"
    [ -f "$out" ] && { echo "content $out present"; return; }
    [ -n "$SRC_DIR" ] || { echo "SRC_DIR is not set and $out is missing: point SRC_DIR at the 4K source recordings" >&2; exit 1; }
    printf "file '%s'\n" "$SRC_DIR"/*.mp4 > "$MEDIA/ceiling_concat.txt"
    "$FFMPEG" -y -hide_banner -loglevel error -f concat -safe 0 -i "$MEDIA/ceiling_concat.txt" \
        -ss "$SEG_START" -t "$SEG_LEN" -vf "scale=${DIMS[$res]}:flags=lanczos" -an -c:v libx264 -profile:v main \
        -pix_fmt yuv420p -g 30 -keyint_min 30 -sc_threshold 0 -b:v 2500k -bf 0 "$out"
    echo "built $out"
}
farm_up() {  # $1 = res, $2 = N
    (cd "$REPO/tools/stream_farm" && ./farm.sh up 0 >/dev/null 2>&1)   # mediamtx only
    ./deepstream/plate_farm.sh down >/dev/null 2>&1; pkill -KILL -f "ff[m]peg.*rtsp://localhost:8554/plate" 2>/dev/null
    CLIP="$MEDIA/plate_$1_g30.mp4" ./deepstream/plate_farm.sh up "$2" >/dev/null
    sleep 6
}
gpu_mean() { tail -n +6 "$1" 2>/dev/null | awk -F', ' '{s+=$1;n++} END{if(n>0) printf "%.1f", s/n; else print "NA"}'; }

run_ds() {  # $1 res $2 N $3 default|tuned
    local n=$2 v=$3 cfg tag="$1_ds_${3}_N$2"
    if [[ $v == tuned ]]; then cfg="app_plate${n}_tuned.txt"; [[ -f deepstream/configs/$cfg ]] || (cd deepstream && ./gen_app_config_tuned.sh "$n" plate >/dev/null)
    else cfg="app_plate${n}.txt"; [[ -f deepstream/configs/$cfg ]] || (cd deepstream && ./gen_app_config.sh "$n" plate >/dev/null); fi
    (cd deepstream && ./sample_gpu.sh "$OUTDIR/gpu_$tag.csv" $((DUR + 8))) & local gp=$!
    (cd deepstream && ./run_bench.sh "$cfg" "$DUR" --latency) > "$OUTDIR/run_$tag.log" 2>&1
    wait $gp 2>/dev/null
    "$PY" deepstream/parse_latency.py --warmup 60 < "$OUTDIR/run_$tag.log" > "$OUTDIR/latency_$tag.txt" 2>&1
    # per-source average fps from the last PERF line: "**PERF:  cur (avg)\tcur (avg)..."
    local perf; perf=$(grep "PERF:" "$OUTDIR/run_$tag.log" | tail -1 | grep -oE "\(([0-9.]+)\)" | tr -d '()' | awk '{s+=$1;n++} END{if(n>0) printf "%.2f", s; else print "0"}')
    local post; post=$(grep -E "^ ALL" "$OUTDIR/latency_$tag.txt" | awk '{printf "%.2f", $4+$5}')
    local offered=$((n * 30)); local hold; hold=$(awk -v a="$perf" -v o="$offered" 'BEGIN{printf "%.1f", 100*a/o}')
    echo "$1,deepstream_$v,$n,$offered,$perf,$hold,${post:-NA},$(gpu_mean "$OUTDIR/gpu_$tag.csv")," >> "$SUMMARY"
    echo "    DS $v N=$n: $perf fps of $offered ($hold%), post-decode ${post:-NA} ms"
}
run_pycamtrt() {  # $1 res $2 N
    local tag="$1_pycamtrt_N$2"
    (cd deepstream && ./sample_gpu.sh "$OUTDIR/gpu_$tag.csv" $((DUR + 20))) & local gp=$!
    docker run --rm --gpus all --network host -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
        -v "$REPO":/workspace -v "$HOME/anaconda3":"$HOME/anaconda3" -w /workspace tensorrt-dev bash -c "
        apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq pkg-config libavformat-dev libavcodec-dev libavutil-dev >/dev/null 2>&1;
        ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so;
        PYTHONPATH=/workspace/build:/workspace/python $CONTAINER_PY bench/ceiling_pycamtrt.py /workspace/$(realpath --relative-to="$REPO" "$OUTDIR")/pycamtrt_cells.csv $2 plate $DUR" > "$OUTDIR/run_$tag.log" 2>&1
    wait $gp 2>/dev/null
    local row n off dec hold res pre q g post drop; row=$(grep "^CELL" "$OUTDIR/run_$tag.log" | tail -1 | cut -d' ' -f2)
    IFS=, read -r n off dec hold res pre q g post drop <<< "$row"
    echo "$1,pycamtrt,$2,${off:-},${dec:-0},${hold:-0},${post:-NA},$(gpu_mean "$OUTDIR/gpu_$tag.csv"),dropped=${drop:-?}" >> "$SUMMARY"
    echo "    PyCamTRT N=$2: ${dec:-0} fps of ${off:-?} (${hold:-0}%), post-decode ${post:-NA} ms"
}
run_diy() {  # $1 res $2 N  (host, .diyenv; detect only)
    local tag="$1_diy_N$2" frames=$((DUR * 30))
    (cd deepstream && ./sample_gpu.sh "$OUTDIR/gpu_$tag.csv" $((DUR + 20))) & local gp=$!
    (cd "$REPO" && timeout $((DUR * 3)) .diyenv/bin/python bench/diy_pynvc_trt.py "$OUTDIR/diy_cells.csv" "$2" plate "$frames" models/yolov8n_plates_b1-48_fp16_sm86.engine) > "$OUTDIR/run_$tag.log" 2>&1
    wait $gp 2>/dev/null
    local row c n fr nres wall pre inf post e2e steady; row=$(grep "^CELL" "$OUTDIR/run_$tag.log" | tail -1 | cut -d' ' -f2)
    IFS=, read -r c n fr nres wall pre inf post e2e steady <<< "$row"
    local offered=$(( $2 * 30 )); local ach="${steady:-0}"
    local hold; hold=$(awk -v a="$ach" -v o="$offered" 'BEGIN{printf "%.1f", 100*a/o}')
    echo "$1,diy_detect_only,$2,$offered,$ach,$hold,${e2e:-NA},$(gpu_mean "$OUTDIR/gpu_$tag.csv"),steady_state" >> "$SUMMARY"
    echo "    DIY N=$2: $ach fps of $offered ($hold%), e2e ${e2e:-NA} ms"
}

CONTAINER_PY="${CONTAINER_PY:-$HOME/anaconda3/envs/Python-dev/bin/python3}"
for res in ${RESES//,/ }; do
    make_content "$res"
    grid=${GRID[$res]}; nmax=${grid##* }
    echo "=== $res: farm up with $nmax publishers"; farm_up "$res" "$nmax"
    for n in $grid; do
        echo ">>> $res N=$n"
        run_ds "$res" "$n" default
        run_ds "$res" "$n" tuned
        run_pycamtrt "$res" "$n"
        run_diy "$res" "$n"
    done
done
./deepstream/plate_farm.sh down >/dev/null 2>&1; pkill -KILL -f "ff[m]peg.*rtsp://localhost:8554/plate" 2>/dev/null
echo "=== campaign complete: $SUMMARY ==="
