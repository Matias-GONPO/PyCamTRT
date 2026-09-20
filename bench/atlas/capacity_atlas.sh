#!/usr/bin/env bash
# THE CAPACITY ATLAS: for each (N cameras, decode rate, inference rate) cell, the timing split,
# hold rate, NVDEC/SM utilisation, VRAM, plates read and a bottleneck label, full plate cascade
# (detector + LPRNet) in every cell, measured with build/rtsp_infer_multi inside tensorrt-dev.
#
# Usage: bench/atlas/capacity_atlas.sh <res> <outdir> [full|mini|smoke]
#   full  = the 336-cell grid of the v2 atlas (N 1..100 x 5 decode rates x 14 skips, dead cells pruned)
#   mini  = the validation grid (every-frame cells around the ceiling + keyframe anchors), ~30 cells
#   smoke = one cell
# RESUME: a cell already in <outdir>/atlas.csv (keyed N,decode,skip) is skipped, so re-running the
# same command continues where it stopped. Env: WALL (target seconds per cell, 40), CELL_CAP (150),
# NS / DECODES / SKIPS override the grid of any mode.
set -u; export LC_ALL=C
RES=${1:?usage: capacity_atlas.sh <res> <outdir> [full|mini|smoke]}; OUTDIR=${2:?}; MODE=${3:-full}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); cd "$REPO"; mkdir -p "$OUTDIR"
S1="${ATLAS_S1:-models/yolov8n_plates_b1-48_fp16_sm86.engine}"
OCR="${ATLAS_OCR:-models/lprnet_b1-32_fp32_sm86.engine}"
CONTAINER_IMAGE="${IMAGE:-tensorrt-dev}"
WALL=${WALL:-40}; CELL_CAP=${CELL_CAP:-150}

case "$MODE" in
  smoke) NS="${NS:-8}"; DECODES="${DECODES:-all}"; SKIPS="${SKIPS:-1}"; WALL=${WALL_SMOKE:-15};;
  mini)
    # validation grid: the every-frame cells the v0.4.0 decoder fix moved, plus keyframe and
    # single-camera anchors; per-resolution N lists stop short of the VRAM wall (v2 atlas)
    case "$RES" in
      720p)  NS_ALL="1 8 16 32 50 75"; NS_KEY30="32 50 75 100"; NS_KEY150="100";;
      1080p) NS_ALL="1 8 16 32 50";    NS_KEY30="32 50 75 100"; NS_KEY150="100";;
      1440p) NS_ALL="1 8 16 32";       NS_KEY30="32 50 75";     NS_KEY150="100";;
      4k)    NS_ALL="1 4 8 16";        NS_KEY30="16 32";        NS_KEY150="32";;
      *) echo "bad res $RES"; exit 1;;
    esac
    NS="${NS:-}"; DECODES="${DECODES:-all key30 key150}"; SKIPS="${SKIPS:-1 2 3 6 15}";;
  full)
    NS="${NS:-1 4 8 16 32 50 75 100}"; DECODES="${DECODES:-all key15 key30 key60 key150}"
    SKIPS="${SKIPS:-1 2 3 4 6 8 12 15 24 30 45 60 90 150}";;
  *) echo "bad mode $MODE"; exit 1;;
esac

CSV="$OUTDIR/atlas.csv"
HDR="N,decode,gop,inf_per_s_per_stream,skip,decoded_per_s,offered_per_s,hold_pct,pre_ms,queue_ms,gpu_ms,total_ms,nvdec_pct,sm_pct,vram_mb,plates,bottleneck"
[ -f "$CSV" ] || echo "$HDR" > "$CSV"
done_cell() { grep -q "^${1},${2},[^,]*,[^,]*,${3}," "$CSV" 2>/dev/null; }
decode_rate() { case $1 in all) echo 30;; key15) echo 2;; key30) echo 1;; key60) echo 0.5;; key150) echo 0.2;; esac; }
decode_gop()  { case $1 in all) echo 30;; key15) echo 15;; key30) echo 30;; key60) echo 60;; key150) echo 150;; esac; }
decode_flag() { case $1 in all) echo "all";; *) echo "key";; esac; }
classify() { awk -v h="$1" -v nd="$2" -v sm="$3" -v pre="$4" -v q="$5" 'BEGIN{
    if (h+0 >= 95) { print "none(offered-bound)"; exit }
    if (nd+0 >= 80) { print "NVDEC"; exit }
    if (sm+0 >= 80) { print "GPU-compute"; exit }
    if (pre+0 >= 8 && pre+0 > q+0) { print "producer-threads"; exit }
    if (q+0 >= 8) { print "batch-queue"; exit }
    print "mixed/ramp" }'; }
cells_for() {  # decode token -> N list for this mode
  if [ "$MODE" = mini ] && [ -z "$NS" ]; then
    case $1 in all) echo "$NS_ALL";; key30) echo "$NS_KEY30";; key150) echo "$NS_KEY150";; *) echo "";; esac
  else echo "$NS"; fi
}
echo "=== capacity atlas $RES ($MODE) -> $CSV ==="
for dec in $DECODES; do
  gop=$(decode_gop "$dec"); dflag=$(decode_flag "$dec"); drate=$(decode_rate "$dec")
  ns=$(cells_for "$dec"); [ -z "$ns" ] && continue
  maxn=$(echo $ns | tr ' ' '\n' | sort -n | tail -1); farm_up=0
  for n in $ns; do
    for skip in $SKIPS; do
      inf=$(awk -v d="$drate" -v s="$skip" 'BEGIN{printf "%.4f", d/s}')
      [ "$(awk -v i="$inf" 'BEGIN{print (i < 0.045) ? 1 : 0}')" = 1 ] && continue   # interval > ~22 s: dead cell
      [ "$MODE" = mini ] && [ "$n" = 1 ] && [ "$skip" != 1 ] && continue              # single camera: skip 1 only
      [ "$MODE" = mini ] && [ "$dec" != all ] && [ "$skip" != 1 ] && continue          # keyframe anchors: skip 1 only
      done_cell "$n" "$dec" "$skip" && continue
      if [ "$farm_up" = 0 ]; then "$HERE/atlas_farm.sh" "$RES" "$gop" "$maxn" > "$OUTDIR/farm_${dec}.log" 2>&1; farm_up=1; fi
      frames=$(awk -v d="$drate" -v w="$WALL" 'BEGIN{f=int(d*w); if(f<8)f=8; print f}')
      tag="${dec}_n${n}_s${skip}"; urls=""; for i in $(seq 1 "$n"); do urls="$urls rtsp://127.0.0.1:8554/cam$i"; done
      nvidia-smi dmon -s u -d 1 -c $((WALL+8)) > "$OUTDIR/dmon_$tag.txt" 2>&1 & DP=$!
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 -f "$OUTDIR/vram_$tag.csv" > /dev/null 2>&1 < /dev/null & VP=$!
      CN="atlascell_$$_${tag}"
      timeout --signal=KILL "$CELL_CAP" docker run --rm --name "$CN" --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video --network host \
        -v "$REPO":/workspace -w /workspace "$CONTAINER_IMAGE" bash -c \
        "ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so 2>/dev/null; ./build/rtsp_infer_multi --engine $S1 --ocr $OCR --decode $dflag --skip $skip --frames $frames $urls" \
        > "$OUTDIR/run_$tag.log" 2>&1 || true
      docker kill "$CN" >/dev/null 2>&1 || true
      kill $DP $VP 2>/dev/null || true; while kill -0 $VP 2>/dev/null; do sleep 0.2; done
      line=$(grep -E "inferred of .* decoded / .* wall" "$OUTDIR/run_$tag.log" | tail -1)
      decps=$(echo "$line" | grep -oP '[0-9.]+(?= decoded/s)' | tail -1)
      sp=$(grep "latency split" "$OUTDIR/run_$tag.log" | grep -oP '[0-9.]+(?= ms)' | head -3)
      pre=$(echo "$sp" | sed -n 1p); q=$(echo "$sp" | sed -n 2p); g=$(echo "$sp" | sed -n 3p)
      tot=$(awk -v a="${pre:-0}" -v b="${q:-0}" -v c="${g:-0}" 'BEGIN{printf "%.2f", a+b+c}')
      plates=$(grep -oP 'ocr: \K[0-9]+' "$OUTDIR/run_$tag.log" | head -1)
      offered=$(awk -v d="$drate" -v n="$n" 'BEGIN{printf "%.1f", d*n}')
      hold=$(awk -v dp="${decps:-0}" -v of="$offered" 'BEGIN{if(of>0)printf "%.1f", 100*dp/of; else print 0}')
      nvdec=$(grep -E "^ " "$OUTDIR/dmon_$tag.txt" | tail -n +4 | awk '{print $5}' | sort -n | awk '{a[NR]=$1}END{print a[int(NR/2)+1]+0}')
      smu=$(grep -E "^ " "$OUTDIR/dmon_$tag.txt" | tail -n +4 | awk '{print $2}' | sort -n | awk '{a[NR]=$1}END{print a[int(NR/2)+1]+0}')
      vram=$(tail -n +3 "$OUTDIR/vram_$tag.csv" 2>/dev/null | awk '$1>m{m=$1}END{print m+0}')
      bott=$(classify "${hold:-0}" "${nvdec:-0}" "${smu:-0}" "${pre:-0}" "${q:-0}")
      grep -q "OUT_OF_MEMORY" "$OUTDIR/run_$tag.log" && bott="VRAM-OOM"
      echo "$n,$dec,$gop,$inf,$skip,${decps:-0},$offered,${hold:-0},${pre:-0},${q:-0},${g:-0},$tot,${nvdec:-0},${smu:-0},${vram:-0},${plates:-0},$bott" >> "$CSV"
      echo "$(date +%T) [$RES $tag] hold ${hold}% inf ${inf}/s/str split ${pre}/${q}/${g} nvdec ${nvdec}% sm ${smu}% -> $bott"
    done
  done
done
echo "=== capacity atlas $RES complete: $CSV ($(($(wc -l < "$CSV")-1)) cells) ==="
