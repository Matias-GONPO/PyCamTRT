#!/usr/bin/env bash
# DeepStream on the atlas dials grid: same farm and GOP content as bench/atlas (cam1..N), latency-tuned
# fp16 config, decode=all with nvinfer interval {0,1,2,5,14} (= skip 1,2,3,6,15) and keyframes-only
# (intra-decode-enable) at skip 1, the same N lists as the atlas `mini` grid. Resume-capable.
# Usage: bench/deepstream/ds_dials.sh <res> <outdir> [WALL seconds, 40]
set -u; export LC_ALL=C
RES=${1:?usage: ds_dials.sh <res> <outdir> [wall]}; OUTDIR=${2:?}; WALL=${3:-40}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); cd "$REPO"; mkdir -p "$OUTDIR"
case "$RES" in
  720p)  NS_ALL="1 8 16 32 50 75"; NS_KEY="32 50 75 100";;
  1080p) NS_ALL="1 8 16 32 50";    NS_KEY="32 50 75 100";;
  1440p) NS_ALL="1 8 16 32";       NS_KEY="32 50 75";;
  4k)    NS_ALL="1 4 8 16";        NS_KEY="16 32";;
  *) echo "bad res $RES"; exit 1;;
esac
SKIPS="1 2 3 6 15"
CSV="$OUTDIR/ds_dials.csv"
[ -f "$CSV" ] || echo "N,decode,gop,inf_per_s_per_stream,skip,achieved_fps,offered_per_s,hold_pct,post_decode_ms,nvdec_pct,sm_pct,note" > "$CSV"
done_cell() { grep -q "^${1},${2},[^,]*,[^,]*,${3}," "$CSV" 2>/dev/null; }
run_cell() {  # N decode skip
  local n=$1 dec=$2 skip=$3 intra=0 interval=$((skip - 1)) drate=30 gop=30
  [ "$dec" = key30 ] && { intra=1; interval=0; drate=1; }
  local inf; inf=$(awk -v d="$drate" -v s="$skip" 'BEGIN{printf "%.4f", d/s}')
  local tag="${dec}_n${n}_s${skip}" cfg="app_cam${n}_tuned_i${interval}_k${intra}.txt"
  [ -f "$HERE/configs/$cfg" ] || "$HERE/gen_app_config_dials.sh" "$n" cam "$interval" "$intra" >/dev/null
  nvidia-smi dmon -s u -d 1 -c $((WALL+12)) > "$OUTDIR/dmon_$tag.txt" 2>&1 & local DP=$!
  (cd "$HERE" && timeout --signal=KILL $((WALL + 240)) ./run_bench.sh "$cfg" "$WALL" --latency) > "$OUTDIR/run_$tag.log" 2>&1
  kill $DP 2>/dev/null; wait $DP 2>/dev/null
  python3 "$HERE/parse_latency.py" --warmup 60 < "$OUTDIR/run_$tag.log" > "$OUTDIR/latency_$tag.txt" 2>&1
  local perf post offered hold nvdec smu
  perf=$(grep "PERF:" "$OUTDIR/run_$tag.log" | tail -1 | grep -oE "\(([0-9.]+)\)" | tr -d '()' | awk '{s+=$1;c++} END{if(c>0) printf "%.2f", s; else print "0"}')
  post=$(grep -E "^ ALL" "$OUTDIR/latency_$tag.txt" | awk '{printf "%.2f", $4+$5}')
  offered=$(awk -v d="$drate" -v n="$n" 'BEGIN{printf "%.1f", d*n}'); hold=$(awk -v a="$perf" -v o="$offered" 'BEGIN{printf "%.1f", 100*a/o}')
  nvdec=$(grep -E "^ " "$OUTDIR/dmon_$tag.txt" | tail -n +8 | awk '{print $5}' | sort -n | awk '{a[NR]=$1}END{print a[int(NR/2)+1]+0}')
  smu=$(grep -E "^ " "$OUTDIR/dmon_$tag.txt" | tail -n +8 | awk '{print $2}' | sort -n | awk '{a[NR]=$1}END{print a[int(NR/2)+1]+0}')
  local note=""; [ "$skip" != 1 ] && note="latency mixes inferred and skipped frames"
  echo "$n,$dec,$gop,$inf,$skip,$perf,$offered,$hold,${post:-NA},${nvdec:-0},${smu:-0},$note" >> "$CSV"
  echo "$(date +%T) [$RES ds $tag] $perf of $offered ($hold%) post ${post:-NA} ms nvdec ${nvdec}% sm ${smu}%"
}
echo "=== DeepStream dials $RES -> $CSV ==="
maxn=$(echo $NS_ALL | tr ' ' '\n' | sort -n | tail -1); farm=0
for n in $NS_ALL; do for skip in $SKIPS; do
  [ "$n" = 1 ] && [ "$skip" != 1 ] && continue
  done_cell "$n" all "$skip" && continue
  [ "$farm" = 0 ] && { bench/atlas/atlas_farm.sh "$RES" 30 "$maxn" > "$OUTDIR/farm_all.log" 2>&1; farm=1; }
  run_cell "$n" all "$skip"
done; done
maxn=$(echo $NS_KEY | tr ' ' '\n' | sort -n | tail -1); farm=0
for n in $NS_KEY; do
  done_cell "$n" key30 1 && continue
  [ "$farm" = 0 ] && { bench/atlas/atlas_farm.sh "$RES" 30 "$maxn" > "$OUTDIR/farm_key.log" 2>&1; farm=1; }
  run_cell "$n" key30 1
done
echo "=== DeepStream dials $RES complete: $CSV ($(($(wc -l < "$CSV")-1)) cells) ==="
