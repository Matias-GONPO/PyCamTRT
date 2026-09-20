#!/usr/bin/env bash
# One-command atlas at one resolution: builds the GOP content if missing, then runs capacity_atlas.sh.
# Usage: bench/atlas/run_atlas.sh <720p|1080p|1440p|4k> [full|mini|smoke] [outdir]
# Content: tools/stream_farm/media/atlas_<res>_g{15,30,60,150}.mp4. The g30 clip is the same recipe as
# the ceiling campaign's plate_<res>_g30.mp4 (concat of the 4K plate recordings in SRC_DIR, lanczos,
# x264 main, 2.5 Mbps, no B-frames); the other GOPs are transcoded from it. Needs ffmpeg (FFMPEG env).
set -euo pipefail; export LC_ALL=C
RES=${1:?usage: run_atlas.sh <res> [full|mini|smoke] [outdir]}; MODE=${2:-full}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); cd "$REPO"
OUTDIR=${3:-bench/atlas/runs/$(date +%Y-%m-%d)_${MODE}/${RES}}
FFMPEG="${FFMPEG:-ffmpeg}"; MEDIA=tools/stream_farm/media
case "$RES" in 720p) DIMS="1280:720";; 1080p) DIMS="1920:1080";; 1440p) DIMS="2560:1440";; 4k) DIMS="3840:2160";; *) echo "bad res $RES"; exit 1;; esac
if [ ! -f "$MEDIA/atlas_${RES}_g30.mp4" ]; then
  if [ -f "$MEDIA/plate_${RES}_g30.mp4" ]; then cp "$MEDIA/plate_${RES}_g30.mp4" "$MEDIA/atlas_${RES}_g30.mp4"
  else
    : "${SRC_DIR:?set SRC_DIR to the directory of 4K source recordings to build atlas_${RES}_g30.mp4}"
    printf "file '%s'\n" "$SRC_DIR"/*.mp4 > "$MEDIA/atlas_concat_${RES}.txt"
    "$FFMPEG" -y -hide_banner -loglevel error -f concat -safe 0 -i "$MEDIA/atlas_concat_${RES}.txt" -ss 0 -t 60 \
      -vf "scale=${DIMS}:flags=lanczos" -an -c:v libx264 -profile:v main -pix_fmt yuv420p -g 30 -keyint_min 30 -sc_threshold 0 -b:v 2500k -bf 0 "$MEDIA/atlas_${RES}_g30.mp4"
  fi
fi
for g in 15 60 150; do
  [ -f "$MEDIA/atlas_${RES}_g${g}.mp4" ] || "$FFMPEG" -y -hide_banner -loglevel error -i "$MEDIA/atlas_${RES}_g30.mp4" \
    -c:v libx264 -profile:v main -pix_fmt yuv420p -g $g -keyint_min $g -sc_threshold 0 -b:v 2500k -bf 0 "$MEDIA/atlas_${RES}_g${g}.mp4"
done
echo ">>> content ready: $(ls "$MEDIA"/atlas_${RES}_g*.mp4 | wc -l) GOP variants at $RES"
"$HERE/capacity_atlas.sh" "$RES" "$OUTDIR" "$MODE"
echo ">>> analyze: python3 bench/atlas/atlas_analyze.py $OUTDIR/atlas.csv ; compare: python3 bench/atlas/atlas_compare.py bench/atlas/atlas_${RES}_v2.csv $OUTDIR/atlas.csv"
