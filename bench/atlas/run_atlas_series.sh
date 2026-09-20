#!/usr/bin/env bash
# Run the atlas at several resolutions back to back (resume-capable per resolution).
# Usage: bench/atlas/run_atlas_series.sh <full|mini> <res> [<res> ...]     e.g. full 720p 1080p
# Full grid: 336 cells per resolution, about 4 to 5 hours each. Output: bench/atlas/runs/<date>_<mode>/<res>/atlas.csv
set -u; export LC_ALL=C
MODE=${1:?usage: run_atlas_series.sh <full|mini> <res>...}; shift
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); cd "$REPO"
RUN="bench/atlas/runs/$(date +%Y-%m-%d)_${MODE}"; mkdir -p "$RUN"
for res in "$@"; do
  echo "=== $(date +%T) $res ($MODE)"
  "$HERE/run_atlas.sh" "$res" "$MODE" "$RUN/$res" 2>&1 | grep -E "^[0-9:]+ \[|complete|content ready|rror"
  [ -f "bench/atlas/atlas_${res}_v2.csv" ] && python3 "$HERE/atlas_compare.py" "bench/atlas/atlas_${res}_v2.csv" "$RUN/$res/atlas.csv" > "$RUN/$res/compare_v2.txt" 2>&1
done
(cd tools/stream_farm && ./farm.sh down >/dev/null 2>&1)
echo "$(date +%T) ATLAS SERIES DONE: $RUN"
