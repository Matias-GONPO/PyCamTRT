#!/usr/bin/env bash
# Serve N publishers of one atlas GOP variant on rtsp://localhost:8554/cam1..camN.
# Usage: bench/atlas/atlas_farm.sh <res> <gop:15|30|60|150> <N>
# Clip: tools/stream_farm/media/atlas_<res>_g<gop>.mp4 (built by run_atlas.sh).
# Tears down existing cam publishers first; leaves mediamtx running.
set -u; export LC_ALL=C
RES=${1:?usage: atlas_farm.sh <res> <gop> <N>}; GOP=${2:?}; N=${3:?}
cd "$(dirname "$0")/../../tools/stream_farm"
FFMPEG="${FFMPEG:-ffmpeg}"
CLIP="media/atlas_${RES}_g${GOP}.mp4"; PIDDIR="run"; mkdir -p "$PIDDIR"
[ -f "$CLIP" ] || { echo "no such clip: $CLIP (run bench/atlas/run_atlas.sh $RES to build it)"; exit 1; }
if [ ! -f "$PIDDIR/mediamtx.pid" ] || ! kill -0 "$(cat "$PIDDIR/mediamtx.pid" 2>/dev/null)" 2>/dev/null; then
    ./bin/mediamtx ./bin/mediamtx.yml > "$PIDDIR/mediamtx.log" 2>&1 &
    echo $! > "$PIDDIR/mediamtx.pid"; sleep 1
fi
for f in "$PIDDIR"/cam*.pid; do [ -f "$f" ] || continue; kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; done
# stray publishers: match the ffmpeg command line itself (clip path + rtsp output), nothing else
for p in $(pgrep -f "media/atlas_[0-9a-z]+_g[0-9]+[.]mp4 -c copy -f rtsp"); do kill "$p" 2>/dev/null; done
sleep 1
for i in $(seq 1 "$N"); do
    "$FFMPEG" -hide_banner -loglevel error -re -stream_loop -1 -i "$CLIP" -c copy -f rtsp -rtsp_transport tcp "rtsp://localhost:8554/cam$i" &
    echo $! > "$PIDDIR/cam$i.pid"
    sleep 0.15
done
sleep 2
echo "atlas_farm: $RES GOP $GOP, $(pgrep -fc "media/atlas_${RES}_g${GOP}[.]mp4 -c copy -f rtsp")/$N publishers live"
