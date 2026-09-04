#!/usr/bin/env bash
# Publishes the REAL plate clip on rtsp://localhost:8554/plate1..plateN,
# alongside the webcam farm's cam1..camN - so the sweep's two content
# types are live simultaneously (its loop expects both).
#
# Replaces the discarded looped-still plate_test.mp4 publishers (see
# ../METHODOLOGY.md: the still image's keyframe-burst delivery gave
# bimodal decode artifacts). This uses the real temporal plate recording
# with -c copy, same as the webcam farm - decoder-side identical to a
# live camera.
#
# mediamtx must already be up (tools/stream_farm/farm.sh up N starts it).
# No -ss content offset: with -c copy a mid-GOP seek corrupts KEYFRAME
# packet PTS (see farm.sh's own WHY-comment); stagger provides the
# keyframe decorrelation.
#
# Usage: ./plate_farm.sh up [N] | down
set -u
export LC_ALL=C
cd "$(dirname "$0")"
FFMPEG="${FFMPEG:-$HOME/anaconda3/envs/Python-dev/bin/ffmpeg}"
CLIP="${CLIP:-$(cd ../../tools/stream_farm && pwd)/media/atlas_plate_g30.mp4}"
PIDDIR="run_plate"
STAGGER=0.7

up() {
    local n="${1:-16}"
    mkdir -p "$PIDDIR"
    for i in $(seq 1 "$n"); do
        if [ -f "$PIDDIR/plate$i.pid" ] && kill -0 "$(cat "$PIDDIR/plate$i.pid")" 2>/dev/null; then
            echo "plate$i already running"; continue
        fi
        "$FFMPEG" -hide_banner -loglevel error \
            -re -stream_loop -1 -i "$CLIP" \
            -c copy -f rtsp -rtsp_transport tcp \
            "rtsp://localhost:8554/plate$i" &
        echo $! > "$PIDDIR/plate$i.pid"
        echo "plate$i up (pid $!)"
        sleep "$STAGGER"
    done
}

down() {
    for f in "$PIDDIR"/plate*.pid; do
        [ -f "$f" ] || continue
        kill "$(cat "$f")" 2>/dev/null && echo "stopped $(basename "$f" .pid)"
        rm -f "$f"
    done
}

case "${1:-}" in
    up) up "${2:-16}" ;;
    down) down ;;
    *) echo "usage: $0 up [N] | down"; exit 1 ;;
esac
