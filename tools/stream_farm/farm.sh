#!/usr/bin/env bash
# Simulated RTSP camera farm: mediamtx + N ffmpeg file loops, no re-encoding.
#
# Each "camera" replays media/webcam_60s.mp4 (a real recorded camera
# bitstream) with -c copy: the decoder-side work is identical to a live
# camera, at ~zero CPU cost per stream. Starts are staggered and each
# stream begins at a different offset into the clip, so keyframes and
# content are decorrelated across streams (like real cameras).
#
# Usage:
#   ./farm.sh up [N]      start mediamtx + N streams (default 4)
#   ./farm.sh down        stop everything
#   ./farm.sh status      list streams + farm CPU usage
#
# Streams are served at rtsp://<host>:8554/cam1 .. camN.

set -u
export LC_ALL=C   # decimal points in awk output regardless of system locale
cd "$(dirname "$0")"

FFMPEG="${FFMPEG:-ffmpeg}"
CLIP="${CLIP:-media/webcam_60s.mp4}"
PIDDIR="run"
STAGGER=0.7   # seconds between stream starts (decorrelates keyframes)

up() {
    local n="${1:-4}"
    mkdir -p "$PIDDIR"
    if [ ! -f "$PIDDIR/mediamtx.pid" ] || ! kill -0 "$(cat "$PIDDIR/mediamtx.pid")" 2>/dev/null; then
        ./bin/mediamtx ./bin/mediamtx.yml > "$PIDDIR/mediamtx.log" 2>&1 &
        echo $! > "$PIDDIR/mediamtx.pid"
        sleep 1
        echo "mediamtx up (pid $(cat "$PIDDIR/mediamtx.pid"))"
    fi
    local dur
    dur=$("$FFMPEG" -i "$CLIP" 2>&1 | grep -oP 'Duration: \K[0-9:.]+' | awk -F: '{print $1*3600+$2*60+$3}')
    for i in $(seq 1 "$n"); do
        if [ -f "$PIDDIR/cam$i.pid" ] && kill -0 "$(cat "$PIDDIR/cam$i.pid")" 2>/dev/null; then
            echo "cam$i already running"; continue
        fi
        # NO -ss content offset: with -c copy, a mid-GOP seek makes ffmpeg
        # emit corrupted PTS on KEYFRAME packets specifically (found while
        # verifying --decode key: K-packet pts stepped 0.33 ms instead of
        # 1 s; source file clean, no-ss publisher clean). Keyframe/GOP
        # phase decorrelation still comes from the staggered starts below;
        # content decorrelation is sacrificed for correct timestamps.
        "$FFMPEG" -hide_banner -loglevel error \
            -re -stream_loop -1 -i "$CLIP" \
            -c copy -f rtsp -rtsp_transport tcp \
            "rtsp://localhost:8554/cam$i" &
        echo $! > "$PIDDIR/cam$i.pid"
        echo "cam$i up (pid $!)"
        sleep "$STAGGER"
    done
}

down() {
    for f in "$PIDDIR"/cam*.pid "$PIDDIR"/mediamtx.pid; do
        [ -f "$f" ] || continue
        kill "$(cat "$f")" 2>/dev/null && echo "stopped $(basename "$f" .pid)"
        rm -f "$f"
    done
}

status() {
    local pids=()
    for f in "$PIDDIR"/cam*.pid "$PIDDIR"/mediamtx.pid; do
        [ -f "$f" ] || continue
        local pid; pid=$(cat "$f")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$(basename "$f" .pid): running (pid $pid)"
            pids+=("$pid")
        else
            echo "$(basename "$f" .pid): DEAD"
        fi
    done
    if [ "${#pids[@]}" -gt 0 ]; then
        echo "--- farm CPU/mem (ps) ---"
        ps -o pid,pcpu,pmem,comm -p "$(IFS=,; echo "${pids[*]}")"
    fi
}

case "${1:-}" in
    up) up "${2:-4}" ;;
    down) down ;;
    status) status ;;
    *) echo "usage: $0 up [N] | down | status"; exit 1 ;;
esac
