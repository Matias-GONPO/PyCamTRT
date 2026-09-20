#!/usr/bin/env bash
# Publish the AIC22 manifest (tools/prep_aic22.py's data/manifest.json) as
# RTSP streams via mediamtx, reusing tools/stream_farm/farm.sh's exact
# publisher pattern (ffmpeg -re -c copy, no re-encode) and PID-file
# conventions - see that script's header for the underlying rationale.
#
# Usage:
#   ./publish_aic22.sh up [--scenarios S02,S01]
#                                      start mediamtx (if needed) + one
#                                      looping publisher per manifest camera
#   ./publish_aic22.sh once [--scenarios S02,S01]
#                                      same, but single-pass (no loop) - not
#                                      the ops-mode path (loop is), kept for
#                                      one-shot smoke checks
#   ./publish_aic22.sh down [--all]   stop our publishers; with --all also
#                                      stop mediamtx (via farm.sh down)
#   ./publish_aic22.sh status         list our publishers + mediamtx state
#
# --scenarios (optional, up/once only): comma-separated scenario ids (e.g.
# S02,S01), same flag name/shape as tools/prep_aic22.py's own --scenarios -
# restricts which manifest.json entries get a publisher this run (the
# manifest itself is untouched; other scenarios' cameras are just skipped).
# Default: S01 only (the shipped 5-camera demo scenario - see publish_all()'s
# own `scenarios="S01"` default below). Pass --scenarios S02,S01 (or any
# other combination) to publish more, e.g. for the P2 ground-truth-aligned
# eval, which needs S02 (the GT-aligned scenario) instead.
#
# Streams are served at rtsp://localhost:8554/aic/<slug> (e.g. aic/s02_c006).
set -u
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # examples/fleet_demo
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"        # PyCamTRT
FARM_DIR="$REPO_ROOT/tools/stream_farm"
DATA_DIR="$EXAMPLE_DIR/data"
RUN_DIR="$EXAMPLE_DIR/run"
MANIFEST="$DATA_DIR/manifest.json"

FFMPEG="${FFMPEG:-ffmpeg}"
PYTHON3="${PYTHON3:-python3}"
# Fixed inter-spawn gap between publishers - NOT a per-camera dataset
# offset_s stagger (that was a GT-alignment/re-ID need; ops uses loop mode
# only, where no cross-camera timestamp alignment matters). Was previously
# `0.7 + offset_s` per camera and, critically, applied as a BLOCKING sleep
# before every spawn inside a single sequential loop - so the delays
# ACCUMULATED across all 65 cameras (~42 min to finish starting). Fixed:
# every publisher spawns back-to-back with only this small fixed gap
# between spawns (65 * 0.15s ~= 10s total, comfortably under the <60s gate).
INTER_SPAWN_GAP=0.15

# Emits one tab-separated "slug<TAB>rtsp_path<TAB>file" line per manifest
# camera, in manifest order, optionally restricted to $1's comma-separated
# scenario list (empty/unset = every camera). jq would also do this, but
# python3 (`PYTHON3`, PATH by default) is always present here, so
# prefer it.
manifest_rows() {
    local scenarios="${1:-}"
    "$PYTHON3" -c '
import json, sys
scenarios = {s.strip().upper() for s in sys.argv[2].split(",") if s.strip()}
with open(sys.argv[1]) as f:
    data = json.load(f)
for e in data:
    if scenarios and e.get("scenario", "").upper() not in scenarios:
        continue
    print("\t".join([e["slug"], e["rtsp_path"], e["file"]]))
' "$MANIFEST" "$scenarios"
}

start_mediamtx() {
    if [ -f "$FARM_DIR/run/mediamtx.pid" ] && kill -0 "$(cat "$FARM_DIR/run/mediamtx.pid")" 2>/dev/null; then
        echo "mediamtx already running (pid $(cat "$FARM_DIR/run/mediamtx.pid"))"
    else
        # "up 0" starts mediamtx only (seq 1 0 loops zero times) - verified
        # against farm.sh's up() body.
        (cd "$FARM_DIR" && ./farm.sh up 0)
    fi
}

start_publisher() {
    local slug="$1" rtsp_path="$2" file="$3" mode="$4"
    if [ -f "$RUN_DIR/$slug.pid" ] && kill -0 "$(cat "$RUN_DIR/$slug.pid")" 2>/dev/null; then
        echo "$slug already running"
        return
    fi

    # STALE-PID HARDENING (found at teardown after a re-up during
    # acceptance left ~5 orphaned publishers): the check above only trusts
    # the RECORDED pid. If that pid is dead/missing but a REAL ffmpeg for
    # this exact slug is still alive anyway - e.g. a survivor of a lost
    # pid-file-overwrite race from a prior overlapping `up`, or the pid file
    # was hand-edited/lost - the old code would spawn a SECOND publisher for
    # the same camera and overwrite the pid file with the new one, silently
    # orphaning the first (no pid file will ever point to it again, so
    # `down` can never find it). Match on the unique source file path (each
    # camera has its own .mp4, so a substring match can't cross slugs)
    # instead of trusting bookkeeping, and adopt a live survivor rather than
    # duplicating it.
    local live_pid
    live_pid=$(pgrep -f "$DATA_DIR/$file" | head -1)
    if [ -n "$live_pid" ]; then
        echo "$slug: adopting untracked live publisher (pid $live_pid; pid file was stale/missing) instead of spawning a duplicate"
        echo "$live_pid" > "$RUN_DIR/$slug.pid"
        return
    fi

    local loop_args=()
    if [ "$mode" = "loop" ]; then
        loop_args=(-stream_loop -1)
    fi
    # </dev/null: this loop's "done < <(manifest_rows)" keeps a pipe open as
    # its own stdin for the whole iteration; without an explicit redirect
    # here, a backgrounded ffmpeg inherits that SAME pipe fd and can steal
    # bytes meant for the loop's `read` (observed: truncated/dropped slugs
    # when several cams' delays were close together) - classic
    # background-inside-`while read <(...)` bash gotcha.
    "$FFMPEG" -hide_banner -loglevel error \
        -re "${loop_args[@]}" -i "$DATA_DIR/$file" \
        -c copy -f rtsp -rtsp_transport tcp \
        "rtsp://localhost:8554/$rtsp_path" \
        < /dev/null > "$RUN_DIR/$slug.log" 2>&1 &
    echo $! > "$RUN_DIR/$slug.pid"
    echo "$slug up (pid $!, rtsp://localhost:8554/$rtsp_path)"
    sleep "$INTER_SPAWN_GAP"
}

publish_all() {
    local mode="$1"; shift
    local scenarios="S01"  # shipped demo default: one intersection; override with --scenarios
    while [ $# -gt 0 ]; do
        case "$1" in
            --scenarios) scenarios="$2"; shift 2 ;;
            *) echo "unknown option: $1" >&2; exit 1 ;;
        esac
    done
    mkdir -p "$RUN_DIR"
    if [ ! -f "$MANIFEST" ]; then
        echo "manifest not found: $MANIFEST (run tools/prep_aic22.py first)" >&2
        exit 1
    fi
    start_mediamtx
    while IFS=$'\t' read -r slug rtsp_path file; do
        start_publisher "$slug" "$rtsp_path" "$file" "$mode"
    done < <(manifest_rows "$scenarios")
}

down() {
    local all="${1:-}"
    for f in "$RUN_DIR"/*.pid; do
        [ -f "$f" ] || continue
        # api.pid is run.sh's own PID file (the FastAPI/uvicorn process) -
        # NOT one of our publishers, but it lives in this same run/ dir
        # (run.sh's choice) so the glob above would otherwise match it too.
        # run.sh's own down() already removes api.pid before ever calling
        # us with --all, so this only matters when this script is run
        # standalone - skip it either way, "our publishers" never includes
        # the API.
        [ "$(basename "$f")" = "api.pid" ] && continue
        kill "$(cat "$f")" 2>/dev/null && echo "stopped $(basename "$f" .pid)"
        rm -f "$f"
    done
    # Defensive sweep: catch any AIC22 publisher ffmpeg that is running
    # WITHOUT a pid file tracking it - e.g. an orphan from a lost
    # pid-file-overwrite race (see start_publisher's own adopt-instead-of-
    # duplicate guard above, which stops new ones from being created this
    # way but can't retroactively fix ones already orphaned before it ran).
    # Matches only OUR publishers: every one of them - and nothing else in
    # this repo - runs ffmpeg with a source file under data/aic22_h264/.
    local orphan_pids
    orphan_pids=$(pgrep -f "$DATA_DIR/aic22_h264" 2>/dev/null)
    if [ -n "$orphan_pids" ]; then
        local opid
        for opid in $orphan_pids; do
            kill "$opid" 2>/dev/null && echo "reaped orphaned publisher (pid $opid, no pid file was tracking it)"
        done
    fi
    # Escalate: an ffmpeg blocked on a write to a server that already went
    # away can sit on SIGTERM indefinitely (observed: 27-44 of 65 survived).
    # Give them 2 s, then SIGKILL whatever is still publishing our clips.
    local i survivors
    for i in $(seq 1 20); do
        survivors=$(pgrep -f "$DATA_DIR/aic22_h264" 2>/dev/null) || true
        [ -z "$survivors" ] && break
        sleep 0.1
    done
    if [ -n "$survivors" ]; then
        kill -KILL $survivors 2>/dev/null || true
        echo "killed $(echo "$survivors" | wc -w) publisher(s) that ignored SIGTERM"
    fi
    if [ "$all" = "--all" ]; then
        (cd "$FARM_DIR" && ./farm.sh down)
        echo "mediamtx stopped (--all)"
    fi
}

status() {
    local pids=()
    for f in "$RUN_DIR"/*.pid; do
        [ -f "$f" ] || continue
        [ "$(basename "$f")" = "api.pid" ] && continue  # see down()'s comment
        local pid; pid=$(cat "$f")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$(basename "$f" .pid): running (pid $pid)"
            pids+=("$pid")
        else
            echo "$(basename "$f" .pid): DEAD"
        fi
    done
    if [ "${#pids[@]}" -gt 0 ]; then
        echo "--- publisher CPU/mem (ps) ---"
        ps -o pid,pcpu,pmem,comm -p "$(IFS=,; echo "${pids[*]}")"
    fi
    echo "--- mediamtx ---"
    if [ -f "$FARM_DIR/run/mediamtx.pid" ] && kill -0 "$(cat "$FARM_DIR/run/mediamtx.pid")" 2>/dev/null; then
        echo "mediamtx: running (pid $(cat "$FARM_DIR/run/mediamtx.pid"))"
    else
        echo "mediamtx: DEAD"
    fi
}

case "${1:-}" in
    up) shift; publish_all loop "$@" ;;
    once) shift; publish_all once "$@" ;;
    down) down "${2:-}" ;;
    status) status ;;
    *) echo "usage: $0 up|once [--scenarios S02,S01] | down [--all] | status"; exit 1 ;;
esac
