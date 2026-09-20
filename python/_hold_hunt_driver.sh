#!/usr/bin/env bash
# hold_frames hunt driver (run INSIDE the tensorrt-dev container from
# /workspace). Loops the variants under eager-abort malloc checking,
# tallies crash signatures, then chases a backtrace with gdb on the most
# crash-prone variant. Timeboxed by the caller.
#
# Usage: _hold_hunt_driver.sh <out_dir> <src> [reps]
set -u
OUT=${1:?out dir}
SRC=${2:?source (file path or rtsp url)}
REPS=${3:-30}
PY="${PYTHON3:-python3}"
export PYTHONPATH=/workspace/build:/workspace/python
mkdir -p "$OUT"
summary="$OUT/tally.txt"
: > "$summary"

for variant in vnohold v0 v2; do
    crash=0; clean=0
    for rep in $(seq 1 "$REPS"); do
        log="$OUT/${variant}_r${rep}.log"
        MALLOC_CHECK_=3 MALLOC_PERTURB_=42 timeout 120 \
            "$PY" python/_hold_hunt.py "$variant" "$SRC" > "$log" 2>&1
        rc=$?
        if grep -q CLEAN_BODY_EXIT "$log" && [ "$rc" -eq 0 ]; then
            clean=$((clean+1)); rm -f "$log"
        else
            crash=$((crash+1))
            echo "CRASH $variant rep$rep rc=$rc: $(grep -iE 'malloc|corrupt|Sig|free\(\)' "$log" | tail -1)" >> "$summary"
        fi
    done
    echo "TALLY $variant: $crash/$REPS crashed" | tee -a "$summary"
done

# per-frame vs per-process discriminator: fewer reps, longer drain
crash=0
for rep in $(seq 1 10); do
    log="$OUT/v0long_r${rep}.log"
    MALLOC_CHECK_=3 MALLOC_PERTURB_=42 timeout 300 \
        "$PY" python/_hold_hunt.py v0long "$SRC" > "$log" 2>&1
    rc=$?
    if grep -q CLEAN_BODY_EXIT "$log" && [ "$rc" -eq 0 ]; then
        rm -f "$log"
    else
        crash=$((crash+1))
        echo "CRASH v0long rep$rep rc=$rc" >> "$summary"
    fi
done
echo "TALLY v0long(2000 frames): $crash/10 crashed" | tee -a "$summary"

# backtrace chase: rerun v2 under gdb until it crashes (cap 25 tries)
if command -v gdb >/dev/null 2>&1; then
    for try in $(seq 1 25); do
        MALLOC_CHECK_=3 MALLOC_PERTURB_=42 timeout 240 \
            gdb --batch -ex run -ex bt -ex 'thread apply all bt' \
                --args "$PY" python/_hold_hunt.py v2 "$SRC" \
                > "$OUT/gdb_try${try}.log" 2>&1
        if grep -qiE "SIGABRT|SIGSEGV|corrupt" "$OUT/gdb_try${try}.log"; then
            echo "GDB BACKTRACE CAPTURED: gdb_try${try}.log" | tee -a "$summary"
            break
        fi
        rm -f "$OUT/gdb_try${try}.log"
    done
else
    echo "gdb not present in container - backtrace pass skipped" | tee -a "$summary"
fi
echo "HUNT COMPLETE" | tee -a "$summary"
