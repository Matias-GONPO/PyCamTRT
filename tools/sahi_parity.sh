#!/usr/bin/env bash
# SAHI PARITY TEST — closes CORDERO limitation #7 (Reports/report 9/
# PYCAMTRT_V0_REPORT.md §5): the pooled (cross-slot) SAHI path's design
# comments assert it produces IDENTICAL per-slot detections to the serial
# (per-slot tile loop) path — decode/NMS/merge are per-tile/per-slot
# independent, only the engine-call grouping differs. This had no
# automated check. This script is that check.
#
# KEY INSIGHT: FFmpegDemuxer opens plain files as well as RTSP URLs, and a
# file source decodes DETERMINISTICALLY - two runs over the same file with
# the same --frames N see byte-identical frames in identical order
# (single stream, single decoded pass; with --frames N less than the
# file's frame count the producer's reconnect-on-EOF path is never
# reached, so there is no wraparound to break determinism). That makes a
# byte-for-byte detection diff between --sahi-serial and --sahi (pooled)
# meaningful: any difference is either a real ordering/merge divergence or
# a genuine bug, not run-to-run frame jitter (which is all an RTSP source
# could ever give us).
#
# HISTORICAL NOTE (the bug this worked around is FIXED - v0.2.0's AVBSF
# path makes MP4 files first-class inputs; see CHANGELOG.md): this script
# predates the fix and extracts a raw .h264 elementary stream before
# running. Kept as-is deliberately - the raw-stream input is maximally
# deterministic for a parity check, and changing a verification
# instrument without need is against house rules. (Original symptom, for
# the record: moov-at-end MP4 probing starved the demuxer into spurious
# EOFs before the AVBSF/file-EOF rework.)
# construction order via targeted instrumentation (see report). A RAW
# H.264 elementary stream (no container, no seek/duration probing) does
# NOT hit this: rtsp_infer_multi decodes it cleanly end to end. So this
# script extracts one once (host ffmpeg, same convention as
# tools/atlas_farm.sh) and feeds THAT to rtsp_infer_multi, sidestepping the
# bug entirely without touching src/core/.
#
# Usage: ./sahi_parity.sh [MEDIA_PATH]
#   MEDIA_PATH defaults to tools/stream_farm/media/atlas_plate_g30.mp4
#   (any mp4/mov container works - it's re-muxed to raw .h264 before use)
set -u
cd "$(dirname "$0")/.."

MEDIA=${1:-tools/stream_farm/media/atlas_plate_g30.mp4}
ENGINE=models/yolov8n_plates_b1-16_fp16_sm86.engine
FRAMES=300
FFMPEG="${FFMPEG:-$HOME/anaconda3/envs/Python-dev/bin/ffmpeg}"
H264=_sahi_parity_media.h264

if [ ! -f "$MEDIA" ]; then
    echo "!! media not found: $MEDIA" >&2
    exit 2
fi
if [ ! -f "$ENGINE" ]; then
    echo "!! engine not found: $ENGINE" >&2
    exit 2
fi
if [ ! -x "$FFMPEG" ]; then
    echo "!! host ffmpeg not found at $FFMPEG (set FFMPEG=/path/to/ffmpeg)" >&2
    exit 2
fi

T0=$(date +%s)

echo "--- extracting raw H.264 elementary stream (historical determinism choice, see script header) ---"
"$FFMPEG" -y -loglevel error -i "$MEDIA" -c copy -bsf:v h264_mp4toannexb -f h264 "$H264"
if [ ! -s "$H264" ]; then
    echo "!! h264 extraction produced an empty/missing file" >&2
    exit 2
fi

# One container, one shell: build (picks up the --dump-dets CLI change),
# run serial, run pooled, run the plain (no --sahi) smoke check task 4
# asks for, then diff - all inside tensorrt-dev so the host never needs
# OpenCV/TensorRT/FFmpeg C++ dev headers (see
# Reports/Reading_CORDERO/chapters/08_build_run.md). File source needs
# neither --network host nor the RTSP farm.
docker run --rm --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v "$(pwd)":/workspace -w /workspace tensorrt-dev bash -c "
set -e
ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so 2>/dev/null
cd build && make -j\$(nproc) rtsp_infer_multi && cd ..

echo '--- plain (no --sahi) file-source smoke check ---'
./build/rtsp_infer_multi '$H264' --engine $ENGINE --frames $FRAMES \
    > /workspace/_sahi_plain.log 2>&1
tail -5 /workspace/_sahi_plain.log

echo '--- serial ---'
./build/rtsp_infer_multi '$H264' --engine $ENGINE --sahi --sahi-serial \
    --frames $FRAMES --dump-dets /workspace/_sahi_serial.tsv \
    > /workspace/_sahi_serial.log 2>&1
tail -5 /workspace/_sahi_serial.log

echo '--- pooled ---'
./build/rtsp_infer_multi '$H264' --engine $ENGINE --sahi \
    --frames $FRAMES --dump-dets /workspace/_sahi_pooled.tsv \
    > /workspace/_sahi_pooled.log 2>&1
tail -5 /workspace/_sahi_pooled.log

echo '--- serial control rerun (same path, same binary, second invocation) ---'
# Baseline-noise control: TensorRT/cuDNN kernel execution is not
# guaranteed bit-deterministic run-to-run (batch-composition-dependent
# reduction order), independent of the SAHI serial/pooled question. This
# reruns --sahi-serial a SECOND time so a serial-vs-pooled diff can be
# judged against the serial-vs-itself noise floor rather than assumed to
# be zero.
./build/rtsp_infer_multi '$H264' --engine $ENGINE --sahi --sahi-serial \
    --frames $FRAMES --dump-dets /workspace/_sahi_serial_control.tsv \
    > /workspace/_sahi_serial_control.log 2>&1
tail -5 /workspace/_sahi_serial_control.log
"
DOCKER_RC=$?

T1=$(date +%s)

if [ $DOCKER_RC -ne 0 ]; then
    echo "!! docker run failed (rc=$DOCKER_RC) - see _sahi_{plain,serial,pooled}.log" >&2
    exit 1
fi

if ! grep -q "Success" _sahi_plain.log; then
    echo "!! plain (no --sahi) file-source run did not report Success - see _sahi_plain.log" >&2
    exit 1
fi

if [ ! -s _sahi_serial.tsv ] || [ ! -s _sahi_pooled.tsv ] || [ ! -s _sahi_serial_control.tsv ]; then
    echo "!! one or more dumps are missing/empty - see _sahi_{serial,pooled,serial_control}.log" >&2
    exit 1
fi

if diff -q _sahi_serial.tsv _sahi_pooled.tsv > /dev/null; then
    echo "SAHI PARITY PASS - byte-identical ($(wc -l < _sahi_serial.tsv) lines, ${FRAMES} frames, $((T1 - T0))s)"
    rm -f _sahi_serial.tsv _sahi_pooled.tsv _sahi_serial_control.tsv \
        _sahi_plain.log _sahi_serial.log _sahi_pooled.log _sahi_serial_control.log "$H264"
    exit 0
fi

# Not byte-identical - before calling this a real bug, measure the
# baseline noise floor: does serial-vs-itself (two independent
# invocations of the SAME --sahi-serial path) already diverge by a
# similar amount? If so this is TensorRT/cuDNN run-to-run float
# non-determinism (batch-composition-dependent reduction order), not a
# serial/pooled merge-logic bug - see the header comment.
n_pooled_vs_serial=$(diff _sahi_serial.tsv _sahi_pooled.tsv | grep -c '^<')
n_serial_vs_control=$(diff _sahi_serial.tsv _sahi_serial_control.tsv | grep -c '^<')

echo "SAHI PARITY: not byte-identical" >&2
echo "  serial vs pooled      : $n_pooled_vs_serial differing detection/F lines" >&2
echo "  serial vs serial-rerun: $n_serial_vs_control differing detection/F lines (baseline noise floor)" >&2
echo "  first 10 serial-vs-pooled differing lines:" >&2
diff _sahi_serial.tsv _sahi_pooled.tsv | head -10 >&2

if [ "$n_serial_vs_control" -gt 0 ] && [ "$n_pooled_vs_serial" -le $((n_serial_vs_control * 2)) ]; then
    # PASS criterion (orchestrator ruling 2026-08-27): byte-identical is
    # provably too strict across separate runs - TensorRT's kernel/tactic
    # choice varies with batch composition, giving sub-pixel run-to-run
    # drift even serial-vs-itself. The honest gate is the CONTROL
    # comparison: pooled counts as equivalent when its divergence from
    # serial does not exceed ~2x the serial-vs-itself noise floor.
    echo "SAHI PARITY PASS (within noise floor) - pooled-vs-serial divergence" \
         "($n_pooled_vs_serial lines) is within ~2x the serial-vs-itself" \
         "baseline ($n_serial_vs_control lines): inherent TensorRT run-to-run" \
         "float non-determinism, no evidence of a pooled-path bug."
    rm -f _sahi_serial.tsv _sahi_pooled.tsv _sahi_serial_control.tsv \
        _sahi_plain.log _sahi_serial.log _sahi_pooled.log _sahi_serial_control.log "$H264"
    exit 0
else
    echo "SAHI PARITY FAIL - pooled-vs-serial divergence ($n_pooled_vs_serial lines)" \
         "clearly EXCEEDS the serial-vs-itself baseline ($n_serial_vs_control lines)." \
         "This looks like a REAL divergence introduced by the pooled path, not" \
         "just run-to-run engine noise - investigate before trusting the pooled" \
         "path's equivalence claim." >&2
    echo "(dumps kept: _sahi_serial.tsv _sahi_pooled.tsv _sahi_serial_control.tsv; logs: _sahi_{serial,pooled,serial_control}.log; media: $H264)" >&2
    exit 1
fi
