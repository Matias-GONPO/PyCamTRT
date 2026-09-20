#!/usr/bin/env python3
"""prep_aic22.py - transcode AIC22/CityFlowV2 camera clips to H.264 once.

Each camera directory in the dataset ships `vdo.avi` encoded as MS-MPEG4v2
(not H.264) - NVDEC in the pycamtrt pipeline can't decode that, so this
does a one-time CPU transcode to H.264 (no B-frames; keyframe interval
`--gop`, default 30 frames = one keyframe every 3 s at the dataset's 10 fps)
before the clip can ever be published over RTSP with `-c copy` (see
tools/publish_aic22.sh and tools/stream_farm/farm.sh's PTS-safety note -
`-c copy` republishing requires a clean, seek-free source bitstream).

Run with your Python environment; ffmpeg/ffprobe come from PATH unless
FFMPEG/FFPROBE are set:

    python3 tools/prep_aic22.py \\
        --scenarios S02,S01 --jobs 4 --crf 20

Writes transcoded clips under --out (default data/aic22_h264/) and a
data/manifest.json describing every camera (scenario, cam, file, fps,
frames, offset_s, slug, rtsp_path, name) that tools/publish_aic22.sh and
later phases (engine service, GT-aligned eval) consume. Safe to re-run
per-scenario: existing manifest entries for other scenarios are preserved
(merge, not overwrite); already-transcoded, up-to-date outputs are skipped
unless --force.

`--jobs N` runs N ffmpeg transcodes concurrently (each is a separate CPU-
bound subprocess - `subprocess.run` releases the GIL while it blocks, so a
plain `ThreadPoolExecutor` is enough, no multiprocessing needed). `--crf`
(default 20) is the libx264 quality knob - the full-65-camera transcode's
disk-tightening fallback is a caller-side `--crf 23` re-run of whichever
scenario needs it, not an automatic in-script decision.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent.parent  # examples/fleet_demo/
DEFAULT_DATASET_ROOT = None   # --dataset-root is required: the CityFlowV2/AIC22 root with validation/ train/ test/
DEFAULT_SCENARIOS = "S02,S01"
DEFAULT_CRF = 20
DEFAULT_GOP = 30
GOP_WARNING = (
    "NOTE: --gop only imitates a camera's keyframe-interval setting. Real IP cameras set\n"
    "their own (typically 1-2 s, per stream), so an intermediate keyframes-only decode\n"
    "rate on a real fleet is a camera setting, not a dataset change. A short GOP raises\n"
    "bitrate and file size; a long one makes keyframes-only decode very sparse. The\n"
    "selected scenarios are re-encoded in place: stop the publishers first (run.sh down)."
)
SPLITS = ("validation", "train", "test")  # scan order; scenario dir found under one of these
# S03/S04/S05 don't ship their own cam_loc PNG upstream - they share one
# combined overview image (S0345.png). The maps route needs a per-scenario
# alias so GET /api/v1/maps/S03 (etc.) resolves to something.
S0345_ALIAS_SCENARIOS = ("S03", "S04", "S05")

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")

_print_lock = threading.Lock()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, required=True,
                   help="AIC22/CityFlowV2 dataset root (default: %(default)s)")
    p.add_argument("--scenarios", default=DEFAULT_SCENARIOS,
                   help="comma-separated scenario ids, e.g. S02,S01 (default: %(default)s)")
    p.add_argument("--out", default=None,
                   help="H.264 output dir (default: <example>/data/aic22_h264)")
    p.add_argument("--force", action="store_true",
                   help="retranscode even if an up-to-date output already exists")
    p.add_argument("--jobs", type=int, default=1,
                   help="number of concurrent ffmpeg transcodes (default: %(default)s)")
    p.add_argument("--crf", type=int, default=DEFAULT_CRF,
                   help="libx264 -crf value, lower = higher quality/bigger file (default: %(default)s)")
    p.add_argument("--gop", type=int, default=DEFAULT_GOP,
                   help="keyframe interval in frames (default: %(default)s = one keyframe per 3 s at 10 fps); "
                        "recorded per camera in the manifest; a camera whose clip was encoded with a "
                        "different --gop is re-encoded even without --force. " + GOP_WARNING.replace("\n", " "))
    return p.parse_args()


def find_scenario_split(dataset_root: Path, scenario: str):
    """Scan validation/, train/, test/ for the scenario dir - don't hardcode
    which split a scenario lives under, verify on disk (S02 turned out to be
    under validation/, S01 under train/)."""
    hits = [split for split in SPLITS if (dataset_root / split / scenario).is_dir()]
    if not hits:
        return None
    if len(hits) > 1:
        print(f"WARNING: scenario {scenario} found under multiple splits {hits}, using {hits[0]}",
              file=sys.stderr)
    return hits[0]


def parse_kv_file(path: Path):
    """Parse 'c006 0.061' / 'c006 2110' style files into {cam: str-value}."""
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        out[parts[0]] = parts[1]
    return out


def probe_fps(vdo_path: Path) -> float:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(vdo_path)],
        capture_output=True, text=True,
    )
    rate = result.stdout.strip()
    if "/" in rate:
        num, den = rate.split("/")
        den = float(den) or 1.0
        return round(float(num) / den, 3)
    try:
        return float(rate)
    except ValueError:
        return 0.0


def transcode(vdo_path: Path, out_path: Path, force: bool, crf: int, gop: int) -> str:
    """Returns 'skipped' | 'transcoded' | 'failed'."""
    if out_path.exists() and not force and out_path.stat().st_mtime > vdo_path.stat().st_mtime:
        return "skipped"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-i", str(vdo_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-x264-params", f"keyint={gop}:min-keyint={gop}:scenecut=0",
        "-bf", "0", "-an",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg FAILED for {vdo_path}:\n{result.stderr[-2000:]}", file=sys.stderr)
        return "failed"
    return "transcoded"


def _process_camera(scenario: str, cam: str, vdo_path: Path, out_dir: Path, force: bool, crf: int,
                     offsets: dict, framenums: dict, data_dir: Path, gop: int, prev_entry: dict):
    """Runs in a worker thread: transcode + probe + build the manifest entry
    (or None on failure). Returns (slug, entry_or_None, status, deviations)."""
    slug = f"{scenario.lower()}_{cam}"
    out_path = out_dir / f"{slug}.mp4"
    deviations = []

    # The same output file encoded with a different --gop must be redone: the
    # manifest remembers what is on disk (older manifests without "gop" are 30).
    if (prev_entry and prev_entry.get("file") == str(out_path.relative_to(data_dir))
            and int(prev_entry.get("gop", DEFAULT_GOP)) != gop):
        force = True
    status = transcode(vdo_path, out_path, force, crf, gop)
    with _print_lock:
        print(f"{scenario}/{cam}: {status} -> {out_path}")
    if status == "failed":
        deviations.append(f"{scenario}/{cam}: ffmpeg transcode failed")
        return slug, None, status, deviations

    fps = probe_fps(vdo_path)
    offset_s = float(offsets.get(cam, 0.0))
    if cam not in offsets:
        deviations.append(f"{scenario}/{cam}: no cam_timestamp entry, offset_s defaulted to 0.0")
    frames_raw = framenums.get(cam)
    frames = int(frames_raw) if frames_raw is not None else 0
    if frames_raw is None:
        deviations.append(f"{scenario}/{cam}: no cam_framenum entry, frames defaulted to 0")

    entry = {
        "scenario": scenario,
        "cam": cam,
        "file": str(out_path.relative_to(data_dir)),
        "fps": fps,
        "gop": gop,
        "frames": frames,
        "offset_s": offset_s,
        "slug": slug,
        "rtsp_path": f"aic/{slug}",
        "name": f"{scenario} {cam}",
    }
    return slug, entry, status, deviations


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    data_dir = EXAMPLE_DIR / "data"
    out_dir = Path(args.out).resolve() if args.out else data_dir / "aic22_h264"
    if data_dir not in out_dir.parents:
        print(f"FATAL: --out must be inside {data_dir} (manifest paths are relative to it): {out_dir}", file=sys.stderr)
        sys.exit(1)
    manifest_path = data_dir / "manifest.json"

    if not dataset_root.is_dir():
        print(f"FATAL: dataset root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    deviations = []
    new_entries = {}
    existing = json.loads(manifest_path.read_text()) if manifest_path.is_file() else []
    prev_by_slug = {e["slug"]: e for e in existing}
    if args.gop != DEFAULT_GOP:
        print(f"\n--gop {args.gop}: {GOP_WARNING}\n")

    # Collect every camera job across all requested scenarios first, so
    # --jobs concurrency isn't scoped to one scenario at a time.
    jobs = []  # (scenario, cam, vdo_path, offsets, framenums)
    for scenario in scenarios:
        split = find_scenario_split(dataset_root, scenario)
        if split is None:
            msg = f"scenario {scenario} not found under any of {SPLITS} in {dataset_root}"
            print(f"WARNING: {msg}", file=sys.stderr)
            deviations.append(msg)
            continue

        scenario_dir = dataset_root / split / scenario
        offsets = parse_kv_file(dataset_root / "cam_timestamp" / f"{scenario}.txt")
        framenums = parse_kv_file(dataset_root / "cam_framenum" / f"{scenario}.txt")

        cam_dirs = sorted(d.name for d in scenario_dir.iterdir() if d.is_dir())
        for cam in cam_dirs:
            vdo_path = scenario_dir / cam / "vdo.avi"
            if not vdo_path.is_file():
                msg = f"{scenario}/{cam}: no vdo.avi on disk, skipped"
                print(f"WARNING: {msg}", file=sys.stderr)
                deviations.append(msg)
                continue
            jobs.append((scenario, cam, vdo_path, offsets, framenums))

    total = len(jobs)
    done = 0
    print(f"\n---- transcoding {total} camera(s) with --jobs {args.jobs} (crf={args.crf}, gop={args.gop}) ----")
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {
                pool.submit(_process_camera, scenario, cam, vdo_path, out_dir, args.force, args.crf,
                            offsets, framenums, data_dir, args.gop,
                            prev_by_slug.get(f"{scenario.lower()}_{cam}")): (scenario, cam)
                for scenario, cam, vdo_path, offsets, framenums in jobs
            }
            for fut in as_completed(futures):
                scenario, cam = futures[fut]
                done += 1
                try:
                    slug, entry, status, devs = fut.result()
                except Exception as exc:  # noqa: BLE001 - one bad job must not kill the batch
                    print(f"ERROR: {scenario}/{cam} raised {exc!r}", file=sys.stderr)
                    deviations.append(f"{scenario}/{cam}: worker raised {exc!r}")
                    continue
                deviations.extend(devs)
                if entry is not None:
                    new_entries[slug] = entry
                with _print_lock:
                    print(f"  [{done}/{total}] done ({scenario}/{cam})")

    # Merge into any existing manifest (other scenarios prepped earlier survive).
    by_slug = dict(prev_by_slug)
    by_slug.update(new_entries)
    merged = sorted(by_slug.values(), key=lambda e: (e["scenario"], e["cam"]))

    data_dir.mkdir(parents=True, exist_ok=True)
    # Copy each scenario's cam_loc overview PNG next to the clips so the
    # dashboard can serve it as a camera-location guidance map
    # (GET /api/v1/maps/<scenario>). Dataset asset -> stays under gitignored
    # data/, same license rule as the videos. Best-effort: not every
    # scenario ships a map (S03/S04/S05 share one combined image upstream -
    # aliased below).
    maps_dir = data_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    s0345_src = dataset_root / "cam_loc" / "S0345.png"
    for scenario in scenarios:
        src_png = dataset_root / "cam_loc" / f"{scenario}.png"
        if src_png.is_file():
            shutil.copyfile(src_png, maps_dir / f"{scenario}.png")
            print(f"map: copied cam_loc/{scenario}.png -> {maps_dir / (scenario + '.png')}")
        elif scenario in S0345_ALIAS_SCENARIOS and s0345_src.is_file():
            shutil.copyfile(s0345_src, maps_dir / f"{scenario}.png")
            print(f"map: aliased cam_loc/S0345.png -> {maps_dir / (scenario + '.png')}")
        else:
            print(f"map: no cam_loc/{scenario}.png (or S0345.png alias) in dataset (skipped)")

    manifest_path.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"\nwrote {manifest_path} ({len(merged)} cameras)")

    # Summary table (full merged manifest, so a partial re-run still shows everything).
    print("\n---- manifest summary ----")
    header = f"{'slug':<14} {'file':<28} {'fps':>6} {'gop':>4} {'frames':>7} {'offset_s':>9} {'rtsp_path':<16} name"
    print(header)
    print("-" * len(header))
    for e in merged:
        print(f"{e['slug']:<14} {e['file']:<28} {e['fps']:>6.2f} {int(e.get('gop', DEFAULT_GOP)):>4d} {e['frames']:>7d} "
              f"{e['offset_s']:>9.3f} {e['rtsp_path']:<16} {e['name']}")

    if deviations:
        print("\n---- deviations ----")
        for d in deviations:
            print(f"  - {d}")


if __name__ == "__main__":
    main()
