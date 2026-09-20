#!/usr/bin/env bash
# fleet_demo lifecycle: up | down | status
#   up     mediamtx + all 65 AIC22 publishers (tools/publish_aic22.sh), then demo.py in the
#          tensorrt-dev container (repo + conda env bind-mounted, host network)
#   down   stop the container, the publishers and mediamtx
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON3:-python3}"   # must be the interpreter the extension in build/ was built with (BUILD.md)
CONDA_MOUNT=""; [ -d "$HOME/anaconda3" ] && CONDA_MOUNT="-v $HOME/anaconda3:$HOME/anaconda3"   # makes a host env visible inside
NAME="${NAME:-fleet_demo}"
IMAGE="${IMAGE:-tensorrt-dev}"

up() {
    local scenarios
    scenarios=$("$PY" -c "import json; print(','.join(sorted({e['scenario'] for e in json.load(open('$HERE/data/manifest.json'))})))")
    "$HERE/tools/publish_aic22.sh" up --scenarios "$scenarios"   # also starts mediamtx (tools/stream_farm/farm.sh up 0)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    # The image lacks libav; _pycamtrt links against it, so install per run. libnvcuvid is mounted
    # by the driver as .so.1 only, so the unversioned link is recreated per run as well.
    docker run -d --rm --name "$NAME" --gpus all --network host \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
        -v "$REPO":/workspace $CONDA_MOUNT -w /workspace "$IMAGE" bash -c "
apt-get update -qq >/dev/null 2>&1; \
apt-get install -y -qq pkg-config libavformat-dev libavcodec-dev libavutil-dev >/dev/null 2>&1; \
ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so; \
PYTHONPATH=/workspace/build:/workspace/python $PY examples/fleet_demo/demo.py" >/dev/null
    echo "demo starting in container $NAME (about a minute: apt + engine load). Follow with: docker logs -f $NAME"
    echo "open http://localhost:8000/ once the log says 'ready gen 1'"
}

down() {
    docker stop -t 30 "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || true
    "$HERE/tools/publish_aic22.sh" down --all
}

status() {
    docker ps --filter "name=$NAME" --format 'container {{.Names}}: {{.Status}}' | grep . || echo "container $NAME: not running"
    "$HERE/tools/publish_aic22.sh" status | tail -3
    curl -s localhost:8000/stats || true
    echo
}

case "${1:-}" in
    up) up ;;
    down) down ;;
    status) status ;;
    *) echo "usage: $0 up|down|status"; exit 1 ;;
esac
