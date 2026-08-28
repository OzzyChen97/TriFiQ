#!/usr/bin/env bash
set -euo pipefail

# Collect the v2 cross-context selection buffer: 12 tasks (4 per split) x
# 3 seeds x 4 consecutive replans from a hash-assigned early/middle/late
# window of a 12-replan FP16 on-policy rollout, then pool the shards into
# the canonical cross-model selection archive.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="${V2_SELECT_COLLECT_PY:-/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python}"
COLLECTOR="$REPO_ROOT/scripts/tools/pi05_collect_fp16_onpolicy_probe.py"
SPEC_BUILDER="$REPO_ROOT/scripts/tools/build_v2_selection_context.py"
POOLER="$REPO_ROOT/scripts/tools/pool_v2_selection_buffer.py"
V2_ROOT="${FULL_CONTEXT_V2_ROOT:-$REPO_ROOT/runs/full_context_v2}"
SPEC_PATH="$V2_ROOT/selection_context_spec.json"
SHARD_DIR="$V2_ROOT/selection_shards"
BUFFER_PATH="$V2_ROOT/selection_buffer.npz"
HOST="${V2_SELECT_HOST:-127.0.0.1}"
PORT="${V2_SELECT_PORT:-18602}"
EGL_POOL="${V2_SELECT_EGL_POOL:-4,7}"
MAX_REPLANS=12
REPLANS_PER_TRIAL=4

usage() {
    echo "usage: $0 build-spec | capture | pool | status" >&2
}

build_spec() {
    "$PYTHON" "$SPEC_BUILDER" --out "$SPEC_PATH"
}

capture() {
    [[ -f "$SPEC_PATH" ]] || { echo "missing spec; run $0 build-spec first" >&2; return 1; }
    mkdir -p "$SHARD_DIR"
    "$PYTHON" - "$SPEC_PATH" "$SHARD_DIR" "$COLLECTOR" "$HOST" "$PORT" "$EGL_POOL" \
        "$MAX_REPLANS" "$REPLANS_PER_TRIAL" <<'PY'
import json
import subprocess
import sys

spec_path, shard_dir, collector, host, port, egl_pool, max_replans, replans = sys.argv[1:]
spec = json.load(open(spec_path, encoding="utf-8"))
egl_devices = [int(value) for value in egl_pool.split(",")]
index = 0
for entry in spec["entries"]:
    for seed_row in entry["seeds"]:
        seed = int(seed_row["seed"])
        window_start = int(seed_row["window_start"])
        egl = egl_devices[index % len(egl_devices)]
        index += 1
        out = f"{shard_dir}/{entry['task']}_{seed}.npz"
        print(f"[v2-select-capture] {entry['task']}/{seed} window={window_start} egl={egl}", flush=True)
        subprocess.run(
            [
                sys.executable, collector,
                "--host", host,
                "--port", port,
                "--tasks", entry["task"],
                "--trial-seeds", str(seed),
                "--egl-device", str(egl),
                "--max-replans", max_replans,
                "--replans-per-trial", replans,
                "--selection", "window",
                "--window-start", str(window_start),
                "--out", out,
                "--force",
            ],
            check=True,
        )
PY
}

pool() {
    [[ -f "$SPEC_PATH" ]] || { echo "missing spec; run $0 build-spec first" >&2; return 1; }
    "$PYTHON" "$POOLER" --spec "$SPEC_PATH" --shard-dir "$SHARD_DIR" --out "$BUFFER_PATH"
}

status() {
    "$PYTHON" - "$SPEC_PATH" "$SHARD_DIR" <<'PY'
import json
from pathlib import Path
import sys

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
shard_dir = Path(sys.argv[2])
missing = []
done = 0
for entry in spec["entries"]:
    for seed_row in entry["seeds"]:
        path = shard_dir / f"{entry['task']}_{seed_row['seed']}.npz"
        if path.is_file():
            done += 1
        else:
            missing.append(str(path.name))
print(f"shards={done}/{spec['total_sequences']} missing={missing}")
PY
}

case "${1:-}" in
    build-spec) build_spec ;;
    capture) capture ;;
    pool) pool ;;
    status) status ;;
    *) usage ;;
esac