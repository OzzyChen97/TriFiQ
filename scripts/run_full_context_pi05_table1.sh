#!/usr/bin/env bash
set -euo pipefail

# Formal RoboCasa365 row for the frozen pi0.5 DyPAC-VLA plan.
# The row uses the same four-flow-step solver as the registered pi0.5
# references. It is idempotent at the (config, task-set, task, seed) key.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
MANIFEST_TOOL="$REPO_ROOT/scripts/tools/pi05_make_dypac_table1_manifest.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_pi05_dypac_table1.py"

RUN_ROOT="${FULL_CONTEXT_PI05_TABLE1_ROOT:-$REPO_ROOT/runs/full_context_v2/pi05_table1}"
CONTROL_DIR="$RUN_ROOT/control"
WORKER_CONTROL="$CONTROL_DIR/workers"
RESULTS_DIR="$RUN_ROOT/results/full_context_w4a8_dynamic_profile"
PREFLIGHT_DIR="$RUN_ROOT/preflight"
MANIFEST="$RUN_ROOT/manifest.json"
AGGREGATE="$RUN_ROOT/aggregate.json"

CONFIG_ID="full_context_w4a8_dynamic_profile"
PLAN="${FULL_CONTEXT_PI05_TABLE1_PLAN:-$REPO_ROOT/runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json}"
HESSIAN="${FULL_CONTEXT_PI05_TABLE1_HESSIAN:-$REPO_ROOT/runs/full_context_v2/pi05_quick/artifacts/hessian_w4.npz}"
PACK_DIR="${FULL_CONTEXT_PI05_TABLE1_PACK_DIR:-$REPO_ROOT/runs/archive_previous_versions/errorfold_v4_iter/calibration/pi05_full_w4/identity_pack}"
BUFFER="${FULL_CONTEXT_PI05_TABLE1_BUFFER:-$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz}"
CHECKPOINT="${PI05_CHECKPOINT:-$REPO_ROOT/checkpoints/robocasa/pi05_pretrain_human300_pytorch}"
GPUS="${FULL_CONTEXT_PI05_TABLE1_GPUS:-1,4,5,6,7}"
PORTS="${FULL_CONTEXT_PI05_TABLE1_PORTS:-19701,19704,19705,19706,19707}"

usage() {
    echo "usage: $0 preflight | start | status | aggregate | stop" >&2
}

metadata_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib, json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

server_env() {
    env \
        PI05_CONTROL_DIR="$CONTROL_DIR" \
        PI05_FLOW_STEPS=4 \
        PI05_FULL_CONTEXT_PLAN="$PLAN" \
        PI05_FULL_CONTEXT_HESSIAN_W4="$HESSIAN" \
        PI05_FULL_CONTEXT_PACK_DIR="$PACK_DIR" \
        PI05_FULL_CONTEXT_BUFFER="$BUFFER" \
        "$@"
}

parse_resources() {
    IFS=, read -r -a GPU_ARRAY <<<"$GPUS"
    IFS=, read -r -a PORT_ARRAY <<<"$PORTS"
    if [[ "${#GPU_ARRAY[@]}" -lt 1 || "${#GPU_ARRAY[@]}" -ne "${#PORT_ARRAY[@]}" ]]; then
        echo "GPU and port lists must be non-empty and have equal lengths" >&2
        return 1
    fi
    local seen=" " gpu port
    for gpu in "${GPU_ARRAY[@]}"; do
        if [[ ! "$gpu" =~ ^[0-7]$ || "$seen" == *" $gpu "* ]]; then
            echo "invalid or duplicate GPU: $gpu" >&2
            return 1
        fi
        seen+="$gpu "
    done
    seen=" "
    for port in "${PORT_ARRAY[@]}"; do
        if [[ ! "$port" =~ ^[0-9]+$ || "$seen" == *" $port "* ]]; then
            echo "invalid or duplicate port: $port" >&2
            return 1
        fi
        seen+="$port "
    done
}

instance_for_gpu() {
    printf 'dypac_pi05_table1_g%s' "$1"
}

preflight() {
    parse_resources
    local required=(
        "$PLAN" "$HESSIAN" "$HESSIAN.json" "$PACK_DIR/manifest.json"
        "$BUFFER" "$CHECKPOINT/model.safetensors" "$MANIFEST_TOOL" "$AGGREGATOR"
    )
    local path
    for path in "${required[@]}"; do
        [[ -f "$path" ]] || { echo "missing required artifact: $path" >&2; return 1; }
    done
    FULL_CONTEXT_PI05_QUICK_ROOT="$REPO_ROOT/runs/full_context_v2/pi05_quick" \
    FULL_CONTEXT_PI05_CANDIDATE_PLAN="$PLAN" \
    FULL_CONTEXT_PI05_CANDIDATE_HESSIAN="$HESSIAN" \
        "$REPO_ROOT/scripts/run_full_context_pi05_quick.sh" preflight
    "$ROBOCASA_PY" - "$PACK_DIR/manifest.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "kind": "hessian_w4_identity_pack",
    "model_adapter": "pi05",
    "permutation": False,
    "row_rotation": "identity",
}
bad = {key: (value.get(key), wanted) for key, wanted in expected.items() if value.get(key) != wanted}
if bad:
    raise SystemExit(f"identity-pack preflight failed: {bad}")
PY
    echo "DyPAC-VLA pi0.5 Table-1 artifact preflight passed"
}

start_servers() {
    parse_resources
    mkdir -p "$CONTROL_DIR" "$WORKER_CONTROL" "$RESULTS_DIR" "$PREFLIGHT_DIR"
    local pids=() started_instances=() index gpu port instance failed=0 pid=""
    for index in "${!GPU_ARRAY[@]}"; do
        gpu="${GPU_ARRAY[$index]}"
        port="${PORT_ARRAY[$index]}"
        instance="$(instance_for_gpu "$gpu")"
        pid=""
        if [[ -f "$CONTROL_DIR/$instance.pid" ]]; then
            pid="$(<"$CONTROL_DIR/$instance.pid")"
        fi
        if [[ "$pid" =~ ^[0-9]+$ ]] \
            && kill -0 "$pid" 2>/dev/null \
            && [[ -s "$CONTROL_DIR/$instance.runtime.json" ]] \
            && curl -fsS --max-time 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
            echo "reuse healthy server=$instance pid=$pid port=$port"
            continue
        fi
        server_env "$SERVER_MANAGER" start "$CONFIG_ID" "$gpu" "$port" "$instance" \
            >"$CONTROL_DIR/$instance.start.log" 2>&1 &
        pids+=("$!")
        started_instances+=("$instance")
    done
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            failed=1
            instance="${started_instances[$index]}"
            echo "server startup failed: $instance" >&2
            tail -n 100 "$CONTROL_DIR/$instance.start.log" >&2 || true
        fi
    done
    if [[ "$failed" != 0 ]]; then
        stop_servers
        return 1
    fi
}

smoke() {
    parse_resources
    local gpu="${GPU_ARRAY[0]}" port="${PORT_ARRAY[0]}" instance runtime hash output
    instance="$(instance_for_gpu "$gpu")"
    runtime="$CONTROL_DIR/$instance.runtime.json"
    hash="$(metadata_hash "$runtime")"
    output="$PREFLIGHT_DIR/runtime_smoke_${hash:0:12}.jsonl"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" \
        --host 127.0.0.1 \
        --port "$port" \
        --config-id "$CONFIG_ID" \
        --task-set atomic_seen \
        --tasks OpenDrawer \
        --trial-seeds 0 \
        --split target \
        --replan-steps 16 \
        --flow-steps 4 \
        --egl-device "$gpu" \
        --expected-server-metadata-sha256 "$hash" \
        --max-steps 1 \
        --resume-dir "$PREFLIGHT_DIR" \
        --out "$output" >"$PREFLIGHT_DIR/runtime_smoke.log" 2>&1
    echo "runtime smoke passed: $output"
}

manifest_args() {
    parse_resources
    MANIFEST_ARGS=(
        --out "$MANIFEST"
        --run-dir "$RUN_ROOT"
        --plan "$PLAN"
        --hessian "$HESSIAN"
        --pack-dir "$PACK_DIR"
        --checkpoint "$CHECKPOINT/model.safetensors"
        --calibration-buffer "$BUFFER"
    )
    local index gpu port instance runtime
    for index in "${!GPU_ARRAY[@]}"; do
        gpu="${GPU_ARRAY[$index]}"
        port="${PORT_ARRAY[$index]}"
        instance="$(instance_for_gpu "$gpu")"
        runtime="$CONTROL_DIR/$instance.runtime.json"
        MANIFEST_ARGS+=(--server "$instance,$gpu,$port,$runtime")
    done
}

freeze_manifest() {
    manifest_args
    if [[ -f "$MANIFEST" ]]; then
        "$ROBOCASA_PY" "$MANIFEST_TOOL" "${MANIFEST_ARGS[@]}" --verify
    else
        "$ROBOCASA_PY" "$MANIFEST_TOOL" "${MANIFEST_ARGS[@]}"
    fi
}

start_one_worker() {
    local gpu="$1" port="$2" shard="$3" shard_count="$4" seeds="$5" suffix="$6"
    local instance runtime hash worker_id pid_file log_file pid=""
    instance="$(instance_for_gpu "$gpu")"
    runtime="$CONTROL_DIR/$instance.runtime.json"
    hash="$(metadata_hash "$runtime")"
    worker_id="dypac_g${gpu}_shard${shard}_${suffix}"
    pid_file="$WORKER_CONTROL/$worker_id.pid"
    log_file="$WORKER_CONTROL/$worker_id.log"
    if [[ -f "$pid_file" ]]; then pid="$(<"$pid_file")"; fi
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "reuse worker=$worker_id pid=$pid"
        return
    fi
    nohup setsid env PI05_FLOW_STEPS=4 \
        "$WORKER" "$CONFIG_ID" "$port" "$gpu" "$shard" "$shard_count" \
        "$seeds" "$worker_id" "$hash" "$RUN_ROOT" \
        >"$log_file" 2>&1 </dev/null &
    pid=$!
    echo "$pid" >"$pid_file"
    echo "started worker=$worker_id pid=$pid seeds=$seeds shard=$shard/$shard_count gpu=$gpu"
}

start_workers() {
    parse_resources
    local shard_count="${#GPU_ARRAY[@]}" index gpu port
    for index in "${!GPU_ARRAY[@]}"; do
        gpu="${GPU_ARRAY[$index]}"
        port="${PORT_ARRAY[$index]}"
        start_one_worker "$gpu" "$port" "$index" "$shard_count" 0-24 lo
        start_one_worker "$gpu" "$port" "$index" "$shard_count" 25-49 hi
    done
}

status() {
    parse_resources
    "$ROBOCASA_PY" - "$RUN_ROOT" "$CONFIG_ID" "$CONTROL_DIR" "$WORKER_CONTROL" "$GPUS" "$PORTS" <<'PY'
import glob, json, os, sys, urllib.request
from collections import Counter
from pathlib import Path

run_dir, config, control, worker_control = map(Path, sys.argv[1:5])
gpus = sys.argv[5].split(",")
ports = sys.argv[6].split(",")
keys = set()
counts = Counter()
duplicates = []
for name in glob.glob(str(run_dir / "results" / config / "*.jsonl")):
    for line_number, line in enumerate(open(name, encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["task_set"]), str(row["task"]), int(row["seed"]))
        if key in keys:
            duplicates.append((name, line_number, key))
        keys.add(key)
        counts[str(row["task_set"])] += 1
if duplicates:
    raise SystemExit(f"duplicate formal keys: {duplicates[:3]}")

def alive(path):
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False

workers = sorted(Path(worker_control).glob("*.pid")) if Path(worker_control).is_dir() else []
servers = []
for gpu, port in zip(gpus, ports):
    instance = f"dypac_pi05_table1_g{gpu}"
    pid_path = Path(control) / f"{instance}.pid"
    healthy = False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as reply:
            healthy = reply.status == 200
    except Exception:
        pass
    servers.append({"instance": instance, "pid_alive": alive(pid_path), "healthy": healthy})
print(json.dumps({
    "completed": len(keys),
    "expected": 2500,
    "remaining": 2500 - len(keys),
    "by_task_set": dict(sorted(counts.items())),
    "result_feedback_withheld_until_complete": len(keys) < 2500,
    "workers_alive": sum(alive(path) for path in workers),
    "workers_registered": len(workers),
    "servers": servers,
}, indent=2, sort_keys=True))
PY
}

aggregate() {
    "$ROBOCASA_PY" "$AGGREGATOR" \
        --run-dir "$RUN_ROOT" \
        --out "$AGGREGATE"
}

stop_workers() {
    local pid_file pid
    [[ -d "$WORKER_CONTROL" ]] || return 0
    for pid_file in "$WORKER_CONTROL"/*.pid; do
        [[ -f "$pid_file" ]] || continue
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        fi
    done
}

stop_servers() {
    parse_resources
    local gpu instance
    for gpu in "${GPU_ARRAY[@]}"; do
        instance="$(instance_for_gpu "$gpu")"
        server_env "$SERVER_MANAGER" stop "$instance" || true
    done
}

run_all() {
    mkdir -p "$CONTROL_DIR"
    # Release the short orchestration lock before spawning long-lived services,
    # otherwise a server can inherit the descriptor and hold it for the run.
    exec 9>"$CONTROL_DIR/start.v2.lock"
    flock -n 9 || { echo "another pi0.5 Table-1 start holds the lock" >&2; exit 1; }
    preflight
    flock -u 9
    exec 9>&-
    start_servers
    smoke
    freeze_manifest
    start_workers
    status
}

case "${1:-}" in
    preflight) preflight ;;
    start) run_all ;;
    status) status ;;
    aggregate) aggregate ;;
    stop) stop_workers; stop_servers ;;
    *) usage; exit 2 ;;
esac
