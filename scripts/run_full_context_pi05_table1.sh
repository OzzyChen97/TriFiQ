#!/usr/bin/env bash
set -euo pipefail

# Frozen π0.5 Table-1 matrix.  This launcher is fail-closed on the fresh
# seeds-50..59 advancement gate and runs every row with the model-native
# 10-step flow solver.  GPU placement and worker counts are execution-only.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
MATERIALIZER="$REPO_ROOT/scripts/tools/materialize_full_context_table1.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_full_context_table1.py"

ROOT="${FULL_CONTEXT_PI05_TABLE1_ROOT:-$REPO_ROOT/runs/full_context_v1/pi05/table1}"
CONTROL_DIR="$ROOT/control"
WORKER_DIR="$CONTROL_DIR/workers"
RESULTS_DIR="$ROOT/results"
MANIFEST="$ROOT/manifest.json"
QUICK_REPORT="$REPO_ROOT/runs/full_context_v1/pi05/quick/aggregate.json"
PLAN="$REPO_ROOT/runs/full_context_v1/pi05/round1/pi05_full_context_round1_frozen.json"
ACTIVATION="$REPO_ROOT/runs/full_context_v1/pi05/a8_attribution/selection.json"
HESSIAN="$REPO_ROOT/runs/full_context_v1/pi05/deployment/hessian_w4.npz"
FULL_A8="$REPO_ROOT/runs/full_context_v1/pi05/calibration_flow10/a8_scales.npz"
GDSQ_A8="$REPO_ROOT/runs/full_context_v1/pi05/gdsq_main_flow10/a8_scales.npz"
CHECKPOINT="$REPO_ROOT/checkpoints/robocasa/pi05_pretrain_human300_pytorch/model.safetensors"

SERVERS=(
    "table1_fp16,fp16,4,19700"
    "table1_quantvla,quantvla_w4a8_paper,5,19701"
    "table1_gdsq,gdsq_vla_ohb_only,6,19702"
    "table1_candidate,full_context_w4a8_dynamic_profile,7,19703"
)

usage() {
    echo "usage: $0 prepare | run | status | aggregate | stop" >&2
}

server_env() {
    env \
        PI05_CONTROL_DIR="$CONTROL_DIR" \
        PI05_FLOW_STEPS=10 \
        PI05_FULL_CONTEXT_PLAN="$PLAN" \
        PI05_FULL_CONTEXT_HESSIAN_W4="$HESSIAN" \
        PI05_FULL_A8="$FULL_A8" \
        PI05_GDSQ_A8="$GDSQ_A8" \
        "$@"
}

runtime_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

require_quick_pass() {
    PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" - "$QUICK_REPORT" "$PLAN" <<'PY'
import json
from pathlib import Path
import sys

from quantvla_cross_model_protocol import sha256_file
from quantvla_full_context import require_protocol_attestation

quick_path = Path(sys.argv[1]).resolve()
plan_path = Path(sys.argv[2]).resolve()
quick = json.loads(quick_path.read_text(encoding="utf-8"))
require_protocol_attestation(quick, source=str(quick_path))
checks = {
    "model": quick.get("model_adapter") == "pi05",
    "quick_pass": quick.get("advance_to_table1") is True,
    "plan_lineage": quick.get("candidate_plan_sha256") == sha256_file(plan_path),
    "coverage": len(quick.get("paired_rows") or []) == 50,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"π0.5 Table-1 gate failed: {failed}")
PY
}

prepare() {
    require_quick_pass
    mkdir -p "$ROOT" "$CONTROL_DIR" "$WORKER_DIR" "$RESULTS_DIR"
    if [[ ! -f "$MANIFEST" ]]; then
        PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}" \
            "$ROBOCASA_PY" "$MATERIALIZER" \
            --model pi05 \
            --frozen-plan "$PLAN" \
            --activation-attribution "$ACTIVATION" \
            --quick-report "$QUICK_REPORT" \
            --checkpoint "$CHECKPOINT" \
            --hessian-w4 "$HESSIAN" \
            --out "$MANIFEST"
    fi
    PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" - "$MANIFEST" "$QUICK_REPORT" "$PLAN" <<'PY'
import json
from pathlib import Path
import sys

from quantvla_cross_model_protocol import sha256_file
from quantvla_full_context import require_protocol_attestation

manifest_path, quick_path, plan_path = map(lambda value: Path(value).resolve(), sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
require_protocol_attestation(manifest, source=str(manifest_path))
assert manifest.get("immutable") is True
assert manifest.get("result_feedback_allowed") is False
assert manifest.get("model_adapter") == "pi05"
assert manifest["quick_gate"]["sha256"] == sha256_file(quick_path)
assert manifest["candidate"]["plan"]["sha256"] == sha256_file(plan_path)
assert manifest["protocol"]["flow_steps"] == 10
assert manifest["protocol"]["episodes"] == 2500
PY
}

server_alive() {
    local instance="$1" port="$2" pid_file="$CONTROL_DIR/$instance.pid" pid
    [[ -f "$pid_file" ]] || return 1
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null \
        && curl -fsS --max-time 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1
}

start_servers() {
    local pids=() spec instance config gpu port failed=0
    for spec in "${SERVERS[@]}"; do
        IFS=, read -r instance config gpu port <<<"$spec"
        if server_alive "$instance" "$port"; then
            continue
        fi
        server_env "$SERVER_MANAGER" stop "$instance" || true
        server_env "$SERVER_MANAGER" start "$config" "$gpu" "$port" "$instance" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]]
}

stop_servers() {
    local spec instance _rest
    for spec in "${SERVERS[@]}"; do
        IFS=, read -r instance _rest <<<"$spec"
        server_env "$SERVER_MANAGER" stop "$instance" || true
    done
}

start_workers() {
    local spec instance config gpu port runtime hash shard seeds egl worker_id pid
    local pids=() failed=0 index=0
    for spec in "${SERVERS[@]}"; do
        IFS=, read -r instance config gpu port <<<"$spec"
        runtime="$CONTROL_DIR/$instance.runtime.json"
        hash="$(runtime_hash "$runtime")"
        for shard in 0 1 2 3 4; do
            for seeds in 0-24 25-49; do
                egl=$((index % 8))
                worker_id="${config}_s${shard}_${seeds}"
                PI05_FLOW_STEPS=10 nohup setsid "$WORKER" \
                    "$config" "$port" "$egl" "$shard" 5 "$seeds" \
                    "$worker_id" "$hash" "$ROOT" \
                    >"$WORKER_DIR/$worker_id.log" 2>&1 </dev/null &
                pid="$!"
                pids+=("$pid")
                index=$((index + 1))
            done
        done
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]]
}

aggregate() {
    PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$AGGREGATOR" \
        --manifest "$MANIFEST" \
        --candidate-dir "$RESULTS_DIR/full_context_w4a8_dynamic_profile" \
        --baseline "fp16=$RESULTS_DIR/fp16" \
        --baseline "quantvla_w4a8=$RESULTS_DIR/quantvla_w4a8_paper" \
        --baseline "gdsq_vla_main=$RESULTS_DIR/gdsq_vla_ohb_only" \
        --out "$ROOT/aggregate.json"
}

status() {
    "$ROBOCASA_PY" - "$RESULTS_DIR" <<'PY'
import json
from collections import Counter
from pathlib import Path
import sys

configs = ("fp16", "quantvla_w4a8_paper", "gdsq_vla_ohb_only", "full_context_w4a8_dynamic_profile")
counts = Counter()
for path in Path(sys.argv[1]).glob("*/*.jsonl"):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "complete":
            counts[str(row.get("config"))] += 1
print(" ".join(f"{config}={counts[config]}/2500" for config in configs))
PY
}

run_all() {
    prepare
    exec 9>"$CONTROL_DIR/run.lock"
    flock -n 9 || { echo "another π0.5 full-context Table-1 run holds the lock" >&2; exit 1; }
    trap stop_servers EXIT INT TERM HUP
    start_servers
    start_workers
    aggregate
    stop_servers
    trap - EXIT INT TERM HUP
}

case "${1:-}" in
    prepare) prepare ;;
    run) run_all ;;
    status) status ;;
    aggregate) aggregate ;;
    stop) stop_servers ;;
    *) usage; exit 2 ;;
esac
