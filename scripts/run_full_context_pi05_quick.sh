#!/usr/bin/env bash
set -euo pipefail

# Frozen π0.5 advancement gate.  Both the historical GDSQ-VLA main and the
# selector-free candidate run with the model-native 10-step flow solver.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_full_context_quick.py"
QUICK_ROOT="${FULL_CONTEXT_PI05_QUICK_ROOT:-$REPO_ROOT/runs/full_context_v1/pi05/quick}"
CONTROL_DIR="$QUICK_ROOT/control"
RESULTS_DIR="$QUICK_ROOT/results"
GDSQ_GPU="${FULL_CONTEXT_PI05_GDSQ_GPU:-5}"
CANDIDATE_GPU="${FULL_CONTEXT_PI05_CANDIDATE_GPU:-6}"
EGL_POOL="${FULL_CONTEXT_PI05_EGL_POOL:-4,7}"
GDSQ_PORT="${FULL_CONTEXT_PI05_GDSQ_PORT:-19655}"
CANDIDATE_PORT="${FULL_CONTEXT_PI05_CANDIDATE_PORT:-19656}"
GDSQ_INSTANCE="full_context_quick_gdsq"
CANDIDATE_INSTANCE="full_context_quick_candidate"
# Seed groups per task; the frozen gate uses 50-59.  Expansion waves override
# this (e.g. "60-64 65-69") and aggregate separately in the combined report.
SEED_GROUPS="${FULL_CONTEXT_PI05_SEED_GROUPS:-50-54 55-59}"
# The quick baseline must be the Table-1 runtime-selector main. The legacy
# ohb-only static path is admissible only with a per-request bitwise
# equivalence artifact proving it reproduces that main in this quick scope.
GDSQ_CONFIG="${FULL_CONTEXT_PI05_GDSQ_CONFIG:-gdsq_vla_runtime_selector}"
GDSQ_EQUIVALENCE_ARTIFACT="${FULL_CONTEXT_PI05_GDSQ_EQUIVALENCE_ARTIFACT:-}"
CANDIDATE_CONFIG="full_context_w4a8_dynamic_profile"
CANDIDATE_PLAN="${FULL_CONTEXT_PI05_CANDIDATE_PLAN:-$REPO_ROOT/runs/full_context_v1/pi05/round1/pi05_full_context_round1_frozen.json}"
CANDIDATE_HESSIAN="${FULL_CONTEXT_PI05_CANDIDATE_HESSIAN:-$REPO_ROOT/runs/full_context_v1/pi05/deployment/hessian_w4.npz}"
GDSQ_FLOW10_A8="$REPO_ROOT/runs/full_context_v1/pi05/gdsq_main_flow10/a8_scales.npz"
GDSQ_PLAN="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/plans/pi05_gdsq_cscka_16to1_d4.final_plan.json"
QUANTVLA_TABLE1_BYTES=1490466816

usage() {
    echo "usage: $0 run | preflight | status | aggregate | stop" >&2
}

preflight_baseline_requirement() {
    if [[ "$GDSQ_CONFIG" == "gdsq_vla_ohb_only" && -z "$GDSQ_EQUIVALENCE_ARTIFACT" ]]; then
        echo "pi0.5 quick baseline requirement violated: gdsq_vla_ohb_only needs" >&2
        echo "a bitwise equivalence artifact (FULL_CONTEXT_PI05_GDSQ_EQUIVALENCE_ARTIFACT)" >&2
        echo "or the Table-1 runtime-selector main (default gdsq_vla_runtime_selector)." >&2
        return 1
    fi
    return 0
}

preflight_candidate() {
    [[ -f "$GDSQ_FLOW10_A8" && -f "$GDSQ_FLOW10_A8.json" ]] || {
        echo "missing frozen 10-step GDSQ main A8 artifact: $GDSQ_FLOW10_A8" >&2
        return 1
    }
    PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" - "$CANDIDATE_PLAN" "$QUANTVLA_TABLE1_BYTES" \
        "$GDSQ_FLOW10_A8" "$GDSQ_PLAN" "$CANDIDATE_HESSIAN" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from quantvla_full_context import PROTOCOL, require_protocol_attestation
from quantvla_cross_model_protocol import PROTOCOL as CROSS_MODEL_PROTOCOL

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

path = Path(sys.argv[1]).resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
meta = payload.get("meta") or {}
require_protocol_attestation(meta, source=str(path))
checks = {
    "frozen": meta.get("frozen") is True,
    "model": meta.get("model_adapter") == "pi05",
    "noise_a": meta.get("selection_noise") == "A",
    "dynamic_a8": meta.get("activation_mode") == "dynamic_a8",
    "no_selector": meta.get("runtime_selector") is False,
    "no_correction": meta.get("runtime_correction") is False,
    "w4_present": int(payload.get("quantized_w4_layers", 0)) > 0,
}
from quantvla_table1_bytes import (
    TABLE1_QUANTVLA_BYTES,
    table1_total_static_budget,
    table1_total_static_bytes,
)
table1_bytes = int(sys.argv[2])
static_budget = table1_total_static_budget("pi05")
static_total = table1_total_static_bytes(
    "pi05", int(payload.get("total_bytes", static_budget + 1))
)
checks["anchor"] = table1_bytes == TABLE1_QUANTVLA_BYTES["pi05"]
checks["bytes"] = static_total <= static_budget
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"candidate quick preflight failed: {failed}")
a8_path = Path(sys.argv[3]).resolve()
a8_sidecar = json.loads(Path(str(a8_path) + ".json").read_text(encoding="utf-8"))
a8_meta = a8_sidecar.get("metadata") or {}
plan_hash = digest(Path(sys.argv[4]).resolve())
a8_checks = {
    "flow_steps": int(a8_meta.get("denoising_steps", -1)) == 4,
    "plan": a8_meta.get("plan_sha256") == plan_hash,
    "wrapped": int(a8_meta.get("wrapped_layers", -1)) == 80,
    "buffer": a8_meta.get("calibration_buffer_sha256")
        == CROSS_MODEL_PROTOCOL["data"]["calibration_buffer"]["sha256"],
}
failed = [name for name, passed in a8_checks.items() if not passed]
if failed:
    raise SystemExit(f"GDSQ-main four-flow-step A8 preflight failed: {failed}")
hessian_path = Path(sys.argv[5]).resolve()
hessian_meta = json.loads(Path(str(hessian_path) + ".json").read_text(encoding="utf-8"))
hessian_checks = {
    "hash": hessian_meta.get("npz_sha256")
        == digest(hessian_path),
    "plan": hessian_meta.get("deployment_plan_sha256")
        == digest(path),
    "inventory": len(hessian_meta.get("layer_names") or [])
        == int(payload.get("quantized_w4_layers", -1)),
    "no_requantize": hessian_meta.get("requantized") is False,
    "buffer": hessian_meta.get("calibration_buffer_sha256")
        == CROSS_MODEL_PROTOCOL["data"]["calibration_buffer"]["sha256"],
}
failed = [name for name, passed in hessian_checks.items() if not passed]
if failed:
    raise SystemExit(f"candidate Hessian subset preflight failed: {failed}")
PY
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

server_env() {
    env \
        PI05_CONTROL_DIR="$CONTROL_DIR" \
        PI05_FLOW_STEPS=4 \
        PI05_FULL_CONTEXT_PLAN="$CANDIDATE_PLAN" \
        PI05_FULL_CONTEXT_HESSIAN_W4="$CANDIDATE_HESSIAN" \
        PI05_GDSQ_A8="$GDSQ_FLOW10_A8" \
        "$@"
}

start_servers() {
    mkdir -p "$CONTROL_DIR" "$RESULTS_DIR/$GDSQ_CONFIG" "$RESULTS_DIR/$CANDIDATE_CONFIG"
    local pids=() failed=0
    server_env "$SERVER_MANAGER" start "$GDSQ_CONFIG" "$GDSQ_GPU" "$GDSQ_PORT" "$GDSQ_INSTANCE" &
    pids+=("$!")
    server_env "$SERVER_MANAGER" start "$CANDIDATE_CONFIG" "$CANDIDATE_GPU" "$CANDIDATE_PORT" "$CANDIDATE_INSTANCE" &
    pids+=("$!")
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]]
}

stop_servers() {
    server_env "$SERVER_MANAGER" stop "$GDSQ_INSTANCE" || true
    server_env "$SERVER_MANAGER" stop "$CANDIDATE_INSTANCE" || true
}

launch_one() {
    local config="$1" port="$2" metadata_hash="$3" task_set="$4" task="$5"
    local seeds="$6" egl="$7" label="$8"
    local output="$RESULTS_DIR/$config/${task_set}_${task}_${seeds}.jsonl"
    local log="$CONTROL_DIR/$label.log"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" \
        --host 127.0.0.1 \
        --port "$port" \
        --config-id "$config" \
        --task-set "$task_set" \
        --tasks "$task" \
        --trial-seeds "$seeds" \
        --split target \
        --replan-steps 16 \
        --flow-steps 4 \
        --egl-device "$egl" \
        --expected-server-metadata-sha256 "$metadata_hash" \
        --resume-dir "$RESULTS_DIR/$config" \
        --out "$output" >"$log" 2>&1 &
    LAST_PID="$!"
}

run_workers() {
    local gdsq_hash candidate_hash
    gdsq_hash="$(runtime_hash "$CONTROL_DIR/$GDSQ_INSTANCE.runtime.json")"
    candidate_hash="$(runtime_hash "$CONTROL_DIR/$CANDIDATE_INSTANCE.runtime.json")"
    local tasks=(
        "atomic_seen,CloseFridge"
        "atomic_seen,OpenDrawer"
        "composite_seen,LoadDishwasher"
        "composite_seen,PrepareCoffee"
        "composite_unseen,MakeIceLemonade"
    )
    IFS=, read -r -a egl_devices <<<"$EGL_POOL"
    [[ "${#egl_devices[@]}" -gt 0 ]] || { echo "empty EGL pool" >&2; return 1; }
    local pids=() spec task_set task seeds config port hash egl label index=0
    for config in "$GDSQ_CONFIG" "$CANDIDATE_CONFIG"; do
        if [[ "$config" == "$GDSQ_CONFIG" ]]; then
            port="$GDSQ_PORT"; hash="$gdsq_hash"
        else
            port="$CANDIDATE_PORT"; hash="$candidate_hash"
        fi
        for spec in "${tasks[@]}"; do
            IFS=, read -r task_set task <<<"$spec"
            for seeds in $SEED_GROUPS; do
                egl="${egl_devices[$((index % ${#egl_devices[@]}))]}"
                label="${config}_${task}_${seeds}"
                launch_one "$config" "$port" "$hash" "$task_set" "$task" "$seeds" "$egl" "$label"
                pids+=("$LAST_PID")
                index=$((index + 1))
            done
        done
    done
    local failed=0 pid
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]]
}

status() {
    "$ROBOCASA_PY" - "$RESULTS_DIR" "$GDSQ_CONFIG" "$CANDIDATE_CONFIG" <<'PY'
import json
from collections import Counter
from pathlib import Path
import sys

counts = Counter()
root = Path(sys.argv[1])
for path in root.glob("**/*.jsonl"):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "complete":
                counts[str(row.get("config"))] += 1
print(" ".join(f"{name}={counts[name]}/50" for name in sys.argv[2:]))
PY
}

aggregate() {
    "$ROBOCASA_PY" "$AGGREGATOR" \
        --model pi05 \
        --main-dir "$RESULTS_DIR" \
        --candidate-dir "$RESULTS_DIR" \
        --main-config "$GDSQ_CONFIG" \
        --candidate-config "$CANDIDATE_CONFIG" \
        --candidate-plan "$CANDIDATE_PLAN" \
        --quantvla-table1-bytes "$QUANTVLA_TABLE1_BYTES" \
        --out "$QUICK_ROOT/aggregate.json"
}

run_all() {
    mkdir -p "$CONTROL_DIR"
    exec 9>"$CONTROL_DIR/run.lock"
    flock -n 9 || { echo "another π0.5 quick run holds the lock" >&2; exit 1; }
    preflight_candidate
    trap stop_servers EXIT INT TERM HUP
    start_servers
    run_workers
    if [[ "$SEED_GROUPS" == "50-54 55-59" ]]; then
        aggregate
    fi
    stop_servers
    trap - EXIT INT TERM HUP
}

case "${1:-}" in
    run) run_all ;;
    preflight) preflight_baseline_requirement; preflight_candidate ;;
    status) status ;;
    aggregate) aggregate ;;
    stop) stop_servers ;;
    *) usage; exit 2 ;;
esac
