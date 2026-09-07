#!/usr/bin/env bash
set -euo pipefail

# Prospective pi0.5 FCP mechanism diagnostic. Each arm uses a separate result
# root because the mature formal service keeps its attested runtime config ID.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
SUBSETTER="$REPO_ROOT/scripts/tools/materialize_full_context_hessian_subset.py"
MATERIALIZER="$REPO_ROOT/scripts/tools/materialize_pi05_fcp_diagnostic.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_pi05_fcp_diagnostic.py"

ROOT="${PI05_FCP_ROOT:-$REPO_ROOT/runs/full_context_v2/pi05_fcp_diagnostic}"
CONTROL="$ROOT/control"
WORKER_CONTROL="$CONTROL/workers"
PREFLIGHT="$ROOT/preflight"
ROLLOUTS="$ROOT/rollouts"
CONFIG_ID="full_context_w4a8_dynamic_profile"
PARENT_HESSIAN="$REPO_ROOT/runs/archive_previous_versions/errorfold_v4_iter/calibration/pi05_full_w4/hessian_w4.npz"
PACK_DIR="$REPO_ROOT/runs/archive_previous_versions/errorfold_v4_iter/calibration/pi05_full_w4/identity_pack"
BUFFER="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
SHARD_COUNT=9

# wave,arm,task-shard,gpu,port. The default distribution uses every requested
# GPU while leaving enough transient headroom to load the next packed model.
SCHEDULE=(
    "0,transferred_initializer,0,2,19800"
    "0,transferred_initializer,1,3,19801"
    "0,transferred_initializer,2,4,19802"
    "0,transferred_initializer,3,5,19803"
    "0,transferred_initializer,4,6,19804"
    "0,transferred_initializer,8,7,19808"
    "1,single_best,0,3,19820"
    "1,transferred_initializer,5,4,19805"
    "1,transferred_initializer,6,5,19806"
    "1,transferred_initializer,7,6,19807"
    "1,single_best,7,7,19827"
    "2,two_best,0,3,19840"
    "2,single_best,1,4,19821"
    "2,single_best,2,5,19822"
    "2,single_best,3,6,19823"
    "2,single_best,8,7,19828"
    "3,single_best,4,4,19824"
    "3,single_best,5,5,19825"
    "3,single_best,6,6,19826"
    "3,two_best,7,7,19847"
    "4,two_best,1,4,19841"
    "4,two_best,2,5,19842"
    "4,two_best,3,6,19843"
    "4,two_best,8,7,19848"
    "5,two_best,4,3,19844"
    "5,two_best,5,5,19845"
    "5,two_best,6,6,19846"
)

usage() {
    echo "usage: $0 prepare | preflight | start-servers [GPU_CSV] | finalize-start | status | aggregate | stop" >&2
}

plan_for() {
    case "$1" in
        transferred_initializer) echo "$REPO_ROOT/runs/full_context_v2/pi05_p2/interventions_round_main/context_base.json" ;;
        single_best) echo "$REPO_ROOT/runs/full_context_v2/pi05_p2/proposals_dynamic/single_best.json" ;;
        two_best) echo "$REPO_ROOT/runs/full_context_v2/pi05_p2/proposals_dynamic/two_best.json" ;;
        projected_anchor) echo "$REPO_ROOT/runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json" ;;
        *) return 2 ;;
    esac
}

hessian_for() {
    case "$1" in
        projected_anchor) echo "$REPO_ROOT/runs/full_context_v2/pi05_quick/artifacts/hessian_w4.npz" ;;
        transferred_initializer|single_best|two_best) echo "$ROOT/artifacts/$1/hessian_w4.npz" ;;
        *) return 2 ;;
    esac
}

instance_for() { printf 'pi05_fcp_%s_s%s' "$1" "$2"; }

metadata_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib,json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
print(hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest())
PY
}

server_env() {
    local arm="$1"; shift
    env \
        PI05_CONTROL_DIR="$CONTROL" \
        PI05_FLOW_STEPS=4 \
        PI05_FULL_CONTEXT_PLAN="$(plan_for "$arm")" \
        PI05_FULL_CONTEXT_HESSIAN_W4="$(hessian_for "$arm")" \
        PI05_FULL_CONTEXT_PACK_DIR="$PACK_DIR" \
        PI05_FULL_CONTEXT_BUFFER="$BUFFER" \
        "$@"
}

prepare() {
    mkdir -p "$ROOT/artifacts" "$CONTROL" "$WORKER_CONTROL" "$PREFLIGHT" "$ROLLOUTS"
    local arm
    for arm in transferred_initializer single_best two_best; do
        mkdir -p "$ROOT/artifacts/$arm"
        "$OPENPI_PY" "$SUBSETTER" \
            --parent "$PARENT_HESSIAN" \
            --plan "$(plan_for "$arm")" \
            --model pi05 \
            --out "$(hessian_for "$arm")"
    done
    "$OPENPI_PY" "$MATERIALIZER"
}

preflight() {
    local required=(
        "$ROOT/preregistration.json" "$PACK_DIR/manifest.json" "$BUFFER"
        "$REPO_ROOT/checkpoints/robocasa/pi05_pretrain_human300_pytorch/model.safetensors"
    )
    local arm path
    for arm in transferred_initializer single_best two_best projected_anchor; do
        required+=("$(plan_for "$arm")" "$(hessian_for "$arm")" "$(hessian_for "$arm").json")
    done
    for path in "${required[@]}"; do [[ -e "$path" ]] || { echo "missing preflight artifact: $path" >&2; return 1; }; done
    "$OPENPI_PY" "$MATERIALIZER" --verify
    echo "pi0.5 FCP plan/Hessian/code inventory preflight passed"
}

gpu_selected() {
    local gpu="$1" csv="$2"
    [[ ",$csv," == *",$gpu,"* ]]
}

server_healthy() {
    local instance="$1" port="$2" pid=""
    [[ -f "$CONTROL/$instance.pid" ]] && pid="$(<"$CONTROL/$instance.pid")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null \
        && [[ -s "$CONTROL/$instance.runtime.json" ]] \
        && curl -fsS --max-time 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1
}

start_one_server() {
    local arm="$1" shard="$2" gpu="$3" port="$4" instance free
    instance="$(instance_for "$arm" "$shard")"
    if server_healthy "$instance" "$port"; then
        echo "reuse healthy server=$instance gpu=$gpu port=$port"
        return
    fi
    free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ ! "$free" =~ ^[0-9]+$ || "$free" -lt 13000 ]]; then
        echo "GPU $gpu has only ${free:-unknown} MiB free; refusing unsafe server load for $instance" >&2
        return 1
    fi
    server_env "$arm" "$SERVER_MANAGER" start "$CONFIG_ID" "$gpu" "$port" "$instance"
}

start_servers() {
    local selected="${1:-2,3,4,5,6,7}" wave spec spec_wave arm shard gpu port
    mkdir -p "$CONTROL"
    for wave in 0 1 2 3 4 5; do
        local pids=() labels=() failed=0
        for spec in "${SCHEDULE[@]}"; do
            IFS=, read -r spec_wave arm shard gpu port <<<"$spec"
            [[ "$spec_wave" == "$wave" ]] || continue
            gpu_selected "$gpu" "$selected" || continue
            start_one_server "$arm" "$shard" "$gpu" "$port" \
                >"$CONTROL/$(instance_for "$arm" "$shard").start.log" 2>&1 &
            pids+=("$!"); labels+=("$(instance_for "$arm" "$shard")")
        done
        local index
        for index in "${!pids[@]}"; do
            if ! wait "${pids[$index]}"; then
                failed=1
                echo "server startup failed: ${labels[$index]}" >&2
                tail -n 100 "$CONTROL/${labels[$index]}.start.log" >&2 || true
            fi
        done
        [[ "$failed" == 0 ]] || return 1
    done
}

assert_all_servers() {
    local spec wave arm shard gpu port instance failed=0
    for spec in "${SCHEDULE[@]}"; do
        IFS=, read -r wave arm shard gpu port <<<"$spec"
        instance="$(instance_for "$arm" "$shard")"
        if ! server_healthy "$instance" "$port"; then
            echo "missing/unhealthy scheduled server: $instance gpu=$gpu port=$port" >&2
            failed=1
        fi
    done
    [[ "$failed" == 0 ]]
}

smoke_one() {
    local arm="$1" shard="$2" gpu="$3" port="$4" instance runtime hash output
    instance="$(instance_for "$arm" "$shard")"
    runtime="$CONTROL/$instance.runtime.json"
    hash="$(metadata_hash "$runtime")"
    mkdir -p "$PREFLIGHT/$arm"
    output="$PREFLIGHT/$arm/smoke_${hash:0:12}.jsonl"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" \
        --host 127.0.0.1 --port "$port" --config-id "$CONFIG_ID" \
        --task-set atomic_seen --tasks OpenDrawer --trial-seeds 0 --split target \
        --replan-steps 16 --flow-steps 4 --egl-device "$gpu" \
        --expected-server-metadata-sha256 "$hash" --max-steps 1 \
        --resume-dir "$PREFLIGHT/$arm" --out "$output" \
        >"$PREFLIGHT/$arm/smoke.log" 2>&1
    echo "smoke passed arm=$arm output=$output"
}

smoke_all() {
    local pids=() labels=(transferred_initializer single_best two_best) index failed=0
    smoke_one transferred_initializer 0 2 19800 & pids+=("$!")
    smoke_one single_best 1 4 19821 & pids+=("$!")
    smoke_one two_best 2 5 19842 & pids+=("$!")
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            failed=1
            echo "smoke failed: ${labels[$index]}" >&2
        fi
    done
    [[ "$failed" == 0 ]]
}

freeze_execution() {
    local args=(--freeze-runtime) spec wave arm shard gpu port instance
    for spec in "${SCHEDULE[@]}"; do
        IFS=, read -r wave arm shard gpu port <<<"$spec"
        instance="$(instance_for "$arm" "$shard")"
        args+=(--server "$arm,$instance,$gpu,$port,$CONTROL/$instance.runtime.json")
    done
    "$OPENPI_PY" "$MATERIALIZER" "${args[@]}"
}

start_worker() {
    local arm="$1" shard="$2" gpu="$3" port="$4" seeds="$5" suffix="$6"
    local instance runtime hash worker_id pid_file log_file pid=""
    instance="$(instance_for "$arm" "$shard")"
    runtime="$CONTROL/$instance.runtime.json"
    hash="$(metadata_hash "$runtime")"
    worker_id="${arm}_s${shard}_${suffix}"
    pid_file="$WORKER_CONTROL/$worker_id.pid"
    log_file="$WORKER_CONTROL/$worker_id.log"
    [[ -f "$pid_file" ]] && pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "reuse worker=$worker_id pid=$pid"
        return
    fi
    mkdir -p "$ROLLOUTS/$arm"
    nohup setsid env PI05_FLOW_STEPS=4 "$WORKER" "$CONFIG_ID" "$port" "$gpu" \
        "$shard" "$SHARD_COUNT" "$seeds" "$worker_id" "$hash" "$ROLLOUTS/$arm" \
        >"$log_file" 2>&1 </dev/null &
    pid=$!; echo "$pid" >"$pid_file"
    echo "started worker=$worker_id pid=$pid gpu=$gpu task_shard=$shard/$SHARD_COUNT seeds=$seeds"
}

start_workers() {
    mkdir -p "$WORKER_CONTROL"
    local spec wave arm shard gpu port
    for spec in "${SCHEDULE[@]}"; do
        IFS=, read -r wave arm shard gpu port <<<"$spec"
        start_worker "$arm" "$shard" "$gpu" "$port" 0-4 lo
        start_worker "$arm" "$shard" "$gpu" "$port" 5-9 hi
    done
}

finalize_start() {
    preflight
    assert_all_servers
    smoke_all
    freeze_execution
    start_workers
    status
}

status() {
    "$ROBOCASA_PY" - "$ROLLOUTS" "$WORKER_CONTROL" <<'PY'
import json, os, sys
from collections import Counter
from pathlib import Path
root=Path(sys.argv[1]); control=Path(sys.argv[2]); config="full_context_w4a8_dynamic_profile"
summary={}
for arm in ("transferred_initializer","single_best","two_best"):
    keys=set(); duplicates=[]; successes=0; splits=Counter()
    for path in sorted((root/arm/"results"/config).glob("*.jsonl")):
        for number,line in enumerate(path.read_text().splitlines(),1):
            if not line.strip(): continue
            row=json.loads(line)
            if row.get("config") != config or row.get("status") != "complete": continue
            key=(str(row.get("task_set")),str(row.get("task")),int(row.get("seed")))
            if key in keys: duplicates.append((str(path),number,key))
            keys.add(key); successes += int(bool(row.get("success"))); splits[key[0]] += 1
    summary[arm]={"completed":len(keys),"expected":500,"remaining":500-len(keys),"successes_withheld":None if len(keys)<500 else successes,"by_split":dict(splits),"duplicates":len(duplicates)}
def alive(path):
    try: os.kill(int(path.read_text().strip()),0); return True
    except (OSError,ValueError): return False
pids=list(control.glob("*.pid")) if control.is_dir() else []
print(json.dumps({"arms":summary,"workers_alive":sum(alive(p) for p in pids),"workers_registered":len(pids),"result_feedback_withheld_until_complete":any(v["completed"]<500 for v in summary.values())},indent=2,sort_keys=True))
PY
}

aggregate() {
    "$ROBOCASA_PY" "$AGGREGATOR" \
        --preregistration "$ROOT/preregistration.json" \
        --execution-manifest "$ROOT/execution_manifest.json" \
        --results-root "$ROLLOUTS" \
        --raw-out "$ROOT/raw_aggregate.json" \
        --out "$ROOT/experiment_summary.json"
}

stop_workers() {
    local pid_file pid command
    for pid_file in "$WORKER_CONTROL"/*.pid; do
        [[ -f "$pid_file" ]] || continue
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            command="$(ps -p "$pid" -o args=)"
            if [[ "$command" == *"run_pi05_formal_worker_seeded.sh"*"$ROOT"* ]]; then
                kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
            fi
        fi
    done
}

stop_servers() {
    local spec wave arm shard gpu port instance
    for spec in "${SCHEDULE[@]}"; do
        IFS=, read -r wave arm shard gpu port <<<"$spec"
        instance="$(instance_for "$arm" "$shard")"
        server_env "$arm" "$SERVER_MANAGER" stop "$instance" || true
    done
}

case "${1:-}" in
    prepare) prepare ;;
    preflight) preflight ;;
    start-servers) start_servers "${2:-2,3,4,5,6,7}" ;;
    finalize-start) finalize_start ;;
    status) status ;;
    aggregate) aggregate ;;
    stop) stop_workers; stop_servers ;;
    *) usage; exit 2 ;;
esac
