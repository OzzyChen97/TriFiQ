#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROOT="$REPO_ROOT/runs/full_context_v2/pi05_fcp_diagnostic"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
FREEZER="$REPO_ROOT/scripts/tools/freeze_pi05_fcp_tail_supervisor_erratum.py"
AGGREGATOR="$REPO_ROOT/scripts/run_pi05_fcp_diagnostic.sh"
CONTROL="$ROOT/control"
WORKER_CONTROL="$CONTROL/workers"
PREFLIGHT="$ROOT/preflight/tail_extra_gpu1"
ROLLOUTS="$ROOT/rollouts"
CONFIG_ID="full_context_w4a8_dynamic_profile"
PACK_DIR="$REPO_ROOT/runs/archive_previous_versions/errorfold_v4_iter/calibration/pi05_full_w4/identity_pack"
BUFFER="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
SHARD_COUNT=9

plan_for() {
    case "$1" in
        single_best) echo "$REPO_ROOT/runs/full_context_v2/pi05_p2/proposals_dynamic/single_best.json" ;;
        two_best) echo "$REPO_ROOT/runs/full_context_v2/pi05_p2/proposals_dynamic/two_best.json" ;;
        *) return 2 ;;
    esac
}
hessian_for() { echo "$ROOT/artifacts/$1/hessian_w4.npz"; }
semantic_hash() { jq -r '.openpi_runtime.semantic_metadata_sha256' "$1"; }

server_env() {
    local arm="$1"; shift
    env PI05_CONTROL_DIR="$CONTROL" PI05_FLOW_STEPS=4 \
        PI05_FULL_CONTEXT_PLAN="$(plan_for "$arm")" \
        PI05_FULL_CONTEXT_HESSIAN_W4="$(hessian_for "$arm")" \
        PI05_FULL_CONTEXT_PACK_DIR="$PACK_DIR" PI05_FULL_CONTEXT_BUFFER="$BUFFER" "$@"
}

start_extra_servers() {
    local free
    free="$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    [[ "$free" =~ ^[0-9]+$ && "$free" -ge 30000 ]] || {
        echo "GPU 1 has only ${free:-unknown} MiB free; refusing two-server load" >&2; return 1;
    }
    server_env single_best "$SERVER_MANAGER" start "$CONFIG_ID" 1 19829 pi05_fcp_single_best_x1
    server_env two_best "$SERVER_MANAGER" start "$CONFIG_ID" 1 19849 pi05_fcp_two_best_x1
}

smoke_one() {
    local arm="$1" instance="$2" port="$3" runtime hash output
    runtime="$CONTROL/$instance.runtime.json"; hash="$(semantic_hash "$runtime")"
    output="$PREFLIGHT/$arm.jsonl"; mkdir -p "$PREFLIGHT/$arm"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" --host 127.0.0.1 --port "$port" \
        --config-id "$CONFIG_ID" --task-set atomic_seen --tasks OpenDrawer \
        --trial-seeds 5 --split target --replan-steps 16 --flow-steps 4 \
        --egl-device 0 --expected-server-metadata-sha256 "$hash" --max-steps 1 \
        --resume-dir "$PREFLIGHT/$arm" --out "$output" >"$PREFLIGHT/$arm.log" 2>&1
}

worker_alive() {
    local worker_id pid_file pid command
    worker_id="$1"; pid_file="$WORKER_CONTROL/$worker_id.pid"
    [[ -f "$pid_file" ]] || return 1
    pid="$(<"$pid_file")"; [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    command="$(ps -p "$pid" -o args=)"
    [[ "$command" == *run_pi05_formal_worker_seeded.sh*"$ROOT"* ]]
}

launch_worker() {
    local arm="$1" shard="$2" instance="$3" egl="$4" port="$5"
    local runtime hash worker_id log pid
    runtime="$CONTROL/$instance.runtime.json"; hash="$(semantic_hash "$runtime")"
    worker_id="tail2_${arm}_s${shard}"; log="$WORKER_CONTROL/$worker_id.log"
    [[ ! -e "$WORKER_CONTROL/$worker_id.pid" && ! -e "$log" ]] || {
        echo "refusing duplicate tail launch: $worker_id" >&2; return 1;
    }
    nohup setsid env PI05_FLOW_STEPS=4 "$WORKER" "$CONFIG_ID" "$port" "$egl" \
        "$shard" "$SHARD_COUNT" 0-9 "$worker_id" "$hash" "$ROLLOUTS/$arm" \
        >"$log" 2>&1 </dev/null &
    pid=$!; echo "$pid" >"$WORKER_CONTROL/$worker_id.pid"
    echo "started $worker_id pid=$pid server=$instance egl=$egl"
}

any_formal_alive() {
    local path worker_id
    for path in "$WORKER_CONTROL"/clean1_*.pid "$WORKER_CONTROL"/tail2_*.pid; do
        [[ -f "$path" ]] || continue
        worker_id="$(basename "$path" .pid)"
        worker_alive "$worker_id" && return 0
    done
    return 1
}

check_all() {
    local logs completed
    logs="$(find "$WORKER_CONTROL" -maxdepth 1 -name 'clean1_*.log' | wc -l)"
    [[ "$logs" -eq 16 ]] || { echo "clean1 log count $logs != 16" >&2; return 1; }
    logs="$(find "$WORKER_CONTROL" -maxdepth 1 -name 'tail2_*.log' | wc -l)"
    [[ "$logs" -eq 11 ]] || { echo "tail2 log count $logs != 11" >&2; return 1; }
    if rg -n "Traceback|Segmentation fault|CUDA out of memory|CUDNN_STATUS|CUBLAS_STATUS|unable to find an engine" \
        "$WORKER_CONTROL"/clean1_*.log "$WORKER_CONTROL"/tail2_*.log; then
        echo "formal clean/tail logs contain a failure" >&2; return 1
    fi
    completed="$({ rg -l 'formal seeded worker complete:' "$WORKER_CONTROL"/clean1_*.log || true; } | wc -l)"
    [[ "$completed" -eq 16 ]] || { echo "clean1 completion count $completed != 16" >&2; return 1; }
    completed="$({ rg -l 'formal seeded worker complete:' "$WORKER_CONTROL"/tail2_*.log || true; } | wc -l)"
    [[ "$completed" -eq 11 ]] || { echo "tail2 completion count $completed != 11" >&2; return 1; }
}

supervise() {
    while any_formal_alive; do sleep 30; done
    check_all
    "$AGGREGATOR" status
    "$AGGREGATOR" aggregate
}

recover() {
    mkdir -p "$PREFLIGHT"
    start_extra_servers
    smoke_one single_best pi05_fcp_single_best_x1 19829
    smoke_one two_best pi05_fcp_two_best_x1 19849
    "$OPENPI_PY" "$FREEZER"
    launch_worker single_best 8 pi05_fcp_single_best_x1 0 19829
    launch_worker two_best 8 pi05_fcp_two_best_x1 1 19849
    nohup setsid "$0" supervise >"$CONTROL/tail_schedule_supervisor_v2.log" 2>&1 </dev/null &
    echo $! >"$CONTROL/tail_schedule_supervisor_v2.pid"
    "$AGGREGATOR" status
}

case "${1:-}" in
    recover) recover ;;
    supervise) supervise ;;
    *) echo "usage: $0 recover | supervise" >&2; exit 2 ;;
esac
