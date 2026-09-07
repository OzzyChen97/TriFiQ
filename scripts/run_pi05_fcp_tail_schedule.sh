#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROOT="$REPO_ROOT/runs/full_context_v2/pi05_fcp_diagnostic"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
FREEZER="$REPO_ROOT/scripts/tools/freeze_pi05_fcp_tail_schedule_adjustment.py"
AGGREGATOR="$REPO_ROOT/scripts/run_pi05_fcp_diagnostic.sh"
CONTROL="$ROOT/control"
WORKER_CONTROL="$CONTROL/workers"
ROLLOUTS="$ROOT/rollouts"
CONFIG_ID="full_context_w4a8_dynamic_profile"
SHARD_COUNT=9

# arm,task-shard,server-shard,EGL-GPU,port
IMMEDIATE=(
    "transferred_initializer,6,0,2,19800" "transferred_initializer,7,1,3,19801"
    "transferred_initializer,8,2,4,19802"
    "single_best,5,2,5,19822" "single_best,6,3,6,19823"
    "single_best,7,7,7,19827"
    "two_best,5,2,5,19842" "two_best,6,3,6,19843"
    "two_best,7,7,7,19847"
)
# predecessor-worker,arm,task-shard,server-shard,EGL-GPU,port
DELAYED=(
    "clean1_single_best_s0,single_best,8,0,3,19820"
    "clean1_two_best_s0,two_best,8,0,3,19840"
)

instance_for() { printf 'pi05_fcp_%s_s%s' "$1" "$2"; }
semantic_hash() { jq -r '.openpi_runtime.semantic_metadata_sha256' "$1"; }

worker_alive() {
    local worker_id="$1" pid_file="$WORKER_CONTROL/$worker_id.pid" pid
    [[ -f "$pid_file" ]] || return 1
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

validate_completed_worker() {
    local worker_id="$1" log="$WORKER_CONTROL/$worker_id.log"
    [[ -f "$log" ]] || { echo "missing predecessor log: $log" >&2; return 1; }
    if rg -n "Traceback|Segmentation fault|CUDA out of memory|CUDNN_STATUS|CUBLAS_STATUS|unable to find an engine" "$log"; then
        echo "predecessor failed: $worker_id" >&2; return 1
    fi
    rg -q 'formal seeded worker complete:' "$log" || { echo "predecessor incomplete: $worker_id" >&2; return 1; }
}

launch_worker() {
    local arm="$1" shard="$2" server_shard="$3" egl="$4" port="$5"
    local instance runtime hash worker_id log pid
    instance="$(instance_for "$arm" "$server_shard")"; runtime="$CONTROL/$instance.runtime.json"
    hash="$(semantic_hash "$runtime")"; worker_id="tail2_${arm}_s${shard}"
    log="$WORKER_CONTROL/$worker_id.log"
    [[ ! -e "$WORKER_CONTROL/$worker_id.pid" && ! -e "$log" ]] || {
        echo "refusing duplicate tail launch: $worker_id" >&2; return 1;
    }
    nohup setsid env PI05_FLOW_STEPS=4 "$WORKER" "$CONFIG_ID" "$port" "$egl" \
        "$shard" "$SHARD_COUNT" 0-9 "$worker_id" "$hash" "$ROLLOUTS/$arm" \
        >"$log" 2>&1 </dev/null &
    pid=$!; echo "$pid" >"$WORKER_CONTROL/$worker_id.pid"
    echo "started $worker_id pid=$pid server=$instance egl=$egl"
}

launch_immediate() {
    local row arm shard server_shard egl port
    for row in "${IMMEDIATE[@]}"; do
        IFS=, read -r arm shard server_shard egl port <<<"$row"
        launch_worker "$arm" "$shard" "$server_shard" "$egl" "$port"
    done
}

launch_ready_delayed() {
    local row predecessor arm shard server_shard egl port worker_id
    for row in "${DELAYED[@]}"; do
        IFS=, read -r predecessor arm shard server_shard egl port <<<"$row"
        worker_id="tail2_${arm}_s${shard}"
        [[ -e "$WORKER_CONTROL/$worker_id.pid" ]] && continue
        if ! worker_alive "$predecessor"; then
            validate_completed_worker "$predecessor"
            launch_worker "$arm" "$shard" "$server_shard" "$egl" "$port"
        fi
    done
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
    while true; do
        launch_ready_delayed
        delayed_count="$(find "$WORKER_CONTROL" -maxdepth 1 -name 'tail2_*.pid' | wc -l)"
        if [[ "$delayed_count" -eq 11 ]] && ! any_formal_alive; then break; fi
        sleep 30
    done
    check_all
    "$AGGREGATOR" status
    "$AGGREGATOR" aggregate
}

recover() {
    "$OPENPI_PY" "$FREEZER"
    launch_immediate
    nohup setsid "$0" supervise >"$CONTROL/tail_schedule_supervisor.log" 2>&1 </dev/null &
    echo $! >"$CONTROL/tail_schedule_supervisor.pid"
    "$AGGREGATOR" status
}

case "${1:-}" in
    recover) recover ;;
    supervise) supervise ;;
    *) echo "usage: $0 recover | supervise" >&2; exit 2 ;;
esac
