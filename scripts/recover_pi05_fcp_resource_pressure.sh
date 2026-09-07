#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROOT="$REPO_ROOT/runs/full_context_v2/pi05_fcp_diagnostic"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
FREEZER="$REPO_ROOT/scripts/tools/freeze_pi05_fcp_resource_adjustment.py"
AGGREGATOR="$REPO_ROOT/scripts/run_pi05_fcp_diagnostic.sh"
CONTROL="$ROOT/control"
WORKER_CONTROL="$CONTROL/workers"
PREFLIGHT="$ROOT/preflight/resource_recovery"
ROLLOUTS="$ROOT/rollouts"
CONFIG_ID="full_context_w4a8_dynamic_profile"
SHARD_COUNT=9

# arm,task-shard,retained-server-shard,EGL-GPU,port,run-hi-in-stage1
UNITS=(
    "transferred_initializer,0,0,2,19800,1" "transferred_initializer,1,1,3,19801,1"
    "transferred_initializer,2,2,4,19802,1" "transferred_initializer,3,3,5,19803,1"
    "transferred_initializer,4,4,6,19804,1" "transferred_initializer,5,2,4,19802,1"
    "transferred_initializer,6,3,5,19803,1" "transferred_initializer,7,4,6,19804,1"
    "transferred_initializer,8,8,7,19808,1"
    "single_best,0,0,3,19820,1" "single_best,1,1,4,19821,1"
    "single_best,2,2,5,19822,0" "single_best,3,3,6,19823,0"
    "single_best,4,1,4,19821,0" "single_best,5,2,5,19822,0"
    "single_best,6,3,6,19823,0" "single_best,7,7,7,19827,1"
    "single_best,8,7,7,19827,1"
    "two_best,0,0,3,19840,1" "two_best,1,1,4,19841,0"
    "two_best,2,2,5,19842,0" "two_best,3,3,6,19843,0"
    "two_best,4,0,3,19840,1" "two_best,5,2,5,19842,0"
    "two_best,6,3,6,19843,0" "two_best,7,7,7,19847,0"
    "two_best,8,7,7,19847,0"
)

instance_for() { printf 'pi05_fcp_%s_s%s' "$1" "$2"; }
semantic_hash() { jq -r '.openpi_runtime.semantic_metadata_sha256' "$1"; }

smoke_one() {
    local arm="$1" server_shard="$2" egl="$3" port="$4" instance runtime hash output
    instance="$(instance_for "$arm" "$server_shard")"; runtime="$CONTROL/$instance.runtime.json"
    hash="$(semantic_hash "$runtime")"; output="$PREFLIGHT/${arm}.jsonl"
    mkdir -p "$PREFLIGHT/$arm"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" --host 127.0.0.1 --port "$port" \
        --config-id "$CONFIG_ID" --task-set atomic_seen --tasks OpenDrawer \
        --trial-seeds 1 --split target --replan-steps 16 --flow-steps 4 \
        --egl-device "$egl" --expected-server-metadata-sha256 "$hash" --max-steps 1 \
        --resume-dir "$PREFLIGHT/$arm" --out "$output" >"$PREFLIGHT/${arm}.log" 2>&1
}

launch_worker() {
    local stage="$1" arm="$2" shard="$3" server_shard="$4" egl="$5" port="$6" seeds="$7" suffix="$8"
    local instance runtime hash worker_id log pid
    instance="$(instance_for "$arm" "$server_shard")"; runtime="$CONTROL/$instance.runtime.json"
    hash="$(semantic_hash "$runtime")"; worker_id="safe${stage}_${arm}_s${shard}_${suffix}"
    log="$WORKER_CONTROL/$worker_id.log"
    nohup setsid env PI05_FLOW_STEPS=4 "$WORKER" "$CONFIG_ID" "$port" "$egl" \
        "$shard" "$SHARD_COUNT" "$seeds" "$worker_id" "$hash" "$ROLLOUTS/$arm" \
        >"$log" 2>&1 </dev/null &
    pid=$!; echo "$pid" >"$WORKER_CONTROL/$worker_id.pid"
    echo "started $worker_id pid=$pid server=$instance egl=$egl"
}

launch_stage1() {
    local row arm shard server_shard egl port hi
    for row in "${UNITS[@]}"; do
        IFS=, read -r arm shard server_shard egl port hi <<<"$row"
        launch_worker 1 "$arm" "$shard" "$server_shard" "$egl" "$port" 0-4 lo
        if [[ "$hi" == 1 ]]; then
            launch_worker 1 "$arm" "$shard" "$server_shard" "$egl" "$port" 5-9 hi
        fi
    done
}

launch_stage2() {
    local row arm shard server_shard egl port hi
    for row in "${UNITS[@]}"; do
        IFS=, read -r arm shard server_shard egl port hi <<<"$row"
        if [[ "$hi" == 0 ]]; then
            launch_worker 2 "$arm" "$shard" "$server_shard" "$egl" "$port" 5-9 hi
        fi
    done
}

stage_alive() {
    local stage="$1" pid_file pid command
    for pid_file in "$WORKER_CONTROL"/safe"$stage"_*.pid; do
        [[ -f "$pid_file" ]] || continue
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            command="$(ps -p "$pid" -o args=)"
            [[ "$command" == *run_pi05_formal_worker_seeded.sh*"$ROOT"* ]] && return 0
        fi
    done
    return 1
}

check_stage() {
    local stage="$1" expected="$2" logs completed
    logs="$(find "$WORKER_CONTROL" -maxdepth 1 -name "safe${stage}_*.log" | wc -l)"
    [[ "$logs" -eq "$expected" ]] || { echo "stage $stage log count $logs != $expected" >&2; return 1; }
    if rg -n "Traceback|Segmentation fault|CUDA out of memory|CUDNN_STATUS|CUBLAS_STATUS|unable to find an engine" "$WORKER_CONTROL"/safe"$stage"_*.log; then
        echo "stage $stage contains a resource/runtime failure" >&2; return 1
    fi
    completed="$({ rg -l 'formal seeded worker complete:' "$WORKER_CONTROL"/safe"$stage"_*.log || true; } | wc -l)"
    [[ "$completed" -eq "$expected" ]] || { echo "stage $stage completion count $completed != $expected" >&2; return 1; }
}

supervise() {
    while stage_alive 1; do sleep 30; done
    check_stage 1 42
    launch_stage2
    while stage_alive 2; do sleep 30; done
    check_stage 2 12
    "$AGGREGATOR" status
    "$AGGREGATOR" aggregate
}

recover() {
    local p1 p2 p3
    mkdir -p "$PREFLIGHT" "$WORKER_CONTROL"
    smoke_one transferred_initializer 0 2 19800 & p1=$!
    smoke_one single_best 0 3 19820 & p2=$!
    smoke_one two_best 1 4 19841 & p3=$!
    wait "$p1"; wait "$p2"; wait "$p3"
    "$OPENPI_PY" "$FREEZER"
    launch_stage1
    nohup setsid "$0" supervise >"$CONTROL/resource_recovery_supervisor.log" 2>&1 </dev/null &
    echo $! >"$CONTROL/resource_recovery_supervisor.pid"
    "$AGGREGATOR" status
}

case "${1:-}" in
    recover) recover ;;
    supervise) supervise ;;
    *) echo "usage: $0 recover | supervise" >&2; exit 2 ;;
esac
