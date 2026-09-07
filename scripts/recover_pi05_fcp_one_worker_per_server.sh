#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROOT="$REPO_ROOT/runs/full_context_v2/pi05_fcp_diagnostic"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
FREEZER="$REPO_ROOT/scripts/tools/freeze_pi05_fcp_concurrency_adjustment.py"
AGGREGATOR="$REPO_ROOT/scripts/run_pi05_fcp_diagnostic.sh"
CONTROL="$ROOT/control"
WORKER_CONTROL="$CONTROL/workers"
PREFLIGHT="$ROOT/preflight/one_worker_per_server"
ROLLOUTS="$ROOT/rollouts"
CONFIG_ID="full_context_w4a8_dynamic_profile"
SHARD_COUNT=9

# arm,task-shard,server-shard,EGL-GPU,port. Each stage contains no repeated server.
STAGE1=(
    "transferred_initializer,0,0,2,19800" "transferred_initializer,1,1,3,19801"
    "transferred_initializer,2,2,4,19802" "transferred_initializer,3,3,5,19803"
    "transferred_initializer,4,4,6,19804" "transferred_initializer,5,8,7,19808"
    "single_best,0,0,3,19820" "single_best,1,1,4,19821"
    "single_best,2,2,5,19822" "single_best,3,3,6,19823"
    "single_best,4,7,7,19827"
    "two_best,0,0,3,19840" "two_best,1,1,4,19841"
    "two_best,2,2,5,19842" "two_best,3,3,6,19843"
    "two_best,4,7,7,19847"
)
STAGE2=(
    "transferred_initializer,6,0,2,19800" "transferred_initializer,7,1,3,19801"
    "transferred_initializer,8,2,4,19802"
    "single_best,5,2,5,19822" "single_best,6,3,6,19823"
    "single_best,7,7,7,19827" "single_best,8,0,3,19820"
    "two_best,5,2,5,19842" "two_best,6,3,6,19843"
    "two_best,7,7,7,19847" "two_best,8,0,3,19840"
)

instance_for() { printf 'pi05_fcp_%s_s%s' "$1" "$2"; }
semantic_hash() { jq -r '.openpi_runtime.semantic_metadata_sha256' "$1"; }

smoke_one() {
    local arm="$1" server_shard="$2" egl="$3" port="$4" instance runtime hash output
    instance="$(instance_for "$arm" "$server_shard")"; runtime="$CONTROL/$instance.runtime.json"
    hash="$(semantic_hash "$runtime")"; output="$PREFLIGHT/${arm}_gpu${egl}.jsonl"
    mkdir -p "$PREFLIGHT/${arm}_gpu${egl}"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" --host 127.0.0.1 --port "$port" \
        --config-id "$CONFIG_ID" --task-set atomic_seen --tasks OpenDrawer \
        --trial-seeds 2 --split target --replan-steps 16 --flow-steps 4 \
        --egl-device "$egl" --expected-server-metadata-sha256 "$hash" --max-steps 1 \
        --resume-dir "$PREFLIGHT/${arm}_gpu${egl}" --out "$output" \
        >"$PREFLIGHT/${arm}_gpu${egl}.log" 2>&1
}

launch_worker() {
    local stage="$1" arm="$2" shard="$3" server_shard="$4" egl="$5" port="$6"
    local instance runtime hash worker_id log pid
    instance="$(instance_for "$arm" "$server_shard")"; runtime="$CONTROL/$instance.runtime.json"
    hash="$(semantic_hash "$runtime")"; worker_id="serial${stage}_${arm}_s${shard}"
    log="$WORKER_CONTROL/$worker_id.log"
    nohup setsid env PI05_FLOW_STEPS=4 "$WORKER" "$CONFIG_ID" "$port" "$egl" \
        "$shard" "$SHARD_COUNT" 0-9 "$worker_id" "$hash" "$ROLLOUTS/$arm" \
        >"$log" 2>&1 </dev/null &
    pid=$!; echo "$pid" >"$WORKER_CONTROL/$worker_id.pid"
    echo "started $worker_id pid=$pid server=$instance egl=$egl"
}

launch_stage() {
    local stage="$1" row arm shard server_shard egl port
    local -n jobs="STAGE${stage}"
    for row in "${jobs[@]}"; do
        IFS=, read -r arm shard server_shard egl port <<<"$row"
        launch_worker "$stage" "$arm" "$shard" "$server_shard" "$egl" "$port"
    done
}

stage_alive() {
    local stage="$1" pid_file pid command
    for pid_file in "$WORKER_CONTROL"/serial"$stage"_*.pid; do
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
    logs="$(find "$WORKER_CONTROL" -maxdepth 1 -name "serial${stage}_*.log" | wc -l)"
    [[ "$logs" -eq "$expected" ]] || { echo "stage $stage log count $logs != $expected" >&2; return 1; }
    if rg -n "Traceback|Segmentation fault|CUDA out of memory|CUDNN_STATUS|CUBLAS_STATUS|unable to find an engine" "$WORKER_CONTROL"/serial"$stage"_*.log; then
        echo "stage $stage contains a resource/runtime failure" >&2; return 1
    fi
    completed="$({ rg -l 'formal seeded worker complete:' "$WORKER_CONTROL"/serial"$stage"_*.log || true; } | wc -l)"
    [[ "$completed" -eq "$expected" ]] || { echo "stage $stage completion count $completed != $expected" >&2; return 1; }
}

supervise() {
    while stage_alive 1; do sleep 30; done
    check_stage 1 16
    launch_stage 2
    while stage_alive 2; do sleep 30; done
    check_stage 2 11
    "$AGGREGATOR" status
    "$AGGREGATOR" aggregate
}

recover() {
    local pids=() p
    mkdir -p "$PREFLIGHT" "$WORKER_CONTROL"
    smoke_one transferred_initializer 0 2 19800 & pids+=("$!")
    smoke_one single_best 0 3 19820 & pids+=("$!")
    smoke_one two_best 1 4 19841 & pids+=("$!")
    smoke_one transferred_initializer 3 5 19803 & pids+=("$!")
    smoke_one single_best 3 6 19823 & pids+=("$!")
    smoke_one two_best 7 7 19847 & pids+=("$!")
    for p in "${pids[@]}"; do wait "$p"; done
    "$OPENPI_PY" "$FREEZER"
    launch_stage 1
    nohup setsid "$0" supervise >"$CONTROL/one_worker_per_server_supervisor.log" 2>&1 </dev/null &
    echo $! >"$CONTROL/one_worker_per_server_supervisor.pid"
    "$AGGREGATOR" status
}

case "${1:-}" in
    recover) recover ;;
    supervise) supervise ;;
    *) echo "usage: $0 recover | supervise" >&2; exit 2 ;;
esac
