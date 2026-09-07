#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROOT="$REPO_ROOT/runs/full_context_v2/pi05_fcp_diagnostic"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
MATERIALIZER="$REPO_ROOT/scripts/tools/materialize_pi05_fcp_residual_gpu4567.py"
AGGREGATOR="$REPO_ROOT/scripts/run_pi05_fcp_diagnostic.sh"
SCHEDULE="$ROOT/residual_schedule_gpu4567.json"
CONTROL="$ROOT/control"
WORKER_CONTROL="$CONTROL/workers"
ROLLOUTS="$ROOT/rollouts"
CONFIG_ID="full_context_w4a8_dynamic_profile"

runner_alive() {
    local runner_id pid_file pid command
    runner_id="$1"; pid_file="$WORKER_CONTROL/$runner_id.pid"
    [[ -f "$pid_file" ]] || return 1
    pid="$(<"$pid_file")"; [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    command="$(ps -p "$pid" -o args=)"
    [[ "$command" == *run_pi05_fcp_residual_gpu4567.sh*runner* ]]
}

run_queue() {
    local index="$1" runner_id arm port egl hash result_dir task_set task seed output
    runner_id="$(jq -r ".runners[$index].runner_id" "$SCHEDULE")"
    arm="$(jq -r ".runners[$index].arm" "$SCHEDULE")"
    port="$(jq -r ".runners[$index].port" "$SCHEDULE")"
    egl="$(jq -r ".runners[$index].egl_gpu" "$SCHEDULE")"
    hash="$(jq -r ".runners[$index].server_metadata_sha256" "$SCHEDULE")"
    result_dir="$ROLLOUTS/$arm/results/$CONFIG_ID"; mkdir -p "$result_dir"
    while IFS=$'\t' read -r task_set task seed; do
        output="$result_dir/${task_set}_${runner_id}.jsonl"
        PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
            "$ROBOCASA_PY" "$EVALUATOR" --host 127.0.0.1 --port "$port" \
            --config-id "$CONFIG_ID" --task-set "$task_set" --tasks "$task" \
            --trial-seeds "$seed" --split target --replan-steps 16 --flow-steps 4 \
            --egl-device "$egl" --expected-server-metadata-sha256 "$hash" \
            --resume-dir "$result_dir" --out "$output"
    done < <(jq -r ".runners[$index].jobs[] | [.task_set,.task,(.seed|tostring)] | @tsv" "$SCHEDULE")
    echo "GPU4567 residual runner complete: $runner_id"
}

launch() {
    local count index runner_id log pid
    "$OPENPI_PY" "$MATERIALIZER"
    count="$(jq '.runners | length' "$SCHEDULE")"
    for index in $(seq 0 $((count - 1))); do
        runner_id="$(jq -r ".runners[$index].runner_id" "$SCHEDULE")"
        log="$WORKER_CONTROL/$runner_id.log"
        nohup setsid "$0" runner "$index" >"$log" 2>&1 </dev/null &
        pid=$!; echo "$pid" >"$WORKER_CONTROL/$runner_id.pid"
        echo "started $runner_id pid=$pid"
    done
    nohup setsid "$0" supervise >"$CONTROL/residual_gpu4567_supervisor.log" 2>&1 </dev/null &
    echo $! >"$CONTROL/residual_gpu4567_supervisor.pid"
    "$AGGREGATOR" status
}

supervise() {
    local count index runner_id alive completed
    count="$(jq '.runners | length' "$SCHEDULE")"
    while true; do
        alive=0
        for index in $(seq 0 $((count - 1))); do
            runner_id="$(jq -r ".runners[$index].runner_id" "$SCHEDULE")"
            runner_alive "$runner_id" && alive=$((alive + 1))
        done
        [[ "$alive" -gt 0 ]] || break
        sleep 15
    done
    if rg -n "Traceback|Segmentation fault|CUDA out of memory|CUDNN_STATUS|CUBLAS_STATUS|unable to find an engine" "$WORKER_CONTROL"/residual47_*.log; then
        echo "GPU4567 residual rollout contains a failure" >&2; return 1
    fi
    completed="$({ rg -l 'GPU4567 residual runner complete:' "$WORKER_CONTROL"/residual47_*.log || true; } | wc -l)"
    [[ "$completed" -eq "$count" ]] || { echo "GPU4567 completion count $completed != $count" >&2; return 1; }
    "$AGGREGATOR" status
    "$AGGREGATOR" aggregate
}

case "${1:-}" in
    launch) launch ;;
    runner) run_queue "$2" ;;
    supervise) supervise ;;
    *) echo "usage: $0 launch | runner INDEX | supervise" >&2; exit 2 ;;
esac
