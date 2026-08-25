#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
PAPER_ROOT="${PI05_PAPER_ROOT:-$REPO_ROOT/runs/pi05_quantvla_paper}"
RUN_ROOT="${PI05_PAPER_SANITY_ROOT:-$PAPER_ROOT/sanity30}"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_paper_server.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
MANIFEST_TOOL="$REPO_ROOT/scripts/tools/pi05_make_paper_sanity_manifest.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/pi05_aggregate_paper_sanity30.py"
TASKS="OpenCabinet,OpenStandMixerHead,PickPlaceDrawerToCounter,CoffeeSetupMug"
CONFIGS=(fp16 quantvla_w4a8_atmohb gdsq_vla_atmohb gdsq_vla)
GPUS=(1 2 3 4)
PORTS=(18601 18602 18603 18604)

usage() {
    echo "usage: $0 run-mode paired|native | run-all | status" >&2
}

formal_matrix_running() {
    pgrep -f '/scripts/supervise_pi05_faithful_parallel.sh' >/dev/null 2>&1
}

runtime_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

instance_name() {
    echo "sanity_${1}"
}

start_servers() {
    local index config gpu port instance
    for index in "${!CONFIGS[@]}"; do
        config="${CONFIGS[$index]}"; gpu="${GPUS[$index]}"; port="${PORTS[$index]}"
        instance="$(instance_name "$config")"
        "$SERVER_MANAGER" start "$config" "$gpu" "$port" "$instance"
    done
}

stop_servers() {
    local config
    for config in "${CONFIGS[@]}"; do
        "$SERVER_MANAGER" stop "$(instance_name "$config")" || true
    done
}

make_manifest() {
    local mode="$1" run_dir="$RUN_ROOT/$1" args=() index config runtime
    mkdir -p "$run_dir"
    for index in "${!CONFIGS[@]}"; do
        config="${CONFIGS[$index]}"
        runtime="$PAPER_ROOT/control/$(instance_name "$config").runtime.json"
        args+=(--server "$config,${GPUS[$index]},${PORTS[$index]},$runtime")
    done
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$MANIFEST_TOOL" --out "$run_dir/manifest.json" \
        --noise-mode "$mode" --paper-root "$PAPER_ROOT" "${args[@]}"
}

run_worker() {
    local mode="$1" config="$2" gpu="$3" port="$4" shard="$5" shard_count="$6"
    local run_dir="$RUN_ROOT/$mode" result_dir="$RUN_ROOT/$mode/results/$config"
    local runtime="$PAPER_ROOT/control/$(instance_name "$config").runtime.json" hash
    hash="$(runtime_hash "$runtime")"
    mkdir -p "$result_dir" "$run_dir/logs"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" --host 127.0.0.1 --port "$port" \
        --config-id "$config" --task-set atomic_seen --tasks "$TASKS" \
        --task-shard-index "$shard" --task-shard-count "$shard_count" \
        --trial-seeds 0-29 --split pretrain --replan-steps 5 --egl-device "$gpu" \
        --action-noise-mode "$mode" --expected-server-metadata-sha256 "$hash" \
        --resume-dir "$result_dir" --out "$result_dir/atomic_seen_${mode}_shard${shard}.jsonl" \
        >"$run_dir/logs/${config}_shard${shard}.log" 2>&1
}

run_workers_once() {
    local mode="$1" shards=2
    [[ "$mode" == native ]] && shards=1
    local pids=() index config shard
    for index in "${!CONFIGS[@]}"; do
        config="${CONFIGS[$index]}"
        for shard in $(seq 0 $((shards - 1))); do
            run_worker "$mode" "$config" "${GPUS[$index]}" "${PORTS[$index]}" "$shard" "$shards" &
            pids+=("$!")
        done
    done
    local failed=0 pid
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    return "$failed"
}

run_mode() {
    [[ $# -eq 1 && ( "$1" == paired || "$1" == native ) ]] || { usage; exit 2; }
    local mode="$1"
    if formal_matrix_running; then
        echo "formal Table-1 matrix is still running; paper sanity launch is deferred" >&2
        exit 3
    fi
    "$ROBOCASA_PY" "$REPO_ROOT/scripts/tools/pi05_audit_paper_quantvla.py" --root "$PAPER_ROOT"
    trap stop_servers EXIT INT TERM
    start_servers
    make_manifest "$mode"
    local attempt
    for attempt in 1 2 3; do
        if run_workers_once "$mode"; then break; fi
        echo "sanity workers failed on attempt=$attempt; resuming committed rows" >&2
        [[ "$attempt" != 3 ]] || exit 1
    done
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$AGGREGATOR" --run-dir "$RUN_ROOT/$mode" \
        >"$RUN_ROOT/$mode/aggregate.log"
    stop_servers
    trap - EXIT INT TERM
    echo "paper sanity complete: mode=$mode"
}

status() {
    "$REPO_ROOT/scripts/prepare_pi05_paper_quantvla.sh" status
    "$SERVER_MANAGER" status
    local mode
    for mode in paired native; do
        if [[ -f "$RUN_ROOT/$mode/summary.json" ]]; then
            echo "complete $mode $RUN_ROOT/$mode/summary.json"
        else
            echo "pending $mode"
        fi
    done
}

case "${1:-}" in
    run-mode) shift; run_mode "$@" ;;
    run-all)
        shift
        run_mode paired
        run_mode native
        ;;
    status) status ;;
    *) usage; exit 2 ;;
esac
