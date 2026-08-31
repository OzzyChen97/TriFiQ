#!/usr/bin/env bash
set -euo pipefail

# Hot-join persistent policy workers to an active dynamic Table-6 episode
# queue.  The parent launcher waits for running queue entries before merging,
# so completed episodes are preserved and exact validation remains unchanged.

ROOT="/home1/gyy/vla/QuantVLA"
DRIVER="$ROOT/external/Omega-QVLA/scripts/run_libero_duquant_benchmark_multi_gpu.py"

if [[ $# -ne 4 ]]; then
    echo "usage: $0 PARENT_PID START_SHARD GPU_CSV STOP_PENDING" >&2
    exit 2
fi

parent_pid="$1"
start_shard="$2"
gpu_csv="$3"
stop_pending="$4"

[[ "$parent_pid" =~ ^[0-9]+$ && -r "/proc/$parent_pid/environ" ]] || {
    echo "invalid or exited parent PID: $parent_pid" >&2
    exit 2
}
[[ "$start_shard" =~ ^[0-9]+$ && "$stop_pending" =~ ^[0-9]+$ ]] || exit 2

# Reuse the exact frozen runtime environment of the active formal launcher.
while IFS= read -r -d '' entry; do
    export "$entry"
done < "/proc/$parent_pid/environ"

: "${OUTPUT_ROOT:?}" "${TASK_SUITE:?}" "${MODEL_PATH:?}" "${DATA_CONFIG:?}"
: "${PACKDIR:?}" "${GROOT_PY:?}" "${PORT_BASE:?}"

queue="$OUTPUT_ROOT/episode_queue.json"
[[ -f "$queue" && "$OUTPUT_ROOT" == "$ROOT"/runs/table6_libero_v1/staging/* ]] || {
    echo "refusing non-formal or missing episode queue: $queue" >&2
    exit 2
}

export GR00T_DYNAMIC_EPISODE_QUEUE_PATH="$queue"
export GR00T_DYNAMIC_EPISODE_QUEUE_STOP_PENDING="$stop_pending"

IFS=',' read -r -a gpus <<<"$gpu_csv"
pids=()
for offset in "${!gpus[@]}"; do
    gpu="${gpus[$offset]}"
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU: $gpu" >&2; exit 2; }
    shard=$((start_shard + offset))
    port=$((PORT_BASE + shard))
    log="$OUTPUT_ROOT/logs/driver_shard_${shard}.log"
    (
        "$GROOT_PY" "$DRIVER" --run-shard \
            "$shard" "$gpu" "$port" '[0]' \
            "$TASK_SUITE" "$MODEL_PATH" "$DATA_CONFIG" \
            1 "${NUM_STEPS_WAIT:-10}" "${DENOISING_STEPS:-8}" \
            "${WBITS:-4}" "${ABITS:-8}" "$OUTPUT_ROOT" "$PACKDIR" \
            1 "${GR00T_EVAL_INIT_OFFSET:-10}"
    ) >"$log" 2>&1 &
    pids+=("$!")
    echo "hotjoin shard=$shard gpu=$gpu port=$port pid=${pids[-1]}"
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
