#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
    echo "usage: $0 CONFIG PORT EGL_GPU SHARD_INDEX SHARD_COUNT TRIAL_SEEDS WORKER_ID METADATA_SHA256 RUN_DIR" >&2
    exit 2
fi

CONFIG="$1"
PORT="$2"
EGL_GPU="$3"
SHARD_INDEX="$4"
SHARD_COUNT="$5"
TRIAL_SEEDS="$6"
WORKER_ID="$7"
METADATA_SHA256="$8"
RUN_DIR="$9"

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"

for numeric in "$PORT" "$EGL_GPU" "$SHARD_INDEX" "$SHARD_COUNT"; do
    if [[ ! "$numeric" =~ ^[0-9]+$ ]]; then
        echo "invalid numeric argument: $numeric" >&2
        exit 2
    fi
done
if [[ ! "$TRIAL_SEEDS" =~ ^[0-9]+-[0-9]+$ ]]; then
    echo "formal seeded worker requires one inclusive seed range" >&2
    exit 2
fi
if [[ ! "$WORKER_ID" =~ ^[A-Za-z0-9_.-]+$ || ! "$METADATA_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid worker id or metadata hash" >&2
    exit 2
fi

mkdir -p "$RUN_DIR/results/$CONFIG"
export PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

TASK_SETS="${PI05_TASK_SETS:-atomic_seen composite_seen composite_unseen}"
selector_args=()
if [[ "$CONFIG" == "gdsq_vla_runtime_selector" ]]; then
    selector_args+=(--expect-runtime-selector)
fi
for task_set in $TASK_SETS; do
    output="$RUN_DIR/results/$CONFIG/${task_set}_${WORKER_ID}.jsonl"
    "$PYTHON" "$EVALUATOR" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --config-id "$CONFIG" \
        --task-set "$task_set" \
        --task-shard-index "$SHARD_INDEX" \
        --task-shard-count "$SHARD_COUNT" \
        --trial-seeds "$TRIAL_SEEDS" \
        --split target \
        --replan-steps 16 \
        --egl-device "$EGL_GPU" \
        --expected-server-metadata-sha256 "$METADATA_SHA256" \
        "${selector_args[@]}" \
        --resume-dir "$RUN_DIR/results/$CONFIG" \
        --out "$output"
done

echo "formal seeded worker complete: $WORKER_ID seeds=$TRIAL_SEEDS config=$CONFIG shard=$SHARD_INDEX/$SHARD_COUNT"
