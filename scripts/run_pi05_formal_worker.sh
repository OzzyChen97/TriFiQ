#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
    echo "usage: $0 CONFIG PORT EGL_GPU SHARD_INDEX SHARD_COUNT WORKER_ID METADATA_SHA256 RUN_DIR" >&2
    exit 2
fi

CONFIG="$1"
PORT="$2"
EGL_GPU="$3"
SHARD_INDEX="$4"
SHARD_COUNT="$5"
WORKER_ID="$6"
METADATA_SHA256="$7"
RUN_DIR="$8"

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
if [[ ! "$WORKER_ID" =~ ^[A-Za-z0-9_.-]+$ || ! "$METADATA_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid worker id or metadata hash" >&2
    exit 2
fi

mkdir -p "$RUN_DIR/results/$CONFIG"
export PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

for task_set in atomic_seen composite_seen composite_unseen; do
    output="$RUN_DIR/results/$CONFIG/${task_set}_${WORKER_ID}.jsonl"
    "$PYTHON" "$EVALUATOR" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --config-id "$CONFIG" \
        --task-set "$task_set" \
        --task-shard-index "$SHARD_INDEX" \
        --task-shard-count "$SHARD_COUNT" \
        --trial-seeds 0-49 \
        --split target \
        --replan-steps 16 \
        --egl-device "$EGL_GPU" \
        --expected-server-metadata-sha256 "$METADATA_SHA256" \
        --out "$output"
done

echo "formal worker complete: $WORKER_ID config=$CONFIG shard=$SHARD_INDEX/$SHARD_COUNT"
