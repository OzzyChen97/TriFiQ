#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 MANIFEST WORKER_ID EXPECTED_MANIFEST_SHA256" >&2
    exit 2
fi

MANIFEST="$1"
WORKER_ID="$2"
EXPECTED_MANIFEST_SHA256="$3"
REPO_ROOT="/home1/gyy/vla/QuantVLA"
RUN_ROOT="$REPO_ROOT/runs/full_context_v2/pi05_table1"
CONFIG_ID="full_context_w4a8_dynamic_profile"
PYTHON="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"

actual_manifest_sha256="$(sha256sum "$MANIFEST" | awk '{print $1}')"
[[ "$actual_manifest_sha256" == "$EXPECTED_MANIFEST_SHA256" ]]
jq -e --arg worker "$WORKER_ID" '.immutable == true and any(.assignments[]; .worker_id == $worker)' \
    "$MANIFEST" >/dev/null

port="$(jq -r --arg worker "$WORKER_ID" '.assignments[] | select(.worker_id==$worker) | .port' "$MANIFEST")"
gpu="$(jq -r --arg worker "$WORKER_ID" '.assignments[] | select(.worker_id==$worker) | .gpu' "$MANIFEST")"
metadata_hash="$(jq -r --arg worker "$WORKER_ID" '.assignments[] | select(.worker_id==$worker) | .server_metadata_sha256' "$MANIFEST")"
export PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

while IFS=$'\t' read -r task_set task seed; do
    output="$RUN_ROOT/results/$CONFIG_ID/${task_set}_${WORKER_ID}_${task}_${seed}.jsonl"
    "$PYTHON" "$EVALUATOR" \
        --host 127.0.0.1 --port "$port" --config-id "$CONFIG_ID" \
        --task-set "$task_set" --tasks "$task" --trial-seeds "$seed" \
        --split target --replan-steps 16 --flow-steps 4 --egl-device "$gpu" \
        --expected-server-metadata-sha256 "$metadata_hash" \
        --resume-dir "$RUN_ROOT/results/$CONFIG_ID" --out "$output"
done < <(jq -r --arg worker "$WORKER_ID" \
    '.assignments[] | select(.worker_id==$worker) | .keys[] | [.task_set,.task,(.seed|tostring)] | @tsv' \
    "$MANIFEST")

echo "finish-sprint worker complete: $WORKER_ID"
