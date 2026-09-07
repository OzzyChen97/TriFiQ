#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROOT="$REPO_ROOT/runs/full_context_v2/pi05_fcp_diagnostic"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
FREEZER="$REPO_ROOT/scripts/tools/freeze_pi05_fcp_semantic_runtime_erratum.py"
CONTROL="$ROOT/control"
WORKER_CONTROL="$CONTROL/workers"
PREFLIGHT="$ROOT/preflight"
ROLLOUTS="$ROOT/rollouts"
CONFIG_ID="full_context_w4a8_dynamic_profile"
SHARD_COUNT=9

SCHEDULE=(
    "transferred_initializer,0,2,19800" "transferred_initializer,1,3,19801"
    "transferred_initializer,2,4,19802" "transferred_initializer,3,5,19803"
    "transferred_initializer,4,6,19804" "transferred_initializer,5,4,19805"
    "transferred_initializer,6,5,19806" "transferred_initializer,7,6,19807"
    "transferred_initializer,8,7,19808"
    "single_best,0,3,19820" "single_best,1,4,19821" "single_best,2,5,19822"
    "single_best,3,6,19823" "single_best,4,4,19824" "single_best,5,5,19825"
    "single_best,6,6,19826" "single_best,7,7,19827" "single_best,8,7,19828"
    "two_best,0,3,19840" "two_best,1,4,19841" "two_best,2,5,19842"
    "two_best,3,6,19843" "two_best,4,3,19844" "two_best,5,5,19845"
    "two_best,6,6,19846" "two_best,7,7,19847" "two_best,8,7,19848"
)

instance_for() { printf 'pi05_fcp_%s_s%s' "$1" "$2"; }

semantic_hash() {
    local value
    value="$(jq -r '.openpi_runtime.semantic_metadata_sha256 // empty' "$1")"
    [[ "$value" =~ ^[0-9a-f]{64}$ ]] || {
        echo "invalid semantic metadata hash in $1" >&2
        return 1
    }
    printf '%s\n' "$value"
}

smoke_one() {
    local arm="$1" shard="$2" gpu="$3" port="$4" instance runtime hash output
    instance="$(instance_for "$arm" "$shard")"; runtime="$CONTROL/$instance.runtime.json"
    hash="$(semantic_hash "$runtime")"
    mkdir -p "$PREFLIGHT/$arm/semantic"
    output="$PREFLIGHT/$arm/semantic_smoke_${hash:0:12}.jsonl"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" --host 127.0.0.1 --port "$port" \
        --config-id "$CONFIG_ID" --task-set atomic_seen --tasks OpenDrawer \
        --trial-seeds 0 --split target --replan-steps 16 --flow-steps 4 \
        --egl-device "$gpu" --expected-server-metadata-sha256 "$hash" --max-steps 1 \
        --resume-dir "$PREFLIGHT/$arm/semantic" --out "$output" \
        >"$PREFLIGHT/$arm/semantic_smoke.log" 2>&1
    echo "semantic smoke passed arm=$arm hash=$hash"
}

freeze_execution() {
    local args=() spec arm shard gpu port instance
    for spec in "${SCHEDULE[@]}"; do
        IFS=, read -r arm shard gpu port <<<"$spec"
        instance="$(instance_for "$arm" "$shard")"
        args+=(--server "$arm,$instance,$gpu,$port,$CONTROL/$instance.runtime.json")
    done
    "$OPENPI_PY" "$FREEZER" "${args[@]}"
}

start_worker() {
    local arm="$1" shard="$2" gpu="$3" port="$4" seeds="$5" suffix="$6"
    local instance runtime hash worker_id log pid
    instance="$(instance_for "$arm" "$shard")"; runtime="$CONTROL/$instance.runtime.json"
    hash="$(semantic_hash "$runtime")"
    worker_id="${arm}_s${shard}_${suffix}"; log="$WORKER_CONTROL/$worker_id.log"
    nohup setsid env PI05_FLOW_STEPS=4 "$WORKER" "$CONFIG_ID" "$port" "$gpu" \
        "$shard" "$SHARD_COUNT" "$seeds" "$worker_id" "$hash" "$ROLLOUTS/$arm" \
        >"$log" 2>&1 </dev/null &
    pid=$!; echo "$pid" >"$WORKER_CONTROL/$worker_id.pid"
    echo "started worker=$worker_id pid=$pid gpu=$gpu shard=$shard/$SHARD_COUNT seeds=$seeds"
}

main() {
    mkdir -p "$WORKER_CONTROL" "$ROLLOUTS"
    if find "$ROLLOUTS" -type f -name '*.jsonl' -print -quit | grep -q .; then
        echo "refusing semantic-hash correction after rollout rows exist" >&2; exit 1
    fi
    local pids=() labels=(transferred_initializer single_best two_best) index failed=0
    smoke_one transferred_initializer 0 2 19800 & pids+=("$!")
    smoke_one single_best 1 4 19821 & pids+=("$!")
    smoke_one two_best 2 5 19842 & pids+=("$!")
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then failed=1; echo "corrected smoke failed: ${labels[$index]}" >&2; fi
    done
    [[ "$failed" == 0 ]]
    freeze_execution
    local spec arm shard gpu port
    for spec in "${SCHEDULE[@]}"; do
        IFS=, read -r arm shard gpu port <<<"$spec"
        start_worker "$arm" "$shard" "$gpu" "$port" 0-4 lo
        start_worker "$arm" "$shard" "$gpu" "$port" 5-9 hi
    done
    "$REPO_ROOT/scripts/run_pi05_fcp_diagnostic.sh" status
}

main "$@"
