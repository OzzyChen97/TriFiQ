#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROOT="$REPO_ROOT/runs/full_context_v2/pi05_fcp_diagnostic"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
FREEZER="$REPO_ROOT/scripts/tools/freeze_pi05_fcp_server_restart_adjustment.py"
AGGREGATOR="$REPO_ROOT/scripts/run_pi05_fcp_diagnostic.sh"
CONTROL="$ROOT/control"
WORKER_CONTROL="$CONTROL/workers"
AUDIT="$ROOT/operational_audit/pre_poisoned_restart"
PREFLIGHT="$ROOT/preflight/post_restart"
ROLLOUTS="$ROOT/rollouts"
CONFIG_ID="full_context_w4a8_dynamic_profile"
PACK_DIR="$REPO_ROOT/runs/archive_previous_versions/errorfold_v4_iter/calibration/pi05_full_w4/identity_pack"
BUFFER="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
SHARD_COUNT=9

RESTARTS=(
    "single_best,2,5,19822" "single_best,3,6,19823"
    "two_best,1,4,19841" "two_best,3,6,19843"
)
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

plan_for() {
    case "$1" in
        transferred_initializer) echo "$REPO_ROOT/runs/full_context_v2/pi05_p2/interventions_round_main/context_base.json" ;;
        single_best) echo "$REPO_ROOT/runs/full_context_v2/pi05_p2/proposals_dynamic/single_best.json" ;;
        two_best) echo "$REPO_ROOT/runs/full_context_v2/pi05_p2/proposals_dynamic/two_best.json" ;;
        *) return 2 ;;
    esac
}
hessian_for() { echo "$ROOT/artifacts/$1/hessian_w4.npz"; }
instance_for() { printf 'pi05_fcp_%s_s%s' "$1" "$2"; }
semantic_hash() { jq -r '.openpi_runtime.semantic_metadata_sha256' "$1"; }

server_env() {
    local arm="$1"; shift
    env PI05_CONTROL_DIR="$CONTROL" PI05_FLOW_STEPS=4 \
        PI05_FULL_CONTEXT_PLAN="$(plan_for "$arm")" \
        PI05_FULL_CONTEXT_HESSIAN_W4="$(hessian_for "$arm")" \
        PI05_FULL_CONTEXT_PACK_DIR="$PACK_DIR" PI05_FULL_CONTEXT_BUFFER="$BUFFER" "$@"
}

archive_and_restart() {
    local row arm shard gpu port instance source target
    mkdir -p "$AUDIT"
    for row in "${RESTARTS[@]}"; do
        IFS=, read -r arm shard gpu port <<<"$row"; instance="$(instance_for "$arm" "$shard")"
        for suffix in runtime.json server.log; do
            source="$CONTROL/$instance.$suffix"; target="$AUDIT/$instance.$suffix"
            if [[ -e "$target" ]]; then cmp -s "$source" "$target" || { echo "restart audit archive drift: $target" >&2; return 1; }
            else
                cp --preserve=timestamps "$source" "$target"
            fi
        done
        env PI05_CONTROL_DIR="$CONTROL" "$SERVER_MANAGER" stop "$instance"
    done
    for row in "${RESTARTS[@]}"; do
        IFS=, read -r arm shard gpu port <<<"$row"; instance="$(instance_for "$arm" "$shard")"
        server_env "$arm" "$SERVER_MANAGER" start "$CONFIG_ID" "$gpu" "$port" "$instance"
    done
}

smoke_one() {
    local label="$1" arm="$2" server_shard="$3" egl="$4" port="$5"
    local instance runtime hash output
    instance="$(instance_for "$arm" "$server_shard")"; runtime="$CONTROL/$instance.runtime.json"
    hash="$(semantic_hash "$runtime")"; output="$PREFLIGHT/$label.jsonl"
    mkdir -p "$PREFLIGHT/$label"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" --host 127.0.0.1 --port "$port" \
        --config-id "$CONFIG_ID" --task-set atomic_seen --tasks OpenDrawer \
        --trial-seeds 3 --split target --replan-steps 16 --flow-steps 4 \
        --egl-device "$egl" --expected-server-metadata-sha256 "$hash" --max-steps 1 \
        --resume-dir "$PREFLIGHT/$label" --out "$output" >"$PREFLIGHT/$label.log" 2>&1
}

post_restart_smoke() {
    local pids=() p
    mkdir -p "$PREFLIGHT"
    smoke_one gpu2 transferred_initializer 0 2 19800 & pids+=("$!")
    smoke_one gpu3 single_best 0 3 19820 & pids+=("$!")
    smoke_one gpu4_two1 two_best 1 4 19841 & pids+=("$!")
    smoke_one gpu5_single2 single_best 2 5 19822 & pids+=("$!")
    smoke_one gpu6_single3 single_best 3 6 19823 & pids+=("$!")
    smoke_one gpu6_two3 two_best 3 6 19843 & pids+=("$!")
    smoke_one gpu7 two_best 7 7 19847 & pids+=("$!")
    for p in "${pids[@]}"; do wait "$p"; done
}

launch_worker() {
    local stage="$1" arm="$2" shard="$3" server_shard="$4" egl="$5" port="$6"
    local instance runtime hash worker_id log pid
    instance="$(instance_for "$arm" "$server_shard")"; runtime="$CONTROL/$instance.runtime.json"
    hash="$(semantic_hash "$runtime")"; worker_id="clean${stage}_${arm}_s${shard}"
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
    for pid_file in "$WORKER_CONTROL"/clean"$stage"_*.pid; do
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
    logs="$(find "$WORKER_CONTROL" -maxdepth 1 -name "clean${stage}_*.log" | wc -l)"
    [[ "$logs" -eq "$expected" ]] || { echo "stage $stage log count $logs != $expected" >&2; return 1; }
    if rg -n "Traceback|Segmentation fault|CUDA out of memory|CUDNN_STATUS|CUBLAS_STATUS|unable to find an engine" "$WORKER_CONTROL"/clean"$stage"_*.log; then
        echo "stage $stage contains a resource/runtime failure" >&2; return 1
    fi
    completed="$({ rg -l 'formal seeded worker complete:' "$WORKER_CONTROL"/clean"$stage"_*.log || true; } | wc -l)"
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
    archive_and_restart
    post_restart_smoke
    "$OPENPI_PY" "$FREEZER"
    launch_stage 1
    nohup setsid "$0" supervise >"$CONTROL/post_restart_supervisor.log" 2>&1 </dev/null &
    echo $! >"$CONTROL/post_restart_supervisor.pid"
    "$AGGREGATOR" status
}

case "${1:-}" in
    recover) recover ;;
    supervise) supervise ;;
    *) echo "usage: $0 recover | supervise" >&2; exit 2 ;;
esac
