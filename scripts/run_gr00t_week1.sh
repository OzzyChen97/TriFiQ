#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
GROOT_PY="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
EXECUTION="$REPO_ROOT/runs/gdsq_week1_preregistered_v1/execution"
MATERIALIZER="$REPO_ROOT/scripts/tools/gdsq_week1_execution.py"
CALIBRATOR="$REPO_ROOT/scripts/tools/calibrate_a8_plan_gr00t.py"
SCREEN="$REPO_ROOT/scripts/tools/gdsq_week1_gr00t_screen.py"
MATRIX="$REPO_ROOT/scripts/tools/run_robocasa_atomic_matrix.py"
PARSER="$REPO_ROOT/scripts/tools/parse_robocasa_atomic_matrix.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_robocasa365_official.py"
CHECKPOINT_ROOT="$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining"
PACK_ROOT="$REPO_ROOT/checkpoints/packs/robocasa365"
GPU="${GDSQ_WEEK1_GPU:-3}"
SEEDS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49"
DEV_TASKS="CoffeeSetupMug,OpenCabinet,OpenStandMixerHead,PickPlaceDrawerToCounter"
PRIMARY14="CloseBlenderLid,CloseFridge,CloseToasterOvenDoor,NavigateKitchen,OpenDrawer,PickPlaceCounterToCabinet,PickPlaceCounterToStove,PickPlaceSinkToCounter,PickPlaceToasterToCounter,SlideDishwasherRack,TurnOffStove,TurnOnElectricKettle,TurnOnMicrowave,TurnOnSinkFaucet"
UNIFORM_SPEC_SUFFIX="${GROOT_WEEK1_UNIFORM_SPEC_SUFFIX:-8gpu_v1}"
UNIFORM_EGL_POOL="${GROOT_WEEK1_UNIFORM_EGL_POOL:-0,1,2,3,4,5,6,7}"
UNIFORM_SHARDS="${GROOT_WEEK1_UNIFORM_SHARDS:-18}"

usage() {
    echo "usage: $0 materialize | prepare-uniform | run-uniform | run-uniform-unseen-accelerated | prepare-ablations | run-ablations | screen-controls | run-dev-controls | select-controls | run-formal-controls | run-all | status" >&2
}

gpu_free() {
    local gpu="$1" output
    output="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    [[ -z "${output//[[:space:]]/}" ]]
}

require_gpu_free() {
    local gpu="$1" attempt
    # CUDA processes can remain visible to nvidia-smi briefly after a
    # sequential calibration exits.  Retry that teardown window, while still
    # refusing a genuinely occupied device.
    for attempt in {1..15}; do
        gpu_free "$gpu" && return
        sleep 2
    done
    echo "refusing to start GR00T calibration/screen: GPU $gpu is occupied" >&2
    return 1
}

valid_a8() {
    local plan="$1" checkpoint="$2" output="$3"
    [[ -s "$output" && -s "$output.meta.json" ]] || return 1
    "$GROOT_PY" - "$plan" "$checkpoint" "$output" <<'PY'
import hashlib, json, pathlib, sys
import numpy as np

plan, checkpoint, output = map(pathlib.Path, sys.argv[1:])
meta = json.loads(pathlib.Path(str(output) + ".meta.json").read_text())
checks = [
    meta.get("plan_sha256") == hashlib.sha256(plan.read_bytes()).hexdigest(),
    pathlib.Path(meta.get("checkpoint_path", "")).resolve() == checkpoint.resolve(),
    int(meta.get("denoising_steps", -1)) == 4,
    int(meta.get("calibration_seed", -1)) == 0,
    int(meta.get("calib_batches", -1)) == 32,
    int(meta.get("wrapped_layers", 0)) > 0,
]
with np.load(output, allow_pickle=False) as arrays:
    checks.extend((bool(arrays.files), all(np.isfinite(arrays[key]).all() for key in arrays.files)))
raise SystemExit(0 if all(checks) else 1)
PY
}

materialize() {
    "$ROBOCASA_PY" "$MATERIALIZER" materialize
}

calibrate() {
    local plan="$1" checkpoint="$2" pack="$3" output="$4"
    if valid_a8 "$plan" "$checkpoint" "$output"; then
        echo "[week1] validated existing GR00T A8: $output"
        return
    fi
    require_gpu_free "$GPU"
    mkdir -p "$(dirname "$output")"
    CUDA_VISIBLE_DEVICES="$GPU" "$GROOT_PY" "$CALIBRATOR" \
        --model-path "$checkpoint" \
        --plan "$plan" \
        --packdir "$pack" \
        --out "$output" \
        --suite robocasa365_atomic \
        --obs-format robocasa365 \
        --denoising-steps 4 \
        --batch-size 8 \
        --calib-steps 32 \
        --calibration-seed 0
}

prepare_uniform() {
    materialize
    local task_set suffix
    for task_set in atomic_seen composite_seen composite_unseen; do
        case "$task_set" in
            atomic_seen) suffix=protocolfix ;;
            composite_seen) suffix=composite_seen ;;
            composite_unseen) suffix=composite_unseen ;;
        esac
        calibrate \
            "$EXECUTION/plans/gr00t/uniform_w6/$task_set.plan.json" \
            "$CHECKPOINT_ROOT/$task_set/checkpoint-60000" \
            "$PACK_ROOT/duquant_packed_robocasa365_${suffix}_d4_w4a8_b64c32ls015" \
            "$EXECUTION/a8/gr00t/uniform_w6_$task_set.npz"
    done
}

run_matrix_until_complete() {
    local spec="$1" run_dir="$2" task_set="$3" checkpoint="$4" tasks="$5" phase="$6" shards="$7"
    local args=(
        --spec "$spec"
        --run-dir "$run_dir"
        --phase "$phase"
        --task-set "$task_set"
        --seeds "$SEEDS"
        --checkpoint "$checkpoint"
        --n-shards "$shards"
        --trial-batch-size 10
        --trial-timeout 3600
        --action-noise paired
        --formal-provenance-v2
        --gpu-sample-interval 10
        --egl-device-pool "$UNIFORM_EGL_POOL"
    )
    if [[ -n "$tasks" ]]; then
        args+=(--tasks "$tasks")
    fi
    if [[ "$phase" == "dev" ]]; then
        args+=(--dev-tasks "$DEV_TASKS")
    fi
    until "$GROOT_PY" "$MATRIX" "${args[@]}"; do
        echo "[week1] matrix retry in 30s: $run_dir" >&2
        sleep 30
    done
    "$GROOT_PY" "$PARSER" --run-dir "$run_dir" --bootstrap 10000 \
        >"$run_dir/strict_parse.log"
}

preflight_valid() {
    local run_dir="$1" expected_configs="$2"
    "$ROBOCASA_PY" - "$run_dir" "$expected_configs" <<'PY'
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1]); expected=int(sys.argv[2])
manifest_path=root / "manifest.json"; summary_path=root / "summary.json"
if not manifest_path.is_file() or not summary_path.is_file(): raise SystemExit(1)
manifest=json.load(open(manifest_path)); summary=json.load(open(summary_path))
sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
checks=[
    manifest.get("schema_version") == 2,
    manifest.get("diagnostic_only") is True,
    manifest.get("phase") == "formal",
    len(manifest.get("tasks", [])) == 2,
    manifest.get("seeds") == [0, 1],
    len(manifest.get("configs", [])) == expected,
    summary.get("manifest_sha256") == sha,
    summary.get("bootstrap_draws") == 10000,
    not summary.get("validation_errors"),
    summary.get("primary_scope", {}).get("n_tasks") == 2,
    len(summary.get("configs", {})) == expected,
    all(row.get("episodes") == 4 for row in summary.get("configs", {}).values()),
]
raise SystemExit(0 if all(checks) else 1)
PY
}

run_preflight_until_complete() {
    local spec="$1" run_dir="$2" task_set="$3" checkpoint="$4" tasks="$5" expected_configs="$6"
    if preflight_valid "$run_dir" "$expected_configs"; then
        echo "[week1] reuse strict 2-task x 2-seed preflight: $run_dir"
        return
    fi
    local args=(
        --spec "$spec"
        --run-dir "$run_dir"
        --phase formal
        --diagnostic-only
        --formal-provenance-v2
        --task-set "$task_set"
        --tasks "$tasks"
        --seeds 0,1
        --checkpoint "$checkpoint"
        --n-shards 2
        --trial-batch-size 2
        --trial-timeout 3600
        --action-noise paired
        --gpu-sample-interval 10
        --egl-device-pool "$UNIFORM_EGL_POOL"
    )
    until "$GROOT_PY" "$MATRIX" "${args[@]}"; do
        echo "[week1] preflight retry in 30s: $run_dir" >&2
        sleep 30
    done
    "$GROOT_PY" "$PARSER" --run-dir "$run_dir" --bootstrap 10000 \
        >"$run_dir/strict_parse.log"
    preflight_valid "$run_dir" "$expected_configs"
}

run_uniform() {
    local task_set run_dir spec checkpoint preflight_tasks
    for task_set in atomic_seen composite_seen composite_unseen; do
        run_dir="$EXECUTION/runs/gr00t_uniform_w6_$task_set"
        spec="$EXECUTION/specs/gr00t_uniform_w6_${task_set}_${UNIFORM_SPEC_SUFFIX}.json"
        checkpoint="$CHECKPOINT_ROOT/$task_set/checkpoint-60000"
        case "$task_set" in
            atomic_seen) preflight_tasks="OpenDrawer,TurnOnMicrowave" ;;
            composite_seen) preflight_tasks="DeliverStraw,PrepareCoffee" ;;
            composite_unseen) preflight_tasks="ArrangeTea,PanTransfer" ;;
        esac
        run_preflight_until_complete "$spec" \
            "$EXECUTION/preflight/gr00t_uniform_w6_$task_set" \
            "$task_set" "$checkpoint" "$preflight_tasks" 1
        # A task-level horizon-balanced shard schedule hides simulator latency
        # without changing the formal task/seed key set. Resource placement is
        # frozen by the spec and manifest before the first result row.
        run_matrix_until_complete "$spec" "$run_dir" "$task_set" "$checkpoint" "" formal "$UNIFORM_SHARDS"
    done
    mkdir -p "$EXECUTION/aggregate/gr00t_uniform_w6"
    "$GROOT_PY" "$AGGREGATOR" \
        --run-dir "$EXECUTION/runs/gr00t_uniform_w6_atomic_seen" \
        --run-dir "$EXECUTION/runs/gr00t_uniform_w6_composite_seen" \
        --run-dir "$EXECUTION/runs/gr00t_uniform_w6_composite_unseen" \
        --out-dir "$EXECUTION/aggregate/gr00t_uniform_w6" \
        --bootstrap 10000
}

run_uniform_unseen_accelerated() {
    local task_set="composite_unseen"
    local run_dir="$EXECUTION/runs/gr00t_uniform_w6_composite_unseen_14shard_v2"
    local spec="$EXECUTION/specs/gr00t_uniform_w6_composite_unseen_7gpu_14shard_v2.json"
    local checkpoint="$CHECKPOINT_ROOT/composite_unseen/checkpoint-60000"
    local egl_pool="1,2,3,4,5,6,7"
    UNIFORM_EGL_POOL="$egl_pool" run_preflight_until_complete "$spec" \
        "$EXECUTION/preflight/gr00t_uniform_w6_composite_unseen_14shard_v2" \
        "$task_set" "$checkpoint" "ArrangeTea,PanTransfer" 1
    UNIFORM_EGL_POOL="$egl_pool" run_matrix_until_complete "$spec" "$run_dir" \
        "$task_set" "$checkpoint" "" formal 14
    mkdir -p "$EXECUTION/aggregate/gr00t_uniform_w6"
    "$GROOT_PY" "$AGGREGATOR" \
        --run-dir "$EXECUTION/runs/gr00t_uniform_w6_atomic_seen" \
        --run-dir "$EXECUTION/runs/gr00t_uniform_w6_composite_seen" \
        --run-dir "$run_dir" \
        --out-dir "$EXECUTION/aggregate/gr00t_uniform_w6" \
        --bootstrap 10000
}

prepare_ablations() {
    materialize
    local checkpoint="$CHECKPOINT_ROOT/atomic_seen/checkpoint-60000"
    local pack="$PACK_ROOT/duquant_packed_robocasa365_protocolfix_d4_w4a8_b64c32ls015"
    calibrate \
        "$REPO_ROOT/runs/gdsq_week1_preregistered_v1/plans/gr00t/ablations/weights_uniform.plan.json" \
        "$checkpoint" "$pack" "$EXECUTION/a8/gr00t/ablation_weights_uniform.npz"
    calibrate \
        "$REPO_ROOT/runs/gdsq_week1_preregistered_v1/plans/gr00t/ablations/no_guards.plan.json" \
        "$checkpoint" "$pack" "$EXECUTION/a8/gr00t/ablation_no_guards.npz"
    calibrate \
        "$REPO_ROOT/runs/gdsq_week1_preregistered_v1/plans/gr00t/ablations/no_functional_adjudication.plan.json" \
        "$checkpoint" "$pack" "$EXECUTION/a8/gr00t/ablation_no_functional_adjudication.npz"
}

run_ablations() {
    run_preflight_until_complete \
        "$EXECUTION/specs/gr00t_core_ablations_primary14_v3_8gpu.json" \
        "$EXECUTION/preflight/gr00t_core_ablations_primary14" \
        atomic_seen \
        "$CHECKPOINT_ROOT/atomic_seen/checkpoint-60000" \
        "OpenDrawer,TurnOnMicrowave" 5
    run_matrix_until_complete \
        "$EXECUTION/specs/gr00t_core_ablations_primary14_v3_8gpu.json" \
        "$EXECUTION/runs/gr00t_core_ablations_primary14" \
        atomic_seen \
        "$CHECKPOINT_ROOT/atomic_seen/checkpoint-60000" \
        "$PRIMARY14" formal 4
}

screen_controls() {
    require_gpu_free "$GPU"
    mkdir -p "$EXECUTION/screen"
    CUDA_VISIBLE_DEVICES="$GPU" "$GROOT_PY" "$SCREEN" \
        --family random \
        --out "$EXECUTION/screen/gr00t_search_matched_random.json"
    CUDA_VISIBLE_DEVICES="$GPU" "$GROOT_PY" "$SCREEN" \
        --family action_only \
        --out "$EXECUTION/screen/gr00t_action_only.json"
    "$ROBOCASA_PY" "$MATERIALIZER" materialize-dev-specs
}

run_dev_controls() {
    "$ROBOCASA_PY" "$MATERIALIZER" materialize-dev-specs
    local wave expected_configs
    for wave in 1 2 3; do
        expected_configs=5
        [[ "$wave" == 3 ]] && expected_configs=4
        run_preflight_until_complete \
            "$EXECUTION/specs/gr00t_controls_dev4_wave${wave}.json" \
            "$EXECUTION/preflight/gr00t_controls_dev4_wave${wave}" \
            atomic_seen \
            "$CHECKPOINT_ROOT/atomic_seen/checkpoint-60000" \
            "CoffeeSetupMug,OpenCabinet" \
            "$expected_configs"
        run_matrix_until_complete \
            "$EXECUTION/specs/gr00t_controls_dev4_wave${wave}.json" \
            "$EXECUTION/runs/gr00t_controls_dev4_wave${wave}" \
            atomic_seen \
            "$CHECKPOINT_ROOT/atomic_seen/checkpoint-60000" \
            "$DEV_TASKS" dev 4
    done
}

select_controls() {
    "$ROBOCASA_PY" "$MATERIALIZER" freeze-dev-selection
    "$ROBOCASA_PY" "$MATERIALIZER" materialize-formal-control-spec
}

run_formal_controls() {
    select_controls
    run_preflight_until_complete \
        "$EXECUTION/specs/gr00t_same_budget_controls_primary14_v2_8gpu.json" \
        "$EXECUTION/preflight/gr00t_same_budget_controls_primary14" \
        atomic_seen \
        "$CHECKPOINT_ROOT/atomic_seen/checkpoint-60000" \
        "OpenDrawer,TurnOnMicrowave" 2
    run_matrix_until_complete \
        "$EXECUTION/specs/gr00t_same_budget_controls_primary14_v2_8gpu.json" \
        "$EXECUTION/runs/gr00t_same_budget_controls_primary14" \
        atomic_seen \
        "$CHECKPOINT_ROOT/atomic_seen/checkpoint-60000" \
        "$PRIMARY14" formal 8
}

status() {
    "$ROBOCASA_PY" - "$EXECUTION" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
for path in sorted((root / "runs").glob("*")):
    if not path.is_dir():
        continue
    rows = 0
    formal_failures = 0
    for result in path.glob("*.jsonl"):
        for line in result.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            formal_failures += int(bool(row.get("formal_failure")))
    print(f"{path.name}: rows={rows} formal_failures={formal_failures} (coverage only)")
for path in sorted((root / "screen").glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"{path.name}: complete={payload.get('complete')} scores={len(payload.get('scores', []))}")
PY
}

run_all() {
    prepare_uniform
    prepare_ablations
    screen_controls
    run_uniform
    run_dev_controls
    run_formal_controls
    run_ablations
}

cd "$REPO_ROOT"
case "${1:-}" in
    materialize) materialize ;;
    prepare-uniform) prepare_uniform ;;
    run-uniform) run_uniform ;;
    run-uniform-unseen-accelerated) run_uniform_unseen_accelerated ;;
    prepare-ablations) prepare_ablations ;;
    run-ablations) run_ablations ;;
    screen-controls) screen_controls ;;
    run-dev-controls) run_dev_controls ;;
    select-controls) select_controls ;;
    run-formal-controls) run_formal_controls ;;
    run-all) run_all ;;
    status) status ;;
    *) usage; exit 2 ;;
esac
