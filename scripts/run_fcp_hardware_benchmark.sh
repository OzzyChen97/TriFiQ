#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
BENCH="$REPO_ROOT/scripts/tools/benchmark_gr00t_hardware.py"
AGGREGATE="$REPO_ROOT/scripts/tools/aggregate_gr00t_hardware.py"
SUBSET="$REPO_ROOT/scripts/tools/materialize_full_context_hessian_subset.py"
ROOT="$REPO_ROOT/runs/full_context_v2/fcp_completion/hardware"
PROTOCOL="$REPO_ROOT/scripts/quantvla_fcp_hardware_protocol.json"
GPU="${FCP_HARDWARE_GPU:-4}"

CHECKPOINT="$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000"
BUFFER="$REPO_ROOT/runs/full_context_v2/selection_buffer_splits/selection_buffer_atomic_seen.npz"
IDENTITY_PACK="$REPO_ROOT/runs/errorfold_v3_15x20/calibration/gr00t/atomic_seen/identity_pack"
PARENT_HESSIAN="$REPO_ROOT/runs/errorfold_v3_15x20/calibration/gr00t/atomic_seen/hessian_w4.npz"
M0_PLAN="$REPO_ROOT/runs/full_context_v2/fcp_completion/gr00t_corrected_fcp_frozen.json"
M0_HESSIAN="$REPO_ROOT/runs/full_context_v2/gr00t_main_hessian/atomic_seen/hessian_w4.npz"
QVLA_PLAN="$REPO_ROOT/checkpoints/packs/robocasa365/gr00t_quant_plan_robocasa365_cscka_16to1_adjudicated.final_plan.json"
QVLA_PACK="$REPO_ROOT/checkpoints/packs/robocasa365/duquant_packed_robocasa365_protocolfix_d4_w4a8_b64c32ls015"
QVLA_A8="$REPO_ROOT/runs/full_context_v1/gr00t/deployment/atomic_seen/gdsq_main_a8_scales.npz"
CANDIDATES=(single_best two_best attention_6 mlp_2 ff_pair_0 dp_full_lambda_1p0)
CONFIGS=(native_fp16 quantvla_w4a8 context_base "${CANDIDATES[@]}")

prepare() {
    mkdir -p "$ROOT/hessian" "$ROOT/trials"
    local candidate_id plan_path subset_path
    for candidate_id in "${CANDIDATES[@]}"; do
        plan_path="$REPO_ROOT/runs/full_context_v2/statistics_correction/proposals/$candidate_id.json"
        subset_path="$ROOT/hessian/$candidate_id/hessian_w4.npz"
        if [[ ! -f "$subset_path" || ! -f "$subset_path.json" ]]; then
            mkdir -p "$(dirname "$subset_path")"
            "$PYTHON" "$SUBSET" --parent "$PARENT_HESSIAN" --plan "$plan_path" --model gr00t --out "$subset_path"
        fi
    done
}

run_one() {
    local trial config_id output_path
    trial="$1"
    config_id="$2"
    output_path="$ROOT/trials/trial${trial}_${config_id}.json"
    if [[ -f "$output_path" ]]; then
        echo "[hardware] reuse trial=$trial config=$config_id" >&2
        return
    fi
    local args=(
        --config-id "$config_id" --trial "$trial" --physical-gpu "$GPU"
        --checkpoint "$CHECKPOINT" --buffer "$BUFFER"
        --activation-mode fp16 --out "$output_path"
    )
    case "$config_id" in
        native_fp16)
            ;;
        quantvla_w4a8)
            args=(
                --config-id "$config_id" --trial "$trial" --physical-gpu "$GPU"
                --checkpoint "$CHECKPOINT" --buffer "$BUFFER"
                --plan "$QVLA_PLAN" --pack-dir "$QVLA_PACK" --act-scale "$QVLA_A8"
                --activation-mode static_a8 --out "$output_path"
            )
            ;;
        context_base)
            args=(
                --config-id "$config_id" --trial "$trial" --physical-gpu "$GPU"
                --checkpoint "$CHECKPOINT" --buffer "$BUFFER"
                --plan "$M0_PLAN" --pack-dir "$IDENTITY_PACK" --hessian-w4 "$M0_HESSIAN"
                --activation-mode dynamic_a8 --out "$output_path"
            )
            ;;
        *)
            args=(
                --config-id "$config_id" --trial "$trial" --physical-gpu "$GPU"
                --checkpoint "$CHECKPOINT" --buffer "$BUFFER"
                --plan "$REPO_ROOT/runs/full_context_v2/statistics_correction/proposals/$config_id.json"
                --pack-dir "$IDENTITY_PACK" --hessian-w4 "$ROOT/hessian/$config_id/hessian_w4.npz"
                --activation-mode dynamic_a8 --out "$output_path"
            )
            ;;
    esac
    echo "[hardware] trial=$trial config=$config_id gpu=$GPU" >&2
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$BENCH" "${args[@]}"
}

run_all() {
    prepare
    local n="${#CONFIGS[@]}"
    for trial in 0 1 2; do
        for ((position=0; position<n; position++)); do
            local index=$(((position + trial) % n))
            run_one "$trial" "${CONFIGS[$index]}"
        done
    done
    "$PYTHON" "$AGGREGATE" \
        --trials "$ROOT/trials" --protocol "$PROTOCOL" --out "$ROOT/summary.json"
}

case "${1:-}" in
    prepare) prepare ;;
    run) run_all ;;
    aggregate)
        "$PYTHON" "$AGGREGATE" \
            --trials "$ROOT/trials" --protocol "$PROTOCOL" --out "$ROOT/summary.json"
        ;;
    *) echo "usage: $0 prepare | run | aggregate" >&2; exit 2 ;;
esac
