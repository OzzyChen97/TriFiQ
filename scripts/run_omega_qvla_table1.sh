#!/usr/bin/env bash
set -euo pipefail

# Reproduce Omega-QVLA v1 Table 1 with the pinned official release.  Formal
# cells are written through isolated staging directories and promoted only
# after exact task/trial validation, so a crashed attempt cannot contaminate a
# paper-visible cell.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
OFFICIAL_ROOT="$REPO_ROOT/external/Omega-QVLA"
ROOT="$REPO_ROOT/runs/gdsq_extension_preregistered_v1/omega_qvla_table1"
FORMAL_ROOT="$ROOT/results"
PREFLIGHT_ROOT="$ROOT/preflight"
STAGING_ROOT="$ROOT/staging"
PROTOCOL="$ROOT/protocol_v1.json"
CONDA_ROOT_PATH="/home1/gyy/probe/miniforge3"
GROOT_ENV="groot_test"
OPENPI_ROOT_PATH="$REPO_ROOT/code/pi05/openpi"
OPENPI_PY_PATH="$CONDA_ROOT_PATH/envs/openpi/bin/python"
LIBERO_ROOT_PATH="$REPO_ROOT/code/LIBERO"
GPU_LIST_DEFAULT="${OMEGA_GPU_LIST:-3,4,5,6,7}"

usage() {
    echo "usage: $0 preflight | formal | run-cell MODEL CONFIG SUITE | status" >&2
}

checkpoint_for_suite() {
    case "$1" in
        goal) echo "$REPO_ROOT/checkpoints/gr00t/libero-goal" ;;
        spatial) echo "$REPO_ROOT/checkpoints/gr00t/libero-spatial" ;;
        object) echo "$REPO_ROOT/checkpoints/gr00t/libero-object" ;;
        long) echo "$REPO_ROOT/checkpoints/gr00t/libero-long" ;;
        *) echo "unknown suite: $1" >&2; return 2 ;;
    esac
}

validate_output() {
    local output="$1" expected_episodes="$2" expected_tasks="$3" trials="$4"
    local summary="$output/merged_summary.json"
    [[ -f "$summary" ]] || return 1
    jq -e \
        --argjson episodes "$expected_episodes" \
        --argjson tasks "$expected_tasks" \
        --argjson trials "$trials" \
        '.total_episodes == $episodes and
         .num_trials_per_task == $trials and
         (.task_summaries | length) == $tasks and
         (all(.task_summaries[]; .episodes == $trials)) and
         ([.task_summaries[].task_id] | unique | length) == $tasks' \
        "$summary" >/dev/null
}

invoke_official() {
    local model="$1" config="$2" suite="$3" output="$4" trials="$5" task_override="$6" gpu_list="$7" port_base="$8"
    mkdir -p "$output" "$ROOT/cache" "$ROOT/libero_config" "$ROOT/numba_cache"
    (
        export QUANTVLA_ROOT="$OFFICIAL_ROOT"
        export QUANTVLA_CONDA_ENV="$GROOT_ENV"
        export CONDA_ROOT="$CONDA_ROOT_PATH"
        export LIBERO_ROOT="$LIBERO_ROOT_PATH"
        export LIBERO_CONFIG_PATH="$ROOT/libero_config"
        export QUANTVLA_CACHE_ROOT="$ROOT/cache"
        export NUMBA_CACHE_DIR="$ROOT/numba_cache"
        export SUITE="$suite"
        export WBITS=4 ABITS=4
        export GPU_LIST="$gpu_list"
        export PORT_BASE="$port_base"
        export NUM_TRIALS_PER_TASK="$trials"
        export GR00T_EVAL_INIT_OFFSET=10
        export GR00T_SHARD_STAGGER_MODE=wait_eval
        export GR00T_SHARD_WAIT_EVAL_TIMEOUT_S=180
        export OUTPUT_ROOT="$output"
        if [[ -n "$task_override" ]]; then
            export GR00T_TASK_IDS_OVERRIDE="$task_override"
        else
            unset GR00T_TASK_IDS_OVERRIDE 2>/dev/null || true
        fi

        if [[ "$model" == "gr00t" ]]; then
            export CHECKPOINT
            CHECKPOINT="$(checkpoint_for_suite "$suite")"
            if [[ "$config" == "fp16" ]]; then
                export PRESET=fp16
                unset GR00T_GPTQ_PATH_OVERRIDE GR00T_GPTQ_INCLUDE_OVERRIDE 2>/dev/null || true
            else
                local llm_re dit_re pack
                llm_re='.*backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*'
                dit_re='.*action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2)).*'
                pack="$REPO_ROOT/checkpoints/omega_qvla/gr00t/gr00t_${suite}/quantized.pt"
                [[ -f "$pack" ]] || { echo "missing pack: $pack" >&2; return 1; }
                unset PRESET 2>/dev/null || true
                export LLM_QUANT=gptq DIT_QUANT=gptq DIT_ATTN=1 DIT_PERSTEP=1
                export GR00T_GPTQ_PATH_OVERRIDE="$pack"
                export GR00T_GPTQ_INCLUDE_OVERRIDE="(${llm_re}|${dit_re})"
                export GR00T_GPTQ_MISSING=fallback
            fi
            bash "$OFFICIAL_ROOT/scripts/run_groot_benchmark.sh"
        elif [[ "$model" == "pi05" ]]; then
            export METHOD="$config"
            [[ "$config" == "omega_qvla_w4a4" ]] && export METHOD=gptq
            export DENOISING_STEPS=8 REPLAN_STEPS=5
            export OPENPI_ROOT="$OPENPI_ROOT_PATH"
            export OPENPI_PY="$OPENPI_PY_PATH"
            export OPENPI_CONFIG=pi05_libero
            export OPENPI_CHECKPOINT="$REPO_ROOT/code/pi05/checkpoints/pi05_libero_pytorch"
            if [[ "$config" == "omega_qvla_w4a4" ]]; then
                local include_both pack
                include_both='.*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*'
                pack="$REPO_ROOT/checkpoints/omega_qvla/pi05/pi05_${suite}/quantized.pt"
                [[ -f "$pack" ]] || { echo "missing pack: $pack" >&2; return 1; }
                export OPENPI_GPTQ_PATH="$pack"
                export OPENPI_GPTQ_INCLUDE="$include_both"
            else
                unset OPENPI_GPTQ_PATH OPENPI_GPTQ_INCLUDE 2>/dev/null || true
            fi
            bash "$OFFICIAL_ROOT/scripts/run_pi05_libero_benchmark.sh"
        else
            echo "unknown model: $model" >&2
            return 2
        fi
    )
}

run_formal_cell() {
    local model="$1" config="$2" suite="$3"
    [[ "$model" == "gr00t" || "$model" == "pi05" ]] || { echo "invalid model: $model" >&2; return 2; }
    [[ "$config" == "fp16" || "$config" == "omega_qvla_w4a4" ]] || { echo "invalid config: $config" >&2; return 2; }
    checkpoint_for_suite "$suite" >/dev/null
    local final="$FORMAL_ROOT/$model/$config/$suite"
    if validate_output "$final" 100 10 10; then
        echo "[omega table1] complete cell already present: $model/$config/$suite"
        return
    fi
    [[ ! -e "$final" ]] || { echo "refusing to overwrite incomplete formal cell: $final" >&2; return 1; }
    mkdir -p "$STAGING_ROOT/$model/$config"
    local stage
    stage="$(mktemp -d "$STAGING_ROOT/$model/$config/${suite}.XXXXXX")"
    invoke_official "$model" "$config" "$suite" "$stage" 10 "" "$GPU_LIST_DEFAULT" 18800
    validate_output "$stage" 100 10 10 || { echo "formal output failed validation: $stage" >&2; return 1; }
    mkdir -p "$(dirname "$final")"
    mv "$stage" "$final"
    echo "[omega table1] promoted $model/$config/$suite"
}

run_preflight_model() {
    local model="$1" output="$PREFLIGHT_ROOT/$model"
    if validate_output "$output" 1 1 1; then
        echo "[omega table1] preflight already valid: $model"
        return
    fi
    [[ ! -e "$output" ]] || { echo "refusing to overwrite incomplete preflight: $output" >&2; return 1; }
    invoke_official "$model" omega_qvla_w4a4 object "$output" 1 0 "${OMEGA_PREFLIGHT_GPU:-3}" 18700
    validate_output "$output" 1 1 1
}

formal_all() {
    /home1/gyy/probe/miniforge3/envs/robocasa365/bin/python "$REPO_ROOT/scripts/tools/omega_qvla_table1_protocol.py" verify
    local model config suite
    for model in gr00t pi05; do
        for config in fp16 omega_qvla_w4a4; do
            for suite in goal spatial object long; do
                run_formal_cell "$model" "$config" "$suite"
            done
        done
    done
}

status_all() {
    local complete=0 total=0 model config suite
    for model in gr00t pi05; do
        for config in fp16 omega_qvla_w4a4; do
            for suite in goal spatial object long; do
                total=$((total + 1))
                if validate_output "$FORMAL_ROOT/$model/$config/$suite" 100 10 10; then complete=$((complete + 1)); fi
            done
        done
    done
    echo "omega_table1 cells=$complete/$total episodes=$((complete * 100))/1600"
}

cd "$REPO_ROOT"
[[ -f "$PROTOCOL" || "${1:-}" == "status" ]] || { echo "missing frozen protocol: $PROTOCOL" >&2; exit 1; }
case "${1:-}" in
    preflight) run_preflight_model gr00t; run_preflight_model pi05 ;;
    formal) formal_all ;;
    run-cell) [[ $# -eq 4 ]] || { usage; exit 2; }; run_formal_cell "$2" "$3" "$4" ;;
    status) status_all ;;
    *) usage; exit 2 ;;
esac
