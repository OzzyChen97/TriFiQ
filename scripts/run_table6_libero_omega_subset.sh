#!/usr/bin/env bash
set -euo pipefail

# Resumable local FP16/Omega-QVLA subset of the preregistered Table 6 matrix.
# Every cell uses all safely available policy GPUs and is promoted only after
# exact (task_id, initial_state_index) validation.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
OFFICIAL_ROOT="$REPO_ROOT/external/Omega-QVLA"
RUN_ROOT="${TABLE6_RUN_ROOT:-$REPO_ROOT/runs/table6_libero_v1}"
RESULTS_ROOT="$RUN_ROOT/results"
STAGING_ROOT="$RUN_ROOT/staging"
CONTROL_ROOT="$RUN_ROOT/control"
VALIDATOR="$REPO_ROOT/scripts/tools/validate_table6_libero_cell.py"
CONDA_ROOT_PATH="/home1/gyy/probe/miniforge3"
OPENPI_ROOT_PATH="$REPO_ROOT/code/pi05/openpi"
OPENPI_PY_PATH="$CONDA_ROOT_PATH/envs/openpi/bin/python"
OPENPI_CLIENT_PATH="$OPENPI_ROOT_PATH/packages/openpi-client/src"
LIBERO_ROOT_PATH="$REPO_ROOT/code/LIBERO"
LIBERO_PY_PATH="$CONDA_ROOT_PATH/envs/libero_test/bin/python"
PY="$CONDA_ROOT_PATH/envs/robocasa365/bin/python"
GPU_LIST_DEFAULT="${TABLE6_GPU_LIST:-auto}"
GR00T_FP16_PROCS_PER_GPU="${TABLE6_GR00T_FP16_PROCS_PER_GPU:-5}"
PI05_FP16_PROCS_PER_GPU="${TABLE6_PI05_FP16_PROCS_PER_GPU:-4}"
OMEGA_PROCS_PER_GPU="${TABLE6_OMEGA_PROCS_PER_GPU:-2}"
GR00T_FP16_STARTS_PER_GPU="${TABLE6_GR00T_FP16_STARTS_PER_GPU:-2}"
PI05_FP16_STARTS_PER_GPU="${TABLE6_PI05_FP16_STARTS_PER_GPU:-1}"

usage() {
    echo "usage: $0 run | run-cell MODEL CONFIG SUITE | run-stage MODEL CONFIG SUITE OUTPUT GPU_LIST | status" >&2
}

available_gpu_list() {
    local procs_per_gpu="$1" process_mib="$2" reserve_mib="$3"
    if [[ "$GPU_LIST_DEFAULT" != auto ]]; then
        echo "$GPU_LIST_DEFAULT"
        return
    fi
    [[ "$procs_per_gpu" =~ ^[1-5]$ ]] || {
        echo "per-GPU process count must be an integer in [1,5]" >&2
        return 2
    }
    local cards=() capacities=() rows=() index free utilization replica capacity card_index
    while IFS=',' read -r index free utilization; do
        index="${index//[[:space:]]/}"
        free="${free//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        capacity=$(( (free - reserve_mib) / process_mib ))
        (( capacity > procs_per_gpu )) && capacity="$procs_per_gpu"
        if (( capacity > 0 && utilization <= 10 )); then
            cards+=("$index")
            capacities+=("$capacity")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
    for ((replica = 0; replica < procs_per_gpu; replica++)); do
        for card_index in "${!cards[@]}"; do
            if (( replica < capacities[card_index] )); then rows+=("${cards[$card_index]}"); fi
        done
    done
    [[ "${#rows[@]}" -gt 0 ]] || {
        echo "no safely free GPU for process=${process_mib}MiB reserve=${reserve_mib}MiB" >&2
        return 1
    }
    local IFS=,
    echo "${rows[*]}"
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

valid_quant_runtime() {
    local cell="$1" log
    local server_logs=("$cell"/logs/server_shard_*.log)
    local driver_logs=("$cell"/logs/driver_shard_*.log)
    (( ${#server_logs[@]} > 0 && ${#server_logs[@]} == ${#driver_logs[@]} )) || return 1
    for log in "${server_logs[@]}"; do
        rg -q 'Total layers replaced: [1-9][0-9]*' "$log" || return 1
        rg -q '\[REPLACED\].*W4 A4 quant=1' "$log" || return 1
    done
}

valid_cell() {
    local cell="$1" config="${2:-}"
    "$PY" "$VALIDATOR" "$cell/merged_summary.json" >/dev/null 2>&1 || return 1
    if [[ "$config" == omega_qvla_w4a4 ]]; then
        valid_quant_runtime "$cell"
    fi
}

invoke_cell() {
    local model="$1" config="$2" suite="$3" output="$4" gpu_list="$5"
    mkdir -p "$output" "$RUN_ROOT/cache" "$RUN_ROOT/libero_config" "$RUN_ROOT/numba_cache"
    (
        export QUANTVLA_ROOT="$OFFICIAL_ROOT"
        export QUANTVLA_CONDA_ENV=groot_test
        export CONDA_ROOT="$CONDA_ROOT_PATH"
        export LIBERO_ROOT="$LIBERO_ROOT_PATH"
        export LIBERO_CONDA_ENV=libero_test
        export LIBERO_PY="$LIBERO_PY_PATH" EVAL_PY="$LIBERO_PY_PATH"
        export LIBERO_CONFIG_PATH="/home1/gyy/.libero"
        export PYTHONPATH="$OPENPI_CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}"
        export QUANTVLA_CACHE_ROOT="$RUN_ROOT/cache"
        export NUMBA_CACHE_DIR="$RUN_ROOT/numba_cache"
        export SUITE="$suite" GPU_LIST="$gpu_list" PORT_BASE="${TABLE6_PORT_BASE:-19100}"
        export WBITS=4 ABITS=4 NUM_TRIALS_PER_TASK=10
        export GR00T_EVAL_INIT_OFFSET=10 DENOISING_STEPS=8 REPLAN_STEPS=5
        export GR00T_SHARD_STAGGER_MODE=wait_eval_batch
        if [[ "$config" == fp16 && "$model" == gr00t ]]; then
            export GR00T_SHARD_STARTS_PER_GPU="$GR00T_FP16_STARTS_PER_GPU"
        elif [[ "$config" == fp16 ]]; then
            export GR00T_SHARD_STARTS_PER_GPU="$PI05_FP16_STARTS_PER_GPU"
        else
            export GR00T_SHARD_STARTS_PER_GPU=1
        fi
        export GR00T_SHARD_WAIT_EVAL_TIMEOUT_S=240
        # Keep the episode queue elastic for cards that become free after this
        # cell starts.  The Python parent owns hot-joined workers and waits for
        # the complete queue before exact validation.
        export GR00T_DYNAMIC_AUTO_JOIN=1
        export GR00T_DYNAMIC_GPU_POOL="${TABLE6_DYNAMIC_GPU_POOL:-auto}"
        export GR00T_DYNAMIC_JOIN_POLL_S="${TABLE6_DYNAMIC_JOIN_POLL_S:-15}"
        export GR00T_DYNAMIC_STARTS_PER_GPU_SCAN=1
        export GR00T_DYNAMIC_QUEUE_DRAIN_TIMEOUT_S=14400
        if [[ "$config" == fp16 && "$model" == gr00t ]]; then
            export GR00T_DYNAMIC_MAX_PROCS_PER_GPU="$GR00T_FP16_PROCS_PER_GPU"
            export GR00T_DYNAMIC_PROCESS_MIB=7600 GR00T_DYNAMIC_RESERVE_MIB=4000
        elif [[ "$config" == fp16 ]]; then
            export GR00T_DYNAMIC_MAX_PROCS_PER_GPU="$PI05_FP16_PROCS_PER_GPU"
            export GR00T_DYNAMIC_PROCESS_MIB=9500 GR00T_DYNAMIC_RESERVE_MIB=4000
        else
            export GR00T_DYNAMIC_MAX_PROCS_PER_GPU="$OMEGA_PROCS_PER_GPU"
            export GR00T_DYNAMIC_PROCESS_MIB=17000 GR00T_DYNAMIC_RESERVE_MIB=2000
        fi
        export GR00T_SAVE_VIDEOS=0 OUTPUT_ROOT="$output"
        export NO_ALBUMENTATIONS_UPDATE=1
        export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3
        unset GR00T_TASK_IDS_OVERRIDE 2>/dev/null || true

        if [[ "$model" == gr00t ]]; then
            export CHECKPOINT
            CHECKPOINT="$(checkpoint_for_suite "$suite")"
            if [[ "$config" == fp16 ]]; then
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
        elif [[ "$model" == pi05 ]]; then
            export METHOD="$config"
            [[ "$config" == omega_qvla_w4a4 ]] && export METHOD=gptq
            export OPENPI_ROOT="$OPENPI_ROOT_PATH" OPENPI_PY="$OPENPI_PY_PATH"
            export OPENPI_CONFIG=pi05_libero
            export OPENPI_CHECKPOINT="$REPO_ROOT/code/pi05/checkpoints/pi05_libero_pytorch"
            export OPENPI_MODEL_DTYPE=float16
            if [[ "$config" == omega_qvla_w4a4 ]]; then
                local include_both pack
                include_both='.*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*'
                pack="$REPO_ROOT/checkpoints/omega_qvla/pi05/pi05_${suite}/quantized.pt"
                [[ -f "$pack" ]] || { echo "missing pack: $pack" >&2; return 1; }
                export OPENPI_GPTQ_PATH="$pack" OPENPI_GPTQ_INCLUDE="$include_both"
            else
                unset OPENPI_GPTQ_PATH OPENPI_GPTQ_INCLUDE 2>/dev/null || true
            fi
            bash "$OFFICIAL_ROOT/scripts/run_pi05_libero_benchmark.sh"
        else
            echo "unknown model: $model" >&2; return 2
        fi
    )
}

run_cell() {
    local model="$1" config="$2" suite="$3"
    [[ "$model" == gr00t || "$model" == pi05 ]] || { echo "invalid model: $model" >&2; return 2; }
    [[ "$config" == fp16 || "$config" == omega_qvla_w4a4 ]] || { echo "invalid config: $config" >&2; return 2; }
    checkpoint_for_suite "$suite" >/dev/null
    local final="$RESULTS_ROOT/$model/$config/$suite" gpu_list procs_per_gpu process_mib reserve_mib
    if valid_cell "$final" "$config"; then
        echo "[table6] reuse complete cell $model/$config/$suite"
        return
    fi
    if [[ -e "$final" ]]; then
        echo "[table6] incomplete cell retained for audit: $final" >&2
        return 1
    fi
    mkdir -p "$STAGING_ROOT/$model/$config"
    local stage
    stage="$(mktemp -d "$STAGING_ROOT/$model/$config/${suite}.XXXXXX")"
    if [[ "$config" == fp16 ]]; then
        if [[ "$model" == gr00t ]]; then
            procs_per_gpu="$GR00T_FP16_PROCS_PER_GPU"
            process_mib=7600
        else
            procs_per_gpu="$PI05_FP16_PROCS_PER_GPU"
            process_mib=9500
        fi
        reserve_mib=4000
    else
        procs_per_gpu="$OMEGA_PROCS_PER_GPU"
        process_mib=17000
        reserve_mib=2000
    fi
    gpu_list="$(available_gpu_list "$procs_per_gpu" "$process_mib" "$reserve_mib")"
    echo "[table6] $model/$config/$suite GPUs=$gpu_list"
    invoke_cell "$model" "$config" "$suite" "$stage" "$gpu_list"
    "$PY" "$VALIDATOR" "$stage/merged_summary.json"
    if [[ "$config" == omega_qvla_w4a4 ]]; then
        valid_quant_runtime "$stage" || {
            echo "[table6] quantized runtime evidence failed: $stage" >&2
            return 1
        }
    fi
    mkdir -p "$(dirname "$final")"
    mv "$stage" "$final"
    echo "[table6] promoted $model/$config/$suite"
}

run_all() {
    mkdir -p "$CONTROL_ROOT"
    exec 9>"$CONTROL_ROOT/omega_subset.lock"
    flock -n 9 || { echo "Table 6 Omega subset is already running" >&2; return 1; }
    local model config suite
    for model in gr00t pi05; do
        for config in fp16 omega_qvla_w4a4; do
            for suite in goal spatial object long; do
                run_cell "$model" "$config" "$suite"
            done
        done
    done
}

status_all() {
    local complete=0 total=0 episodes=0 model config suite
    for model in gr00t pi05; do
        for config in fp16 omega_qvla_w4a4; do
            for suite in goal spatial object long; do
                total=$((total + 1))
                if valid_cell "$RESULTS_ROOT/$model/$config/$suite" "$config"; then
                    complete=$((complete + 1)); episodes=$((episodes + 100))
                fi
            done
        done
    done
    echo "table6_omega_subset cells=$complete/$total episodes=$episodes/1600 gpu_policy=$GPU_LIST_DEFAULT gr00t_fp16_procs_per_gpu=$GR00T_FP16_PROCS_PER_GPU pi05_fp16_procs_per_gpu=$PI05_FP16_PROCS_PER_GPU omega_procs_per_gpu=$OMEGA_PROCS_PER_GPU"
}

cd "$REPO_ROOT"
case "${1:-}" in
    run) run_all ;;
    run-cell) [[ $# -eq 4 ]] || { usage; exit 2; }; run_cell "$2" "$3" "$4" ;;
    run-stage)
        [[ $# -eq 6 ]] || { usage; exit 2; }
        invoke_cell "$2" "$3" "$4" "$5" "$6"
        ;;
    status) status_all ;;
    *) usage; exit 2 ;;
esac
