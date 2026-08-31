#!/usr/bin/env bash
set -euo pipefail

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=${LIBERO_DYPAC_RUN_ROOT:-$ROOT/runs/libero_dypac_v1}
OFFICIAL=$ROOT/external/Omega-QVLA
LIBERO_ROOT=$ROOT/code/LIBERO
CONDA=/home1/gyy/probe/miniforge3
PY=$CONDA/envs/groot_test/bin/python
EVAL_PY=$CONDA/envs/libero_test/bin/python
ART=$RUN_ROOT/artifacts/gr00t
VALIDATOR=$ROOT/scripts/tools/validate_table6_libero_cell.py
MODE=${1:-formal}
SUITE_FILTER=${2:-all}

[[ "$MODE" == canary || "$MODE" == formal ]] || { echo "usage: $0 canary|formal [goal|spatial|object|long|all]" >&2; exit 2; }
SUITES=(goal spatial object long)
if [[ "$SUITE_FILTER" != all ]]; then SUITES=("$SUITE_FILTER"); fi

suite_name() {
    case "$1" in
        goal) echo libero_goal ;;
        spatial) echo libero_spatial ;;
        object) echo libero_object ;;
        long) echo libero_10 ;;
        *) return 2 ;;
    esac
}

data_config() {
    if [[ "$1" == goal ]]; then
        echo examples.Libero.custom_data_config:LiberoDataConfigMeanStd
    else
        echo examples.Libero.custom_data_config:LiberoDataConfig
    fi
}

automatic_eval_gpus() {
    local max_workers=$1
    local first=() second=() selected=() index free
    while IFS=, read -r index free; do
        index=${index// /}; free=${free// /}
        [[ "$index" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || continue
        (( free < 18000 )) && continue
        first+=("$index")
        (( free >= 36000 )) && second+=("$index")
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    selected=("${first[@]}" "${second[@]}")
    (( ${#selected[@]} > 0 )) || return 1
    (( ${#selected[@]} > max_workers )) && selected=("${selected[@]:0:max_workers}")
    local IFS=,
    printf '%s\n' "${selected[*]}"
}

cell_valid() {
    local summary="$1"
    if [[ "$MODE" == formal ]]; then
        if [[ -n "${GR00T_DYPAC_TASK_IDS_OVERRIDE:-}" ]]; then
            if [[ "${GR00T_DYPAC_REQUIRE_PROTOCOL:-0}" == 1 ]]; then
                "$PY" "$VALIDATOR" "$summary" \
                    --trials "${GR00T_DYPAC_NUM_TRIALS:-10}" \
                    --task-ids "$GR00T_DYPAC_TASK_IDS_OVERRIDE" --offset 10 \
                    --require-paired-action-noise --groot-replan-steps 5 \
                    --replan-steps 5 --policy-backend groot_zmq \
                    --action-noise-generator torch-cpu-normal-v1 >/dev/null 2>&1
            else
                "$PY" "$VALIDATOR" "$summary" \
                    --task-ids "$GR00T_DYPAC_TASK_IDS_OVERRIDE" >/dev/null 2>&1
            fi
        elif [[ "${GR00T_DYPAC_REQUIRE_PROTOCOL:-0}" == 1 ]]; then
            "$PY" "$VALIDATOR" "$summary" \
                --require-paired-action-noise --groot-replan-steps 5 \
                --replan-steps 5 --policy-backend groot_zmq \
                --action-noise-generator torch-cpu-normal-v1 >/dev/null 2>&1
        else
            "$PY" "$VALIDATOR" "$summary" >/dev/null 2>&1
        fi
    else
        "$PY" "$VALIDATOR" "$summary" --trials 1 --tasks 1 --offset 10 >/dev/null 2>&1
    fi
}

wait_for_pi05_ours() {
    local suite
    while true; do
        local ready=1
        for suite in goal spatial object long; do
            if ! "$PY" "$VALIDATOR" \
                "$RUN_ROOT/results/pi05/dypac_vla/$suite/merged_summary.json" \
                >/dev/null 2>&1; then
                ready=0
                break
            fi
        done
        if (( ready == 1 )); then return; fi
        echo "[libero-dypac] waiting for all four pi0.5 Ours cells before GR00T formal evaluation"
        sleep 30
    done
}

if [[ "$MODE" == formal ]]; then wait_for_pi05_ours; fi

for suite in "${SUITES[@]}"; do
    TASK_SUITE=$(suite_name "$suite")
    CHECKPOINT=$ROOT/checkpoints/gr00t/libero-$suite
    PLAN=$ART/$suite/dypac_vla_libero.frozen.plan.json
    HESSIAN=$ART/$suite/hessian_w4.frozen_subset.npz
    PACK=$ART/$suite/identity_pack
    for path in "$CHECKPOINT/config.json" "$PLAN" "$HESSIAN" "$HESSIAN.json" "$PACK"; do
        [[ -e "$path" ]] || { echo "missing formal GR00T DyPAC artifact: $path" >&2; exit 1; }
    done
    if [[ "$MODE" == canary ]]; then
        OUTPUT_ROOT=${GR00T_DYPAC_CANARY_OUTPUT_BASE:-$RUN_ROOT/canary/gr00t/dypac_vla}/$suite
        if [[ -n "${GR00T_DYPAC_CANARY_GPUS:-}" ]]; then
            GPU_LIST=$GR00T_DYPAC_CANARY_GPUS
        else
            GPU_LIST=$(automatic_eval_gpus 1) || { echo "no GPU has the 18 GiB canary margin" >&2; exit 1; }
        fi
        NUM_TRIALS=1
        TASK_OVERRIDE=0
        STARTS=1
        PORT_BASE=${GR00T_DYPAC_PORT_BASE_OVERRIDE:-$((21900 + ${#suite}))}
    else
        OUTPUT_ROOT=${GR00T_DYPAC_FORMAL_OUTPUT_BASE:-$RUN_ROOT/results/gr00t/dypac_vla}/$suite
        if [[ -n "${GR00T_DYPAC_FORMAL_GPUS:-}" ]]; then
            GPU_LIST=$GR00T_DYPAC_FORMAL_GPUS
        else
            GPU_LIST=$(automatic_eval_gpus 10) || { echo "no GPU has the 18 GiB formal-eval margin" >&2; exit 1; }
        fi
        NUM_TRIALS=${GR00T_DYPAC_NUM_TRIALS:-10}
        TASK_OVERRIDE=${GR00T_DYPAC_TASK_IDS_OVERRIDE:-}
        STARTS=2
        PORT_BASE=${GR00T_DYPAC_PORT_BASE_OVERRIDE:-$((22000 + ${#suite}))}
    fi
    if cell_valid "$OUTPUT_ROOT/merged_summary.json"; then
        echo "[libero-dypac] reuse GR00T $MODE $suite"
        continue
    fi
    mkdir -p "$OUTPUT_ROOT/logs"
    (
        export CONDA_ROOT="$CONDA" GROOT_PY="$PY" LIBERO_CONDA_ENV=libero_test
        export LIBERO_PY="$EVAL_PY" EVAL_PY="$EVAL_PY" LIBERO_ROOT LIBERO_CONFIG_PATH=/home1/gyy/.libero
        export QUANTVLA_ROOT="$ROOT" QUANTVLA_CONDA_ENV=groot_test
        export QUANTVLA_CACHE_ROOT="$RUN_ROOT/cache" NUMBA_CACHE_DIR="$RUN_ROOT/numba_cache"
        export PYTHONPATH="$ROOT:$ROOT/code:$ROOT/scripts/tools:$OFFICIAL:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
        export EVAL_SCRIPT="$OFFICIAL/examples/Libero/eval/run_libero_eval.py" EVAL_CWD="$OFFICIAL/examples/Libero/eval"
        export EVAL_EXTRA_ARGS_JSON="${GR00T_DYPAC_EVAL_EXTRA_ARGS_JSON:-[\"--paired-action-noise\",\"--groot-replan-steps\",\"5\"]}"
        export INFERENCE_PY="$PY" INFERENCE_SCRIPT="$ROOT/scripts/inference_service.py" INFERENCE_CWD="$ROOT"
        export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 NO_ALBUMENTATIONS_UPDATE=1 PYTHONNOUSERSITE=1
        export OUTPUT_ROOT TASK_SUITE GPU_LIST PORT_BASE MODEL_PATH="$CHECKPOINT" DATA_CONFIG="$(data_config "$suite")"
        export PACKDIR="$PACK" BENCHMARK_LABEL="dypac_vla_gr00t_${suite}"
        export NUM_TRIALS_PER_TASK="$NUM_TRIALS" GR00T_EVAL_INIT_OFFSET=10 DENOISING_STEPS=10 NUM_STEPS_WAIT=10 REPLAN_STEPS=5
        export GR00T_SHARD_STARTS_PER_GPU="$STARTS" GR00T_SAVE_VIDEOS="${GR00T_DYPAC_SAVE_VIDEOS:-0}" HEADLESS=1 WBITS=4 ABITS=8 METHOD=duquant
        export GR00T_SHARD_STAGGER_MODE=wait_eval_batch GR00T_SHARD_WAIT_EVAL_TIMEOUT_S=600
        if [[ "$MODE" == formal ]]; then
            export GR00T_DYNAMIC_EPISODE_QUEUE=1 GR00T_DYNAMIC_RESUME_QUEUE=1
            export GR00T_DYNAMIC_RESTART_FAILED_WORKERS=1
            export GR00T_DYNAMIC_AUTO_JOIN="${GR00T_DYPAC_DYNAMIC_AUTO_JOIN:-1}"
            export GR00T_DYNAMIC_GPU_POOL="${GR00T_DYPAC_DYNAMIC_GPU_POOL:-auto}" GR00T_DYNAMIC_JOIN_UTIL_MAX=100
            export GR00T_DYNAMIC_MAX_PROCS_PER_GPU=2 GR00T_DYNAMIC_PROCESS_MIB=17000
            export GR00T_DYNAMIC_RESERVE_MIB=2000 GR00T_DYNAMIC_STARTS_PER_GPU_SCAN=1
            export GR00T_DYNAMIC_CLAIM_TIMEOUT_S=1800
            export GR00T_DYNAMIC_JOIN_POLL_S=10 GR00T_DYNAMIC_QUEUE_DRAIN_TIMEOUT_S=86400
        else
            export GR00T_DYNAMIC_EPISODE_QUEUE=0 GR00T_DYNAMIC_AUTO_JOIN=0
        fi
        if [[ -n "$TASK_OVERRIDE" ]]; then export GR00T_TASK_IDS_OVERRIDE="$TASK_OVERRIDE"; else unset GR00T_TASK_IDS_OVERRIDE 2>/dev/null || true; fi
        export GR00T_CONFIG_ID=dypac_vla_libero GR00T_LIBERO_DYPAC_PROTOCOL=1 GR00T_DENOISING_STEPS=10
        export GR00T_DUQUANT_SCOPE=""
        export GR00T_DUQUANT_INCLUDE='.*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*'
        export GR00T_DUQUANT_EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)'
        export GR00T_DUQUANT_PLAN="$PLAN" GR00T_DUQUANT_WBITS_DEFAULT=4 GR00T_DUQUANT_ABITS=8
        export GR00T_DUQUANT_BLOCK=64 GR00T_DUQUANT_BLOCK_OUT=64 GR00T_DUQUANT_PERMUTE=0 GR00T_DUQUANT_ROW_ROT=0
        export GR00T_DUQUANT_LS=0 GR00T_DUQUANT_ACT_PCT=99.9 GR00T_DUQUANT_CALIB_STEPS=32
        export GR00T_DUQUANT_PACKDIR="$PACK" GR00T_DUQUANT_HESSIAN_W4_PATH="$HESSIAN"
        export GR00T_DUQUANT_ACT_DYNAMIC=1 GR00T_DUQUANT_FUSED=1 GR00T_DUQUANT_PRECACHE_WEIGHTS=0
        export GR00T_DUQUANT_DEBUG=0 QUANTVLA_ADAPTER_ONLY=1
        unset GR00T_DUQUANT_ACT_SCALE_PATH GR00T_ATM_ENABLE GR00T_OHB_ENABLE GR00T_ERRORFOLD_PATH GR00T_RUNTIME_SELECTOR_PATH
        "$PY" -u "$OFFICIAL/scripts/run_libero_duquant_benchmark_multi_gpu.py"
    ) >"$OUTPUT_ROOT/logs/run.log" 2>&1
    if [[ "$MODE" == canary ]]; then
        "$PY" "$VALIDATOR" \
            "$OUTPUT_ROOT/merged_summary.json" --trials 1 --tasks 1 --offset 10
    else
        if [[ -n "${GR00T_DYPAC_TASK_IDS_OVERRIDE:-}" ]]; then
            if [[ "${GR00T_DYPAC_REQUIRE_PROTOCOL:-0}" == 1 ]]; then
                "$PY" "$VALIDATOR" "$OUTPUT_ROOT/merged_summary.json" \
                    --trials "$NUM_TRIALS" --task-ids "$TASK_OVERRIDE" --offset 10 \
                    --require-paired-action-noise --groot-replan-steps 5 \
                    --replan-steps 5 --policy-backend groot_zmq \
                    --action-noise-generator torch-cpu-normal-v1
            else
                "$PY" "$VALIDATOR" "$OUTPUT_ROOT/merged_summary.json" \
                    --trials "$NUM_TRIALS" --task-ids "$TASK_OVERRIDE" --offset 10
            fi
        elif [[ "${GR00T_DYPAC_REQUIRE_PROTOCOL:-0}" == 1 ]]; then
            "$PY" "$VALIDATOR" "$OUTPUT_ROOT/merged_summary.json" \
                --require-paired-action-noise --groot-replan-steps 5 \
                --replan-steps 5 --policy-backend groot_zmq \
                --action-noise-generator torch-cpu-normal-v1
        else
            "$PY" "$VALIDATOR" "$OUTPUT_ROOT/merged_summary.json"
        fi
    fi
    echo "[libero-dypac] completed GR00T $MODE $suite"
done
