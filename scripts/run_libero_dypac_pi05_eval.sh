#!/usr/bin/env bash
set -euo pipefail

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=${LIBERO_DYPAC_RUN_ROOT:-$ROOT/runs/libero_dypac_v1}
OFFICIAL=$ROOT/external/Omega-QVLA
LIBERO_ROOT=$ROOT/code/LIBERO
OPENPI_ROOT=$ROOT/code/pi05/openpi
OPENPI_CLIENT=$OPENPI_ROOT/packages/openpi-client/src
CONDA=/home1/gyy/probe/miniforge3
DRIVER_PY=$CONDA/envs/groot_test/bin/python
EVAL_PY=$CONDA/envs/libero_test/bin/python
OPENPI_PY=$CONDA/envs/openpi/bin/python
CHECKPOINT=$ROOT/code/pi05/checkpoints/pi05_libero_pytorch
CHECKPOINT_SHA=0f8c489e37b01c72251c45f2e73595894f3933fc6297f4f1cf95fc8737db4c74
PLAN=$RUN_ROOT/artifacts/pi05/dypac_vla_libero.frozen.plan.json
HESSIAN=$RUN_ROOT/artifacts/pi05/hessian_w4.frozen_subset.npz
PACK=$RUN_ROOT/artifacts/pi05/identity_pack
BUFFER=$RUN_ROOT/calibration/pi05/calibration_256.npz
PARITY=$RUN_ROOT/artifacts/pi05/triton_parity.json
VALIDATOR=$ROOT/scripts/tools/validate_table6_libero_cell.py
MODE=${1:-formal}
SUITE_FILTER=${2:-all}

[[ "$MODE" == canary || "$MODE" == formal ]] || { echo "usage: $0 canary|formal [goal|spatial|object|long|all]" >&2; exit 2; }
for path in "$PLAN" "$HESSIAN" "$HESSIAN.json" "$PACK/manifest.json" "$BUFFER" "$PARITY"; do
    [[ -e "$path" ]] || { echo "missing formal DyPAC artifact: $path" >&2; exit 1; }
done

read -r WRAPPED < <("$OPENPI_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["quantized_w4_layers"])' "$PLAN")
BUFFER_SHA=$(sha256sum "$BUFFER" | cut -d' ' -f1)
PACK_MANIFEST_SHA=$(sha256sum "$PACK/manifest.json" | cut -d' ' -f1)
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

automatic_eval_gpus() {
    local max_workers=$1
    local first=() second=() third=() fourth=() selected=() index free
    while IFS=, read -r index free; do
        index=${index// /}; free=${free// /}
        [[ "$index" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || continue
        (( free < 12000 )) && continue
        first+=("$index")
        # A served pi0.5 policy occupies about 7--9 GiB.  Keep a margin for
        # initialization while filling every card round-robin before replicas.
        (( free >= 22000 )) && second+=("$index")
        (( free >= 31500 )) && third+=("$index")
        (( free >= 41000 )) && fourth+=("$index")
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    selected=("${first[@]}" "${second[@]}" "${third[@]}" "${fourth[@]}")
    (( ${#selected[@]} > 0 )) || return 1
    (( ${#selected[@]} > max_workers )) && selected=("${selected[@]:0:max_workers}")
    local IFS=,
    printf '%s\n' "${selected[*]}"
}

cell_valid() {
    local summary="$1"
    if [[ "$MODE" == formal ]]; then
        if [[ -n "${PI05_DYPAC_TASK_IDS_OVERRIDE:-}" ]]; then
            "$DRIVER_PY" "$VALIDATOR" "$summary" \
                --trials "${PI05_DYPAC_NUM_TRIALS:-10}" \
                --task-ids "$PI05_DYPAC_TASK_IDS_OVERRIDE" --offset 10 \
                --require-paired-action-noise --replan-steps 5 \
                --policy-backend openpi_ws \
                --action-noise-generator numpy-pcg64-standard-normal-v1 \
                --server-config-id dypac_vla_libero --server-flow-steps 10 \
                >/dev/null 2>&1
        elif [[ "${PI05_DYPAC_REQUIRE_PAIRED:-0}" == 1 ]]; then
            "$DRIVER_PY" "$VALIDATOR" "$summary" \
                --require-paired-action-noise --replan-steps 5 \
                --policy-backend openpi_ws \
                --action-noise-generator numpy-pcg64-standard-normal-v1 \
                --server-config-id dypac_vla_libero --server-flow-steps 10 \
                >/dev/null 2>&1
        else
            "$DRIVER_PY" "$VALIDATOR" "$summary" >/dev/null 2>&1
        fi
    else
        "$DRIVER_PY" "$VALIDATOR" "$summary" --trials 1 --tasks 1 --offset 10 >/dev/null 2>&1
    fi
}

for suite in "${SUITES[@]}"; do
    TASK_SUITE=$(suite_name "$suite")
    if [[ "$MODE" == canary ]]; then
        OUTPUT_ROOT=$RUN_ROOT/canary/pi05/dypac_vla/$suite
        if [[ -n "${LIBERO_DYPAC_CANARY_GPUS:-}" ]]; then
            GPU_LIST=$LIBERO_DYPAC_CANARY_GPUS
        else
            GPU_LIST=$(automatic_eval_gpus 1) || { echo "no GPU has the 18 GiB canary margin" >&2; exit 1; }
        fi
        NUM_TRIALS=1
        TASK_OVERRIDE=0
        STARTS=1
        PORT_BASE=${LIBERO_DYPAC_PORT_BASE:-$((21600 + ${#suite}))}
    else
        OUTPUT_ROOT=${PI05_DYPAC_FORMAL_OUTPUT_BASE:-$RUN_ROOT/results/pi05/dypac_vla}/$suite
        if [[ -n "${LIBERO_DYPAC_FORMAL_GPUS:-}" ]]; then
            GPU_LIST=$LIBERO_DYPAC_FORMAL_GPUS
        else
            GPU_LIST=$(automatic_eval_gpus 10) || { echo "no GPU has the 18 GiB formal-eval margin" >&2; exit 1; }
        fi
        NUM_TRIALS=${PI05_DYPAC_NUM_TRIALS:-10}
        TASK_OVERRIDE=${PI05_DYPAC_TASK_IDS_OVERRIDE:-}
        STARTS=2
        PORT_BASE=${LIBERO_DYPAC_PORT_BASE:-$((21700 + ${#suite}))}
    fi
    if cell_valid "$OUTPUT_ROOT/merged_summary.json"; then
        echo "[libero-dypac] reuse $MODE $suite"
        continue
    fi
    mkdir -p "$OUTPUT_ROOT/logs"
    (
        export CONDA_ROOT="$CONDA" GROOT_PY="$DRIVER_PY" LIBERO_CONDA_ENV=libero_test
        export LIBERO_PY="$EVAL_PY" EVAL_PY="$EVAL_PY" LIBERO_ROOT LIBERO_CONFIG_PATH=/home1/gyy/.libero
        export QUANTVLA_ROOT="$ROOT" QUANTVLA_CONDA_ENV=groot_test
        export QUANTVLA_CACHE_ROOT="$RUN_ROOT/cache" NUMBA_CACHE_DIR="$RUN_ROOT/numba_cache"
        export PYTHONPATH="$ROOT:$OFFICIAL:$OPENPI_CLIENT:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
        export EVAL_SCRIPT="$OFFICIAL/examples/Libero/eval/run_libero_eval.py" EVAL_CWD="$OFFICIAL/examples/Libero/eval"
        export INFERENCE_PY="$OPENPI_PY" INFERENCE_SCRIPT="$ROOT/scripts/table6_pi05_inference_service.py" INFERENCE_CWD="$OPENPI_ROOT"
        export INFERENCE_EXTRA_ARGS_JSON='["--device","cuda"]'
        export EVAL_EXTRA_ARGS_JSON='["--policy-backend","openpi_ws","--max-steps-profile","openpi","--replan-steps","5","--paired-action-noise","--action-noise-stream","A"]'
        export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 NO_ALBUMENTATIONS_UPDATE=1 PYTHONNOUSERSITE=1
        export OUTPUT_ROOT TASK_SUITE GPU_LIST PORT_BASE MODEL_PATH="$CHECKPOINT" DATA_CONFIG=pi05_libero
        export PACKDIR="$PACK" BENCHMARK_LABEL="dypac_vla_pi05_${suite}"
        export NUM_TRIALS_PER_TASK="$NUM_TRIALS" GR00T_EVAL_INIT_OFFSET=10 DENOISING_STEPS=10 NUM_STEPS_WAIT=10 REPLAN_STEPS=5
        export GR00T_SHARD_STARTS_PER_GPU="$STARTS" GR00T_SAVE_VIDEOS="${PI05_DYPAC_SAVE_VIDEOS:-0}" HEADLESS=1 WBITS=4 ABITS=8 METHOD=duquant
        export GR00T_SHARD_STAGGER_MODE=wait_eval_batch GR00T_SHARD_WAIT_EVAL_TIMEOUT_S=600
        if [[ "$MODE" == formal ]]; then
            # Persistent workers pull exact (task, state) episodes from one
            # atomic queue.  This removes task-level long tails and lets a card
            # that becomes free after launch hot-join without duplicating work.
            export GR00T_DYNAMIC_EPISODE_QUEUE=1 GR00T_DYNAMIC_RESUME_QUEUE=1
            export GR00T_DYNAMIC_RESTART_FAILED_WORKERS=1 GR00T_DYNAMIC_AUTO_JOIN="${PI05_DYPAC_DYNAMIC_AUTO_JOIN:-1}"
            export GR00T_DYNAMIC_GPU_POOL=auto GR00T_DYNAMIC_JOIN_UTIL_MAX=100
            export GR00T_DYNAMIC_MAX_PROCS_PER_GPU=4 GR00T_DYNAMIC_PROCESS_MIB=9500
            export GR00T_DYNAMIC_RESERVE_MIB=1500 GR00T_DYNAMIC_STARTS_PER_GPU_SCAN=1
            export GR00T_DYNAMIC_CLAIM_TIMEOUT_S=1800
            export GR00T_DYNAMIC_JOIN_POLL_S=10 GR00T_DYNAMIC_QUEUE_DRAIN_TIMEOUT_S=86400
        else
            export GR00T_DYNAMIC_EPISODE_QUEUE=0 GR00T_DYNAMIC_AUTO_JOIN=0
        fi
        if [[ -n "$TASK_OVERRIDE" ]]; then export GR00T_TASK_IDS_OVERRIDE="$TASK_OVERRIDE"; else unset GR00T_TASK_IDS_OVERRIDE 2>/dev/null || true; fi
        export OPENPI_MODEL_DTYPE=float16 OPENPI_CONFIG_ID=dypac_vla_libero OPENPI_CHECKPOINT_SHA256="$CHECKPOINT_SHA"
        export OPENPI_LIBERO_DYPAC_PROTOCOL=1 OPENPI_FORMAL_MODE=1 OPENPI_FORMAL_FLOW_STEPS=10 OPENPI_FORMAL_EXPECT_WRAPPED="$WRAPPED"
        export OPENPI_DUQUANT_PLAN="$PLAN" OPENPI_DUQUANT_PLAN_STRICT=1 OPENPI_DUQUANT_WBITS_DEFAULT=4 OPENPI_DUQUANT_ABITS=8
        export OPENPI_DUQUANT_BLOCK=64 OPENPI_DUQUANT_BLOCK_OUT=64 OPENPI_DUQUANT_EXPECT_BLOCK=64 OPENPI_DUQUANT_EXPECT_WRAPPED="$WRAPPED"
        export OPENPI_DUQUANT_LS=0 OPENPI_DUQUANT_PERMUTE=0 OPENPI_DUQUANT_ROW_ROT=0 OPENPI_DUQUANT_DENOISING_STEPS=10
        export OPENPI_DUQUANT_PACKDIR="$PACK" OPENPI_DUQUANT_PACK_MANIFEST_SHA256="$PACK_MANIFEST_SHA"
        export OPENPI_DUQUANT_HESSIAN_W4_PATH="$HESSIAN" OPENPI_DUQUANT_ACT_DYNAMIC=1 OPENPI_DUQUANT_REQUIRE_ACT_SCALE=0
        export OPENPI_DUQUANT_CALIB_BUFFER_SHA256="$BUFFER_SHA" OPENPI_DUQUANT_STRICT_ARTIFACTS=0
        export OPENPI_DUQUANT_PRECACHE_WEIGHTS=0 OPENPI_DUQUANT_TRITON=1 OPENPI_DUQUANT_QUIET=1 QUANTVLA_ADAPTER_ONLY=1
        unset OPENPI_DUQUANT_ACT_SCALE_PATH OPENPI_ATM_ENABLE OPENPI_OHB_ENABLE OPENPI_ERRORFOLD_PATH
        "$DRIVER_PY" -u "$OFFICIAL/scripts/run_libero_duquant_benchmark_multi_gpu.py"
    ) >"$OUTPUT_ROOT/logs/run.log" 2>&1
    if [[ "$MODE" == formal ]]; then
        if [[ -n "${PI05_DYPAC_TASK_IDS_OVERRIDE:-}" ]]; then
            "$DRIVER_PY" "$VALIDATOR" "$OUTPUT_ROOT/merged_summary.json" \
                --trials "$NUM_TRIALS" --task-ids "$TASK_OVERRIDE" --offset 10 \
                --require-paired-action-noise --replan-steps 5 \
                --policy-backend openpi_ws \
                --action-noise-generator numpy-pcg64-standard-normal-v1 \
                --server-config-id dypac_vla_libero --server-flow-steps 10
        else
            "$DRIVER_PY" "$VALIDATOR" "$OUTPUT_ROOT/merged_summary.json" \
                --require-paired-action-noise --replan-steps 5 \
                --policy-backend openpi_ws \
                --action-noise-generator numpy-pcg64-standard-normal-v1 \
                --server-config-id dypac_vla_libero --server-flow-steps 10
        fi
    else
        "$DRIVER_PY" "$VALIDATOR" "$OUTPUT_ROOT/merged_summary.json" --trials 1 --tasks 1 --offset 10
    fi
    echo "[libero-dypac] completed $MODE $suite"
done
