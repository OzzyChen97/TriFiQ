#!/usr/bin/env bash
set -euo pipefail

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=${LIBERO_DYPAC_RUN_ROOT:-$ROOT/runs/libero_dypac_v1}
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
ART=$RUN_ROOT/artifacts/gr00t
CALIB=$RUN_ROOT/calibration/gr00t
CONTROL=$RUN_ROOT/control/gr00t_artifacts
INVENTORY=$ART/candidate_inventory.json
SUITES=(goal spatial object long)
MODE=${1:-all}
[[ "$MODE" == all || "$MODE" == prepare || "$MODE" == capture || "$MODE" == hessian ]] || {
    echo "usage: $0 all|prepare|capture|hessian" >&2; exit 2;
}
mkdir -p "$ART" "$CONTROL"
export PYTHONPATH="$ROOT/code:$ROOT/scripts/tools"
export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3
export NO_ALBUMENTATIONS_UPDATE=1

checkpoint() { echo "$ROOT/checkpoints/gr00t/libero-$1"; }

prepare() {
    if [[ ! -f "$INVENTORY" ]]; then
        "$PY" "$ROOT/scripts/tools/prepare_libero_dypac_gr00t.py" \
            --out-dir "$ART" >"$CONTROL/prepare.log" 2>&1
    fi
    for suite in "${SUITES[@]}"; do
        if [[ ! -f "$ART/$suite/identity_pack/shard_00_of_01.json" ]]; then
            "$PY" -u "$ROOT/scripts/tools/build_libero_dypac_gr00t_identity_pack.py" \
                --suite "$suite" --checkpoint "$(checkpoint "$suite")" \
                --inventory "$INVENTORY" --out-dir "$ART/$suite/identity_pack" \
                >"$CONTROL/identity_pack_${suite}.log" 2>&1
        fi
    done
}

capture() {
    IFS=, read -r -a capture_gpus <<<"${GR00T_DYPAC_CAPTURE_GPUS:-1,2,3,5}"
    local batch_size=${GR00T_DYPAC_CAPTURE_BATCH_SIZE:-4}
    (( ${#capture_gpus[@]} == 4 )) || { echo "GR00T_DYPAC_CAPTURE_GPUS must contain four GPUs" >&2; return 2; }
    (( batch_size >= 1 && batch_size <= 4 )) || { echo "invalid GR00T_DYPAC_CAPTURE_BATCH_SIZE" >&2; return 2; }
    local pids=() index suite gpu failed=0 batch_start batch_end pid
    for ((batch_start=0; batch_start<${#SUITES[@]}; batch_start+=batch_size)); do
        pids=()
        batch_end=$((batch_start + batch_size))
        (( batch_end > ${#SUITES[@]} )) && batch_end=${#SUITES[@]}
        for ((index=batch_start; index<batch_end; index++)); do
            suite=${SUITES[$index]}; gpu=${capture_gpus[$index]}
            if [[ -f "$ART/$suite/fp16_hessian_capture.npz" ]]; then
                echo "[libero-dypac] reuse GR00T Hessian capture: $suite"
                continue
            fi
            (
                export CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1
                "$PY" -u "$ROOT/scripts/tools/capture_libero_dypac_gr00t_hessian.py" \
                    --suite "$suite" --checkpoint "$(checkpoint "$suite")" \
                    --inventory "$INVENTORY" --buffer "$CALIB/shards/$suite.npz" \
                    --out "$ART/$suite/fp16_hessian_capture.npz" --batch-size 4
            ) >"$CONTROL/capture_${suite}.log" 2>&1 &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
        (( failed == 0 )) || return 1
    done
}

hessian() {
    IFS=, read -r -a hessian_gpus <<<"${GR00T_DYPAC_HESSIAN_GPUS:-1,2,3,5,6,7,5,6}"
    local batch_size=${GR00T_DYPAC_HESSIAN_BATCH_SIZE:-8}
    (( ${#hessian_gpus[@]} == 8 )) || { echo "GR00T_DYPAC_HESSIAN_GPUS must contain eight entries" >&2; return 2; }
    (( batch_size >= 1 && batch_size <= 8 )) || { echo "invalid GR00T_DYPAC_HESSIAN_BATCH_SIZE" >&2; return 2; }
    local pids=() suite shard index gpu failed=0 batch_start batch_end pid suite_index
    for suite in "${SUITES[@]}"; do mkdir -p "$ART/$suite/hessian_layers"; done
    for ((batch_start=0; batch_start<8; batch_start+=batch_size)); do
        pids=()
        batch_end=$((batch_start + batch_size))
        (( batch_end > 8 )) && batch_end=8
        for ((index=batch_start; index<batch_end; index++)); do
            suite_index=$((index / 2)); shard=$((index % 2))
            suite=${SUITES[$suite_index]}; gpu=${hessian_gpus[$index]}
            (
                export CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1
                "$PY" -u "$ROOT/scripts/tools/build_libero_dypac_hessian_shard.py" \
                    --suite "$suite" --checkpoint "$(checkpoint "$suite")" \
                    --capture "$ART/$suite/fp16_hessian_capture.npz" \
                    --inventory "$INVENTORY" --out-dir "$ART/$suite/hessian_layers" \
                    --shard-index "$shard" --shard-count 2
            ) >"$CONTROL/hessian_${suite}_${shard}.log" 2>&1 &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
        (( failed == 0 )) || return 1
    done
    for suite in "${SUITES[@]}"; do
        if [[ ! -f "$ART/$suite/hessian_w4.npz" ]]; then
            "$PY" "$ROOT/scripts/tools/merge_libero_dypac_hessian.py" \
                --model gr00t --suite "$suite" --layer-dir "$ART/$suite/hessian_layers" \
                --inventory "$INVENTORY" --capture "$ART/$suite/fp16_hessian_capture.npz" \
                --out "$ART/$suite/hessian_w4.npz" >"$CONTROL/hessian_merge_${suite}.log" 2>&1
        fi
    done
}

case "$MODE" in
    prepare) prepare ;;
    capture) [[ -f "$INVENTORY" ]] || prepare; capture ;;
    hessian) hessian ;;
    all) prepare; capture; hessian ;;
esac
echo "[libero-dypac] GR00T base artifacts complete: $MODE"
