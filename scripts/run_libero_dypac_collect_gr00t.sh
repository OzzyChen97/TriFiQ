#!/usr/bin/env bash
set -euo pipefail

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=${LIBERO_DYPAC_RUN_ROOT:-$ROOT/runs/libero_dypac_v1}
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
EVAL_PY=/home1/gyy/probe/miniforge3/envs/libero_test/bin/python
CALIB=$RUN_ROOT/calibration/gr00t
CONTROL=$RUN_ROOT/control/gr00t_calibration
IFS=, read -r -a GPUS <<<"${GR00T_DYPAC_CALIB_GPUS:-1,2,3,5}"
SUITES=(goal spatial object long)
BATCH_SIZE=${GR00T_DYPAC_CALIB_BATCH_SIZE:-4}
(( ${#GPUS[@]} == 4 )) || { echo "GR00T_DYPAC_CALIB_GPUS must contain four GPUs" >&2; exit 2; }
(( BATCH_SIZE >= 1 && BATCH_SIZE <= 4 )) || { echo "invalid GR00T_DYPAC_CALIB_BATCH_SIZE" >&2; exit 2; }
mkdir -p "$CALIB/shards" "$CONTROL"
export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3
export NO_ALBUMENTATIONS_UPDATE=1

checkpoint() {
    echo "$ROOT/checkpoints/gr00t/libero-$1"
}

data_config() {
    if [[ "$1" == goal ]]; then
        echo examples.Libero.custom_data_config:LiberoDataConfigMeanStd
    else
        echo examples.Libero.custom_data_config:LiberoDataConfig
    fi
}

server_pids=()
cleanup() {
    local pid
    for pid in "${server_pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${server_pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
    server_pids=()
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

for ((batch_start=0; batch_start<${#SUITES[@]}; batch_start+=BATCH_SIZE)); do
    active_indices=()
    batch_end=$((batch_start + BATCH_SIZE))
    (( batch_end > ${#SUITES[@]} )) && batch_end=${#SUITES[@]}
    for ((index=batch_start; index<batch_end; index++)); do
        suite=${SUITES[$index]}
        if [[ -f "$CALIB/shards/$suite.npz" ]]; then
            echo "[libero-dypac] reuse GR00T calibration shard: $suite"
            continue
        fi
        gpu=${GPUS[$index]}
        port=$((21810 + index))
        (
            while IFS='=' read -r name _; do
                [[ "$name" == GR00T_* || "$name" == QUANTVLA_* ]] && unset "$name"
            done < <(env)
            export CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1
            export PYTHONPATH="$ROOT/code:$ROOT/scripts/tools:$ROOT/code/LIBERO"
            export GR00T_CONFIG_ID=fp16_libero_dypac_calibration
            export GR00T_LIBERO_DYPAC_PROTOCOL=1 GR00T_DENOISING_STEPS=10
            exec "$PY" -u "$ROOT/scripts/inference_service.py" --server \
                --model-path "$(checkpoint "$suite")" \
                --data-config "$(data_config "$suite")" \
                --embodiment-tag new_embodiment --port "$port" --denoising-steps 10
        ) >"$CONTROL/server_${suite}.log" 2>&1 &
        server_pids+=("$!")
        active_indices+=("$index")
    done
    (( ${#active_indices[@]} > 0 )) || continue

    for position in "${!active_indices[@]}"; do
        index=${active_indices[$position]}
        suite=${SUITES[$index]}
        port=$((21810 + index))
        server_pid=${server_pids[$position]}
        ready=0
        for _ in $(seq 1 180); do
            kill -0 "$server_pid" 2>/dev/null || break
            if timeout 10 "$PY" - "$port" >/dev/null 2>&1 <<'PY'
import sys
from gr00t.eval.service import ExternalRobotInferenceClient
value = ExternalRobotInferenceClient(host="127.0.0.1", port=int(sys.argv[1])).call_endpoint("get_runtime_info")
assert value["config_id"] == "fp16_libero_dypac_calibration"
PY
            then ready=1; break; fi
            sleep 2
        done
        (( ready == 1 )) || { echo "GR00T FP16 server did not become ready: $suite" >&2; exit 1; }
    done

    collector_pids=()
    for index in "${active_indices[@]}"; do
        suite=${SUITES[$index]}
        gpu=${GPUS[$index]}
        port=$((21810 + index))
        (
            export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$gpu"
            export LIBERO_CONFIG_PATH=/home1/gyy/.libero NO_ALBUMENTATIONS_UPDATE=1 PYTHONNOUSERSITE=1
            export PYTHONPATH="$ROOT/code:$ROOT/code/LIBERO:$ROOT/scripts/tools"
            "$EVAL_PY" -u "$ROOT/scripts/tools/collect_libero_dypac_gr00t_onpolicy.py" \
                --suite "$suite" --port "$port" --egl-device "$gpu" \
                --out "$CALIB/shards/$suite.npz"
        ) >"$CONTROL/collector_${suite}.log" 2>&1 &
        collector_pids+=("$!")
    done

    failed=0
    for pid in "${collector_pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
    cleanup
    (( failed == 0 )) || exit 1
done

export PYTHONPATH="$ROOT/scripts/tools"
"$PY" "$ROOT/scripts/tools/merge_libero_dypac_gr00t_calibration.py" \
    --shard-dir "$CALIB/shards" --out "$CALIB/calibration_256.npz" \
    --selection-out "$CALIB/selection_144.npz" >"$CONTROL/merge.log" 2>&1
echo "[libero-dypac] GR00T calibration complete"
