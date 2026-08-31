#!/usr/bin/env bash
set -euo pipefail

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=${LIBERO_DYPAC_RUN_ROOT:-$ROOT/runs/libero_dypac_v1}
OPENPI_PY=/home1/gyy/probe/miniforge3/envs/openpi/bin/python
LIBERO_PY=/home1/gyy/probe/miniforge3/envs/libero_test/bin/python
CHECKPOINT=$ROOT/code/pi05/checkpoints/pi05_libero_pytorch
CHECKPOINT_SHA=0f8c489e37b01c72251c45f2e73595894f3933fc6297f4f1cf95fc8737db4c74
LOG_DIR=$RUN_ROOT/control/calibration_logs
SHARD_DIR=$RUN_ROOT/calibration/pi05/shards
SUITES=(goal spatial object long)
GPUS=(1 2 3 4)
PORTS=(21101 21102 21103 21104)
SERVER_PIDS=()
COLLECTOR_PIDS=()

mkdir -p "$LOG_DIR" "$SHARD_DIR"

cleanup() {
    local pid
    for pid in "${SERVER_PIDS[@]:-}"; do
        kill -- "-$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

for index in "${!SUITES[@]}"; do
    suite=${SUITES[$index]}
    gpu=${GPUS[$index]}
    port=${PORTS[$index]}
    runtime=$RUN_ROOT/control/fp16_calibration_${suite}.runtime.json
    (
        cd "$ROOT/code/pi05/openpi"
        export CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 TORCHDYNAMO_DISABLE=1
        export OPENPI_MODEL_DTYPE=float16 OPENPI_CONFIG_ID=fp16_libero_dypac_calibration
        export OPENPI_CHECKPOINT_SHA256="$CHECKPOINT_SHA"
        export OPENPI_LIBERO_DYPAC_PROTOCOL=1 OPENPI_RUNTIME_INFO_PATH="$runtime"
        unset OPENPI_DUQUANT_PLAN OPENPI_DUQUANT_PACKDIR OPENPI_DUQUANT_HESSIAN_W4_PATH
        unset OPENPI_DUQUANT_ACT_SCALE_PATH OPENPI_OMEGA_QVLA OPENPI_ATM_ENABLE OPENPI_OHB_ENABLE
        exec setsid "$OPENPI_PY" "$ROOT/scripts/table6_pi05_inference_service.py" \
            --model-path "$CHECKPOINT" --data-config pi05_libero --port "$port" \
            --denoising-steps 10 --server
    ) >"$LOG_DIR/server_${suite}.log" 2>&1 &
    SERVER_PIDS+=("$!")
done

for index in "${!SUITES[@]}"; do
    suite=${SUITES[$index]}
    port=${PORTS[$index]}
    for _ in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then break; fi
        sleep 2
    done
    curl -fsS "http://127.0.0.1:$port/healthz" >/dev/null
    echo "[libero-dypac] FP16 $suite server ready on $port"
done

for index in "${!SUITES[@]}"; do
    suite=${SUITES[$index]}
    gpu=${GPUS[$index]}
    port=${PORTS[$index]}
    (
        export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$gpu"
        export LIBERO_CONFIG_PATH=/home1/gyy/.libero NO_ALBUMENTATIONS_UPDATE=1 PYTHONNOUSERSITE=1
        export PYTHONPATH="$ROOT/code/LIBERO:$ROOT/external/Omega-QVLA:$ROOT/code/pi05/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
        exec "$LIBERO_PY" "$ROOT/scripts/tools/collect_libero_dypac_onpolicy.py" \
            --suite "$suite" --host 127.0.0.1 --port "$port" --egl-device "$gpu" \
            --out "$SHARD_DIR/$suite.npz"
    ) >"$LOG_DIR/collector_${suite}.log" 2>&1 &
    COLLECTOR_PIDS+=("$!")
done

failed=0
for pid in "${COLLECTOR_PIDS[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
if (( failed )); then
    echo "one or more LIBERO DyPAC collectors failed; see $LOG_DIR" >&2
    exit 1
fi

"$OPENPI_PY" "$ROOT/scripts/tools/merge_libero_dypac_calibration.py" \
    --shard-dir "$SHARD_DIR" \
    --out "$RUN_ROOT/calibration/pi05/calibration_256.npz" \
    --selection-out "$RUN_ROOT/calibration/pi05/selection_144.npz" \
    >"$LOG_DIR/merge.log" 2>&1
echo "[libero-dypac] merged calibration complete"
