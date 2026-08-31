#!/usr/bin/env bash
set -euo pipefail

# Prepare and evaluate the four formal pi0.5 paper-method cells while the
# Omega-QVLA subset drains.  The formal runner owns a per-cell lock, so the
# normal unattended pass can safely wait for and reuse these results.

ROOT="/home1/gyy/vla/QuantVLA"
RUN_ROOT="${TABLE6_RUN_ROOT:-$ROOT/runs/table6_libero_v1}"
ARTIFACTS="$RUN_ROOT/artifacts"
CONTROL="$RUN_ROOT/control"
LOGS="$CONTROL/pi05_ours_overlap_logs"
LIBERO_ROOT="$ROOT/code/LIBERO"
OFFICIAL="$ROOT/external/Omega-QVLA"
CONDA="/home1/gyy/probe/miniforge3"
LIBERO_PY="$CONDA/envs/libero_test/bin/python"
OPENPI_PY="$CONDA/envs/openpi/bin/python"
AUDIT_PY="$CONDA/envs/robocasa365/bin/python"
PI_CHECKPOINT_SHA="0f8c489e37b01c72251c45f2e73595894f3933fc6297f4f1cf95fc8737db4c74"
PACK="$ARTIFACTS/pi05/pack_block64"
FORMAL_GPUS="${TABLE6_PI05_OVERLAP_GPUS:-1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7}"
CALIBRATION_GPUS=(1 2 3 4)
SUITES=(goal spatial object long)

mkdir -p "$LOGS" "$CONTROL"
cd "$ROOT"

exec 9>"$CONTROL/pi05_ours_overlap.lock"
flock -n 9 || { echo "pi0.5 Ours overlap runner already active" >&2; exit 1; }

phase() {
    printf '%s phase=%s detail=%s\n' "$(date --iso-8601=seconds)" "$1" "${2:-}" \
        | tee -a "$CONTROL/pi05_ours_overlap.log"
}

calibration_valid() {
    "$AUDIT_PY" - "$1" <<'PY' >/dev/null 2>&1
import hashlib
import json
import numpy as np
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
sidecar = json.loads(pathlib.Path(str(path) + ".json").read_text())
assert sidecar["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
assert sidecar["rows"] == 256 and sidecar["tasks"] == 10
assert sidecar["initial_state_indices"] == list(range(5))
assert sidecar["held_out_initial_state_indices"] == list(range(10, 20))
assert sidecar["overlap_with_held_out"] is False
assert sidecar["policy_queries"] == 0
assert sidecar["test_rollout_feedback_used"] is False
with np.load(path, allow_pickle=False) as archive:
    assert archive["states"].shape == (256, 8)
    assert archive["action_noises"].shape == (256, 50, 32)
PY
}

selector_valid() {
    "$OPENPI_PY" - "$1" "$2" "$PI_CHECKPOINT_SHA" <<'PY' >/dev/null 2>&1
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text())
meta = value["meta"]
assert meta["suite"] == sys.argv[2]
assert meta["checkpoint_sha256"] == sys.argv[3]
assert meta["candidate_layers"] == len(value["layers"]) == 180
assert meta["uses_task_success"] is False
assert meta["uses_test_rollout_feedback"] is False
assert meta["uses_cka"] is False and meta["uses_cs"] is False
assert meta["held_out_initial_state_indices"] == list(range(10, 20))
PY
}

pack_valid() {
    "$OPENPI_PY" - "$PACK/manifest.json" "$PI_CHECKPOINT_SHA" <<'PY' >/dev/null 2>&1
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text())
assert value["complete"] is True
assert value["checkpoint_sha256"] == sys.argv[2]
assert value["wrapped_layer_count"] == 180
assert len(value["files"]) == 180
assert all((path.parent / row["file"]).is_file() for row in value["files"])
PY
}

collect_calibration() {
    phase calibration "four result-blind suites"
    local pids=() index suite gpu out
    for index in "${!SUITES[@]}"; do
        suite="${SUITES[$index]}"
        gpu="${CALIBRATION_GPUS[$index]}"
        out="$ARTIFACTS/calibration/$suite.npz"
        if calibration_valid "$out"; then
            echo "reuse calibration $suite"
            continue
        fi
        if [[ -e "$out" || -e "$out.json" ]]; then
            echo "invalid calibration artifact retained: $out" >&2
            return 1
        fi
        mkdir -p "$(dirname "$out")"
        (
            export PYTHONPATH="$LIBERO_ROOT:$OFFICIAL${PYTHONPATH:+:$PYTHONPATH}"
            export LIBERO_CONFIG_PATH=/home1/gyy/.libero
            export NO_ALBUMENTATIONS_UPDATE=1 PYTHONNOUSERSITE=1
            "$LIBERO_PY" "$ROOT/scripts/tools/collect_table6_libero_calibration.py" \
                --suite "$suite" --egl-device "$gpu" --out "$out"
        ) >"$LOGS/calibration_${suite}.log" 2>&1 &
        pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do wait "$pid"; done
    for suite in "${SUITES[@]}"; do
        calibration_valid "$ARTIFACTS/calibration/$suite.npz"
    done
}

select_masks() {
    phase select_masks "four result-blind pi0.5 Ours plans"
    local pids=() index suite gpu out
    for index in "${!SUITES[@]}"; do
        suite="${SUITES[$index]}"
        gpu="${CALIBRATION_GPUS[$index]}"
        out="$ARTIFACTS/pi05/$suite/gdsq_vla_selector.plan.json"
        if selector_valid "$out" "$suite"; then
            echo "reuse selector $suite"
            continue
        fi
        if [[ -e "$out" ]]; then
            echo "invalid selector artifact retained: $out" >&2
            return 1
        fi
        mkdir -p "$(dirname "$out")"
        (
            export CUDA_VISIBLE_DEVICES="$gpu" OPENPI_MODEL_DTYPE=float16
            export TORCHDYNAMO_DISABLE=1 PYTHONNOUSERSITE=1
            "$OPENPI_PY" "$ROOT/scripts/tools/select_table6_pi05_libero_plan.py" \
                --buffer "$ARTIFACTS/calibration/$suite.npz" \
                --suite "$suite" --out "$out"
        ) >"$LOGS/selector_${suite}.log" 2>&1 &
        pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do wait "$pid"; done
    for suite in "${SUITES[@]}"; do
        selector_valid "$ARTIFACTS/pi05/$suite/gdsq_vla_selector.plan.json" "$suite"
    done
}

run_canary() {
    phase canary "pi0.5 Ours goal task0 seed10 on GPU1"
    env TABLE6_GPU_LIST=1 TABLE6_PORT_BASE=19700 TABLE6_NUM_TRIALS=1 \
        TABLE6_TASK_IDS_OVERRIDE=0 TABLE6_SMOKE_TASKS=1 \
        TABLE6_SMOKE_TAG=overlap_canary \
        bash "$ROOT/scripts/run_table6_libero_quant_subset.sh" \
            smoke-cell pi05 gdsq_vla_selector goal \
        >"$LOGS/canary.log" 2>&1
}

run_formal() {
    local suite
    for suite in "${SUITES[@]}"; do
        phase formal "$suite GPUs=$FORMAL_GPUS"
        env TABLE6_GPU_LIST="$FORMAL_GPUS" TABLE6_PORT_BASE=19800 \
            TABLE6_PI05_SELECTOR_PROCS_PER_GPU=1 \
            bash "$ROOT/scripts/run_table6_libero_quant_subset.sh" \
                run-cell pi05 gdsq_vla_selector "$suite" \
            >"$LOGS/formal_${suite}.log" 2>&1
    done
}

pack_valid
collect_calibration
select_masks
run_canary
run_formal
phase complete "four formal pi0.5 Ours cells"
