#!/usr/bin/env bash
set -uo pipefail

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=$ROOT/runs/table6_libero_v1
ARTIFACT=$RUN_ROOT/artifacts/gr00t/goal/uniform_w6.a8.npz
RESULT=$RUN_ROOT/results/gr00t/uniform_w6/goal/merged_summary.json
VALIDATOR=$ROOT/scripts/tools/validate_table6_libero_cell.py
PY=/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python
LOG=$RUN_ROOT/control/early_gr00t_goal_uniform.log

artifact_valid() {
    "$PY" - "$ARTIFACT" <<'PY' >/dev/null 2>&1
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
sidecar = json.loads(pathlib.Path(str(path) + ".meta.json").read_text())
assert sidecar["wrapped_layers"] == 116
assert sidecar["calibration_buffer_sha256"]
PY
}

result_valid() {
    "$PY" "$VALIDATOR" "$RESULT" >/dev/null 2>&1
}

mkdir -p "$(dirname "$LOG")"
while ! artifact_valid; do sleep 15; done

attempt=0
while ! result_valid; do
    attempt=$((attempt + 1))
    printf '%s attempt=%d gpus=3,4\n' "$(date --iso-8601=seconds)" "$attempt" >>"$LOG"
    env \
        TABLE6_GPU_LIST=3,4 TABLE6_GPU_UTIL_MAX=100 \
        TABLE6_DYNAMIC_AUTO_JOIN=1 TABLE6_DYNAMIC_GPU_POOL=3,4 \
        TABLE6_DYNAMIC_CLAIM_TIMEOUT_S=1800 TABLE6_PORT_BASE=27000 \
        TABLE6_GR00T_W6_PROCS_PER_GPU=2 \
        bash "$ROOT/scripts/run_table6_libero_quant_subset.sh" \
            run-cell gr00t uniform_w6 goal >>"$LOG" 2>&1 || true
    result_valid || sleep 30
done
printf '%s exact_valid\n' "$(date --iso-8601=seconds)" >>"$LOG"
