#!/usr/bin/env bash
set -euo pipefail

ROOT="/home1/gyy/vla/QuantVLA"
VALIDATOR="$ROOT/scripts/tools/validate_table6_libero_cell.py"
PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
FP16_SMOKE="$ROOT/runs/table6_libero_v1/preflight/gr00t_fp16_4x_gpu1_v1"
OMEGA_SMOKE="$ROOT/runs/table6_libero_v1/preflight/gr00t_omega_2x_gpu3_v1"

while tmux has-session -t table6_smoke_fp16_4x 2>/dev/null \
    || tmux has-session -t table6_smoke_omega_2x 2>/dev/null; do
    sleep 15
done

"$PY" "$VALIDATOR" "$FP16_SMOKE/merged_summary.json" \
    --tasks 4 --trials 1 --offset 10
"$PY" "$VALIDATOR" "$OMEGA_SMOKE/merged_summary.json" \
    --tasks 2 --trials 1 --offset 10

cd "$ROOT"
exec bash scripts/run_table6_libero_omega_subset.sh run
