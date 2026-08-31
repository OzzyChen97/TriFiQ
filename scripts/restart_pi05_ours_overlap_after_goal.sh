#!/usr/bin/env bash
set -euo pipefail

ROOT="/home1/gyy/vla/QuantVLA"
RESULT="$ROOT/runs/table6_libero_v1/results/pi05/gdsq_vla_selector/goal/merged_summary.json"
VALIDATOR="$ROOT/scripts/tools/validate_table6_libero_cell.py"
PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"

while ! "$PY" "$VALIDATOR" "$RESULT" >/dev/null 2>&1; do
    sleep 2
done

# The old shell captured the one-worker GPU list before the concurrency
# change.  Restart only after Goal is promoted, then reuse it and launch the
# remaining suites with the new three-workers-per-card list.
if tmux has-session -t table6_pi05_ours_overlap 2>/dev/null; then
    tmux kill-session -t table6_pi05_ours_overlap
fi
sleep 2
tmux new-session -d -s table6_pi05_ours_overlap \
    "cd $ROOT && bash scripts/run_table6_pi05_ours_overlap.sh >> runs/table6_libero_v1/control/pi05_ours_overlap.session.log 2>&1"
