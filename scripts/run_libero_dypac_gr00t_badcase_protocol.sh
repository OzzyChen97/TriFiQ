#!/usr/bin/env bash
set -euo pipefail

# Targeted protocol-alignment audit for the three GR00T LIBERO bad cases.
# This never overwrites the frozen formal cell.  It reuses the frozen mask and
# suite-specific Hessian artifacts, while making the rollout match the frozen
# DyPAC calibration protocol: five executed actions per replan and deterministic
# request-local flow-matching noise.

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=${LIBERO_DYPAC_RUN_ROOT:-$ROOT/runs/libero_dypac_v1}
OUT_BASE=${GR00T_DYPAC_BADCASE_OUTPUT_BASE:-$RUN_ROOT/repairs/gr00t_badcase_protocol_v2/current_mask}
CONTROL=$OUT_BASE/control
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
VALIDATOR=$ROOT/scripts/tools/validate_table6_libero_cell.py
GPU=${GR00T_DYPAC_BADCASE_GPU:-5}
TRIALS=${GR00T_DYPAC_BADCASE_TRIALS:-10}
SAVE_VIDEOS=${GR00T_DYPAC_BADCASE_SAVE_VIDEOS:-0}
EVAL_ARGS='["--paired-action-noise","--groot-replan-steps","5","--action-noise-stream","A"]'

mkdir -p "$CONTROL"
exec 9>"$CONTROL/supervisor.lock"
flock -n 9 || { echo "GR00T bad-case protocol audit already active" >&2; exit 1; }
printf '%s\n' "$$" >"$CONTROL/supervisor.pid"

valid() {
    local suite=$1 task_ids=$2
    "$PY" "$VALIDATOR" "$OUT_BASE/$suite/merged_summary.json" \
        --trials "$TRIALS" --task-ids "$task_ids" --offset 10 \
        --require-paired-action-noise --groot-replan-steps 5 \
        >/dev/null 2>&1
}

suite_worker_active() {
    local suite=$1
    # A launcher can be interrupted after spawning its persistent shard.  The
    # shard deliberately survives so completed episodes are not lost.  Detect
    # that case and wait for the orphaned worker before resuming the queue,
    # instead of launching a second process against the same queue and port.
    pgrep -af 'run_libero_duquant_benchmark_multi_gpu.py.*--run-shard' \
        | grep -F -- "$OUT_BASE/$suite" >/dev/null 2>&1
}

run_suite() {
    local suite=$1 task_ids=$2 port=$3 attempt=0
    while ! valid "$suite" "$task_ids"; do
        if suite_worker_active "$suite"; then
            printf '%s suite=%s attach_existing_worker=1\n' \
                "$(date --iso-8601=seconds)" "$suite"
            while suite_worker_active "$suite" && ! valid "$suite" "$task_ids"; do
                sleep 15
            done
            valid "$suite" "$task_ids" && break
        fi
        attempt=$((attempt + 1))
        printf '%s suite=%s attempt=%d gpu=%s trials=%s\n' \
            "$(date --iso-8601=seconds)" "$suite" "$attempt" "$GPU" "$TRIALS"
        if env \
            GR00T_DYPAC_FORMAL_OUTPUT_BASE="$OUT_BASE" \
            GR00T_DYPAC_FORMAL_GPUS="$GPU" \
            GR00T_DYPAC_PORT_BASE_OVERRIDE="$port" \
            GR00T_DYPAC_DYNAMIC_AUTO_JOIN=0 \
            GR00T_DYPAC_TASK_IDS_OVERRIDE="$task_ids" \
            GR00T_DYPAC_NUM_TRIALS="$TRIALS" \
            GR00T_DYPAC_EVAL_EXTRA_ARGS_JSON="$EVAL_ARGS" \
            GR00T_DYPAC_SAVE_VIDEOS="$SAVE_VIDEOS" \
            bash "$ROOT/scripts/run_libero_dypac_gr00t_eval.sh" formal "$suite"; then
            valid "$suite" "$task_ids" && break
        fi
        printf '%s suite=%s retry_after=30s\n' \
            "$(date --iso-8601=seconds)" "$suite" >&2
        sleep 30
    done
    printf '%s suite=%s complete\n' "$(date --iso-8601=seconds)" "$suite"
}

# Long-8 is the stable 2/10 failure and therefore has diagnostic priority.
run_suite long 8 25000
run_suite goal 0,6 25100

"$PY" - "$OUT_BASE" >"$OUT_BASE/paired_badcase_summary.json" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
cells = {}
for suite in ("long", "goal"):
    value = json.loads((root / suite / "merged_summary.json").read_text())
    if value.get("paired_action_noise") is not True:
        raise RuntimeError(f"{suite}: paired action noise attestation missing")
    if int(value.get("groot_replan_steps", -1)) != 5:
        raise RuntimeError(f"{suite}: GR00T replan protocol mismatch")
    if value.get("action_noise_suite_key") != suite:
        raise RuntimeError(f"{suite}: DyPAC action-noise suite namespace mismatch")
    cells[suite] = {
        "episodes": value["total_episodes"],
        "successes": value["total_successes"],
        "tasks": value["task_summaries"],
        "summary": str(root / suite / "merged_summary.json"),
    }
payload = {
    "schema_version": 1,
    "kind": "gr00t_libero_badcase_protocol_alignment",
    "mask": "frozen_dypac_vla_libero_context_base",
    "paired_action_noise": True,
    "groot_replan_steps": 5,
    "action_noise_suite_keys": {suite: suite for suite in cells},
    "formal_result_overwritten": False,
    "cells": cells,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

printf '%s complete exact_protocol_valid\n' "$(date --iso-8601=seconds)"
