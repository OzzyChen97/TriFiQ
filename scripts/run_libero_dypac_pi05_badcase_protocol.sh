#!/usr/bin/env bash
set -euo pipefail

# Targeted pi0.5 protocol-alignment audit.  Keep the existing four formal
# cells immutable and rerun only their weakest tasks with the exact frozen
# PCG64 Noise-A stream used by DyPAC calibration.

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=${LIBERO_DYPAC_RUN_ROOT:-$ROOT/runs/libero_dypac_v1}
OUT_BASE=${PI05_DYPAC_BADCASE_OUTPUT_BASE:-$RUN_ROOT/repairs/pi05_badcase_protocol_v2/current_mask}
CONTROL=$OUT_BASE/control
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
VALIDATOR=$ROOT/scripts/tools/validate_table6_libero_cell.py
LONG_GPU=${PI05_DYPAC_BADCASE_LONG_GPU:-7}
GOAL_GPU=${PI05_DYPAC_BADCASE_GOAL_GPU:-4}
TRIALS=${PI05_DYPAC_BADCASE_TRIALS:-10}

mkdir -p "$CONTROL"
exec 9>"$CONTROL/supervisor.lock"
flock -n 9 || { echo "pi0.5 bad-case protocol audit already active" >&2; exit 1; }
printf '%s\n' "$$" >"$CONTROL/supervisor.pid"

valid() {
    local suite=$1 task_ids=$2
    "$PY" "$VALIDATOR" "$OUT_BASE/$suite/merged_summary.json" \
        --trials "$TRIALS" --task-ids "$task_ids" --offset 10 \
        --require-paired-action-noise --replan-steps 5 \
        --policy-backend openpi_ws \
        --action-noise-generator numpy-pcg64-standard-normal-v1 \
        --server-config-id dypac_vla_libero --server-flow-steps 10 \
        >/dev/null 2>&1
}

suite_worker_active() {
    local suite=$1
    pgrep -af 'run_libero_duquant_benchmark_multi_gpu.py.*--run-shard' \
        | grep -F -- "$OUT_BASE/$suite" >/dev/null 2>&1
}

run_suite() {
    local suite=$1 task_ids=$2 port=$3 gpu=$4 attempt=0
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
            "$(date --iso-8601=seconds)" "$suite" "$attempt" "$gpu" "$TRIALS"
        if env \
            PI05_DYPAC_FORMAL_OUTPUT_BASE="$OUT_BASE" \
            LIBERO_DYPAC_FORMAL_GPUS="$gpu" \
            LIBERO_DYPAC_PORT_BASE="$port" \
            PI05_DYPAC_DYNAMIC_AUTO_JOIN=0 \
            PI05_DYPAC_TASK_IDS_OVERRIDE="$task_ids" \
            PI05_DYPAC_NUM_TRIALS="$TRIALS" \
            PI05_DYPAC_SAVE_VIDEOS=0 \
            bash "$ROOT/scripts/run_libero_dypac_pi05_eval.sh" formal "$suite"; then
            valid "$suite" "$task_ids" && break
        fi
        printf '%s suite=%s retry_after=30s\n' \
            "$(date --iso-8601=seconds)" "$suite" >&2
        sleep 30
    done
    printf '%s suite=%s complete\n' "$(date --iso-8601=seconds)" "$suite"
}

# Run the weakest Long task and the only failing Goal task concurrently.
run_suite long 8 26100 "$LONG_GPU" &
long_pid=$!
run_suite goal 2 26200 "$GOAL_GPU" &
goal_pid=$!
wait "$long_pid"
wait "$goal_pid"

"$PY" - "$OUT_BASE" >"$OUT_BASE/paired_badcase_summary.json" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
cells = {}
for suite in ("long", "goal"):
    value = json.loads((root / suite / "merged_summary.json").read_text())
    expected_task = 8 if suite == "long" else 2
    if value.get("paired_action_noise") is not True:
        raise RuntimeError(f"{suite}: paired action noise attestation missing")
    if value.get("action_noise_suite_key") != suite:
        raise RuntimeError(f"{suite}: action-noise suite namespace mismatch")
    if value.get("action_noise_generator") != "numpy-pcg64-standard-normal-v1":
        raise RuntimeError(f"{suite}: pi0.5 noise generator mismatch")
    if int(value.get("executed_replan_steps", -1)) != 5:
        raise RuntimeError(f"{suite}: replan protocol mismatch")
    if int((value.get("server_runtime_protocol") or {}).get("flow_steps", -1)) != 10:
        raise RuntimeError(f"{suite}: flow solver mismatch")
    task = value["task_summaries"]
    if len(task) != 1 or int(task[0]["task_id"]) != expected_task:
        raise RuntimeError(f"{suite}: unexpected task coverage")
    cells[suite] = {
        "episodes": value["total_episodes"],
        "successes": value["total_successes"],
        "tasks": task,
        "summary": str(root / suite / "merged_summary.json"),
    }
payload = {
    "schema_version": 1,
    "kind": "pi05_libero_badcase_protocol_alignment",
    "mask": "frozen_dypac_vla_libero_context_base",
    "paired_action_noise": True,
    "action_noise_generator": "numpy-pcg64-standard-normal-v1",
    "replan_steps": 5,
    "flow_steps": 10,
    "formal_result_overwritten": False,
    "cells": cells,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

printf '%s complete exact_protocol_valid\n' "$(date --iso-8601=seconds)"
