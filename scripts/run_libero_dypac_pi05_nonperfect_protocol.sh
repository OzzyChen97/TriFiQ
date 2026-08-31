#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate every pi0.5 task that was below 10/10 in the first formal
# LIBERO pass.  Selection is frozen from that pass, while every rerun uses
# the corrected DyPAC protocol and held-out initial states 10--19.

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=${LIBERO_DYPAC_RUN_ROOT:-$ROOT/runs/libero_dypac_v1}
SOURCE_BASE=${PI05_DYPAC_NONPERFECT_SOURCE_BASE:-$RUN_ROOT/results/pi05/dypac_vla}
OUT_BASE=${PI05_DYPAC_NONPERFECT_OUTPUT_BASE:-$RUN_ROOT/repairs/pi05_nonperfect_protocol_v2/current_mask}
CONTROL=$OUT_BASE/control
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
VALIDATOR=$ROOT/scripts/tools/validate_table6_libero_cell.py
TRIALS=${PI05_DYPAC_NONPERFECT_TRIALS:-10}

GOAL_TASKS=2
OBJECT_TASKS=2,3,4,5
LONG_TASKS=0,2,8
GOAL_GPU=${PI05_DYPAC_NONPERFECT_GOAL_GPU:-5}
OBJECT_GPU=${PI05_DYPAC_NONPERFECT_OBJECT_GPU:-6}
LONG_GPU=${PI05_DYPAC_NONPERFECT_LONG_GPU:-7}

mkdir -p "$CONTROL"
exec 9>"$CONTROL/supervisor.lock"
flock -n 9 || { echo "pi0.5 non-perfect-task rerun already active" >&2; exit 1; }
printf '%s\n' "$$" >"$CONTROL/supervisor.pid"

"$PY" - "$SOURCE_BASE" "$OUT_BASE" <<'PY'
import hashlib
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2]).resolve()
expected = {"goal": [2], "spatial": [], "object": [2, 3, 4, 5], "long": [0, 2, 8]}
records = {}
for suite, expected_ids in expected.items():
    path = source / suite / "merged_summary.json"
    value = json.loads(path.read_text())
    observed = sorted(
        int(row["task_id"])
        for row in value["task_summaries"]
        if int(row["successes"]) < int(row["episodes"])
    )
    if observed != expected_ids:
        raise RuntimeError(f"{suite}: non-perfect task set drift: {observed} != {expected_ids}")
    records[suite] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "task_ids": observed,
    }
out.mkdir(parents=True, exist_ok=True)
(out / "selection_manifest.json").write_text(json.dumps({
    "schema_version": 1,
    "kind": "pi05_libero_nonperfect_task_selection",
    "criterion": "successes < 10 among 10 formal trials",
    "source_results": records,
}, indent=2, sort_keys=True) + "\n")
PY

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
        printf '%s suite=%s tasks=%s attempt=%d gpu=%s trials=%s\n' \
            "$(date --iso-8601=seconds)" "$suite" "$task_ids" "$attempt" "$gpu" "$TRIALS"
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

run_suite goal "$GOAL_TASKS" 26300 "$GOAL_GPU" &
goal_pid=$!
run_suite object "$OBJECT_TASKS" 26400 "$OBJECT_GPU" &
object_pid=$!
run_suite long "$LONG_TASKS" 26500 "$LONG_GPU" &
long_pid=$!
wait "$goal_pid"
wait "$object_pid"
wait "$long_pid"

"$PY" - "$OUT_BASE" >"$OUT_BASE/nonperfect_retest_summary.json" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
expected = {"goal": [2], "object": [2, 3, 4, 5], "long": [0, 2, 8]}
cells = {}
for suite, task_ids in expected.items():
    path = root / suite / "merged_summary.json"
    value = json.loads(path.read_text())
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
    observed = sorted(int(row["task_id"]) for row in value["task_summaries"])
    if observed != task_ids:
        raise RuntimeError(f"{suite}: task coverage mismatch: {observed} != {task_ids}")
    cells[suite] = {
        "episodes": value["total_episodes"],
        "successes": value["total_successes"],
        "tasks": value["task_summaries"],
        "summary": str(path),
    }
payload = {
    "schema_version": 1,
    "kind": "pi05_libero_nonperfect_protocol_rerun",
    "mask": "frozen_dypac_vla_libero_context_base",
    "selection_manifest": str(root / "selection_manifest.json"),
    "paired_action_noise": True,
    "action_noise_generator": "numpy-pcg64-standard-normal-v1",
    "replan_steps": 5,
    "flow_steps": 10,
    "formal_result_overwritten": False,
    "cells": cells,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

printf '%s complete exact_80_episode_protocol_valid\n' "$(date --iso-8601=seconds)"
