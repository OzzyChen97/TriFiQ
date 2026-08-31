#!/usr/bin/env bash
set -euo pipefail

# Full 4-suite GR00T LIBERO rerun under the exact frozen DyPAC protocol.
# Existing formal cells remain immutable.  Each suite resumes an exact
# 100-episode queue and is accepted only with paired Noise A and five-step
# action-chunk execution attested in the merged summary.

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=${LIBERO_DYPAC_RUN_ROOT:-$ROOT/runs/libero_dypac_v1}
OUT_BASE=${GR00T_DYPAC_FULL_V2_OUTPUT_BASE:-$RUN_ROOT/results_v2/gr00t/dypac_vla}
CONTROL=$OUT_BASE/control
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
VALIDATOR=$ROOT/scripts/tools/validate_table6_libero_cell.py
TRIALS=${GR00T_DYPAC_FULL_V2_TRIALS:-10}

mkdir -p "$CONTROL"
exec 9>"$CONTROL/supervisor.lock"
flock -n 9 || { echo "GR00T full protocol-v2 evaluation already active" >&2; exit 1; }
printf '%s\n' "$$" >"$CONTROL/supervisor.pid"

valid() {
    local suite=$1
    "$PY" "$VALIDATOR" "$OUT_BASE/$suite/merged_summary.json" \
        --trials "$TRIALS" --tasks 10 --offset 10 \
        --require-paired-action-noise --groot-replan-steps 5 \
        --replan-steps 5 --policy-backend groot_zmq \
        --action-noise-generator torch-cpu-normal-v1 >/dev/null 2>&1
}

suite_worker_active() {
    local suite=$1
    pgrep -af 'run_libero_duquant_benchmark_multi_gpu.py.*--run-shard' \
        | grep -F -- "$OUT_BASE/$suite" >/dev/null 2>&1
}

run_suite() {
    local suite=$1 port=$2 attempt=0
    while ! valid "$suite"; do
        if suite_worker_active "$suite"; then
            printf '%s suite=%s attach_existing_worker=1\n' \
                "$(date --iso-8601=seconds)" "$suite"
            while suite_worker_active "$suite" && ! valid "$suite"; do
                sleep 20
            done
            valid "$suite" && break
        fi
        attempt=$((attempt + 1))
        printf '%s suite=%s attempt=%d trials=%s\n' \
            "$(date --iso-8601=seconds)" "$suite" "$attempt" "$TRIALS"
        if env \
            GR00T_DYPAC_FORMAL_OUTPUT_BASE="$OUT_BASE" \
            GR00T_DYPAC_PORT_BASE_OVERRIDE="$port" \
            GR00T_DYPAC_DYNAMIC_AUTO_JOIN=1 \
            GR00T_DYPAC_NUM_TRIALS="$TRIALS" \
            GR00T_DYPAC_REQUIRE_PROTOCOL=1 \
            GR00T_DYPAC_SAVE_VIDEOS=0 \
            bash "$ROOT/scripts/run_libero_dypac_gr00t_eval.sh" formal "$suite"; then
            valid "$suite" && break
        fi
        printf '%s suite=%s retry_after=30s\n' \
            "$(date --iso-8601=seconds)" "$suite" >&2
        sleep 30
    done
    printf '%s suite=%s complete\n' "$(date --iso-8601=seconds)" "$suite"
}

# Start with the two suites whose old one-step rollout exposed bad cases.
run_suite long 27100
run_suite goal 27200
run_suite spatial 27300
run_suite object 27400

"$PY" - "$OUT_BASE" >"$OUT_BASE/full_protocol_summary.json" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
cells = {}
total_episodes = 0
total_successes = 0
for suite in ("goal", "spatial", "object", "long"):
    value = json.loads((root / suite / "merged_summary.json").read_text())
    if value.get("paired_action_noise") is not True:
        raise RuntimeError(f"{suite}: paired action noise missing")
    if value.get("action_noise_suite_key") != suite:
        raise RuntimeError(f"{suite}: action-noise suite namespace mismatch")
    if value.get("action_noise_generator") != "torch-cpu-normal-v1":
        raise RuntimeError(f"{suite}: GR00T noise generator mismatch")
    if int(value.get("executed_replan_steps", -1)) != 5:
        raise RuntimeError(f"{suite}: replan protocol mismatch")
    episodes = int(value["total_episodes"])
    successes = int(value["total_successes"])
    total_episodes += episodes
    total_successes += successes
    cells[suite] = {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "summary": str(root / suite / "merged_summary.json"),
    }
if total_episodes != 400:
    raise RuntimeError(f"full LIBERO coverage mismatch: {total_episodes} != 400")
payload = {
    "schema_version": 1,
    "kind": "gr00t_libero_dypac_full_protocol_v2",
    "paired_action_noise": True,
    "action_noise_generator": "torch-cpu-normal-v1",
    "replan_steps": 5,
    "flow_steps": 10,
    "formal_result_overwritten": False,
    "total_episodes": total_episodes,
    "total_successes": total_successes,
    "macro_success_rate": sum(row["success_rate"] for row in cells.values()) / 4,
    "cells": cells,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

printf '%s complete exact_400_episode_protocol_valid\n' "$(date --iso-8601=seconds)"
