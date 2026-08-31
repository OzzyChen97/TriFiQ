#!/usr/bin/env bash
set -uo pipefail

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=${LIBERO_DYPAC_RUN_ROOT:-$ROOT/runs/libero_dypac_v1}
OUT_BASE=${GR00T_DYPAC_RECHECK_BASE:-$RUN_ROOT/rechecks/gr00t_dypac_tasks_0_6_8_v1}
CONTROL=$OUT_BASE/control
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
VALIDATOR=$ROOT/scripts/tools/validate_table6_libero_cell.py
SUPERVISOR_PID_FILE=$ROOT/runs/table5_followup/control/supervisor.pid
mkdir -p "$CONTROL"
exec 9>"$CONTROL/recheck.lock"
flock -n 9 || { echo "Goal/Long recheck already active" >&2; exit 1; }

supervisor_pid=
if [[ -f "$SUPERVISOR_PID_FILE" ]]; then
    supervisor_pid=$(<"$SUPERVISOR_PID_FILE")
fi
resume_supervisor() {
    if [[ "$supervisor_pid" =~ ^[0-9]+$ ]]; then
        kill -CONT "$supervisor_pid" 2>/dev/null || true
    fi
}
trap resume_supervisor EXIT INT TERM

valid() {
    local suite=$1 task_ids=$2
    "$PY" "$VALIDATOR" "$OUT_BASE/$suite/merged_summary.json" \
        --task-ids "$task_ids" >/dev/null 2>&1
}

run_suite() {
    local suite=$1 task_ids=$2 gpu_list=$3 port=$4 attempt=0
    while ! valid "$suite" "$task_ids"; do
        attempt=$((attempt + 1))
        printf '%s suite=%s attempt=%d gpus=%s\n' \
            "$(date --iso-8601=seconds)" "$suite" "$attempt" "$gpu_list"
        if env \
            GR00T_DYPAC_FORMAL_OUTPUT_BASE="$OUT_BASE" \
            GR00T_DYPAC_FORMAL_GPUS="$gpu_list" \
            GR00T_DYPAC_PORT_BASE_OVERRIDE="$port" \
            GR00T_DYPAC_DYNAMIC_AUTO_JOIN=0 \
            GR00T_DYPAC_TASK_IDS_OVERRIDE="$task_ids" \
            bash "$ROOT/scripts/run_libero_dypac_gr00t_eval.sh" formal "$suite"; then
            valid "$suite" "$task_ids" && break
        fi
        printf '%s suite=%s retry_after=30s\n' \
            "$(date --iso-8601=seconds)" "$suite" >&2
        sleep 30
    done
    printf '%s suite=%s complete\n' "$(date --iso-8601=seconds)" "$suite"
}

run_suite goal 0,6 1,2,3,4,5,2 23000 >"$CONTROL/goal.log" 2>&1 &
goal_pid=$!
run_suite long 8 4,5,6,6,7 24000 >"$CONTROL/long.log" 2>&1 &
long_pid=$!
failed=0
wait "$goal_pid" || failed=1
wait "$long_pid" || failed=1
(( failed == 0 )) || exit 1
valid goal 0,6 && valid long 8
printf '%s complete goal_and_long_exact_valid\n' "$(date --iso-8601=seconds)"
