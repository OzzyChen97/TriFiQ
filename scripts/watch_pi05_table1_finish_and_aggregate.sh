#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
RUN_ROOT="$REPO_ROOT/runs/full_context_v2/pi05_table1"
MANIFEST="$RUN_ROOT/manifest.schedule_finish_sprint_v4.json"
WORKER_CONTROL="$RUN_ROOT/control/workers"
LOG="$RUN_ROOT/control/finish_sprint_aggregate.log"

while true; do
    alive=0
    while IFS= read -r worker_id; do
        pid_file="$WORKER_CONTROL/$worker_id.pid"
        [[ -f "$pid_file" ]] || continue
        pid="$(<"$pid_file")"
        [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || continue
        cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
        [[ "$cmdline" == *"run_pi05_table1_finish_worker.sh"*"$MANIFEST"* ]] \
            && alive=$((alive+1))
    done < <(jq -r '.assignments[].worker_id' "$MANIFEST")
    [[ "$alive" == 0 ]] && break
    sleep 30
done

cd "$REPO_ROOT"
scripts/run_full_context_pi05_table1.sh aggregate >"$LOG" 2>&1
