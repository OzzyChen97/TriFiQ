#!/usr/bin/env bash
set -uo pipefail

# Independent watchdog for the Table-5 completion supervisor.  The guardian
# has its own lock and session, so it can restart the supervisor after an
# unexpected exit without creating duplicate evaluators.  Exact-result locks
# and queues remain authoritative inside the supervisor.

ROOT=/home1/gyy/vla/QuantVLA
CONTROL=$ROOT/runs/table5_followup/control
SUPERVISOR=$ROOT/scripts/run_table5_followup_supervisor.sh
AUDIT_PY=/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python
POLL_SECONDS=${TABLE5_GUARDIAN_POLL_SECONDS:-30}
STALE_SECONDS=${TABLE5_GUARDIAN_STALE_SECONDS:-900}
STATUS=$CONTROL/status.json
GUARDIAN_STATUS=$CONTROL/guardian_status.json
GUARDIAN_PROGRESS=$CONTROL/guardian_progress.log
UNATTENDED_LOG=$CONTROL/unattended.log

mkdir -p "$CONTROL"
cd "$ROOT" || exit 1

log() {
    printf '%s guardian %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$GUARDIAN_PROGRESS"
}

supervisor_pids() {
    pgrep -f '[r]un_table5_followup_supervisor.sh run' 2>/dev/null || true
}

relevant_child_count() {
    pgrep -f \
        '[r]un_libero_dypac_(pi05_eval|gr00t_(resume|artifacts|pipeline|eval))\.sh|[p]robe_libero_dypac_gr00t_outputimpact\.py|[s]core_libero_dypac_gr00t_masks\.py|[r]un_table6_libero_(unattended|quant_subset)\.sh|[r]un_libero_duquant_benchmark_multi_gpu\.py' \
        2>/dev/null | wc -l
}

completion_valid() {
    [[ -f "$CONTROL/complete" && -f "$STATUS" ]] || return 1
    "$AUDIT_PY" - "$STATUS" >/dev/null 2>&1 <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text())
cells = list(value.get("ours", {}).values()) + list(value.get("baselines", {}).values())
assert value.get("phase") == "complete"
assert len(cells) == 6
assert all(len(group) == 4 and all(group.values()) for group in cells)
PY
}

write_heartbeat() {
    local supervisors children status_age complete_flag
    supervisors=$(supervisor_pids | paste -sd, -)
    children=$(relevant_child_count)
    if [[ -f "$STATUS" ]]; then
        status_age=$(( $(date +%s) - $(stat -c %Y "$STATUS") ))
    else
        status_age=-1
    fi
    if completion_valid; then complete_flag=1; else complete_flag=0; fi
    "$AUDIT_PY" - "$GUARDIAN_STATUS" "$$" "$supervisors" "$children" \
        "$status_age" "$complete_flag" "$1" <<'PY'
import datetime
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
supervisors = [int(item) for item in sys.argv[3].split(",") if item]
payload = {
    "schema_version": 1,
    "guardian_pid": int(sys.argv[2]),
    "supervisor_pids": supervisors,
    "relevant_child_processes": int(sys.argv[4]),
    "supervisor_status_age_seconds": int(sys.argv[5]),
    "completion_valid": bool(int(sys.argv[6])),
    "restart_count": int(sys.argv[7]),
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
temporary = pathlib.Path(str(path) + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
}

run_guardian() {
    exec 9>"$CONTROL/guardian.lock"
    flock -n 9 || { echo "Table-5 follow-up guardian is already active" >&2; return 1; }
    printf '%s\n' "$$" >"$CONTROL/guardian.pid"
    local restart_count=0 supervisors children status_age
    log "started pid=$$ poll_seconds=$POLL_SECONDS"

    while true; do
        if completion_valid; then
            write_heartbeat "$restart_count"
            log "complete exact_valid_cells=24"
            return 0
        fi

        supervisors=$(supervisor_pids)
        if [[ -z "$supervisors" ]]; then
            restart_count=$((restart_count + 1))
            log "starting_supervisor restart_count=$restart_count"
            setsid bash "$SUPERVISOR" run >>"$UNATTENDED_LOG" 2>&1 < /dev/null &
            sleep 2
        else
            children=$(relevant_child_count)
            if [[ -f "$STATUS" ]]; then
                status_age=$(( $(date +%s) - $(stat -c %Y "$STATUS") ))
            else
                status_age=$STALE_SECONDS
            fi
            # The supervisor updates status every retry cycle.  A stale status
            # with no relevant child means the scheduler itself is wedged, not
            # merely waiting on a long-running evaluation stage.
            if (( status_age >= STALE_SECONDS && children == 0 )); then
                log "restarting_stale_supervisor pids=$(paste -sd, <<<"$supervisors") status_age_seconds=$status_age"
                while read -r pid; do
                    [[ "$pid" =~ ^[0-9]+$ ]] && kill -TERM "$pid" 2>/dev/null || true
                done <<<"$supervisors"
                sleep 5
            fi
        fi

        write_heartbeat "$restart_count"
        sleep "$POLL_SECONDS"
    done
}

case "${1:-run}" in
    run) run_guardian ;;
    status)
        [[ -f "$GUARDIAN_STATUS" ]] && cat "$GUARDIAN_STATUS" || echo "guardian status has not been initialized"
        ;;
    *) echo "usage: $0 [run|status]" >&2; exit 2 ;;
esac
