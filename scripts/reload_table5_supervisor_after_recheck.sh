#!/usr/bin/env bash
set -uo pipefail

# A stopped Bash process has already parsed its function definitions.  Wait
# for the focused GR00T recheck, retire that old supervisor, then launch the
# current script so subsequent baseline work uses the latest scheduler.

ROOT=/home1/gyy/vla/QuantVLA
RECHECK_PID=${1:?recheck pid required}
OLD_SUPERVISOR_PID=${2:?old supervisor pid required}
CONTROL=$ROOT/runs/table5_followup/control
LOG=$CONTROL/reload_after_recheck.log
SUPERVISOR=$ROOT/scripts/run_table5_followup_supervisor.sh
mkdir -p "$CONTROL"

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"$LOG"
}

expected_command() {
    local pid="$1" needle="$2" command
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    command=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
    [[ "$command" == *"$needle"* ]]
}

log "watching recheck_pid=$RECHECK_PID old_supervisor_pid=$OLD_SUPERVISOR_PID"
while expected_command "$RECHECK_PID" run_libero_dypac_gr00t_goal_long_recheck.sh; do
    sleep 15
done

if expected_command "$OLD_SUPERVISOR_PID" run_table5_followup_supervisor.sh; then
    # A stopped process must first be continued for TERM to be handled.
    kill -CONT "$OLD_SUPERVISOR_PID" 2>/dev/null || true
    kill -TERM "$OLD_SUPERVISOR_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
        [[ -e "/proc/$OLD_SUPERVISOR_PID" ]] || break
        sleep 1
    done
    if expected_command "$OLD_SUPERVISOR_PID" run_table5_followup_supervisor.sh; then
        kill -KILL "$OLD_SUPERVISOR_PID" 2>/dev/null || true
    fi
    log "retired old supervisor pid=$OLD_SUPERVISOR_PID"
fi

setsid bash "$SUPERVISOR" run \
    >>"$CONTROL/unattended.log" 2>&1 < /dev/null &
log "launched current supervisor pid=$!"
