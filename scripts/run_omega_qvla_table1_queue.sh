#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROOT="$REPO_ROOT/runs/gdsq_extension_preregistered_v1/omega_qvla_table1/queue"
PID_FILE="$ROOT/queue.pid"
PHASE_FILE="$ROOT/phase.txt"
LOG_FILE="$ROOT/queue.log"
WEEK1_PHASE="$REPO_ROOT/runs/gdsq_week1_preregistered_v1/execution/queue/phase.txt"
DOWNLOAD="$REPO_ROOT/checkpoints/omega_qvla/download_manifest.json"
PROTOCOL="$REPO_ROOT/runs/gdsq_extension_preregistered_v1/omega_qvla_table1/protocol_v1.json"
PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
TMUX_SESSION="omega_qvla_table1_queue"

set_phase() {
    mkdir -p "$ROOT"
    printf '%s\n' "$1" >"$PHASE_FILE"
    printf '[%s] phase=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

live_pid() {
    local pid=""
    [[ -f "$PID_FILE" ]] && pid="$(<"$PID_FILE")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

wait_for_file() {
    local path="$1" phase="$2"
    set_phase "$phase"
    while [[ ! -f "$path" ]]; do sleep 60; done
}

run_queue() {
    mkdir -p "$ROOT"
    echo "$$" >"$PID_FILE"
    trap 'set_phase "failed_at_${CURRENT_PHASE:-unknown}"' ERR
    CURRENT_PHASE=waiting_for_official_packs
    wait_for_file "$DOWNLOAD" "$CURRENT_PHASE"

    CURRENT_PHASE=freezing_protocol
    set_phase "$CURRENT_PHASE"
    if [[ -f "$PROTOCOL" ]]; then
        "$PY" "$REPO_ROOT/scripts/tools/omega_qvla_table1_protocol.py" verify
    else
        "$PY" "$REPO_ROOT/scripts/tools/omega_qvla_table1_protocol.py" materialize
    fi

    CURRENT_PHASE=waiting_for_week1_complete
    set_phase "$CURRENT_PHASE"
    while [[ ! -f "$WEEK1_PHASE" || "$(<"$WEEK1_PHASE")" != "complete" ]]; do sleep 60; done

    CURRENT_PHASE=waiting_for_gpu_driver
    set_phase "$CURRENT_PHASE"
    while ! nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; do sleep 60; done

    CURRENT_PHASE=preflight
    set_phase "$CURRENT_PHASE"
    bash "$REPO_ROOT/scripts/run_omega_qvla_table1.sh" preflight

    CURRENT_PHASE=formal_1600
    set_phase "$CURRENT_PHASE"
    bash "$REPO_ROOT/scripts/run_omega_qvla_table1.sh" formal

    CURRENT_PHASE=strict_aggregate
    set_phase "$CURRENT_PHASE"
    "$PY" "$REPO_ROOT/scripts/tools/aggregate_omega_qvla_table1.py" --strict

    CURRENT_PHASE=paper_promotion
    set_phase "$CURRENT_PHASE"
    "$PY" "$REPO_ROOT/scripts/tools/finalize_omega_qvla_table1.py"
    "$PY" "$REPO_ROOT/scripts/tools/render_gdsq_main_table.py"
    "$PY" "$REPO_ROOT/scripts/tools/render_gdsq_claim_status.py"
    "$PY" "$REPO_ROOT/scripts/tools/render_gdsq_component_ablation.py"
    "$PY" "$REPO_ROOT/scripts/tools/render_omega_qvla_libero_table.py"
    make -C "$REPO_ROOT/docs/gdsq_vla_iclr2027" check

    CURRENT_PHASE=complete
    set_phase "$CURRENT_PHASE"
}

start_queue() {
    mkdir -p "$ROOT"
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null || live_pid; then
        echo "Omega-QVLA Table 1 queue already running: pid=$(<"$PID_FILE")"
        return
    fi
    tmux new-session -d -s "$TMUX_SESSION" "$0 run >$LOG_FILE 2>&1"
    sleep 1
    echo "Omega-QVLA Table 1 queue started in tmux session $TMUX_SESSION; log=$LOG_FILE"
}

status_queue() {
    local pid="none" state="stopped" phase="not_started"
    [[ -f "$PID_FILE" ]] && pid="$(<"$PID_FILE")"
    [[ -f "$PHASE_FILE" ]] && phase="$(<"$PHASE_FILE")"
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null || live_pid; then state="running"; fi
    [[ "$phase" == "complete" ]] && state="complete"
    echo "omega_table1_queue pid=$pid state=$state phase=$phase"
    bash "$REPO_ROOT/scripts/run_omega_qvla_table1.sh" status
    [[ -f "$LOG_FILE" ]] && tail -n 12 "$LOG_FILE"
}

cd "$REPO_ROOT"
case "${1:-}" in
    start) start_queue ;;
    run) run_queue ;;
    status) status_queue ;;
    *) echo "usage: $0 start | run | status" >&2; exit 2 ;;
esac
