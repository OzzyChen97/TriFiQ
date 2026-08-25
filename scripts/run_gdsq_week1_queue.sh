#!/usr/bin/env bash
set -euo pipefail

# Persistent result-blind week-1 queue.  It never preempts a process and does
# not advance past the pi0.5 selector until its strict 2500-row aggregate is
# complete.  P0 same-budget controls precede P1 ablations.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROOT="$REPO_ROOT/runs/gdsq_week1_preregistered_v1/execution/queue"
PID_FILE="$ROOT/queue.pid"
PHASE_FILE="$ROOT/phase.txt"
LOG_FILE="$ROOT/queue.log"
SELECTOR_PID_FILE="$REPO_ROOT/runs/gdsq_week1_preregistered_v1/pi05_selector_official50/control/orchestrator.pid"
SELECTOR_SUMMARY="$REPO_ROOT/runs/gdsq_week1_preregistered_v1/pi05_selector_official50/aggregate/summary.json"
SELECTOR="$REPO_ROOT/scripts/run_pi05_selector_official.sh"
GROOT="$REPO_ROOT/scripts/run_gr00t_week1.sh"
PI05_W6="$REPO_ROOT/scripts/run_pi05_uniform_w6_official.sh"
PI05_CONTROLS="$REPO_ROOT/scripts/run_pi05_controls_official.sh"
OMEGA_ROBOCASA="$REPO_ROOT/scripts/run_omega_qvla_robocasa365.sh"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
FINALIZER="$REPO_ROOT/scripts/tools/gdsq_finalize_week1.py"
RENDER_TABLE="$REPO_ROOT/scripts/tools/render_gdsq_main_table.py"
RENDER_CLAIMS="$REPO_ROOT/scripts/tools/render_gdsq_claim_status.py"
RENDER_ABLATIONS="$REPO_ROOT/scripts/tools/render_gdsq_component_ablation.py"
JOINT_STATS="$REPO_ROOT/scripts/tools/gdsq_week1_joint_statistics.py"
ABLATION_STATS="$REPO_ROOT/scripts/tools/gdsq_week1_ablation_statistics.py"
REUSE_ATTESTATION="$REPO_ROOT/scripts/tools/gdsq_gr00t_selector_reuse_attestation.py"
EVIDENCE_GATE="$REPO_ROOT/scripts/tools/gdsq_evidence_registry.py"

promote_phase() {
    local phase="$1"
    "$ROBOCASA_PY" "$FINALIZER" "$phase" || return 1
    "$ROBOCASA_PY" "$RENDER_TABLE"
    "$ROBOCASA_PY" "$RENDER_CLAIMS"
    "$ROBOCASA_PY" "$RENDER_ABLATIONS"
}

usage() {
    echo "usage: $0 start | start-recovery | run | run-recovery | status" >&2
}

verify_gr00t_w6_complete() {
    "$ROBOCASA_PY" - "$REPO_ROOT/runs/gdsq_week1_preregistered_v1/execution" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
expected={
    'atomic_seen': ('gr00t_uniform_w6_atomic_seen', 900),
    'composite_seen': ('gr00t_uniform_w6_composite_seen', 800),
    'composite_unseen': ('gr00t_uniform_w6_composite_unseen_14shard_v2', 800),
}
for task_set, (run_name, count) in expected.items():
    path=root/'runs'/run_name/'summary.json'
    value=json.load(open(path)); row=value.get('configs',{}).get('uniform_w6',{})
    assert row.get('episodes') == count, (task_set, row.get('episodes'), count)
    assert not value.get('validation_errors'), (task_set, value.get('validation_errors'))
aggregate=root/'aggregate'/'gr00t_uniform_w6'/'summary.json'
value=json.load(open(aggregate)); row=value.get('configs',{}).get('uniform_w6',{})
assert row.get('episodes') == 2500, row.get('episodes')
assert not value.get('validation_errors'), value.get('validation_errors')
print('GR00T uniform-W6 strict 2500-episode gate passed')
PY
}

wait_for_external_gr00t_w6() {
    set_phase "waiting_for_active_gr00t_uniform_w6"
    while pgrep -f "scripts/run_gr00t_week1.sh run-uniform" >/dev/null 2>&1; do
        sleep 60
    done
    set_phase "closing_gr00t_uniform_w6"
    # Never ask the matrix runner to recreate a completed immutable manifest:
    # later source edits are expected to drift from its frozen provenance.
    # Resume only when the strict result/coverage gate is not yet closed.
    if ! verify_gr00t_w6_complete; then
        "$GROOT" run-uniform-unseen-accelerated
    fi
    verify_gr00t_w6_complete
}

run_recovery_queue() {
    trap 'set_phase "failed_at_${CURRENT_PHASE:-unknown}"' ERR
    CURRENT_PHASE="waiting_for_active_gr00t_uniform_w6"
    wait_for_external_gr00t_w6
    # Result coverage is already closed by the strict gate above.  Promotion
    # is allowed to defer when immutable older manifests report expected
    # source-hash drift; that audit must never idle the independent Omega run.
    if ! promote_phase gr00t-w6; then
        set_phase "gr00t_w6_complete_promotion_deferred"
    fi

    # User-requested priority handoff: benchmark the external W4A4 baseline on
    # RoboCasa365 immediately after GR00T W6, before controls and ablations.
    CURRENT_PHASE="omega_qvla_robocasa365"
    set_phase "$CURRENT_PHASE"
    bash "$OMEGA_ROBOCASA" run-all

    CURRENT_PHASE="gr00t_screen_controls"
    set_phase "$CURRENT_PHASE"
    "$GROOT" screen-controls

    CURRENT_PHASE="gr00t_dev_controls"
    set_phase "$CURRENT_PHASE"
    "$GROOT" run-dev-controls

    CURRENT_PHASE="gr00t_formal_controls"
    set_phase "$CURRENT_PHASE"
    "$GROOT" run-formal-controls
    promote_phase gr00t-p0

    CURRENT_PHASE="gr00t_prepare_ablations"
    set_phase "$CURRENT_PHASE"
    "$GROOT" prepare-ablations

    CURRENT_PHASE="gr00t_core_ablations"
    set_phase "$CURRENT_PHASE"
    "$GROOT" run-ablations
    PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$ABLATION_STATS"
    promote_phase gr00t-ablations

    CURRENT_PHASE="pi05_same_budget_controls_deferred"
    set_phase "$CURRENT_PHASE"
    "$PI05_CONTROLS" run-all
    promote_phase pi05-controls

    CURRENT_PHASE="final_claim_and_paper_gates"
    set_phase "$CURRENT_PHASE"
    PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$JOINT_STATS"
    promote_phase joint
    make test-gdsq

    CURRENT_PHASE="complete"
    set_phase "$CURRENT_PHASE"
}

set_phase() {
    local value="$1"
    mkdir -p "$ROOT"
    printf '%s\n' "$value" >"$PHASE_FILE"
    printf '[%s] phase=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$value"
}

live_pid_file() {
    local file="$1" pid=""
    [[ -f "$file" ]] && pid="$(<"$file")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

verify_selector_complete() {
    "$SELECTOR" aggregate
    "$ROBOCASA_PY" - "$SELECTOR_SUMMARY" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value.get("complete") is True
assert value.get("formal_result") is True
assert value.get("completed_episodes") == value.get("expected_episodes") == 2500
assert value.get("missing_episodes") == value.get("duplicate_episodes") == 0
assert value.get("selector_attestation", {}).get("validated_rows") == 2500
print("selector strict completion gate passed")
PY
}

wait_for_selector() {
    set_phase "waiting_for_pi05_selector"
    while live_pid_file "$SELECTOR_PID_FILE"; do
        sleep 60
    done
    set_phase "verifying_pi05_selector"
    verify_selector_complete
}

run_queue() {
    trap 'set_phase "failed_at_${CURRENT_PHASE:-unknown}"' ERR
    CURRENT_PHASE="startup_evidence_audit"
    set_phase "$CURRENT_PHASE"
    "$ROBOCASA_PY" "$REUSE_ATTESTATION" verify
    "$ROBOCASA_PY" "$EVIDENCE_GATE"

    CURRENT_PHASE="selector"
    wait_for_selector
    promote_phase selector

    # Non-deferrable GR00T P0: W6 plus result-blind functional/dev selection
    # and Primary14 random/action-only controls.
    CURRENT_PHASE="gr00t_prepare_uniform"
    set_phase "$CURRENT_PHASE"
    "$GROOT" prepare-uniform

    CURRENT_PHASE="gr00t_uniform_w6"
    set_phase "$CURRENT_PHASE"
    "$GROOT" run-uniform

    # Functional screening is independent of uniform-W6 rollout outcomes.
    # Run the eight-GPU formal W6 matrix first so the fleet is not idle while
    # a single-model screen is executing.
    CURRENT_PHASE="gr00t_screen_controls"
    set_phase "$CURRENT_PHASE"
    "$GROOT" screen-controls

    CURRENT_PHASE="gr00t_dev_controls"
    set_phase "$CURRENT_PHASE"
    "$GROOT" run-dev-controls

    CURRENT_PHASE="gr00t_formal_controls"
    set_phase "$CURRENT_PHASE"
    "$GROOT" run-formal-controls
    promote_phase gr00t-p0

    # The second non-deferrable all-50 W6 arm.  Its manifest is frozen after
    # the exact preflight and before the first formal row.
    CURRENT_PHASE="pi05_uniform_w6"
    set_phase "$CURRENT_PHASE"
    "$PI05_W6" start
    promote_phase pi05-w6

    # P1 follows every currently executable P0 arm.
    CURRENT_PHASE="gr00t_prepare_ablations"
    set_phase "$CURRENT_PHASE"
    "$GROOT" prepare-ablations

    CURRENT_PHASE="gr00t_core_ablations"
    set_phase "$CURRENT_PHASE"
    "$GROOT" run-ablations
    PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$ABLATION_STATS"
    promote_phase gr00t-ablations

    # Explicitly last per the throughput fallback order in the preregistration.
    CURRENT_PHASE="pi05_same_budget_controls_deferred"
    set_phase "$CURRENT_PHASE"
    "$PI05_CONTROLS" run-all
    promote_phase pi05-controls

    CURRENT_PHASE="final_claim_and_paper_gates"
    set_phase "$CURRENT_PHASE"
    PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$JOINT_STATS"
    promote_phase joint
    make test-gdsq

    CURRENT_PHASE="complete"
    set_phase "$CURRENT_PHASE"
}

start_queue() {
    mkdir -p "$ROOT"
    if live_pid_file "$PID_FILE"; then
        echo "week-1 queue already running: pid=$(<"$PID_FILE")"
        return
    fi
    nohup setsid "$0" run >"$LOG_FILE" 2>&1 </dev/null &
    echo "$!" >"$PID_FILE"
    echo "week-1 queue started: pid=$! log=$LOG_FILE"
}

start_recovery_queue() {
    mkdir -p "$ROOT"
    if live_pid_file "$PID_FILE"; then
        echo "week-1 queue already running: pid=$(<"$PID_FILE")"
        return
    fi
    nohup setsid "$0" run-recovery >"$LOG_FILE" 2>&1 </dev/null &
    echo "$!" >"$PID_FILE"
    echo "week-1 recovery queue started: pid=$! log=$LOG_FILE"
}

status_queue() {
    local pid="" phase="not_started" state="stopped"
    [[ -f "$PID_FILE" ]] && pid="$(<"$PID_FILE")"
    [[ -f "$PHASE_FILE" ]] && phase="$(<"$PHASE_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then state="running"; fi
    echo "week1_queue pid=${pid:-none} state=$state phase=$phase"
    if [[ -f "$LOG_FILE" ]]; then tail -n 12 "$LOG_FILE"; fi
}

cd "$REPO_ROOT"
case "${1:-}" in
    start) start_queue ;;
    start-recovery) start_recovery_queue ;;
    run) run_queue ;;
    run-recovery) run_recovery_queue ;;
    status) status_queue ;;
    *) usage; exit 2 ;;
esac
