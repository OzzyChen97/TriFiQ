#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
TOOLS_PATH="$REPO_ROOT/scripts/tools"
PAPER_ROOT="${PI05_PAPER_ROOT:-$REPO_ROOT/runs/pi05_quantvla_paper}"
RUN_DIR="${PI05_CORRECTED_TABLE1_ROOT:-$PAPER_ROOT/official_pretrain_paired50}"
CONTROL_DIR="$RUN_DIR/control"
WORKER_DIR="$CONTROL_DIR/workers"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_paper_server.sh"
WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
MANIFEST_TOOL="$REPO_ROOT/scripts/tools/pi05_make_corrected_table1_manifest.py"
SCHEDULE_TOOL="$REPO_ROOT/scripts/tools/pi05_make_parallel_schedule.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_pi05_robocasa365.py"
AUDITOR="$REPO_ROOT/scripts/tools/pi05_audit_corrected_table1_completion.py"
REPORTER="$REPO_ROOT/scripts/tools/render_pi05_table1_report.py"
DOC_UPDATER="$REPO_ROOT/scripts/tools/update_pi05_final_doc.py"
FINAL_DOC="$REPO_ROOT/docs/quantvla_v2_pi05_porting_final.md"
GPU_MONITOR="$REPO_ROOT/scripts/tools/monitor_pi05_formal_gpu.py"
MANIFEST="$RUN_DIR/manifest.json"
SCHEDULE="$RUN_DIR/manifest.schedule.json"

SERVERS=(
    "corrected_fp16_g1,fp16,1,18701"
    "corrected_w4_g2,quantvla_w4a8_atmohb,2,18702"
    "corrected_w4_g3,quantvla_w4a8_atmohb,3,18703"
    "corrected_gdsqatm_g4,gdsq_vla_atmohb,4,18704"
    "corrected_gdsq_g5,gdsq_vla,5,18705"
    "corrected_gdsqatm_g6,gdsq_vla_atmohb,6,18706"
    "corrected_gdsq_g1,gdsq_vla,1,18707"
    "corrected_fp16_g4,fp16,4,18708"
    "corrected_fp16_g6,fp16,6,18709"
)

WORKERS=(
    "corr_fp_g1_t0_lo,fp16,corrected_fp16_g1,0,3,0-24,1,18701"
    "corr_fp_g1_t0_hi,fp16,corrected_fp16_g1,0,3,25-49,1,18701"
    "corr_fp_g4_t1_lo,fp16,corrected_fp16_g4,1,3,0-24,4,18708"
    "corr_fp_g4_t1_hi,fp16,corrected_fp16_g4,1,3,25-49,4,18708"
    "corr_fp_g6_t2_lo,fp16,corrected_fp16_g6,2,3,0-24,6,18709"
    "corr_fp_g6_t2_hi,fp16,corrected_fp16_g6,2,3,25-49,6,18709"
    "corr_w4_g2_t0_q0,quantvla_w4a8_atmohb,corrected_w4_g2,0,2,0-12,2,18702"
    "corr_w4_g2_t0_q1,quantvla_w4a8_atmohb,corrected_w4_g2,0,2,13-24,2,18702"
    "corr_w4_g2_t0_q2,quantvla_w4a8_atmohb,corrected_w4_g2,0,2,25-37,2,18702"
    "corr_w4_g2_t0_q3,quantvla_w4a8_atmohb,corrected_w4_g2,0,2,38-49,2,18702"
    "corr_w4_g3_t1_q0,quantvla_w4a8_atmohb,corrected_w4_g3,1,2,0-12,3,18703"
    "corr_w4_g3_t1_q1,quantvla_w4a8_atmohb,corrected_w4_g3,1,2,13-24,3,18703"
    "corr_w4_g3_t1_q2,quantvla_w4a8_atmohb,corrected_w4_g3,1,2,25-37,3,18703"
    "corr_w4_g3_t1_q3,quantvla_w4a8_atmohb,corrected_w4_g3,1,2,38-49,3,18703"
    "corr_gatm_g4_t0_lo,gdsq_vla_atmohb,corrected_gdsqatm_g4,0,2,0-24,4,18704"
    "corr_gatm_g4_t0_hi,gdsq_vla_atmohb,corrected_gdsqatm_g4,0,2,25-49,4,18704"
    "corr_gatm_g6_t1_lo,gdsq_vla_atmohb,corrected_gdsqatm_g6,1,2,0-24,6,18706"
    "corr_gatm_g6_t1_hi,gdsq_vla_atmohb,corrected_gdsqatm_g6,1,2,25-49,6,18706"
    "corr_gdsq_g5_t0_q0,gdsq_vla,corrected_gdsq_g5,0,2,0-12,5,18705"
    "corr_gdsq_g5_t0_q1,gdsq_vla,corrected_gdsq_g5,0,2,13-24,5,18705"
    "corr_gdsq_g5_t0_q2,gdsq_vla,corrected_gdsq_g5,0,2,25-37,5,18705"
    "corr_gdsq_g5_t0_q3,gdsq_vla,corrected_gdsq_g5,0,2,38-49,5,18705"
    "corr_gdsq_g1_t1_lo,gdsq_vla,corrected_gdsq_g1,1,2,0-24,1,18707"
    "corr_gdsq_g1_t1_hi,gdsq_vla,corrected_gdsq_g1,1,2,25-49,1,18707"
)

usage() {
    echo "usage: $0 prepare | run-all | status | stop" >&2
}

runtime_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

server_alive() {
    local instance="$1" port="$2" pid_file="$CONTROL_DIR/$1.pid" pid
    [[ -f "$pid_file" ]] || return 1
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null \
        && curl -fsS --max-time 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1
}

start_servers() {
    mkdir -p "$CONTROL_DIR"
    local spec instance config gpu port
    for spec in "${SERVERS[@]}"; do
        IFS=, read -r instance config gpu port <<<"$spec"
        if server_alive "$instance" "$port"; then
            continue
        fi
        PI05_PAPER_CONTROL_DIR="$CONTROL_DIR" "$SERVER_MANAGER" stop "$instance" || true
        PI05_PAPER_CONTROL_DIR="$CONTROL_DIR" "$SERVER_MANAGER" start "$config" "$gpu" "$port" "$instance"
    done
}

stop_servers() {
    local spec instance _rest
    for spec in "${SERVERS[@]}"; do
        IFS=, read -r instance _rest <<<"$spec"
        PI05_PAPER_CONTROL_DIR="$CONTROL_DIR" "$SERVER_MANAGER" stop "$instance" || true
    done
}

make_manifest() {
    local args=() spec instance config gpu port runtime
    for spec in "${SERVERS[@]}"; do
        IFS=, read -r instance config gpu port <<<"$spec"
        runtime="$CONTROL_DIR/$instance.runtime.json"
        args+=(--server "$instance,$config,$gpu,$port,$runtime")
    done
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$MANIFEST_TOOL" --out "$MANIFEST" --paper-root "$PAPER_ROOT" "${args[@]}"
}

make_schedule() {
    local args=() spec instance config gpu port runtime worker_id shard count seeds _wgpu _wport
    for spec in "${SERVERS[@]}"; do
        IFS=, read -r instance config gpu port <<<"$spec"
        runtime="$CONTROL_DIR/$instance.runtime.json"
        args+=(--server "$instance,$config,$gpu,$port,$runtime")
    done
    for spec in "${WORKERS[@]}"; do
        IFS=, read -r worker_id config instance shard count seeds _wgpu _wport <<<"$spec"
        args+=(--worker "$worker_id,$config,$instance,$shard,$count,$seeds")
    done
    if [[ -f "$SCHEDULE" ]]; then
        "$ROBOCASA_PY" "$SCHEDULE_TOOL" --base-manifest "$MANIFEST" \
            --scheduler "$REPO_ROOT/scripts/run_pi05_corrected_table1.sh" --run-dir "$RUN_DIR" \
            --out "$SCHEDULE" --verify "${args[@]}"
    else
        "$ROBOCASA_PY" "$SCHEDULE_TOOL" --base-manifest "$MANIFEST" \
            --scheduler "$REPO_ROOT/scripts/run_pi05_corrected_table1.sh" --run-dir "$RUN_DIR" \
            --out "$SCHEDULE" "${args[@]}"
    fi
}

prepare() {
    "$ROBOCASA_PY" "$REPO_ROOT/scripts/tools/pi05_audit_paper_quantvla.py" --root "$PAPER_ROOT"
    start_servers
    make_manifest
    make_schedule
}

start_workers() {
    mkdir -p "$WORKER_DIR"
    local spec worker_id config instance shard count seeds gpu port runtime hash
    for spec in "${WORKERS[@]}"; do
        IFS=, read -r worker_id config instance shard count seeds gpu port <<<"$spec"
        if [[ -f "$WORKER_DIR/$worker_id.pid" ]]; then
            local old_pid
            old_pid="$(<"$WORKER_DIR/$worker_id.pid")"
            if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
                echo "refusing to duplicate live corrected worker: $worker_id pid=$old_pid" >&2
                return 1
            fi
        fi
        runtime="$CONTROL_DIR/$instance.runtime.json"
        hash="$(runtime_hash "$runtime")"
        nohup setsid "$WORKER" "$config" "$port" "$gpu" "$shard" "$count" "$seeds" \
            "$worker_id" "$hash" "$RUN_DIR" >"$WORKER_DIR/$worker_id.log" 2>&1 </dev/null &
        echo "$!" >"$WORKER_DIR/$worker_id.pid"
    done
}

stop_workers() {
    local spec worker_id _rest pid_file pid
    for spec in "${WORKERS[@]}"; do
        IFS=, read -r worker_id _rest <<<"$spec"
        pid_file="$WORKER_DIR/$worker_id.pid"
        [[ -f "$pid_file" ]] || continue
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    done
}

wait_workers() {
    local spec worker_id _rest pid failed=0
    for spec in "${WORKERS[@]}"; do
        IFS=, read -r worker_id _rest <<<"$spec"
        pid="$(<"$WORKER_DIR/$worker_id.pid")"
        wait "$pid" || failed=1
    done
    return "$failed"
}

start_monitor() {
    if [[ -f "$CONTROL_DIR/gpu_monitor.pid" ]]; then
        local old_pid
        old_pid="$(<"$CONTROL_DIR/gpu_monitor.pid")"
        if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
            return
        fi
    fi
    nohup setsid "$ROBOCASA_PY" "$GPU_MONITOR" --run-dir "$RUN_DIR" --interval 30 \
        >"$CONTROL_DIR/gpu_monitor.log" 2>&1 </dev/null &
    echo "$!" >"$CONTROL_DIR/gpu_monitor.pid"
}

stop_monitor() {
    local pid_file="$CONTROL_DIR/gpu_monitor.pid" pid
    [[ -f "$pid_file" ]] || return
    pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
}

complete() {
    PYTHONPATH="$TOOLS_PATH:$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$AGGREGATOR" --run-dir "$RUN_DIR" \
        --out-dir "$RUN_DIR/aggregate_incomplete" --allow-incomplete >/dev/null
    "$ROBOCASA_PY" - "$RUN_DIR/aggregate_incomplete/summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
ok = all(row["completed_episodes"] == 2500 and row["missing_episodes"] == 0 for row in summary["configs"].values())
raise SystemExit(0 if ok else 1)
PY
}

finalize() {
    PYTHONPATH="$TOOLS_PATH:$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$AGGREGATOR" --run-dir "$RUN_DIR" --out-dir "$RUN_DIR/aggregate"
    PYTHONPATH="$TOOLS_PATH:$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$AUDITOR" --run-dir "$RUN_DIR" --schedule "$SCHEDULE" \
        --out "$RUN_DIR/aggregate/completion_audit.json"
    "$ROBOCASA_PY" "$REPORTER" --summary "$RUN_DIR/aggregate/summary.json" \
        --audit "$RUN_DIR/aggregate/completion_audit.json" --manifest "$MANIFEST" \
        --out "$RUN_DIR/aggregate/table1_final_report.md"
    "$ROBOCASA_PY" "$DOC_UPDATER" --doc "$FINAL_DOC" \
        --summary "$RUN_DIR/aggregate/summary.json" \
        --audit "$RUN_DIR/aggregate/completion_audit.json" \
        --report "$RUN_DIR/aggregate/table1_final_report.md"
    touch "$CONTROL_DIR/complete"
}

run_all() {
    mkdir -p "$CONTROL_DIR"
    exec 9>"$CONTROL_DIR/run_all.lock"
    if ! flock -n 9; then
        echo "another corrected Table-1 runner already holds $CONTROL_DIR/run_all.lock" >&2
        exit 1
    fi
    trap 'stop_workers; stop_monitor; stop_servers' EXIT INT TERM HUP
    prepare
    if complete; then
        finalize
        stop_monitor
        stop_servers
        trap - EXIT INT TERM HUP
        echo "corrected paper-faithful Table-1 matrix already complete"
        return
    fi
    local attempt
    for attempt in 1 2 3; do
        echo "[corrected Table1] attempt=$attempt"
        start_servers
        start_workers
        start_monitor
        wait_workers || true
        stop_workers
        stop_monitor
        if complete; then
            finalize
            stop_servers
            trap - EXIT INT TERM HUP
            echo "corrected paper-faithful Table-1 matrix complete"
            return
        fi
        echo "[corrected Table1] incomplete after attempt=$attempt; committed-key resume" >&2
    done
    echo "corrected Table-1 matrix did not complete after three attempts" >&2
    exit 1
}

status() {
    if [[ -f "$MANIFEST" ]]; then
        PYTHONPATH="$TOOLS_PATH:$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
            "$ROBOCASA_PY" "$AGGREGATOR" --run-dir "$RUN_DIR" \
            --out-dir "$RUN_DIR/aggregate_incomplete" --allow-incomplete
    else
        echo "corrected manifest pending: $MANIFEST"
    fi
    PI05_PAPER_CONTROL_DIR="$CONTROL_DIR" "$SERVER_MANAGER" status
}

case "${1:-}" in
    prepare) prepare ;;
    run-all) run_all ;;
    status) status ;;
    stop) stop_workers; stop_monitor; stop_servers ;;
    *) usage; exit 2 ;;
esac
