#!/usr/bin/env bash
set -euo pipefail

# Seven-replica, config-sequential GR00T-aligned Table-1 scheduler. Every config runs
# the frozen 50 tasks x 50 explicit seeds with identical paired noise.  The
# wave layout only changes resource allocation: all available experiment GPUs
# serve one config at a time, reducing the critical path versus one GPU/config.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
FINAL_ROOT="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned"
RUN_DIR="$FINAL_ROOT/official_target_paired50"
CONTROL_DIR="$RUN_DIR/control"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
SEEDED_WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
MANIFEST_TOOL="$REPO_ROOT/scripts/tools/pi05_make_formal_manifest.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_pi05_robocasa365.py"
GPU_MONITOR="$REPO_ROOT/scripts/tools/monitor_pi05_formal_gpu.py"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
ALIGNMENT_AUDIT="$REPO_ROOT/scripts/tools/audit_pi05_against_gr00t_final.py"

export PI05_REQUIRE_FAITHFUL_FINAL=1
export PI05_CONTROL_DIR="$CONTROL_DIR"
export PI05_PACK_DIR="$REPO_ROOT/runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
export PI05_PACK_MANIFEST="$PI05_PACK_DIR/manifest.json"
export PI05_CALIBRATION_BUFFER="$FINAL_ROOT/calibration/pi05_robocasa365_seed0_n256.npz"
export PI05_SENSITIVITY="$FINAL_ROOT/sensitivity/pi05_sensitivity_action_n16_d4_merged.json"
export PI05_FULL_PLAN="$FINAL_ROOT/plans/pi05_quantvla_uniform_w4a8_d4.plan.json"
export PI05_FULL_A8="$FINAL_ROOT/a8/pi05_quantvla_uniform_w4a8_d4_p999_b32x8.npz"
export PI05_FULL_ATM="$FINAL_ROOT/atm_ohb/pi05_quantvla_uniform_w4a8_d4_static_perhead.json"
export PI05_GDSQ_PLAN="$FINAL_ROOT/plans/pi05_gdsq_cscka_16to1_d4.final_plan.json"
export PI05_GDSQ_A8="$FINAL_ROOT/a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz"
export PI05_GDSQ_ATM="$FINAL_ROOT/atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json"
export PI05_FINAL_SELECTION="$FINAL_ROOT/selection/final_ratio_selection.json"
export PI05_DEV_SUMMARY="$FINAL_ROOT/selection/dev_summary.json"
export PI05_DEV_EQUIVALENCE="$FINAL_ROOT/selection/executable_equivalence.json"

GPUS=(1 2 3 4 5 6 7)
PORTS=(18401 18402 18403 18404 18405 18406 18407)
CONFIGS=(gdsq_vla gdsq_vla_atmohb quantvla_w4a8_atmohb fp16)

usage() {
    echo "usage: $0 prepare | run-all | run-wave CONFIG | status | stop" >&2
}

instance_name() {
    local config="$1" gpu="$2"
    echo "wave_${config}_g${gpu}"
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

start_servers() {
    local config="$1"
    local pids=()
    local index gpu port instance
    for index in "${!GPUS[@]}"; do
        gpu="${GPUS[$index]}"
        port="${PORTS[$index]}"
        instance="$(instance_name "$config" "$gpu")"
        "$SERVER_MANAGER" start "$config" "$gpu" "$port" "$instance" &
        pids+=("$!")
    done
    local failed=0 pid
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if [[ "$failed" != 0 ]]; then
        echo "one or more $config servers failed startup" >&2
        return 1
    fi
}

stop_servers() {
    local config="$1"
    local pids=()
    local gpu instance
    for gpu in "${GPUS[@]}"; do
        instance="$(instance_name "$config" "$gpu")"
        "$SERVER_MANAGER" stop "$instance" &
        pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
}

prepare_runtimes() {
    mkdir -p "$CONTROL_DIR"
    local config
    for config in "${CONFIGS[@]}"; do
        echo "[pi05 waves] preflight runtime config=$config"
        start_servers "$config"
        stop_servers "$config"
    done
}

create_manifest() {
    local args=()
    local config index gpu port instance runtime
    for config in "${CONFIGS[@]}"; do
        for index in "${!GPUS[@]}"; do
            gpu="${GPUS[$index]}"
            port="${PORTS[$index]}"
            instance="$(instance_name "$config" "$gpu")"
            runtime="$CONTROL_DIR/$instance.runtime.json"
            args+=(--server "$instance,$config,$gpu,$port,$runtime")
        done
    done
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$MANIFEST_TOOL" --out "$RUN_DIR/manifest.json" "${args[@]}"
}

prepare() {
    "$OPENPI_PY" "$ALIGNMENT_AUDIT" --phase artifacts \
        --out "$FINAL_ROOT/audit/gr00t_final_alignment.json"
    prepare_runtimes
    create_manifest
    echo "[pi05 waves] immutable manifest ready"
}

config_completed() {
    local config="$1"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$AGGREGATOR" \
        --run-dir "$RUN_DIR" \
        --out-dir "$RUN_DIR/aggregate_incomplete" \
        --allow-incomplete >/dev/null
    "$ROBOCASA_PY" - "$RUN_DIR/aggregate_incomplete/summary.json" "$config" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
row = summary["configs"][sys.argv[2]]
raise SystemExit(0 if row["completed_episodes"] == 2500 and row["missing_episodes"] == 0 else 1)
PY
}

start_worker() {
    local config="$1" port="$2" gpu="$3" shard="$4" seeds="$5" worker_id="$6" runtime="$7"
    local worker_dir="$CONTROL_DIR/workers"
    local log_file="$worker_dir/$worker_id.log"
    local pid_file="$worker_dir/$worker_id.pid"
    local hash
    mkdir -p "$worker_dir" "$RUN_DIR/results/$config"
    hash="$(runtime_hash "$runtime")"
    nohup setsid "$SEEDED_WORKER" \
        "$config" "$port" "$gpu" "$shard" "${#GPUS[@]}" "$seeds" "$worker_id" "$hash" "$RUN_DIR" \
        >"$log_file" 2>&1 </dev/null &
    echo "$!" >"$pid_file"
    echo "[pi05 waves] worker=$worker_id pid=$! config=$config shard=$shard/$((${#GPUS[@]} - 1)) seeds=$seeds"
}

start_wave_workers() {
    local config="$1"
    local index gpu port instance runtime prefix
    for index in "${!GPUS[@]}"; do
        gpu="${GPUS[$index]}"
        port="${PORTS[$index]}"
        instance="$(instance_name "$config" "$gpu")"
        runtime="$CONTROL_DIR/$instance.runtime.json"
        prefix="${instance}_t${index}"
        start_worker "$config" "$port" "$gpu" "$index" 0-24 "${prefix}_lo" "$runtime"
        start_worker "$config" "$port" "$gpu" "$index" 25-49 "${prefix}_hi" "$runtime"
    done
}

wait_wave_workers() {
    local config="$1"
    local prefix="wave_${config}_"
    local pid_file pid alive
    while true; do
        alive=0
        shopt -s nullglob
        for pid_file in "$CONTROL_DIR/workers"/${prefix}*.pid; do
            pid="$(<"$pid_file")"
            if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
                alive=$((alive + 1))
            fi
        done
        echo "[pi05 waves] config=$config workers_alive=$alive"
        [[ "$alive" == 0 ]] && return
        sleep 60
    done
}

start_monitor() {
    local config="$1"
    local log_file="$CONTROL_DIR/gpu_monitor.$config.log"
    nohup setsid "$ROBOCASA_PY" "$GPU_MONITOR" --run-dir "$RUN_DIR" --interval 30 \
        >"$log_file" 2>&1 </dev/null &
    echo "$!" >"$CONTROL_DIR/gpu_monitor.$config.pid"
}

run_wave() {
    local config="$1"
    if [[ ! " ${CONFIGS[*]} " =~ [[:space:]]${config}[[:space:]] ]]; then
        echo "unknown config: $config" >&2
        exit 2
    fi
    if config_completed "$config"; then
        echo "[pi05 waves] config already complete: $config"
        return
    fi
    local attempt
    for attempt in 1 2 3; do
        echo "[pi05 waves] config=$config attempt=$attempt"
        # Idempotently reuse healthy replicas and restart any replica that
        # exited during a previous attempt before resuming committed keys.
        start_servers "$config"
        start_wave_workers "$config"
        start_monitor "$config"
        wait_wave_workers "$config"
        if config_completed "$config"; then
            stop_servers "$config"
            echo "[pi05 waves] config complete: $config"
            return
        fi
        echo "[pi05 waves] config incomplete after attempt=$attempt; retrying committed-key resume" >&2
    done
    stop_servers "$config"
    echo "[pi05 waves] config failed strict completion after three attempts: $config" >&2
    exit 1
}

run_all() {
    [[ -f "$RUN_DIR/manifest.json" ]] || prepare
    "$OPENPI_PY" "$ALIGNMENT_AUDIT" --phase final \
        --out "$FINAL_ROOT/audit/gr00t_final_alignment.json"
    local config
    for config in "${CONFIGS[@]}"; do
        run_wave "$config"
    done
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$AGGREGATOR" \
        --run-dir "$RUN_DIR" --out-dir "$RUN_DIR/aggregate"
    echo "[pi05 waves] complete 10,000-episode matrix"
}

status() {
    if [[ -f "$RUN_DIR/manifest.json" ]]; then
        PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
            "$ROBOCASA_PY" "$AGGREGATOR" \
            --run-dir "$RUN_DIR" --out-dir "$RUN_DIR/aggregate_incomplete" --allow-incomplete
    else
        echo "manifest pending: $RUN_DIR/manifest.json"
    fi
    "$SERVER_MANAGER" status
}

stop_all() {
    local pid_file pid command config
    local worker_pids=()
    shopt -s nullglob
    for pid_file in "$CONTROL_DIR/workers"/*.pid; do
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            command="$(ps -p "$pid" -o args=)"
            if [[ "$command" == *run_pi05_formal_worker_seeded.sh* ]]; then
                kill -- "-$pid" 2>/dev/null || kill "$pid"
                worker_pids+=("$pid")
            else
                echo "refusing to stop unrelated worker pid=$pid command=$command" >&2
            fi
        fi
    done
    local worker_pid
    for worker_pid in "${worker_pids[@]}"; do
        for _ in $(seq 1 30); do
            kill -0 "$worker_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$worker_pid" 2>/dev/null; then
            command="$(ps -p "$worker_pid" -o args=)"
            if [[ "$command" == *run_pi05_formal_worker_seeded.sh* ]]; then
                kill -KILL -- "-$worker_pid" 2>/dev/null || kill -KILL "$worker_pid"
            else
                echo "refusing worker KILL escalation pid=$worker_pid command=$command" >&2
            fi
        fi
    done
    for config in "${CONFIGS[@]}"; do
        stop_servers "$config"
    done
}

case "${1:-}" in
    prepare) prepare ;;
    run-all) run_all ;;
    run-wave) [[ $# == 2 ]] || { usage; exit 2; }; run_wave "$2" ;;
    status) status ;;
    stop) stop_all ;;
    *) usage; exit 2 ;;
esac
