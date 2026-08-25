#!/usr/bin/env bash
set -euo pipefail

# Four-config-parallel Table-1 scheduler.  This changes only resource layout;
# frozen models, tasks, seeds, paired noise, horizons, and committed-key resume
# semantics remain identical to the base immutable manifest.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
FINAL_ROOT="$REPO_ROOT/runs/pi05_gdsq_final"
RUN_DIR="$FINAL_ROOT/official_pretrain_paired50"
CONTROL_DIR="$RUN_DIR/control"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
SEEDED_WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
SCHEDULE_TOOL="$REPO_ROOT/scripts/tools/pi05_make_parallel_schedule.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_pi05_robocasa365.py"
GPU_MONITOR="$REPO_ROOT/scripts/tools/monitor_pi05_formal_gpu.py"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
BASE_MANIFEST="$RUN_DIR/manifest.json"
SUPERSEDED_SCHEDULE="$RUN_DIR/manifest.schedule_parallel_v3.json"
SCHEDULE="$RUN_DIR/manifest.schedule_parallel_v4.json"

export PI05_REQUIRE_FAITHFUL_FINAL=1
export PI05_CONTROL_DIR="$CONTROL_DIR"
export PI05_PACK_DIR="$REPO_ROOT/runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
export PI05_PACK_MANIFEST="$PI05_PACK_DIR/manifest.json"
export PI05_CALIBRATION_BUFFER="$REPO_ROOT/runs/pi05_gdsq_port/probes/pi05_robocasa_official_buffer32_state16.npz"
export PI05_SENSITIVITY="$FINAL_ROOT/probes/pi05_sensitivity_action_n16_w4_merged.json"
export PI05_FULL_PLAN="$REPO_ROOT/runs/pi05_gdsq_port/plans/pi05_quantvla_full_w4a8.plan.json"
export PI05_FULL_A8="$REPO_ROOT/runs/pi05_gdsq_port/a8_scales/pi05_full_w4a8_state16_p999_b32.npz"
export PI05_FULL_ATM="$REPO_ROOT/runs/pi05_gdsq_port/atm_ohb/pi05_full_w4a8_state16_static_expert.json"
export PI05_GDSQ_PLAN="$FINAL_ROOT/final/pi05_gdsq_vla_final.plan.json"
export PI05_GDSQ_A8="$FINAL_ROOT/final/pi05_gdsq_vla_final.a8_p999_b32.npz"
export PI05_GDSQ_ATM="$FINAL_ROOT/final/pi05_gdsq_vla_final.static_atm_ohb.json"
export PI05_FINAL_SELECTION="$FINAL_ROOT/final/final_ratio_selection.json"

SERVER_ROWS=(
    "wave_fp16_g1,fp16,1,18401"
    "parallel_quantvla_w4a8_atmohb_g2,quantvla_w4a8_atmohb,2,18402"
    "wave_quantvla_w4a8_atmohb_g3,quantvla_w4a8_atmohb,3,18403"
    "wave_gdsq_vla_atmohb_g4,gdsq_vla_atmohb,4,18404"
    "wave_gdsq_vla_g5,gdsq_vla,5,18405"
    "wave_gdsq_vla_atmohb_g6,gdsq_vla_atmohb,6,18406"
    "parallel_v3_gdsq_vla_g1,gdsq_vla,1,18407"
    "parallel_v3_fp16_g4,fp16,4,18408"
    "parallel_v3_fp16_g6,fp16,6,18409"
)

WORKER_ROWS=(
    "parallel_v3_fp16_g1_t0_lo,fp16,wave_fp16_g1,0,3,0-24"
    "parallel_v3_fp16_g1_t0_hi,fp16,wave_fp16_g1,0,3,25-49"
    "parallel_v3_fp16_g4_t1_lo,fp16,parallel_v3_fp16_g4,1,3,0-24"
    "parallel_v3_fp16_g4_t1_hi,fp16,parallel_v3_fp16_g4,1,3,25-49"
    "parallel_v3_fp16_g6_t2_lo,fp16,parallel_v3_fp16_g6,2,3,0-24"
    "parallel_v3_fp16_g6_t2_hi,fp16,parallel_v3_fp16_g6,2,3,25-49"
    "parallel_v4_w4_g2_t0_q0,quantvla_w4a8_atmohb,parallel_quantvla_w4a8_atmohb_g2,0,2,0-12"
    "parallel_v4_w4_g2_t0_q1,quantvla_w4a8_atmohb,parallel_quantvla_w4a8_atmohb_g2,0,2,13-24"
    "parallel_v4_w4_g2_t0_q2,quantvla_w4a8_atmohb,parallel_quantvla_w4a8_atmohb_g2,0,2,25-37"
    "parallel_v4_w4_g2_t0_q3,quantvla_w4a8_atmohb,parallel_quantvla_w4a8_atmohb_g2,0,2,38-49"
    "parallel_v4_w4_g3_t1_q0,quantvla_w4a8_atmohb,wave_quantvla_w4a8_atmohb_g3,1,2,0-12"
    "parallel_v4_w4_g3_t1_q1,quantvla_w4a8_atmohb,wave_quantvla_w4a8_atmohb_g3,1,2,13-24"
    "parallel_v4_w4_g3_t1_q2,quantvla_w4a8_atmohb,wave_quantvla_w4a8_atmohb_g3,1,2,25-37"
    "parallel_v4_w4_g3_t1_q3,quantvla_w4a8_atmohb,wave_quantvla_w4a8_atmohb_g3,1,2,38-49"
    "parallel_v3_gdsqatm_g4_t0_lo,gdsq_vla_atmohb,wave_gdsq_vla_atmohb_g4,0,2,0-24"
    "parallel_v3_gdsqatm_g4_t0_hi,gdsq_vla_atmohb,wave_gdsq_vla_atmohb_g4,0,2,25-49"
    "parallel_v3_gdsqatm_g6_t1_lo,gdsq_vla_atmohb,wave_gdsq_vla_atmohb_g6,1,2,0-24"
    "parallel_v3_gdsqatm_g6_t1_hi,gdsq_vla_atmohb,wave_gdsq_vla_atmohb_g6,1,2,25-49"
    "parallel_v4_gdsq_g5_t0_q0,gdsq_vla,wave_gdsq_vla_g5,0,2,0-12"
    "parallel_v4_gdsq_g5_t0_q1,gdsq_vla,wave_gdsq_vla_g5,0,2,13-24"
    "parallel_v4_gdsq_g5_t0_q2,gdsq_vla,wave_gdsq_vla_g5,0,2,25-37"
    "parallel_v4_gdsq_g5_t0_q3,gdsq_vla,wave_gdsq_vla_g5,0,2,38-49"
    "parallel_v3_gdsq_g1_t1_lo,gdsq_vla,parallel_v3_gdsq_vla_g1,1,2,0-24"
    "parallel_v3_gdsq_g1_t1_hi,gdsq_vla,parallel_v3_gdsq_vla_g1,1,2,25-49"
)

usage() {
    echo "usage: $0 prepare | prepare-running | start | start-servers | start-workers | status | stop-workers | stop" >&2
}

runtime_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib, json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

start_servers() {
    local pids=() row instance config gpu port
    for row in "${SERVER_ROWS[@]}"; do
        IFS=',' read -r instance config gpu port <<<"$row"
        "$SERVER_MANAGER" start "$config" "$gpu" "$port" "$instance" &
        pids+=("$!")
    done
    local pid failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]] || { stop_servers; return 1; }
}

stop_servers() {
    local pids=() row instance _config _gpu _port
    for row in "${SERVER_ROWS[@]}"; do
        IFS=',' read -r instance _config _gpu _port <<<"$row"
        "$SERVER_MANAGER" stop "$instance" &
        pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do wait "$pid" || true; done
}

schedule_args() {
    local row instance config gpu port runtime
    for row in "${SERVER_ROWS[@]}"; do
        IFS=',' read -r instance config gpu port <<<"$row"
        runtime="$CONTROL_DIR/$instance.runtime.json"
        printf '%s\0%s\0' --server "$instance,$config,$gpu,$port,$runtime"
    done
    local worker
    for worker in "${WORKER_ROWS[@]}"; do
        printf '%s\0%s\0' --worker "$worker"
    done
}

run_schedule_tool() {
    local mode="${1:-create}"
    local args=(
        --base-manifest "$BASE_MANIFEST"
        --scheduler "$REPO_ROOT/scripts/run_pi05_faithful_parallel.sh"
        --run-dir "$RUN_DIR"
        --out "$SCHEDULE"
    )
    if [[ -f "$SUPERSEDED_SCHEDULE" ]]; then
        args+=(--supersedes "$SUPERSEDED_SCHEDULE")
    fi
    local item
    while IFS= read -r -d '' item; do args+=("$item"); done < <(schedule_args)
    if [[ "$mode" == verify ]]; then args+=(--verify); fi
    "$ROBOCASA_PY" "$SCHEDULE_TOOL" "${args[@]}"
}

prepare() {
    [[ -f "$BASE_MANIFEST" ]] || { echo "base manifest missing" >&2; exit 1; }
    [[ ! -e "$SCHEDULE" ]] || { echo "parallel schedule already exists: $SCHEDULE" >&2; exit 1; }
    start_servers
    run_schedule_tool create
    stop_servers
}

prepare_running() {
    [[ -f "$BASE_MANIFEST" ]] || { echo "base manifest missing" >&2; exit 1; }
    [[ ! -e "$SCHEDULE" ]] || { echo "parallel schedule already exists: $SCHEDULE" >&2; exit 1; }
    local row instance _config _gpu _port runtime pid_file pid
    for row in "${SERVER_ROWS[@]}"; do
        IFS=',' read -r instance _config _gpu _port <<<"$row"
        runtime="$CONTROL_DIR/$instance.runtime.json"
        pid_file="$CONTROL_DIR/$instance.pid"
        [[ -f "$runtime" && -f "$pid_file" ]] || {
            echo "running server evidence missing for $instance" >&2
            exit 1
        }
        pid="$(<"$pid_file")"
        [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null || {
            echo "server is not running: $instance pid=$pid" >&2
            exit 1
        }
    done
    run_schedule_tool create
}

start_worker() {
    local worker_id="$1" config="$2" instance="$3" gpu="$4" port="$5"
    local shard="$6" shard_count="$7" seeds="$8"
    local worker_dir="$CONTROL_DIR/workers"
    local runtime="$CONTROL_DIR/$instance.runtime.json"
    local pid_file="$worker_dir/$worker_id.pid"
    local log_file="$worker_dir/$worker_id.log"
    local pid=""
    mkdir -p "$worker_dir" "$RUN_DIR/results/$config"
    if [[ -f "$pid_file" ]]; then pid="$(<"$pid_file")"; fi
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[pi05 parallel] reuse worker=$worker_id pid=$pid"
        return
    fi
    local hash
    hash="$(runtime_hash "$runtime")"
    nohup setsid "$SEEDED_WORKER" \
        "$config" "$port" "$gpu" "$shard" "$shard_count" "$seeds" \
        "$worker_id" "$hash" "$RUN_DIR" >"$log_file" 2>&1 </dev/null &
    echo "$!" >"$pid_file"
    echo "[pi05 parallel] worker=$worker_id pid=$! config=$config shard=$shard/$shard_count seeds=$seeds"
}

start_workers() {
    local worker worker_id config instance shard shard_count seeds
    local row server_instance _server_config gpu port found
    for worker in "${WORKER_ROWS[@]}"; do
        IFS=',' read -r worker_id config instance shard shard_count seeds <<<"$worker"
        found=0
        for row in "${SERVER_ROWS[@]}"; do
            IFS=',' read -r server_instance _server_config gpu port <<<"$row"
            if [[ "$server_instance" == "$instance" ]]; then
                start_worker "$worker_id" "$config" "$instance" "$gpu" "$port" \
                    "$shard" "$shard_count" "$seeds"
                found=1
                break
            fi
        done
        [[ "$found" == 1 ]] || { echo "worker server not found: $worker" >&2; exit 1; }
    done
}

stop_workers() {
    local pid_file pid command
    shopt -s nullglob
    for pid_file in "$CONTROL_DIR/workers"/parallel_*.pid; do
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            command="$(ps -p "$pid" -o args=)"
            if [[ "$command" == *run_pi05_formal_worker_seeded.sh* ]]; then
                kill -- "-$pid" 2>/dev/null || kill "$pid"
            else
                echo "refusing unrelated pid=$pid command=$command" >&2
            fi
        fi
    done
}

workers_alive() {
    local count=0 pid_file pid
    shopt -s nullglob
    for pid_file in "$CONTROL_DIR/workers"/parallel_*.pid; do
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

progress() {
    "$ROBOCASA_PY" - "$RUN_DIR/results" <<'PY'
import glob, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
parts = []
for config in ("fp16", "quantvla_w4a8_atmohb", "gdsq_vla_atmohb", "gdsq_vla"):
    rows = []
    for name in glob.glob(str(root / config / "*.jsonl")):
        rows.extend(json.loads(line) for line in open(name, encoding="utf-8") if line.strip())
    keys = {(row["config"], row["task_set"], row["task"], row["seed"]) for row in rows}
    if len(keys) != len(rows):
        raise SystemExit(f"duplicate committed rows: {config}")
    parts.append(f"{config}={len(rows)}/2500")
print("[pi05 parallel] " + " ".join(parts))
PY
}

complete() {
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$AGGREGATOR" --run-dir "$RUN_DIR" \
        --out-dir "$RUN_DIR/aggregate_incomplete" --allow-incomplete >/dev/null
    "$ROBOCASA_PY" - "$RUN_DIR/aggregate_incomplete/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
ok = all(row["completed_episodes"] == 2500 and row["missing_episodes"] == 0 for row in x["configs"].values())
raise SystemExit(0 if ok else 1)
PY
}

start_monitor() {
    local pid_file="$CONTROL_DIR/gpu_monitor.parallel.pid" pid=""
    if [[ -f "$pid_file" ]]; then pid="$(<"$pid_file")"; fi
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then return; fi
    nohup setsid "$ROBOCASA_PY" "$GPU_MONITOR" --run-dir "$RUN_DIR" --interval 30 \
        >"$CONTROL_DIR/gpu_monitor.parallel.log" 2>&1 </dev/null &
    echo "$!" >"$pid_file"
}

run_all() {
    [[ -f "$SCHEDULE" ]] || { echo "run prepare first" >&2; exit 1; }
    local attempt alive
    for attempt in 1 2 3; do
        echo "[pi05 parallel] attempt=$attempt"
        start_servers
        run_schedule_tool verify
        start_workers
        start_monitor
        while true; do
            alive="$(workers_alive)"
            progress
            echo "[pi05 parallel] workers_alive=$alive"
            [[ "$alive" == 0 ]] && break
            sleep 60
        done
        if complete; then
            PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
                "$ROBOCASA_PY" "$AGGREGATOR" --run-dir "$RUN_DIR" \
                --out-dir "$RUN_DIR/aggregate"
            stop_servers
            echo "[pi05 parallel] complete 10,000-episode matrix"
            return
        fi
        echo "[pi05 parallel] incomplete; committed-key retry" >&2
    done
    stop_servers
    echo "parallel matrix incomplete after three attempts" >&2
    exit 1
}

status() {
    progress
    echo "[pi05 parallel] workers_alive=$(workers_alive)"
    "$SERVER_MANAGER" status
}

stop_all() {
    stop_workers
    stop_servers
}

case "${1:-}" in
    prepare) prepare ;;
    prepare-running) prepare_running ;;
    start) run_all ;;
    start-servers) start_servers ;;
    start-workers)
        [[ -f "$SCHEDULE" ]] || { echo "run prepare-running first" >&2; exit 1; }
        run_schedule_tool verify
        start_workers
        start_monitor
        ;;
    status) status ;;
    stop-workers) stop_workers ;;
    stop) stop_all ;;
    *) usage; exit 2 ;;
esac
