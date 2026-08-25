#!/usr/bin/env bash
set -euo pipefail

# Four-config-parallel resource schedule for the GR00T-final-aligned pi0.5
# Table-1 run.  The base manifest remains immutable; this scheduler is bound by
# a separate immutable schedule amendment that records the exact cutover
# keyset, runtime hashes, task shards, and seed intervals.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
FINAL_ROOT="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned"
RUN_DIR="$FINAL_ROOT/official_target_paired50"
CONTROL_DIR="$RUN_DIR/control"
WORKER_DIR="$CONTROL_DIR/workers"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
SEEDED_WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
SCHEDULE_TOOL="$REPO_ROOT/scripts/tools/pi05_make_parallel_schedule.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_pi05_robocasa365.py"
GPU_MONITOR="$REPO_ROOT/scripts/tools/monitor_pi05_formal_gpu.py"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
BASE_MANIFEST="$RUN_DIR/manifest.json"
SCHEDULE="$RUN_DIR/manifest.schedule_parallel_v1.json"

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

# One model replica per GPU.  The nearly-complete GDSQ configuration receives
# one GPU; each new configuration receives two.  Two disjoint seed workers per
# replica keep the total at fourteen concurrent environments.
SERVER_ROWS=(
    "parallel_v1_fp16_g1,fp16,1,18601"
    "parallel_v1_fp16_g2,fp16,2,18602"
    "parallel_v1_w4_g3,quantvla_w4a8_atmohb,3,18603"
    "parallel_v1_w4_g4,quantvla_w4a8_atmohb,4,18604"
    "parallel_v1_gdsqatm_g5,gdsq_vla_atmohb,5,18605"
    "parallel_v1_gdsqatm_g6,gdsq_vla_atmohb,6,18606"
    "parallel_v1_gdsq_g7,gdsq_vla,7,18607"
)

WORKER_ROWS=(
    "parallel_v1_fp16_g1_t0_lo,fp16,parallel_v1_fp16_g1,0,2,0-24"
    "parallel_v1_fp16_g1_t0_hi,fp16,parallel_v1_fp16_g1,0,2,25-49"
    "parallel_v1_fp16_g2_t1_lo,fp16,parallel_v1_fp16_g2,1,2,0-24"
    "parallel_v1_fp16_g2_t1_hi,fp16,parallel_v1_fp16_g2,1,2,25-49"
    "parallel_v1_w4_g3_t0_lo,quantvla_w4a8_atmohb,parallel_v1_w4_g3,0,2,0-24"
    "parallel_v1_w4_g3_t0_hi,quantvla_w4a8_atmohb,parallel_v1_w4_g3,0,2,25-49"
    "parallel_v1_w4_g4_t1_lo,quantvla_w4a8_atmohb,parallel_v1_w4_g4,1,2,0-24"
    "parallel_v1_w4_g4_t1_hi,quantvla_w4a8_atmohb,parallel_v1_w4_g4,1,2,25-49"
    "parallel_v1_gdsqatm_g5_t0_lo,gdsq_vla_atmohb,parallel_v1_gdsqatm_g5,0,2,0-24"
    "parallel_v1_gdsqatm_g5_t0_hi,gdsq_vla_atmohb,parallel_v1_gdsqatm_g5,0,2,25-49"
    "parallel_v1_gdsqatm_g6_t1_lo,gdsq_vla_atmohb,parallel_v1_gdsqatm_g6,1,2,0-24"
    "parallel_v1_gdsqatm_g6_t1_hi,gdsq_vla_atmohb,parallel_v1_gdsqatm_g6,1,2,25-49"
    "parallel_v1_gdsq_g7_t0_lo,gdsq_vla,parallel_v1_gdsq_g7,0,1,0-24"
    "parallel_v1_gdsq_g7_t0_hi,gdsq_vla,parallel_v1_gdsq_g7,0,1,25-49"
)

usage() {
    echo "usage: $0 start | start-servers | create-schedule | start-workers | status | stop" >&2
}

runtime_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib, json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

start_servers() {
    local pids=() row instance config gpu port pid failed=0
    for row in "${SERVER_ROWS[@]}"; do
        IFS=',' read -r instance config gpu port <<<"$row"
        "$SERVER_MANAGER" start "$config" "$gpu" "$port" "$instance" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]] || { stop_servers; return 1; }
}

stop_servers() {
    local pids=() row instance _config _gpu _port pid
    for row in "${SERVER_ROWS[@]}"; do
        IFS=',' read -r instance _config _gpu _port <<<"$row"
        "$SERVER_MANAGER" stop "$instance" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
}

schedule_args() {
    local row instance config gpu port runtime worker
    for row in "${SERVER_ROWS[@]}"; do
        IFS=',' read -r instance config gpu port <<<"$row"
        runtime="$CONTROL_DIR/$instance.runtime.json"
        printf '%s\0%s\0' --server "$instance,$config,$gpu,$port,$runtime"
    done
    for worker in "${WORKER_ROWS[@]}"; do
        printf '%s\0%s\0' --worker "$worker"
    done
}

schedule_tool() {
    local mode="$1" item
    local args=(
        --base-manifest "$BASE_MANIFEST"
        --scheduler "$REPO_ROOT/scripts/run_pi05_gr00t_aligned_parallel.sh"
        --run-dir "$RUN_DIR"
        --out "$SCHEDULE"
    )
    while IFS= read -r -d '' item; do args+=("$item"); done < <(schedule_args)
    [[ "$mode" == verify ]] && args+=(--verify)
    "$ROBOCASA_PY" "$SCHEDULE_TOOL" "${args[@]}"
}

create_schedule() {
    [[ -f "$BASE_MANIFEST" ]] || { echo "base manifest missing" >&2; return 1; }
    [[ ! -e "$SCHEDULE" ]] || { echo "immutable schedule already exists: $SCHEDULE" >&2; return 1; }
    schedule_tool create
}

start_worker() {
    local worker_id="$1" config="$2" instance="$3" gpu="$4" port="$5"
    local shard="$6" shard_count="$7" seeds="$8"
    local runtime="$CONTROL_DIR/$instance.runtime.json"
    local pid_file="$WORKER_DIR/$worker_id.pid"
    local log_file="$WORKER_DIR/$worker_id.log"
    local pid="" hash
    mkdir -p "$WORKER_DIR" "$RUN_DIR/results/$config"
    [[ -f "$pid_file" ]] && pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[pi05 parallel-v1] reuse worker=$worker_id pid=$pid"
        return
    fi
    if [[ -f "$log_file" ]] && grep -Fq "formal seeded worker complete:" "$log_file"; then
        echo "[pi05 parallel-v1] completed worker=$worker_id"
        return
    fi
    hash="$(runtime_hash "$runtime")"
    nohup setsid "$SEEDED_WORKER" \
        "$config" "$port" "$gpu" "$shard" "$shard_count" "$seeds" \
        "$worker_id" "$hash" "$RUN_DIR" >"$log_file" 2>&1 </dev/null &
    echo "$!" >"$pid_file"
    echo "[pi05 parallel-v1] worker=$worker_id pid=$! config=$config shard=$shard/$shard_count seeds=$seeds"
}

start_workers() {
    local worker worker_id config instance shard shard_count seeds
    local server_row server_instance server_config gpu port found
    for worker in "${WORKER_ROWS[@]}"; do
        IFS=',' read -r worker_id config instance shard shard_count seeds <<<"$worker"
        found=0
        for server_row in "${SERVER_ROWS[@]}"; do
            IFS=',' read -r server_instance server_config gpu port <<<"$server_row"
            if [[ "$server_instance" == "$instance" && "$server_config" == "$config" ]]; then
                start_worker "$worker_id" "$config" "$instance" "$gpu" "$port" \
                    "$shard" "$shard_count" "$seeds"
                found=1
                break
            fi
        done
        [[ "$found" == 1 ]] || { echo "worker server not found: $worker" >&2; return 1; }
    done
}

stop_workers() {
    local pid_file pid command
    shopt -s nullglob
    for pid_file in "$WORKER_DIR"/parallel_v1_*.pid; do
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
    for pid_file in "$WORKER_DIR"/parallel_v1_*.pid; do
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

failed_workers() {
    local row worker_id _config _instance _shard _count _seeds pid_file log_file pid=""
    for row in "${WORKER_ROWS[@]}"; do
        IFS=',' read -r worker_id _config _instance _shard _count _seeds <<<"$row"
        pid_file="$WORKER_DIR/$worker_id.pid"
        log_file="$WORKER_DIR/$worker_id.log"
        [[ -f "$pid_file" ]] && pid="$(<"$pid_file")" || pid=""
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then continue; fi
        if [[ -f "$log_file" ]] && grep -Fq "formal seeded worker complete:" "$log_file"; then continue; fi
        echo "$worker_id"
    done
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
    keys = {(row["config"], row["task_set"], row["task"], int(row["seed"])) for row in rows}
    if len(keys) != len(rows) or any(row.get("status") != "complete" for row in rows):
        raise SystemExit(f"invalid committed rows: {config}")
    parts.append(f"{config}={len(rows)}/2500")
print("[pi05 parallel-v1] " + " ".join(parts))
PY
}

complete() {
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$AGGREGATOR" --run-dir "$RUN_DIR" \
        --out-dir "$RUN_DIR/aggregate_incomplete" --allow-incomplete >/dev/null
    "$ROBOCASA_PY" - "$RUN_DIR/aggregate_incomplete/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
ok = all(row["completed_episodes"] == 2500 and row["missing_episodes"] == 0 for row in summary["configs"].values())
raise SystemExit(0 if ok else 1)
PY
}

start_monitor() {
    local pid_file pid
    shopt -s nullglob
    for pid_file in "$CONTROL_DIR"/gpu_monitor.*.pid; do
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            echo "[pi05 parallel-v1] reuse gpu monitor pid=$pid"
            return
        fi
    done
    nohup setsid "$ROBOCASA_PY" "$GPU_MONITOR" --run-dir "$RUN_DIR" --interval 30 \
        >"$CONTROL_DIR/gpu_monitor.parallel_v1.log" 2>&1 </dev/null &
    echo "$!" >"$CONTROL_DIR/gpu_monitor.parallel_v1.pid"
}

run_all() {
    start_servers
    if [[ -f "$SCHEDULE" ]]; then schedule_tool verify; else create_schedule; fi
    start_workers
    start_monitor
    local failed alive
    while true; do
        progress
        failed="$(failed_workers)"
        if [[ -n "$failed" ]]; then
            echo "[pi05 parallel-v1] restarting abnormal workers: $failed" >&2
            stop_workers
            start_workers
        fi
        alive="$(workers_alive)"
        echo "[pi05 parallel-v1] workers_alive=$alive"
        if [[ "$alive" == 0 ]]; then
            if complete; then
                PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
                    "$ROBOCASA_PY" "$AGGREGATOR" --run-dir "$RUN_DIR" --out-dir "$RUN_DIR/aggregate"
                stop_servers
                echo "[pi05 parallel-v1] complete 10,000-episode matrix"
                return
            fi
            echo "parallel workers exited before strict completion" >&2
            return 1
        fi
        sleep 60
    done
}

status() {
    progress
    echo "[pi05 parallel-v1] workers_alive=$(workers_alive)"
    "$SERVER_MANAGER" status
}

stop_all() {
    stop_workers
    stop_servers
}

case "${1:-}" in
    start) run_all ;;
    start-servers) start_servers ;;
    create-schedule) create_schedule ;;
    start-workers) schedule_tool verify; start_workers; start_monitor ;;
    status) status ;;
    stop) stop_all ;;
    *) usage; exit 2 ;;
esac
