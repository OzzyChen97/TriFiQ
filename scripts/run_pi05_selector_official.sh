#!/usr/bin/env bash
set -euo pipefail

# Preregistered pi0.5 GDSQ-VLA + v8 runtime-selector official evaluation.
# Five model replicas occupy GPUs 3-7; ten disjoint workers cover 50 tasks x
# 50 seeds exactly.  The immutable manifest is frozen after the diagnostic
# 2-task x 2-seed preflight and before any official result row is written.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
WEEK1_ROOT="$REPO_ROOT/runs/gdsq_week1_preregistered_v1"
RUN_DIR="${PI05_SELECTOR_RUN_DIR:-$WEEK1_ROOT/pi05_selector_official50}"
CONTROL_DIR="$RUN_DIR/control"
WORKER_DIR="$CONTROL_DIR/workers"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
WORKER_LAUNCHER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
MANIFEST_TOOL="$REPO_ROOT/scripts/tools/pi05_make_selector_manifest.py"
PREFLIGHT_AUDITOR="$REPO_ROOT/scripts/tools/audit_pi05_selector_preflight.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_pi05_selector_official.py"
GPU_MONITOR="$REPO_ROOT/scripts/tools/monitor_pi05_formal_gpu.py"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
MANIFEST="$RUN_DIR/manifest.json"
CONFIG="gdsq_vla_runtime_selector"

export PI05_CONTROL_DIR="$CONTROL_DIR"
export PI05_PACK_DIR="$REPO_ROOT/runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
export PI05_PACK_MANIFEST="$PI05_PACK_DIR/manifest.json"
export PI05_CALIBRATION_BUFFER="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
export PI05_GDSQ_PLAN="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/plans/pi05_gdsq_cscka_16to1_d4.final_plan.json"
export PI05_GDSQ_A8="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz"
export PI05_GDSQ_ATM="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json"
export PI05_RUNTIME_SELECTOR="$REPO_ROOT/runs/atmohb_dynamic_selector_v8/selector.json"

SERVER_ROWS=(
    "selector_v8_g3,$CONFIG,3,19703"
    "selector_v8_g4,$CONFIG,4,19704"
    "selector_v8_g5,$CONFIG,5,19705"
    "selector_v8_g6,$CONFIG,6,19706"
    "selector_v8_g7,$CONFIG,7,19707"
)

WORKER_ROWS=(
    "selector_v8_g3_s0_lo,selector_v8_g3,0,5,0-24"
    "selector_v8_g3_s0_hi,selector_v8_g3,0,5,25-49"
    "selector_v8_g4_s1_lo,selector_v8_g4,1,5,0-24"
    "selector_v8_g4_s1_hi,selector_v8_g4,1,5,25-49"
    "selector_v8_g5_s2_lo,selector_v8_g5,2,5,0-24"
    "selector_v8_g5_s2_hi,selector_v8_g5,2,5,25-49"
    "selector_v8_g6_s3_lo,selector_v8_g6,3,5,0-24"
    "selector_v8_g6_s3_hi,selector_v8_g6,3,5,25-49"
    "selector_v8_g7_s4_lo,selector_v8_g7,4,5,0-24"
    "selector_v8_g7_s4_hi,selector_v8_g7,4,5,25-49"
)

usage() {
    echo "usage: $0 start | start-servers | preflight | freeze-manifest | start-workers | aggregate | status | stop" >&2
}

runtime_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib, json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

server_running() {
    local instance="$1" port="$2" pid_file="$CONTROL_DIR/$instance.pid" pid=""
    [[ -f "$pid_file" ]] && pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    curl -fsS --max-time 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 || return 1
    [[ -s "$CONTROL_DIR/$instance.runtime.json" ]]
}

start_one_server() {
    local row="$1" instance config gpu port
    IFS=',' read -r instance config gpu port <<<"$row"
    if server_running "$instance" "$port"; then
        echo "[pi05 selector] reuse server=$instance gpu=$gpu port=$port"
        return
    fi
    "$SERVER_MANAGER" start "$config" "$gpu" "$port" "$instance"
}

start_servers() {
    mkdir -p "$CONTROL_DIR"
    local row pid failed=0 pids=()
    for row in "${SERVER_ROWS[@]}"; do
        start_one_server "$row" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if [[ "$failed" != 0 ]]; then
        stop_servers
        return 1
    fi
}

stop_servers() {
    local row instance _config _gpu _port pid pids=()
    for row in "${SERVER_ROWS[@]}"; do
        IFS=',' read -r instance _config _gpu _port <<<"$row"
        "$SERVER_MANAGER" stop "$instance" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
}

server_row_for_instance() {
    local wanted="$1" row instance config gpu port
    for row in "${SERVER_ROWS[@]}"; do
        IFS=',' read -r instance config gpu port <<<"$row"
        if [[ "$instance" == "$wanted" ]]; then
            printf '%s,%s,%s,%s\n' "$instance" "$config" "$gpu" "$port"
            return
        fi
    done
    return 1
}

run_preflight() {
    local instance="selector_v8_g3" gpu=3 port=19703
    local runtime="$CONTROL_DIR/$instance.runtime.json"
    local directory="$RUN_DIR/preflight"
    local output="$directory/pi05_selector_2task2seed.jsonl"
    local summary="$directory/summary.json"
    local hash
    server_running "$instance" "$port" || {
        echo "preflight server is not running: $instance" >&2
        return 1
    }
    mkdir -p "$directory"
    hash="$(runtime_hash "$runtime")"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" \
        --host 127.0.0.1 \
        --port "$port" \
        --config-id "$CONFIG" \
        --task-set atomic_seen \
        --tasks OpenDrawer,TurnOnMicrowave \
        --trial-seeds 0-1 \
        --split target \
        --replan-steps 16 \
        --egl-device "$gpu" \
        --expected-server-metadata-sha256 "$hash" \
        --expect-runtime-selector \
        --resume-dir "$directory" \
        --out "$output"
    "$ROBOCASA_PY" "$PREFLIGHT_AUDITOR" \
        --results "$output" --runtime "$runtime" --out "$summary"
}

manifest_args() {
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

run_manifest_tool() {
    local mode="$1" item
    local args=(
        --run-dir "$RUN_DIR"
        --out "$MANIFEST"
        --orchestrator "$REPO_ROOT/scripts/run_pi05_selector_official.sh"
    )
    while IFS= read -r -d '' item; do args+=("$item"); done < <(manifest_args)
    [[ "$mode" == verify ]] && args+=(--verify)
    "$ROBOCASA_PY" "$MANIFEST_TOOL" "${args[@]}"
}

freeze_manifest() {
    if [[ -f "$MANIFEST" ]]; then
        run_manifest_tool verify
    else
        run_manifest_tool create
    fi
}

start_worker() {
    local worker_id="$1" instance="$2" shard="$3" count="$4" seeds="$5"
    local server_row config gpu port runtime hash pid_file log_file pid=""
    server_row="$(server_row_for_instance "$instance")"
    IFS=',' read -r _instance config gpu port <<<"$server_row"
    runtime="$CONTROL_DIR/$instance.runtime.json"
    pid_file="$WORKER_DIR/$worker_id.pid"
    log_file="$WORKER_DIR/$worker_id.log"
    mkdir -p "$WORKER_DIR" "$RUN_DIR/results/$CONFIG"
    [[ -f "$pid_file" ]] && pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[pi05 selector] reuse worker=$worker_id pid=$pid"
        return
    fi
    if [[ -f "$log_file" ]] && grep -Fq "formal seeded worker complete:" "$log_file"; then
        echo "[pi05 selector] completed worker=$worker_id"
        return
    fi
    server_running "$instance" "$port" || {
        echo "worker server is not running: $instance" >&2
        return 1
    }
    hash="$(runtime_hash "$runtime")"
    nohup setsid "$WORKER_LAUNCHER" \
        "$config" "$port" "$gpu" "$shard" "$count" "$seeds" \
        "$worker_id" "$hash" "$RUN_DIR" >"$log_file" 2>&1 </dev/null &
    echo "$!" >"$pid_file"
    echo "[pi05 selector] worker=$worker_id pid=$! shard=$shard/$count seeds=$seeds"
}

start_workers() {
    [[ -f "$MANIFEST" ]] || { echo "selector manifest is not frozen" >&2; return 1; }
    run_manifest_tool verify
    local row worker instance shard count seeds
    for row in "${WORKER_ROWS[@]}"; do
        IFS=',' read -r worker instance shard count seeds <<<"$row"
        start_worker "$worker" "$instance" "$shard" "$count" "$seeds"
    done
}

stop_workers() {
    local pid_file pid command process_group
    shopt -s nullglob
    for pid_file in "$WORKER_DIR"/selector_v8_*.pid; do
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            command="$(ps -p "$pid" -o args=)"
            if [[ "$command" != *run_pi05_formal_worker_seeded.sh* ]]; then
                echo "refusing to stop unrelated pid=$pid command=$command" >&2
                continue
            fi
            process_group="$(ps -p "$pid" -o pgid= | tr -d ' ')"
            if [[ "$process_group" == "$pid" ]]; then
                kill -- "-$pid"
            else
                kill "$pid"
            fi
        fi
    done
}

workers_alive() {
    local count=0 pid_file pid
    shopt -s nullglob
    for pid_file in "$WORKER_DIR"/selector_v8_*.pid; do
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

failed_workers() {
    local row worker _instance _shard _count _seeds pid_file log_file pid=""
    for row in "${WORKER_ROWS[@]}"; do
        IFS=',' read -r worker _instance _shard _count _seeds <<<"$row"
        pid_file="$WORKER_DIR/$worker.pid"
        log_file="$WORKER_DIR/$worker.log"
        [[ -f "$pid_file" ]] && pid="$(<"$pid_file")" || pid=""
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then continue; fi
        if [[ -f "$log_file" ]] && grep -Fq "formal seeded worker complete:" "$log_file"; then continue; fi
        echo "$worker"
    done
}

progress() {
    "$ROBOCASA_PY" - "$RUN_DIR/results/$CONFIG" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.jsonl")):
    rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
keys = {(row["config"], row["task_set"], row["task"], int(row["seed"])) for row in rows}
if len(keys) != len(rows):
    raise SystemExit("duplicate committed selector rows")
if any(row.get("status") != "complete" for row in rows):
    raise SystemExit("non-complete selector row committed")
if any(not row.get("runtime_selector_enabled") or row.get("selected_variant") != "ohb" for row in rows):
    raise SystemExit("invalid selector attestation in committed row")
print(f"[pi05 selector] completed={len(rows)}/2500 successes={sum(bool(row['success']) for row in rows)}")
PY
}

aggregate() {
    local mode="${1:-formal}" args=()
    [[ "$mode" == progress ]] && args+=(--allow-incomplete)
    local out_dir="$RUN_DIR/aggregate"
    [[ "$mode" == progress ]] && out_dir="$RUN_DIR/aggregate_incomplete"
    PYTHONPATH="$REPO_ROOT/scripts/tools:$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$AGGREGATOR" --run-dir "$RUN_DIR" \
        --out-dir "$out_dir" --bootstrap 10000 "${args[@]}"
}

is_complete() {
    aggregate progress >/dev/null
    "$ROBOCASA_PY" - "$RUN_DIR/aggregate_incomplete/summary.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value.get("complete") and value.get("completed_episodes") == 2500 else 1)
PY
}

start_monitor() {
    local pid_file="$CONTROL_DIR/gpu_monitor.selector_v8.pid" pid=""
    [[ -f "$pid_file" ]] && pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[pi05 selector] reuse GPU monitor pid=$pid"
        return
    fi
    nohup setsid "$ROBOCASA_PY" "$GPU_MONITOR" --run-dir "$RUN_DIR" --interval 30 \
        >"$CONTROL_DIR/gpu_monitor.selector_v8.log" 2>&1 </dev/null &
    echo "$!" >"$pid_file"
}

run_all() {
    start_servers
    run_preflight
    freeze_manifest
    start_workers
    start_monitor
    local failed alive
    while true; do
        progress
        failed="$(failed_workers)"
        if [[ -n "$failed" ]]; then
            echo "[pi05 selector] restarting abnormal workers: $failed" >&2
            start_workers
        fi
        alive="$(workers_alive)"
        echo "[pi05 selector] workers_alive=$alive"
        if [[ "$alive" == 0 ]]; then
            if is_complete; then
                aggregate formal
                stop_servers
                echo "[pi05 selector] complete: 2500 attested episodes"
                return
            fi
            echo "selector workers exited before strict completion" >&2
            return 1
        fi
        sleep 60
    done
}

status() {
    progress
    echo "[pi05 selector] workers_alive=$(workers_alive)"
    local failed
    failed="$(failed_workers)"
    [[ -z "$failed" ]] || echo "[pi05 selector] abnormal_workers=$failed"
    "$SERVER_MANAGER" status
}

stop_all() {
    stop_workers
    stop_servers
}

case "${1:-}" in
    start) run_all ;;
    start-servers) start_servers ;;
    preflight) run_preflight ;;
    freeze-manifest) freeze_manifest ;;
    start-workers) start_workers; start_monitor ;;
    aggregate) aggregate formal ;;
    status) status ;;
    stop) stop_all ;;
    *) usage; exit 2 ;;
esac
