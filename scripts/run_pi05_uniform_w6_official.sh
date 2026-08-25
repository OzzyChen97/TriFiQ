#!/usr/bin/env bash
set -euo pipefail

# Preregistered pi0.5 uniform-W6, static-A8, 50-task x 50-seed evaluation.
# The server GPU list is resource-configurable before the immutable manifest
# is frozen. Four clients per replica hide simulator latency. The Latin
# task/seed mapping spreads every task shard across four servers, avoiding the
# long-task tail observed in the already-frozen selector schedule.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
WEEK1_ROOT="$REPO_ROOT/runs/gdsq_week1_preregistered_v1"
RUN_DIR="${PI05_UNIFORM_W6_RUN_DIR:-$WEEK1_ROOT/execution/runs/pi05_uniform_w6_official50}"
CONTROL_DIR="$RUN_DIR/control"
WORKER_DIR="$CONTROL_DIR/workers"
CONFIG="uniform_w6"
PLAN="$WEEK1_ROOT/plans/pi05/uniform_w6.plan.json"
A8="$WEEK1_ROOT/execution/a8/pi05/uniform_w6_p999_b32x8.npz"
EXPECTED_WRAPPED=180
MANIFEST="$RUN_DIR/manifest.json"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_week1_server.sh"
WORKER_LAUNCHER="$REPO_ROOT/scripts/run_pi05_week1_worker.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
CALIBRATOR="$REPO_ROOT/scripts/tools/pi05_calibrate_a8.py"
MANIFEST_TOOL="$REPO_ROOT/scripts/tools/pi05_make_week1_manifest.py"
PREFLIGHT_AUDITOR="$REPO_ROOT/scripts/tools/audit_pi05_week1_preflight.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_pi05_week1.py"
GPU_MONITOR="$REPO_ROOT/scripts/tools/monitor_pi05_formal_gpu.py"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
SELECTOR_MANIFEST="$WEEK1_ROOT/pi05_selector_official50/manifest.json"
STATIC_MANIFEST="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/official_target_paired50/manifest.json"
PREP_GPU="${PI05_UNIFORM_W6_PREP_GPU:-3}"
PREFLIGHT_GPU="${PI05_UNIFORM_W6_PREFLIGHT_GPU:-2}"
SERVER_GPU_SPEC="${PI05_UNIFORM_W6_SERVER_GPUS:-0,1,2,3,4,5,6,7}"

export PI05_WEEK1_CONTROL_DIR="$CONTROL_DIR"

SERVER_ROWS=()
WORKER_ROWS=()
SEED_RANGES=("0-12" "13-24" "25-37" "38-49")
IFS=',' read -r -a SERVER_GPUS <<<"$SERVER_GPU_SPEC"
[[ "${#SERVER_GPUS[@]}" -gt 0 ]] || { echo "empty PI05_UNIFORM_W6_SERVER_GPUS" >&2; exit 2; }
declare -A SEEN_SERVER_GPU=()
for gpu in "${SERVER_GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-7]$ ]] || { echo "invalid server GPU: $gpu" >&2; exit 2; }
    [[ -z "${SEEN_SERVER_GPU[$gpu]:-}" ]] || { echo "duplicate server GPU: $gpu" >&2; exit 2; }
    SEEN_SERVER_GPU[$gpu]=1
    SERVER_ROWS+=("pi05_w6_g${gpu},$CONFIG,$gpu,$((20200 + gpu))")
done
SHARD_COUNT="${#SERVER_GPUS[@]}"
for ((shard = 0; shard < SHARD_COUNT; shard++)); do
    for quarter in {0..3}; do
        server_index=$(( (shard + quarter) % SHARD_COUNT ))
        gpu="${SERVER_GPUS[$server_index]}"
        WORKER_ROWS+=(
            "pi05_w6_g${gpu}_s${shard}_q${quarter},pi05_w6_g${gpu},$shard,$SHARD_COUNT,${SEED_RANGES[$quarter]}"
        )
    done
done

usage() {
    echo "usage: $0 prepare | schedule | start | start-servers | preflight | concurrency-preflight | freeze-manifest | start-workers | aggregate | status | stop" >&2
}

gpu_free() {
    local gpu="$1" output
    output="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    [[ -z "${output//[[:space:]]/}" ]]
}

require_gpus_free() {
    local gpu
    for gpu in "${SERVER_GPUS[@]}"; do
        if ! gpu_free "$gpu"; then
            echo "refusing to start: GPU $gpu has a live compute process" >&2
            return 1
        fi
    done
}

print_schedule() {
    local row gpu count
    echo "[pi05 W6] server_gpus=$SERVER_GPU_SPEC servers=${#SERVER_ROWS[@]} workers=${#WORKER_ROWS[@]} task_shards=$SHARD_COUNT"
    for gpu in "${SERVER_GPUS[@]}"; do
        count=0
        for row in "${WORKER_ROWS[@]}"; do
            [[ "$row" == *,"pi05_w6_g${gpu}",* ]] && count=$((count + 1))
        done
        echo "[pi05 W6] gpu=$gpu assigned_workers=$count"
    done
}

require_gpu_free() {
    local gpu="$1"
    if ! gpu_free "$gpu"; then
        echo "refusing to prepare W6 A8: GPU $gpu has a live compute process" >&2
        return 1
    fi
}

runtime_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["server_metadata_sha256"])
PY
}

server_running() {
    local instance="$1" port="$2" pid_file="$CONTROL_DIR/$instance.pid" pid=""
    [[ -f "$pid_file" ]] && pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    curl -fsS --max-time 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 || return 1
    [[ -s "$CONTROL_DIR/$instance.runtime.json" && -s "$CONTROL_DIR/$instance.audit.json" ]]
}

prepare() {
    if [[ -f "$A8" && -f "$A8.json" ]]; then
        "$ROBOCASA_PY" - "$PLAN" "$A8" "$EXPECTED_WRAPPED" <<'PY'
import hashlib, json, pathlib, sys
plan, a8, wrapped = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
side = json.loads(pathlib.Path(str(a8) + ".json").read_text())
meta = side.get("metadata") or {}
assert side.get("npz_sha256") == sha(a8)
assert meta.get("plan_sha256") == sha(plan)
assert int(meta.get("wrapped_layers", -1)) == wrapped
assert int(meta.get("calibration_observations", -1)) == 256
assert int(meta.get("calibration_batch_size", -1)) == 8
print(f"validated existing A8: {a8}")
PY
        return
    fi
    require_gpu_free "$PREP_GPU"
    mkdir -p "$(dirname "$A8")"
    CUDA_VISIBLE_DEVICES="$PREP_GPU" "$OPENPI_PY" "$CALIBRATOR" \
        --plan "$PLAN" \
        --out "$A8" \
        --expected-wrapped "$EXPECTED_WRAPPED" \
        --n-frames 256 \
        --batch-size 8 \
        --device cuda
}

start_one_server() {
    local row="$1" instance config gpu port
    IFS=',' read -r instance config gpu port <<<"$row"
    if server_running "$instance" "$port"; then
        echo "[pi05 W6] reuse server=$instance gpu=$gpu port=$port"
        return
    fi
    if ! gpu_free "$gpu"; then
        echo "refusing to start $instance: GPU $gpu is occupied" >&2
        return 1
    fi
    "$SERVER_MANAGER" start "$config" "$gpu" "$port" "$instance" "$PLAN" "$A8" "$EXPECTED_WRAPPED"
}

start_servers() {
    [[ -f "$A8" && -f "$A8.json" ]] || { echo "missing W6 A8; run prepare" >&2; return 1; }
    mkdir -p "$CONTROL_DIR"
    local row pid failed=0 pids=()
    for row in "${SERVER_ROWS[@]}"; do
        start_one_server "$row" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if [[ "$failed" != 0 ]]; then stop_servers; return 1; fi
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
    local gpu="$PREFLIGHT_GPU" instance="pi05_w6_g${PREFLIGHT_GPU}" port=$((20200 + PREFLIGHT_GPU))
    local directory="$RUN_DIR/preflight" output="$RUN_DIR/preflight/pi05_w6_2task2seed.jsonl"
    local summary="$RUN_DIR/preflight/summary.json" audit="$CONTROL_DIR/$instance.audit.json" hash
    server_running "$instance" "$port" || { echo "preflight server is not running" >&2; return 1; }
    mkdir -p "$directory"
    hash="$(runtime_hash "$audit")"
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
        --resume-dir "$directory" \
        --out "$output"
    "$ROBOCASA_PY" "$PREFLIGHT_AUDITOR" \
        --results "$output" \
        --runtime-audit "$audit" \
        --config "$CONFIG" \
        --tasks OpenDrawer,TurnOnMicrowave \
        --out "$summary" >/dev/null
}

run_concurrency_preflight() {
    local gpu="$PREFLIGHT_GPU" instance="pi05_w6_g${PREFLIGHT_GPU}" port=$((20200 + PREFLIGHT_GPU))
    local directory="$RUN_DIR/preflight/concurrency4" audit="$CONTROL_DIR/$instance.audit.json"
    local hash start elapsed failed=0 index task seed pid
    local specs=("OpenDrawer,2" "OpenDrawer,3" "TurnOnMicrowave,2" "TurnOnMicrowave,3")
    local pids=()
    server_running "$instance" "$port" || { echo "concurrency preflight server is not running" >&2; return 1; }
    mkdir -p "$directory"
    hash="$(runtime_hash "$audit")"
    start="$(date +%s)"
    for index in "${!specs[@]}"; do
        IFS=',' read -r task seed <<<"${specs[$index]}"
        mkdir -p "$directory/client_$index"
        PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
            "$ROBOCASA_PY" "$EVALUATOR" \
            --host 127.0.0.1 \
            --port "$port" \
            --config-id "$CONFIG" \
            --task-set atomic_seen \
            --tasks "$task" \
            --trial-seeds "$seed" \
            --split target \
            --replan-steps 16 \
            --egl-device "$gpu" \
            --expected-server-metadata-sha256 "$hash" \
            --resume-dir "$directory/client_$index" \
            --out "$directory/client_$index/result.jsonl" \
            >"$directory/client_$index/client.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]] || { echo "four-client concurrency preflight failed" >&2; return 1; }
    elapsed=$(( $(date +%s) - start ))
    "$ROBOCASA_PY" - "$directory" "$hash" "$elapsed" <<'PY'
import json, pathlib, sys
root, metadata_sha, elapsed = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rows = []
for path in sorted(root.glob("client_*/result.jsonl")):
    rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
keys = {(row.get("task"), int(row.get("seed", -1))) for row in rows}
expected = {("OpenDrawer", 2), ("OpenDrawer", 3), ("TurnOnMicrowave", 2), ("TurnOnMicrowave", 3)}
assert len(rows) == 4 and keys == expected
assert all(row.get("status") == "complete" for row in rows)
assert all(row.get("server_metadata_sha256") == metadata_sha for row in rows)
summary = {
    "schema_version": 1,
    "kind": "pi05_uniform_w6_four_client_safety_preflight",
    "result_blind": True,
    "clients": 4,
    "completed_episodes": 4,
    "episode_keys": sorted([list(key) for key in keys]),
    "server_metadata_sha256": metadata_sha,
    "wall_seconds": elapsed,
}
temporary = root / ".summary.json.tmp"
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "summary.json")
print(f"validated four-client concurrency preflight: wall_seconds={elapsed}")
PY
}

manifest_args() {
    local row instance config gpu port worker
    for row in "${SERVER_ROWS[@]}"; do
        IFS=',' read -r instance config gpu port <<<"$row"
        printf '%s\0%s\0' --server \
            "$instance,$config,$gpu,$port,$CONTROL_DIR/$instance.runtime.json,$CONTROL_DIR/$instance.audit.json"
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
        --config "$CONFIG"
        --scope all50
        --plan "$PLAN"
        --a8 "$A8"
        --expected-wrapped "$EXPECTED_WRAPPED"
        --orchestrator "$REPO_ROOT/scripts/run_pi05_uniform_w6_official.sh"
        --preflight-results "$RUN_DIR/preflight/pi05_w6_2task2seed.jsonl"
        --preflight-audit "$RUN_DIR/preflight/summary.json"
        --reference "gdsq_vla_selector,$SELECTOR_MANIFEST,gdsq_vla_runtime_selector"
        --reference "static_gdsq,$STATIC_MANIFEST,gdsq_vla"
    )
    while IFS= read -r -d '' item; do args+=("$item"); done < <(manifest_args)
    [[ "$mode" == verify ]] && args+=(--verify)
    "$ROBOCASA_PY" "$MANIFEST_TOOL" "${args[@]}"
}

freeze_manifest() {
    if [[ -f "$MANIFEST" ]]; then run_manifest_tool verify; else run_manifest_tool create; fi
}

start_worker() {
    local worker_id="$1" instance="$2" shard="$3" count="$4" seeds="$5"
    local server_row config gpu port audit hash pid_file log_file pid=""
    server_row="$(server_row_for_instance "$instance")"
    IFS=',' read -r _instance config gpu port <<<"$server_row"
    audit="$CONTROL_DIR/$instance.audit.json"
    pid_file="$WORKER_DIR/$worker_id.pid"
    log_file="$WORKER_DIR/$worker_id.log"
    mkdir -p "$WORKER_DIR" "$RUN_DIR/results/$CONFIG"
    [[ -f "$pid_file" ]] && pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[pi05 W6] reuse worker=$worker_id pid=$pid"
        return
    fi
    if [[ -f "$log_file" ]] && grep -Fq "pi05 week1 worker complete:" "$log_file"; then
        echo "[pi05 W6] completed worker=$worker_id"
        return
    fi
    server_running "$instance" "$port" || { echo "worker server is not running: $instance" >&2; return 1; }
    hash="$(runtime_hash "$audit")"
    nohup setsid "$WORKER_LAUNCHER" \
        "$config" "$port" "$gpu" "$shard" "$count" "$seeds" \
        "$worker_id" "$hash" "$RUN_DIR" all50 >"$log_file" 2>&1 </dev/null &
    echo "$!" >"$pid_file"
    echo "[pi05 W6] worker=$worker_id pid=$! shard=$shard/$count seeds=$seeds"
}

start_workers() {
    [[ -f "$MANIFEST" ]] || { echo "W6 manifest is not frozen" >&2; return 1; }
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
    for pid_file in "$WORKER_DIR"/pi05_w6_*.pid; do
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            command="$(ps -p "$pid" -o args=)"
            if [[ "$command" != *run_pi05_week1_worker.sh* ]]; then
                echo "refusing to stop unrelated pid=$pid command=$command" >&2
                continue
            fi
            process_group="$(ps -p "$pid" -o pgid= | tr -d ' ')"
            if [[ "$process_group" == "$pid" ]]; then kill -- "-$pid"; else kill "$pid"; fi
        fi
    done
}

workers_alive() {
    local count=0 pid_file pid
    shopt -s nullglob
    for pid_file in "$WORKER_DIR"/pi05_w6_*.pid; do
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then count=$((count + 1)); fi
    done
    echo "$count"
}

failed_workers() {
    local row worker _instance _shard _count _seeds pid_file log_file pid=""
    for row in "${WORKER_ROWS[@]}"; do
        IFS=',' read -r worker _instance _shard _count _seeds <<<"$row"
        pid_file="$WORKER_DIR/$worker.pid"; log_file="$WORKER_DIR/$worker.log"
        [[ -f "$pid_file" ]] && pid="$(<"$pid_file")" || pid=""
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then continue; fi
        if [[ -f "$log_file" ]] && grep -Fq "pi05 week1 worker complete:" "$log_file"; then continue; fi
        echo "$worker"
    done
}

progress() {
    "$ROBOCASA_PY" - "$RUN_DIR/results/$CONFIG" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.jsonl")):
    rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
keys = {(row["config"], row["task_set"], row["task"], int(row["seed"])) for row in rows}
if len(keys) != len(rows): raise SystemExit("duplicate committed W6 rows")
if any(row.get("status") != "complete" for row in rows): raise SystemExit("non-complete W6 row")
if any(row.get("runtime_selector_enabled") for row in rows): raise SystemExit("selector enabled in W6 row")
print(f"[pi05 W6] completed={len(rows)}/2500")
PY
}

aggregate() {
    local mode="${1:-formal}" args=() out_dir="$RUN_DIR/aggregate"
    if [[ "$mode" == progress ]]; then args+=(--allow-incomplete); out_dir="$RUN_DIR/aggregate_incomplete"; fi
    PYTHONPATH="$REPO_ROOT/scripts/tools:$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$AGGREGATOR" \
        --run-dir "$RUN_DIR" --out-dir "$out_dir" --bootstrap 10000 "${args[@]}"
}

is_complete() {
    aggregate progress >/dev/null
    "$ROBOCASA_PY" - "$RUN_DIR/aggregate_incomplete/summary.json" <<'PY'
import json, sys
m=json.load(open(sys.argv[1]))
raise SystemExit(0 if m.get("complete") and m.get("completed_episodes") == 2500 else 1)
PY
}

start_monitor() {
    local pid_file="$CONTROL_DIR/gpu_monitor.pi05_w6.pid" pid=""
    [[ -f "$pid_file" ]] && pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then return; fi
    nohup setsid "$ROBOCASA_PY" "$GPU_MONITOR" --run-dir "$RUN_DIR" --interval 30 \
        >"$CONTROL_DIR/gpu_monitor.pi05_w6.log" 2>&1 </dev/null &
    echo "$!" >"$pid_file"
}

run_all() {
    local succeeded=0
    cleanup_on_exit() {
        local rc=$?
        if [[ "$succeeded" != 1 ]]; then
            stop_workers || true
            stop_servers || true
        fi
        return "$rc"
    }
    trap cleanup_on_exit EXIT
    prepare
    start_servers
    run_preflight
    run_concurrency_preflight
    freeze_manifest
    start_workers
    start_monitor
    local failed alive
    while true; do
        progress
        failed="$(failed_workers)"
        if [[ -n "$failed" ]]; then
            echo "[pi05 W6] restarting abnormal workers: $failed" >&2
            start_workers
        fi
        alive="$(workers_alive)"
        echo "[pi05 W6] workers_alive=$alive"
        if [[ "$alive" == 0 ]]; then
            if is_complete; then
                aggregate formal
                stop_servers
                succeeded=1
                trap - EXIT
                echo "[pi05 W6] complete: 2500 attested episodes"
                return
            fi
            echo "W6 workers exited before strict completion" >&2
            return 1
        fi
        sleep 60
    done
}

status() {
    progress
    if [[ ! -f "$MANIFEST" ]]; then
        echo "[pi05 W6] state=not_started (manifest not frozen)"
        return
    fi
    echo "[pi05 W6] workers_alive=$(workers_alive)"
    local failed
    failed="$(failed_workers)"
    [[ -z "$failed" ]] || echo "[pi05 W6] abnormal_workers=$failed"
    "$SERVER_MANAGER" status
}

case "${1:-}" in
    prepare) prepare ;;
    schedule) print_schedule ;;
    start) run_all ;;
    start-servers) start_servers ;;
    preflight) run_preflight ;;
    concurrency-preflight) run_concurrency_preflight ;;
    freeze-manifest) freeze_manifest ;;
    start-workers) start_workers; start_monitor ;;
    aggregate) aggregate formal ;;
    status) status ;;
    stop) stop_workers; stop_servers ;;
    *) usage; exit 2 ;;
esac
