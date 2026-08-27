#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROOT="$REPO_ROOT/runs/gdsq_extension_preregistered_v1/omega_qvla_pi05_robocasa365_v1"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
BUILDER="$REPO_ROOT/scripts/tools/omega_qvla_pi05_robocasa365_build.py"
AUDITOR="$REPO_ROOT/scripts/tools/omega_qvla_pi05_robocasa365_eval.py"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
WORKER_LAUNCHER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
CONFIG="omega_qvla_w4a4"
GPU_LAYOUT=(1 2 4 5 6 7)

usage() {
    echo "usage: $0 start | run-all | build-packs | run-task TASK_SET | aggregate | status | stop" >&2
}

port_base() {
    case "$1" in
        atomic_seen) echo 20910 ;;
        composite_seen) echo 20920 ;;
        composite_unseen) echo 20930 ;;
        *) echo "unknown task set: $1" >&2; return 2 ;;
    esac
}

pack_for() { echo "$ROOT/packs/$1/omega_qvla_w4a4.pt"; }
calibration_for() { echo "$ROOT/calibration/$1.manifest.v3.json"; }
run_dir_for() { echo "$ROOT/results/$1"; }
control_dir_for() { echo "$(run_dir_for "$1")/control"; }

runtime_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib, json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

build_one() {
    local task_set="$1" stage="$2" gpu="$3"
    local log="$ROOT/logs/build_${task_set}_${stage}.log"
    mkdir -p "$ROOT/logs"
    {
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pi05 Omega build $task_set/$stage gpu=$gpu"
        CUDA_VISIBLE_DEVICES="$gpu" "$OPENPI_PY" "$BUILDER" build \
            --task-set "$task_set" --stage "$stage"
    } >>"$log" 2>&1
}

build_packs() {
    local task_set
    for task_set in atomic_seen composite_seen composite_unseen; do
        "$OPENPI_PY" "$BUILDER" prepare --task-set "$task_set"
    done
    local jobs=(
        "atomic_seen,paligemma,1" "atomic_seen,expert,2"
        "composite_seen,paligemma,4" "composite_seen,expert,5"
        "composite_unseen,paligemma,6" "composite_unseen,expert,7"
    )
    local job task stage gpu pid failed=0
    local pids=()
    for job in "${jobs[@]}"; do
        IFS=',' read -r task stage gpu <<<"$job"
        build_one "$task" "$stage" "$gpu" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]] || { echo "one or more pi05 Omega build stages failed" >&2; return 1; }
    for task_set in atomic_seen composite_seen composite_unseen; do
        "$OPENPI_PY" "$BUILDER" merge --task-set "$task_set"
        "$OPENPI_PY" "$BUILDER" verify --task-set "$task_set"
    done
}

instance_for() { echo "omega_pi05_${1}_g${2}"; }

server_running() {
    local task_set="$1" gpu="$2" port="$3"
    local control instance pid_file pid=""
    control="$(control_dir_for "$task_set")"
    instance="$(instance_for "$task_set" "$gpu")"
    pid_file="$control/$instance.pid"
    [[ -f "$pid_file" ]] && pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    curl -fsS --max-time 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 || return 1
    [[ -s "$control/$instance.runtime.json" ]]
}

start_one_server() {
    local task_set="$1" gpu="$2" port="$3"
    local control instance pack calibration
    control="$(control_dir_for "$task_set")"
    instance="$(instance_for "$task_set" "$gpu")"
    pack="$(pack_for "$task_set")"
    calibration="$(calibration_for "$task_set")"
    if server_running "$task_set" "$gpu" "$port"; then
        echo "[pi05 Omega] reuse server=$instance gpu=$gpu port=$port"
        return
    fi
    PI05_CONTROL_DIR="$control" \
    PI05_OMEGA_PACK="$pack" \
    PI05_OMEGA_CALIBRATION_MANIFEST="$calibration" \
        "$SERVER_MANAGER" start "$CONFIG" "$gpu" "$port" "$instance"
}

start_servers() {
    local task_set="$1" base index gpu pid failed=0
    base="$(port_base "$task_set")"
    mkdir -p "$(control_dir_for "$task_set")"
    local pids=()
    for index in "${!GPU_LAYOUT[@]}"; do
        gpu="${GPU_LAYOUT[$index]}"
        start_one_server "$task_set" "$gpu" "$((base + index))" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]] || { stop_servers "$task_set"; return 1; }
}

stop_servers() {
    local task_set="$1" control gpu instance pid pids=()
    control="$(control_dir_for "$task_set")"
    for gpu in "${GPU_LAYOUT[@]}"; do
        instance="$(instance_for "$task_set" "$gpu")"
        PI05_CONTROL_DIR="$control" "$SERVER_MANAGER" stop "$instance" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
}

preflight() {
    local task_set="$1" run_dir control base gpu=1 instance runtime hash tasks output
    run_dir="$(run_dir_for "$task_set")"
    control="$(control_dir_for "$task_set")"
    base="$(port_base "$task_set")"
    instance="$(instance_for "$task_set" "$gpu")"
    runtime="$control/$instance.runtime.json"
    hash="$(runtime_hash "$runtime")"
    case "$task_set" in
        atomic_seen) tasks="CloseBlenderLid,CloseFridge" ;;
        composite_seen) tasks="DeliverStraw,GetToastedBread" ;;
        composite_unseen) tasks="ArrangeBreadBasket,ArrangeTea" ;;
    esac
    output="$run_dir/preflight/${task_set}_2task2seed.jsonl"
    mkdir -p "$(dirname "$output")"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" \
        --host 127.0.0.1 --port "$base" --config-id "$CONFIG" \
        --task-set "$task_set" --tasks "$tasks" --task-shard-index 0 \
        --task-shard-count 1 --trial-seeds 0-1 --split target \
        --replan-steps 16 --egl-device "$gpu" \
        --expected-server-metadata-sha256 "$hash" \
        --resume-dir "$(dirname "$output")" --out "$output"
    "$ROBOCASA_PY" - "$output" "$task_set" "$hash" <<'PY'
import json, pathlib, sys
path, task_set, server_hash = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
keys={(row.get("task"), row.get("seed")) for row in rows}
checks=[len(rows)==4, len(keys)==4]
for row in rows:
    checks += [row.get("status")=="complete", row.get("task_set")==task_set,
               row.get("server_metadata_sha256")==server_hash,
               row.get("flow_steps")==4, row.get("replan_steps")==16,
               row.get("paired_action_noise") is True]
if not all(checks): raise SystemExit("strict preflight failed")
print(f"strict preflight complete: {task_set} 4/4")
PY
}

freeze_manifest() {
    local task_set="$1" run_dir control preflight_file
    run_dir="$(run_dir_for "$task_set")"
    control="$(control_dir_for "$task_set")"
    preflight_file="$run_dir/preflight/${task_set}_2task2seed.jsonl"
    "$ROBOCASA_PY" "$AUDITOR" freeze --task-set "$task_set" \
        --run-dir "$run_dir" --control-dir "$control" \
        --pack "$(pack_for "$task_set")" \
        --calibration "$(calibration_for "$task_set")" \
        --preflight "$preflight_file"
}

worker_running() {
    local pid_file="$1" pid=""
    [[ -f "$pid_file" ]] && pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

start_worker() {
    local task_set="$1" shard="$2" half="$3" seeds="$4"
    local run_dir control workers gpu base port instance runtime hash worker pid_file log
    run_dir="$(run_dir_for "$task_set")"
    control="$(control_dir_for "$task_set")"
    workers="$control/workers"
    gpu="${GPU_LAYOUT[$shard]}"
    base="$(port_base "$task_set")"
    port="$((base + shard))"
    instance="$(instance_for "$task_set" "$gpu")"
    runtime="$control/$instance.runtime.json"
    hash="$(runtime_hash "$runtime")"
    worker="${task_set}_g${gpu}_${half}"
    pid_file="$workers/$worker.pid"
    log="$workers/$worker.log"
    mkdir -p "$workers" "$run_dir/results/$CONFIG"
    if worker_running "$pid_file"; then return; fi
    if [[ -f "$log" ]] && grep -Fq "formal seeded worker complete:" "$log"; then return; fi
    (
        export PI05_TASK_SETS="$task_set"
        exec nohup setsid "$WORKER_LAUNCHER" \
            "$CONFIG" "$port" "$gpu" "$shard" 6 "$seeds" \
            "$worker" "$hash" "$run_dir"
    ) >"$log" 2>&1 </dev/null &
    echo "$!" >"$pid_file"
    echo "[pi05 Omega] worker=$worker pid=$! shard=$shard/6 seeds=$seeds"
}

start_workers() {
    local task_set="$1" shard
    [[ -f "$(run_dir_for "$task_set")/manifest.json" ]] || {
        echo "formal manifest is not frozen: $task_set" >&2; return 1;
    }
    for shard in 0 1 2 3 4 5; do
        start_worker "$task_set" "$shard" lo 0-24
        start_worker "$task_set" "$shard" hi 25-49
    done
}

task_complete() {
    local task_set="$1" run_dir
    run_dir="$(run_dir_for "$task_set")"
    "$ROBOCASA_PY" "$AUDITOR" progress --run-dir "$run_dir" 2>/dev/null | \
        "$ROBOCASA_PY" -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("complete") else 1)'
}

wait_for_task() {
    local task_set="$1" run_dir snapshot
    run_dir="$(run_dir_for "$task_set")"
    while ! task_complete "$task_set"; do
        start_workers "$task_set"
        snapshot="$($ROBOCASA_PY "$AUDITOR" progress --run-dir "$run_dir" || true)"
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $snapshot"
        sleep 30
    done
    "$ROBOCASA_PY" "$AUDITOR" progress --run-dir "$run_dir" --strict
}

run_task() {
    local task_set="$1"
    "$OPENPI_PY" "$BUILDER" verify --task-set "$task_set"
    start_servers "$task_set"
    preflight "$task_set"
    freeze_manifest "$task_set"
    start_workers "$task_set"
    wait_for_task "$task_set"
    stop_servers "$task_set"
}

aggregate() {
    "$ROBOCASA_PY" "$AUDITOR" aggregate --root "$ROOT" --bootstrap 10000 \
        >"$ROOT/aggregate.log"
    tail -n 80 "$ROOT/aggregate.log"
}

run_all() {
    echo "build_packs" >"$ROOT/phase"
    build_packs
    local task_set
    for task_set in atomic_seen composite_seen composite_unseen; do
        echo "formal:$task_set" >"$ROOT/phase"
        run_task "$task_set"
    done
    echo "aggregate" >"$ROOT/phase"
    aggregate
    echo "complete" >"$ROOT/phase"
}

start_queue() {
    mkdir -p "$ROOT/logs"
    local pid=""
    [[ -f "$ROOT/queue.pid" ]] && pid="$(<"$ROOT/queue.pid")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "pi05 Omega queue already running pid=$pid"
        return
    fi
    nohup setsid "$0" run-all >"$ROOT/logs/queue.log" 2>&1 </dev/null &
    echo "$!" >"$ROOT/queue.pid"
    echo "started pi05 Omega queue pid=$! log=$ROOT/logs/queue.log"
}

status() {
    local phase="not-started" pid="" state="stopped" task_set pack run_dir
    [[ -f "$ROOT/phase" ]] && phase="$(<"$ROOT/phase")"
    [[ -f "$ROOT/queue.pid" ]] && pid="$(<"$ROOT/queue.pid")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && state="running"
    echo "queue=$state pid=${pid:-none} phase=$phase"
    for task_set in atomic_seen composite_seen composite_unseen; do
        pack="$(pack_for "$task_set")"
        run_dir="$(run_dir_for "$task_set")"
        if [[ -f "$pack" ]]; then
            echo "$task_set pack=ready bytes=$(stat -c %s "$pack")"
        else
            echo "$task_set pack=pending stages=$(find "$ROOT/packs/$task_set" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)"
        fi
        if [[ -f "$run_dir/manifest.json" ]]; then
            "$ROBOCASA_PY" "$AUDITOR" progress --run-dir "$run_dir" || true
        else
            echo "$task_set formal=not-frozen"
        fi
    done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
}

stop_all() {
    local task_set pid="" pid_file worker_pid command pgid
    for task_set in atomic_seen composite_seen composite_unseen; do
        shopt -s nullglob
        for pid_file in "$(control_dir_for "$task_set")"/workers/*.pid; do
            worker_pid="$(<"$pid_file")"
            if [[ "$worker_pid" =~ ^[0-9]+$ ]] && kill -0 "$worker_pid" 2>/dev/null; then
                command="$(ps -p "$worker_pid" -o args=)"
                if [[ "$command" == *run_pi05_formal_worker_seeded.sh* ]]; then
                    pgid="$(ps -p "$worker_pid" -o pgid= | tr -d ' ')"
                    [[ "$pgid" == "$worker_pid" ]] && kill -- "-$worker_pid" || kill "$worker_pid"
                fi
            fi
        done
        stop_servers "$task_set" || true
    done
    [[ -f "$ROOT/queue.pid" ]] && pid="$(<"$ROOT/queue.pid")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        command="$(ps -p "$pid" -o args=)"
        if [[ "$command" == *run_omega_qvla_pi05_robocasa365.sh* ]]; then
            pgid="$(ps -p "$pid" -o pgid= | tr -d ' ')"
            [[ "$pgid" == "$pid" ]] && kill -- "-$pid" || kill "$pid"
        fi
    fi
}

cd "$REPO_ROOT"
case "${1:-}" in
    start) start_queue ;;
    run-all) run_all ;;
    build-packs) build_packs ;;
    run-task) [[ $# -eq 2 ]] || { usage; exit 2; }; run_task "$2" ;;
    aggregate) aggregate ;;
    status) status ;;
    stop) stop_all ;;
    *) usage; exit 2 ;;
esac
