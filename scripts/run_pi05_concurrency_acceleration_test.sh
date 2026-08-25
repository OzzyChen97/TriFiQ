#!/usr/bin/env bash
set -euo pipefail

# Result-blind 2-vs-4 client closed-loop throughput test on an isolated GPU.
# It clones the frozen pi0.5 selector runtime but never writes into a formal
# run directory and never uses rollout success for a decision.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROOT="$REPO_ROOT/runs/gdsq_week1_preregistered_v1/diagnostics/pi05_concurrency_acceleration_v1"
CONTROL="$ROOT/control"
LOG="$ROOT/orchestrator.log"
PID_FILE="$ROOT/orchestrator.pid"
SUMMARY="$ROOT/summary.json"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
SELECTOR="$REPO_ROOT/runs/atmohb_dynamic_selector_v8/selector.json"
CONFIG="gdsq_vla_runtime_selector"
INSTANCE="accel_selector_g2"
GPU=2
PORT=20702
TMUX_SESSION="pi05_concurrency_acceleration_v1"
GPU_SAMPLER_PID=""

usage() {
    echo "usage: $0 start | run | status | stop" >&2
}

live_pid() {
    local pid=""
    [[ -f "$PID_FILE" ]] && pid="$(<"$PID_FILE")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

server_running() {
    local pid_file="$CONTROL/$INSTANCE.pid" pid=""
    [[ -f "$pid_file" ]] && pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    curl -fsS --max-time 2 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1
}

gpu_free() {
    local output
    output="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    [[ -z "${output//[[:space:]]/}" ]]
}

runtime_hash() {
    "$ROBOCASA_PY" - "$CONTROL/$INSTANCE.runtime.json" <<'PY'
import hashlib, json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

start_server() {
    if server_running; then return; fi
    gpu_free || { echo "GPU $GPU is occupied; refusing acceleration diagnostic" >&2; return 1; }
    mkdir -p "$CONTROL"
    PI05_CONTROL_DIR="$CONTROL" \
        bash "$SERVER_MANAGER" start "$CONFIG" "$GPU" "$PORT" "$INSTANCE"
}

stop_server() {
    PI05_CONTROL_DIR="$CONTROL" bash "$SERVER_MANAGER" stop "$INSTANCE" || true
}

start_gpu_sampler() {
    local mode="$1" output="$ROOT/$mode/gpu_efficiency.jsonl"
    "$ROBOCASA_PY" - "$GPU" "$output" <<'PY' &
import csv, json, pathlib, subprocess, sys, time
gpu, output = int(sys.argv[1]), pathlib.Path(sys.argv[2])
query = [
    "nvidia-smi", "-i", str(gpu),
    "--query-gpu=memory.used,utilization.gpu,power.draw",
    "--format=csv,noheader,nounits",
]
while True:
    row = next(csv.reader([subprocess.check_output(query, text=True).strip()]))
    value = {
        "timestamp": time.time(),
        "gpu": gpu,
        "memory_used_mib": float(row[0].strip()),
        "utilization_gpu_pct": float(row[1].strip()),
        "power_draw_w": float(row[2].strip()),
    }
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
    time.sleep(2)
PY
    GPU_SAMPLER_PID="$!"
}

run_client() {
    local mode="$1" client="$2" tasks="$3" seeds="$4" hash="$5"
    local directory="$ROOT/$mode/$client"
    mkdir -p "$directory"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --config-id "$CONFIG" \
        --task-set atomic_seen \
        --tasks "$tasks" \
        --trial-seeds "$seeds" \
        --split target \
        --replan-steps 16 \
        --egl-device "$GPU" \
        --expected-server-metadata-sha256 "$hash" \
        --expect-runtime-selector \
        --resume-dir "$directory" \
        --out "$directory/results.jsonl" \
        >"$directory/client.log" 2>&1
}

run_mode() {
    local mode="$1" hash="$2" start elapsed sampler_pid failed=0 pid
    local pids=()
    mkdir -p "$ROOT/$mode"
    start="$(date +%s)"
    start_gpu_sampler "$mode"
    sampler_pid="$GPU_SAMPLER_PID"
    if [[ "$mode" == "clients2" ]]; then
        run_client "$mode" c0 "OpenDrawer,TurnOnMicrowave" 40-41 "$hash" & pids+=("$!")
        run_client "$mode" c1 "CloseToasterOvenDoor,TurnOnElectricKettle" 40-41 "$hash" & pids+=("$!")
    elif [[ "$mode" == "clients4" ]]; then
        run_client "$mode" c0 OpenDrawer 40-41 "$hash" & pids+=("$!")
        run_client "$mode" c1 TurnOnMicrowave 40-41 "$hash" & pids+=("$!")
        run_client "$mode" c2 CloseToasterOvenDoor 40-41 "$hash" & pids+=("$!")
        run_client "$mode" c3 TurnOnElectricKettle 40-41 "$hash" & pids+=("$!")
    else
        echo "unknown mode: $mode" >&2; return 2
    fi
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    [[ "$failed" == 0 ]] || { echo "$mode client failed" >&2; return 1; }
    elapsed=$(( $(date +%s) - start ))
    printf '%s\n' "$elapsed" >"$ROOT/$mode/wall_seconds.txt"
    echo "completed mode=$mode wall_seconds=$elapsed"
}

write_summary() {
    local hash="$1"
    "$ROBOCASA_PY" - "$ROOT" "$SUMMARY" "$hash" "$SELECTOR" <<'PY'
import hashlib, json, pathlib, statistics, sys
root, output, metadata_sha, selector_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4])
selector_sha = hashlib.sha256(selector_path.read_bytes()).hexdigest()
expected = {
    (task, seed)
    for task in ("OpenDrawer", "TurnOnMicrowave", "CloseToasterOvenDoor", "TurnOnElectricKettle")
    for seed in (40, 41)
}
modes = {}
for mode, clients in (("clients2", 2), ("clients4", 4)):
    rows = []
    for path in sorted((root / mode).glob("c*/results.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    keys = {(row["task"], int(row["seed"])) for row in rows}
    assert len(rows) == 8 and keys == expected
    assert all(row.get("status") == "complete" for row in rows)
    assert all(row.get("runtime_selector_enabled") is True for row in rows)
    assert all(row.get("selected_variant") == "ohb" for row in rows)
    assert all(row.get("selector_sha256") == selector_sha for row in rows)
    assert all(row.get("server_metadata_sha256") == metadata_sha for row in rows)
    wall = int((root / mode / "wall_seconds.txt").read_text())
    gpu_rows = [json.loads(line) for line in (root / mode / "gpu_efficiency.jsonl").read_text().splitlines() if line.strip()]
    modes[mode] = {
        "clients": clients,
        "completed_episodes": len(rows),
        "wall_seconds": wall,
        "throughput_episodes_per_hour": 3600.0 * len(rows) / wall,
        "episode_wall_seconds_mean": statistics.mean(float(row["episode_wall_seconds"]) for row in rows),
        "environment_step_seconds_mean": statistics.mean(float(row["env_step_seconds"]) for row in rows),
        "inference_seconds_mean": statistics.mean(float(row["inference_seconds"]) for row in rows),
        "gpu_utilization_mean_pct": statistics.mean(float(row["utilization_gpu_pct"]) for row in gpu_rows),
        "gpu_utilization_max_pct": max(float(row["utilization_gpu_pct"]) for row in gpu_rows),
        "gpu_samples": len(gpu_rows),
    }
summary = {
    "schema_version": 1,
    "kind": "pi05_result_blind_concurrency_acceleration_test",
    "formal_result": False,
    "outcomes_used": False,
    "gpu": 2,
    "config_id": "gdsq_vla_runtime_selector",
    "selector_sha256": selector_sha,
    "server_metadata_sha256": metadata_sha,
    "episode_keyset_sha256": hashlib.sha256(json.dumps(sorted(expected), separators=(",", ":")).encode()).hexdigest(),
    "modes": modes,
    "throughput_speedup_clients4_vs_clients2": modes["clients4"]["throughput_episodes_per_hour"] / modes["clients2"]["throughput_episodes_per_hour"],
    "acceptance_rule": "Retain the pending four-client schedules only when all protocol attestations pass and throughput speedup exceeds 1.10; rollout success is never read.",
}
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

run_all() {
    mkdir -p "$ROOT"
    echo "$$" >"$PID_FILE"
    if [[ -f "$SUMMARY" ]]; then
        echo "acceleration test already complete: $SUMMARY"
        return
    fi
    local succeeded=0 hash
    cleanup() {
        local rc=$?
        stop_server
        [[ "$succeeded" == 1 ]] || echo "acceleration test failed rc=$rc" >&2
        return "$rc"
    }
    trap cleanup EXIT
    start_server
    hash="$(runtime_hash)"
    run_client warmup c0 TurnOnMicrowave 49 "$hash"
    run_mode clients2 "$hash"
    run_mode clients4 "$hash"
    write_summary "$hash"
    succeeded=1
    trap - EXIT
    stop_server
    echo "acceleration test complete: $SUMMARY"
}

start_detached() {
    mkdir -p "$ROOT"
    if [[ -f "$SUMMARY" ]]; then
        echo "acceleration test already complete: $SUMMARY"
        return
    fi
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null || live_pid; then
        echo "acceleration test already running: pid=$(<"$PID_FILE")"
        return
    fi
    gpu_free || { echo "GPU $GPU is occupied; refusing acceleration diagnostic" >&2; return 1; }
    tmux new-session -d -s "$TMUX_SESSION" "bash $0 run >$LOG 2>&1"
    sleep 1
    echo "acceleration test started: session=$TMUX_SESSION log=$LOG"
}

status_run() {
    local pid="none" state="stopped"
    [[ -f "$PID_FILE" ]] && pid="$(<"$PID_FILE")"
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null || live_pid; then state="running"; fi
    [[ -f "$SUMMARY" ]] && state="complete"
    echo "pi05_acceleration_test pid=$pid state=$state summary=$([[ -f "$SUMMARY" ]] && echo yes || echo no)"
    [[ -f "$LOG" ]] && tail -n 12 "$LOG"
}

stop_run() {
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then tmux kill-session -t "$TMUX_SESSION"; fi
    stop_server
    echo "acceleration test stopped"
}

cd "$REPO_ROOT"
case "${1:-}" in
    start) start_detached ;;
    run) run_all ;;
    status) status_run ;;
    stop) stop_run ;;
    *) usage; exit 2 ;;
esac
