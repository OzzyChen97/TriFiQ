#!/usr/bin/env bash
set -euo pipefail

# Resource-only amendment: expand the frozen pi0.5 DyPAC-VLA run to every GPU
# that currently has enough headroom for the measured ~12.7 GiB startup peak.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
SCHEDULE_TOOL="$REPO_ROOT/scripts/tools/pi05_make_dypac_table1_schedule.py"

RUN_ROOT="${FULL_CONTEXT_PI05_TABLE1_ROOT:-$REPO_ROOT/runs/full_context_v2/pi05_table1}"
CONTROL_DIR="$RUN_ROOT/control"
WORKER_CONTROL="$CONTROL_DIR/workers"
PREFLIGHT_DIR="$RUN_ROOT/preflight"
BASE_MANIFEST="$RUN_ROOT/manifest.json"
SCHEDULE="$RUN_ROOT/manifest.schedule_all_free_v1.json"
CONFIG_ID="full_context_w4a8_dynamic_profile"
PLAN="$REPO_ROOT/runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json"
HESSIAN="$REPO_ROOT/runs/full_context_v2/pi05_quick/artifacts/hessian_w4.npz"
PACK_DIR="$REPO_ROOT/runs/archive_previous_versions/errorfold_v4_iter/calibration/pi05_full_w4/identity_pack"
BUFFER="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
GPUS="${FULL_CONTEXT_PI05_TABLE1_GPUS:-1,2,4,5,6,7}"
PORTS="${FULL_CONTEXT_PI05_TABLE1_PORTS:-19701,19702,19704,19705,19706,19707}"

metadata_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib, json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

parse_resources() {
    IFS=, read -r -a GPU_ARRAY <<<"$GPUS"
    IFS=, read -r -a PORT_ARRAY <<<"$PORTS"
    [[ "${#GPU_ARRAY[@]}" -eq "${#PORT_ARRAY[@]}" && "${#GPU_ARRAY[@]}" -gt 0 ]]
}

instance_for_gpu() { printf 'dypac_pi05_table1_g%s' "$1"; }

server_env() {
    env PI05_CONTROL_DIR="$CONTROL_DIR" PI05_FLOW_STEPS=4 \
        PI05_FULL_CONTEXT_PLAN="$PLAN" \
        PI05_FULL_CONTEXT_HESSIAN_W4="$HESSIAN" \
        PI05_FULL_CONTEXT_PACK_DIR="$PACK_DIR" \
        PI05_FULL_CONTEXT_BUFFER="$BUFFER" "$@"
}

start_servers() {
    parse_resources
    local starts=() labels=() index gpu port instance pid="" failed=0
    for index in "${!GPU_ARRAY[@]}"; do
        gpu="${GPU_ARRAY[$index]}"; port="${PORT_ARRAY[$index]}"
        instance="$(instance_for_gpu "$gpu")"; pid=""
        [[ -f "$CONTROL_DIR/$instance.pid" ]] && pid="$(<"$CONTROL_DIR/$instance.pid")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null \
            && [[ -s "$CONTROL_DIR/$instance.runtime.json" ]] \
            && curl -fsS --max-time 10 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
            echo "reuse server=$instance pid=$pid"
            continue
        fi
        server_env "$SERVER_MANAGER" start "$CONFIG_ID" "$gpu" "$port" "$instance" \
            >"$CONTROL_DIR/$instance.scale_start.log" 2>&1 &
        starts+=("$!"); labels+=("$instance")
    done
    for index in "${!starts[@]}"; do
        if ! wait "${starts[$index]}"; then
            failed=1
            echo "failed server=${labels[$index]}" >&2
            tail -n 100 "$CONTROL_DIR/${labels[$index]}.scale_start.log" >&2 || true
        fi
    done
    [[ "$failed" == 0 ]]
}

smoke_new_server() {
    local gpu="${GPU_ARRAY[0]}" port="${PORT_ARRAY[0]}" instance runtime hash output
    instance="$(instance_for_gpu "$gpu")"; runtime="$CONTROL_DIR/$instance.runtime.json"
    hash="$(metadata_hash "$runtime")"; output="$PREFLIGHT_DIR/all_free_smoke_${hash:0:12}.jsonl"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" --host 127.0.0.1 --port "$port" \
        --config-id "$CONFIG_ID" --task-set atomic_seen --tasks OpenDrawer \
        --trial-seeds 1 --split target --replan-steps 16 --flow-steps 4 \
        --egl-device "$gpu" --expected-server-metadata-sha256 "$hash" \
        --max-steps 1 --resume-dir "$PREFLIGHT_DIR" --out "$output" \
        >"$PREFLIGHT_DIR/all_free_smoke.log" 2>&1
    echo "new-server smoke passed: $output"
}

stop_old_workers() {
    local pid_file pid
    for pid_file in "$WORKER_CONTROL"/*.pid; do
        [[ -f "$pid_file" ]] || continue
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        fi
    done
    for _ in $(seq 1 30); do
        local alive=0
        for pid_file in "$WORKER_CONTROL"/*.pid; do
            [[ -f "$pid_file" ]] || continue
            pid="$(<"$pid_file")"
            if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then alive=$((alive+1)); fi
        done
        [[ "$alive" == 0 ]] && return
        sleep 1
    done
    echo "workers did not stop within 30 seconds" >&2
    return 1
}

schedule_args() {
    parse_resources
    SCHEDULE_ARGS=(--base-manifest "$BASE_MANIFEST" --run-dir "$RUN_ROOT" \
        --scheduler "$REPO_ROOT/scripts/reschedule_full_context_pi05_table1_all_free.sh" \
        --out "$SCHEDULE")
    local count="${#GPU_ARRAY[@]}" index gpu port instance runtime suffix seeds worker_id
    for index in "${!GPU_ARRAY[@]}"; do
        gpu="${GPU_ARRAY[$index]}"; port="${PORT_ARRAY[$index]}"
        instance="$(instance_for_gpu "$gpu")"; runtime="$CONTROL_DIR/$instance.runtime.json"
        SCHEDULE_ARGS+=(--server "$instance,$gpu,$port,$runtime")
        for suffix in lo hi; do
            [[ "$suffix" == lo ]] && seeds=0-24 || seeds=25-49
            worker_id="allfree_g${gpu}_shard${index}_${suffix}"
            SCHEDULE_ARGS+=(--worker "$worker_id,$instance,$gpu,$port,$index,$count,$seeds")
        done
    done
}

freeze_schedule() {
    schedule_args
    if [[ -f "$SCHEDULE" ]]; then
        "$ROBOCASA_PY" "$SCHEDULE_TOOL" "${SCHEDULE_ARGS[@]}" --verify
    else
        "$ROBOCASA_PY" "$SCHEDULE_TOOL" "${SCHEDULE_ARGS[@]}"
    fi
}

start_workers() {
    parse_resources
    local count="${#GPU_ARRAY[@]}" index gpu port instance runtime hash suffix seeds worker_id pid
    for index in "${!GPU_ARRAY[@]}"; do
        gpu="${GPU_ARRAY[$index]}"; port="${PORT_ARRAY[$index]}"
        instance="$(instance_for_gpu "$gpu")"; runtime="$CONTROL_DIR/$instance.runtime.json"
        hash="$(metadata_hash "$runtime")"
        for suffix in lo hi; do
            [[ "$suffix" == lo ]] && seeds=0-24 || seeds=25-49
            worker_id="allfree_g${gpu}_shard${index}_${suffix}"
            nohup setsid env PI05_FLOW_STEPS=4 "$WORKER" "$CONFIG_ID" "$port" "$gpu" \
                "$index" "$count" "$seeds" "$worker_id" "$hash" "$RUN_ROOT" \
                >"$WORKER_CONTROL/$worker_id.log" 2>&1 </dev/null &
            pid=$!; echo "$pid" >"$WORKER_CONTROL/$worker_id.pid"
            echo "started worker=$worker_id pid=$pid seeds=$seeds shard=$index/$count"
        done
    done
}

main() {
    mkdir -p "$CONTROL_DIR" "$WORKER_CONTROL" "$PREFLIGHT_DIR"
    [[ -f "$BASE_MANIFEST" ]] || { echo "missing base manifest" >&2; exit 1; }
    start_servers
    smoke_new_server
    stop_old_workers
    freeze_schedule
    start_workers
    FULL_CONTEXT_PI05_TABLE1_GPUS="$GPUS" FULL_CONTEXT_PI05_TABLE1_PORTS="$PORTS" \
        "$REPO_ROOT/scripts/run_full_context_pi05_table1.sh" status
}

main "$@"
