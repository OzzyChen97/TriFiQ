#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
SCHEDULE_TOOL="$REPO_ROOT/scripts/tools/pi05_make_dypac_table1_schedule_v2.py"

RUN_ROOT="$REPO_ROOT/runs/full_context_v2/pi05_table1"
CONTROL_DIR="$RUN_ROOT/control"
WORKER_CONTROL="$CONTROL_DIR/workers"
PREFLIGHT_DIR="$RUN_ROOT/preflight"
BASE_MANIFEST="$RUN_ROOT/manifest.json"
PARENT_SCHEDULE="$RUN_ROOT/manifest.schedule_max_free_vram_v2.json"
SCHEDULE="$RUN_ROOT/manifest.schedule_gpu3_free_v3.json"
CONFIG_ID="full_context_w4a8_dynamic_profile"
PLAN="$REPO_ROOT/runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json"
HESSIAN="$REPO_ROOT/runs/full_context_v2/pi05_quick/artifacts/hessian_w4.npz"
PACK_DIR="$REPO_ROOT/runs/archive_previous_versions/errorfold_v4_iter/calibration/pi05_full_w4/identity_pack"
BUFFER="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"

# Immutable task-shard order. Shards 6, 8, and 10 are moved from GPUs with
# external compute load to the newly free GPU 3. Coverage remains exactly 2500.
SERVER_ROWS=(
    "g1_base,1,19701,dypac_pi05_table1_g1"
    "g2_base,2,19702,dypac_pi05_table1_g2"
    "g4_base,4,19704,dypac_pi05_table1_g4"
    "g5_base,5,19705,dypac_pi05_table1_g5"
    "g6_base,6,19706,dypac_pi05_table1_g6"
    "g7_base,7,19707,dypac_pi05_table1_g7"
    "g3_s0,3,19831,dypac_pi05_table1_g3_s0"
    "g4_s1,4,19841,dypac_pi05_table1_g4_s1"
    "g3_s1,3,19832,dypac_pi05_table1_g3_s1"
    "g6_s1,6,19861,dypac_pi05_table1_g6_s1"
    "g3_s2,3,19833,dypac_pi05_table1_g3_s2"
    "g1_s2,1,19812,dypac_pi05_table1_g1_s2"
    "g4_s2,4,19842,dypac_pi05_table1_g4_s2"
    "g5_s2,5,19852,dypac_pi05_table1_g5_s2"
    "g6_s2,6,19862,dypac_pi05_table1_g6_s2"
    "g7_s2,7,19872,dypac_pi05_table1_g7_s2"
)

metadata_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib,json,sys
x=json.load(open(sys.argv[1],encoding="utf-8"))
print(hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",", ":")).encode()).hexdigest())
PY
}

server_env() {
    env PI05_CONTROL_DIR="$CONTROL_DIR" PI05_FLOW_STEPS=4 \
        PI05_FULL_CONTEXT_PLAN="$PLAN" PI05_FULL_CONTEXT_HESSIAN_W4="$HESSIAN" \
        PI05_FULL_CONTEXT_PACK_DIR="$PACK_DIR" PI05_FULL_CONTEXT_BUFFER="$BUFFER" "$@"
}

start_one() {
    local port="$1" instance="$2" pid=""
    [[ -f "$CONTROL_DIR/$instance.pid" ]] && pid="$(<"$CONTROL_DIR/$instance.pid")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null \
        && [[ -s "$CONTROL_DIR/$instance.runtime.json" ]] \
        && curl -fsS --max-time 10 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
        echo "reuse server=$instance pid=$pid"
        return
    fi
    server_env "$SERVER_MANAGER" start "$CONFIG_ID" 3 "$port" "$instance" \
        >"$CONTROL_DIR/$instance.gpu3_start.log" 2>&1
}

start_gpu3_servers() {
    local port instance
    for row in "19831,dypac_pi05_table1_g3_s0" \
               "19832,dypac_pi05_table1_g3_s1" \
               "19833,dypac_pi05_table1_g3_s2"; do
        IFS=, read -r port instance <<<"$row"
        if ! start_one "$port" "$instance"; then
            echo "failed server=$instance" >&2
            tail -n 100 "$CONTROL_DIR/$instance.gpu3_start.log" >&2 || true
            return 1
        fi
    done
}

smoke_gpu3() {
    local instance="dypac_pi05_table1_g3_s0" port=19831 runtime hash output
    runtime="$CONTROL_DIR/$instance.runtime.json"
    hash="$(metadata_hash "$runtime")"
    output="$PREFLIGHT_DIR/gpu3_smoke_${hash:0:12}.jsonl"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" --host 127.0.0.1 --port "$port" \
        --config-id "$CONFIG_ID" --task-set atomic_seen --tasks OpenDrawer \
        --trial-seeds 2 --split target --replan-steps 16 --flow-steps 4 \
        --egl-device 3 --expected-server-metadata-sha256 "$hash" \
        --max-steps 1 --resume-dir "$PREFLIGHT_DIR" --out "$output" \
        >"$PREFLIGHT_DIR/gpu3_smoke.log" 2>&1
    echo "GPU3 smoke passed: $output"
}

stop_workers() {
    local pid_file pid cmdline alive
    for pid_file in "$WORKER_CONTROL"/*.pid; do
        [[ -f "$pid_file" ]] || continue
        pid="$(<"$pid_file")"
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        [[ -r "/proc/$pid/cmdline" ]] || continue
        cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
        [[ "$cmdline" == *"run_pi05_formal_worker_seeded.sh"*"$RUN_ROOT"* ]] || continue
        kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    done
    for _ in $(seq 1 30); do
        alive=0
        for pid_file in "$WORKER_CONTROL"/*.pid; do
            [[ -f "$pid_file" ]] || continue
            pid="$(<"$pid_file")"
            [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || continue
            cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
            [[ "$cmdline" == *"run_pi05_formal_worker_seeded.sh"*"$RUN_ROOT"* ]] \
                && alive=$((alive+1))
        done
        [[ "$alive" == 0 ]] && return
        sleep 1
    done
    echo "workers did not stop within 30 seconds" >&2
    return 1
}

build_schedule_args() {
    SCHEDULE_ARGS=(--base-manifest "$BASE_MANIFEST" --parent-schedule "$PARENT_SCHEDULE" \
        --run-dir "$RUN_ROOT" --scheduler "$REPO_ROOT/scripts/reschedule_full_context_pi05_table1_gpu3_free.sh" \
        --out "$SCHEDULE")
    local index row label gpu port instance runtime suffix seeds worker_id
    for index in "${!SERVER_ROWS[@]}"; do
        row="${SERVER_ROWS[$index]}"
        IFS=, read -r label gpu port instance <<<"$row"
        runtime="$CONTROL_DIR/$instance.runtime.json"
        SCHEDULE_ARGS+=(--server "$instance,$gpu,$port,$runtime")
        for suffix in lo hi; do
            [[ "$suffix" == lo ]] && seeds=0-24 || seeds=25-49
            worker_id="gpu3free_${label}_${suffix}"
            SCHEDULE_ARGS+=(--worker "$worker_id,$instance,$gpu,$index,16,$seeds")
        done
    done
}

freeze_schedule() {
    build_schedule_args
    if [[ -f "$SCHEDULE" ]]; then
        "$ROBOCASA_PY" "$SCHEDULE_TOOL" "${SCHEDULE_ARGS[@]}" --verify
    else
        "$ROBOCASA_PY" "$SCHEDULE_TOOL" "${SCHEDULE_ARGS[@]}"
    fi
}

start_workers() {
    local index row label gpu port instance runtime hash suffix seeds worker_id pid
    for index in "${!SERVER_ROWS[@]}"; do
        row="${SERVER_ROWS[$index]}"
        IFS=, read -r label gpu port instance <<<"$row"
        runtime="$CONTROL_DIR/$instance.runtime.json"
        hash="$(metadata_hash "$runtime")"
        for suffix in lo hi; do
            [[ "$suffix" == lo ]] && seeds=0-24 || seeds=25-49
            worker_id="gpu3free_${label}_${suffix}"
            nohup setsid env PI05_FLOW_STEPS=4 "$WORKER" "$CONFIG_ID" "$port" "$gpu" \
                "$index" 16 "$seeds" "$worker_id" "$hash" "$RUN_ROOT" \
                >"$WORKER_CONTROL/$worker_id.log" 2>&1 </dev/null &
            pid=$!
            echo "$pid" >"$WORKER_CONTROL/$worker_id.pid"
            echo "started worker=$worker_id pid=$pid shard=$index/16 seeds=$seeds"
        done
    done
}

main() {
    mkdir -p "$CONTROL_DIR" "$WORKER_CONTROL" "$PREFLIGHT_DIR"
    [[ -f "$BASE_MANIFEST" && -f "$PARENT_SCHEDULE" ]]
    start_gpu3_servers
    smoke_gpu3
    stop_workers
    freeze_schedule
    start_workers
}

main "$@"
