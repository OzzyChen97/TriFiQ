#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
FINISH_WORKER="$REPO_ROOT/scripts/run_pi05_table1_finish_worker.sh"
SCHEDULE_TOOL="$REPO_ROOT/scripts/tools/pi05_make_table1_finish_sprint.py"
RUN_ROOT="$REPO_ROOT/runs/full_context_v2/pi05_table1"
CONTROL_DIR="$RUN_ROOT/control"
WORKER_CONTROL="$CONTROL_DIR/workers"
PREFLIGHT_DIR="$RUN_ROOT/preflight/finish_sprint"
BASE_MANIFEST="$RUN_ROOT/manifest.json"
PARENT_SCHEDULE="$RUN_ROOT/manifest.schedule_gpu3_free_v3.json"
SCHEDULE="$RUN_ROOT/manifest.schedule_finish_sprint_v4.json"
CONFIG_ID="full_context_w4a8_dynamic_profile"
PLAN="$REPO_ROOT/runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json"
HESSIAN="$REPO_ROOT/runs/full_context_v2/pi05_quick/artifacts/hessian_w4.npz"
PACK_DIR="$REPO_ROOT/runs/archive_previous_versions/errorfold_v4_iter/calibration/pi05_full_w4/identity_pack"
BUFFER="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"

SERVER_ROWS=(
    "dypac_pi05_table1_g1,1,19701"
    "dypac_pi05_table1_g2,2,19702"
    "dypac_pi05_table1_g4,4,19704"
    "dypac_pi05_table1_g5,5,19705"
    "dypac_pi05_table1_g6,6,19706"
    "dypac_pi05_table1_g7,7,19707"
    "dypac_pi05_table1_g1_s1,1,19811"
    "dypac_pi05_table1_g4_s1,4,19841"
    "dypac_pi05_table1_g5_s1,5,19851"
    "dypac_pi05_table1_g6_s1,6,19861"
    "dypac_pi05_table1_g7_s1,7,19871"
    "dypac_pi05_table1_g1_s2,1,19812"
    "dypac_pi05_table1_g4_s2,4,19842"
    "dypac_pi05_table1_g5_s2,5,19852"
    "dypac_pi05_table1_g6_s2,6,19862"
    "dypac_pi05_table1_g7_s2,7,19872"
    "dypac_pi05_table1_g3_s0,3,19831"
    "dypac_pi05_table1_g3_s1,3,19832"
    "dypac_pi05_table1_g3_s2,3,19833"
    "dypac_pi05_table1_g1_s3,1,19813"
    "dypac_pi05_table1_g2_s1,2,19821"
    "dypac_pi05_table1_g3_s3,3,19834"
    "dypac_pi05_table1_g4_s3,4,19843"
    "dypac_pi05_table1_g5_s3,5,19853"
    "dypac_pi05_table1_g6_s3,6,19863"
    "dypac_pi05_table1_g7_s3,7,19873"
)

NEW_SERVER_ROWS=(
    "dypac_pi05_table1_g1_s3,1,19813"
    "dypac_pi05_table1_g2_s1,2,19821"
    "dypac_pi05_table1_g3_s3,3,19834"
    "dypac_pi05_table1_g4_s3,4,19843"
    "dypac_pi05_table1_g5_s3,5,19853"
    "dypac_pi05_table1_g6_s3,6,19863"
    "dypac_pi05_table1_g7_s3,7,19873"
)

server_env() {
    env PI05_CONTROL_DIR="$CONTROL_DIR" PI05_FLOW_STEPS=4 \
        PI05_FULL_CONTEXT_PLAN="$PLAN" PI05_FULL_CONTEXT_HESSIAN_W4="$HESSIAN" \
        PI05_FULL_CONTEXT_PACK_DIR="$PACK_DIR" PI05_FULL_CONTEXT_BUFFER="$BUFFER" "$@"
}

start_one() {
    local instance="$1" gpu="$2" port="$3" pid=""
    [[ -f "$CONTROL_DIR/$instance.pid" ]] && pid="$(<"$CONTROL_DIR/$instance.pid")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null \
        && [[ -s "$CONTROL_DIR/$instance.runtime.json" ]] \
        && curl -fsS --max-time 10 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
        echo "reuse server=$instance pid=$pid"
        return
    fi
    server_env "$SERVER_MANAGER" start "$CONFIG_ID" "$gpu" "$port" "$instance" \
        >"$CONTROL_DIR/$instance.finish_start.log" 2>&1
}

start_new_servers() {
    local starts=() instances=() row instance gpu port failed=0 index
    for row in "${NEW_SERVER_ROWS[@]}"; do
        IFS=, read -r instance gpu port <<<"$row"
        start_one "$instance" "$gpu" "$port" &
        starts+=("$!"); instances+=("$instance")
    done
    for index in "${!starts[@]}"; do
        if ! wait "${starts[$index]}"; then
            failed=1
            echo "failed server=${instances[$index]}" >&2
            tail -n 100 "$CONTROL_DIR/${instances[$index]}.finish_start.log" >&2 || true
        fi
    done
    [[ "$failed" == 0 ]]
}

smoke_new_server() {
    local instance="dypac_pi05_table1_g7_s3" runtime hash output
    runtime="$CONTROL_DIR/$instance.runtime.json"
    hash="$($ROBOCASA_PY - "$runtime" <<'PY'
import hashlib,json,sys
x=json.load(open(sys.argv[1],encoding="utf-8"))
print(hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",", ":")).encode()).hexdigest())
PY
)"
    mkdir -p "$PREFLIGHT_DIR"
    output="$PREFLIGHT_DIR/smoke.jsonl"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" --host 127.0.0.1 --port 19873 \
        --config-id "$CONFIG_ID" --task-set atomic_seen --tasks OpenDrawer \
        --trial-seeds 17 --split target --replan-steps 16 --flow-steps 4 \
        --egl-device 7 --expected-server-metadata-sha256 "$hash" --max-steps 1 \
        --resume-dir "$PREFLIGHT_DIR" --out "$output" >"$PREFLIGHT_DIR/smoke.log" 2>&1
    echo "finish-sprint smoke passed"
}

stop_current_workers() {
    local pid_file pid cmdline alive
    for pid_file in "$WORKER_CONTROL"/gpu3free_*.pid; do
        [[ -f "$pid_file" ]] || continue
        pid="$(<"$pid_file")"
        [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || continue
        cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
        [[ "$cmdline" == *"run_pi05_formal_worker_seeded.sh"*"$RUN_ROOT"* ]] || continue
        kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    done
    for _ in $(seq 1 30); do
        alive=0
        for pid_file in "$WORKER_CONTROL"/gpu3free_*.pid; do
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
    echo "current workers did not stop" >&2
    return 1
}

freeze_schedule() {
    local args=() row instance gpu port runtime
    for row in "${SERVER_ROWS[@]}"; do
        IFS=, read -r instance gpu port <<<"$row"
        runtime="$CONTROL_DIR/$instance.runtime.json"
        args+=(--server "$instance,$gpu,$port,$runtime")
    done
    "$ROBOCASA_PY" "$SCHEDULE_TOOL" --base-manifest "$BASE_MANIFEST" \
        --parent-schedule "$PARENT_SCHEDULE" --run-dir "$RUN_ROOT" \
        --scheduler "$REPO_ROOT/scripts/reschedule_pi05_table1_finish_sprint.sh" \
        --out "$SCHEDULE" "${args[@]}"
}

start_finish_workers() {
    local manifest_hash worker_id pid
    manifest_hash="$(sha256sum "$SCHEDULE" | awk '{print $1}')"
    while IFS= read -r worker_id; do
        nohup setsid "$FINISH_WORKER" "$SCHEDULE" "$worker_id" "$manifest_hash" \
            >"$WORKER_CONTROL/$worker_id.log" 2>&1 </dev/null &
        pid=$!
        echo "$pid" >"$WORKER_CONTROL/$worker_id.pid"
        echo "started worker=$worker_id pid=$pid"
    done < <(jq -r '.assignments[].worker_id' "$SCHEDULE")
}

main() {
    mkdir -p "$CONTROL_DIR" "$WORKER_CONTROL" "$PREFLIGHT_DIR"
    [[ -f "$BASE_MANIFEST" && -f "$PARENT_SCHEDULE" && ! -e "$SCHEDULE" ]]
    start_new_servers
    smoke_new_server
    stop_current_workers
    freeze_schedule
    start_finish_workers
}

main "$@"
