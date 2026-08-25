#!/usr/bin/env bash
set -euo pipefail

# The public formal entry now fail-closes onto the GR00T N1.5-aligned runner.
# Set PI05_ENABLE_LEGACY_PRETRAIN_D10=1 only for historical diagnosis; rows
# produced by that branch are rejected by the current formal aggregator.
if [[ "${PI05_ENABLE_LEGACY_PRETRAIN_D10:-0}" == "0" ]]; then
    aligned_runner="/home1/gyy/vla/QuantVLA/scripts/run_pi05_faithful_waves.sh"
    case "${1:-}" in
        start|run-all) exec "$aligned_runner" run-all ;;
        prepare) exec "$aligned_runner" prepare ;;
        status) exec "$aligned_runner" status ;;
        stop-workers|stop-all|stop) exec "$aligned_runner" stop ;;
        run-wave) exec "$aligned_runner" "$@" ;;
        *)
            echo "usage: $0 prepare | start | run-all | run-wave CONFIG | status | stop" >&2
            exit 2
            ;;
    esac
fi

REPO_ROOT="/home1/gyy/vla/QuantVLA"
RUN_DIR="${PI05_FORMAL_RUN_DIR:-$REPO_ROOT/runs/pi05_gdsq_port/official_pretrain_paired50}"
CONTROL_DIR="$RUN_DIR/control"
export PI05_CONTROL_DIR="$CONTROL_DIR"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
SEEDED_WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
MANIFEST_TOOL="$REPO_ROOT/scripts/tools/pi05_make_formal_manifest.py"
GPU_MONITOR="$REPO_ROOT/scripts/tools/monitor_pi05_formal_gpu.py"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"

usage() {
    echo "usage: $0 start | status | stop-workers | stop-all" >&2
}

if [[ "${PI05_AVOID_GPU2:-0}" != "0" ]]; then
    # Shared-machine layout when GPU2 is occupied by an external process.
    # Keep two instances for the slow full-W4 baseline and one for each other
    # formal configuration; GPU0/GPU7 remain untouched.
    server_rows=(
        "faithful_fp16_g1,fp16,1,18101"
        "faithful_w4_g3,quantvla_w4a8_atmohb,3,18102"
        "faithful_gdsqatm_g4,gdsq_vla_atmohb,4,18103"
        "faithful_gdsq_g5,gdsq_vla,5,18104"
        "faithful_w4_g6,quantvla_w4a8_atmohb,6,18105"
    )
else
    server_rows=(
        "smoke_fp16,fp16,1,18101"
        "smoke_quantvla_w4a8_atmohb,quantvla_w4a8_atmohb,2,18102"
        "smoke_gdsq_vla_atmohb,gdsq_vla_atmohb,3,18103"
        "smoke_gdsq_vla,gdsq_vla,4,18104"
        "official_quantvla_w4a8_atmohb_g5,quantvla_w4a8_atmohb,5,18105"
        "official_gdsq_vla_g6,gdsq_vla,6,18106"
    )
fi

metadata_hash() {
    local runtime_file="$1"
    "$ROBOCASA_PY" - "$runtime_file" <<'PY'
import hashlib, json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(encoded).hexdigest())
PY
}

ensure_server() {
    local instance="$1" config="$2" gpu="$3" port="$4"
    local pid_file="$CONTROL_DIR/$instance.pid"
    local pid=""
    if [[ -f "$pid_file" ]]; then
        pid="$(<"$pid_file")"
    fi
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        local command
        command="$(ps -p "$pid" -o args=)"
        if [[ "$command" == *serve_pi05_quant_policy.py* \
            && -s "$CONTROL_DIR/$instance.runtime.json" ]]; then
            # A synchronous policy inference temporarily blocks the asyncio
            # health handler.  PID ownership plus the frozen runtime
            # attestation is the reliable reuse check while workers are live.
            echo "reuse server instance=$instance pid=$pid port=$port"
            return
        fi
    fi
    "$SERVER_MANAGER" start "$config" "$gpu" "$port" "$instance"
}

create_manifest() {
    local args=()
    local specs=()
    local row instance config gpu port runtime
    for row in "${server_rows[@]}"; do
        IFS=',' read -r instance config gpu port <<<"$row"
        runtime="$CONTROL_DIR/$instance.runtime.json"
        args+=(--server "$instance,$config,$gpu,$port,$runtime")
        specs+=("$instance,$config,$gpu,$port,$runtime")
    done
    if [[ -f "$RUN_DIR/manifest.json" \
        && "${PI05_REQUIRE_FAITHFUL_FINAL:-0}" == "0" ]]; then
        local expected_manifest_hash="0bd05bf31434cc5f0df505bbe99e9e7782364c9ae84dd9d4023b39173ec4b703"
        local actual_manifest_hash
        actual_manifest_hash="$(sha256sum "$RUN_DIR/manifest.json" | cut -d' ' -f1)"
        if [[ "$actual_manifest_hash" != "$expected_manifest_hash" ]]; then
            echo "frozen base manifest hash mismatch: $actual_manifest_hash" >&2
            exit 1
        fi
        "$ROBOCASA_PY" - "$RUN_DIR/manifest.json" "${specs[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


manifest = json.load(open(sys.argv[1], encoding="utf-8"))
# The v3 schedule amendment separately attests the changed evaluator and
# seeded worker.  Model/probe/plan/calibration/server artifacts remain frozen.
schedule_owned = {"evaluator", "worker_launcher"}
for name, artifact in manifest["artifacts"].items():
    if name in schedule_owned:
        continue
    actual = sha256_file(artifact["path"])
    if actual != artifact["sha256"]:
        raise SystemExit(f"frozen artifact hash mismatch for {name}: {actual}")

expected = {row["instance"]: row for row in manifest["servers"]}
for spec in sys.argv[2:]:
    instance, config, gpu, port, runtime_path = spec.split(",", 4)
    runtime = json.load(open(runtime_path, encoding="utf-8"))
    row = expected.get(instance)
    if row is None:
        raise SystemExit(f"server instance absent from frozen manifest: {instance}")
    identity = (row["config_id"], row["gpu"], row["port"])
    if identity != (config, int(gpu), int(port)):
        raise SystemExit(f"server assignment changed for {instance}: {identity}")
    actual = canonical_hash(runtime)
    if actual != row["server_metadata_sha256"]:
        raise SystemExit(f"server runtime changed for {instance}: {actual}")
print("frozen base manifest/model artifacts/server runtimes verified")
PY
        return
    fi
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" "$ROBOCASA_PY" "$MANIFEST_TOOL" \
        --out "$RUN_DIR/manifest.json" "${args[@]}"
}

start_worker() {
    local config="$1" port="$2" gpu="$3" shard="$4" count="$5" seeds="$6"
    local worker_id="$7" runtime="$8"
    local worker_control="$CONTROL_DIR/workers"
    mkdir -p "$worker_control"
    local pid_file="$worker_control/$worker_id.pid"
    local log_file="$worker_control/$worker_id.log"
    local pid=""
    if [[ -f "$pid_file" ]]; then
        pid="$(<"$pid_file")"
    fi
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "reuse worker=$worker_id pid=$pid"
        return
    fi
    local hash
    hash="$(metadata_hash "$runtime")"
    nohup setsid "$SEEDED_WORKER" "$config" "$port" "$gpu" "$shard" "$count" \
        "$seeds" "$worker_id" "$hash" "$RUN_DIR" >"$log_file" 2>&1 </dev/null &
    pid=$!
    echo "$pid" >"$pid_file"
    echo "started worker=$worker_id pid=$pid config=$config shard=$shard/$count seeds=$seeds gpu=$gpu"
}

start_seed_halves() {
    local config="$1" port="$2" gpu="$3" shard="$4" count="$5" prefix="$6" runtime="$7"
    start_worker "$config" "$port" "$gpu" "$shard" "$count" 0-24 \
        "${prefix}_lo" "$runtime"
    start_worker "$config" "$port" "$gpu" "$shard" "$count" 25-49 \
        "${prefix}_hi" "$runtime"
}

start_all() {
    mkdir -p "$CONTROL_DIR"
    local row instance config gpu port
    for row in "${server_rows[@]}"; do
        IFS=',' read -r instance config gpu port <<<"$row"
        ensure_server "$instance" "$config" "$gpu" "$port"
    done
    create_manifest

    if [[ "${PI05_AVOID_GPU2:-0}" != "0" ]]; then
        # Capacity-matched schedule: two clients per eager quant server.  The
        # two full-W4 instances split tasks; each single-instance config uses
        # the two disjoint seed halves.  This avoids the severe queue inflation
        # seen when 12+ clients shared one synchronous policy server.
        start_seed_halves fp16 18101 1 0 1 "faithful_fp16_g1" \
            "$CONTROL_DIR/faithful_fp16_g1.runtime.json"
        start_seed_halves quantvla_w4a8_atmohb 18102 3 0 2 "faithful_w4_g3" \
            "$CONTROL_DIR/faithful_w4_g3.runtime.json"
        start_seed_halves quantvla_w4a8_atmohb 18105 6 1 2 "faithful_w4_g6" \
            "$CONTROL_DIR/faithful_w4_g6.runtime.json"
        start_seed_halves gdsq_vla_atmohb 18103 4 0 1 "faithful_gdsqatm_g4" \
            "$CONTROL_DIR/faithful_gdsqatm_g4.runtime.json"
        start_seed_halves gdsq_vla 18104 5 0 1 "faithful_gdsq_g5" \
            "$CONTROL_DIR/faithful_gdsq_g5.runtime.json"
    else
    # Schedule v3: repartition only the task dimension; 0-24/25-49 remain
    # disjoint.  Each evaluator uses the config-wide read-only resume index,
    # so rows committed by v1/v2 are preserved without duplication.
    local i
    for i in $(seq 0 7); do
        start_seed_halves fp16 18101 1 "$i" 8 "v3_fp16_g1_t$i" \
            "$CONTROL_DIR/smoke_fp16.runtime.json"
    done
    for i in $(seq 0 4); do
        start_seed_halves quantvla_w4a8_atmohb 18102 2 "$i" 10 "v3_w4_g2_t$i" \
            "$CONTROL_DIR/smoke_quantvla_w4a8_atmohb.runtime.json"
    done
    for i in $(seq 5 9); do
        start_seed_halves quantvla_w4a8_atmohb 18105 5 "$i" 10 "v3_w4_g5_t$i" \
            "$CONTROL_DIR/official_quantvla_w4a8_atmohb_g5.runtime.json"
    done
    for i in $(seq 0 5); do
        start_seed_halves gdsq_vla_atmohb 18103 3 "$i" 6 "v3_gdsqatm_g3_t$i" \
            "$CONTROL_DIR/smoke_gdsq_vla_atmohb.runtime.json"
    done
    for i in $(seq 0 5); do
        start_seed_halves gdsq_vla 18104 4 "$i" 12 "v3_gdsq_g4_t$i" \
            "$CONTROL_DIR/smoke_gdsq_vla.runtime.json"
    done
    for i in $(seq 6 11); do
        start_seed_halves gdsq_vla 18106 6 "$i" 12 "v3_gdsq_g6_t$i" \
            "$CONTROL_DIR/official_gdsq_vla_g6.runtime.json"
    done
    fi
    local monitor_pid=""
    if [[ -f "$CONTROL_DIR/gpu_monitor.pid" ]]; then
        monitor_pid="$(<"$CONTROL_DIR/gpu_monitor.pid")"
    fi
    if [[ ! "$monitor_pid" =~ ^[0-9]+$ ]] || ! kill -0 "$monitor_pid" 2>/dev/null; then
        nohup setsid "$ROBOCASA_PY" "$GPU_MONITOR" --run-dir "$RUN_DIR" --interval 30 \
            >"$CONTROL_DIR/gpu_monitor.log" 2>&1 </dev/null &
        monitor_pid=$!
        echo "$monitor_pid" >"$CONTROL_DIR/gpu_monitor.pid"
        echo "started gpu monitor pid=$monitor_pid"
    fi
}

status_all() {
    "$SERVER_MANAGER" status
    local config rows successes failures
    for config in fp16 quantvla_w4a8_atmohb gdsq_vla_atmohb gdsq_vla; do
        read -r rows successes < <(
            "$ROBOCASA_PY" - "$RUN_DIR/results/$config" <<'PY'
import glob, json, pathlib, sys
rows = []
for name in glob.glob(str(pathlib.Path(sys.argv[1]) / "*.jsonl")):
    for line in open(name, encoding="utf-8"):
        if line.strip(): rows.append(json.loads(line))
print(len(rows), sum(bool(row.get("success")) for row in rows))
PY
        )
        failures=$((rows - successes))
        echo "matrix config=$config complete=$rows/2500 successes=$successes failures=$failures"
    done
    local pid_file pid state
    shopt -s nullglob
    for pid_file in "$CONTROL_DIR/workers"/*.pid; do
        pid="$(<"$pid_file")"
        state="stale"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then state="running"; fi
        echo "worker=$(basename "$pid_file" .pid) pid=$pid state=$state"
    done
    if [[ -f "$CONTROL_DIR/gpu_monitor.pid" ]]; then
        pid="$(<"$CONTROL_DIR/gpu_monitor.pid")"
        state="stale"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then state="running"; fi
        echo "gpu_monitor pid=$pid state=$state"
    fi
}

stop_workers() {
    local pid_file pid command
    shopt -s nullglob
    for pid_file in "$CONTROL_DIR/workers"/*.pid; do
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            command="$(ps -p "$pid" -o args=)"
            if [[ "$command" == *run_pi05_formal_worker.sh* \
                || "$command" == *run_pi05_formal_worker_seeded.sh* ]]; then
                kill -- "-$pid" 2>/dev/null || kill "$pid"
                echo "stopped worker=$(basename "$pid_file" .pid) pid=$pid"
            else
                echo "refusing unrelated pid=$pid command=$command" >&2
            fi
        fi
    done
}

stop_all() {
    stop_workers
    local row instance _config _gpu _port
    for row in "${server_rows[@]}"; do
        IFS=',' read -r instance _config _gpu _port <<<"$row"
        "$SERVER_MANAGER" stop "$instance"
    done
}

case "${1:-}" in
    start) start_all ;;
    status) status_all ;;
    stop-workers) stop_workers ;;
    stop-all) stop_all ;;
    *) usage; exit 2 ;;
esac
