#!/usr/bin/env bash
set -euo pipefail

# ATM-vs-OHB component ablation for the frozen pi0.5 GDSQ-VLA mask.
#
# Runs two wave configs on the atomic_seen task group only (18 tasks x 50
# paired seeds = 900 episodes per config):
#   gdsq_vla_atm_only : static ATM enabled, OHB disabled
#   gdsq_vla_ohb_only : static OHB enabled, ATM disabled
#
# Artifacts are exactly the ones used by the completed 10,000-episode
# Table-1 matrix in runs/pi05_gdsq_gr00t_aligned/official_target_paired50
# (d4 target-split protocol), so the new rows are directly paired (same
# deterministic per-(task,seed,replan) noise) with the existing gdsq_vla and
# gdsq_vla_atmohb atomic rows.
#
# Wave 1 (atm_only) uses GPUs 1-5 while the GR00T ablation holds GPUs 6-7;
# wave 2 (ohb_only) expands to GPUs 1-7 after GPUs 6-7 are idle.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
RUN_DIR="$REPO_ROOT/runs/pi05_atmohb_ablation_atomic"
CONTROL_DIR="$RUN_DIR/control"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
SEEDED_WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"

ATM_ONLY_GPUS="1 2 3 4 5"
OHB_ONLY_GPUS="1 2 3 4 5 6 7"
PORTS_BASE=18501

export PI05_CONTROL_DIR="$CONTROL_DIR"
export PI05_PACK_DIR="$REPO_ROOT/runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
export PI05_CALIBRATION_BUFFER="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
export PI05_GDSQ_PLAN="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/plans/pi05_gdsq_cscka_16to1_d4.final_plan.json"
export PI05_GDSQ_A8="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz"
export PI05_GDSQ_ATM="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json"
export PI05_TASK_SETS="atomic_seen"

usage() {
    echo "usage: $0 prepare | run-all | run-wave CONFIG | status | stop" >&2
}

instance_name() {
    local config="$1" gpu="$2"
    echo "ab_${config}_g${gpu}"
}

runtime_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

config_gpus() {
    case "$1" in
        gdsq_vla_atm_only) echo "$ATM_ONLY_GPUS" ;;
        gdsq_vla_ohb_only) echo "$OHB_ONLY_GPUS" ;;
        *) echo "unknown config: $1" >&2; exit 2 ;;
    esac
}

start_servers() {
    local config="$1"
    local gpus
    gpus="$(config_gpus "$config")"
    local pids=() gpu port index instance
    index=0
    for gpu in $gpus; do
        port=$((PORTS_BASE + index))
        instance="$(instance_name "$config" "$gpu")"
        "$SERVER_MANAGER" start "$config" "$gpu" "$port" "$instance" &
        pids+=("$!")
        index=$((index + 1))
    done
    local failed=0 pid
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if [[ "$failed" != 0 ]]; then
        echo "[pi05 ablation] one or more $config servers failed startup" >&2
        return 1
    fi
}

stop_servers() {
    local config="$1"
    local gpus
    gpus="$(config_gpus "$config")"
    local pids=() gpu instance
    for gpu in $gpus; do
        instance="$(instance_name "$config" "$gpu")"
        "$SERVER_MANAGER" stop "$instance" &
        pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
}

wait_for_gpus_idle() {
    local wanted="$1" poll_seconds="${2:-60}"
    while true; do
        if "$ROBOCASA_PY" - "$wanted" <<'PY'
import subprocess
import sys

wanted = {int(x) for x in sys.argv[1].split()}
rows = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
    text=True, capture_output=True,
).stdout.splitlines()
uuid_to_index = {}
for row in rows:
    index, uuid = [x.strip() for x in row.split(",", 1)]
    uuid_to_index[uuid] = int(index)
procs = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
    text=True, capture_output=True,
).stdout.splitlines()
busy = {uuid_to_index[u.strip()] for u in procs if u.strip() in uuid_to_index}
raise SystemExit(0 if not (busy & wanted) else 1)
PY
        then
            return
        fi
        echo "[pi05 ablation] waiting for GPUs $wanted to become idle" >&2
        sleep "$poll_seconds"
    done
}

preflight_one() {
    local config="$1" gpu="$2" port="$3" instance="$4"
    local wrapped
    wrapped="$("$ROBOCASA_PY" - "$PI05_GDSQ_PLAN" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
layers = payload.get("layers") or {}
selected = sum(
    not bool(row.get("skip", not int(row.get("bits", 0) or 0)))
    and int(row.get("bits", 0) or 0) > 0
    for row in layers.values()
)
if selected <= 0:
    raise SystemExit("quant plan selects zero layers")
print(selected)
PY
)"
    "$SERVER_MANAGER" start "$config" "$gpu" "$port" "$instance"
    local runtime="$CONTROL_DIR/$instance.runtime.json"
    "$ROBOCASA_PY" - "$runtime" "$config" "$wrapped" <<'PY'
import json
import sys

runtime_path, config, wrapped_text = sys.argv[1:4]
payload = json.load(open(runtime_path, encoding="utf-8"))
runtime = payload["openpi_runtime"]
if runtime.get("config_id") != config:
    raise SystemExit(f"runtime config mismatch: {runtime.get('config_id')!r}")
wrapped = int((runtime.get("duquant") or {}).get("wrapped_layers", 0))
if wrapped != int(wrapped_text):
    raise SystemExit(f"unexpected wrapped layers: {wrapped} != {wrapped_text}")
scaling = runtime.get("atm_ohb") or {}
if not scaling.get("enabled"):
    raise SystemExit("ATM/OHB scaling is not enabled")
expect_atm = config.endswith("atm_only")
expect_ohb = config.endswith("ohb_only")
if bool(scaling.get("atm_enabled")) != expect_atm:
    raise SystemExit(f"atm_enabled mismatch: {scaling}")
if bool(scaling.get("ohb_enabled")) != expect_ohb:
    raise SystemExit(f"ohb_enabled mismatch: {scaling}")
print(f"preflight {config} OK: wrapped={wrapped} atm={scaling.get('atm_enabled')} ohb={scaling.get('ohb_enabled')}")
PY
    "$SERVER_MANAGER" stop "$instance"
}

preflight() {
    mkdir -p "$CONTROL_DIR"
    preflight_one gdsq_vla_atm_only 1 18601 ab_pre_atm_g1
    preflight_one gdsq_vla_ohb_only 2 18602 ab_pre_ohb_g2
}

write_manifest() {
    if [[ -f "$RUN_DIR/manifest.json" ]]; then
        echo "[pi05 ablation] manifest already frozen: $RUN_DIR/manifest.json"
        return
    fi
    PYTHONPATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" - "$RUN_DIR/manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import robocasa  # noqa: F401
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

from openpi_client.paired_noise import PROTOCOL as NOISE_PROTOCOL

out = Path(sys.argv[1])
repo = Path("/home1/gyy/vla/QuantVLA")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def artifact(rel: str) -> dict:
    path = (repo / rel).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}

plan_path = (repo / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_gdsq_cscka_16to1_d4.final_plan.json").resolve()
plan = json.loads(plan_path.read_text(encoding="utf-8"))
wrapped = sum(
    not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) > 0
    for row in plan["layers"].values()
)
payload = {
    "kind": "pi05_gdsq_atmohb_component_ablation",
    "schema_version": 1,
    "immutable": True,
    "purpose": (
        "ATM-vs-OHB ablation on the frozen pi0.5 GDSQ-VLA mask. "
        "atomic_seen only; 50 paired scenarios per task; rows are paired with "
        "runs/pi05_gdsq_gr00t_aligned/official_target_paired50 via the shared "
        "deterministic (task,seed,replan) action-noise protocol."
    ),
    "base_run_dir": str(repo / "runs/pi05_gdsq_gr00t_aligned/official_target_paired50"),
    "benchmark": "RoboCasa365",
    "task_sets": {"atomic_seen": list(TASK_SET_REGISTRY["atomic_seen"])},
    "trial_seeds": list(range(50)),
    "protocol": {
        "split": "target",
        "n_action_steps": 16,
        "replan_steps": 16,
        "flow_steps": 4,
        "action_horizon": 50,
        "state_dim": 16,
        "action_dim": 12,
        "fresh_environment_per_trial": True,
        "render_enabled": True,
        "official_task_horizon": True,
        "paired_action_noise": True,
        "paired_action_noise_protocol": NOISE_PROTOCOL,
    },
    "configs": {
        "gdsq_vla_atm_only": {"wrapped_layers": wrapped, "atm": True, "ohb": False},
        "gdsq_vla_ohb_only": {"wrapped_layers": wrapped, "atm": False, "ohb": True},
    },
    "artifacts": {
        "checkpoint": artifact("checkpoints/robocasa/pi05_pretrain_human300_pytorch/model.safetensors"),
        "checkpoint_config": artifact("checkpoints/robocasa/pi05_pretrain_human300_pytorch/config.json"),
        "pack_manifest": artifact("runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015/manifest.json"),
        "calibration_buffer": artifact("runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"),
        "gdsq_plan": artifact("runs/pi05_gdsq_gr00t_aligned/plans/pi05_gdsq_cscka_16to1_d4.final_plan.json"),
        "gdsq_a8": artifact("runs/pi05_gdsq_gr00t_aligned/a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz"),
        "gdsq_a8_sidecar": artifact("runs/pi05_gdsq_gr00t_aligned/a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz.json"),
        "gdsq_atm_ohb": artifact("runs/pi05_gdsq_gr00t_aligned/atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json"),
        "server_launcher": artifact("scripts/run_pi05_formal_server.sh"),
        "worker_launcher": artifact("scripts/run_pi05_formal_worker_seeded.sh"),
        "evaluator": artifact("scripts/run_robocasa365_pi05_eval.py"),
    },
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"manifest frozen: {out}")
PY
}

record_servers() {
    local config="$1"
    local gpus
    gpus="$(config_gpus "$config")"
    "$ROBOCASA_PY" - "$CONTROL_DIR/manifest.$config.servers.json" "$config" "$gpus" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
config = sys.argv[2]
gpus = sys.argv[3].split()
rows = []
for gpu in gpus:
    instance = f"ab_{config}_g{gpu}"
    runtime = Path(f"/home1/gyy/vla/QuantVLA/runs/pi05_atmohb_ablation_atomic/control/{instance}.runtime.json")
    raw = runtime.read_bytes()
    parsed = json.loads(runtime.read_text(encoding="utf-8"))
    rows.append({
        "instance": instance,
        "config_id": config,
        "gpu": int(gpu),
        "runtime_file": str(runtime),
        "runtime_file_sha256": hashlib.sha256(raw).hexdigest(),
        "server_metadata_sha256": hashlib.sha256(
            json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    })
out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"server records written: {out}")
PY
}

config_completed() {
    local config="$1"
    "$ROBOCASA_PY" - "$RUN_DIR" "$config" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
config = sys.argv[2]
root = run_dir / "results" / config
if not root.is_dir():
    raise SystemExit(1)
rows = {}
for path in sorted(root.glob("*.jsonl")):
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "complete":
            raise SystemExit(1)
        if row.get("config") != config or row.get("task_set") != "atomic_seen":
            raise SystemExit(1)
        key = (str(row["task"]), int(row["seed"]))
        if key in rows:
            raise SystemExit(1)
        rows[key] = row
import robocasa  # noqa: E402
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY  # noqa: E402

tasks = list(TASK_SET_REGISTRY["atomic_seen"])
expected = {(task, seed) for task in tasks for seed in range(50)}
missing = expected - set(rows)
if missing:
    print(f"{config}: {len(missing)} missing episodes, e.g. {sorted(missing)[:5]}")
    raise SystemExit(1)
successes = sum(bool(rows[key]["success"]) for key in expected)
print(f"{config}: 900/900 episodes, {successes} successes")
PY
}

start_workers() {
    local config="$1"
    local gpus
    gpus="$(config_gpus "$config")"
    local count=0
    local gpu
    for gpu in $gpus; do count=$((count + 1)); done
    local worker_dir="$CONTROL_DIR/workers"
    mkdir -p "$worker_dir" "$RUN_DIR/results/$config"
    local index=0 gpu port instance runtime prefix hash
    for gpu in $gpus; do
        port=$((PORTS_BASE + index))
        instance="$(instance_name "$config" "$gpu")"
        runtime="$CONTROL_DIR/$instance.runtime.json"
        prefix="${instance}_t${index}"
        hash="$(runtime_hash "$runtime")"
        nohup setsid "$SEEDED_WORKER" \
            "$config" "$port" "$gpu" "$index" "$count" 0-24 "${prefix}_lo" "$hash" "$RUN_DIR" \
            >"$worker_dir/${prefix}_lo.log" 2>&1 </dev/null &
        echo "$!" >"$worker_dir/${prefix}_lo.pid"
        nohup setsid "$SEEDED_WORKER" \
            "$config" "$port" "$gpu" "$index" "$count" 25-49 "${prefix}_hi" "$hash" "$RUN_DIR" \
            >"$worker_dir/${prefix}_hi.log" 2>&1 </dev/null &
        echo "$!" >"$worker_dir/${prefix}_hi.pid"
        echo "[pi05 ablation] worker ${prefix}_lo and ${prefix}_hi on gpu=$gpu port=$port shard=$index/$count"
        index=$((index + 1))
    done
}

wait_workers() {
    local config="$1"
    local prefix="ab_${config}_"
    local pid_file pid alive
    while true; do
        alive=0
        shopt -s nullglob
        for pid_file in "$CONTROL_DIR/workers"/${prefix}*.pid; do
            pid="$(<"$pid_file")"
            if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
                alive=$((alive + 1))
            fi
        done
        echo "[pi05 ablation] config=$config workers_alive=$alive"
        [[ "$alive" == 0 ]] && return
        sleep 60
    done
}

stop_workers() {
    local pid_file pid command
    shopt -s nullglob
    for pid_file in "$CONTROL_DIR/workers"/*.pid; do
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            command="$(ps -p "$pid" -o args=)"
            if [[ "$command" == *run_pi05_formal_worker_seeded.sh* ]]; then
                kill -- "-$pid" 2>/dev/null || kill "$pid"
            fi
        fi
        rm -f "$pid_file"
    done
}

run_wave() {
    local config="$1"
    if config_completed "$config"; then
        echo "[pi05 ablation] config already complete: $config"
        return
    fi
    local attempt
    for attempt in 1 2 3; do
        echo "[pi05 ablation] config=$config attempt=$attempt"
        start_servers "$config"
        record_servers "$config"
        start_workers "$config"
        wait_workers "$config"
        if config_completed "$config"; then
            stop_servers "$config"
            echo "[pi05 ablation] config complete: $config"
            return
        fi
        echo "[pi05 ablation] config incomplete after attempt=$attempt; committed-key resume" >&2
        stop_workers
        stop_servers "$config"
    done
    stop_servers "$config"
    echo "[pi05 ablation] config failed strict completion: $config" >&2
    exit 1
}

run_all() {
    mkdir -p "$CONTROL_DIR"
    exec 9>"$CONTROL_DIR/run_all.lock"
    if ! flock -n 9; then
        echo "another ablation runner holds $CONTROL_DIR/run_all.lock" >&2
        exit 1
    fi
    preflight
    write_manifest
    run_wave gdsq_vla_atm_only
    wait_for_gpus_idle "6 7" 120
    run_wave gdsq_vla_ohb_only
    echo "[pi05 ablation] both waves complete: $RUN_DIR"
}

status() {
    local config
    for config in gdsq_vla_atm_only gdsq_vla_ohb_only; do
        config_completed "$config" || true
    done
    "$SERVER_MANAGER" status
}

stop() {
    stop_workers || true
    local config
    for config in gdsq_vla_atm_only gdsq_vla_ohb_only; do
        stop_servers "$config" || true
    done
}

case "${1:-}" in
    prepare) preflight; write_manifest ;;
    run-all) run_all ;;
    run-wave) [[ $# == 2 ]] || { usage; exit 2; }; run_wave "$2" ;;
    status) status ;;
    stop) stop ;;
    *) usage; exit 2 ;;
esac