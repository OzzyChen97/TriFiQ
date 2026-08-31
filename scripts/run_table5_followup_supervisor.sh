#!/usr/bin/env bash
set -uo pipefail

# Persistent, fail-closed completion supervisor for the remaining LIBERO
# Table-5 cells.  Child failures are retried forever, while exact episode
# validators and resumable atomic queues prevent duplicate or partial cells.

ROOT=/home1/gyy/vla/QuantVLA
DYPAC_ROOT=$ROOT/runs/libero_dypac_v1
TABLE_ROOT=$ROOT/runs/table6_libero_v1
CONTROL=$ROOT/runs/table5_followup/control
STATUS=$CONTROL/status.json
PROGRESS=$CONTROL/progress.log
VALIDATOR=$ROOT/scripts/tools/validate_table6_libero_cell.py
AUDIT_PY=/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python
RETRY_SECONDS=${TABLE5_RETRY_SECONDS:-30}
CLAIM_TIMEOUT_SECONDS=${TABLE5_CLAIM_TIMEOUT_SECONDS:-1800}

mkdir -p "$CONTROL"
cd "$ROOT" || exit 1

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$PROGRESS"
}

phase() {
    local name="$1" detail="${2:-}"
    "$AUDIT_PY" - "$STATUS" "$name" "$detail" "$DYPAC_ROOT" "$TABLE_ROOT" <<'PY'
import datetime
import json
import os
import pathlib
import sys

status = pathlib.Path(sys.argv[1])
phase, detail = sys.argv[2:4]
dypac, table = map(pathlib.Path, sys.argv[4:6])

def exact(path: pathlib.Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("episode_summaries") or []
        keys = [(int(row["task_id"]), int(row["trial_id"])) for row in rows]
        expected = {(task, trial) for task in range(10) for trial in range(10, 20)}
        return (
            int(value.get("total_episodes", -1)) == 100
            and len(keys) == len(set(keys)) == 100
            and set(keys) == expected
            and len(value.get("task_summaries") or []) == 10
            and int(value.get("num_trials_per_task", -1)) == 10
        )
    except Exception:
        return False

suites = ("goal", "spatial", "object", "long")
ours = {
    model: {
        suite: exact(dypac / "results" / model / "dypac_vla" / suite / "merged_summary.json")
        for suite in suites
    }
    for model in ("pi05", "gr00t")
}
baselines = {
    f"{config}/{model}": {
        suite: exact(table / "results" / model / config / suite / "merged_summary.json")
        for suite in suites
    }
    for config in ("uniform_w6", "quantvla_w4a8")
    for model in ("pi05", "gr00t")
}
payload = {
    "schema_version": 1,
    "phase": phase,
    "detail": detail,
    "pid": os.getppid(),
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "order": [
        "pi05/dypac_vla",
        "gr00t/dypac_vla",
        "uniform_w6/pi05",
        "uniform_w6/gr00t",
        "quantvla_w4a8/pi05",
        "quantvla_w4a8/gr00t",
    ],
    "ours": ours,
    "baselines": baselines,
}
temporary = pathlib.Path(str(status) + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(status)
PY
    log "phase=$name detail=$detail"
}

cell_valid() {
    "$AUDIT_PY" "$VALIDATOR" "$1/merged_summary.json" >/dev/null 2>&1
}

ours_cell() {
    echo "$DYPAC_ROOT/results/$1/dypac_vla/$2"
}

baseline_cell() {
    echo "$TABLE_ROOT/results/$1/$2/$3"
}

first_incomplete_ours() {
    local model="$1" suite
    for suite in goal spatial object long; do
        if ! cell_valid "$(ours_cell "$model" "$suite")"; then
            echo "$suite"
            return
        fi
    done
}

all_ours_valid() {
    [[ -z "$(first_incomplete_ours "$1")" ]]
}

pi05_controller_active() {
    pgrep -f '[r]un_libero_dypac_pi05_eval.sh formal' >/dev/null 2>&1
}

gr00t_controller_active() {
    pgrep -f '[r]un_libero_dypac_gr00t_resume.sh' >/dev/null 2>&1 ||
        pgrep -f '[r]un_libero_dypac_gr00t_artifacts.sh' >/dev/null 2>&1 ||
        pgrep -f '[r]un_libero_dypac_gr00t_pipeline.sh' >/dev/null 2>&1 ||
        pgrep -f '[r]un_libero_dypac_gr00t_eval.sh' >/dev/null 2>&1
}

terminate_exact_tree() {
    local root_pid="$1" expected_root="$2" command descendants target
    [[ -r "/proc/$root_pid/cmdline" ]] || return
    command=$(tr '\0' ' ' <"/proc/$root_pid/cmdline" 2>/dev/null || true)
    [[ "$command" == *"--run-shard"* && "$command" == *"$expected_root"* ]] || return
    mapfile -t descendants < <("$AUDIT_PY" - "$root_pid" <<'PY'
import pathlib
import sys

root = int(sys.argv[1])
children = {}
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        lines = (entry / "status").read_text().splitlines()
        ppid = int(next(line.split()[1] for line in lines if line.startswith("PPid:")))
        children.setdefault(ppid, []).append(int(entry.name))
    except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration, ValueError):
        pass

ordered = []
def visit(pid):
    for child in children.get(pid, []):
        visit(child)
    ordered.append(pid)
visit(root)
print("\n".join(map(str, ordered)))
PY
    )
    (( ${#descendants[@]} > 0 )) || return
    log "retiring stalled worker pid=$root_pid root=$expected_root descendants=${descendants[*]}"
    for target in "${descendants[@]}"; do kill -TERM "$target" 2>/dev/null || true; done
}

reap_stale_queue_workers() {
    local model="$1" row pid output_root
    while IFS=$'\t' read -r pid output_root; do
        [[ -n "${pid:-}" && -n "${output_root:-}" ]] || continue
        terminate_exact_tree "$pid" "$output_root"
    done < <("$AUDIT_PY" - "$DYPAC_ROOT" "$model" "$CLAIM_TIMEOUT_SECONDS" <<'PY'
import json
import pathlib
import sys
import time

root = pathlib.Path(sys.argv[1])
model = sys.argv[2]
timeout = float(sys.argv[3])
now = time.time()
for queue_path in (root / "results" / model / "dypac_vla").glob("*/episode_queue.json"):
    try:
        value = json.loads(queue_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        continue
    for row in value.get("running", {}).values():
        pid = int(row.get("pid", -1))
        age = now - float(row.get("claimed_unix", now))
        if pid <= 1 or age <= timeout:
            continue
        try:
            command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "--run-shard" in command and str(queue_path.parent) in command:
            print(pid, queue_path.parent, sep="\t")
PY
    )
}

complete_pi05_ours() {
    local suite attempt=0
    while ! all_ours_valid pi05; do
        suite=$(first_incomplete_ours pi05)
        reap_stale_queue_workers pi05
        if pi05_controller_active; then
            phase wait_pi05_ours "suite=$suite controller_active=1"
            sleep "$RETRY_SECONDS"
            continue
        fi
        attempt=$((attempt + 1))
        phase run_pi05_ours "suite=$suite attempt=$attempt"
        if env \
            GR00T_DYNAMIC_RESUME_QUEUE=1 \
            GR00T_DYNAMIC_RESTART_FAILED_WORKERS=1 \
            GR00T_DYNAMIC_AUTO_JOIN=1 \
            bash "$ROOT/scripts/run_libero_dypac_pi05_eval.sh" formal "$suite"; then
            log "pi0.5 Ours suite=$suite evaluator returned successfully"
        else
            log "pi0.5 Ours suite=$suite failed; retrying after ${RETRY_SECONDS}s"
            sleep "$RETRY_SECONDS"
        fi
    done
    phase pi05_ours_complete "four exact 100-episode cells"
}

complete_gr00t_ours() {
    local suite attempt=0
    while ! all_ours_valid gr00t; do
        suite=$(first_incomplete_ours gr00t)
        reap_stale_queue_workers gr00t
        if gr00t_controller_active; then
            phase wait_gr00t_ours "suite=$suite controller_active=1"
            sleep "$RETRY_SECONDS"
            continue
        fi
        attempt=$((attempt + 1))
        phase run_gr00t_ours "suite=$suite attempt=$attempt"
        if env \
            GR00T_DYNAMIC_RESUME_QUEUE=1 \
            GR00T_DYNAMIC_RESTART_FAILED_WORKERS=1 \
            GR00T_DYNAMIC_AUTO_JOIN=1 \
            bash "$ROOT/scripts/run_libero_dypac_gr00t_resume.sh"; then
            log "GR00T Ours controller returned successfully"
        else
            log "GR00T Ours controller failed; retrying after ${RETRY_SECONDS}s"
            sleep "$RETRY_SECONDS"
        fi
    done
    phase gr00t_ours_complete "four exact 100-episode cells"
}

prepare_baseline_artifacts() {
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        phase prepare_baseline_artifacts "attempt=$attempt"
        if env TABLE6_GPU_UTIL_MAX=100 TABLE6_A8_MIN_FREE_MIB=32000 \
            bash "$ROOT/scripts/run_table6_libero_unattended.sh" prepare-quant; then
            phase baseline_artifacts_complete "16 validated static-A8 artifacts"
            return
        fi
        log "baseline artifact preparation failed; retrying after ${RETRY_SECONDS}s"
        sleep "$RETRY_SECONDS"
    done
}

plan_baseline_pair_slots() {
    # Pack both methods onto the currently free memory on every card.  The
    # planner maximizes occupied memory, requires both methods to receive at
    # least one worker, and keeps their aggregate memory shares balanced when
    # multiple maximum-fill layouts exist.
    local model="$1"
    "$AUDIT_PY" - "$model" <<'PY'
import subprocess
import sys

model = sys.argv[1]
if model == "pi05":
    uniform_mib, quant_mib = 26000, 9500
    uniform_cap, quant_cap = 1, 4
elif model == "gr00t":
    uniform_mib = quant_mib = 17000
    uniform_cap = quant_cap = 2
else:
    raise SystemExit(2)

raw = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
cards = []
for line in raw.splitlines():
    index, free = (int(field.strip()) for field in line.split(","))
    cards.append((index, max(0, free - 2000)))

# (uniform workers, quant workers) -> (occupied MiB, per-card assignments)
states = {(0, 0): (0, [])}
for index, budget in cards:
    options = []
    for uniform in range(uniform_cap + 1):
        for quant in range(quant_cap + 1):
            occupied = uniform * uniform_mib + quant * quant_mib
            if occupied <= budget:
                options.append((uniform, quant, occupied))
    updated = {}
    for (total_uniform, total_quant), (used, assignment) in states.items():
        for uniform, quant, occupied in options:
            key = (total_uniform + uniform, total_quant + quant)
            candidate = (used + occupied, assignment + [(index, uniform, quant)])
            if key not in updated or candidate[0] > updated[key][0]:
                updated[key] = candidate
    states = updated

candidates = []
for (uniform, quant), (used, assignment) in states.items():
    if uniform <= 0 or quant <= 0:
        continue
    uniform_share = uniform * uniform_mib
    quant_share = quant * quant_mib
    # Full memory packing is primary.  For equally full layouts prefer equal
    # memory shares, then more total workers.
    score = (used, -abs(uniform_share - quant_share), uniform + quant)
    candidates.append((score, assignment))
if not candidates:
    raise SystemExit(1)

_, assignment = max(candidates, key=lambda item: item[0])
uniform_gpus = []
quant_gpus = []
for index, uniform, quant in assignment:
    uniform_gpus.extend([str(index)] * uniform)
    quant_gpus.extend([str(index)] * quant)
print(",".join(uniform_gpus) + "\t" + ",".join(quant_gpus))
PY
}

baseline_port_base() {
    local config="$1" model="$2" suite="$3" base offset=0
    [[ "$config" == uniform_w6 ]] && base=26000 || base=28000
    [[ "$model" == gr00t ]] && offset=$((offset + 1000))
    case "$suite" in
        goal) : ;;
        spatial) offset=$((offset + 100)) ;;
        object) offset=$((offset + 200)) ;;
        long) offset=$((offset + 300)) ;;
        *) return 2 ;;
    esac
    echo $((base + offset))
}

run_baseline_cell_once() {
    local model="$1" config="$2" suite="$3" gpu_list="$4" port="$5" auto_join="$6"
    if env \
        TABLE6_GPU_LIST="$gpu_list" TABLE6_GPU_UTIL_MAX=100 \
        TABLE6_DYNAMIC_AUTO_JOIN="$auto_join" \
        TABLE6_DYNAMIC_GPU_POOL="$gpu_list" TABLE6_DYNAMIC_JOIN_POLL_S=10 \
        TABLE6_DYNAMIC_CLAIM_TIMEOUT_S=1800 TABLE6_PORT_BASE="$port" \
        TABLE6_PI05_W6_PROCS_PER_GPU=1 TABLE6_PI05_W4_PROCS_PER_GPU=4 \
        TABLE6_GR00T_W6_PROCS_PER_GPU=2 TABLE6_GR00T_W4_PROCS_PER_GPU=2 \
        bash "$ROOT/scripts/run_table6_libero_quant_subset.sh" \
            run-cell "$model" "$config" "$suite"; then
        log "baseline evaluator returned successfully $config/$model/$suite"
        return 0
    fi
    log "baseline evaluator failed $config/$model/$suite"
    return 1
}

complete_baseline_pair() {
    local model="$1" suite="$2" uniform_final quant_final assignment
    local uniform_gpus quant_gpus uniform_port quant_port attempt=0
    local uniform_pid quant_pid uniform_rc quant_rc remaining
    uniform_final=$(baseline_cell "$model" uniform_w6 "$suite")
    quant_final=$(baseline_cell "$model" quantvla_w4a8 "$suite")
    uniform_port=$(baseline_port_base uniform_w6 "$model" "$suite")
    quant_port=$(baseline_port_base quantvla_w4a8 "$model" "$suite")

    while ! cell_valid "$uniform_final" || ! cell_valid "$quant_final"; do
        attempt=$((attempt + 1))
        if ! cell_valid "$uniform_final" && ! cell_valid "$quant_final"; then
            if ! assignment=$(plan_baseline_pair_slots "$model"); then
                phase wait_baseline_slots "$model/$suite both_methods attempt=$attempt"
                sleep "$RETRY_SECONDS"
                continue
            fi
            IFS=$'\t' read -r uniform_gpus quant_gpus <<<"$assignment"
            if [[ -z "$uniform_gpus" || -z "$quant_gpus" ]]; then
                phase wait_baseline_slots "$model/$suite invalid_assignment attempt=$attempt"
                sleep "$RETRY_SECONDS"
                continue
            fi
            phase run_baseline_pair \
                "$model/$suite attempt=$attempt uniform_gpus=$uniform_gpus quant_gpus=$quant_gpus"
            run_baseline_cell_once "$model" uniform_w6 "$suite" \
                "$uniform_gpus" "$uniform_port" 1 &
            uniform_pid=$!
            run_baseline_cell_once "$model" quantvla_w4a8 "$suite" \
                "$quant_gpus" "$quant_port" 1 &
            quant_pid=$!
            uniform_rc=0; quant_rc=0
            wait "$uniform_pid" || uniform_rc=$?
            wait "$quant_pid" || quant_rc=$?
            log "baseline pair returned $model/$suite uniform_rc=$uniform_rc quant_rc=$quant_rc"
        else
            if cell_valid "$uniform_final"; then
                remaining=quantvla_w4a8
                quant_port=$(baseline_port_base "$remaining" "$model" "$suite")
                phase run_baseline_tail "$remaining/$model/$suite attempt=$attempt all_available_gpus"
                run_baseline_cell_once "$model" "$remaining" "$suite" auto "$quant_port" 1 || true
            else
                remaining=uniform_w6
                uniform_port=$(baseline_port_base "$remaining" "$model" "$suite")
                phase run_baseline_tail "$remaining/$model/$suite attempt=$attempt all_available_gpus"
                run_baseline_cell_once "$model" "$remaining" "$suite" auto "$uniform_port" 1 || true
            fi
        fi

        if ! cell_valid "$uniform_final" || ! cell_valid "$quant_final"; then
            log "baseline pair incomplete $model/$suite; retrying after ${RETRY_SECONDS}s"
            sleep "$RETRY_SECONDS"
        fi
    done
    phase baseline_pair_complete "$model/$suite uniform_w6+quantvla_w4a8 exact-valid"
}

complete_baselines() {
    local model suite
    for model in pi05 gr00t; do
        for suite in goal spatial object long; do
            complete_baseline_pair "$model" "$suite"
        done
    done
}

run_supervisor() {
    exec 9>"$CONTROL/supervisor.lock"
    flock -n 9 || { echo "Table-5 follow-up supervisor is already active" >&2; return 1; }
    printf '%s\n' "$$" >"$CONTROL/supervisor.pid"
    phase starting "persistent exact-resume scheduler"
    complete_pi05_ours
    complete_gr00t_ours
    prepare_baseline_artifacts
    complete_baselines
    touch "$CONTROL/complete"
    phase complete "all 24 requested follow-up cells are exact-valid"
}

case "${1:-run}" in
    run) run_supervisor ;;
    status)
        [[ -f "$STATUS" ]] && cat "$STATUS" || echo "status has not been initialized"
        ;;
    *) echo "usage: $0 [run|status]" >&2; exit 2 ;;
esac
