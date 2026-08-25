#!/usr/bin/env bash
set -euo pipefail

# Persistent outer supervisor for the immutable parallel schedule.  The
# evaluator/server workers are already nohup process groups; this process only
# restores automatic strict aggregation and committed-key retry if the desktop
# PTY that launched run_pi05_faithful_parallel.sh is reclaimed.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
RUN_DIR="$REPO_ROOT/runs/pi05_gdsq_final/official_pretrain_paired50"
CONTROL_DIR="$RUN_DIR/control"
WORKER_DIR="$CONTROL_DIR/workers"
RUNNER="$REPO_ROOT/scripts/run_pi05_faithful_parallel.sh"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_pi05_robocasa365.py"
COMPLETION_AUDITOR="$REPO_ROOT/scripts/tools/audit_pi05_table1_completion.py"
FINAL_REPORTER="$REPO_ROOT/scripts/tools/render_pi05_table1_report.py"
FINAL_DOC_UPDATER="$REPO_ROOT/scripts/tools/update_pi05_final_doc.py"
FINAL_DOC="$REPO_ROOT/docs/quantvla_v2_pi05_porting_final.md"
RESOURCE_SCHEDULE="$RUN_DIR/manifest.schedule_parallel_v4.json"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"

workers_alive() {
    local count=0 pid_file pid
    shopt -s nullglob
    for pid_file in "$WORKER_DIR"/parallel_*.pid; do
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

schedule_values() {
    local field="$1"
    "$ROBOCASA_PY" - "$RESOURCE_SCHEDULE" "$field" <<'PY'
import json
import sys

schedule = json.load(open(sys.argv[1], encoding="utf-8"))
field = sys.argv[2]
if field == "workers":
    for row in schedule["workers"]:
        print(row["worker_id"])
elif field == "servers":
    for row in schedule["servers"]:
        print(row["instance"])
else:
    raise SystemExit(f"unknown schedule field: {field}")
PY
}

failed_workers() {
    local worker pid_file log_file pid
    while IFS= read -r worker; do
        pid_file="$WORKER_DIR/$worker.pid"
        log_file="$WORKER_DIR/$worker.log"
        if [[ ! -f "$pid_file" ]]; then
            echo "$worker:missing-pid"
            continue
        fi
        pid="$(<"$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            continue
        fi
        if [[ -f "$log_file" ]] && grep -Fq "formal seeded worker complete:" "$log_file"; then
            continue
        fi
        echo "$worker:abnormal-exit"
    done < <(schedule_values workers)
}

failed_servers() {
    local instance pid_file pid command
    while IFS= read -r instance; do
        pid_file="$CONTROL_DIR/$instance.pid"
        if [[ ! -f "$pid_file" ]]; then
            echo "$instance:missing-pid"
            continue
        fi
        pid="$(<"$pid_file")"
        if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
            echo "$instance:not-running"
            continue
        fi
        command="$(ps -p "$pid" -o args=)"
        if [[ "$command" != *serve_pi05_quant_policy.py* ]]; then
            echo "$instance:unexpected-pid-command"
        fi
    done < <(schedule_values servers)
}

progress() {
    "$ROBOCASA_PY" - "$RUN_DIR/results" <<'PY'
import glob
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
parts = []
for config in ("fp16", "quantvla_w4a8_atmohb", "gdsq_vla_atmohb", "gdsq_vla"):
    rows = []
    for name in glob.glob(str(root / config / "*.jsonl")):
        rows.extend(json.loads(line) for line in open(name, encoding="utf-8") if line.strip())
    keys = {(row["config"], row["task_set"], row["task"], row["seed"]) for row in rows}
    if len(keys) != len(rows) or any(row.get("status") != "complete" for row in rows):
        raise SystemExit(f"invalid committed rows: {config}")
    parts.append(f"{config}={len(rows)}/2500")
print("[pi05 persistent] " + " ".join(parts))
PY
}

complete() {
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$AGGREGATOR" --run-dir "$RUN_DIR" \
        --out-dir "$RUN_DIR/aggregate_incomplete" --allow-incomplete >/dev/null
    "$ROBOCASA_PY" - "$RUN_DIR/aggregate_incomplete/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
ok = all(
    row["completed_episodes"] == 2500 and row["missing_episodes"] == 0
    for row in summary["configs"].values()
)
raise SystemExit(0 if ok else 1)
PY
}

while true; do
    failed_server_list="$(failed_servers)"
    failed_worker_list="$(failed_workers)"
    if [[ -n "$failed_server_list" || -n "$failed_worker_list" ]]; then
        echo "[pi05 persistent] abnormal runtime detected"
        [[ -z "$failed_server_list" ]] || echo "$failed_server_list"
        [[ -z "$failed_worker_list" ]] || echo "$failed_worker_list"
        "$RUNNER" stop
        exec "$RUNNER" start
    fi

    alive="$(workers_alive)"
    progress
    echo "[pi05 persistent] workers_alive=$alive"
    if [[ "$alive" != 0 ]]; then
        sleep 60
        continue
    fi

    if complete; then
        PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
            "$ROBOCASA_PY" "$AGGREGATOR" --run-dir "$RUN_DIR" \
            --out-dir "$RUN_DIR/aggregate"
        legacy_audit_status="failed_after_source_correction"
        if PYTHONPATH="$REPO_ROOT/scripts/tools:$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
            "$ROBOCASA_PY" "$COMPLETION_AUDITOR" \
            --run-dir "$RUN_DIR" \
            --schedule "$RESOURCE_SCHEDULE" \
            --out "$RUN_DIR/aggregate/diagnostic_legacy_audit.json" \
            >"$CONTROL_DIR/diagnostic_legacy_audit.log" 2>&1; then
            legacy_audit_status="passed_but_scientifically_invalidated"
        fi
        "$ROBOCASA_PY" - "$RUN_DIR" "$legacy_audit_status" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1]).resolve()
summary = json.loads((run_dir / "aggregate/summary.json").read_text(encoding="utf-8"))
if summary.get("complete") is not True:
    raise SystemExit("diagnostic matrix aggregate is incomplete")
manifest = run_dir / "manifest.json"
payload = {
    "schema_version": 1,
    "status": "diagnostic_predecessor_complete_not_final",
    "episodes": sum(row["completed_episodes"] for row in summary["configs"].values()),
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "legacy_audit_status": sys.argv[2],
    "reason_not_final": (
        "full-W4 baseline used permute=false, synthetic calibration and custom "
        "per-head OHB; no predecessor row may enter corrected Table 1"
    ),
}
out = run_dir / "aggregate/diagnostic_completion.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
        "$RUNNER" stop
        echo "[pi05 persistent] complete 10,000-episode diagnostic predecessor; final report intentionally skipped"
        exit 0
    fi

    echo "[pi05 persistent] incomplete after worker exit; restarting committed-key resume"
    "$RUNNER" stop
    exec "$RUNNER" start
done
