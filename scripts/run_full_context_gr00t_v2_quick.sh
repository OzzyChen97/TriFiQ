#!/usr/bin/env bash
set -euo pipefail

# v2 quick advancement gate: gdsq_main (exact historical main) vs the
# P2-frozen full-context v2 winner on the hash-drawn 2/2/1 tasks with
# seeds 60-69 (50 episodes per config).  Advancement requires every
# ``advance_if`` rule from the frozen quick spec:
#   micro strictly higher, task macro not lower, W > L,
#   static bytes <= 1.10 x QuantVLA Table-1 cell.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
export PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}"
RUNNER="$REPO_ROOT/scripts/tools/run_robocasa_atomic_matrix.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_full_context_quick.py"
SPEC_BUILDER="$REPO_ROOT/scripts/tools/build_v2_quick_execution_specs.py"
SPEC_ROOT="${FULL_CONTEXT_V2_QUICK_SPEC_ROOT:-$REPO_ROOT/runs/full_context_v2/quick/specs}"
QUICK_SPEC="$REPO_ROOT/runs/full_context_v2/quick_spec.json"
QUICK_ROOT="${FULL_CONTEXT_V2_QUICK_ROOT:-$REPO_ROOT/runs/full_context_v2/quick}"
GDSQ_GPU="${FULL_CONTEXT_V2_QUICK_GDSQ_GPU:-5}"
CANDIDATE_GPU="${FULL_CONTEXT_V2_QUICK_CANDIDATE_GPU:-6}"
EGL_POOL="${FULL_CONTEXT_V2_QUICK_EGL_POOL:-5,6}"
SEED_SHARDS="${FULL_CONTEXT_V2_QUICK_SEED_SHARDS:-5}"
SEEDS="60,61,62,63,64,65,66,67,68,69"
FROZEN_WINNER="${FULL_CONTEXT_V2_QUICK_WINNER:-$REPO_ROOT/runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json}"
QUANTVLA_TABLE1_BYTES=963772416

usage() {
    echo "usage: $0 build | preflight | run | status | aggregate" >&2
}

rewrite_execution_placement() {
    local source="$1" target="$2"
    "$PYTHON" - "$source" "$target" "$GDSQ_GPU" "$CANDIDATE_GPU" <<'PY'
import json
import os
import sys
import tempfile

source, target, gdsq_gpu, candidate_gpu = sys.argv[1:]
payload = json.load(open(source, encoding="utf-8"))
placements = {
    "gdsq_main": (int(gdsq_gpu), 19565),
    "full_context_v2": (int(candidate_gpu), 19566),
}
for row in payload["configs"]:
    row["gpu"], row["port"] = placements[row["id"]]
directory = os.path.dirname(target)
os.makedirs(directory, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".execution.", suffix=".json", dir=directory)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}

build() {
    "$PYTHON" "$SPEC_BUILDER" --frozen-winner "$FROZEN_WINNER"
}

preflight() {
    "$PYTHON" - "$SPEC_ROOT" "$FROZEN_WINNER" "$QUANTVLA_TABLE1_BYTES" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

spec_root, winner_path, table1_bytes = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
from quantvla_full_context import require_protocol_attestation
from quantvla_table1_bytes import (
    TABLE1_QUANTVLA_BYTES,
    table1_total_static_budget,
    table1_total_static_bytes,
)

winner = json.loads(winner_path.read_text(encoding="utf-8"))
meta = winner.get("meta") or {}
require_protocol_attestation(meta, source=str(winner_path))
checks = {
    "frozen": meta.get("frozen") is True,
    "model": meta.get("model_adapter") == "gr00t",
    "noise_a": meta.get("selection_noise") == "A",
    "no_selector": meta.get("runtime_selector") is False,
    "no_correction": meta.get("runtime_correction") is False,
    "anchor": table1_bytes == TABLE1_QUANTVLA_BYTES["gr00t"],
    "bytes": table1_total_static_bytes("gr00t", int(winner.get("total_bytes", 0)))
        <= table1_total_static_budget("gr00t"),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"v2 quick preflight failed: {failed}")

for spec_path in sorted(spec_root.glob("*.json")):
    if spec_path.name.startswith("."):
        continue
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("kind") != "full_context_v2_quick_execution_spec":
        raise SystemExit(f"{spec_path}: wrong spec kind")
    rows = {row["id"]: row for row in spec["configs"]}
    if set(rows) != {"gdsq_main", "full_context_v2"}:
        raise SystemExit(f"{spec_path}: config inventory drift")
    if rows["full_context_v2"]["plan"] != str(winner_path):
        raise SystemExit(f"{spec_path}: candidate plan is not the frozen winner")
    if rows["full_context_v2"]["activation_mode"] != "dynamic_a8":
        raise SystemExit(f"{spec_path}: candidate must run dynamic A8")
    if rows["gdsq_main"]["activation_mode"] != "static_a8":
        raise SystemExit(f"{spec_path}: main must run static A8")
print(f"[preflight] v2 quick specs OK ({len(list(spec_root.glob('*.json')))} task sets)")
PY
}

run_wave() {
    local task_set="$1" tasks="$2" checkpoint="$3"
    local frozen_spec="$SPEC_ROOT/$task_set.json"
    local execution_spec="$QUICK_ROOT/specs/.execution-$task_set.json"
    local run_dir="$QUICK_ROOT/$task_set"
    rewrite_execution_placement "$frozen_spec" "$execution_spec"
    "$PYTHON" "$RUNNER" \
        --spec "$execution_spec" \
        --run-dir "$run_dir" \
        --phase dev \
        --seeds "$SEEDS" \
        --checkpoint "$checkpoint" \
        --task-set "$task_set" \
        --tasks "$tasks" \
        --dev-tasks "$tasks" \
        --n-shards "$(awk -F, '{print NF}' <<<"$tasks")" \
        --seed-shards-per-task "$SEED_SHARDS" \
        --egl-device-pool "$EGL_POOL" \
        --trial-batch-size 5 \
        --action-noise paired \
        --allow-shared-gpus
}

run_all() {
    build
    preflight
    TASKS="$("$PYTHON" - "$QUICK_SPEC" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1], encoding="utf-8"))
tasks = spec["tasks"]
for split in ("atomic_seen", "composite_seen", "composite_unseen"):
    print(f"{split}|{','.join(tasks[split])}")
PY
)"
    for row in $TASKS; do
        local task_set="${row%%|*}"
        local tasks="${row#*|}"
        run_wave \
            "$task_set" \
            "$tasks" \
            "$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/$task_set/checkpoint-60000"
    done
    aggregate
}

status() {
    "$PYTHON" - "$QUICK_ROOT" <<'PY'
import json
from collections import Counter
from pathlib import Path
import sys

counts = Counter()
for path in Path(sys.argv[1]).glob("**/*.jsonl"):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "complete":
                counts[str(row.get("config"))] += 1
print(" ".join(f"{name}={counts[name]}/50" for name in ("gdsq_main", "full_context_v2")))
PY
}

aggregate() {
    "$PYTHON" "$AGGREGATOR" \
        --model gr00t \
        --main-dir "$QUICK_ROOT" \
        --candidate-dir "$QUICK_ROOT" \
        --main-config gdsq_main \
        --candidate-config full_context_v2 \
        --candidate-plan "$FROZEN_WINNER" \
        --quantvla-table1-bytes "$QUANTVLA_TABLE1_BYTES" \
        --out "$QUICK_ROOT/aggregate.json"
}

case "${1:-}" in
    build) build ;;
    preflight) preflight ;;
    run) run_all ;;
    status) status ;;
    aggregate) aggregate ;;
    *) usage; exit 2 ;;
esac
