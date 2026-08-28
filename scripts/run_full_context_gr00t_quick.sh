#!/usr/bin/env bash
set -euo pipefail

# Frozen 5-task x seeds-50..59 advancement gate for the selector-free
# full-context FP16-protection candidate.  The three waves are required because
# RoboCasa365 uses one checkpoint per target task set.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
RUNNER="$REPO_ROOT/scripts/tools/run_robocasa_atomic_matrix.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_full_context_quick.py"
SPEC_ROOT="${FULL_CONTEXT_GR00T_SPEC_ROOT:-$REPO_ROOT/runs/full_context_v1/gr00t/quick/specs}"
# Keep the first, zero-episode protocol-failure attempt immutable.  A repaired
# launch writes a new manifest and result lineage under quick_real.  Shared
# placement is explicitly allowed by the experiment protocol as long as the
# runner's conservative free-memory preflight passes; no process is evicted.
QUICK_ROOT="${FULL_CONTEXT_GR00T_QUICK_ROOT:-$REPO_ROOT/runs/full_context_v1/gr00t/quick_real}"
GDSQ_GPU="${FULL_CONTEXT_GR00T_GDSQ_GPU:-5}"
CANDIDATE_GPU="${FULL_CONTEXT_GR00T_CANDIDATE_GPU:-6}"
EGL_POOL="${FULL_CONTEXT_GR00T_EGL_POOL:-4,7}"
SEED_SHARDS="${FULL_CONTEXT_GR00T_SEED_SHARDS:-5}"
SEEDS="50,51,52,53,54,55,56,57,58,59"
CANDIDATE_PLAN="$REPO_ROOT/runs/full_context_v1/gr00t/round1/gr00t_full_context_round1_frozen.json"
QUANTVLA_TABLE1_BYTES=963772416

usage() {
    echo "usage: $0 run | preflight | status | aggregate" >&2
}

preflight_candidate() {
    PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" - "$CANDIDATE_PLAN" "$QUANTVLA_TABLE1_BYTES" <<'PY'
import json
from pathlib import Path
import sys

from quantvla_full_context import require_protocol_attestation
from quantvla_table1_bytes import (
    TABLE1_QUANTVLA_BYTES,
    table1_total_static_budget,
    table1_total_static_bytes,
)

path = Path(sys.argv[1]).resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
meta = payload.get("meta") or {}
require_protocol_attestation(meta, source=str(path))
table1_bytes = int(sys.argv[2])
static_budget = table1_total_static_budget("gr00t")
static_total = table1_total_static_bytes(
    "gr00t", int(payload.get("total_bytes", static_budget + 1))
)
checks = {
    "frozen": meta.get("frozen") is True,
    "model": meta.get("model_adapter") == "gr00t",
    "noise_a": meta.get("selection_noise") == "A",
    "no_selector": meta.get("runtime_selector") is False,
    "no_correction": meta.get("runtime_correction") is False,
    "anchor": table1_bytes == TABLE1_QUANTVLA_BYTES["gr00t"],
    "bytes": static_total <= static_budget,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"candidate quick preflight failed: {failed}")
PY
}

preflight_deployment_artifacts() {
    PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" - "$SPEC_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from quantvla_cross_model_protocol import PROTOCOL, sha256_file

root = Path(sys.argv[1]).resolve()
spec_files = [path for path in sorted(root.glob("*.json")) if not path.name.startswith(".")]
expected_buffer = PROTOCOL["data"]["calibration_buffer"]["sha256"]
accepted_hessian_protocols = {
    "quantvla-gr00t-pi05-errorfold-v3":
        "f43aa056b5633b555acfecc48af2f2b5eec81ca7e2cf83c8b86f12fde91498e2",
    "quantvla-gr00t-pi05-errorfold-v4":
        "8ab2aa09c72bc881f7275121717081a011e81d3b39937598d619c6a58d5c5aea",
}
for spec_path in spec_files:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in spec["configs"]}
    baseline = rows["gdsq_main"]
    plan = Path(baseline["plan"])
    scale = Path(baseline["act_scale"])
    scale_meta = json.loads(Path(str(scale) + ".meta.json").read_text())
    expected_plan = sha256_file(plan)
    checks = {
        "a8_buffer": scale_meta.get("buffer_sha256") == expected_buffer,
        "a8_source_buffer": scale_meta.get("source_buffer_sha256") == expected_buffer,
        "a8_plan": scale_meta.get("plan_sha256") == expected_plan,
        "a8_wrapped": int(scale_meta.get("wrapped_layers", -1))
            == int(baseline["expected_wrapped"]),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{spec_path}: GDSQ deployment drift: {failed}")

    candidate = rows["full_context_v1"]
    hessian = Path(candidate["hessian_w4"])
    hessian_meta = json.loads(Path(str(hessian) + ".json").read_text())
    protocol_id = hessian_meta.get("protocol_id")
    checks = {
        "hessian_buffer": hessian_meta.get("calibration_buffer_sha256")
            == expected_buffer,
        "hessian_protocol": hessian_meta.get("protocol_sha256")
            == accepted_hessian_protocols.get(protocol_id),
        "hessian_hash": hessian_meta.get("npz_sha256") == sha256_file(hessian),
        "hessian_inventory": len(hessian_meta.get("layer_names") or [])
            == int(candidate["expected_wrapped"]),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{spec_path}: candidate deployment drift: {failed}")
print(f"[preflight] strict deployment artifacts OK for {len(spec_files)} task sets")
PY
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
    "gdsq_main": (int(gdsq_gpu), 19555),
    "full_context_v1": (int(candidate_gpu), 19556),
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
    preflight_candidate
    preflight_deployment_artifacts
    run_wave \
        atomic_seen \
        CloseFridge,OpenDrawer \
        "$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000"
    run_wave \
        composite_seen \
        LoadDishwasher,PrepareCoffee \
        "$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"
    run_wave \
        composite_unseen \
        MakeIceLemonade \
        "$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_unseen/checkpoint-60000"
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
print(" ".join(f"{name}={counts[name]}/50" for name in ("gdsq_main", "full_context_v1")))
PY
}

aggregate() {
    "$PYTHON" "$AGGREGATOR" \
        --model gr00t \
        --main-dir "$QUICK_ROOT" \
        --candidate-dir "$QUICK_ROOT" \
        --main-config gdsq_main \
        --candidate-config full_context_v1 \
        --candidate-plan "$CANDIDATE_PLAN" \
        --quantvla-table1-bytes "$QUANTVLA_TABLE1_BYTES" \
        --out "$QUICK_ROOT/aggregate.json"
}

case "${1:-}" in
    run) run_all ;;
    preflight) preflight_candidate; preflight_deployment_artifacts ;;
    status) status ;;
    aggregate) aggregate ;;
    *) usage; exit 2 ;;
esac
