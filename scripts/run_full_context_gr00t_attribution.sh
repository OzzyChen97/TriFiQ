#!/usr/bin/env bash
set -euo pipefail

# Four-config GR00T attribution (H / M / C / C16) over the already-consumed
# quick tasks and seeds 50-59, marked as development diagnostics.  H and C
# replay the frozen quick configs; M runs the historical main mask on the
# common Hessian/rotation=0/dynamic-A8 runtime; C16 runs the candidate mask
# with FP16 activations.  These runs are not eligible for Table 1.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
RUNNER="$REPO_ROOT/scripts/tools/run_robocasa_atomic_matrix.py"
SPEC_BUILDER="$REPO_ROOT/scripts/tools/build_gr00t_four_config_attribution_specs.py"
INTERPRETER="$REPO_ROOT/scripts/tools/interpret_four_config_attribution.py"
SPEC_ROOT="${FULL_CONTEXT_GR00T_ATTR_SPEC_ROOT:-$REPO_ROOT/runs/full_context_v1/gr00t/attribution/specs}"
ATTR_ROOT="${FULL_CONTEXT_GR00T_ATTR_ROOT:-$REPO_ROOT/runs/full_context_v1/gr00t/attribution}"
EGL_POOL="${FULL_CONTEXT_GR00T_EGL_POOL:-4,7}"
SEED_SHARDS="${FULL_CONTEXT_GR00T_SEED_SHARDS:-5}"
SEEDS="50,51,52,53,54,55,56,57,58,59"
H_GPU="${FULL_CONTEXT_GR00T_H_GPU:-4}"
M_GPU="${FULL_CONTEXT_GR00T_M_GPU:-6}"
C_GPU="${FULL_CONTEXT_GR00T_C_GPU:-4}"
C16_GPU="${FULL_CONTEXT_GR00T_C16_GPU:-7}"
ATTR_CONFIGS="${FULL_CONTEXT_GR00T_ATTR_CONFIGS:-h,m,c,c16}"

usage() {
    echo "usage: $0 build | preflight | run | status | interpret" >&2
}

rewrite_execution_placement() {
    local source="$1" target="$2" subset="$3"
    "$PYTHON" - "$source" "$target" "$subset" "$H_GPU" "$M_GPU" "$C_GPU" "$C16_GPU" <<'PY'
import json
import os
import sys
import tempfile

source, target, subset, h_gpu, m_gpu, c_gpu, c16_gpu = sys.argv[1:]
payload = json.load(open(source, encoding="utf-8"))
placements = {
    "h": (int(h_gpu), 19555),
    "m": (int(m_gpu), 19556),
    "c": (int(c_gpu), 19557),
    "c16": (int(c16_gpu), 19558),
}
wanted = {item.strip() for item in subset.split(",") if item.strip()}
payload["configs"] = [row for row in payload["configs"] if row["id"] in wanted]
for row in payload["configs"]:
    row["gpu"], row["port"] = placements[row["id"]]
payload["config_subset"] = sorted(wanted)
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

preflight() {
    "$PYTHON" - "$SPEC_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

spec_root = Path(sys.argv[1])
task_sets = ("atomic_seen", "composite_seen", "composite_unseen")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


for task_set in task_sets:
    spec_path = spec_root / f"{task_set}.json"
    if not spec_path.is_file():
        raise SystemExit(f"missing spec: {spec_path}")
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if payload.get("kind") != "full_context_four_config_attribution_spec":
        raise SystemExit(f"{spec_path}: wrong spec kind")
    if payload.get("diagnostic_only") is not True:
        raise SystemExit(f"{spec_path}: attribution spec must be diagnostic_only")
    rows = {str(row["id"]): row for row in payload["configs"]}
    if set(rows) != {"h", "m", "c", "c16"}:
        raise SystemExit(f"{spec_path}: config inventory drift")
    for identifier, row in rows.items():
        plan = Path(row["plan"]).resolve()
        if not plan.is_file():
            raise SystemExit(f"{spec_path}: {identifier} plan missing: {plan}")
        packdir = Path(row["packdir"]).resolve()
        if not packdir.is_dir():
            raise SystemExit(f"{spec_path}: {identifier} packdir missing: {packdir}")
        if row.get("act_scale"):
            act_scale = Path(row["act_scale"]).resolve()
            if not act_scale.is_file():
                raise SystemExit(f"{spec_path}: {identifier} act_scale missing: {act_scale}")
        if row.get("hessian_w4"):
            hessian = Path(row["hessian_w4"]).resolve()
            sidecar = json.loads(Path(str(hessian) + ".json").read_text())
            if sidecar.get("npz_sha256") != digest(hessian):
                raise SystemExit(f"{spec_path}: {identifier} hessian sha drift")
    if rows["m"]["plan"] != rows["h"]["plan"]:
        raise SystemExit(f"{spec_path}: M must reuse the historical main mask")
    if rows["m"]["hessian_w4"] == rows["c"]["hessian_w4"]:
        raise SystemExit(f"{spec_path}: M must use an inventory-exact hessian subset")
    if rows["c16"]["plan"] != rows["c"]["plan"]:
        raise SystemExit(f"{spec_path}: C16 must reuse the candidate mask")
    if rows["c16"]["activation_mode"] != "fp16":
        raise SystemExit(f"{spec_path}: C16 must use FP16 activations")
    if rows["m"]["activation_mode"] != "dynamic_a8":
        raise SystemExit(f"{spec_path}: M must use dynamic A8")
print(f"[preflight] four-config attribution specs OK ({len(task_sets)} task sets)")
PY
}

run_wave() {
    local task_set="$1" tasks="$2" checkpoint="$3"
    local frozen_spec="$SPEC_ROOT/$task_set.json"
    local subset_tag
    subset_tag="$(echo "$ATTR_CONFIGS" | tr ',' '_')"
    local execution_spec="$ATTR_ROOT/specs/.execution-${task_set}_${subset_tag}.json"
    local run_dir="$ATTR_ROOT/${task_set}_${subset_tag}"
    rewrite_execution_placement "$frozen_spec" "$execution_spec" "$ATTR_CONFIGS"
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
        --diagnostic-only \
        --allow-shared-gpus
}

run_all() {
    preflight
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
    interpret
}

status() {
    "$PYTHON" - "$ATTR_ROOT" <<'PY'
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
print(" ".join(f"{name}={counts[name]}/50" for name in ("h", "m", "c", "c16")))
PY
}

interpret() {
    "$PYTHON" "$INTERPRETER" \
        --attribution-root "$ATTR_ROOT" \
        --out "$ATTR_ROOT/interpretation.json"
}

case "${1:-}" in
    build) "$PYTHON" "$SPEC_BUILDER" ;;
    preflight) preflight ;;
    run) run_all ;;
    status) status ;;
    interpret) interpret ;;
    *) usage ;;
esac