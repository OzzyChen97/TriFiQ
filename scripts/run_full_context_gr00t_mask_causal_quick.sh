#!/usr/bin/env bash
set -euo pipefail

# Paired 50-episode causal quick test: inherited M0 mask vs full W4 under
# the same Hessian group-64 + DyRange-A8 runtime. Reuses the frozen quick
# tasks and seeds 60-69; only the full-W4 side needs to be executed.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
export PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}"
RUNNER="$REPO_ROOT/scripts/tools/run_robocasa_atomic_matrix.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_gr00t_mask_causal_quick.py"
FORMAL_BUILDER="$REPO_ROOT/scripts/run_full_context_gr00t_mask_causal.sh"

ROOT="${FULL_CONTEXT_GR00T_MASK_CAUSAL_QUICK_ROOT:-$REPO_ROOT/runs/full_context_v2/mask_causal_quick}"
SPEC_ROOT="$ROOT/specs"
RESULTS="$ROOT/results/full_w4_dyrange"
M0_RESULTS="$REPO_ROOT/runs/full_context_v2/quick"
SOURCE_SPECS="$REPO_ROOT/runs/full_context_v2/quick/specs"
ATTR_SPECS="$REPO_ROOT/runs/full_context_v1/gr00t/attribution/specs"
FULL_W4_PLAN="$REPO_ROOT/runs/full_context_v2/mask_causal/full_w4_current_protocol.json"
M0_PLAN="$REPO_ROOT/runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json"
MANIFEST="$ROOT/preregistration.json"
GPU="${FULL_CONTEXT_GR00T_MASK_CAUSAL_QUICK_GPU:-6}"
PORT="${FULL_CONTEXT_GR00T_MASK_CAUSAL_QUICK_PORT:-19579}"
EGL_POOL="${FULL_CONTEXT_GR00T_MASK_CAUSAL_QUICK_EGL_POOL:-1,4,7}"
SEEDS="60,61,62,63,64,65,66,67,68,69"

usage() {
    echo "usage: $0 prepare | preflight | run | status | aggregate" >&2
}

prepare() {
    # This materializes the current-protocol full-W4 plan before any result is seen.
    bash "$FORMAL_BUILDER" prepare >/dev/null
    mkdir -p "$SPEC_ROOT"
    "$PYTHON" - "$SPEC_ROOT" "$SOURCE_SPECS" "$ATTR_SPECS" \
        "$FULL_W4_PLAN" "$M0_PLAN" "$MANIFEST" <<'PY'
import copy
import json
import sys
from pathlib import Path

from quantvla_cross_model_protocol import sha256_file, validate_quant_plan
from quantvla_outputimpact import atomic_json

spec_root, source_root, attr_root, full_path, m0_path, manifest_path = map(Path, sys.argv[1:])
splits = ("atomic_seen", "composite_seen", "composite_unseen")


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def stable_write(path, payload):
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != payload:
            raise SystemExit(f"immutable quick preregistration drift: {path}")
        return
    atomic_json(path, payload)


full_plan = json.loads(full_path.read_text(encoding="utf-8"))
m0_plan = json.loads(m0_path.read_text(encoding="utf-8"))
full_attestation = validate_quant_plan(full_plan, model="gr00t", source=str(full_path))
m0_attestation = validate_quant_plan(m0_plan, model="gr00t", source=str(m0_path))
if full_attestation["quantized_w4_layers"] != 116 or m0_attestation["quantized_w4_layers"] != 100:
    raise SystemExit("quick contrast W4 count drift")
full_modes = {name: bool(row.get("skip", False)) for name, row in full_plan["layers"].items()}
m0_modes = {name: bool(row.get("skip", False)) for name, row in m0_plan["layers"].items()}
changed = sorted(name for name in full_modes if full_modes[name] != m0_modes[name])
if len(changed) != 16:
    raise SystemExit("quick contrast must differ in exactly 16 mask entries")

tasks = {}
spec_artifacts = {}
runtime_audits = {}
for split in splits:
    source = json.loads((source_root / f"{split}.json").read_text(encoding="utf-8"))
    attr = json.loads((attr_root / f"{split}.json").read_text(encoding="utf-8"))
    m0 = copy.deepcopy(next(row for row in source["configs"] if row["id"] == "full_context_v2"))
    full_reference = copy.deepcopy(next(row for row in attr["configs"] if row["id"] == "c"))
    full = copy.deepcopy(m0)
    full.update(
        {
            "id": "full_w4_dyrange",
            "expected_wrapped": 116,
            "plan": str(full_path.resolve()),
            "hessian_w4": full_reference["hessian_w4"],
            "meta": {
                "role": "mask_causal_quick_full_w4",
                "diagnostic_only": True,
                "runtime_correction": False,
                "runtime_selector": False,
            },
        }
    )
    for field in ("gpu", "port", "egl_device"):
        full.pop(field, None)
    same_fields = ("packdir", "act_scale", "errorfold", "atm", "ohb", "ohb_only", "activation_mode")
    mismatch = {field: [full.get(field), m0.get(field)] for field in same_fields if full.get(field) != m0.get(field)}
    if mismatch:
        raise SystemExit(f"{split}: non-mask runtime mismatch: {mismatch}")
    if full["activation_mode"] != "dynamic_a8":
        raise SystemExit("quick contrast must use DyRange-A8")
    tasks[split] = list(source["tasks"])
    spec = {
        "schema_version": 1,
        "kind": "gr00t_mask_causal_quick_execution_spec",
        "purpose": "paired mask-only quick diagnostic",
        "task_set": split,
        "seeds": "60-69",
        "tasks": tasks[split],
        "configs": [full],
        "comparisons": {
            "paired_external_baseline": "full_context_v2",
            "unit": "paired_task_seed",
        },
        "decision": {
            "role": "development_diagnostic",
            "diagnostic_only": True,
            "result_feedback_allowed": False,
        },
    }
    path = spec_root / f"{split}.json"
    stable_write(path, spec)
    spec_artifacts[split] = artifact(path)
    runtime_audits[split] = {
        "same_identity_pack": full["packdir"] == m0["packdir"],
        "same_activation_mode": full["activation_mode"] == m0["activation_mode"],
        "same_correction_state": all(full.get(field) == m0.get(field) for field in ("errorfold", "atm", "ohb", "ohb_only")),
        "full_w4_hessian": artifact(full["hessian_w4"]),
        "m0_hessian": artifact(m0["hessian_w4"]),
    }

manifest = {
    "schema_version": 1,
    "kind": "gr00t_mask_causal_quick_preregistration",
    "immutable": True,
    "diagnostic_only": True,
    "result_feedback_allowed": False,
    "contrast": {
        "m0": {"plan": artifact(m0_path), "w4_layers": 100, "fp16_layers": 16},
        "full_w4": {"plan": artifact(full_path), "w4_layers": 116, "fp16_layers": 0},
        "changed_layers": changed,
        "only_intended_difference": "precision mask and inventory-exact Hessian container",
    },
    "runtime": {
        "weight_quantization": "Hessian group-64 W4",
        "activation_quantization": "DyRange-A8",
        "runtime_correction": False,
        "runtime_selector": False,
        "paired_action_noise": True,
        "flow_steps": 4,
    },
    "evaluation": {
        "tasks": tasks,
        "trial_seeds": list(range(60, 70)),
        "episodes_per_config": 50,
        "baseline_results": "/home1/gyy/vla/QuantVLA/runs/full_context_v2/quick",
    },
    "statistics": {
        "primary": "paired M0 wins vs losses",
        "secondary": ["task_macro_success_rate", "micro_success_rate", "exact McNemar"],
        "uncertainty": "task-then-seed hierarchical bootstrap, 10000 draws, seed 0",
        "scope": "directional development diagnostic; no formal equivalence claim",
    },
    "runtime_audits": runtime_audits,
    "execution_specs": spec_artifacts,
}
stable_write(manifest_path, manifest)
print(json.dumps({"manifest": str(manifest_path), "episodes": 50, "changed_layers": len(changed)}, indent=2))
PY
}

rewrite_placement() {
    local source="$1" target="$2"
    "$PYTHON" - "$source" "$target" "$GPU" "$PORT" <<'PY'
import json
import os
import sys
import tempfile
source, target, gpu, port = sys.argv[1:]
payload = json.load(open(source, encoding="utf-8"))
payload["configs"][0].update({"gpu": int(gpu), "port": int(port)})
os.makedirs(os.path.dirname(target), exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".execution-", suffix=".json", dir=os.path.dirname(target))
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
    prepare
    "$PYTHON" - "$MANIFEST" "$M0_RESULTS" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    (task, seed)
    for tasks in manifest["evaluation"]["tasks"].values()
    for task in tasks
    for seed in manifest["evaluation"]["trial_seeds"]
}
observed = set()
for path in Path(sys.argv[2]).glob("**/*.jsonl"):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("config") == "full_context_v2" and row.get("status") == "complete":
                observed.add((str(row["task"]), int(row["seed"])))
if observed != expected:
    raise SystemExit(f"M0 quick coverage drift: {len(observed)}/{len(expected)}")
if len(manifest["contrast"]["changed_layers"]) != 16:
    raise SystemExit("mask contrast drift")
print("[preflight] paired M0 quick coverage=50; mask-only contrast verified")
PY
}

run_split() {
    local split="$1" tasks="$2"
    local source_spec="$SPEC_ROOT/$split.json"
    local execution_spec="$SPEC_ROOT/.execution-$split.json"
    local run_dir="$ROOT/matrix/$split"
    local checkpoint="$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/$split/checkpoint-60000"
    rewrite_placement "$source_spec" "$execution_spec"
    "$PYTHON" "$RUNNER" \
        --spec "$execution_spec" \
        --run-dir "$run_dir" \
        --phase dev \
        --seeds "$SEEDS" \
        --checkpoint "$checkpoint" \
        --task-set "$split" \
        --tasks "$tasks" \
        --dev-tasks "$tasks" \
        --n-shards "$(awk -F, '{print NF}' <<<"$tasks")" \
        --seed-shards-per-task 5 \
        --egl-device-pool "$EGL_POOL" \
        --trial-batch-size 5 \
        --action-noise paired \
        --diagnostic-only \
        --allow-shared-gpus
    mkdir -p "$RESULTS/$split"
    cp -n "$run_dir"/full_w4_dyrange_s*.jsonl "$RESULTS/$split/" || true
}

run_all() {
    preflight
    while read -r split tasks; do
        run_split "$split" "$tasks"
    done < <("$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for split, tasks in manifest["evaluation"]["tasks"].items():
    print(split, ",".join(tasks))
PY
    )
    aggregate
}

status() {
    "$PYTHON" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
seen = set()
for path in Path(sys.argv[1]).glob("**/*.jsonl"):
    if path.name.startswith(("gpu_efficiency", "gpu_server_efficiency")):
        continue
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("config") == "full_w4_dyrange" and row.get("status") == "complete":
                seen.add((str(row["task"]), int(row["seed"])))
print(json.dumps({"completed": len(seen), "expected": 50, "remaining": 50-len(seen)}, indent=2))
PY
}

aggregate() {
    "$PYTHON" "$AGGREGATOR" \
        --manifest "$MANIFEST" \
        --full-w4-dir "$RESULTS" \
        --m0-dir "$M0_RESULTS" \
        --prior-attribution-root "$REPO_ROOT/runs/full_context_v1/gr00t/attribution" \
        --out "$ROOT/aggregate.json"
}

case "${1:-}" in
    prepare) prepare ;;
    preflight) preflight ;;
    run) run_all ;;
    status) status ;;
    aggregate) aggregate ;;
    *) usage; exit 2 ;;
esac
