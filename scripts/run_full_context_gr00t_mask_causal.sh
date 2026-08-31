#!/usr/bin/env bash
set -euo pipefail

# Formal causal ablation for the GR00T precision mask.
# The only deployed change is M0 (100 W4 / 16 FP16) -> full W4 (116 W4).
# Both configurations use the same checkpoint family, paired seeds, group-64
# Hessian packs, identity row rotation, DyRange-A8, and no runtime correction.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
export PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}"
RUNNER="$REPO_ROOT/scripts/tools/run_robocasa_atomic_matrix.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_gr00t_mask_causal.py"

ROOT="${FULL_CONTEXT_GR00T_MASK_CAUSAL_ROOT:-$REPO_ROOT/runs/full_context_v2/mask_causal}"
SPEC_ROOT="$ROOT/specs"
RESULTS="$ROOT/results/full_w4_dyrange"
M0_RESULTS="$REPO_ROOT/runs/full_context_v2/table1/results/full_context_v2"
ATTR_SPECS="$REPO_ROOT/runs/full_context_v1/gr00t/attribution/specs"
TABLE1_SPECS="$REPO_ROOT/runs/full_context_v2/table1/specs"
FULL_W4_PLAN="$ROOT/full_w4_current_protocol.json"
M0_PLAN="$REPO_ROOT/runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json"
MANIFEST="$ROOT/preregistration.json"
GPU="${FULL_CONTEXT_GR00T_MASK_CAUSAL_GPU:-5}"
PORT="${FULL_CONTEXT_GR00T_MASK_CAUSAL_PORT:-19579}"
EGL_POOL="${FULL_CONTEXT_GR00T_MASK_CAUSAL_EGL_POOL:-1,4,7}"
SEEDS="$(seq -s, 0 49)"

usage() {
    echo "usage: $0 prepare | preflight | run | wait-run | status | aggregate" >&2
}

prepare() {
    mkdir -p "$SPEC_ROOT"
    "$PYTHON" - "$REPO_ROOT" "$SPEC_ROOT" "$ATTR_SPECS" "$TABLE1_SPECS" \
        "$FULL_W4_PLAN" "$M0_PLAN" "$MANIFEST" <<'PY'
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from quantvla_cross_model_protocol import sha256_file, validate_quant_plan
from quantvla_full_context import PROTOCOL, protocol_attestation, require_protocol_attestation
from quantvla_outputimpact import atomic_json
from quantvla_table1_bytes import (
    table1_total_static_bytes,
    table1_total_static_compression,
)

repo, spec_root, attr_root, table_root, full_path, m0_path, manifest_path = map(Path, sys.argv[1:])
splits = ("atomic_seen", "composite_seen", "composite_unseen")


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def stable_write(path, payload):
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != payload:
            raise SystemExit(f"immutable preregistration drift: {path}")
        return
    atomic_json(path, payload)


def load_plan(path, *, require_current_full_context=True):
    document = json.loads(path.read_text(encoding="utf-8"))
    attestation = validate_quant_plan(document, model="gr00t", source=str(path))
    if require_current_full_context:
        require_protocol_attestation(document.get("meta") or {}, source=str(path))
    return document, attestation


def layer_mode(document):
    return {
        name: "fp16" if bool(row.get("skip", False)) else "w4"
        for name, row in document["layers"].items()
    }


def hessian_common_audit(full_hessian, subset_hessian, common_layers):
    full = np.load(full_hessian, allow_pickle=False)
    subset = np.load(subset_hessian, allow_pickle=False)
    full_names = [str(value) for value in full["layer_names"].tolist()]
    subset_names = [str(value) for value in subset["layer_names"].tolist()]
    if set(subset_names) != set(common_layers):
        raise SystemExit(f"subset Hessian inventory mismatch: {subset_hessian}")
    if set(full_names) < set(subset_names):
        raise SystemExit(f"full Hessian misses subset layers: {full_hessian}")
    full_index = {name: index for index, name in enumerate(full_names)}
    for subset_index, name in enumerate(subset_names):
        source_index = full_index[name]
        for prefix in ("packed", "scales", "clipping", "error"):
            left = full[f"{prefix}_{source_index:04d}"]
            right = subset[f"{prefix}_{subset_index:04d}"]
            if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                raise SystemExit(f"Hessian payload drift for {name}/{prefix}")
    return {
        "full_layers": len(full_names),
        "common_layers": len(subset_names),
        "common_payloads_byte_identical": True,
    }


m0_plan, m0_attestation = load_plan(m0_path)
full_plan = copy.deepcopy(m0_plan)
for row in full_plan["layers"].values():
    row.update(
        {
            "bits": 4,
            "group": 64,
            "skip": False,
            "reason": "formal_mask_causal_full_w4_control",
        }
    )
full_plan["protected_layers"] = []
full_plan["quantized_w4_layers"] = len(full_plan["layers"])
full_plan["retained_fp16_layers"] = 0
full_plan["total_bytes"] = int(full_plan["all_w4_total_bytes"])
full_plan["achieved_target_matrix_compression"] = float(
    full_plan["fp16_total_bytes"] / full_plan["total_bytes"]
)
full_plan["table1_total_static_bytes"] = table1_total_static_bytes(
    "gr00t", full_plan["total_bytes"]
)
full_plan["table1_total_static_compression"] = table1_total_static_compression(
    "gr00t", full_plan["total_bytes"]
)
full_meta = dict(full_plan.get("meta") or {})
for obsolete in (
    "counterfactual_manifest_sha256",
    "full_network_scores_sha256",
    "proposal_manifest_sha256",
    "selection_result",
):
    full_meta.pop(obsolete, None)
full_meta.update(
    {
        "kind": "formal_mask_causal_full_w4_control",
        "candidate_id": "full_w4_dyrange",
        "selected_candidate_id": "full_w4_dyrange",
        "source_m0_plan": str(m0_path.resolve()),
        "source_m0_plan_sha256": sha256_file(m0_path),
        "derivation": "flip exactly the 16 M0 FP16 layers to group-64 W4",
        "result_feedback_used": False,
    }
)
full_plan["meta"] = full_meta
stable_write(full_path, full_plan)
full_plan, full_attestation = load_plan(full_path)
full_modes = layer_mode(full_plan)
m0_modes = layer_mode(m0_plan)
if set(full_modes) != set(m0_modes):
    raise SystemExit("M0/full-W4 target-layer inventory differs")
if set(full_modes.values()) != {"w4"}:
    raise SystemExit("full-W4 plan is not uniformly W4")
changed = sorted(name for name in full_modes if full_modes[name] != m0_modes[name])
if len(changed) != 16 or any(m0_modes[name] != "fp16" for name in changed):
    raise SystemExit("causal contrast must be exactly the 16 M0 FP16 protections")
if full_attestation["quantized_w4_layers"] != 116 or m0_attestation["quantized_w4_layers"] != 100:
    raise SystemExit("unexpected W4 counts")

split_audits = {}
spec_artifacts = {}
for split in splits:
    attr = json.loads((attr_root / f"{split}.json").read_text(encoding="utf-8"))
    table = json.loads((table_root / f"{split}.json").read_text(encoding="utf-8"))
    c = copy.deepcopy(next(row for row in attr["configs"] if row["id"] == "c"))
    m = copy.deepcopy(next(row for row in table["configs"] if row["id"] == "full_context_v2"))
    comparable_fields = (
        "packdir", "act_scale", "errorfold", "atm", "ohb", "ohb_only", "activation_mode"
    )
    mismatches = {field: [c.get(field), m.get(field)] for field in comparable_fields if c.get(field) != m.get(field)}
    if mismatches:
        raise SystemExit(f"{split}: non-mask runtime mismatch: {mismatches}")
    if c.get("activation_mode") != "dynamic_a8":
        raise SystemExit(f"{split}: causal contrast must use DyRange-A8")
    full_hessian = Path(c["hessian_w4"]).resolve()
    m0_hessian = Path(m["hessian_w4"]).resolve()
    hessian_audit = hessian_common_audit(
        full_hessian,
        m0_hessian,
        [name for name, mode in m0_modes.items() if mode == "w4"],
    )
    c.update(
        {
            "id": "full_w4_dyrange",
            "plan": str(full_path.resolve()),
            "meta": {
                "role": "formal_mask_causal_full_w4",
                "formal_failure_on_crash": True,
                "runtime_correction": False,
                "runtime_selector": False,
            },
        }
    )
    for field in ("gpu", "port", "egl_device"):
        c.pop(field, None)
    spec = {
        "schema_version": 1,
        "kind": "gr00t_mask_causal_execution_spec",
        "purpose": "isolate_M0_mask_effect_under_common_Hessian_DyRange_runtime",
        "task_set": split,
        "seeds": "0-49",
        "configs": [c],
        "comparisons": {
            "paired_external_baseline": "full_context_v2",
            "unit": "paired_task_seed",
        },
        "decision": {
            "role": "formal_causal_ablation",
            "result_feedback_allowed": False,
            "practical_equivalence_margin": 0.02,
        },
    }
    spec_path = spec_root / f"{split}.json"
    stable_write(spec_path, spec)
    spec_artifacts[split] = artifact(spec_path)
    split_audits[split] = {
        "identity_pack_same_path": c["packdir"] == m["packdir"],
        "activation_mode": c["activation_mode"],
        "runtime_correction": False,
        "runtime_selector": False,
        "full_w4_hessian": artifact(full_hessian),
        "m0_hessian": artifact(m0_hessian),
        "hessian_common_layer_audit": hessian_audit,
    }

table = PROTOCOL["table1"]
manifest = {
    "schema_version": 1,
    "kind": "gr00t_mask_causal_preregistration",
    "immutable": True,
    "result_feedback_allowed": False,
    "hypothesis": (
        "causal closed-loop effect of the inherited M0 precision mask relative "
        "to full W4 under an otherwise identical runtime"
    ),
    "contrast": {
        "m0": {
            "plan": artifact(m0_path),
            "w4_layers": 100,
            "fp16_layers": 16,
            "existing_formal_results": str(repo / "runs/full_context_v2/table1/results/full_context_v2"),
        },
        "full_w4": {
            "plan": artifact(full_path),
            "w4_layers": 116,
            "fp16_layers": 0,
        },
        "changed_layers": changed,
        "only_intended_difference": "precision mask and its inventory-exact Hessian container",
    },
    "runtime_contract": {
        "weight_quantization": "Hessian group-64 W4",
        "row_rotation": "identity",
        "activation_quantization": "DyRange-A8 dynamic per-forward per-channel amax",
        "runtime_correction": False,
        "runtime_selector": False,
        "paired_action_noise": True,
        "flow_steps": 4,
    },
    "full_context_protocol": protocol_attestation(),
    "execution_specs": spec_artifacts,
    "split_audits": split_audits,
    "evaluation": {
        "benchmark": table["benchmark"],
        "task_sets": table["tasks"],
        "trial_seeds": list(range(50)),
        "episodes_per_config": 2500,
        "missing_episode_policy": table["missing_episode_policy"],
        "duplicate_episode_policy": table["duplicate_episode_policy"],
    },
    "statistics": {
        "primary": "task_macro_success_rate",
        "secondary": ["micro_success_rate", "paired_exact_McNemar"],
        "uncertainty": "task-then-seed hierarchical bootstrap, 10000 draws, seed 0",
        "practical_equivalence_margin": 0.02,
        "decision_rule": {
            "positive": "M0 has positive macro and micro deltas and CI excludes 0 or McNemar p<0.05",
            "negative": "M0 has negative macro and micro deltas and CI excludes 0 or McNemar p<0.05",
            "equivalent": "the full bootstrap CI lies inside [-0.02,+0.02]",
            "otherwise": "inconclusive",
        },
    },
}
stable_write(manifest_path, manifest)
print(json.dumps({"manifest": str(manifest_path), "changed_layers": len(changed), "specs": len(splits)}, indent=2))
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
row = payload["configs"][0]
row["gpu"] = int(gpu)
row["port"] = int(port)
directory = os.path.dirname(target)
os.makedirs(directory, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".execution-", suffix=".json", dir=directory)
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
if manifest["contrast"]["m0"]["w4_layers"] != 100:
    raise SystemExit("M0 W4 count drift")
if manifest["contrast"]["full_w4"]["w4_layers"] != 116:
    raise SystemExit("full-W4 count drift")
if len(manifest["contrast"]["changed_layers"]) != 16:
    raise SystemExit("mask contrast drift")
for split, audit in manifest["split_audits"].items():
    if not audit["identity_pack_same_path"]:
        raise SystemExit(f"{split}: identity pack mismatch")
    if not audit["hessian_common_layer_audit"]["common_payloads_byte_identical"]:
        raise SystemExit(f"{split}: common Hessian payload mismatch")
rows = []
for path in Path(sys.argv[2]).glob("**/*.jsonl"):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "complete":
                rows.append((row["task"], int(row["seed"])))
if len(rows) != 2500 or len(set(rows)) != 2500:
    raise SystemExit(f"M0 formal baseline coverage drift: {len(rows)}/{len(set(rows))}")
print("[preflight] immutable mask-only causal contrast verified; M0 coverage=2500")
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
        --phase formal \
        --seeds "$SEEDS" \
        --checkpoint "$checkpoint" \
        --task-set "$split" \
        --tasks "$tasks" \
        --n-shards "$(awk -F, '{print NF}' <<<"$tasks")" \
        --seed-shards-per-task 1 \
        --egl-device-pool "$EGL_POOL" \
        --trial-batch-size 5 \
        --action-noise paired \
        --formal-provenance-v2 \
        --allow-shared-gpus
    mkdir -p "$RESULTS/$split"
    cp -n "$run_dir"/full_w4_dyrange_s*.jsonl "$RESULTS/$split/" || true
}

run_all() {
    preflight
    local tasks_file="$ROOT/.tasks.txt"
    "$PYTHON" - > "$tasks_file" <<'PY'
from quantvla_full_context import PROTOCOL
for split in ("atomic_seen", "composite_seen", "composite_unseen"):
    print(f"{split} {','.join(PROTOCOL['table1']['tasks'][split])}")
PY
    while read -r split tasks; do
        [[ -n "$split" ]] || continue
        run_split "$split" "$tasks"
    done < "$tasks_file"
    aggregate
}

wait_run() {
    preflight
    while pgrep -f '[s]erve_pi05_quant_policy.py|[r]un_robocasa365_pi05_eval.py|[r]un_pi05_formal_worker_seeded.sh' >/dev/null; do
        echo "[wait-run] pi0.5 formal wave is still active; preserving its GPUs" >&2
        sleep 30
    done
    echo "[wait-run] pi0.5 formal wave released; launching GR00T mask causal ablation" >&2
    run_all
}

status() {
    "$PYTHON" - "$RESULTS" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

counts = Counter()
successes = Counter()
seen = set()
for path in Path(sys.argv[1]).glob("**/*.jsonl"):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "complete":
            continue
        key = (str(row["task"]), int(row["seed"]))
        if key in seen:
            continue
        seen.add(key)
        task_set = path.parent.name
        counts[task_set] += 1
        successes[task_set] += int(bool(row["success"]))
payload = {
    "completed": len(seen),
    "expected": 2500,
    "remaining": 2500 - len(seen),
    "by_task_set": dict(counts),
}
if len(seen) == 2500:
    payload["successes_withheld"] = False
    payload["successes"] = sum(successes.values())
else:
    payload["successes_withheld"] = True
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

aggregate() {
    "$PYTHON" "$AGGREGATOR" \
        --manifest "$MANIFEST" \
        --full-w4-dir "$RESULTS" \
        --m0-dir "$M0_RESULTS" \
        --equivalence-margin 0.02 \
        --out "$ROOT/aggregate.json"
}

case "${1:-}" in
    prepare) prepare ;;
    preflight) preflight ;;
    run) run_all ;;
    wait-run) wait_run ;;
    status) status ;;
    aggregate) aggregate ;;
    *) usage; exit 2 ;;
esac
