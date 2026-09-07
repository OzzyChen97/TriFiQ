#!/usr/bin/env python3
"""Prepare independent, exact-byte-matched projection controls without rollouts.

Only frozen offline scores, plans and protocols are read. All outputs are placed
in the new experiment directory. Existing plans and results are never changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from prune_gr00t_main_to_budget import prune_main_plan
from quantvla_cross_model_protocol import protocol_attestation, sha256_file, validate_quant_plan
from quantvla_dynamic_a8_protocol import protocol_attestation as dynamic_attestation
from quantvla_table1_bytes import TABLE1_FP16_BYTES, fixed_bytes, table1_total_static_budget
from select_full_context_protection import set_plan_mask

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "scripts/budget_projection_comparison_protocol.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def exact_ranked_removals(order: list[str], costs: dict[str, int], target: int) -> list[str]:
    """Prefer each ranked removal if the remaining exact byte target is feasible.

    This preserves the baseline ranking lexicographically, not an additive-risk
    optimum. Suffix subset-sum feasibility is score-independent and deterministic.
    """
    require(len(set(order)) == len(order), "Duplicate layer in removal order")
    require(all(isinstance(costs[n], int) and costs[n] > 0 for n in order), "Invalid byte cost")
    require(isinstance(target, int) and target >= 0, "Invalid removal target")
    if not order:
        require(target == 0, "Empty inventory cannot meet target")
        return []
    unit = math.gcd(*(costs[n] for n in order))
    require(target % unit == 0, "Target outside byte lattice")
    values = [costs[n] // unit for n in order]
    remaining = target // unit
    suffix = [1] * (len(order) + 1)
    for index in range(len(order) - 1, -1, -1):
        suffix[index] = suffix[index + 1] | (suffix[index + 1] << values[index])
    require(bool((suffix[0] >> remaining) & 1), "Unreachable exact byte target")
    removed = []
    for index, name in enumerate(order):
        cost = values[index]
        if remaining >= cost and (suffix[index + 1] >> (remaining - cost)) & 1:
            removed.append(name)
            remaining -= cost
    require(remaining == 0, "Exact matching failed")
    return removed


def serialize_plan(base: dict, byte_rows: dict, protected: set[str], model: str,
                   identifier: str, ceiling: int, protocol_hash: str) -> dict:
    plan = set_plan_mask(copy.deepcopy(base), protected, reason_prefix="budget_projection_v1")
    all_w4 = sum(row["w4_bytes"] for row in byte_rows.values())
    variable = all_w4 + sum(byte_rows[n]["extra_fp16_bytes"] for n in protected)
    fixed = fixed_bytes(model)
    total = fixed + variable
    require(total <= ceiling, "Plan exceeds experiment ceiling")
    fp16 = sum(row["fp16_bytes"] for row in byte_rows.values())
    # Drop inherited selection decisions and stale counts; preserve layer payloads.
    plan = {"schema_version": 5, "layers": plan["layers"],
            "protected_layers": sorted(protected), "quantized_w4_layers": len(byte_rows) - len(protected),
            "retained_fp16_layers": len(protected), "all_w4_total_bytes": all_w4,
            "total_bytes": variable, "fixed_bytes": fixed, "budget_bytes": ceiling - fixed,
            "fp16_total_bytes": fp16, "table1_total_static_bytes": total,
            "table1_total_static_budget_bytes": ceiling,
            "achieved_target_matrix_compression": fp16 / variable,
            "table1_total_static_compression": TABLE1_FP16_BYTES[model] / total,
            "meta": {"kind": "prospective_budget_projection_comparison", "candidate_id": identifier,
                     "experiment_protocol_sha256": protocol_hash, "model_adapter": model,
                     "activation_mode": "dynamic_a8", "flow_steps": 4, "expected_wrapped_layers": len(byte_rows) - len(protected),
                     "cross_model_protocol": protocol_attestation(), "dynamic_a8_protocol": dynamic_attestation(),
                     "uses_success_labels": False, "runtime_selector": False, "runtime_correction": False,
                     "frozen_before_new_rollouts": True, "original_paper_plan": False,
                     "budget_anchor": "independent_projection_experiment", "static_byte_ceiling": ceiling}}
    validated = validate_quant_plan(plan, model=model, source=identifier)
    require(validated["quantized_w4_layers"] == plan["quantized_w4_layers"], "Serialized inventory mismatch")
    return plan


def stable_write(path: Path, value: dict, *, check: bool) -> None:
    if path.exists():
        require(load(path) == value, f"Immutable experiment artifact drift: {path}")
    elif check:
        raise FileNotFoundError(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(*, check: bool = False) -> dict:
    protocol = load(PROTOCOL)
    out = ROOT / protocol["execution"]["output_root"]
    require(out.resolve() == ROOT / "runs/budget_projection_comparison_v1", "Unexpected output directory")
    # Snapshot all algorithm inputs before using any offline scores.
    protocol_hash = sha256_file(PROTOCOL)
    code = [Path(__file__), ROOT / "scripts/tools/prune_gr00t_main_to_budget.py",
            ROOT / "scripts/tools/quantvla_full_context.py", ROOT / "scripts/tools/quantvla_table1_bytes.py",
            ROOT / "scripts/tools/select_full_context_protection.py", ROOT / "scripts/tools/quantvla_cross_model_protocol.py",
            ROOT / "scripts/tools/quantvla_dynamic_a8_protocol.py"]
    task_path = ROOT / protocol["evaluation"]["task_protocol"]
    tasks = load(task_path)["table1"]["tasks"]
    require(sum(map(len, tasks.values())) == 50 and len(set(sum(tasks.values(), []))) == 50, "Task inventory mismatch")
    registry = {"protocol": protocol, "protocol_file": artifact(PROTOCOL),
                "code": [artifact(p) for p in code], "task_source": artifact(task_path),
                "tasks": tasks, "models": {}, "result_feedback_allowed": False,
                "status": "masks_frozen_execution_preflight_pending"}
    for model, spec in protocol["models"].items():
        manifest_path, scores_path = ROOT / spec["interventions"], ROOT / spec["scores"]
        manifest, scores_doc = load(manifest_path), load(scores_path)
        require(scores_doc["complete"] is True and scores_doc["selection_noise"] == "A", "Incomplete or nonselection scores")
        require(scores_doc.get("uses_success_labels") is False, "Scores use success labels or lack attestation")
        rows = {row["candidate_id"]: row for row in manifest["candidates"]}
        require(set(rows) == set(scores_doc["scores"]), "Score/manifest inventory mismatch")
        initial_row = rows["context_base"]
        initial_path = Path(initial_row["path"])
        require(sha256_file(initial_path) == initial_row["sha256"], "Initializer hash mismatch")
        initial = load(initial_path)
        protected = set(initial_row["protected_layers"])
        byte_rows = manifest["byte_rows"]
        require(set(initial["layers"]) == set(byte_rows), "Byte inventory mismatch")
        require(protected == {n for n, r in initial["layers"].items() if r.get("skip", False)}, "Initializer protection mismatch")
        costs = {n: int(byte_rows[n]["extra_fp16_bytes"]) for n in protected}
        all_w4 = sum(row["w4_bytes"] for row in byte_rows.values())
        fixed = fixed_bytes(model)
        initial_bytes = all_w4 + fixed + sum(costs.values())
        base_score = scores_doc["scores"]["context_base"]
        removal_scores = {}
        for identifier, row in rows.items():
            flip = row.get("flip")
            if not flip or flip["layer"] not in protected:
                continue
            name = flip["layer"]
            require(flip["from"] == "fp16" and flip["to"] == "w4", "Wrong intervention direction")
            require(set(row["protected_layers"]) == protected - {name}, "Not an initializer-relative removal")
            require(sha256_file(Path(row["path"])) == row["sha256"], "Intervention artifact drift")
            score = scores_doc["scores"][identifier]
            seq = lambda x: [(r["task"], r["seed"]) for r in x["d_pac_summary"]["sequences"]]
            require(seq(score) == seq(base_score), "Unpaired score sequences")
            error = float(score["d_func_summary"]["d_final"]) - float(base_score["d_func_summary"]["d_final"])
            require(math.isfinite(error), "Nonfinite final-action relative error")
            removal_scores[name] = error
        require(set(removal_scores) == protected, "Missing removal scores")
        plans, conditions = {}, {}
        identity = f"{model}_initializer"
        plans[identity] = serialize_plan(initial, byte_rows, protected, model, identity, initial_bytes, protocol_hash)
        for label, anchor, numerator, denominator in spec["budgets"]:
            anchor_bytes = initial_bytes if anchor == "initializer" else table1_total_static_budget(model)
            nominal = anchor_bytes * numerator // denominator
            require(all_w4 + fixed <= nominal < initial_bytes, "Budget must require feasible deletion")
            _, removed, variable = prune_main_plan(main_plan=initial, byte_rows=byte_rows,
                flip_scores=scores_doc["scores"], flip_manifest=manifest["candidates"], budget=nominal-fixed)
            actual = fixed + variable
            saved = initial_bytes - actual
            removals = {"full_policy": removed,
                        "final_action_error": exact_ranked_removals(sorted(protected, key=lambda n: (removal_scores[n], n)), costs, saved),
                        "largest_first": exact_ranked_removals(sorted(protected, key=lambda n: (-costs[n], n)), costs, saved)}
            conditions[label] = {"nominal_ceiling_bytes": nominal, "matched_actual_bytes": actual, "arms": {}}
            for arm, names in removals.items():
                identifier = f"{model}_{label}_{arm}"
                require(sum(costs[n] for n in names) == saved, "Control byte mismatch")
                plans[identifier] = serialize_plan(initial, byte_rows, protected-set(names), model, identifier, nominal, protocol_hash)
                conditions[label]["arms"][arm] = {"candidate_id": identifier, "removed": names,
                    "fp16_layers": len(protected)-len(names), "static_bytes": plans[identifier]["table1_total_static_bytes"]}
        outputs, seen = {}, {}
        for identifier, plan in plans.items():
            path = out / "plans" / (identifier + ".json")
            signature = canonical_hash({"model": model, "protected": plan["protected_layers"], "activation": "dynamic_a8", "flow_steps": 4})
            stable_write(path, plan, check=check)
            outputs[identifier] = {**artifact(path), "execution_alias": seen.setdefault(signature, identifier)}
        registry["models"][model] = {"sources": [artifact(manifest_path), artifact(scores_path), artifact(initial_path)],
            "initializer_static_bytes": initial_bytes, "initializer_fp16": len(protected), "conditions": conditions,
            "plans": outputs, "unique_configurations": len(seen)}
    registry["logical_configurations"] = sum(len(x["plans"]) for x in registry["models"].values())
    registry["unique_configurations"] = sum(x["unique_configurations"] for x in registry["models"].values())
    registry["planned_episodes"] = registry["unique_configurations"] * protocol["evaluation"]["episodes_per_configuration"]
    stable_write(out / "protocol_snapshot.json", protocol, check=check)
    stable_write(out / "manifest.json", registry, check=check)
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = prepare(check=args.check)
    print(json.dumps({"valid": True, "status": result["status"], "logical_configurations": result["logical_configurations"],
                      "unique_configurations": result["unique_configurations"], "planned_episodes": result["planned_episodes"],
                      "conditions": {m: x["conditions"] for m, x in result["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()