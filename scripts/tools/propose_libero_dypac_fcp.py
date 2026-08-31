#!/usr/bin/env python3
"""Generate exact-byte FCP proposals from complete-network coordinate flips."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from quantvla_full_context import BudgetItem, exact_weighted_knapsack
from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file


def merge_scores(paths: list[str]) -> tuple[dict, list[dict]]:
    scores = {}; sources = []
    for raw in paths:
        path = Path(raw).resolve(); value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("complete") is not True or value.get("noise") != "A":
            raise ValueError(f"invalid coordinate score shard: {path}")
        if set(scores) & set(value["scores"]):
            raise ValueError("duplicate coordinate score")
        scores.update(value["scores"]); sources.append({"path": str(path), "sha256": sha256_file(path)})
    return scores, sources


def paired_lcb(values: np.ndarray) -> float:
    if len(values) < 2:
        return max(float(values.mean()), 0.0)
    return max(float(values.mean() - 1.96 * values.std(ddof=1) / math.sqrt(len(values))), 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinate-manifest", required=True)
    parser.add_argument("--coordinate-scores", required=True, nargs="+")
    parser.add_argument("--initial-plan", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    coordinate_path = Path(args.coordinate_manifest).resolve()
    coordinate = json.loads(coordinate_path.read_text(encoding="utf-8"))
    scores, sources = merge_scores(args.coordinate_scores)
    if set(scores) != set(coordinate["candidates"]):
        raise ValueError("coordinate score coverage mismatch")
    baseline = scores["context_base"]
    initial_path = Path(args.initial_plan).resolve()
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    initial_protected = set(initial["protected_layers"])
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    byte_rows = {row["name"]: row for row in inventory["layers"]}
    items = []
    marginal = {}
    for identifier, candidate in coordinate["candidates"].items():
        flip = candidate.get("flip")
        if not flip:
            continue
        name = flip["layer"]
        score = scores[identifier]
        base_pac = np.asarray(baseline["d_pac_summary"]["per_sequence"])
        flip_pac = np.asarray(score["d_pac_summary"]["per_sequence"])
        base_func = np.asarray(baseline["d_func_summary"]["per_sequence"])
        flip_func = np.asarray(score["d_func_summary"]["per_sequence"])
        if name in initial_protected:
            pac_delta = flip_pac - base_pac
            func_delta = flip_func - base_func
        else:
            pac_delta = base_pac - flip_pac
            func_delta = base_func - flip_func
        pac = paired_lcb(pac_delta); func = paired_lcb(func_delta)
        marginal[name] = {"d_pac_lcb95": pac, "d_func_lcb95": func, "direction_observed": flip["direction"]}
        items.append(BudgetItem(name=name, extra_bytes=int(byte_rows[name]["extra_fp16_bytes"]), benefit_d_func=func, benefit_d_pac=pac))

    output_dir = Path(args.out_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    candidates = {}
    seen = set()

    def materialize(identifier: str, protected: set[str], source: str) -> None:
        key = tuple(sorted(protected))
        if key in seen:
            return
        seen.add(key)
        plan = copy.deepcopy(initial)
        for name, row in plan["layers"].items():
            if name in protected:
                row.update({"bits": None, "skip": True, "reason": "dypac_fcp_fp16"})
            else:
                row.update({"bits": 4, "group": 64, "skip": False, "reason": "dypac_fcp_w4"})
        total = inventory["all_w4_bytes"] + sum(byte_rows[name]["extra_fp16_bytes"] for name in protected)
        if total > inventory["budget_bytes"]:
            raise ValueError(f"{identifier}: byte budget exceeded")
        plan.update({"total_bytes": total, "protected_layers": sorted(protected), "retained_fp16_layers": len(protected), "quantized_w4_layers": len(byte_rows)-len(protected), "achieved_target_matrix_compression": inventory["fp16_bytes"] / total})
        plan["meta"].update({"kind": "dypac_libero_fcp_proposal", "candidate_id": identifier, "proposal_source": source, "coordinate_manifest_sha256": sha256_file(coordinate_path), "uses_success_labels": False})
        path = output_dir / f"{identifier}.plan.json"; atomic_json(path, plan)
        candidates[identifier] = {"path": str(path), "sha256": sha256_file(path), "protected_layers": sorted(protected), "total_bytes": total, "source": source}

    materialize("context_base", initial_protected, "initial_mask_m0")
    capacity = inventory["budget_bytes"] - inventory["all_w4_bytes"]
    for value in (0.0, 0.2, 0.5, 0.8, 1.0):
        result = exact_weighted_knapsack(items, budget_bytes=int(capacity), lambda_d_func=value)
        materialize(f"dp_lambda_{str(value).replace('.', 'p')}", set(result.protected), f"exact_dp_lambda_d_func_{value}")
    manifest = {
        "schema_version": 1,
        "kind": "dypac_libero_fcp_proposal_manifest",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "coordinate_manifest": str(coordinate_path),
        "coordinate_manifest_sha256": sha256_file(coordinate_path),
        "coordinate_score_sources": sources,
        "initial_plan": str(initial_path),
        "initial_plan_sha256": sha256_file(initial_path),
        "marginal_benefits": marginal,
        "exact_solver": "sparse_pareto_01_dynamic_programming",
        "candidates": candidates,
        "uses_success_labels": False,
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output_dir / 'manifest.json'), "candidates": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()
