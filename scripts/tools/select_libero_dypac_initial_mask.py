#!/usr/bin/env python3
"""Solve the exact-byte π0.5 LIBERO DyPAC initial FP16-protection mask."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from quantvla_full_context import BudgetItem, exact_weighted_knapsack
from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file


def lower_confidence(values: list[float], aggregate: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        return max(float(aggregate), 0.0)
    se = float(array.std(ddof=1) / math.sqrt(len(array)))
    return max(float(aggregate) - 1.96 * se, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--base-plan", required=True)
    parser.add_argument("--scores", required=True, nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lambda-d-func", type=float, default=0.2)
    args = parser.parse_args()
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    byte_rows = {row["name"]: row for row in inventory["layers"]}
    merged = {}
    sources = []
    for raw in args.scores:
        path = Path(raw).resolve(); value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("complete") is not True or value.get("protocol_id") != PROTOCOL["protocol_id"]:
            raise ValueError(f"invalid OutputImpact shard: {path}")
        overlap = set(merged) & set(value["layers"])
        if overlap:
            raise ValueError(f"duplicate OutputImpact layers: {sorted(overlap)[:3]}")
        merged.update(value["layers"])
        sources.append({"path": str(path), "sha256": sha256_file(path)})
    if set(merged) != set(byte_rows):
        raise ValueError(f"OutputImpact coverage mismatch: {len(merged)} != {len(byte_rows)}")
    items = []
    reliability = {}
    for name in sorted(merged):
        score = merged[name]
        pac = lower_confidence(score["d_pac_summary"]["per_sequence"], score["d_pac"])
        func = lower_confidence(score["d_func_summary"]["per_sequence"], score["d_func"])
        reliability[name] = {"d_pac_lcb95": pac, "d_func_lcb95": func}
        items.append(BudgetItem(name=name, extra_bytes=int(byte_rows[name]["extra_fp16_bytes"]), benefit_d_func=func, benefit_d_pac=pac))
    all_w4 = int(inventory["all_w4_bytes"])
    budget = int(inventory["budget_bytes"])
    result = exact_weighted_knapsack(items, budget_bytes=budget - all_w4, lambda_d_func=args.lambda_d_func)
    protected = set(result.protected)
    base = json.loads(Path(args.base_plan).read_text(encoding="utf-8"))
    plan = copy.deepcopy(base)
    for name, row in plan["layers"].items():
        if name in protected:
            row.update({"bits": None, "skip": True, "reason": "dypac_outputimpact_exact_dp_fp16"})
        else:
            row.update({"bits": 4, "group": 64, "skip": False, "reason": "dypac_outputimpact_exact_dp_w4"})
        row["outputimpact"] = reliability[name]
    total = all_w4 + sum(byte_rows[name]["extra_fp16_bytes"] for name in protected)
    if total > budget:
        raise RuntimeError("exact DP exceeded the byte budget")
    plan.update({
        "total_bytes": total,
        "quantized_w4_layers": len(byte_rows) - len(protected),
        "retained_fp16_layers": len(protected),
        "protected_layers": sorted(protected),
        "achieved_target_matrix_compression": inventory["fp16_bytes"] / total,
    })
    plan["meta"].update({
        "kind": "dypac_libero_initial_mask_m0",
        "initial_mask": True,
        "selection_metric": "reliability_shrunk_d_pac_with_complementary_d_func",
        "lambda_d_func": args.lambda_d_func,
        "exact_solver": "sparse_pareto_01_dynamic_programming",
        "outputimpact_sources": sources,
        "uses_success_labels": False,
        "uses_test_rollout_feedback": False,
    })
    atomic_json(args.out, plan)
    report = {
        "schema_version": 1,
        "kind": "dypac_libero_initial_mask_report",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "plan": str(Path(args.out).resolve()),
        "plan_sha256": sha256_file(args.out),
        "budget_bytes": budget,
        "total_bytes": total,
        "quantized_w4_layers": plan["quantized_w4_layers"],
        "retained_fp16_layers": plan["retained_fp16_layers"],
        "protected_layers": plan["protected_layers"],
        "utility": result.utility,
        "sources": sources,
    }
    atomic_json(str(args.out) + ".selection.json", report)
    print(json.dumps({k: report[k] for k in ("plan", "total_bytes", "quantized_w4_layers", "retained_fp16_layers")}, indent=2))


if __name__ == "__main__":
    main()
