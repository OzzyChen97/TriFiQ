#!/usr/bin/env python3
"""Prune the historical GR00T main mask down to the Table-1 byte budget.

The GR00T main mask sits only slightly above the v2 ceiling
(floor(1.10 x QuantVLA row) - fixed bytes), so the v2 upper-bound control
flips the least valuable protected layers from FP16 back to W4 in the
shared Hessian/dynamic/no-correction runtime instead of deleting all 16
protections at once. Per-layer value comes from the frozen round-1
counterfactual scores (W4 -> FP16 restore benefit measured against the
full-W4 context); the smallest-benefit protections are pruned first.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from quantvla_full_context import paired_candidate_summary
from quantvla_dynamic_a8_protocol import (
    protocol_attestation as dynamic_a8_protocol_attestation,
)
from quantvla_outputimpact import atomic_json
from quantvla_table1_bytes import (
    TABLE1_FP16_BYTES,
    fixed_bytes,
    table1_total_static_budget,
    table1_total_static_bytes,
    table1_variable_budget,
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def prune_main_plan(
    *,
    main_plan: dict[str, Any],
    byte_rows: dict[str, dict[str, int]],
    flip_scores: dict[str, dict[str, Any]],
    flip_manifest: list[dict[str, Any]],
    budget: int,
) -> dict[str, Any]:
    protected = sorted(
        name
        for name, row in main_plan["layers"].items()
        if bool(row.get("skip", False))
    )
    all_w4 = sum(row["w4_bytes"] for row in byte_rows.values())
    total = all_w4 + sum(byte_rows[name]["extra_fp16_bytes"] for name in protected)
    if total <= budget:
        # Under the deployed byte scope (packed W4 + scales, rotation=0) the
        # historical main mask already satisfies the corrected ceiling; the
        # anchor collapses to the unmodified main mask.
        return copy.deepcopy(main_plan), [], total
    layer_to_flip = {
        candidate["flip"]["layer"]: candidate
        for candidate in flip_manifest
        if candidate.get("flip")
    }
    missing = [name for name in protected if name not in layer_to_flip]
    if missing:
        raise ValueError(f"no counterfactual score for protected layers: {missing}")
    baseline = flip_scores["context_base"]
    damages: list[tuple[float, str]] = []
    for name in protected:
        candidate = layer_to_flip[name]
        score = flip_scores[candidate["candidate_id"]]
        objective = float(paired_candidate_summary(score, baseline)["objective"])
        # damage of dropping the FP16 protection:
        #   flip measured as W4->FP16 restore (gr00t round-1): more negative
        #   objective = FP16 helps more -> damage = -objective;
        #   flip measured as FP16->W4 removal (pi0.5 round-main): more
        #   positive objective = removal hurts more -> damage = objective.
        damage = (
            objective
            if candidate["flip"]["from"] == "fp16"
            else -objective
        )
        damages.append((damage, name))
    plan = copy.deepcopy(main_plan)
    pruned: list[str] = []
    # Prune the least damaging protections first.
    for _damage, name in sorted(damages, key=lambda pair: (pair[0], pair[1])):
        if total <= budget:
            break
        total -= int(byte_rows[name]["extra_fp16_bytes"])
        plan["layers"][name].update(
            {"bits": 4, "group": 64, "skip": False, "reason": "v2_prune_to_table1_budget"}
        )
        pruned.append(name)
    if total > budget:
        raise ValueError("unreachable: full-W4 total exceeds the variable budget")
    return plan, pruned, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-plan", required=True)
    parser.add_argument("--byte-rows-json", required=True, help="round-1 interventions manifest")
    parser.add_argument("--flip-scores-json", required=True, help="round-1 merged flip scores")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=("gr00t", "pi05"), default="gr00t")
    parser.add_argument("--budget", type=int, default=None)
    args = parser.parse_args()
    budget = (
        args.budget if args.budget is not None else table1_variable_budget(args.model)
    )
    main_plan = load_json(args.main_plan)
    manifest = load_json(args.byte_rows_json)
    byte_rows = manifest["byte_rows"]
    scores = load_json(args.flip_scores_json)
    flip_manifest = manifest["candidates"]
    plan, pruned, total = prune_main_plan(
        main_plan=main_plan,
        byte_rows=byte_rows,
        flip_scores=scores["scores"],
        flip_manifest=flip_manifest,
        budget=budget,
    )
    all_w4 = sum(row["w4_bytes"] for row in byte_rows.values())
    static_total = table1_total_static_bytes(args.model, total)
    static_budget = table1_total_static_budget(args.model)
    if static_total > static_budget:
        raise ValueError("pruned plan still exceeds the Table-1 static ceiling")
    flow_steps = {"gr00t": 4, "pi05": 4}[args.model]
    meta = dict(plan.get("meta") or {})
    meta.update(
        {
            "kind": "full_context_v2_main_pruned_upper_bound_control",
            "budget_anchor": "table1_quantvla_storage_cell",
            "variable_budget_bytes": budget,
            "fixed_bytes": fixed_bytes(args.model),
            "flow_steps": flow_steps,
            "activation_mode": "dynamic_a8",
            "dynamic_a8_protocol": dynamic_a8_protocol_attestation(),
            "pruned_layers": pruned,
            "already_within_budget": len(pruned) == 0,
            "source_main_plan": str(Path(args.main_plan).expanduser().resolve()),
            "byte_rows_source": str(Path(args.byte_rows_json).expanduser().resolve()),
            "flip_scores_source": str(Path(args.flip_scores_json).expanduser().resolve()),
            "runtime_selector": False,
            "runtime_correction": False,
        }
    )
    plan["meta"] = meta
    plan.update(
        {
            "budget_bytes": budget,
            "all_w4_total_bytes": all_w4,
            "fp16_total_bytes": sum(row["fp16_bytes"] for row in byte_rows.values()),
            "total_bytes": total,
            "fixed_bytes": fixed_bytes(args.model),
            "table1_total_static_bytes": static_total,
            "table1_total_static_budget_bytes": static_budget,
            "achieved_target_matrix_compression": float(
                sum(row["fp16_bytes"] for row in byte_rows.values()) / total
            ),
            "table1_total_static_compression": float(
                TABLE1_FP16_BYTES[args.model] / static_total
            ),
            "retained_fp16_layers": len(
                [n for n, r in plan["layers"].items() if bool(r.get("skip", False))]
            ),
            "protected_layers": sorted(
                n for n, r in plan["layers"].items() if bool(r.get("skip", False))
            ),
        }
    )
    atomic_json(Path(args.out).expanduser().resolve(), plan)
    print(json.dumps({"out": args.out, "pruned": pruned, "total_bytes": total}, indent=2))


if __name__ == "__main__":
    main()
