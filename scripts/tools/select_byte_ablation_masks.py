#!/usr/bin/env python3
"""Deterministic byte-equality MILP masks for the CS / CKA-DiT / CS+CKA ablation.

Three arms select the FP16-protected subset of the 116-layer GR00T candidate
inventory from all-W4.  The allocation score differs per arm and only per arm::

    CS-only   S_i = w_i * norm(cs)_i
    CKA-only  S_i = w_i * norm(1 - cka_dit)_i
    CS+CKA    S_i = w_i * (16*norm(1-cka_dit)_i + norm(cs)_i) / 17

where norm(.) is the joint min-max normalization over all (layer, bit) pairs
of that term (identical to the frozen headline selector), and w_i are the
shared action-impact weights.  All arms share the RMS/saturation hard guards
(guard-violating layers are forced FP16), the 116-layer inventory, the frozen
Table-1 byte rows, and this deterministic solver.

The MILP maximizes the FP16-protected risk under an EXACT byte equality::

    max  sum_i x_i * S_i      s.t.  sum_i x_i * extra_fp16_bytes_i = B
    x_i in {0,1}, x_i = 1 means "protected" (FP16), B = 125,108,224

Guard-violating layers are forced FP16 and excluded from the decision
variables entirely (their bytes are subtracted from the budget up front), so
every arm lands on exactly 634,716,160 candidate bytes (962,068,480 total
static bytes = 0.896 GiB, 2.2239x compression), identical to the frozen DyPAC
Table-1 row.  Deterministic tie-breaks among objective-optimal masks:
(1) fewest FP16 layers, then (2) the lexicographically smallest FP16 mask over
sorted layer names.  Re-running with the same frozen inputs must reproduce the
same masks and hashes; the tool fails loudly otherwise.

The tool also emits DyPAC-format deployment plans (schema_version 5) for each
arm, and a frozen masks manifest with pairwise Hamming distances, byte and
guard checks, and input attestation hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from quantvla_cross_model_protocol import sha256_file, validate_quant_plan
from quantvla_table1_bytes import (
    TABLE1_FP16_BYTES,
    fixed_bytes,
    table1_total_static_budget,
    table1_total_static_bytes,
    table1_total_static_compression,
    table1_variable_budget,
)
from gr00t_select_plan import (
    build_scores,
    build_weights_with_log,
)

ARMS = {
    "cs_only": {"lambda_cka": 0.0, "lambda_cs": 1.0, "uses_cka": False, "uses_cs": True},
    "cka_only": {"lambda_cka": 1.0, "lambda_cs": 0.0, "uses_cka": True, "uses_cs": False},
    "cs_cka": {"lambda_cka": 16.0, "lambda_cs": 1.0, "uses_cka": True, "uses_cs": True},
}
CKA_FIELD = "cka_dit"
BUDGET_BYTES = 125_108_224  # frozen DyPAC mask variable bytes above all-W4
EXPECTED = {
    "cs_cka": {"w4": 93, "fp16": 23},
    "cka_only": {"w4": 94, "fp16": 22},
    "cs_only": {"w4": 92, "fp16": 24},
}
ALL_W4_BYTES = 509_607_936
FIXED_BYTES = 327_352_320
TOTAL_STATIC_BYTES = 962_068_480


def canonical_sha(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def solve_arm(
    *,
    names: list[str],
    scores: dict[str, float],
    extras: dict[str, int],
    forced_fp16: set[str],
    budget: int,
) -> dict[str, Any]:
    """Exact byte-equality MILP with deterministic lexicographic tie-breaks."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    order = [n for n in names if scores.get(n) is not None and n not in forced_fp16]
    # Guard-violating layers are FP16-protected by construction and are not
    # decision variables at all: their bytes are accounted separately and they
    # never enter the objective, the count, or the lexicographic tie-breaks.
    guard_set = set(forced_fp16)
    unmeasured = {n for n in names if scores.get(n) is None}
    if guard_set & unmeasured:
        raise RuntimeError(f"guard-forced layer lacks measurement: {guard_set & unmeasured}")

    n = len(order)
    cost = np.array([extras[name] for name in order], dtype=float)
    objective = np.array([-scores[name] for name in order], dtype=float)

    fixed = sum(extras[name] for name in unmeasured)
    pinned_bytes = sum(extras[name] for name in guard_set)
    remaining = budget - fixed - pinned_bytes
    if remaining < 0:
        raise RuntimeError(f"forced FP16 bytes exceed budget {budget}")

    def solve(c, extra_constraints=(), fixed_map=None):
        fixed_map = dict(fixed_map or {})
        bounds_low = np.zeros(n)
        bounds_high = np.ones(n)
        for name, value in fixed_map.items():
            index = order.index(name)
            bounds_low[index] = bounds_high[index] = float(value)
        constraints = [
            LinearConstraint(cost.reshape(1, -1), lb=[remaining], ub=[remaining]),
            *extra_constraints,
        ]
        result = milp(
            c=c,
            constraints=constraints,
            integrality=np.ones(n),
            bounds=Bounds(bounds_low, bounds_high),
        )
        return result

    # Stage 1: maximize protected risk under exact byte equality.
    stage1 = solve(objective)
    if not stage1.success:
        raise RuntimeError(f"byte-equality MILP infeasible: {stage1.message}")
    best_objective = float(-stage1.fun)

    # Stage 2: fewest FP16 layers among objective-optimal masks.
    count_objective = np.ones(n)
    stage2 = solve(
        count_objective,
        extra_constraints=[
            LinearConstraint(
                objective.reshape(1, -1),
                lb=[-np.inf],
                ub=[-best_objective + 1e-9],
            )
        ],
    )
    if not stage2.success:
        raise RuntimeError(f"count tie-break infeasible: {stage2.message}")
    best_count = int(round(stage2.fun))

    # Stage 3: lexicographically smallest FP16 mask over sorted layer names.
    fixed_map: dict[str, int] = {}
    for name in order:
        trial = solve(
            count_objective,
            extra_constraints=[
                LinearConstraint(
                    objective.reshape(1, -1),
                    lb=[-np.inf],
                    ub=[-best_objective + 1e-9],
                ),
                LinearConstraint(
                    np.ones((1, n)),
                    lb=[best_count],
                    ub=[best_count],
                ),
            ],
            fixed_map={**fixed_map, name: 0},
        )
        if trial.success and int(round(trial.fun)) <= best_count:
            fixed_map[name] = 0
        else:
            fixed_map[name] = 1

    protected = (
        set(unmeasured)
        | set(guard_set)
        | {name for name, value in fixed_map.items() if value == 1}
    )
    byte_total = (
        fixed
        + pinned_bytes
        + sum(extras[name] for name in protected if name in order)
    )
    objective_total = sum(
        scores[name] for name in sorted(protected) if name in order
    )
    return {
        "protected": sorted(protected),
        "w4": sorted(n for n in names if n not in protected),
        "byte_total": int(byte_total),
        "objective": float(objective_total),
        "fp16_count": len(protected),
        "forced_fp16": sorted(unmeasured | set(guard_set)),
    }


def build_deployment_plan(
    *,
    arm: str,
    protected: list[str],
    byte_rows: dict[str, dict[str, int]],
    template: dict[str, Any],
    extra_meta: dict[str, Any],
    packdirs: dict[str, str],
) -> dict[str, Any]:
    layers = {}
    for name in sorted(byte_rows):
        is_protected = name in set(protected)
        layers[name] = {
            "bits": None if is_protected else 4,
            "group": 64,
            "reason": "byte_ablation_fp16" if is_protected else "byte_ablation_w4",
            "skip": is_protected,
        }
    total = ALL_W4_BYTES + BUDGET_BYTES
    fp16_total = sum(row["fp16_bytes"] for row in byte_rows.values())
    meta = {
        "kind": "byte_ablation_allocation_mask",
        "arm": arm,
        "model_adapter": "gr00t",
        "activation_mode": "dynamic_a8",
        "flow_steps": 4,
        "budget_anchor": "dy_pac_table1_total_static_bytes",
        "fixed_bytes": FIXED_BYTES,
        "target_compression_scope": "candidate",
        "target_compression": float(fp16_total / table1_variable_budget("gr00t")),
        "runtime_selector": False,
        "runtime_correction": False,
        "uses_task_ids_for_stratified_statistics": True,
        "uses_task_labels_for_routing_or_task_specific_mask": False,
        "uses_success_labels": False,
        "uses_gradients": False,
        "uses_extra_training_data": False,
        **extra_meta,
    }
    return {
        "achieved_target_matrix_compression": float(fp16_total / total),
        "all_w4_total_bytes": ALL_W4_BYTES,
        "budget_bytes": table1_variable_budget("gr00t"),
        "fixed_bytes": FIXED_BYTES,
        "fp16_total_bytes": fp16_total,
        "layers": layers,
        "packdirs": dict(packdirs),
        "protected_layers": sorted(protected),
        "quantized_w4_layers": len(byte_rows) - len(protected),
        "retained_fp16_layers": len(protected),
        "schema_version": 5,
        "table1_total_static_budget_bytes": table1_total_static_budget("gr00t"),
        "table1_total_static_bytes": TOTAL_STATIC_BYTES,
        "table1_total_static_compression": table1_total_static_compression(
            "gr00t", total
        ),
        "total_bytes": int(total),
        "meta": meta,
    }


def hamming(a: list[str], b: list[str]) -> int:
    return len(set(a) ^ set(b))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensitivity",
        default="/home1/gyy/vla/QuantVLA/checkpoints/packs/robocasa365/"
        "sensitivity_robocasa365_atomic_protocolfix_d4_g64_b4_6_dit.json",
    )
    parser.add_argument(
        "--byte-rows-manifest",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v2/p2/"
        "interventions_dynamic/manifest.json",
    )
    parser.add_argument(
        "--plan-template",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v2/p2/"
        "gr00t_full_context_v2_frozen.json",
    )
    parser.add_argument(
        "--weight-metric",
        default="d_func",
        choices=["d_solver", "d_func"],
        help=(
            "Shared action-impact weight source (frozen in the manifest). "
            "Default d_func reproduces the frozen 16:1 headline w_i log; "
            "d_solver is the per-layer solver-divergence alternative."
        ),
    )
    parser.add_argument("--budget-bytes", type=int, default=BUDGET_BYTES)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--packdirs",
        default="/home1/gyy/vla/QuantVLA/checkpoints/packs/robocasa365/"
        "duquant_packed_robocasa365_protocolfix_d4_w4a8_b64c32ls015",
        help="Comma-separated plan-level packdirs (deployment overrides per split).",
    )
    args = parser.parse_args()

    sensitivity = load_json(args.sensitivity)
    manifest = load_json(args.byte_rows_manifest)
    byte_rows = manifest["byte_rows"]
    names = sorted(byte_rows)
    if len(names) != 116:
        raise RuntimeError(f"expected 116-layer inventory, got {len(names)}")
    if sum(row["w4_bytes"] for row in byte_rows.values()) != ALL_W4_BYTES:
        raise RuntimeError("byte rows no longer reproduce the frozen all-W4 total")

    guards = sensitivity["meta"]["guard_thresholds"]
    tau_rms = float(guards["tau_rms"])
    tau_sat = float(guards["tau_sat"])
    guarded = {
        name
        for name in names
        if float(sensitivity["layers"][name]["b4"]["rms_ratio"]) > tau_rms
        or float(sensitivity["layers"][name]["b4"]["sat_rate"]) > tau_sat
    }

    template = load_json(args.plan_template)
    packdirs = {
        value.strip().split("=", 1)[0]: value.strip().split("=", 1)[-1]
        for value in args.packdirs.split(",")
    }

    weights, weight_log = build_weights_with_log(
        sensitivity, names, metric=args.weight_metric
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    plans: dict[str, dict[str, Any]] = {}
    for arm, spec in ARMS.items():
        score_table = build_scores(
            sensitivity,
            names,
            spec["lambda_cka"],
            spec["lambda_cs"],
            cka_field=CKA_FIELD,
        )
        scores = {}
        for name in names:
            value = score_table.get(name, {}).get(4)
            if value is not None:
                scores[name] = weights[name] * value
        unavailable = {name for name in names if name not in scores}
        if guarded & unavailable:
            raise RuntimeError("guard layer without measurement")
        result = solve_arm(
            names=names,
            scores=scores,
            extras={name: int(byte_rows[name]["extra_fp16_bytes"]) for name in names},
            forced_fp16=set(guarded),
            budget=args.budget_bytes,
        )
        if result["byte_total"] != args.budget_bytes:
            raise RuntimeError(f"{arm}: byte equality violated: {result['byte_total']}")
        if not guarded.issubset(result["protected"]):
            raise RuntimeError(f"{arm}: guard layer left W4")
        results[arm] = result

        extra_meta = {
            "uses_cka": spec["uses_cka"],
            "uses_cs": spec["uses_cs"],
            "cka_field": CKA_FIELD,
            "lambda": {"cka": spec["lambda_cka"], "cs": spec["lambda_cs"]},
            "weight_metric": args.weight_metric,
            "w_i_log": weight_log,
            "guard_thresholds": guards,
            "guard_filtered_layers": sorted(guarded),
            "byte_rows_manifest_sha256": sha256_file(args.byte_rows_manifest),
            "sensitivity_sha256": sha256_file(args.sensitivity),
            "solver": "scipy-milp-equality",
            "tie_break": (
                "1) fewest FP16 layers; 2) lexicographically smallest FP16 mask "
                "over sorted layer names"
            ),
            "budget_bytes": args.budget_bytes,
        }
        plan = build_deployment_plan(
            arm=arm,
            protected=result["protected"],
            byte_rows=byte_rows,
            template=template,
            extra_meta=extra_meta,
            packdirs=packdirs,
        )
        validation = validate_quant_plan(plan, model="gr00t", source=f"byte-ablation {arm}")
        if validation["quantized_w4_layers"] != plan["quantized_w4_layers"]:
            raise RuntimeError(f"{arm}: plan validation count mismatch")
        if plan["total_bytes"] != TOTAL_STATIC_BYTES - FIXED_BYTES:
            raise RuntimeError(f"{arm}: static byte total mismatch")
        if plan["table1_total_static_bytes"] != TOTAL_STATIC_BYTES:
            raise RuntimeError(f"{arm}: table1 static bytes mismatch")
        plans[arm] = plan

    # Mask diversity: collapse is a hard stop.
    for left, right in (("cs_only", "cka_only"), ("cs_only", "cs_cka"), ("cka_only", "cs_cka")):
        if set(results[left]["protected"]) == set(results[right]["protected"]):
            raise SystemExit(
                f"mask collapse between {left} and {right}; stopping the experiment"
            )

    manifest = {
        "schema_version": 1,
        "kind": "byte_ablation_masks",
        "weight_metric": args.weight_metric,
        "budget_bytes": args.budget_bytes,
        "all_w4_bytes": ALL_W4_BYTES,
        "candidate_total_bytes": ALL_W4_BYTES + args.budget_bytes,
        "total_static_bytes": TOTAL_STATIC_BYTES,
        "compression": table1_total_static_compression(
            "gr00t", ALL_W4_BYTES + args.budget_bytes
        ),
        "sensitivity": {
            "path": str(Path(args.sensitivity).resolve()),
            "sha256": sha256_file(args.sensitivity),
        },
        "byte_rows_manifest": {
            "path": str(Path(args.byte_rows_manifest).resolve()),
            "sha256": sha256_file(args.byte_rows_manifest),
        },
        "guard_thresholds": guards,
        "guarded_layers": sorted(guarded),
        "w_i_log": weight_log,
        "arms": {
            arm: {
                "fp16_count": results[arm]["fp16_count"],
                "w4_count": len(names) - results[arm]["fp16_count"],
                "byte_total": results[arm]["byte_total"],
                "objective": results[arm]["objective"],
                "protected_layers": results[arm]["protected"],
                "expected_counts": EXPECTED[arm],
                "matches_expected_counts": (
                    len(names) - results[arm]["fp16_count"],
                    results[arm]["fp16_count"],
                ) == (EXPECTED[arm]["w4"], EXPECTED[arm]["fp16"]),
            }
            for arm in results
        },
        "hamming_distances": {
            f"{left}_vs_{right}": hamming(
                results[left]["protected"], results[right]["protected"]
            )
            for left, right in (
                ("cs_only", "cka_only"),
                ("cs_only", "cs_cka"),
                ("cka_only", "cs_cka"),
            )
        },
    }

    for arm, plan in plans.items():
        path = out_dir / f"{arm}.plan.json"
        payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise SystemExit(f"mask changed on re-materialization: {path}")
        else:
            path.write_text(payload, encoding="utf-8")
        manifest["arms"][arm]["plan_sha256"] = sha256_file(path)

    manifest_path = out_dir / "masks_manifest.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != payload:
            raise SystemExit(f"masks manifest changed on re-materialization: {manifest_path}")
    else:
        manifest_path.write_text(payload, encoding="utf-8")

    print(json.dumps(
        {
            "weight_metric": args.weight_metric,
            "arms": {
                arm: {
                    "w4": manifest["arms"][arm]["w4_count"],
                    "fp16": manifest["arms"][arm]["fp16_count"],
                    "matches_expected": manifest["arms"][arm]["matches_expected_counts"],
                }
                for arm in results
            },
            "hamming": manifest["hamming_distances"],
            "expected_hamming_multiset": [5, 13, 18],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
