#!/usr/bin/env python3
"""Generate a deduplicated uncertainty grid at one frozen compression budget.

This is a model-agnostic second stage for OutputImpact.  It consumes only the
per-layer D_PAC-v2 mean, jackknife standard error, reliability, and byte costs
already frozen in a valid static plan.  It never observes task success and it
does not change the deployment byte budget.  The resulting masks are intended
for full-network D_PAC-v2/noise-A scoring before any closed-loop evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from quantvla_cross_model_protocol import (
    PROTOCOL,
    protocol_attestation,
    sha256_file,
    validate_quant_plan,
)
from select_errorbudget_plan import select_protected_layers


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_floats(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item.strip()]
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError("grid values must be a non-empty finite list")
    return result


def build(
    *,
    model: str,
    reference_plan_path: str | Path,
    se_weights: list[float],
    reliability_powers: list[float],
    output_dir: str | Path,
    prefix: str,
    target_compression: float | None = None,
) -> dict[str, Any]:
    if model not in PROTOCOL["models"]:
        raise ValueError(f"unknown model adapter: {model}")
    if not all(math.isfinite(value) for value in se_weights):
        raise ValueError("SE weights must be finite")
    if not all(math.isfinite(value) and value >= 0.0 for value in reliability_powers):
        raise ValueError("reliability powers must be finite and non-negative")
    reference_path = Path(reference_plan_path).expanduser().resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    validation = validate_quant_plan(reference, model=model, source=str(reference_path))
    meta = reference.get("meta") or {}
    if meta.get("kind") != "outputimpact_static_compression_profile":
        raise ValueError("reference is not an OutputImpact static profile")
    if meta.get("model_adapter") != model:
        raise ValueError("reference model adapter mismatch")
    if bool(meta.get("uses_success_labels", False)):
        raise ValueError("reference plan used success labels")
    layers = reference.get("layers") or {}
    base_rows: dict[str, dict[str, float]] = {}
    for name, entry in layers.items():
        row = entry.get("outputimpact") or {}
        required = ("d_pac", "jackknife_se", "reliability", "fp16_bytes", "w4_bytes")
        if any(key not in row for key in required):
            raise ValueError(f"{name}: reference lacks frozen OutputImpact statistics")
        values = {key: float(row[key]) for key in required}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"{name}: non-finite OutputImpact statistic")
        if not 0.0 <= values["reliability"] <= 1.0:
            raise ValueError(f"{name}: reliability outside [0,1]")
        base_rows[name] = values
    fp16_total = sum(row["fp16_bytes"] for row in base_rows.values())
    all_w4_total = sum(row["w4_bytes"] for row in base_rows.values())
    if target_compression is None:
        budget = float(reference["budget_bytes"])
        target = float(
            (reference.get("meta") or {}).get("target_candidate_compression")
            or reference.get("achieved_candidate_compression")
        )
    else:
        target = float(target_compression)
        maximum = fp16_total / all_w4_total
        if not math.isfinite(target) or target < 1.0 or target > maximum + 1e-9:
            raise ValueError(
                f"target compression must be in [1,{maximum:.6g}], got {target!r}"
            )
        budget = fp16_total / target

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    seen_masks: dict[str, str] = {}
    for se_weight in se_weights:
        for reliability_power in reliability_powers:
            rows = copy.deepcopy(base_rows)
            for row in rows.values():
                row["risk"] = max(
                    0.0, row["d_pac"] + se_weight * row["jackknife_se"]
                ) * row["reliability"] ** reliability_power
            protected, total = select_protected_layers(rows, budget)
            mask_hash = canonical_hash(sorted(protected))
            candidate_id = (
                f"se{se_weight:+g}_rp{reliability_power:g}"
                .replace("+", "p")
                .replace("-", "m")
                .replace(".", "p")
            )
            if mask_hash in seen_masks:
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "duplicate_of": seen_masks[mask_hash],
                        "mask_sha256": mask_hash,
                        "se_weight": se_weight,
                        "reliability_power": reliability_power,
                    }
                )
                continue
            seen_masks[mask_hash] = candidate_id
            payload = copy.deepcopy(reference)
            for name, entry in payload["layers"].items():
                if name in protected:
                    entry.update(
                        {
                            "bits": None,
                            "skip": True,
                            "reason": "outputimpact_generalized_fp16_protection",
                        }
                    )
                else:
                    entry.update(
                        {
                            "bits": 4,
                            "group": 64,
                            "skip": False,
                            "reason": "outputimpact_generalized_group64_w4",
                        }
                    )
                entry["outputimpact"] = rows[name]
            total_risk = sum(row["risk"] for row in rows.values())
            protected_risk = sum(rows[name]["risk"] for name in protected)
            payload["meta"] = {
                **meta,
                "cross_model_protocol": protocol_attestation(),
                "risk_mode": "generalized",
                "se_weight": float(se_weight),
                "reliability_power": float(reliability_power),
                "selection_formula": (
                    "risk=max(0,d_pac_v2+se_weight*jackknife_se(d_pac_v2))"
                    "*reliability^reliability_power; protect descending "
                    "risk/(fp16_bytes-w4_bytes) under the frozen static byte budget"
                ),
                "risk_grid_source_sha256": sha256_file(Path(__file__).resolve()),
                "risk_grid_reference_plan_sha256": sha256_file(reference_path),
                "uses_success_labels": False,
                "selection_used_closed_loop_success": False,
                "target_compression": target,
                "target_compression_scope": "candidate",
                "target_candidate_compression": target,
                "target_model_compression": None,
                "fixed_model_bytes": None,
            }
            payload["budget_bytes"] = float(budget)
            payload["total_bytes"] = float(total)
            payload["achieved_candidate_compression"] = float(
                payload["fp16_total_bytes"] / total
            )
            payload["achieved_compression"] = payload[
                "achieved_candidate_compression"
            ]
            payload["quantized_w4_layers"] = len(rows) - len(protected)
            payload["retained_fp16_layers"] = len(protected)
            payload["protected_risk_fraction"] = (
                float(protected_risk / total_risk) if total_risk > 0.0 else 0.0
            )
            validate_quant_plan(payload, model=model, source=candidate_id)
            path = out_dir / f"{prefix}_{candidate_id}.json"
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "mask_sha256": mask_hash,
                    "se_weight": se_weight,
                    "reliability_power": reliability_power,
                    "quantized_w4_layers": payload["quantized_w4_layers"],
                    "retained_fp16_layers": payload["retained_fp16_layers"],
                    "achieved_candidate_compression": payload[
                        "achieved_candidate_compression"
                    ],
                }
            )
    manifest = {
        "schema_version": 1,
        "kind": "outputimpact_generalized_risk_grid",
        "model_adapter": model,
        "cross_model_protocol": protocol_attestation(),
        "reference_plan": str(reference_path),
        "reference_plan_sha256": sha256_file(reference_path),
        "reference_validation": validation,
        "budget_bytes": budget,
        "target_candidate_compression": target,
        "selection_metric": "full_network_d_pac_v2_noise_A",
        "selection_used_closed_loop_success": False,
        "se_weights": se_weights,
        "reliability_powers": reliability_powers,
        "unique_masks": len(seen_masks),
        "candidates": candidates,
    }
    manifest_path = out_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest": str(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=PROTOCOL["models"])
    parser.add_argument("--reference-plan", required=True)
    parser.add_argument("--se-weights", default="-2,-1.5,-1,-0.5,0,0.5,1,1.5,2")
    parser.add_argument("--reliability-powers", default="0,0.5,1")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--target-compression", type=float, default=None,
        help="optional candidate-scope compression override",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(
        model=args.model,
        reference_plan_path=args.reference_plan,
        se_weights=parse_floats(args.se_weights),
        reliability_powers=parse_floats(args.reliability_powers),
        output_dir=args.out_dir,
        prefix=args.prefix,
        target_compression=args.target_compression,
    )
    print(
        json.dumps(
            {
                "manifest": result["manifest"],
                "candidates": len(result["candidates"]),
                "unique_masks": result["unique_masks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
