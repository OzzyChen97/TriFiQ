#!/usr/bin/env python3
"""Select a static W4/FP16 compression profile from calibration-only error.

The formula and byte accounting are model agnostic.  GR00T N1.5 and pi0.5
only provide adapter-bound layer names and tensor shapes through their frozen
full-W4 captures.  No CKA, CS, task label, success label, gradient, or extra
training data enters the selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from quantvla_cross_model_protocol import (
    PROTOCOL,
    PROTOCOL_SHA256,
    protocol_attestation,
    require_protocol_attestation,
    sha256_file,
    validate_quant_plan,
)


def select_protected_layers(
    rows: Mapping[str, Mapping[str, float]], budget_bytes: float
) -> tuple[set[str], float]:
    """Greedily spend the FP16 budget on maximum risk removed per byte."""
    if not math.isfinite(float(budget_bytes)) or float(budget_bytes) <= 0.0:
        raise ValueError("budget_bytes must be finite and positive")
    total = sum(float(row["w4_bytes"]) for row in rows.values())
    if total > float(budget_bytes) + 1e-6:
        raise ValueError("target compression is infeasible below the all-W4 bytes")
    protected: set[str] = set()
    ranked = sorted(
        rows.items(),
        key=lambda item: (
            -float(item[1]["risk"]) / max(
                float(item[1]["fp16_bytes"]) - float(item[1]["w4_bytes"]), 1.0
            ),
            -float(item[1]["risk"]),
            item[0],
        ),
    )
    for name, row in ranked:
        delta = float(row["fp16_bytes"]) - float(row["w4_bytes"])
        if delta < -1e-6:
            raise ValueError(f"{name}: FP16 bytes are smaller than W4 bytes")
        if total + delta <= float(budget_bytes) + 1e-6:
            protected.add(name)
            total += delta
    return protected, total


def _risk_rows(
    fp16_capture: Path, hessian_path: Path, raw_errorfold: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    correction_layers = raw_errorfold.get("layers") or {}
    result: dict[str, dict[str, float]] = {}
    with np.load(fp16_capture, allow_pickle=False) as fp16, np.load(
        hessian_path, allow_pickle=False
    ) as hessian:
        names = [str(value) for value in fp16["layer_names"].tolist()]
        hessian_names = [str(value) for value in hessian["layer_names"].tolist()]
        if names != hessian_names:
            raise ValueError("FP16 and Hessian layer inventories differ")
        for index, name in enumerate(names):
            entry = correction_layers.get(name)
            if not entry or entry.get("kind") != "linear":
                raise ValueError(f"raw ErrorFold lacks Linear statistics for {name}")
            output = np.asarray(fp16[f"outputs_{index:04d}"], dtype=np.float64)
            weight = np.asarray(fp16[f"weight_{index:04d}"])
            teacher_energy = max(float(np.mean(np.square(output))), 1e-12)
            local_mse = float(
                np.mean(np.asarray(hessian[f"error_{index:04d}"], dtype=np.float64))
            )
            local_relative_mse = max(local_mse / teacher_energy, 0.0)
            stabilized_relative_mse = max(
                float(entry["stabilized_mse"]) / teacher_energy, 0.0
            )
            stability = np.asarray(
                entry["stability_reliability"], dtype=np.float64
            )
            mean_stability = float(np.clip(stability.mean(), 0.0, 1.0))
            amplification = float(
                np.clip(
                    stabilized_relative_mse / max(local_relative_mse, 1e-12),
                    1.0,
                    16.0,
                )
            )
            # Causal local W4 reconstruction is the anchor.  Full-network
            # residual supplies only a bounded amplification, preventing late
            # layers from receiving all upstream error as their own risk.
            risk = local_relative_mse * math.sqrt(amplification) * (
                1.0 + (1.0 - mean_stability)
            )
            w4_bytes = int(np.asarray(hessian[f"packed_{index:04d}"]).nbytes)
            w4_bytes += int(np.asarray(hessian[f"scales_{index:04d}"]).nbytes)
            fp16_bytes = int(weight.size * 2)
            result[name] = {
                "risk": float(risk),
                "local_relative_mse": float(local_relative_mse),
                "stabilized_relative_mse": float(stabilized_relative_mse),
                "network_amplification": amplification,
                "mean_stability_reliability": mean_stability,
                "w4_bytes": float(w4_bytes),
                "fp16_bytes": float(fp16_bytes),
            }
    return result


def build(
    *,
    model: str,
    base_plan_path: str | Path,
    fp16_capture_path: str | Path,
    hessian_path: str | Path,
    raw_errorfold_path: str | Path,
    target_compression: float,
    output_path: str | Path,
) -> dict[str, Any]:
    if model not in PROTOCOL["models"]:
        raise ValueError(f"unknown model adapter: {model}")
    if not math.isfinite(float(target_compression)) or target_compression < 1.0:
        raise ValueError("target_compression must be finite and >= 1")
    base_path = Path(base_plan_path).expanduser().resolve()
    capture_path = Path(fp16_capture_path).expanduser().resolve()
    hessian = Path(hessian_path).expanduser().resolve()
    raw_path = Path(raw_errorfold_path).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    require_protocol_attestation(raw, source=str(raw_path))
    if raw.get("meta", {}).get("plan_sha256") != sha256_file(base_path):
        raise ValueError("raw ErrorFold and base full-W4 plan lineage differ")
    hessian_meta = json.loads(Path(str(hessian) + ".json").read_text(encoding="utf-8"))
    if hessian_meta.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("Hessian artifact protocol drift")
    rows = _risk_rows(capture_path, hessian, raw)
    if set(rows) != set(base.get("layers") or {}):
        raise ValueError("base full-W4 plan and calibration inventories differ")
    fp16_total = sum(row["fp16_bytes"] for row in rows.values())
    all_w4_total = sum(row["w4_bytes"] for row in rows.values())
    budget = fp16_total / float(target_compression)
    maximum_compression = fp16_total / all_w4_total
    if target_compression > maximum_compression + 1e-9:
        raise ValueError(
            f"target compression {target_compression:g} exceeds all-W4 maximum "
            f"{maximum_compression:.6g}"
        )
    protected, total = select_protected_layers(rows, budget)
    payload = copy.deepcopy(base)
    for name, entry in payload["layers"].items():
        if name in protected:
            entry.update(
                {
                    "bits": None,
                    "skip": True,
                    "reason": "errorbudget_fp16_protection",
                }
            )
        else:
            entry.update(
                {
                    "bits": 4,
                    "group": 64,
                    "skip": False,
                    "reason": "errorbudget_group64_w4",
                }
            )
        entry["errorbudget"] = rows[name]
    total_risk = sum(row["risk"] for row in rows.values())
    protected_risk = sum(rows[name]["risk"] for name in protected)
    payload["schema_version"] = max(int(payload.get("schema_version", 1)), 4)
    payload["meta"] = {
        **dict(payload.get("meta") or {}),
        "kind": "errorbudget_static_compression_profile",
        "cross_model_protocol": protocol_attestation(),
        "model_adapter": model,
        "selection_formula": (
            "local_relative_hessian_mse * sqrt(clipped_full_network_amplification) "
            "* (1 + 1 - mean_stability_reliability), protected by risk/extra_byte"
        ),
        "uses_cka": False,
        "uses_cs": False,
        "uses_task_labels": False,
        "uses_success_labels": False,
        "target_compression": float(target_compression),
        "compression_scope": "adapter_bound_weight_matrices_plus_group64_dequant_scales",
        "fp16_capture_sha256": sha256_file(capture_path),
        "hessian_w4_sha256": sha256_file(hessian),
        "raw_errorfold_sha256": sha256_file(raw_path),
        "base_full_w4_plan_sha256": sha256_file(base_path),
    }
    payload.update(
        {
            "target_compression": float(target_compression),
            "budget_bytes": float(budget),
            "fp16_total_bytes": float(fp16_total),
            "all_w4_total_bytes": float(all_w4_total),
            "total_bytes": float(total),
            "achieved_candidate_compression": float(fp16_total / total),
            "maximum_candidate_compression": float(maximum_compression),
            "quantized_w4_layers": len(rows) - len(protected),
            "retained_fp16_layers": len(protected),
            "protected_risk_fraction": (
                float(protected_risk / total_risk) if total_risk > 0.0 else 0.0
            ),
        }
    )
    validate_quant_plan(payload, model=model, source="generated ErrorBudget plan")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=PROTOCOL["models"])
    parser.add_argument("--base-plan", required=True)
    parser.add_argument("--fp16-capture", required=True)
    parser.add_argument("--hessian", required=True)
    parser.add_argument("--raw-errorfold", required=True)
    parser.add_argument("--target-compression", required=True, type=float)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build(
        model=args.model,
        base_plan_path=args.base_plan,
        fp16_capture_path=args.fp16_capture,
        hessian_path=args.hessian,
        raw_errorfold_path=args.raw_errorfold,
        target_compression=args.target_compression,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve()),
                "target_compression": payload["target_compression"],
                "achieved_candidate_compression": payload[
                    "achieved_candidate_compression"
                ],
                "quantized_w4_layers": payload["quantized_w4_layers"],
                "retained_fp16_layers": payload["retained_fp16_layers"],
                "protected_risk_fraction": payload["protected_risk_fraction"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
