#!/usr/bin/env python3
"""Select a static W4/FP16 plan from single-layer physical-action impact.

The selector is deliberately model agnostic.  A model adapter produces one
``outputimpact_single_layer_dpac_v2`` artifact by intervening on exactly one
Hessian-W4A8 Linear at a time while all other candidate Linears execute their
original FP16 weights.  This file applies the same reliability shrinkage,
byte accounting, and compression-budget rule to GR00T N1.5 and pi0.5.

No CKA, CS, task success, gradients, or additional training data enter the
score.  The output is a selector-free static deployment plan.
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
from quantvla_dynamic_a8_protocol import (
    protocol_attestation as dynamic_a8_protocol_attestation,
    require_protocol_attestation as require_dynamic_a8_protocol_attestation,
)
from select_errorbudget_plan import select_protected_layers


KIND = "outputimpact_single_layer_dpac_v2"
RISK_FORMULAS = {
    "reliability_shrunk": (
        "risk=d_pac_v2*theta^2/(theta^2+jackknife_se(d_pac_v2)^2+1e-12)"
    ),
    "raw": "risk=d_pac_v2",
    "upper_confidence": "risk=d_pac_v2+jackknife_se(d_pac_v2)",
    "lower_confidence": "risk=max(0,d_pac_v2-jackknife_se(d_pac_v2))",
    "generalized": (
        "risk=max(0,d_pac_v2+se_weight*jackknife_se(d_pac_v2))"
        "*reliability^reliability_power"
    ),
}
FORMULA_SUFFIX = (
    "; protect descending risk/(fp16_bytes-w4_bytes) under the static byte budget"
)


def _mean_cvar(values: np.ndarray, alpha: float = 0.9) -> float:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("sequence losses must be a non-empty finite vector")
    tail_count = max(1, int(math.ceil((1.0 - alpha) * values.size)))
    tail = np.partition(values, values.size - tail_count)[-tail_count:]
    return float(values.mean() + tail.mean())


def jackknife_se(sequence_losses: list[float] | np.ndarray) -> float:
    values = np.asarray(sequence_losses, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        return 0.0
    leave_one_out = np.asarray(
        [_mean_cvar(np.delete(values, index)) for index in range(values.size)],
        dtype=np.float64,
    )
    center = float(leave_one_out.mean())
    return float(
        math.sqrt(
            (values.size - 1.0)
            / values.size
            * float(np.square(leave_one_out - center).sum())
        )
    )


def reliability_shrunk_risk(
    row: Mapping[str, Any], *, risk_mode: str = "reliability_shrunk",
    se_weight: float = 0.0, reliability_power: float = 0.0,
) -> dict[str, float]:
    theta = float(row["d_pac"])
    if not math.isfinite(theta) or theta < 0.0:
        raise ValueError("single-layer D_PAC must be finite and non-negative")
    sequence_losses = row.get("per_sequence")
    if not isinstance(sequence_losses, list) or not sequence_losses:
        raise ValueError("single-layer impact row lacks per_sequence losses")
    se = jackknife_se(sequence_losses)
    reliability = theta * theta / (theta * theta + se * se + 1e-12)
    if risk_mode == "reliability_shrunk":
        risk = theta * reliability
    elif risk_mode == "raw":
        risk = theta
    elif risk_mode == "upper_confidence":
        risk = theta + se
    elif risk_mode == "lower_confidence":
        risk = max(0.0, theta - se)
    elif risk_mode == "generalized":
        if not math.isfinite(se_weight):
            raise ValueError("se_weight must be finite")
        if not math.isfinite(reliability_power) or reliability_power < 0.0:
            raise ValueError("reliability_power must be finite and non-negative")
        risk = max(0.0, theta + float(se_weight) * se) * (
            reliability ** float(reliability_power)
        )
    else:
        raise ValueError(f"unsupported OutputImpact risk mode: {risk_mode}")
    return {
        "d_pac": theta,
        "jackknife_se": se,
        "reliability": float(np.clip(reliability, 0.0, 1.0)),
        "risk": float(risk),
    }


def _byte_rows(
    fp16_capture_path: Path,
    hessian_path: Path,
    impact: Mapping[str, Any],
    *,
    risk_mode: str = "reliability_shrunk",
    se_weight: float = 0.0,
    reliability_power: float = 0.0,
) -> dict[str, dict[str, float]]:
    impact_layers = impact.get("layers") or {}
    result: dict[str, dict[str, float]] = {}
    with np.load(fp16_capture_path, allow_pickle=False) as fp16, np.load(
        hessian_path, allow_pickle=False
    ) as hessian:
        fp16_names = [str(value) for value in fp16["layer_names"].tolist()]
        hessian_names = [str(value) for value in hessian["layer_names"].tolist()]
        if fp16_names != hessian_names:
            raise ValueError("FP16 capture and Hessian layer inventories differ")
        if set(impact_layers) != set(fp16_names):
            missing = sorted(set(fp16_names) - set(impact_layers))
            extra = sorted(set(impact_layers) - set(fp16_names))
            raise ValueError(
                f"output-impact inventory mismatch: missing={missing[:3]} extra={extra[:3]}"
            )
        for index, name in enumerate(fp16_names):
            weight = np.asarray(fp16[f"weight_{index:04d}"])
            packed = np.asarray(hessian[f"packed_{index:04d}"])
            scales = np.asarray(hessian[f"scales_{index:04d}"])
            risk = reliability_shrunk_risk(
                impact_layers[name], risk_mode=risk_mode,
                se_weight=se_weight,
                reliability_power=reliability_power,
            )
            result[name] = {
                **risk,
                "fp16_bytes": float(weight.size * 2),
                "w4_bytes": float(packed.nbytes + scales.nbytes),
            }
    return result


def _candidate_budget(
    *,
    fp16_candidate_bytes: float,
    all_w4_candidate_bytes: float,
    target_compression: float | None,
    target_model_compression: float | None,
    fixed_model_bytes: float | None,
) -> tuple[float, dict[str, float | None]]:
    if (target_compression is None) == (target_model_compression is None):
        raise ValueError(
            "specify exactly one of target_compression or target_model_compression"
        )
    if target_compression is not None:
        target = float(target_compression)
        if not math.isfinite(target) or target < 1.0:
            raise ValueError("target_compression must be finite and >= 1")
        budget = fp16_candidate_bytes / target
        maximum = fp16_candidate_bytes / all_w4_candidate_bytes
        if target > maximum + 1e-9:
            raise ValueError(
                f"candidate target {target:g} exceeds all-W4 maximum {maximum:.6g}"
            )
        return budget, {
            # Canonical cross-model knob.  Keep the explicit candidate alias
            # for artifact readability and backward-compatible audits.
            "target_compression": target,
            "target_compression_scope": "candidate",
            "target_candidate_compression": target,
            "target_model_compression": None,
            "fixed_model_bytes": None,
        }

    target = float(target_model_compression)
    fixed = float(fixed_model_bytes) if fixed_model_bytes is not None else math.nan
    if not math.isfinite(target) or target < 1.0:
        raise ValueError("target_model_compression must be finite and >= 1")
    if not math.isfinite(fixed) or fixed < 0.0:
        raise ValueError("fixed_model_bytes must be finite and non-negative")
    fp16_model_bytes = fixed + fp16_candidate_bytes
    budget = fp16_model_bytes / target - fixed
    maximum = fp16_model_bytes / (fixed + all_w4_candidate_bytes)
    if target > maximum + 1e-9 or budget < all_w4_candidate_bytes - 1e-6:
        raise ValueError(
            f"model target {target:g} exceeds all-W4 maximum {maximum:.6g}"
        )
    return budget, {
        "target_compression": None,
        "target_compression_scope": "model",
        "target_candidate_compression": None,
        "target_model_compression": target,
        "fixed_model_bytes": fixed,
    }


def build(
    *,
    model: str,
    base_plan_path: str | Path,
    fp16_capture_path: str | Path,
    hessian_path: str | Path,
    outputimpact_path: str | Path,
    output_path: str | Path,
    target_compression: float | None = None,
    target_model_compression: float | None = None,
    fixed_model_bytes: float | None = None,
    risk_mode: str = "reliability_shrunk",
    se_weight: float = 0.0,
    reliability_power: float = 0.0,
) -> dict[str, Any]:
    if model not in PROTOCOL["models"]:
        raise ValueError(f"unknown model adapter: {model}")
    base_path = Path(base_plan_path).expanduser().resolve()
    capture_path = Path(fp16_capture_path).expanduser().resolve()
    hessian = Path(hessian_path).expanduser().resolve()
    impact_path = Path(outputimpact_path).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    impact = json.loads(impact_path.read_text(encoding="utf-8"))
    require_protocol_attestation(impact, source=str(impact_path))
    activation_mode = impact.get("activation_mode")
    if activation_mode == "dynamic_a8":
        require_dynamic_a8_protocol_attestation(impact, source=str(impact_path))
    if impact.get("kind") != KIND or impact.get("complete") is not True:
        raise ValueError("output-impact artifact is incomplete or has the wrong kind")
    if impact.get("model_adapter") != model:
        raise ValueError("output-impact model adapter mismatch")
    if impact.get("teacher") != "original_fp16":
        raise ValueError("output-impact teacher must be original_fp16")
    if impact.get("selection_metric") != "d_pac_v2":
        raise ValueError("output-impact selection metric must be d_pac_v2")
    for forbidden in (
        "uses_cka",
        "uses_cs",
        "uses_task_success",
        "uses_success_labels",
        "uses_gradients",
        "uses_extra_training_data",
    ):
        if bool(impact.get(forbidden, False)):
            raise ValueError(f"output-impact artifact violates {forbidden}=false")
    if impact.get("base_full_w4_plan_sha256") != sha256_file(base_path):
        raise ValueError("output-impact/base-plan lineage mismatch")
    if impact.get("hessian_w4_sha256") != sha256_file(hessian):
        raise ValueError("output-impact/Hessian lineage mismatch")
    hessian_meta = json.loads(Path(str(hessian) + ".json").read_text(encoding="utf-8"))
    if hessian_meta.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("Hessian artifact protocol drift")

    if risk_mode not in RISK_FORMULAS:
        raise ValueError(f"unsupported OutputImpact risk mode: {risk_mode}")
    rows = _byte_rows(
        capture_path, hessian, impact, risk_mode=risk_mode,
        se_weight=se_weight,
        reliability_power=reliability_power,
    )
    if set(rows) != set(base.get("layers") or {}):
        raise ValueError("base full-W4 plan and output-impact inventories differ")
    fp16_total = sum(row["fp16_bytes"] for row in rows.values())
    all_w4_total = sum(row["w4_bytes"] for row in rows.values())
    budget, budget_meta = _candidate_budget(
        fp16_candidate_bytes=fp16_total,
        all_w4_candidate_bytes=all_w4_total,
        target_compression=target_compression,
        target_model_compression=target_model_compression,
        fixed_model_bytes=fixed_model_bytes,
    )
    protected, total = select_protected_layers(rows, budget)
    payload = copy.deepcopy(base)
    for name, entry in payload["layers"].items():
        if name in protected:
            entry.update(
                {"bits": None, "skip": True, "reason": "outputimpact_fp16_protection"}
            )
        else:
            entry.update(
                {"bits": 4, "group": 64, "skip": False, "reason": "outputimpact_group64_w4"}
            )
        entry["outputimpact"] = rows[name]

    fixed = budget_meta["fixed_model_bytes"]
    fp16_model_total = None if fixed is None else float(fixed + fp16_total)
    deployed_model_total = None if fixed is None else float(fixed + total)
    total_risk = sum(row["risk"] for row in rows.values())
    protected_risk = sum(rows[name]["risk"] for name in protected)
    payload["schema_version"] = max(int(payload.get("schema_version", 1)), 4)
    payload["meta"] = {
        **dict(payload.get("meta") or {}),
        "kind": "outputimpact_static_compression_profile",
        "cross_model_protocol": protocol_attestation(),
        "dynamic_a8_protocol": (
            dynamic_a8_protocol_attestation()
            if activation_mode == "dynamic_a8"
            else None
        ),
        "activation_mode": activation_mode,
        "model_adapter": model,
        "selection_formula": RISK_FORMULAS[risk_mode] + FORMULA_SUFFIX,
        "risk_mode": risk_mode,
        "se_weight": float(se_weight) if risk_mode == "generalized" else None,
        "reliability_power": (
            float(reliability_power) if risk_mode == "generalized" else None
        ),
        "selector_source_sha256": sha256_file(Path(__file__).resolve()),
        "uses_cka": False,
        "uses_cs": False,
        "uses_task_labels": False,
        "uses_success_labels": False,
        "uses_gradients": False,
        "uses_extra_training_data": False,
        "compression_scope": "adapter_bound_weight_matrices_plus_group64_dequant_scales",
        "fp16_capture_sha256": sha256_file(capture_path),
        "hessian_w4_sha256": sha256_file(hessian),
        "outputimpact_sha256": sha256_file(impact_path),
        "base_full_w4_plan_sha256": sha256_file(base_path),
        **budget_meta,
    }
    payload.update(
        {
            "budget_bytes": float(budget),
            "fp16_total_bytes": float(fp16_total),
            "all_w4_total_bytes": float(all_w4_total),
            "total_bytes": float(total),
            "achieved_candidate_compression": float(fp16_total / total),
            "achieved_compression": float(fp16_total / total),
            "fp16_model_total_bytes": fp16_model_total,
            "deployed_model_total_bytes": deployed_model_total,
            "achieved_model_compression": (
                None
                if fp16_model_total is None
                else float(fp16_model_total / deployed_model_total)
            ),
            "quantized_w4_layers": len(rows) - len(protected),
            "retained_fp16_layers": len(protected),
            "protected_risk_fraction": (
                float(protected_risk / total_risk) if total_risk > 0.0 else 0.0
            ),
        }
    )
    validate_quant_plan(payload, model=model, source="generated OutputImpact plan")
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
    parser.add_argument("--output-impact", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-compression", type=float)
    target.add_argument("--target-model-compression", type=float)
    parser.add_argument("--fixed-model-bytes", type=float)
    parser.add_argument(
        "--risk-mode",
        choices=tuple(RISK_FORMULAS),
        default="reliability_shrunk",
        help=(
            "Offline-only uncertainty treatment used to generate joint-mask "
            "refinement candidates. The final candidate is still selected by "
            "full-network D_PAC-v2 on noise-A."
        ),
    )
    parser.add_argument(
        "--se-weight", type=float, default=0.0,
        help="generalized mode: coefficient on jackknife SE",
    )
    parser.add_argument(
        "--reliability-power", type=float, default=0.0,
        help="generalized mode: exponent on continuous reliability",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build(
        model=args.model,
        base_plan_path=args.base_plan,
        fp16_capture_path=args.fp16_capture,
        hessian_path=args.hessian,
        outputimpact_path=args.output_impact,
        output_path=args.out,
        target_compression=args.target_compression,
        target_model_compression=args.target_model_compression,
        fixed_model_bytes=args.fixed_model_bytes,
        risk_mode=args.risk_mode,
        se_weight=args.se_weight,
        reliability_power=args.reliability_power,
    )
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve()),
                "achieved_candidate_compression": payload[
                    "achieved_candidate_compression"
                ],
                "achieved_model_compression": payload[
                    "achieved_model_compression"
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
