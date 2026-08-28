#!/usr/bin/env python3
"""Materialize every one-layer state flip around a complete W4/FP16 mask."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from quantvla_cross_model_protocol import (
    PROTOCOL as CROSS_MODEL_PROTOCOL,
    PROTOCOL_SHA256 as CROSS_MODEL_PROTOCOL_SHA256,
    protocol_attestation as cross_model_attestation,
    sha256_file,
    validate_quant_plan,
)
from quantvla_dynamic_a8_protocol import (
    protocol_attestation as dynamic_a8_protocol_attestation,
)
from quantvla_full_context import (
    PROTOCOL,
    protocol_attestation as full_context_attestation,
)
from quantvla_outputimpact import atomic_json


def byte_rows(fp16_capture: Path, hessian_w4: Path) -> dict[str, dict[str, int]]:
    with np.load(fp16_capture, allow_pickle=False) as fp16, np.load(
        hessian_w4, allow_pickle=False
    ) as hessian:
        fp16_names = [str(value) for value in fp16["layer_names"].tolist()]
        hessian_names = [str(value) for value in hessian["layer_names"].tolist()]
        if fp16_names != hessian_names:
            raise ValueError("FP16 capture and Hessian layer inventories differ")
        rows = {}
        for index, name in enumerate(fp16_names):
            weight = np.asarray(fp16[f"weight_{index:04d}"])
            packed = np.asarray(hessian[f"packed_{index:04d}"])
            scales = np.asarray(hessian[f"scales_{index:04d}"])
            fp16_bytes = int(weight.size * 2)
            w4_bytes = int(packed.nbytes + scales.nbytes)
            if not 0 < w4_bytes < fp16_bytes:
                raise ValueError(f"{name}: invalid W4/FP16 byte accounting")
            rows[name] = {
                "fp16_bytes": fp16_bytes,
                "w4_bytes": w4_bytes,
                "extra_fp16_bytes": fp16_bytes - w4_bytes,
            }
    return rows


def is_w4(row: dict[str, Any]) -> bool:
    return not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4


def set_state(row: dict[str, Any], *, fp16: bool, reason: str) -> None:
    if fp16:
        row.update({"bits": None, "skip": True, "reason": reason})
    else:
        row.update({"bits": 4, "group": 64, "skip": False, "reason": reason})
    for forbidden in ("errorfold", "blocksoftfold", "atm", "ohb"):
        row.pop(forbidden, None)


def plan_bytes(plan: dict[str, Any], rows: dict[str, dict[str, int]]) -> int:
    return int(
        sum(rows[name]["w4_bytes"] if is_w4(entry) else rows[name]["fp16_bytes"]
            for name, entry in plan["layers"].items())
    )


def finalize_plan(
    plan: dict[str, Any],
    *,
    model: str,
    activation_mode: str,
    rows: dict[str, dict[str, int]],
    base_sha: str,
    current_sha: str,
    round_index: int,
    candidate_id: str,
    flow_steps: int,
) -> dict[str, Any]:
    total = plan_bytes(plan, rows)
    fp16_total = sum(row["fp16_bytes"] for row in rows.values())
    all_w4_total = sum(row["w4_bytes"] for row in rows.values())
    budget = int(math.floor(
        float(PROTOCOL["byte_budget"]["maximum_quantvla_byte_multiplier"])
        * all_w4_total
    ))
    protected = sorted(name for name, entry in plan["layers"].items() if not is_w4(entry))
    meta = dict(plan.get("meta") or {})
    for forbidden in ("errorfold", "atm", "ohb", "blocksoftfold", "runtime_selector"):
        meta.pop(forbidden, None)
    meta.update(
        {
            "kind": "full_context_fp16_protection_intervention",
            "method_id": PROTOCOL["method_id"],
            "candidate_id": candidate_id,
            "round": int(round_index),
            "model_adapter": model,
            "activation_mode": activation_mode,
            "flow_steps": int(flow_steps),
            "cross_model_protocol": cross_model_attestation(),
            "full_context_protocol": full_context_attestation(),
            "dynamic_a8_protocol": (
                dynamic_a8_protocol_attestation()
                if activation_mode == "dynamic_a8"
                else None
            ),
            "base_full_w4_plan_sha256": base_sha,
            "current_plan_sha256": current_sha,
            "target_compression_scope": "candidate",
            "target_compression": float(fp16_total / budget),
            "uses_cka": False,
            "uses_cs": False,
            "uses_task_labels": False,
            "uses_success_labels": False,
            "uses_gradients": False,
            "uses_extra_training_data": False,
            "runtime_selector": False,
            "runtime_correction": False,
        }
    )
    plan["schema_version"] = max(int(plan.get("schema_version", 1)), 5)
    plan["meta"] = meta
    plan.update(
        {
            "budget_bytes": budget,
            "fp16_total_bytes": fp16_total,
            "all_w4_total_bytes": all_w4_total,
            "total_bytes": total,
            "achieved_candidate_compression": float(fp16_total / total),
            "achieved_compression": float(fp16_total / total),
            "quantized_w4_layers": len(rows) - len(protected),
            "retained_fp16_layers": len(protected),
            "protected_layers": protected,
        }
    )
    if total > budget:
        # Single flips are allowed to exceed the final deployment budget only
        # when probing a currently over-budget input plan.  Such plans cannot
        # enter proposal generation and the manifest records the violation.
        plan["probe_only_budget_excess_bytes"] = total - budget
    else:
        plan.pop("probe_only_budget_excess_bytes", None)
    validate_quant_plan(plan, model=model, source=candidate_id)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=CROSS_MODEL_PROTOCOL["models"])
    parser.add_argument("--base-full-w4-plan", required=True)
    parser.add_argument("--current-plan", required=True)
    parser.add_argument("--fp16-capture", required=True)
    parser.add_argument("--hessian-w4", required=True)
    parser.add_argument(
        "--activation-mode", choices=("static_a8", "dynamic_a8"), required=True
    )
    parser.add_argument("--round", type=int, choices=(1, 2), required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_full_w4_plan).expanduser().resolve()
    current_path = Path(args.current_plan).expanduser().resolve()
    capture_path = Path(args.fp16_capture).expanduser().resolve()
    hessian_path = Path(args.hessian_w4).expanduser().resolve()
    for path in (base_path, current_path, capture_path, hessian_path, Path(str(hessian_path) + ".json")):
        if not path.is_file():
            raise FileNotFoundError(path)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    base_selection = validate_quant_plan(base, model=args.model, source=str(base_path))
    validate_quant_plan(current, model=args.model, source=str(current_path))
    if base_selection["retained_fp16_target_layers"] != 0:
        raise ValueError("base-full-w4-plan is not all W4")
    if set(base["layers"]) != set(current["layers"]):
        raise ValueError("base and current plan inventories differ")
    rows = byte_rows(capture_path, hessian_path)
    if set(rows) != set(base["layers"]):
        raise ValueError("plan and capture/Hessian inventories differ")
    hessian_meta = json.loads(Path(str(hessian_path) + ".json").read_text(encoding="utf-8"))
    if hessian_meta.get("protocol_sha256") != CROSS_MODEL_PROTOCOL_SHA256:
        raise ValueError("Hessian artifact cross-model protocol drift")
    if hessian_meta.get("capture_sha256") != sha256_file(capture_path):
        raise ValueError("Hessian artifact does not descend from the supplied FP16 capture")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_sha = sha256_file(base_path)
    current_sha = sha256_file(current_path)
    candidates = []
    flow_steps = int(PROTOCOL["model_hyperparameters"][args.model]["table1_flow_steps"])
    with np.load(capture_path, allow_pickle=False) as capture:
        captured_flow_steps = int(
            np.asarray(capture["capture_flow_steps"]).item()
            if "capture_flow_steps" in capture
            else CROSS_MODEL_PROTOCOL["closed_loop"]["flow_steps"]
        )
    if captured_flow_steps != flow_steps:
        raise ValueError(
            f"FP16/Hessian capture flow-step drift: {captured_flow_steps} != {flow_steps}"
        )

    def write_candidate(identifier: str, plan: dict[str, Any], flip: dict[str, Any] | None) -> None:
        finalized = finalize_plan(
            plan,
            model=args.model,
            activation_mode=args.activation_mode,
            rows=rows,
            base_sha=base_sha,
            current_sha=current_sha,
            round_index=args.round,
            candidate_id=identifier,
            flow_steps=flow_steps,
        )
        path = out_dir / f"{identifier}.json"
        atomic_json(path, finalized)
        candidates.append(
            {
                "candidate_id": identifier,
                "path": str(path),
                "sha256": sha256_file(path),
                "flip": flip,
                "total_bytes": finalized["total_bytes"],
                "retained_fp16_layers": finalized["retained_fp16_layers"],
                "protected_layers": finalized["protected_layers"],
                "within_budget": finalized["total_bytes"] <= finalized["budget_bytes"],
            }
        )

    write_candidate("context_base", copy.deepcopy(current), None)
    for index, name in enumerate(sorted(current["layers"])):
        plan = copy.deepcopy(current)
        from_state = "w4" if is_w4(plan["layers"][name]) else "fp16"
        to_fp16 = from_state == "w4"
        set_state(
            plan["layers"][name],
            fp16=to_fp16,
            reason=(
                "full_context_counterfactual_restore_fp16"
                if to_fp16
                else "full_context_counterfactual_requantize_w4"
            ),
        )
        write_candidate(
            f"flip_{index:04d}",
            plan,
            {"layer": name, "from": from_state, "to": "fp16" if to_fp16 else "w4"},
        )

    manifest = {
        "schema_version": 1,
        "kind": "full_context_single_flip_manifest",
        "method_id": PROTOCOL["method_id"],
        "cross_model_protocol": cross_model_attestation(),
        "full_context_protocol": full_context_attestation(),
        "model_adapter": args.model,
        "activation_mode": args.activation_mode,
        "flow_steps": flow_steps,
        "round": args.round,
        "base_full_w4_plan": str(base_path),
        "base_full_w4_plan_sha256": base_sha,
        "current_plan": str(current_path),
        "current_plan_sha256": current_sha,
        "fp16_capture": str(capture_path),
        "fp16_capture_sha256": sha256_file(capture_path),
        "hessian_w4": str(hessian_path),
        "hessian_w4_sha256": sha256_file(hessian_path),
        "byte_rows": rows,
        "all_w4_total_bytes": sum(row["w4_bytes"] for row in rows.values()),
        "fp16_total_bytes": sum(row["fp16_bytes"] for row in rows.values()),
        "budget_bytes": int(math.floor(
            PROTOCOL["byte_budget"]["maximum_quantvla_byte_multiplier"]
            * sum(row["w4_bytes"] for row in rows.values())
        )),
        "selection_noise": "A",
        "uses_success_labels": False,
        "candidates": candidates,
    }
    manifest_path = out_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "candidates": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()
