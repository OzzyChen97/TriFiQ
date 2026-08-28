#!/usr/bin/env python3
"""Freeze a Table-1 candidate manifest only after the fresh quick gate passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantvla_cross_model_protocol import sha256_file, validate_quant_plan
from quantvla_full_context import (
    PROTOCOL,
    PROTOCOL_V2,
    protocol_attestation,
    require_protocol_attestation,
)
from quantvla_outputimpact import atomic_json
from quantvla_table1_bytes import (
    TABLE1_QUANTVLA_BYTES,
    table1_total_static_bytes,
)


def artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def optional_artifact(path: str | None) -> dict[str, Any] | None:
    return None if path is None else artifact(path)


def checkpoint_tree_artifact(path: str | Path) -> dict[str, Any]:
    """Content-addressed record for a GR00T checkpoint directory tree."""
    import hashlib

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    digest = hashlib.sha256()
    total = 0
    for file_path in files:
        relative = file_path.relative_to(root).as_posix().encode()
        file_digest = sha256_file(file_path)
        size = file_path.stat().st_size
        digest.update(relative + b"\0" + file_digest.encode() + b"\0" + str(size).encode() + b"\n")
        total += size
    return {
        "path": str(root),
        "sha256_tree": digest.hexdigest(),
        "files": len(files),
        "bytes": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("gr00t", "pi05"))
    parser.add_argument("--frozen-plan", required=True)
    parser.add_argument("--activation-attribution", required=True)
    parser.add_argument("--quick-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hessian-w4", required=True)
    parser.add_argument("--a8")
    parser.add_argument("--fp16-evidence")
    parser.add_argument("--quantvla-evidence")
    parser.add_argument("--gdsq-main-evidence")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    plan_path = Path(args.frozen_plan).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_quant_plan(plan, model=args.model, source=str(plan_path))
    require_protocol_attestation(plan.get("meta") or {}, source=str(plan_path))

    quick_path = Path(args.quick_report).expanduser().resolve()
    quick = json.loads(quick_path.read_text(encoding="utf-8"))
    require_protocol_attestation(quick, source=str(quick_path))
    if quick.get("model_adapter") != args.model:
        raise ValueError("quick gate/model mismatch")
    if quick.get("advance_to_table1") is not True:
        raise ValueError("quick development gate did not pass")
    if quick.get("candidate_plan_sha256") != sha256_file(plan_path):
        raise ValueError("quick gate/frozen plan lineage mismatch")

    activation_path = Path(args.activation_attribution).expanduser().resolve()
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    require_protocol_attestation(activation, source=str(activation_path))
    selected_mode = activation["selected_activation_mode"]
    if (plan.get("meta") or {}).get("activation_mode") != selected_mode:
        raise ValueError("frozen plan/activation attribution mismatch")
    if selected_mode == "static_a8" and args.a8 is None:
        raise ValueError("static A8 Table-1 deployment requires an A8 artifact")

    all_w4 = int(plan["all_w4_total_bytes"])
    static_total = table1_total_static_bytes(args.model, int(plan["total_bytes"]))
    maximum_static = int(
        PROTOCOL_V2["byte_budget"]["maximum_quantvla_byte_multiplier"]
        * TABLE1_QUANTVLA_BYTES[args.model]
    )
    if static_total > maximum_static:
        raise ValueError(
            "frozen plan exceeds the registered Table-1 byte cap "
            f"({static_total} > {maximum_static} total-static bytes)"
        )
    table = PROTOCOL["table1"]
    tasks = table["tasks"]
    if {key: len(value) for key, value in tasks.items()} != table["task_counts"]:
        raise ValueError("Table-1 task registry/count drift")
    if len({task for values in tasks.values() for task in values}) != 50:
        raise ValueError("Table-1 task registry must contain 50 unique tasks")

    baseline_evidence = {
        "fp16": optional_artifact(args.fp16_evidence),
        "quantvla_w4a8": optional_artifact(args.quantvla_evidence),
        "gdsq_vla_main": optional_artifact(args.gdsq_main_evidence),
    }
    payload = {
        "schema_version": 1,
        "kind": "full_context_table1_frozen_manifest",
        "immutable": True,
        "result_feedback_allowed": False,
        "full_context_protocol": protocol_attestation(),
        "model_adapter": args.model,
        "model_hyperparameters": PROTOCOL["model_hyperparameters"][args.model],
        "quick_gate": artifact(quick_path),
        "activation_attribution": artifact(activation_path),
        "candidate": {
            "id": "full_context_fp16_protection_v1",
            "plan": artifact(plan_path),
            "activation_mode": selected_mode,
            "hessian_w4": artifact(args.hessian_w4),
            "a8": optional_artifact(args.a8),
            "total_bytes": int(plan["total_bytes"]),
            "all_w4_quantvla_bytes": all_w4,
            "byte_multiplier": float(plan["total_bytes"] / all_w4),
            "achieved_candidate_compression": float(plan.get("achieved_candidate_compression", plan.get("achieved_target_matrix_compression"))),
            "quantized_w4_layers": int(plan["quantized_w4_layers"]),
            "retained_fp16_layers": int(plan["retained_fp16_layers"]),
            "runtime_selector": False,
            "runtime_correction": False,
        },
        "compression_claim": {
            "byte_scope": PROTOCOL_V2["byte_budget"]["scope"],
            "budget_anchor": PROTOCOL_V2["byte_budget"]["budget_anchor"],
            "quantvla_storage_cell_bytes": TABLE1_QUANTVLA_BYTES[args.model],
            "maximum_total_static_bytes": maximum_static,
            "candidate_total_static_bytes": static_total,
            "within_quantvla_1p10": static_total <= maximum_static,
            "candidate_linear_byte_multiplier": float(plan["total_bytes"] / all_w4),
            "candidate_linear_only_compression": float(
                plan.get("achieved_candidate_compression", plan.get("achieved_target_matrix_compression"))
            ),
            "quantvla_anchor_compression": float(
                PROTOCOL["model_hyperparameters"][args.model][
                    "quantvla_anchor_compression"
                ]
            ),
            "registered_conservative_floor": float(
                PROTOCOL["byte_budget"]["minimum_approximate_compression"][args.model]
            ),
            "candidate_linear_byte_scope": PROTOCOL["byte_budget"]["scope"],
            "candidate_linear_quantvla_bytes": all_w4,
            "candidate_linear_deployment_bytes": int(plan["total_bytes"]),
            "note": (
                "The v2 budget scope is the total-static component accounting "
                "against the 1.10x QuantVLA storage cell; the linear-only "
                "multiplier is reported transparently as a secondary figure."
            ),
        },
        "checkpoint": checkpoint_tree_artifact(args.checkpoint),
        "protocol": {
            "benchmark": table["benchmark"],
            "split": table["split"],
            "task_sets": tasks,
            "trial_seeds": list(range(50)),
            "episodes": table["episodes_per_model_per_config"],
            "flow_steps": PROTOCOL["model_hyperparameters"][args.model]["table1_flow_steps"],
            "paired_action_noise": True,
            "missing_episode_policy": table["missing_episode_policy"],
            "duplicate_episode_policy": table["duplicate_episode_policy"],
        },
        "baseline_evidence": baseline_evidence,
        "baseline_reuse_rule": (
            "reuse_only_after_checkpoint_protocol_task_seed_and_raw_result_sha_attestation; "
            "otherwise rerun the affected 2500-episode configuration"
        ),
        "statistics": {
            "primary": table["primary_metric"],
            "secondary": table["secondary_metrics"],
            "multiple_comparisons": table["multiple_comparisons"],
        },
    }
    atomic_json(args.out, payload)
    print(json.dumps({"out": str(Path(args.out).resolve()), "episodes": 2500}, indent=2))


if __name__ == "__main__":
    main()
