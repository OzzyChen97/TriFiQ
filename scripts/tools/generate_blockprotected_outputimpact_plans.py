#!/usr/bin/env python3
"""Generate static compression profiles protected by BlockSoftFold evidence.

Active attention blocks come from the frozen, FP16-relative dual-loss search.
The same deterministic rule then raises the protection risk of every target
Linear in those blocks before re-solving the unchanged byte budget.  This is
an offline structural candidate grid; rollout success is never consumed.
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


def parse_factors(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item.strip()]
    if not result or not all(math.isfinite(item) and item >= 1.0 for item in result):
        raise ValueError("protection factors must be finite and >= 1")
    return result


def block_prefix(attention_name: str) -> str:
    suffix = ".self_attn::attention_logits"
    if not attention_name.endswith(suffix):
        raise ValueError(f"unsupported attention block name: {attention_name}")
    return attention_name[: -len(suffix)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=PROTOCOL["models"])
    parser.add_argument("--reference-plan", required=True)
    parser.add_argument("--block-selection", required=True)
    parser.add_argument("--protection-factors", default="2,4,8,16")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    reference_path = Path(args.reference_plan).expanduser().resolve()
    selection_path = Path(args.block_selection).expanduser().resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_quant_plan(reference, model=args.model, source=str(reference_path))
    if selection.get("kind") != "blockwise_errorfold_coordinate_selection":
        raise ValueError("block selection has the wrong kind")
    if selection.get("selection", {}).get("uses_rollout_success") is not False:
        raise ValueError("block selection must be success-label-free")
    if selection.get("selection", {}).get("uses_task_labels") is not False:
        raise ValueError("block selection must be task-label-free")
    active_attention = sorted(
        name for name, gate in selection["profile"].items() if float(gate) != 0.0
    )
    if not active_attention:
        raise ValueError("block selection has no active attention blocks")
    prefixes = [block_prefix(name) for name in active_attention]
    base_rows = {
        name: {key: float(value) for key, value in entry["outputimpact"].items()}
        for name, entry in reference["layers"].items()
    }
    budget = float(reference["budget_bytes"])
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_masks: set[str] = set()
    candidates = []
    for factor in parse_factors(args.protection_factors):
        rows = copy.deepcopy(base_rows)
        protected_block_layers = []
        for name, row in rows.items():
            in_active_block = any(name.startswith(prefix + ".") for prefix in prefixes)
            row["blocksoftfold_active_block"] = in_active_block
            row["base_risk"] = float(row["risk"])
            if in_active_block:
                row["risk"] *= factor
                protected_block_layers.append(name)
        protected, total = select_protected_layers(rows, budget)
        mask_hash = canonical_hash(sorted(protected))
        candidate_id = f"blockprotect_f{factor:g}".replace(".", "p")
        if mask_hash in seen_masks:
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "duplicate": True,
                    "mask_sha256": mask_hash,
                    "protection_factor": factor,
                }
            )
            continue
        seen_masks.add(mask_hash)
        payload = copy.deepcopy(reference)
        for name, entry in payload["layers"].items():
            if name in protected:
                entry.update(
                    {"bits": None, "skip": True, "reason": "blocksoftfold_fp16_protection"}
                )
            else:
                entry.update(
                    {"bits": 4, "group": 64, "skip": False, "reason": "blocksoftfold_group64_w4"}
                )
            entry["outputimpact"] = rows[name]
        payload["meta"] = {
            **dict(reference.get("meta") or {}),
            "kind": "blocksoftfold_outputimpact_static_compression_profile",
            "cross_model_protocol": protocol_attestation(),
            "method_id": "blocksoftfold-structural-protection-v1",
            "selection_formula": (
                "base_outputimpact_risk*block_protection_factor for target Linears "
                "sharing a transformer block with a nonzero frozen BlockSoftFold ATM; "
                "protect descending risk/byte_saved under unchanged static byte budget"
            ),
            "block_protection_factor": factor,
            "active_attention_blocks": active_attention,
            "active_block_target_layers": protected_block_layers,
            "block_selection_path": str(selection_path),
            "block_selection_sha256": sha256_file(selection_path),
            "reference_plan_sha256": sha256_file(reference_path),
            "uses_task_labels": False,
            "uses_success_labels": False,
            "selection_used_closed_loop_success": False,
            "uses_gradients": False,
            "uses_extra_training_data": False,
        }
        payload["total_bytes"] = float(total)
        payload["achieved_candidate_compression"] = float(
            payload["fp16_total_bytes"] / total
        )
        payload["achieved_compression"] = payload["achieved_candidate_compression"]
        payload["quantized_w4_layers"] = len(rows) - len(protected)
        payload["retained_fp16_layers"] = len(protected)
        total_risk = sum(row["risk"] for row in rows.values())
        payload["protected_risk_fraction"] = (
            sum(rows[name]["risk"] for name in protected) / total_risk
            if total_risk > 0.0
            else 0.0
        )
        validate_quant_plan(payload, model=args.model, source=candidate_id)
        path = out_dir / f"{args.prefix}_{candidate_id}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "mask_sha256": mask_hash,
                "protection_factor": factor,
                "quantized_w4_layers": payload["quantized_w4_layers"],
                "achieved_candidate_compression": payload[
                    "achieved_candidate_compression"
                ],
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "blocksoftfold_outputimpact_plan_grid",
        "method_id": "blocksoftfold-structural-protection-v1",
        "model_adapter": args.model,
        "reference_plan": str(reference_path),
        "reference_plan_sha256": sha256_file(reference_path),
        "block_selection": str(selection_path),
        "block_selection_sha256": sha256_file(selection_path),
        "active_attention_blocks": active_attention,
        "budget_bytes": budget,
        "selection_metric": "d_func_v1_plus_d_pac_v2_noise_A",
        "selection_used_closed_loop_success": False,
        "candidates": candidates,
    }
    manifest_path = out_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "active_blocks": len(active_attention),
                "unique_masks": len(seen_masks),
                "candidates": candidates,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
