#!/usr/bin/env python3
"""Create the exact π0.5 LIBERO DyPAC candidate inventory and all-W4 base plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from safetensors import safe_open

from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file
from quantvla_table1_bytes import table1_variable_budget


PATTERNS = (
    re.compile(r"^paligemma_with_expert\.paligemma\.model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"),
    re.compile(r"^paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight$"),
)


PROJECTION_ORDER = {
    "self_attn.q_proj": 0,
    "self_attn.k_proj": 1,
    "self_attn.v_proj": 2,
    "self_attn.o_proj": 3,
    "mlp.gate_proj": 4,
    "mlp.up_proj": 5,
    "mlp.down_proj": 6,
}


def model_traversal_key(key: str) -> tuple[int, int, int]:
    """Mirror ``nn.Module.named_modules`` order used by the runtime loader."""
    if ".paligemma.model.language_model.layers." in key:
        family = 0
        suffix = key.split(".language_model.layers.", 1)[1]
    elif ".gemma_expert.model.layers." in key:
        family = 1
        suffix = key.split(".gemma_expert.model.layers.", 1)[1]
    else:
        raise ValueError(f"unknown π0.5 candidate key: {key}")
    layer_text, projection = suffix.split(".", 1)
    projection = projection.removesuffix(".weight")
    if projection not in PROJECTION_ORDER:
        raise ValueError(f"unknown π0.5 candidate projection: {projection}")
    return family, int(layer_text), PROJECTION_ORDER[projection]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with safe_open(checkpoint, framework="pt", device="cpu") as archive:
        keys = sorted(
            (key for key in archive.keys() if any(pattern.match(key) for pattern in PATTERNS)),
            key=model_traversal_key,
        )
        if len(keys) != 180:
            raise RuntimeError(f"expected 180 adapter-bound layers, found {len(keys)}")
        for key in keys:
            shape = tuple(archive.get_slice(key).get_shape())
            out_features, in_features = map(int, shape)
            name = key.removesuffix(".weight")
            groups = (in_features + 63) // 64
            fp16_bytes = out_features * in_features * 2
            w4_bytes = out_features * ((in_features + 1) // 2) + out_features * groups * 4
            rows.append({
                "name": name,
                "checkpoint_key": key,
                "shape": [out_features, in_features],
                "parameters": out_features * in_features,
                "fp16_bytes": fp16_bytes,
                "w4_bytes": w4_bytes,
                "extra_fp16_bytes": fp16_bytes - w4_bytes,
            })
    inventory_sha = hashlib.sha256(("\n".join(row["name"] for row in rows) + "\n").encode()).hexdigest()
    all_w4 = sum(row["w4_bytes"] for row in rows)
    fp16 = sum(row["fp16_bytes"] for row in rows)
    budget = table1_variable_budget("pi05")
    inventory = {
        "schema_version": 1,
        "kind": "dypac_libero_pi05_candidate_inventory",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "candidate_inventory_sha256": inventory_sha,
        "candidate_layers": len(rows),
        "fp16_bytes": fp16,
        "all_w4_bytes": all_w4,
        "budget_bytes": budget,
        "layers": rows,
    }
    inventory_path = output / "candidate_inventory.json"
    atomic_json(inventory_path, inventory)
    plan = {
        "schema_version": 5,
        "meta": {
            "kind": "dypac_libero_all_w4_base",
            "method_id": PROTOCOL["method_id"],
            "model": "pi05",
            "model_adapter": "pi05",
            "checkpoint_sha256": args.checkpoint_sha256,
            "candidate_inventory_sha256": inventory_sha,
            "protocol_id": PROTOCOL["protocol_id"],
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "activation_mode": "dynamic_a8",
            "flow_steps": 10,
            "act_bits": 8,
            "block_in": 64,
            "block_out": 64,
            "row_rotation": "identity",
            "uses_success_labels": False,
            "uses_test_rollout_feedback": False,
            "runtime_selector": False,
            "runtime_correction": False,
        },
        "layers": {
            row["name"]: {"bits": 4, "group": 64, "skip": False, "reason": "dypac_all_w4_base"}
            for row in rows
        },
        "all_w4_total_bytes": all_w4,
        "fp16_total_bytes": fp16,
        "budget_bytes": budget,
        "total_bytes": all_w4,
        "quantized_w4_layers": 180,
        "retained_fp16_layers": 0,
        "protected_layers": [],
    }
    atomic_json(output / "all_w4.plan.json", plan)
    print(json.dumps({"inventory": str(inventory_path), "layers": 180, "all_w4_bytes": all_w4, "budget_bytes": budget}, indent=2))


if __name__ == "__main__":
    main()
