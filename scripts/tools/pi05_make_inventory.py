#!/usr/bin/env python3
"""Freeze the π0.5 candidate inventory and full-W4 reference plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"
CONFIG_SHA256 = "3673272c04c1d5eb6bc187d087104b4912a3a33624eb6e02c2250904d593b836"
NORM_STATS_SHA256 = "4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee"
PATTERNS = (
    re.compile(
        r"^paligemma_with_expert\.paligemma\.model\.language_model\.layers\.\d+\."
        r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
    ),
    re.compile(
        r"^paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\."
        r"mlp\.(?:gate_proj|up_proj|down_proj)\.weight$"
    ),
)


def digest_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(
            REPO_ROOT
            / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/model.safetensors"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/plans"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = []
    with safe_open(checkpoint, framework="pt", device="cpu") as archive:
        keys = sorted(key for key in archive.keys() if any(pattern.match(key) for pattern in PATTERNS))
        for key in keys:
            shape = archive.get_slice(key).get_shape()
            if len(shape) != 2:
                raise RuntimeError(f"candidate is not a matrix: {key} {shape}")
            name = key.removesuffix(".weight")
            layers.append(
                {
                    "name": name,
                    "out_features": int(shape[0]),
                    "in_features": int(shape[1]),
                    "parameters": int(shape[0] * shape[1]),
                    "family": "action_expert_mlp" if ".gemma_expert." in name else "paligemma_language",
                }
            )
    if len(layers) != 180:
        raise RuntimeError(f"expected 180 candidates, got {len(layers)}")
    names = [row["name"] for row in layers]
    inventory_hash = digest_lines(names)
    inventory = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "norm_stats_sha256": NORM_STATS_SHA256,
        "candidate_inventory_sha256": inventory_hash,
        "candidate_count": len(layers),
        "families": {
            "paligemma_language": sum(row["family"] == "paligemma_language" for row in layers),
            "action_expert_mlp": sum(row["family"] == "action_expert_mlp" for row in layers),
        },
        "total_weight_parameters": sum(row["parameters"] for row in layers),
        "layers": layers,
    }
    inventory["protocol"] = "GR00T-N1.5-aligned-target16-d4"
    inventory_path = out_dir / "pi05_candidate_inventory_d4.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    full_plan = {
        "schema_version": 1,
        "layers": {
            row["name"]: {
                "bits": 4,
                "group": 64,
                "skip": False,
                "reason": "full_w4a8_reference",
            }
            for row in layers
        },
        "meta": {
            "kind": "quantvla_uniform_w4a8_gr00t_aligned",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "norm_stats_sha256": NORM_STATS_SHA256,
            "candidate_inventory_sha256": inventory_hash,
            "candidate_count": 180,
            "quantized_layers": 180,
            "block_in": 64,
            "block_out": 64,
            "enable_permute": False,
            "lambda_smooth": 0.15,
            "act_bits": 8,
            "act_percentile": 99.9,
            "calibration_batches": 32,
            "denoising_steps": 4,
            "evaluation_split": "target",
            "n_action_steps": 16,
            "paired_noise": "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1",
            "calibration_seed": 0,
            "calibration_batch_size": 8,
            "adjudicated": True,
        },
    }
    plan_path = out_dir / "pi05_quantvla_uniform_w4a8_d4.plan.json"
    plan_path.write_text(json.dumps(full_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "inventory": str(inventory_path),
                "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                "candidate_inventory_sha256": inventory_hash,
                "full_plan": str(plan_path),
                "full_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "candidate_count": len(layers),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
