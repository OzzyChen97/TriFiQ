#!/usr/bin/env python3
"""Create the checkpoint-pinned GR00T LIBERO DyPAC inventory and all-W4 plans."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
import re

from safetensors import safe_open

from gr00t_v2_common import DEFAULT_EXCLUDE, DEFAULT_INCLUDE
from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file
from quantvla_table1_bytes import table1_variable_budget


ROOT = Path(__file__).resolve().parents[2]
SUITES = tuple(PROTOCOL["benchmark"]["suites"])
CHECKPOINTS = {suite: ROOT / "checkpoints/gr00t" / f"libero-{suite}" for suite in SUITES}
TEMPLATE = ROOT / "checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial_v14_adjudicated.final_plan.json"
PROJECTION_ORDER = {
    "self_attn.q_proj": 0,
    "self_attn.k_proj": 1,
    "self_attn.v_proj": 2,
    "self_attn.o_proj": 3,
    "mlp.gate_proj": 4,
    "mlp.up_proj": 5,
    "mlp.down_proj": 6,
}


def traversal_key(name: str) -> tuple[int, int, int]:
    llm = re.search(r"language_model\.model\.layers\.(\d+)\.(.+)$", name)
    if llm:
        projection = llm.group(2)
        if projection not in PROJECTION_ORDER:
            raise ValueError(f"unknown GR00T LLM projection: {name}")
        return 0, int(llm.group(1)), PROJECTION_ORDER[projection]
    dit = re.search(r"transformer_blocks\.(\d+)\.ff\.net\.(0\.proj|2)$", name)
    if dit:
        return 1, int(dit.group(1)), 0 if dit.group(2) == "0.proj" else 1
    raise ValueError(f"unknown GR00T candidate layer: {name}")


def checkpoint_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no checkpoint shards under {directory}")
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def checkpoint_index(directory: Path) -> dict:
    return json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--checkpoint-sha256-json",
        help="Optional precomputed {suite: sha256}; otherwise all checkpoint shards are hashed.",
    )
    args = parser.parse_args()
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    names = sorted((template.get("layers") or {}), key=traversal_key)
    if len(names) != 116:
        raise RuntimeError(f"expected 116 GR00T adapter-bound layers, found {len(names)}")
    inventory_hash = hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()

    precomputed = (
        json.loads(Path(args.checkpoint_sha256_json).read_text(encoding="utf-8"))
        if args.checkpoint_sha256_json
        else {}
    )
    checkpoint_hashes = {
        suite: str(precomputed.get(suite) or checkpoint_sha256(CHECKPOINTS[suite]))
        for suite in SUITES
    }
    indexes = {suite: checkpoint_index(CHECKPOINTS[suite]) for suite in SUITES}
    rows = []
    with ExitStack() as stack:
        archives = {
            suite: {
                filename: stack.enter_context(
                    safe_open(CHECKPOINTS[suite] / filename, framework="pt", device="cpu")
                )
                for filename in sorted(set(indexes[suite].values()))
            }
            for suite in SUITES
        }
        for name in names:
            key = f"{name}.weight"
            checkpoint_files = {}
            shapes = {}
            for suite in SUITES:
                filename = indexes[suite].get(key)
                if not filename:
                    raise KeyError(f"{suite} checkpoint omits {key}")
                checkpoint_files[suite] = filename
                shapes[suite] = tuple(archives[suite][filename].get_slice(key).get_shape())
            if len(set(shapes.values())) != 1:
                raise RuntimeError(f"cross-suite GR00T shape drift for {name}: {shapes}")
            out_features, in_features = map(int, next(iter(shapes.values())))
            groups = (in_features + 63) // 64
            fp16_bytes = out_features * in_features * 2
            w4_bytes = out_features * ((in_features + 1) // 2) + out_features * groups * 4
            rows.append(
                {
                    "name": name,
                    "checkpoint_key": key,
                    "checkpoint_files": checkpoint_files,
                    "shape": [out_features, in_features],
                    "parameters": out_features * in_features,
                    "fp16_bytes": fp16_bytes,
                    "w4_bytes": w4_bytes,
                    "extra_fp16_bytes": fp16_bytes - w4_bytes,
                }
            )

    all_w4 = sum(row["w4_bytes"] for row in rows)
    fp16 = sum(row["fp16_bytes"] for row in rows)
    budget = table1_variable_budget("gr00t")
    inventory = {
        "schema_version": 1,
        "kind": "dypac_libero_gr00t_candidate_inventory",
        "model": "gr00t",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_inventory_sha256": inventory_hash,
        "candidate_layers": 116,
        "candidate_order": "native_model_named_modules",
        "checkpoints": {
            suite: {"path": str(CHECKPOINTS[suite]), "sha256": checkpoint_hashes[suite]}
            for suite in SUITES
        },
        "include_regex": DEFAULT_INCLUDE,
        "exclude_regex": DEFAULT_EXCLUDE,
        "fp16_bytes": fp16,
        "all_w4_bytes": all_w4,
        "budget_bytes": budget,
        "layers": rows,
    }
    inventory_path = output / "candidate_inventory.json"
    atomic_json(inventory_path, inventory)
    for suite in SUITES:
        plan = {
            "schema_version": 5,
            "meta": {
                "kind": "dypac_libero_all_w4_base",
                "method_id": PROTOCOL["method_id"],
                "model": "gr00t",
                "model_adapter": "gr00t",
                "suite": suite,
                "checkpoint_sha256": checkpoint_hashes[suite],
                "candidate_inventory_sha256": inventory_hash,
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
                row["name"]: {
                    "bits": 4,
                    "group": 64,
                    "skip": False,
                    "reason": "dypac_all_w4_base",
                }
                for row in rows
            },
            "all_w4_total_bytes": all_w4,
            "fp16_total_bytes": fp16,
            "budget_bytes": budget,
            "total_bytes": all_w4,
            "quantized_w4_layers": 116,
            "retained_fp16_layers": 0,
            "protected_layers": [],
        }
        atomic_json(output / suite / "all_w4.plan.json", plan)
    print(
        json.dumps(
            {
                "inventory": str(inventory_path),
                "layers": 116,
                "all_w4_bytes": all_w4,
                "budget_bytes": budget,
                "plans": list(SUITES),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
