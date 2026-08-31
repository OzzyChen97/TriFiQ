#!/usr/bin/env python3
"""Materialize every coordinate flip around a LIBERO DyPAC initial mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-plan", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan_path = Path(args.initial_plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    names = [row["name"] for row in inventory["layers"]]
    protected = set(plan["protected_layers"])
    candidates = {
        "context_base": {
            "protected_layers": sorted(protected),
            "flip": None,
            "total_bytes": plan["total_bytes"],
        }
    }
    byte_rows = {row["name"]: row for row in inventory["layers"]}
    for index, name in enumerate(names):
        changed = set(protected)
        if name in changed:
            changed.remove(name); direction = "fp16_to_w4"
        else:
            changed.add(name); direction = "w4_to_fp16"
        total = inventory["all_w4_bytes"] + sum(byte_rows[value]["extra_fp16_bytes"] for value in changed)
        candidates[f"flip_{index:04d}"] = {
            "protected_layers": sorted(changed),
            "flip": {"layer": name, "direction": direction},
            "total_bytes": total,
            "within_budget": total <= inventory["budget_bytes"],
        }
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_coordinate_flip_manifest",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "initial_plan": str(plan_path),
        "initial_plan_sha256": sha256_file(plan_path),
        "inventory_sha256": sha256_file(args.inventory),
        "baseline_id": "context_base",
        "candidates": candidates,
        "uses_success_labels": False,
    }
    atomic_json(args.out, payload)
    print(json.dumps({"out": str(Path(args.out).resolve()), "candidates": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()
