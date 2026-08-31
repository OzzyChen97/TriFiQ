#!/usr/bin/env python3
"""Bind one globally selected GR00T DyPAC mask to four suite checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-mask", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    mask_path = Path(args.frozen_mask).resolve()
    mask = json.loads(mask_path.read_text(encoding="utf-8"))
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    names = {row["name"] for row in inventory["layers"]}
    if set(mask.get("layers") or {}) != names:
        raise ValueError("frozen GR00T mask/inventory mismatch")
    if mask.get("meta", {}).get("protocol_id") != PROTOCOL["protocol_id"]:
        raise ValueError("frozen GR00T mask protocol drift")
    output = Path(args.out_dir).resolve()
    results = {}
    for suite in PROTOCOL["benchmark"]["suites"]:
        plan = copy.deepcopy(mask)
        plan["meta"].update(
            {
                "kind": "dypac_libero_gr00t_suite_frozen",
                "model": "gr00t",
                "model_adapter": "gr00t",
                "suite": suite,
                "checkpoint_sha256": inventory["checkpoints"][suite]["sha256"],
                "global_mask_source": str(mask_path),
                "global_mask_source_sha256": sha256_file(mask_path),
                "protocol_sha256": sha256_file(PROTOCOL_PATH),
            }
        )
        path = output / suite / "dypac_vla_libero.frozen.plan.json"
        atomic_json(path, plan)
        results[suite] = {"path": str(path), "sha256": sha256_file(path)}
    manifest = {
        "schema_version": 1,
        "kind": "dypac_libero_gr00t_suite_plan_manifest",
        "model": "gr00t",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "global_mask": str(mask_path),
        "global_mask_sha256": sha256_file(mask_path),
        "plans": results,
    }
    atomic_json(output / "suite_plans.manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
