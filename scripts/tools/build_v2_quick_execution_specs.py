#!/usr/bin/env python3
"""Build the frozen v2 quick execution specs (gdsq_main vs frozen winner).

Two configurations per task set:

* gdsq_main     historical main mask on the historical main runtime
                (static A8, exact deployment plan + regenerated A8 scales),
                copied verbatim from the four-config attribution h rows;
* full_context_v2  the P2-frozen winner mask on the common runtime
                (Hessian group64 parent artifact, row_rotation=0,
                dynamic A8, corrections off).

Tasks and seeds come from the frozen v2 quick spec (2/2/1 hash-drawn tasks,
seeds 60-69).  Placement (gpu/port/egl) is intentionally omitted; the
launcher rewrites it at execution time.  Specs are frozen before launch.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from quantvla_cross_model_protocol import sha256_file, validate_quant_plan
from quantvla_outputimpact import atomic_json

TASK_SETS = ("atomic_seen", "composite_seen", "composite_unseen")
PLACEMENT_FIELDS = (
    "gpu",
    "port",
    "egl_device",
    "result_files",
    "shard_egl_devices",
    "config_sha256",
)


def strip_placement(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key not in PLACEMENT_FIELDS
    }


def flatten_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "plan",
        "act_scale",
        "hessian_w4",
        "errorfold",
        "omega_pack",
        "omega_calibration_manifest",
        "atm",
        "packdir",
    ):
        value = config.get(key)
        if isinstance(value, dict):
            config[key] = value.get("path")
    return config


def artifact_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_spec(
    attribution_specs: dict[str, dict[str, Any]],
    quick_spec: dict[str, Any],
    winner: dict[str, Any],
    winner_path: Path,
    calib_root: Path,
    main_plan_path: Path,
    subset_hessian_root: Path,
    task_set: str,
) -> dict[str, Any]:
    attribution = attribution_specs[task_set]
    h = None
    for row in attribution["configs"]:
        if row.get("id") == "h":
            h = strip_placement(row)
    if h is None:
        raise ValueError(f"{task_set}: attribution spec lacks the h config")
    h.update({"id": "gdsq_main", "source_config_id": "h"})
    h = flatten_artifacts(h)
    # Inventory-exact W4 artifact: when the winner protects exactly the
    # historical main mask's FP16 layers, the deployment subset hessian
    # (100 layers) matches the wrapped inventory; otherwise the 116-layer
    # parent artifact is required.
    main_plan = json.loads(main_plan_path.read_text(encoding="utf-8"))
    main_protected = sorted(
        name for name, row in main_plan["layers"].items() if bool(row.get("skip", False))
    )
    winner_protected = sorted(winner.get("protected_layers") or [])
    expected_wrapped = int(winner.get("quantized_w4_layers") or 0)
    if expected_wrapped != len(winner.get("layers") or {}) - len(winner_protected):
        raise ValueError("winner quantized-layer accounting drift")
    if winner_protected == main_protected:
        hessian = subset_hessian_root / task_set / "hessian_w4.npz"
    else:
        hessian = calib_root / task_set / "hessian_w4.npz"
    candidate = {
        "id": "full_context_v2",
        "expected_wrapped": expected_wrapped,
        "plan": str(winner_path),
        "plan_sha256": sha256_file(winner_path),
        "packdir": str(calib_root / task_set / "identity_pack"),
        "hessian_w4": artifact_row(hessian),
        "act_scale": None,
        "errorfold": None,
        "omega_pack": None,
        "omega_pack_attestation": None,
        "omega_calibration_manifest": None,
        "omega_include": None,
        "atm": None,
        "ohb": False,
        "ohb_only": False,
        "activation_mode": "dynamic_a8",
        "meta": {
            "role": "frozen_full_context_v2_candidate",
            "frozen_plan_sha256": sha256_file(winner_path),
        },
    }
    candidate = flatten_artifacts(candidate)
    tasks = list(quick_spec["tasks"][task_set])
    return {
        "schema_version": 1,
        "kind": "full_context_v2_quick_execution_spec",
        "purpose": "v2_quick_gdsq_main_vs_frozen_winner",
        "task_set": task_set,
        "seeds": "60-69",
        "tasks": tasks,
        "per_split_task_counts": quick_spec["per_split_task_counts"],
        "episodes_per_config": quick_spec["episodes_per_config"],
        "quick_spec_sha256": sha256_file(
            Path("/home1/gyy/vla/QuantVLA/runs/full_context_v2/quick_spec.json")
        ),
        "advance_if": quick_spec["advance_if"],
        "configs": [h, candidate],
        "comparisons": {
            "pairs": [["gdsq_main", "full_context_v2"]],
            "unit": "paired_task_seed",
        },
        "decision": {
            "role": "development_advancement_gate",
            "advance_requires": "all_advance_if_rules",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attribution-specs-dir",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v1/gr00t/attribution/specs",
    )
    parser.add_argument(
        "--quick-spec",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v2/quick_spec.json",
    )
    parser.add_argument("--frozen-winner", required=True)
    parser.add_argument(
        "--calib-root",
        default="/home1/gyy/vla/QuantVLA/runs/errorfold_v3_15x20/calibration/gr00t",
    )
    parser.add_argument(
        "--main-plan",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v2/gr00t_main_pruned_to_table1_budget.json",
    )
    parser.add_argument(
        "--subset-hessian-root",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v2/gr00t_main_hessian",
    )
    parser.add_argument(
        "--out-dir",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v2/quick/specs",
    )
    args = parser.parse_args()
    attribution_dir = Path(args.attribution_specs_dir).expanduser().resolve()
    quick_path = Path(args.quick_spec).expanduser().resolve()
    winner_path = Path(args.frozen_winner).expanduser().resolve()
    calib_root = Path(args.calib_root).expanduser().resolve()
    main_plan_path = Path(args.main_plan).expanduser().resolve()
    subset_hessian_root = Path(args.subset_hessian_root).expanduser().resolve()
    quick_spec = json.loads(quick_path.read_text(encoding="utf-8"))
    if quick_spec.get("kind") != "full_context_v2_quick_spec":
        raise ValueError(f"{quick_path}: wrong spec kind")
    winner = json.loads(winner_path.read_text(encoding="utf-8"))
    validate_quant_plan(winner, model="gr00t", source=str(winner_path))
    if (winner.get("meta") or {}).get("frozen") is not True:
        raise ValueError(f"{winner_path}: winner plan is not frozen")
    attribution_specs = {
        task_set: json.loads(
            (attribution_dir / f"{task_set}.json").read_text(encoding="utf-8")
        )
        for task_set in TASK_SETS
    }
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_set in TASK_SETS:
        spec = build_spec(
            attribution_specs,
            quick_spec,
            winner,
            winner_path,
            calib_root,
            main_plan_path,
            subset_hessian_root,
            task_set,
        )
        atomic_json(out_dir / f"{task_set}.json", spec)
    print(json.dumps({"specs": str(out_dir), "task_sets": list(TASK_SETS)}, indent=2))


if __name__ == "__main__":
    main()
