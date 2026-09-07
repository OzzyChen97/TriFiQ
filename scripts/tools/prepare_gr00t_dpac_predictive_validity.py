#!/usr/bin/env python3
"""Materialize the immutable 60-mask GR00T predictive-validity library."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from gr00t_probe_outputimpact import checkpoint_sha256  # noqa: E402
from quantvla_cross_model_protocol import (  # noqa: E402
    protocol_attestation as cross_model_attestation,
    validate_quant_plan,
)
from quantvla_full_context import (  # noqa: E402
    PROTOCOL_V2 as FULL_CONTEXT_V2,
    protocol_attestation as full_context_attestation,
)
from quantvla_predictive_validity import (  # noqa: E402
    PROTOCOL,
    REPO_ROOT,
    artifact,
    atomic_json,
    generate_masks,
    is_w4,
    protocol_attestation,
    sha256_file,
    _set_state,
)


SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")
SOURCE_FILES = (
    "scripts/quantvla_dpac_predictive_protocol.json",
    "scripts/tools/quantvla_predictive_validity.py",
    "scripts/tools/prepare_gr00t_dpac_predictive_validity.py",
    "scripts/tools/gr00t_score_outputimpact_plans.py",
    "scripts/tools/combine_gr00t_dpac_predictive_scores.py",
    "scripts/inference_service.py",
    "scripts/run_robocasa365_gr00t_eval.py",
    "scripts/tools/run_gr00t_dpac_predictive_validity.py",
    "scripts/tools/audit_gr00t_predictive_routes.py",
    "scripts/tools/aggregate_gr00t_dpac_predictive_validity.py",
)


def resolve_repo_path(value: str) -> Path:
    return (REPO_ROOT / value).resolve()


def tree_artifact(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    count = 0
    total = 0
    for item in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_hash = sha256_file(item).encode("ascii")
        digest.update(file_hash)
        count += 1
        total += item.stat().st_size
    return {
        "path": str(root),
        "tree_sha256": digest.hexdigest(),
        "files": count,
        "bytes": total,
    }


def plan_for_mask(
    reference: dict[str, Any], mask: dict[str, Any], byte_rows: dict[str, Any]
) -> dict[str, Any]:
    plan = copy.deepcopy(reference)
    protected = set(mask["protected_layers"])
    for name, row in plan["layers"].items():
        _set_state(
            row,
            fp16=name in protected,
            reason="dpac_predictive_validity_matched_swap",
        )
    fp16_total = sum(int(row["fp16_bytes"]) for row in byte_rows.values())
    all_w4_total = sum(int(row["w4_bytes"]) for row in byte_rows.values())
    total = sum(
        int(byte_rows[name]["w4_bytes"] if is_w4(row) else byte_rows[name]["fp16_bytes"])
        for name, row in plan["layers"].items()
    )
    plan.update(
        {
            "fp16_total_bytes": fp16_total,
            "all_w4_total_bytes": all_w4_total,
            "total_bytes": total,
            "quantized_w4_layers": len(plan["layers"]) - len(protected),
            "retained_fp16_layers": len(protected),
            "protected_layers": sorted(protected),
            "achieved_target_matrix_compression": float(fp16_total / total),
        }
    )
    expected = PROTOCOL["mask_library"]
    if (
        plan["quantized_w4_layers"] != int(expected["expected_w4_layers"])
        or plan["retained_fp16_layers"] != int(expected["expected_fp16_layers"])
        or plan["total_bytes"] != int(expected["expected_candidate_bytes"])
        or plan["table1_total_static_bytes"]
        != int(expected["expected_table1_static_bytes"])
    ):
        raise ValueError(f"{mask['candidate_id']}: exact-budget invariant failed")
    meta = dict(plan.get("meta") or {})
    meta.update(
        {
            "kind": "gr00t_dpac_predictive_validity_mask",
            "candidate_id": mask["candidate_id"],
            "predictive_validity_protocol": protocol_attestation(),
            "swap_count": mask["swap_count"],
            "hamming_distance": mask["hamming_distance"],
            "removed_fp16_layers": mask["removed_fp16_layers"],
            "added_fp16_layers": mask["added_fp16_layers"],
            "flow_steps": int(PROTOCOL["runtime"]["flow_steps"]),
            "activation_mode": "dynamic_a8",
            "uses_success_labels": False,
            "uses_task_labels_for_routing_or_task_specific_mask": False,
            "runtime_selector": False,
            "runtime_correction": False,
            "frozen": True,
        }
    )
    plan["meta"] = meta
    validate_quant_plan(plan, model="gr00t", source=mask["candidate_id"])
    return plan


def prepare(root: Path) -> Path:
    library = PROTOCOL["mask_library"]
    reference_path = resolve_repo_path(library["reference_plan"])
    full_w4_path = resolve_repo_path(library["full_w4_plan"])
    byte_manifest_path = resolve_repo_path(library["byte_inventory"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    full_w4 = json.loads(full_w4_path.read_text(encoding="utf-8"))
    byte_manifest = json.loads(byte_manifest_path.read_text(encoding="utf-8"))
    validate_quant_plan(reference, model="gr00t", source=str(reference_path))
    full_attestation = validate_quant_plan(full_w4, model="gr00t", source=str(full_w4_path))
    if int(full_attestation["retained_fp16_target_layers"]) != 0:
        raise ValueError("predictive superset base is not full W4")
    if set(reference["layers"]) != set(full_w4["layers"]):
        raise ValueError("M0 and full-W4 inventories differ")
    byte_rows = byte_manifest.get("byte_rows") or {}
    if set(byte_rows) != set(reference["layers"]):
        raise ValueError("byte inventory differs from the M0 target inventory")
    expected = library
    if (
        int(reference["total_bytes"]) != int(expected["expected_candidate_bytes"])
        or int(reference["table1_total_static_bytes"])
        != int(expected["expected_table1_static_bytes"])
    ):
        raise ValueError("reference plan bytes drifted from the registered protocol")
    protected = set(reference.get("protected_layers") or [])
    masks = generate_masks(
        protected,
        byte_rows,
        seed=int(library["rng_seed"]),
        swap_counts=[int(value) for value in library["swap_counts"]],
        per_count=int(library["masks_per_swap_count"]),
    )

    plans_dir = root / "plans"
    rows = []
    for mask in masks:
        plan = plan_for_mask(reference, mask, byte_rows)
        path = plans_dir / f"{mask['candidate_id']}.json"
        atomic_json(path, plan, immutable=True)
        rows.append(
            {
                **mask,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "total_bytes": int(plan["total_bytes"]),
                "table1_total_static_bytes": int(plan["table1_total_static_bytes"]),
                "quantized_w4_layers": int(plan["quantized_w4_layers"]),
                "retained_fp16_layers": int(plan["retained_fp16_layers"]),
            }
        )

    split_sources = {}
    for split in SPLITS:
        checkpoint = REPO_ROOT / (
            "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
            f"target_posttraining/{split}/checkpoint-60000"
        )
        calibration = REPO_ROOT / f"runs/errorfold_v3_15x20/calibration/gr00t/{split}"
        hessian = calibration / "hessian_w4.npz"
        hessian_sidecar = Path(str(hessian) + ".json")
        selection_buffer = (
            REPO_ROOT
            / f"runs/full_context_v2/selection_buffer_splits/selection_buffer_{split}.npz"
        )
        split_sources[split] = {
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "tree_sha256": checkpoint_sha256(checkpoint),
            },
            "hessian_w4": artifact(hessian),
            "hessian_sidecar": artifact(hessian_sidecar),
            "identity_pack": tree_artifact(calibration / "identity_pack"),
            "selection_buffer": artifact(selection_buffer),
        }

    tasks = {
        split: list(FULL_CONTEXT_V2["table1"]["tasks"][split]) for split in SPLITS
    }
    if sum(len(values) for values in tasks.values()) != 50:
        raise ValueError("RoboCasa365 target task inventory is not 50")
    manifest = {
        "schema_version": 1,
        "kind": "gr00t_dpac_predictive_mask_library",
        "immutable": True,
        "predictive_validity_protocol": protocol_attestation(),
        "cross_model_protocol": cross_model_attestation(),
        "full_context_protocol": full_context_attestation(),
        "generation": {
            "rng_seed": int(library["rng_seed"]),
            "swap_counts": list(library["swap_counts"]),
            "masks_per_swap_count": int(library["masks_per_swap_count"]),
            "rule": library["swap_rule"],
            "metric_independent": True,
            "success_independent": True,
        },
        "reference_plan": artifact(reference_path),
        "full_w4_plan": artifact(full_w4_path),
        "byte_inventory": artifact(byte_manifest_path),
        "selection_buffer": artifact(resolve_repo_path(PROTOCOL["offline"]["selection_buffer"])),
        "split_sources": split_sources,
        "closed_loop": {
            "tasks": tasks,
            "environment_seed": int(PROTOCOL["closed_loop"]["environment_seed"]),
            "episodes_per_mask": 50,
            "paired_action_noise": True,
        },
        "runtime": dict(PROTOCOL["runtime"]),
        "source_files": {
            relative: artifact(REPO_ROOT / relative) for relative in SOURCE_FILES
        },
        "candidate_count": len(rows),
        "candidates": rows,
        "uses_success_labels": False,
        "result_feedback_allowed": False,
    }
    manifest_path = root / "mask_library.json"
    atomic_json(manifest_path, manifest, immutable=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(resolve_repo_path(PROTOCOL["execution"]["root"])),
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    manifest = prepare(root)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "sha256": sha256_file(manifest),
                "candidates": value["candidate_count"],
                "candidate_bytes": value["candidates"][0]["total_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
