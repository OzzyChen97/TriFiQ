#!/usr/bin/env python3
"""Acceptance tests for the GR00T DPAC predictive-validity experiment."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aggregate_gr00t_dpac_predictive_validity import correlation, kendall, validate_inputs
from prepare_gr00t_dpac_predictive_validity import plan_for_mask
from quantvla_metric_protocol import physical_action_scale, summarize_pair
from quantvla_predictive_validity import (
    PredictiveMaskRuntime,
    generate_masks,
    physical_action_mse_summary,
    protocol_attestation,
    sha256_file,
    validate_predictive_response,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json"
BYTE_INVENTORY = ROOT / "runs/full_context_v2/p2/interventions_dynamic/manifest.json"


def source_masks():
    reference = json.loads(REFERENCE.read_text())
    byte_rows = json.loads(BYTE_INVENTORY.read_text())["byte_rows"]
    masks = generate_masks(
        set(reference["protected_layers"]), byte_rows,
        seed=20260901, swap_counts=[1, 2, 4, 8, 12, 16], per_count=10,
    )
    return reference, byte_rows, masks


def runtime_fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    reference, byte_rows, masks = source_masks()
    rows = []
    for mask in masks:
        plan = plan_for_mask(reference, mask, byte_rows)
        path = tmp_path / f"{mask['candidate_id']}.json"
        path.write_text(json.dumps(plan))
        rows.append({**mask, "path": str(path), "sha256": sha256_file(path)})
    manifest = {
        "kind": "gr00t_dpac_predictive_mask_library",
        "predictive_validity_protocol": protocol_attestation(),
        "candidate_count": 60,
        "candidates": rows,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    layers = {name: SimpleNamespace(_outputimpact_fp16=None) for name in reference["layers"]}
    return manifest_path, manifest, layers


def test_mask_generation_is_reproducible_unique_and_exact_budget():
    reference, byte_rows, masks = source_masks()
    again = generate_masks(
        set(reference["protected_layers"]), byte_rows,
        seed=20260901, swap_counts=[1, 2, 4, 8, 12, 16], per_count=10,
    )
    assert masks == again
    assert len(masks) == len({frozenset(row["protected_layers"]) for row in masks}) == 60
    assert {k: sum(row["swap_count"] == k for row in masks) for k in (1, 2, 4, 8, 12, 16)} == {
        k: 10 for k in (1, 2, 4, 8, 12, 16)
    }
    reference_hist = {}
    for name in reference["protected_layers"]:
        group = int(byte_rows[name]["extra_fp16_bytes"])
        reference_hist[group] = reference_hist.get(group, 0) + 1
    reference_bytes = sum(
        int(byte_rows[name]["w4_bytes"] if name not in reference["protected_layers"]
            else byte_rows[name]["fp16_bytes"])
        for name in reference["layers"]
    )
    for row in masks:
        protected = set(row["protected_layers"])
        histogram = {}
        for name in protected:
            group = int(byte_rows[name]["extra_fp16_bytes"])
            histogram[group] = histogram.get(group, 0) + 1
        total = sum(
            int(byte_rows[name]["fp16_bytes"] if name in protected else byte_rows[name]["w4_bytes"])
            for name in reference["layers"]
        )
        assert len(protected) == 16
        assert histogram == reference_hist
        assert total == reference_bytes == 634_716_160


def test_physical_mse_identity_and_equal_mse_dpac_separation():
    generator = torch.Generator().manual_seed(19)
    teacher = torch.randn((8, 16, 12), generator=generator)
    records = [
        {"task": "task", "seed": index // 4, "replan": index % 4}
        for index in range(8)
    ]
    assert physical_action_mse_summary(teacher, teacher, records)["mse"] == 0.0
    constant = teacher.clone()
    alternating = teacher.clone()
    constant[..., 6] += 0.1
    signs = torch.tensor([1.0, -1.0] * 8).reshape(1, 16)
    alternating[..., 6] += 0.1 * signs
    mse_a = physical_action_mse_summary(teacher, constant, records)["mse"]
    mse_b = physical_action_mse_summary(teacher, alternating, records)["mse"]
    assert mse_a == pytest.approx(mse_b, rel=0, abs=2e-11)
    scale = physical_action_scale(teacher)
    dpac_a = summarize_pair(teacher, constant, records, scale=scale)["d_pac_summary"]["d_pac"]
    dpac_b = summarize_pair(teacher, alternating, records, scale=scale)["d_pac_summary"]["d_pac"]
    assert dpac_a != pytest.approx(dpac_b, rel=1e-5, abs=1e-8)


def test_runtime_rejects_missing_unknown_drift_and_switches(tmp_path: Path):
    manifest_path, manifest, layers = runtime_fixture(tmp_path)
    runtime = PredictiveMaskRuntime(manifest_path, layers)
    with pytest.raises(ValueError, match="missing"):
        runtime.activate({})
    with pytest.raises(ValueError, match="unknown"):
        runtime.activate({"predictive_mask_id": "not_registered"})
    first, second = manifest["candidates"][0], manifest["candidates"][-1]
    response_a = runtime.activate({"predictive_mask_id": first["candidate_id"]})
    assert {name for name, layer in layers.items() if layer._outputimpact_fp16} == set(
        first["protected_layers"]
    )
    response_b = runtime.activate({"predictive_mask_id": second["candidate_id"]})
    assert response_a["candidate_id"] != response_b["candidate_id"]
    assert {name for name, layer in layers.items() if layer._outputimpact_fp16} == set(
        second["protected_layers"]
    )
    validate_predictive_response(
        response_b, candidate_id=second["candidate_id"],
        plan_sha256=second["sha256"], library_sha256=sha256_file(manifest_path),
    )
    drift = dict(response_b, plan_sha256="0" * 64)
    with pytest.raises(ValueError, match="drift"):
        validate_predictive_response(
            drift, candidate_id=second["candidate_id"],
            plan_sha256=second["sha256"], library_sha256=sha256_file(manifest_path),
        )
    plan_path = Path(first["path"])
    plan_path.write_text(plan_path.read_text() + "\n")
    with pytest.raises(ValueError, match="plan drift"):
        PredictiveMaskRuntime(manifest_path, layers)


def test_rank_statistics_handle_ties_constants_and_known_order():
    assert correlation([1, 1, 2, 3], [1, 1, 2, 3])["value"] == pytest.approx(1.0)
    assert kendall([1, 1, 2, 3], [1, 1, 2, 3])["value"] == pytest.approx(1.0)
    assert correlation([1, 2, 3], [3, 2, 1])["value"] == pytest.approx(-1.0)
    assert kendall([1, 2, 3], [3, 2, 1])["value"] == pytest.approx(-1.0)
    assert correlation([1, 1, 1], [1, 2, 3])["value"] is None
    assert kendall([1, 2, 3], [0, 0, 0])["value"] is None


def synthetic_coverage(tmp_path: Path):
    manifest_path, manifest, _layers = runtime_fixture(tmp_path / "library")
    tasks = {split: [f"{split}_{index}" for index in range(count)] for split, count in zip(
        ("atomic_seen", "composite_seen", "composite_unseen"), (18, 16, 16)
    )}
    manifest["closed_loop"] = {"environment_seed": 70, "tasks": tasks}
    manifest_path.write_text(json.dumps(manifest))
    score_path = tmp_path / "scores.json"
    scores = {
        "kind": "gr00t_dpac_predictive_offline_scores",
        "predictive_validity_protocol": protocol_attestation(),
        "mask_library": {"sha256": sha256_file(manifest_path)},
        "scores": {
            row["candidate_id"]: {
                "mse": index + 1.0,
                "d_func": index + 2.0,
                "d_pac": index + 3.0,
                "split_scores": {
                    split: {"mse": index + 1.0, "d_func": index + 2.0, "d_pac": index + 3.0}
                    for split in tasks
                },
            }
            for index, row in enumerate(manifest["candidates"])
        },
    }
    score_path.write_text(json.dumps(scores))
    mask_root, fp16_root = tmp_path / "masks", tmp_path / "fp16"
    mask_root.mkdir()
    fp16_root.mkdir()
    mask_lines = []
    for candidate in manifest["candidates"]:
        for split_tasks in tasks.values():
            for task in split_tasks:
                mask_lines.append(json.dumps({
                    "status": "complete", "task": task, "seed": 70, "success": False,
                    "predictive_mask_id": candidate["candidate_id"],
                    "predictive_plan_sha256": candidate["sha256"],
                    "predictive_library_sha256": sha256_file(manifest_path),
                    "predictive_validity_protocol": protocol_attestation(),
                    "predictive_evaluation_only": True,
                }))
    (mask_root / "rows.jsonl").write_text("\n".join(mask_lines) + "\n")
    fp16_lines = [
        json.dumps({
            "status": "complete", "task": task, "seed": 70, "success": True,
            "predictive_mask_id": None,
        })
        for split_tasks in tasks.values() for task in split_tasks
    ]
    (fp16_root / "rows.jsonl").write_text("\n".join(fp16_lines) + "\n")
    return manifest_path, manifest, score_path, scores, mask_root, fp16_root


def test_signed_drop_and_strict_missing_duplicate_coverage(tmp_path: Path):
    args = synthetic_coverage(tmp_path)
    rows, baseline, _ = validate_inputs(*args)
    assert len(rows) == 60 and len(baseline) == 50
    assert all(row["success_drop"] == pytest.approx(1.0) for row in rows)
    mask_file = args[-2] / "rows.jsonl"
    original = mask_file.read_text().splitlines()
    mask_file.write_text("\n".join(original[:-1]) + "\n")
    with pytest.raises(ValueError, match="coverage"):
        validate_inputs(*args)
    mask_file.write_text("\n".join(original + [original[0]]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        validate_inputs(*args)
