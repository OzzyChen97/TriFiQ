#!/usr/bin/env python3
"""Unit tests for combine_full_context_split_scores.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from quantvla_cross_model_protocol import protocol_attestation as cross_attestation  # noqa: E402
from quantvla_full_context import protocol_attestation as full_attestation  # noqa: E402
from quantvla_metric_protocol import aggregate_d_pac_sequences  # noqa: E402
from combine_full_context_split_scores import SPLITS, combine, macro_mean  # noqa: E402


def make_sequence_rows(task: str, seed: int, values: list[float]) -> list[dict]:
    rows = []
    for replan, value in enumerate(values):
        rows.append(
            {
                "task": task,
                "seed": seed,
                "record_indices": [replan],
                "replan_indices": [replan],
                "d_pac_sequence": value,
                "components": {
                    "grip": 0.0,
                    "local": value,
                    "pose": value,
                    "prefix": value,
                    "stitch": value,
                },
                "d_grip": 0.0,
                "d_local": value,
                "d_pose": value,
                "d_prefix": value,
                "d_stitch": value,
                "dimension_scale": [1.0] * 12,
                "formula_id": "d_pac_v2_test",
                "protocol_sha256": "x",
                "teacher": "original_native_fp16_checkpoint",
                "executed_actions": 16,
                "n_replans": 4,
                "weights": {"grip": 1.0, "local": 1.0, "pose": 1.0, "prefix": 1.0, "stitch": 1.0},
            }
        )
    return rows


def make_document(tasks_by_seed: dict[str, list[int]]) -> dict:
    scores = {}
    for identifier in ("context_base", "flip_0000", "flip_0001"):
        pac_rows = []
        for task, seeds in tasks_by_seed.items():
            for seed in seeds:
                pac_rows += make_sequence_rows(task, seed, [0.1, 0.2, 0.3, 0.4])
        pac = aggregate_d_pac_sequences(pac_rows)
        pac.update(
            {
                "teacher": "original_native_fp16_checkpoint",
                "sequence_grouping": "paired_task_seed_sequence",
                "dimension_scale": [1.0] * 12,
                "sequences": pac_rows,
            }
        )
        func_rows = [
            {
                "task": row["task"],
                "seed": row["seed"],
                "record_indices": row["record_indices"],
                "d_func_sequence": float(row["d_pac_sequence"]) * 0.5,
            }
            for row in pac_rows
        ]
        functional = {
            "per_sequence": [row["d_func_sequence"] for row in func_rows],
            "sequences": func_rows,
            "selection_sample_unit": "paired_task_seed_sequence",
            "formula_id": "d_func_v1",
            "protocol_sha256": "x",
        }
        scores[identifier] = {
            "quantized_w4_layers": 100,
            "retained_fp16_layers": 16,
            "d_pac": float(np.mean(pac["per_sequence"])),
            "d_pac_summary": pac,
            "d_func": float(np.mean(functional["per_sequence"])),
            "d_func_summary": functional,
            "elapsed_s": 1.0,
        }
    return {
        "schema_version": 4,
        "kind": "outputimpact_static_mask_scores",
        "cross_model_protocol": cross_attestation(),
        "full_context_protocol": full_attestation(),
        "model_adapter": "gr00t",
        "teacher": "original_fp16",
        "selection_metric": "d_pac_v2",
        "selection_noise": "A",
        "noise_b_used_for_selection": False,
        "complete": True,
        "uses_success_labels": False,
        "n_obs": 12,
        "checkpoint_sha256": "c",
        "selection_buffer_sha256": "b",
        "scores": scores,
    }


def write_document(document: dict, directory: Path, name: str) -> Path:
    path = directory / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_combine_across_splits() -> None:
    tasks = {
        "atomic_seen": {"Ta": [1, 2], "Tb": [3]},
        "composite_seen": {"Tc": [4, 5], "Td": [6]},
        "composite_unseen": {"Tu": [7, 8]},
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        payloads, paths = {}, {}
        for split in SPLITS:
            paths[split] = write_document(
                make_document(tasks[split]), root, f"doc_{split}.json"
            )
            payloads[split] = json.loads(paths[split].read_text(encoding="utf-8"))
        combined = combine(payloads, paths)
        per_candidate = 4 * (2 + 1 + 2 + 1 + 2)  # 8 sequences x 4 replans
        assert combined["n_obs"] == 12 * 3
        assert combined["n_obs_per_split"] == 12
        assert set(combined["combined_splits"]) == set(SPLITS)
        for identifier, score in combined["scores"].items():
            assert len(score["d_pac_summary"]["per_sequence"]) == per_candidate
            assert len(score["d_pac_summary"]["sequences"]) == per_candidate
            assert len(score["d_func_summary"]["per_sequence"]) == per_candidate
            assert score["d_pac"] > 0.0 and score["d_func"] > 0.0
        # every synthetic task carries identical per-sequence values, so the
        # task-macro estimand is unchanged by concatenating the splits
        assert combined["scores"]["context_base"]["d_pac"] == pytest_approx(
            macro_mean(payloads[SPLITS[0]]["scores"]["context_base"])["d_pac"]
        )
        assert combined["scores"]["context_base"]["d_func"] == pytest_approx(
            macro_mean(payloads[SPLITS[0]]["scores"]["context_base"])["d_func"]
        )


def pytest_approx(value: float) -> float:
    import pytest  # noqa: F401

    return pytest.approx(value, rel=1e-12)


def test_combine_rejects_inventory_mismatch() -> None:
    tasks = {
        "atomic_seen": {"Ta": [1]},
        "composite_seen": {"Tc": [2]},
        "composite_unseen": {"Tu": [3]},
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        payloads, paths = {}, {}
        for split in SPLITS:
            document = make_document(tasks[split])
            if split == "composite_seen":
                document["scores"].pop("flip_0001")
            paths[split] = write_document(document, root, f"doc_{split}.json")
            payloads[split] = json.loads(paths[split].read_text(encoding="utf-8"))
        try:
            combine(payloads, paths)
        except ValueError as error:
            assert "inventories differ" in str(error)
        else:
            raise AssertionError("expected inventory mismatch to fail")


if __name__ == "__main__":
    test_combine_across_splits()
    test_combine_rejects_inventory_mismatch()
    print("[combine-full-context-split-scores] selftest OK")
