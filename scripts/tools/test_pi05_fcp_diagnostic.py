#!/usr/bin/env python3
"""Deterministic unit and fail-closed tests for the pi0.5 FCP diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from aggregate_pi05_fcp_diagnostic import (
    exact_mcnemar,
    hierarchical_bootstrap,
    holm_adjust,
    load_rows,
    validate_source_hashes,
)
from quantvla_cross_model_protocol import closed_loop_row_protocol


def row(task: str = "TaskA", seed: int = 0, success: bool = True) -> dict:
    return {
        **closed_loop_row_protocol(),
        "flow_steps": 4,
        "config": "full_context_w4a8_dynamic_profile",
        "task_set": "atomic_seen",
        "task": task,
        "seed": seed,
        "success": success,
        "status": "complete",
        "server_metadata_sha256": "a" * 64,
    }


class StatisticsTests(unittest.TestCase):
    def test_exact_mcnemar(self) -> None:
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_mcnemar(3, 0), 0.25)
        self.assertEqual(exact_mcnemar(0, 3), 0.25)

    def test_holm(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.04})
        self.assertEqual(adjusted, {"a": 0.02, "b": 0.04})

    def test_hierarchical_bootstrap_deterministic(self) -> None:
        left = {
            ("atomic_seen", "A", 0): {"success": True},
            ("atomic_seen", "A", 1): {"success": True},
            ("atomic_seen", "B", 0): {"success": False},
            ("atomic_seen", "B", 1): {"success": True},
        }
        right = {
            ("atomic_seen", "A", 0): {"success": False},
            ("atomic_seen", "A", 1): {"success": True},
            ("atomic_seen", "B", 0): {"success": False},
            ("atomic_seen", "B", 1): {"success": False},
        }
        first = hierarchical_bootstrap(left, right, draws=100, seed=0)
        second = hierarchical_bootstrap(left, right, draws=100, seed=0)
        self.assertEqual(first, second)
        self.assertEqual(first["observed_task_macro_delta"], 0.5)


class IntegrityTests(unittest.TestCase):
    def test_missing_duplicate_and_hash_drift_fail(self) -> None:
        expected = {("atomic_seen", "TaskA", 0), ("atomic_seen", "TaskB", 0)}
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            missing = root / "missing.jsonl"
            missing.write_text(json.dumps(row()) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage drift"):
                load_rows([missing], expected, allowed_metadata={"a" * 64}, source_label="test")

            duplicate = root / "duplicate.jsonl"
            duplicate.write_text(
                "\n".join((json.dumps(row()), json.dumps(row()), json.dumps(row("TaskB")))) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_rows([duplicate], expected, allowed_metadata={"a" * 64}, source_label="test")

            with self.assertRaisesRegex(ValueError, "hash drift"):
                validate_source_hashes(
                    [missing],
                    [{"path": str(missing.resolve()), "sha256": "0" * 64}],
                )


if __name__ == "__main__":
    unittest.main()
