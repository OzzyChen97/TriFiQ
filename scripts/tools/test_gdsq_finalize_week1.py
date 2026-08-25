from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("gdsq_finalize_week1.py")
SPEC = importlib.util.spec_from_file_location("gdsq_finalize_week1", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def matrix_summary(tasks: int = 14, episodes: int = 700):
    per_task = {f"task_{index}": {"sr": 0.0} for index in range(tasks)}
    return {
        "manifest_sha256": "manifest",
        "bootstrap_draws": 10_000,
        "validation_errors": [],
        "primary_scope": {"n_tasks": tasks},
        "secondary_scope": {"n_tasks": tasks},
        "configs": {"control": {"episodes": episodes, "per_task": per_task}},
    }


def test_matrix_promotion_reads_nested_scope_schema() -> None:
    MODULE.require_matrix_summary(
        matrix_summary(),
        manifest_sha="manifest",
        expected_tasks=14,
        expected_configs={"control"},
    )


def test_matrix_promotion_rejects_incomplete_config() -> None:
    summary = matrix_summary(episodes=699)
    with pytest.raises(ValueError, match="episode coverage drift"):
        MODULE.require_matrix_summary(
            summary,
            manifest_sha="manifest",
            expected_tasks=14,
            expected_configs={"control"},
        )


def test_multi_config_coverage_uses_total_and_per_config_counts() -> None:
    assert MODULE.coverage(14, 3500, configurations=5) == {
        "tasks": 14,
        "seeds_per_task": 50,
        "expected_episodes": 3500,
        "observed_episodes": 3500,
        "missing_episodes": 0,
        "duplicate_episodes": 0,
        "configurations": 5,
        "expected_episodes_per_config": 700,
        "observed_episodes_per_config": 700,
    }
