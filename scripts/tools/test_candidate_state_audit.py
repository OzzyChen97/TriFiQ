from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from candidate_state_teacher_audit import (  # noqa: E402
    deterministic_noise,
    j_candidate_state,
    pool_archives,
)


def write_archive(
    path: Path, rows: int, *, task_prefix: str, offset: int = 0
) -> None:
    arrays = {
        "images": np.zeros((rows, 224, 224, 3), dtype=np.uint8),
        "wrist_images": np.zeros((rows, 224, 224, 3), dtype=np.uint8),
        "right_images": np.zeros((rows, 224, 224, 3), dtype=np.uint8),
        "states": np.zeros((rows, 16), dtype=np.float32),
        "prompts": np.array([f"p-{index}" for index in range(rows)]),
        "task_ids": np.array([f"{task_prefix}-t{index}" for index in range(rows)]),
        "env_seeds": np.array([offset + index for index in range(rows)], dtype=np.int64),
        "env_steps": np.full((rows,), 3, dtype=np.int64),
        "replan_indices": np.full((rows,), 1, dtype=np.int64),
        "action_noises": np.zeros((rows, 50, 32), dtype=np.float32),
    }
    np.savez_compressed(path, **arrays)


def test_pool_concatenates_and_orders(tmp_path: Path) -> None:
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    write_archive(first, 2, task_prefix="aa")
    write_archive(second, 3, task_prefix="bb", offset=10)
    out = tmp_path / "pooled.npz"
    report = pool_archives([("main_mask", first), ("candidate", second)], out)
    assert report["rows"] == 5
    archive = np.load(out, allow_pickle=False)
    assert len(archive["states"]) == 5
    assert sorted(archive["task_ids"].tolist()) == [
        "aa-t0",
        "aa-t1",
        "bb-t0",
        "bb-t1",
        "bb-t2",
    ]
    assert report["sources"][0]["identifier"] == "main_mask"
    assert report["sources"][1]["rows"] == 3


def test_pool_rejects_duplicate_keys(tmp_path: Path) -> None:
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    write_archive(first, 2, task_prefix="dup")
    write_archive(second, 2, task_prefix="dup")
    with pytest.raises(ValueError, match="duplicate pooled candidate-state key"):
        pool_archives([("a", first), ("b", second)], tmp_path / "pooled.npz")


def test_deterministic_noise_is_stable_and_distinct() -> None:
    a = deterministic_noise("CloseFridge", 51, 3, 1)
    b = deterministic_noise("CloseFridge", 51, 3, 1)
    c = deterministic_noise("CloseFridge", 51, 4, 1)
    assert a.shape == (50, 32)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_j_candidate_state_normalized_upper_bound() -> None:
    sequences = [1.0, 2.0, 3.0, 4.0]
    score = {
        "d_func_summary": {"per_sequence": sequences},
        "d_pac_summary": {"per_sequence": [2.0 * value for value in sequences]},
    }
    result = j_candidate_state(score, {"d_func": 10.0, "d_pac": 20.0})
    mean = 2.5
    se = np.std(sequences, ddof=1) / 2.0
    assert result["entries"]["d_func"]["upper_bound"] == pytest.approx((mean + se) / 10.0)
    assert result["entries"]["d_pac"]["upper_bound"] == pytest.approx((2 * mean + 2 * se) / 20.0)
    assert result["j"] == pytest.approx((mean + se) / 10.0)


def test_j_candidate_state_rejects_non_positive_scale() -> None:
    score = {
        "d_func_summary": {"per_sequence": [1.0]},
        "d_pac_summary": {"per_sequence": [1.0]},
    }
    with pytest.raises(ValueError, match="non-positive baseline scale"):
        j_candidate_state(score, {"d_func": 0.0, "d_pac": 1.0})