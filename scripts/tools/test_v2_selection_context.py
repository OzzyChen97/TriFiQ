from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from build_v2_selection_context import (  # noqa: E402
    MAX_REPLANS,
    REPLANS_PER_SEQUENCE,
    SEEDS_PER_TASK,
    SPLITS,
    WINDOW_STARTS,
    build_spec,
    select_seeds,
    select_tasks,
    window_for,
)
from pool_v2_selection_buffer import pool as pool_shards  # noqa: E402


SHA = "v2-test-sha"


def test_select_tasks_balanced_and_disjoint() -> None:
    tasks = select_tasks(SHA)
    assert set(tasks) == set(SPLITS)
    for split in SPLITS:
        assert len(tasks[split]) == 4
    all_tasks = [task for split in SPLITS for task in tasks[split]]
    assert len(set(all_tasks)) == 12
    again = select_tasks(SHA)
    assert again == tasks


def test_select_seeds_distinct_and_deterministic() -> None:
    seeds = select_seeds(SHA, "OpenDrawer")
    assert len(seeds) == SEEDS_PER_TASK
    assert len(set(seeds)) == SEEDS_PER_TASK
    assert all(0 <= seed < 50 for seed in seeds)
    assert select_seeds(SHA, "OpenDrawer") == seeds


def test_window_for_hash_fixed() -> None:
    name, start = window_for(SHA, "CloseFridge", 7)
    assert name in WINDOW_STARTS
    assert start == WINDOW_STARTS[name]
    assert window_for(SHA, "CloseFridge", 7) == (name, start)


def test_build_spec_totals_and_checkpoint_mapping() -> None:
    spec = build_spec(SHA)
    assert spec["total_tasks"] == 12
    assert spec["total_sequences"] == 36
    assert spec["total_observations"] == 144
    for entry in spec["entries"]:
        assert entry["gr00t_checkpoint"].endswith(entry["split"] + "/checkpoint-60000")
        for seed_row in entry["seeds"]:
            assert seed_row["window_start"] in (0, 4, 8)


def write_shard(
    shard_dir: Path,
    task: str,
    seed: int,
    window_start: int,
    *,
    drift: str | None = None,
) -> None:
    replans = REPLANS_PER_SEQUENCE
    indices = list(range(window_start, window_start + replans))
    if drift == "window":
        indices = [value + 1 for value in indices]
    arrays = {
        "images": np.zeros((replans, 8, 8, 3), dtype=np.uint8),
        "wrist_images": np.zeros((replans, 8, 8, 3), dtype=np.uint8),
        "right_images": np.zeros((replans, 8, 8, 3), dtype=np.uint8),
        "states": np.zeros((replans, 16), dtype=np.float32),
        "prompts": np.array([f"p-{index}" for index in range(replans)]),
        "task_ids": np.array([task] * replans),
        "env_seeds": np.array([seed] * replans, dtype=np.int64),
        "env_steps": np.full((replans,), 3, dtype=np.int64),
        "replan_indices": np.array(indices, dtype=np.int64),
        "action_noises": np.zeros((replans, 50, 32), dtype=np.float32),
    }
    stacked = {name: np.concatenate([value, value], axis=0) for name, value in arrays.items()}
    shard = shard_dir / f"{task}_{seed}.npz"
    np.savez_compressed(shard, **stacked)
    sidecar = {
        "selection": "window",
        "window_start": window_start,
        "replans_per_trial": replans,
    }
    Path(str(shard) + ".json").write_text(json.dumps(sidecar), encoding="utf-8")


def write_all_shards(tmp_path: Path, spec: dict, *, drift: str | None = None) -> Path:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for entry in spec["entries"]:
        for seed_row in entry["seeds"]:
            write_shard(
                shard_dir,
                entry["task"],
                int(seed_row["seed"]),
                int(seed_row["window_start"]),
                drift=drift,
            )
    return shard_dir


def test_pool_exact_coverage_and_order(tmp_path: Path) -> None:
    spec = build_spec(SHA)
    shard_dir = write_all_shards(tmp_path, spec)
    out = tmp_path / "selection_buffer.npz"
    manifest = pool_shards(spec, shard_dir, out)
    assert manifest["observations"] == 144
    archive = np.load(out, allow_pickle=False)
    assert len(archive["states"]) == 144
    assert len(archive["action_noises"]) == 144
    assert archive["action_noises"].shape[1:] == (50, 32)
    tasks = [str(value) for value in archive["task_ids"]]
    assert len(set(tasks)) == 12


def test_pool_rejects_window_drift(tmp_path: Path) -> None:
    spec = build_spec(SHA)
    shard_dir = write_all_shards(tmp_path, spec, drift="window")
    with pytest.raises(ValueError, match="replan indices"):
        pool_shards(spec, shard_dir, tmp_path / "out.npz")


def test_pool_rejects_missing_shard(tmp_path: Path) -> None:
    spec = build_spec(SHA)
    shard_dir = write_all_shards(tmp_path, spec)
    first = next(iter(shard_dir.glob("*.npz")))
    first.unlink()
    Path(str(first) + ".json").unlink()
    with pytest.raises(FileNotFoundError, match="missing v2 selection shard"):
        pool_shards(spec, shard_dir, tmp_path / "out.npz")


def test_split_buffer_by_split_coverage(tmp_path: Path) -> None:
    from split_v2_buffer_by_split import split_buffer

    spec = build_spec(SHA)
    shard_dir = write_all_shards(tmp_path, spec)
    pooled = tmp_path / "selection_buffer.npz"
    pool_shards(spec, shard_dir, pooled)
    out_dir = tmp_path / "splits"
    report = split_buffer(spec, pooled, out_dir)
    for split in ("atomic_seen", "composite_seen", "composite_unseen"):
        assert report["splits"][split]["rows"] == 48
        archive = np.load(report["splits"][split]["archive"], allow_pickle=False)
        tasks = [str(value) for value in archive["task_ids"]]
        assert len(set(tasks)) == 4
        for entry in spec["entries"]:
            if entry["split"] == split:
                assert entry["task"] in set(tasks)