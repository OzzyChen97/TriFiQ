from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_budget_projection_comparison as runner


def test_gpu_scope():
    assert runner.GPUS == (0, 1, 2, 3)


def test_missing_parity_blocks_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT", tmp_path)
    assert runner.parity_reports() is None


def test_failed_parity_blocks_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT", tmp_path)
    path = tmp_path / "parity/gr00t_atomic_seen.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"failure": "action mismatch"}))
    with pytest.raises(RuntimeError, match="Parity failed"):
        runner.parity_reports()


def test_strict_paired_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT", tmp_path)
    job = {"id": "test", "candidate": "arm", "model": "gr00t", "tasks": ["task"], "seeds": [20000]}
    path = tmp_path / "rollouts/test/arm_s0.jsonl"
    path.parent.mkdir(parents=True)
    row = {"config": "arm", "task": "task", "seed": 20000, "status": "complete", "paired_action_noise": True}
    path.write_text(json.dumps(row) + "\n")
    seen, expected = runner.completed_keys(job)
    assert seen == expected
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        runner.completed_keys(job)
    row["status"] = "error"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Invalid episode"):
        runner.completed_keys(job)


def test_execution_manifest_immutable(tmp_path):
    path = tmp_path / "execution.json"
    runner.save(path, {"v": 1}, immutable=True)
    with pytest.raises(ValueError, match="Immutable"):
        runner.save(path, {"v": 2}, immutable=True)
    assert runner.read(path) == {"v": 1}


def semantic_metadata(payload):
    stable = copy.deepcopy(payload)
    runtime = stable["openpi_runtime"]
    runtime.pop("gpu_memory_bytes", None)
    runtime.pop("semantic_metadata_sha256", None)
    claimed = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["openpi_runtime"]["semantic_metadata_sha256"] = claimed
    return claimed


def test_pi05_semantic_metadata_hash_ignores_allocator_counters():
    metadata = {
        "openpi_runtime": {
            "duquant": {"plan_sha256": "abc"},
            "gpu_memory_bytes": {"allocated": 1},
        }
    }
    claimed = semantic_metadata(metadata)
    assert runner.pi05_semantic_metadata_hash(metadata) == claimed
    metadata["openpi_runtime"]["gpu_memory_bytes"]["allocated"] = 2
    assert runner.pi05_semantic_metadata_hash(metadata) == claimed


def test_pi05_semantic_metadata_hash_rejects_semantic_drift():
    metadata = {"openpi_runtime": {"duquant": {"plan_sha256": "abc"}}}
    semantic_metadata(metadata)
    metadata["openpi_runtime"]["duquant"]["plan_sha256"] = "changed"
    with pytest.raises(ValueError, match="semantic metadata hash mismatch"):
        runner.pi05_semantic_metadata_hash(metadata)
