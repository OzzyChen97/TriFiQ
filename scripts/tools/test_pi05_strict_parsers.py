#!/usr/bin/env python3
"""Synthetic rejection tests for the formal π0.5 Table-1 parser."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import tempfile

import robocasa  # noqa: F401
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/tools"))

from aggregate_pi05_robocasa365 import CONFIG_ORDER, aggregate, load_rows  # noqa: E402


NOISE_PROTOCOL = "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"


def manifest() -> dict:
    task_sets = {
        key: list(TASK_SET_REGISTRY[key])
        for key in ("atomic_seen", "composite_seen", "composite_unseen")
    }
    return {
        "table_1_protocol": {
            "task_sets": task_sets,
            "trial_seeds": list(range(50)),
            "paired_action_noise_protocol": NOISE_PROTOCOL,
        },
        "servers": [
            {
                "config_id": config,
                "server_metadata_sha256": f"runtime-{config}",
            }
            for config in CONFIG_ORDER
        ],
        "artifacts": {
            "inventory": {
                "path": str(
                    REPO_ROOT / "runs/pi05_gdsq_port/plans/pi05_candidate_inventory.json"
                )
            },
            "gdsq_plan": {
                "path": str(
                    REPO_ROOT
                    / "runs/pi05_gdsq_final/adjudication/"
                    "pi05_ckaonly_adjudicated.final_plan.json"
                )
            },
        },
    }


def row(config: str, task_set: str, task: str, seed: int) -> dict:
    return {
        "status": "complete",
        "config": config,
        "task_set": task_set,
        "task": task,
        "seed": seed,
        "split": "target",
        "success": False,
        "steps": int(get_task_horizon(task)),
        "max_steps": int(get_task_horizon(task)),
        "replans": 1,
        "replan_steps": 16,
        "n_action_steps": 16,
        "flow_steps": 4,
        "action_horizon": 50,
        "paired_action_noise": True,
        "action_noise_protocol": NOISE_PROTOCOL,
        "fresh_environment": True,
        "render_enabled": True,
        "server_metadata_sha256": f"runtime-{config}",
        "episode_wall_seconds": 1.0,
        "env_construct_seconds": 0.1,
        "inference_seconds": 0.1,
        "env_step_seconds": 0.1,
        "server_infer_ms_mean": 1.0,
    }


def write_matrix(root: Path, spec: dict) -> None:
    task_sets = spec["table_1_protocol"]["task_sets"]
    for config in CONFIG_ORDER:
        directory = root / "results" / config
        directory.mkdir(parents=True, exist_ok=True)
        rows = [
            row(config, task_set, task, seed)
            for task_set, tasks in task_sets.items()
            for task, seed in itertools.product(tasks, range(50))
        ]
        (directory / "synthetic.jsonl").write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows),
            encoding="utf-8",
        )


def expect_value_error(callable_, contains: str) -> None:
    try:
        callable_()
    except ValueError as error:
        if contains not in str(error):
            raise AssertionError(f"unexpected error: {error}") from error
    else:
        raise AssertionError(f"parser accepted invalid matrix; expected {contains!r}")


def main() -> None:
    spec = manifest()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "manifest.json").write_text(json.dumps(spec), encoding="utf-8")
        write_matrix(root, spec)
        parsed = load_rows(root, spec)
        assert all(len(parsed[config]) == 2500 for config in CONFIG_ORDER)

        target = root / "results" / "fp16" / "synthetic.jsonl"
        original = target.read_text(encoding="utf-8")
        first = original.splitlines()[0]
        target.write_text(original + first + "\n", encoding="utf-8")
        expect_value_error(lambda: load_rows(root, spec), "duplicate result key")

        lines = original.splitlines()
        invalid = json.loads(lines[0])
        invalid["seed"] = 50
        lines[0] = json.dumps(invalid)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        expect_value_error(lambda: load_rows(root, spec), "protocol checks failed")

        lines = original.splitlines()
        invalid = json.loads(lines[0])
        invalid["server_metadata_sha256"] = "wrong-plan-runtime"
        lines[0] = json.dumps(invalid)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        expect_value_error(lambda: load_rows(root, spec), "protocol checks failed")

        target.write_text("\n".join(original.splitlines()[1:]) + "\n", encoding="utf-8")
        expect_value_error(
            lambda: aggregate(root, n_boot=10, allow_incomplete=False),
            "formal matrix is incomplete",
        )
    print("[pi05 strict parser] duplicate/wrong-seed/wrong-runtime/missing rejection OK")


if __name__ == "__main__":
    main()
