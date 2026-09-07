#!/usr/bin/env python3
"""Freeze an outcome-blind, exact-key residual rollout schedule."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tools"))

from quantvla_cross_model_protocol import sha256_file  # noqa: E402
from quantvla_outputimpact import atomic_json  # noqa: E402


ROOT = REPO / "runs/full_context_v2/pi05_fcp_diagnostic"
OUT = ROOT / "residual_schedule.json"
CONFIG = "full_context_w4a8_dynamic_profile"

POOLS = {
    "transferred_initializer": [
        ("pi05_fcp_transferred_initializer_s0", 2, 19800, 0),
        ("pi05_fcp_transferred_initializer_s1", 3, 19801, 1),
        ("pi05_fcp_transferred_initializer_s2", 4, 19802, 2),
        ("pi05_fcp_transferred_initializer_s3", 5, 19803, 3),
        ("pi05_fcp_transferred_initializer_s4", 6, 19804, 4),
        ("pi05_fcp_transferred_initializer_s8", 7, 19808, 5),
    ],
    "single_best": [
        ("pi05_fcp_single_best_s0", 3, 19820, 1),
        ("pi05_fcp_single_best_s1", 4, 19821, 1),
        ("pi05_fcp_single_best_s2", 5, 19822, 1),
        ("pi05_fcp_single_best_s3", 6, 19823, 2),
        ("pi05_fcp_single_best_s7", 7, 19827, 3),
    ],
    "two_best": [
        ("pi05_fcp_two_best_s0", 3, 19840, 4),
        ("pi05_fcp_two_best_s1", 4, 19841, 5),
        ("pi05_fcp_two_best_s2", 5, 19842, 6),
        ("pi05_fcp_two_best_s3", 6, 19843, 7),
        ("pi05_fcp_two_best_s7", 7, 19847, 4),
    ],
}


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def observed_keys(arm: str) -> set[tuple[str, str, int]]:
    seen: set[tuple[str, str, int]] = set()
    result_dir = ROOT / "rollouts" / arm / "results" / CONFIG
    for path in sorted(result_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config") != CONFIG or row.get("status") != "complete":
                continue
            key = (str(row["task_set"]), str(row["task"]), int(row["seed"]))
            if key in seen:
                raise ValueError(f"duplicate committed key before residual schedule: {arm}/{key}")
            seen.add(key)
    return seen


def main() -> None:
    prereg_path = ROOT / "preregistration.json"
    execution_path = ROOT / "execution_manifest.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    task_sets = prereg["evaluation"]["task_sets"]
    seeds = [int(seed) for seed in prereg["evaluation"]["seeds"]]
    expected = {
        (split, task, seed)
        for split, tasks in task_sets.items()
        for task in tasks
        for seed in seeds
    }
    if len(expected) != 500:
        raise ValueError("expected-key contract drift")

    runners = []
    missing_by_arm = {}
    completed_by_arm = {}
    for arm, pool in POOLS.items():
        observed = observed_keys(arm)
        missing = sorted(expected - observed)
        completed_by_arm[arm] = len(observed)
        missing_by_arm[arm] = [
            {"task_set": split, "task": task, "seed": seed}
            for split, task, seed in missing
        ]
        active_pool = pool[: min(len(pool), len(missing))]
        queues: list[list[tuple[str, str, int]]] = [[] for _ in active_pool]
        for index, key in enumerate(missing):
            queues[index % len(active_pool)].append(key)
        allowed = {row["server_metadata_sha256"] for row in execution["servers"][arm]}
        for index, ((instance, server_gpu, port, egl_gpu), jobs) in enumerate(zip(active_pool, queues)):
            runtime_path = ROOT / "control" / f"{instance}.runtime.json"
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            semantic = runtime["openpi_runtime"]["semantic_metadata_sha256"]
            if semantic not in allowed:
                raise ValueError(f"unregistered runtime semantic hash for {instance}")
            runners.append(
                {
                    "runner_id": f"residual_{arm}_q{index}",
                    "arm": arm,
                    "instance": instance,
                    "server_gpu": server_gpu,
                    "port": port,
                    "egl_gpu": egl_gpu,
                    "server_metadata_sha256": semantic,
                    "runtime": artifact(runtime_path),
                    "jobs": [
                        {"task_set": split, "task": task, "seed": seed}
                        for split, task, seed in jobs
                    ],
                }
            )

    total_missing = sum(len(rows) for rows in missing_by_arm.values())
    if total_missing <= 0:
        raise ValueError("no residual rollout keys remain")
    payload = {
        "schema_version": 1,
        "kind": "pi05_fcp_outcome_blind_exact_key_residual_schedule",
        "immutable": True,
        "scientific_protocol_changed": False,
        "model_or_artifact_changed": False,
        "runtime_semantics_changed": False,
        "statistics_changed": False,
        "success_values_used_for_scheduling": False,
        "completed_counts_before_residual_schedule": completed_by_arm,
        "missing_counts": {arm: len(rows) for arm, rows in missing_by_arm.items()},
        "total_missing": total_missing,
        "missing_keys": missing_by_arm,
        "runners": runners,
        "concurrency_rule": "one exact-key queue per retained server; one request per server",
        "egl_assignment": "GPU 0:1 runner; GPU 1:4; GPU 2:1; GPU 3:1; GPU 4:2; GPU 5--7:1 each",
        "preregistration": artifact(prereg_path),
        "execution_manifest": artifact(execution_path),
        "parent_tail_supervisor_erratum": artifact(ROOT / "tail_supervisor_erratum.json"),
        "launcher": artifact(REPO / "scripts/run_pi05_fcp_residual_workers.sh"),
    }
    if OUT.is_file():
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("immutable residual schedule drift")
    else:
        atomic_json(OUT, payload)
    print(
        json.dumps(
            {
                "path": str(OUT),
                "sha256": sha256_file(OUT),
                "total_missing": total_missing,
                "runners": len(runners),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
