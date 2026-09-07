#!/usr/bin/env python3
"""Freeze the user-directed GPU-4/5/6/7 exact-key residual schedule."""

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
OUT = ROOT / "residual_schedule_gpu4567.json"
CONFIG = "full_context_w4a8_dynamic_profile"

# Repeating an instance creates two disjoint exact-key queues for that server.
SLOTS = {
    "transferred_initializer": [
        ("pi05_fcp_transferred_initializer_s4", 6, 19804),
    ],
    "single_best": [
        ("pi05_fcp_single_best_s1", 4, 19821),
        ("pi05_fcp_single_best_s1", 4, 19821),
        ("pi05_fcp_single_best_s2", 5, 19822),
        ("pi05_fcp_single_best_s2", 5, 19822),
        ("pi05_fcp_single_best_s3", 6, 19823),
        ("pi05_fcp_single_best_s3", 6, 19823),
        ("pi05_fcp_single_best_s7", 7, 19827),
        ("pi05_fcp_single_best_s7", 7, 19827),
    ],
    "two_best": [
        ("pi05_fcp_two_best_s1", 4, 19841),
        ("pi05_fcp_two_best_s1", 4, 19841),
        ("pi05_fcp_two_best_s2", 5, 19842),
        ("pi05_fcp_two_best_s2", 5, 19842),
        ("pi05_fcp_two_best_s3", 6, 19843),
        ("pi05_fcp_two_best_s3", 6, 19843),
        ("pi05_fcp_two_best_s7", 7, 19847),
        ("pi05_fcp_two_best_s7", 7, 19847),
    ],
}


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def observed_keys(arm: str) -> set[tuple[str, str, int]]:
    seen: set[tuple[str, str, int]] = set()
    for path in sorted((ROOT / "rollouts" / arm / "results" / CONFIG).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config") != CONFIG or row.get("status") != "complete":
                continue
            key = (str(row["task_set"]), str(row["task"]), int(row["seed"]))
            if key in seen:
                raise ValueError(f"duplicate committed key: {arm}/{key}")
            seen.add(key)
    return seen


def main() -> None:
    prereg_path = ROOT / "preregistration.json"
    execution_path = ROOT / "execution_manifest.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    expected = {
        (split, task, int(seed))
        for split, tasks in prereg["evaluation"]["task_sets"].items()
        for task in tasks
        for seed in prereg["evaluation"]["seeds"]
    }
    if len(expected) != 500:
        raise ValueError("expected-key contract drift")

    runners = []
    missing_by_arm = {}
    completed_by_arm = {}
    for arm, available_slots in SLOTS.items():
        observed = observed_keys(arm)
        missing = sorted(expected - observed)
        completed_by_arm[arm] = len(observed)
        missing_by_arm[arm] = [
            {"task_set": split, "task": task, "seed": seed}
            for split, task, seed in missing
        ]
        slots = available_slots[: min(len(available_slots), len(missing))]
        queues = [[] for _ in slots]
        for index, key in enumerate(missing):
            queues[index % len(slots)].append(key)
        allowed = {row["server_metadata_sha256"] for row in execution["servers"][arm]}
        for index, ((instance, gpu, port), jobs) in enumerate(zip(slots, queues)):
            runtime_path = ROOT / "control" / f"{instance}.runtime.json"
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            semantic = runtime["openpi_runtime"]["semantic_metadata_sha256"]
            if semantic not in allowed:
                raise ValueError(f"unregistered runtime semantic hash for {instance}")
            runners.append(
                {
                    "runner_id": f"residual47_{arm}_q{index}",
                    "arm": arm,
                    "instance": instance,
                    "server_gpu": gpu,
                    "port": port,
                    "egl_gpu": gpu,
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
        raise ValueError("no residual keys remain")
    interrupted_logs = sorted((ROOT / "control" / "workers").glob("residual_*.log"))
    payload = {
        "schema_version": 1,
        "kind": "pi05_fcp_outcome_blind_gpu4567_exact_key_residual_schedule",
        "immutable": True,
        "user_directive": "use only GPUs 4,5,6,7 and fill their available memory",
        "scientific_protocol_changed": False,
        "model_or_artifact_changed": False,
        "runtime_semantics_changed": False,
        "statistics_changed": False,
        "success_values_used_for_scheduling": False,
        "completed_counts_before_schedule": completed_by_arm,
        "missing_counts": {arm: len(rows) for arm, rows in missing_by_arm.items()},
        "total_missing": total_missing,
        "missing_keys": missing_by_arm,
        "runners": runners,
        "gpu_scope": [4, 5, 6, 7],
        "concurrency_rule": (
            "At most two disjoint exact-key queues per model server; four or five "
            "evaluator processes per GPU with no key overlap."
        ),
        "preregistration": artifact(prereg_path),
        "execution_manifest": artifact(execution_path),
        "superseded_residual_schedule": artifact(ROOT / "residual_schedule.json"),
        "interrupted_residual_logs": [artifact(path) for path in interrupted_logs],
        "launcher": artifact(REPO / "scripts/run_pi05_fcp_residual_gpu4567.sh"),
    }
    if OUT.is_file():
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("immutable GPU4567 residual schedule drift")
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
