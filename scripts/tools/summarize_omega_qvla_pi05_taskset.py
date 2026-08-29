#!/usr/bin/env python3
"""Create an audited task-set summary for the pi0.5 Omega-QVLA run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

from omega_qvla_pi05_robocasa365_eval import CONFIG, load_rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def summarize(run_dir: Path, bootstrap_draws: int) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest, rows, errors = load_rows(run_dir, require_complete=True)
    if errors:
        raise ValueError(f"validated run contains errors: {errors[:5]}")
    task_details: dict[str, dict[str, Any]] = {}
    task_values: list[float] = []
    for task in manifest["tasks"]:
        task_rows = [rows[(task, seed)] for seed in manifest["seeds"]]
        successes = sum(bool(row["success"]) for row in task_rows)
        sr = successes / len(task_rows)
        task_values.append(sr)
        task_details[task] = {
            "episodes": len(task_rows),
            "successes": successes,
            "sr": sr,
        }
    seed = 20260826 + ("atomic_seen", "composite_seen", "composite_unseen").index(
        manifest["task_set"]
    )
    rng = random.Random(seed)
    bootstrap = [
        sum(rng.choice(task_values) for _ in task_values) / len(task_values)
        for _ in range(bootstrap_draws)
    ]
    successes = sum(bool(row["success"]) for row in rows.values())
    result = {
        "episodes": len(rows),
        "successes": successes,
        "episode_sr": successes / len(rows),
        "task_macro_sr": sum(task_values) / len(task_values),
        "task_cluster_bootstrap_95ci": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "formal_failures": 0,
        "per_task_details": task_details,
    }
    manifest_path = run_dir / "manifest.json"
    source_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "kind": "omega_qvla_pi05_robocasa365_taskset_summary",
        "config_id": CONFIG,
        "task_set": manifest["task_set"],
        "complete": True,
        "validation_errors": [],
        "bootstrap_draws": bootstrap_draws,
        "manifest_sha256": sha256_file(manifest_path),
        "summarizer": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "configs": {CONFIG: result},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    value = summarize(run_dir, args.bootstrap)
    output = run_dir / "summary.json"
    atomic_json(output, value)
    print(json.dumps({"output": str(output), **value}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
