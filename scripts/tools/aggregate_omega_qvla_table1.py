#!/usr/bin/env python3
"""Strictly aggregate the pinned Omega-QVLA Table 1 reproduction.

One official launcher output is required for every model/configuration/suite
cell.  A formal summary is complete only at 2 models x 2 configurations x
4 suites x 10 tasks x 10 held-out trials = 1,600 episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "runs/gdsq_extension_preregistered_v1/omega_qvla_table1"
MODELS = ("gr00t", "pi05")
CONFIGS = ("fp16", "omega_qvla_w4a4")
SUITES = ("goal", "spatial", "object", "long")
TASK_SUITE_NAMES = {
    "goal": "libero_goal",
    "spatial": "libero_spatial",
    "object": "libero_object",
    "long": "libero_10",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def display_path(path: Path, root: Path | None = None) -> str:
    for parent in (REPO_ROOT, root):
        if parent is None:
            continue
        try:
            return str(path.relative_to(parent))
        except ValueError:
            pass
    return str(path)


def validate_cell(path: Path, suite: str, root: Path | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("task_suite_name") == TASK_SUITE_NAMES[suite], f"suite drift: {path}")
    require(int(value.get("num_trials_per_task", -1)) == 10, f"trial count drift: {path}")
    rows = value.get("task_summaries") or []
    require(len(rows) == 10, f"expected 10 task summaries: {path}")
    task_ids = [int(row.get("task_id", -1)) for row in rows]
    require(sorted(task_ids) == list(range(10)), f"task coverage drift: {path}")
    for row in rows:
        require(int(row.get("episodes", -1)) == 10, f"task trial coverage drift: {path}")
        successes = int(row.get("successes", -1))
        require(0 <= successes <= 10, f"invalid task successes: {path}")
        require(
            abs(float(row.get("success_rate", -1.0)) - successes / 10.0) < 1e-12,
            f"task rate mismatch: {path}",
        )
    successes = sum(int(row["successes"]) for row in rows)
    require(int(value.get("total_episodes", -1)) == 100, f"episode coverage drift: {path}")
    require(int(value.get("total_successes", -1)) == successes, f"success total drift: {path}")
    require(
        abs(float(value.get("total_success_rate", -1.0)) - successes / 100.0) < 1e-12,
        f"suite rate mismatch: {path}",
    )
    return {
        "successes": successes,
        "episodes": 100,
        "success_rate_percent": float(successes),
        "source": display_path(path, root),
        "source_sha256": sha256_file(path),
    }


def aggregate(root: Path) -> dict[str, Any]:
    models: dict[str, Any] = {}
    missing: list[str] = []
    invalid: list[str] = []
    observed_episodes = 0
    for model in MODELS:
        configs: dict[str, Any] = {}
        for config in CONFIGS:
            suites: dict[str, Any] = {}
            for suite in SUITES:
                path = root / "results" / model / config / suite / "merged_summary.json"
                relative = display_path(path, root)
                if not path.is_file():
                    missing.append(relative)
                    continue
                try:
                    row = validate_cell(path, suite, root)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    invalid.append(f"{relative}: {error}")
                    continue
                suites[suite] = row
                observed_episodes += int(row["episodes"])
            metrics = None
            if len(suites) == len(SUITES):
                metrics = {suite: suites[suite]["success_rate_percent"] for suite in SUITES}
                metrics["average"] = sum(metrics[suite] for suite in SUITES) / len(SUITES)
            configs[config] = {"suites": suites, "metrics": metrics}
        models[model] = {"configs": configs}
    complete = not missing and not invalid and observed_episodes == 1600
    return {
        "schema_version": 1,
        "kind": "omega_qvla_table1_local_reproduction",
        "complete": complete,
        "formal_result": complete,
        "protocol": {
            "models": list(MODELS),
            "configs": list(CONFIGS),
            "suites": list(SUITES),
            "tasks_per_suite": 10,
            "trials_per_task": 10,
            "held_out_init_offset": 10,
            "denoising_steps": 8,
            "official_release_packs": True,
        },
        "coverage": {
            "expected_episodes": 1600,
            "observed_episodes": observed_episodes,
            "missing_cells": missing,
            "invalid_cells": invalid,
        },
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = aggregate(root)
    output = (args.output or root / "aggregate/summary.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"complete": result["complete"], "coverage": result["coverage"], "output": str(output)}, indent=2))
    if args.strict and not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
