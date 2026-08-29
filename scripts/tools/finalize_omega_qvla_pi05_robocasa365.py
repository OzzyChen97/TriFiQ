#!/usr/bin/env python3
"""Register complete pi0.5 Omega-QVLA RoboCasa365 task-set evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "runs/gdsq_extension_preregistered_v1/omega_qvla_pi05_robocasa365_v1"
REGISTRY = REPO_ROOT / "docs/gdsq_vla_iclr2027/experiment_registry.json"
CONFIG = "omega_qvla_w4a4"
SPECS = {
    "atomic_seen": (18, 900),
    "composite_seen": (16, 800),
    "composite_unseen": (16, 800),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def task_set_record(task_set: str, tasks: int, episodes: int) -> dict[str, Any] | None:
    run_dir = RUN_ROOT / "results" / task_set
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(manifest.get("task_set") == task_set, f"{task_set} manifest drift")
    require(len(manifest.get("tasks") or []) == tasks, f"{task_set} task-count drift")
    require(manifest.get("seeds") == list(range(50)), f"{task_set} seed drift")
    require(summary.get("task_set") == task_set, f"{task_set} summary drift")
    require(summary.get("complete") is True, f"{task_set} summary incomplete")
    require(summary.get("validation_errors") == [], f"{task_set} validation errors")
    require(summary.get("bootstrap_draws") == 10_000, f"{task_set} bootstrap drift")
    require(summary.get("manifest_sha256") == sha256_file(manifest_path), f"{task_set} link drift")
    result = (summary.get("configs") or {}).get(CONFIG) or {}
    require(result.get("episodes") == episodes, f"{task_set} episode drift")
    require(result.get("formal_failures") == 0, f"{task_set} formal failure")
    details = result.get("per_task_details") or {}
    require(len(details) == tasks, f"{task_set} per-task coverage drift")
    require(all(row.get("episodes") == 50 for row in details.values()), f"{task_set} seed coverage drift")
    return {
        "status": "complete",
        "main_claim_enabled": True,
        "coverage": {
            "tasks": tasks,
            "seeds_per_task": 50,
            "expected_episodes": episodes,
            "observed_episodes": episodes,
            "missing_episodes": 0,
            "duplicate_episodes": 0,
        },
        "manifest": artifact(manifest_path),
        "summary": artifact(summary_path),
    }


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    experiment = registry["experiments"]["omega_qvla_robocasa365"]
    progress = experiment["model_progress"].setdefault("pi05", {})
    records: dict[str, Any] = {}
    observed = 0
    for task_set, (tasks, episodes) in SPECS.items():
        record = task_set_record(task_set, tasks, episodes)
        if record is not None:
            records[task_set] = record
            observed += episodes
        elif task_set in (progress.get("task_sets") or {}):
            records[task_set] = progress["task_sets"][task_set]
    progress["task_sets"] = records
    progress["coverage"] = {
        "expected_episodes": 2500,
        "observed_episodes": observed,
        "missing_episodes": 2500 - observed,
        "duplicate_episodes": 0,
    }

    memory_path = RUN_ROOT / "aggregate/pi05_paper_memory.json"
    if memory_path.is_file():
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        require(memory.get("kind") == "omega_qvla_pi05_theoretical_packed_storage_audit", "memory kind drift")
        require(memory.get("scope_linear_layers") == 252, "memory scope drift")
        require(memory.get("quantized_layers") == 252, "memory layer drift")
        require((memory.get("representation") or {}).get("weight_bits") == 4, "memory W4 drift")
        require((memory.get("representation") or {}).get("activation_bits") == 4, "memory A4 drift")
        progress["memory"] = artifact(memory_path)

    aggregate_path = RUN_ROOT / "aggregate/summary.json"
    if observed == 2500 and aggregate_path.is_file():
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        require(aggregate.get("complete") is True, "aggregate incomplete")
        require(aggregate.get("episodes") == 2500, "aggregate coverage drift")
        require(aggregate.get("bootstrap_samples") == 10_000, "aggregate bootstrap drift")
        progress["aggregate"] = artifact(aggregate_path)
        progress["status"] = "complete"
        progress["main_claim_enabled"] = True
    else:
        progress.pop("aggregate", None)
        progress["status"] = "running_partial_taskset_complete"
        progress["main_claim_enabled"] = False

    gr_observed = int(experiment["model_progress"]["gr00t"]["coverage"]["observed_episodes"])
    total_observed = gr_observed + observed
    experiment["coverage"] = {
        "expected_episodes": 5000,
        "observed_episodes": total_observed,
        "missing_episodes": 5000 - total_observed,
        "duplicate_episodes": 0,
    }
    experiment["main_claim_enabled"] = observed == 2500
    experiment["status"] = (
        "complete" if observed == 2500 else "gr00t_complete_pi05_partial"
    )
    complete_names = [name for name in SPECS if name in records and records[name].get("status") == "complete"]
    experiment["notes"] = (
        "The released Omega-QVLA packs are LIBERO-specific and are not reused. "
        "GR00T has exact 2,500-episode coverage. The independent pi0.5 run uses "
        "task-set-specific RoboCasa365 W4A4 packs; only task sets with exact "
        f"50-seed coverage are enabled. Complete pi0.5 sets: {', '.join(complete_names)}."
    )

    rendered = json.dumps(registry, indent=2) + "\n"
    temporary = REGISTRY.with_name(f".{REGISTRY.name}.tmp.{os.getpid()}")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(REGISTRY)
    print(json.dumps({
        "status": experiment["status"],
        "pi05_observed_episodes": observed,
        "pi05_missing_episodes": 2500 - observed,
        "complete_task_sets": complete_names,
    }, indent=2))


if __name__ == "__main__":
    main()
