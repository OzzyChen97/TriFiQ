#!/usr/bin/env python3
"""Replay-audit a frozen RoboCasa365 result matrix without mutating it.

The original manifest remains immutable.  This audit re-parses every JSONL
episode, checks the preregistered protocol and exact task/seed coverage, and
independently recomputes the summary cells used by the paper.  Source files
that have changed since the run are reported as hash drift; drift is never
silently blessed by rewriting the frozen manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/official_target_paired50"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_path(path_text: str, manifest_path: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    candidate = manifest_path.parent / path
    return candidate if candidate.exists() else REPO_ROOT / path


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def artifact_drift(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, frozen in sorted((manifest.get("artifacts") or {}).items()):
        if not isinstance(frozen, dict) or "path" not in frozen or "sha256" not in frozen:
            continue
        path = resolve_manifest_path(str(frozen["path"]), manifest_path)
        if not path.is_file():
            rows[name] = {
                "path": str(path),
                "frozen_sha256": frozen["sha256"],
                "current_sha256": None,
                "status": "missing",
            }
            continue
        current = sha256_file(path)
        rows[name] = {
            "path": str(path),
            "frozen_sha256": frozen["sha256"],
            "current_sha256": current,
            "status": "match" if current == frozen["sha256"] else "drift",
        }
    return rows


def expected_protocol(manifest: dict[str, Any]) -> tuple[dict[str, str], set[int], dict[str, Any]]:
    protocol = manifest["table_1_protocol"]
    task_to_set = {
        task: task_set
        for task_set, tasks in protocol["task_sets"].items()
        for task in tasks
    }
    seeds = {int(seed) for seed in protocol["trial_seeds"]}
    values = {
        "split": protocol["split"],
        "action_horizon": int(protocol["action_horizon"]),
        "n_action_steps": int(protocol["n_action_steps"]),
        "replan_steps": int(protocol["replan_steps"]),
        "flow_steps": int(protocol["flow_steps"]),
        "paired_action_noise": bool(protocol["paired_action_noise"]),
        "action_noise_protocol": protocol["paired_action_noise_protocol"],
        "fresh_environment": bool(protocol["fresh_environment_per_trial"]),
        "render_enabled": bool(protocol["render_enabled"]),
    }
    return task_to_set, seeds, values


def row_protocol_errors(
    row: dict[str, Any],
    *,
    config: str,
    task_to_set: dict[str, str],
    seeds: set[int],
    protocol: dict[str, Any],
    allowed_server_hashes: set[str],
) -> list[str]:
    task = str(row.get("task", ""))
    try:
        seed = int(row.get("seed", -1))
    except (TypeError, ValueError):
        seed = -1
    checks = {
        "status": row.get("status") == "complete",
        "config": row.get("config") == config,
        "task": task in task_to_set,
        "task_set": row.get("task_set") == task_to_set.get(task),
        "seed": seed in seeds,
        "split": row.get("split") == protocol["split"],
        "action_horizon": row.get("action_horizon") == protocol["action_horizon"],
        "n_action_steps": row.get("n_action_steps") == protocol["n_action_steps"],
        "replan_steps": row.get("replan_steps") == protocol["replan_steps"],
        "flow_steps": row.get("flow_steps") == protocol["flow_steps"],
        "paired_action_noise": (
            row.get("paired_action_noise") is protocol["paired_action_noise"]
        ),
        "action_noise_protocol": (
            row.get("action_noise_protocol") == protocol["action_noise_protocol"]
        ),
        "fresh_environment": (
            row.get("fresh_environment") is protocol["fresh_environment"]
        ),
        "render_enabled": row.get("render_enabled") is protocol["render_enabled"],
        "official_horizon": (
            isinstance(row.get("max_steps"), int) and int(row["max_steps"]) > 0
        ),
        "success_boolean": isinstance(row.get("success"), bool),
        "server_metadata": row.get("server_metadata_sha256") in allowed_server_hashes,
    }
    return [name for name, valid in checks.items() if not valid]


def load_and_check_rows(
    run_dir: Path, manifest: dict[str, Any]
) -> tuple[dict[str, dict[tuple[str, int], dict[str, Any]]], dict[str, Any]]:
    task_to_set, seeds, protocol = expected_protocol(manifest)
    expected = set(itertools.product(task_to_set, seeds))
    configs = list(manifest["configs"])
    server_hashes = {
        config: {
            row["server_metadata_sha256"]
            for row in manifest["servers"]
            if row["config_id"] == config
        }
        for config in configs
    }
    grouped: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        config: {} for config in configs
    }
    parse_errors: list[dict[str, Any]] = []
    duplicates: dict[str, list[list[Any]]] = {config: [] for config in configs}
    files: dict[str, list[dict[str, Any]]] = {config: [] for config in configs}
    for config in configs:
        result_dir = run_dir / "results" / config
        for path in sorted(result_dir.glob("*.jsonl")):
            row_count = 0
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                row_count += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    parse_errors.append(
                        {
                            "file": str(path),
                            "line": line_number,
                            "errors": [f"malformed_json:{error.msg}"],
                        }
                    )
                    continue
                errors = row_protocol_errors(
                    row,
                    config=config,
                    task_to_set=task_to_set,
                    seeds=seeds,
                    protocol=protocol,
                    allowed_server_hashes=server_hashes[config],
                )
                if errors:
                    parse_errors.append(
                        {"file": str(path), "line": line_number, "errors": errors}
                    )
                    continue
                key = (str(row["task"]), int(row["seed"]))
                if key in grouped[config]:
                    duplicates[config].append([key[0], key[1]])
                    continue
                grouped[config][key] = row
            files[config].append(
                {"path": str(path), "sha256": sha256_file(path), "rows": row_count}
            )
    missing = {
        config: [[task, seed] for task, seed in sorted(expected - set(rows))]
        for config, rows in grouped.items()
    }
    unexpected = {
        config: [[task, seed] for task, seed in sorted(set(rows) - expected)]
        for config, rows in grouped.items()
    }
    audit = {
        "expected_tasks": len(task_to_set),
        "expected_seeds_per_task": len(seeds),
        "expected_episodes_per_config": len(expected),
        "observed_episodes_per_config": {
            config: len(rows) for config, rows in grouped.items()
        },
        "missing": missing,
        "duplicates": duplicates,
        "unexpected": unexpected,
        "protocol_errors": parse_errors,
        "result_files": files,
    }
    return grouped, audit


def recompute_cells(
    rows: dict[str, dict[tuple[str, int], dict[str, Any]]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    task_to_set, seeds, _ = expected_protocol(manifest)
    tasks = list(task_to_set)
    result: dict[str, Any] = {}
    for config, config_rows in rows.items():
        per_task: dict[str, float] = {}
        task_set_macro: dict[str, float] = {}
        for task in tasks:
            available = [
                config_rows[(task, seed)]
                for seed in sorted(seeds)
                if (task, seed) in config_rows
            ]
            if len(available) == len(seeds):
                per_task[task] = sum(bool(row["success"]) for row in available) / len(
                    available
                )
        for task_set, set_tasks in manifest["table_1_protocol"]["task_sets"].items():
            values = [per_task[task] for task in set_tasks if task in per_task]
            if len(values) == len(set_tasks):
                task_set_macro[task_set] = sum(values) / len(values)
        task_values = list(per_task.values())
        result[config] = {
            "completed_episodes": len(config_rows),
            "observed_successes": sum(
                bool(row["success"]) for row in config_rows.values()
            ),
            "observed_episode_sr": (
                sum(bool(row["success"]) for row in config_rows.values())
                / len(config_rows)
                if config_rows
                else None
            ),
            "per_task_sr": per_task,
            "task_macro_sr": (
                sum(task_values) / len(task_values)
                if len(task_values) == len(tasks)
                else None
            ),
            "task_set_macro_sr": task_set_macro,
        }
    return result


def compare_summary(
    recomputed: dict[str, Any], summary: dict[str, Any], manifest_sha256: str
) -> list[str]:
    errors: list[str] = []
    if summary.get("complete") is not True:
        errors.append("summary.complete is not true")
    if summary.get("manifest_sha256") != manifest_sha256:
        errors.append("summary manifest_sha256 differs from frozen manifest")
    for config, fresh in recomputed.items():
        frozen = (summary.get("configs") or {}).get(config)
        if frozen is None:
            errors.append(f"summary missing config {config}")
            continue
        for key in ("completed_episodes", "observed_successes"):
            if frozen.get(key) != fresh[key]:
                errors.append(
                    f"{config}.{key}: frozen={frozen.get(key)!r}, recomputed={fresh[key]!r}"
                )
        for key in ("observed_episode_sr", "task_macro_sr"):
            left, right = frozen.get(key), fresh[key]
            if left is None or right is None or not math.isclose(
                float(left), float(right), rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append(
                    f"{config}.{key}: frozen={left!r}, recomputed={right!r}"
                )
        frozen_per_task = frozen.get("per_task_sr") or {}
        if set(frozen_per_task) != set(fresh["per_task_sr"]):
            errors.append(f"{config}.per_task_sr task inventory differs")
        else:
            for task, value in fresh["per_task_sr"].items():
                if not math.isclose(
                    float(frozen_per_task[task]), value, rel_tol=0.0, abs_tol=1e-12
                ):
                    errors.append(f"{config}.per_task_sr.{task} differs")
    return errors


def run_audit(run_dir: Path, summary_path: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows, coverage = load_and_check_rows(run_dir, manifest)
    recomputed = recompute_cells(rows, manifest)
    manifest_hash = sha256_file(manifest_path)
    summary_errors = compare_summary(recomputed, summary, manifest_hash)
    coverage_error_count = (
        len(coverage["protocol_errors"])
        + sum(len(values) for values in coverage["missing"].values())
        + sum(len(values) for values in coverage["duplicates"].values())
        + sum(len(values) for values in coverage["unexpected"].values())
    )
    drift = artifact_drift(manifest, manifest_path)
    drifted = sorted(name for name, row in drift.items() if row["status"] == "drift")
    missing_artifacts = sorted(
        name for name, row in drift.items() if row["status"] == "missing"
    )
    return {
        "schema_version": 1,
        "kind": "immutable_result_replay_audit",
        "run_dir": str(run_dir),
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_hash,
            "immutable": manifest.get("immutable") is True,
        },
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "valid": (
            manifest.get("immutable") is True
            and coverage_error_count == 0
            and not summary_errors
        ),
        "coverage": coverage,
        "recomputed_cells": recomputed,
        "summary_comparison_errors": summary_errors,
        "source_artifact_audit": {
            "note": (
                "Drift is reported against the immutable run manifest. It does not alter "
                "the frozen raw episodes or retroactively attest changed source code."
            ),
            "drifted": drifted,
            "missing": missing_artifacts,
            "artifacts": drift,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--summary", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary
        else run_dir / "aggregate/summary.json"
    )
    report = run_audit(run_dir, summary_path)
    out = Path(args.out).expanduser().resolve()
    atomic_write_json(out, report)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "manifest_sha256": report["manifest"]["sha256"],
                "observed_episodes_per_config": report["coverage"][
                    "observed_episodes_per_config"
                ],
                "source_drift": report["source_artifact_audit"]["drifted"],
                "source_missing": report["source_artifact_audit"]["missing"],
                "summary_comparison_errors": report["summary_comparison_errors"],
                "out": str(out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
