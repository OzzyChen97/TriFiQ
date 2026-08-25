#!/usr/bin/env python3
"""Completion audit for the immutable π0.5 RoboCasa365 Table-1 run.

This tool intentionally wraps, rather than edits, the aggregator frozen in the
base manifest.  It verifies artifact content, the resource-schedule chain,
runtime attestations, exact key coverage, and the final statistical schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import aggregate_pi05_robocasa365 as frozen_aggregator


CONFIGS = tuple(frozen_aggregator.CONFIG_ORDER)
EXPECTED_COMPARISONS = {
    f"{a}_vs_{b}" for a, b in frozen_aggregator.CONTRASTS
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_artifacts(manifest: dict[str, Any]) -> dict[str, Any]:
    verified = {}
    for name, record in sorted(manifest["artifacts"].items()):
        path = Path(record["path"]).resolve()
        require(path.is_file(), f"artifact is missing: {name}: {path}")
        actual_bytes = path.stat().st_size
        require(
            actual_bytes == int(record["bytes"]),
            f"artifact byte count mismatch: {name}: {actual_bytes} != {record['bytes']}",
        )
        actual_sha = sha256_file(path)
        require(
            actual_sha == record["sha256"],
            f"artifact hash mismatch: {name}: {actual_sha} != {record['sha256']}",
        )
        verified[name] = {
            "path": str(path),
            "bytes": actual_bytes,
            "sha256": actual_sha,
        }
    return verified


def schedule_key_coverage(
    schedule: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    task_sets = manifest["table_1_protocol"]["task_sets"]
    tasks = [task for values in task_sets.values() for task in values]
    seeds = {int(seed) for seed in manifest["table_1_protocol"]["trial_seeds"]}
    expected = {(task, seed) for task in tasks for seed in seeds}
    coverage = {}
    for config in CONFIGS:
        keys: set[tuple[str, int]] = set()
        duplicate_count = 0
        workers = [row for row in schedule["workers"] if row["config_id"] == config]
        require(workers, f"schedule has no workers for {config}")
        for worker in workers:
            shard = int(worker["task_shard_index"])
            shard_count = int(worker["task_shard_count"])
            require(0 <= shard < shard_count, f"invalid task shard in {worker['worker_id']}")
            worker_tasks = [
                task for index, task in enumerate(tasks) if index % shard_count == shard
            ]
            worker_seeds = range(
                int(worker["trial_seed_start"]), int(worker["trial_seed_end"]) + 1
            )
            for key in ((task, seed) for task in worker_tasks for seed in worker_seeds):
                if key in keys:
                    duplicate_count += 1
                keys.add(key)
        missing = expected - keys
        extra = keys - expected
        require(not duplicate_count, f"schedule duplicates {duplicate_count} keys for {config}")
        require(not missing, f"schedule misses {len(missing)} keys for {config}")
        require(not extra, f"schedule adds {len(extra)} keys for {config}")
        coverage[config] = {
            "workers": len(workers),
            "keys": len(keys),
            "keyset_sha256": canonical_hash(sorted(keys)),
        }
    return coverage


def verify_schedule_chain(
    leaf_path: Path, manifest_path: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    expected_base_sha = sha256_file(manifest_path)
    allowed_runtime_hashes = {
        config: {
            row["server_metadata_sha256"]
            for row in manifest["servers"]
            if row["config_id"] == config
        }
        for config in CONFIGS
    }
    chain = []
    seen: set[Path] = set()
    path = leaf_path.resolve()
    leaf_coverage = None
    while True:
        require(path.is_file(), f"schedule is missing: {path}")
        require(path not in seen, f"schedule supersedes cycle: {path}")
        seen.add(path)
        schedule = json.loads(path.read_text(encoding="utf-8"))
        schedule_sha = sha256_file(path)
        require(schedule.get("immutable") is True, f"schedule is not immutable: {path}")
        require(schedule.get("protocol_change") is False, f"protocol changed in {path}")
        require(
            schedule.get("base_manifest_sha256") == expected_base_sha,
            f"base manifest hash mismatch in {path}",
        )
        require(
            schedule.get("frozen_protocol") == manifest["table_1_protocol"],
            f"frozen protocol mismatch in {path}",
        )
        for server in schedule["servers"]:
            config = server["config_id"]
            require(config in allowed_runtime_hashes, f"unknown schedule config: {config}")
            require(
                server["server_metadata_sha256"] in allowed_runtime_hashes[config],
                f"unfrozen runtime metadata in {path}: {server['instance']}",
            )
        coverage = schedule_key_coverage(schedule, manifest)
        if not chain:
            leaf_coverage = coverage
            scheduler = Path(schedule["scheduler"]).resolve()
            require(scheduler.is_file(), f"leaf scheduler is missing: {scheduler}")
            require(
                sha256_file(scheduler) == schedule["scheduler_sha256"],
                "leaf scheduler source no longer matches immutable schedule",
            )
            for server in schedule["servers"]:
                runtime_path = Path(server["runtime_path"]).resolve()
                require(runtime_path.is_file(), f"runtime file is missing: {runtime_path}")
                require(
                    sha256_file(runtime_path) == server["runtime_file_sha256"],
                    f"runtime file hash mismatch: {server['instance']}",
                )
                runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                require(
                    canonical_hash(runtime) == server["server_metadata_sha256"],
                    f"runtime canonical hash mismatch: {server['instance']}",
                )
        chain.append(
            {
                "path": str(path),
                "sha256": schedule_sha,
                "schema_version": schedule.get("schema_version"),
                "servers": len(schedule["servers"]),
                "workers": len(schedule["workers"]),
                "coverage": coverage,
            }
        )
        parent = schedule.get("supersedes_schedule")
        if not parent:
            break
        parent_path = Path(parent).resolve()
        require(
            sha256_file(parent_path) == schedule["supersedes_schedule_sha256"],
            f"superseded schedule hash mismatch: {parent_path}",
        )
        path = parent_path
    return {"leaf_coverage": leaf_coverage, "chain_leaf_to_root": chain}


def verify_statistical_schema(summary: dict[str, Any], complete: bool) -> dict[str, Any]:
    require(summary["expected_episodes_per_config"] == 2500, "wrong episode target")
    if not complete:
        return {"complete": False, "comparisons_ready": False}
    require(summary["complete"] is True, "matrix is not complete")
    for config in CONFIGS:
        row = summary["configs"][config]
        require(row["completed_episodes"] == 2500, f"wrong row count: {config}")
        require(row["missing_episodes"] == 0, f"missing rows: {config}")
        require(len(row["per_task_sr"]) == 50, f"wrong task count: {config}")
        require(len(row["per_task_details"]) == 50, f"missing task details: {config}")
        require(
            set(row["task_set_macro_sr"])
            == {"atomic_seen", "composite_seen", "composite_unseen"},
            f"wrong task-set summary: {config}",
        )
        for task, details in row["per_task_details"].items():
            require(details["episodes"] == 50, f"wrong seed count: {config}/{task}")
        require(len(row["task_cluster_ci95"]) == 2, f"missing all-task CI: {config}")
        require(
            len(row["heldout46_task_cluster_ci95"]) == 2,
            f"missing held-out CI: {config}",
        )
    require(
        set(summary["comparisons"]) == EXPECTED_COMPARISONS,
        "prespecified comparison set mismatch",
    )
    for name, comparison in summary["comparisons"].items():
        require(len(comparison["heldout46_task_cluster_ci95"]) == 2, f"missing CI: {name}")
        require("paired_permutation_p" in comparison, f"missing permutation p: {name}")
        require("holm_adjusted_p" in comparison, f"missing Holm p: {name}")
        require("episode_mcnemar" in comparison, f"missing McNemar: {name}")
    return {"complete": True, "comparisons_ready": True}


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("immutable") is True, "base manifest is not immutable")
    manifest_sha = sha256_file(manifest_path)
    require(
        manifest_sha
        == "79dfd759a019c5eb5ccc90cc7c149b59715e254ec3a9beb8452ef261c196e042",
        f"unexpected base manifest: {manifest_sha}",
    )
    artifacts = verify_artifacts(manifest)
    schedule = verify_schedule_chain(
        Path(args.schedule).resolve(), manifest_path, manifest
    )
    summary = frozen_aggregator.aggregate(
        run_dir, args.bootstrap, args.allow_incomplete
    )
    statistics = verify_statistical_schema(summary, summary["complete"])
    return {
        "schema_version": 1,
        "complete": bool(summary["complete"]),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "frozen_aggregator_sha256": artifacts["aggregator"]["sha256"],
        "audit_tool_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap_draws": args.bootstrap,
        "artifacts": {"verified": len(artifacts), "records": artifacts},
        "schedule": schedule,
        "matrix": {
            config: {
                "completed": summary["configs"][config]["completed_episodes"],
                "missing": summary["configs"][config]["missing_episodes"],
            }
            for config in CONFIGS
        },
        "statistics": statistics,
        "efficiency_scope_note": (
            "server-process memory, shared whole-device memory, and theoretical packed "
            "candidate-component storage remain separate; queue-inflated formal rollout "
            "latency is not a single-request service-time speedup claim"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args)
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "manifest_sha256": result["manifest_sha256"],
                "artifacts_verified": result["artifacts"]["verified"],
                "schedule_depth": len(result["schedule"]["chain_leaf_to_root"]),
                "matrix": result["matrix"],
                "out": str(output),
                "out_sha256": sha256_file(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
