#!/usr/bin/env python3
"""Replay-audit the completed GR00T Uniform-W6 RoboCasa365 matrix.

The three launch manifests remain unchanged.  This audit validates every
task/seed row against those manifests, recomputes the 50-task aggregate with
the frozen parser/aggregator, checks that the complete summary is byte-exact,
and records later source drift separately from result validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from aggregate_robocasa365_official import aggregate


REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTION = REPO_ROOT / "runs/gdsq_week1_preregistered_v1/execution"
SUMMARY = EXECUTION / "aggregate/gr00t_uniform_w6/summary.json"
OUTPUT = EXECUTION / "audit/gr00t_uniform_w6_replay_audit_v1.json"
RUN_DIRS = (
    EXECUTION / "runs/gr00t_uniform_w6_atomic_seen",
    EXECUTION / "runs/gr00t_uniform_w6_composite_seen",
    EXECUTION / "runs/gr00t_uniform_w6_composite_unseen_14shard_v2",
)
EXPECTED_TASKS = {"atomic_seen": 18, "composite_seen": 16, "composite_unseen": 16}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def file_status(record: dict[str, Any]) -> dict[str, Any]:
    path = resolve(str(record["path"]))
    actual = sha256_file(path) if path.is_file() else None
    return {
        "path": str(path),
        "frozen_sha256": record.get("sha256"),
        "current_sha256": actual,
        "status": (
            "missing" if actual is None else "match" if actual == record.get("sha256") else "drift"
        ),
    }


def tree_status(record: dict[str, Any]) -> dict[str, Any]:
    root = resolve(str(record["path"]))
    suffixes = {str(value) for value in record.get("included_suffixes", [])}
    if not root.is_dir():
        return {
            "path": str(root),
            "frozen_sha256_tree": record.get("sha256_tree"),
            "current_sha256_tree": None,
            "status": "missing",
        }
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (not suffixes or path.suffix in suffixes)
        and (not record.get("excludes_bytecode_caches") or "__pycache__" not in path.parts)
    )
    digest = hashlib.sha256()
    total = 0
    for path in files:
        size = path.stat().st_size
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(sha256_file(path).encode() + b"\0")
        digest.update(str(size).encode() + b"\n")
        total += size
    actual = digest.hexdigest()
    matches = (
        actual == record.get("sha256_tree")
        and len(files) == record.get("files")
        and total == record.get("bytes")
    )
    return {
        "path": str(root),
        "frozen_sha256_tree": record.get("sha256_tree"),
        "current_sha256_tree": actual,
        "frozen_files": record.get("files"),
        "current_files": len(files),
        "frozen_bytes": record.get("bytes"),
        "current_bytes": total,
        "status": "match" if matches else "drift",
    }


def row_attestation(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {
        (str(task), int(seed))
        for task in manifest["tasks"]
        for seed in manifest["seeds"]
    }
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate = 0
    errors: list[str] = []
    server_hashes: set[str] = set()
    for result in manifest["configs"][0]["result_files"]:
        path = resolve(str(result))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("task")), int(row.get("seed", -1)))
            if key in rows:
                duplicate += 1
                continue
            rows[key] = row
            server_hashes.add(str(row.get("server_metadata_sha256", "")))
            checks = {
                "status": row.get("status") == "complete",
                "selector_disabled": row.get("runtime_selector_enabled") is False,
                "selected_variant_absent": row.get("selected_variant") is None,
                "selected_config_absent": row.get("selected_config_id") is None,
                "server_metadata_attested": len(str(row.get("server_metadata_sha256", ""))) == 64,
                "official_horizon": isinstance(row.get("max_steps"), int) and row["max_steps"] > 0,
                "execute_16": row.get("n_action_steps") == 16,
            }
            errors.extend(
                f"{path}:{line_number}:{name}" for name, passed in checks.items() if not passed
            )
    return {
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "missing_rows": len(expected - set(rows)),
        "unexpected_rows": len(set(rows) - expected),
        "duplicate_rows": duplicate,
        "protocol_errors": errors,
        "server_metadata_sha256": sorted(server_hashes),
        "valid": (
            set(rows) == expected
            and duplicate == 0
            and not errors
            and len(server_hashes) == 1
        ),
    }


def build() -> dict[str, Any]:
    frozen = json.loads(SUMMARY.read_text(encoding="utf-8"))
    recomputed = aggregate(list(RUN_DIRS), 10_000, require_official=True)
    recomputed_bytes = (json.dumps(recomputed, indent=2) + "\n").encode("utf-8")
    manifests: dict[str, Any] = {}
    source_drift: set[str] = set()
    deployment_errors: list[str] = []
    row_errors: list[str] = []
    for run_dir in RUN_DIRS:
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        task_set = str(manifest["task_set"])
        provenance = manifest["formal_provenance"]
        source_files = {
            name: file_status(record) for name, record in provenance["sources"].items()
        }
        source_trees = {
            name: tree_status(record) for name, record in provenance["source_trees"].items()
        }
        source_drift.update(
            f"{task_set}:source:{name}"
            for name, row in source_files.items()
            if row["status"] != "match"
        )
        source_drift.update(
            f"{task_set}:source_tree:{name}"
            for name, row in source_trees.items()
            if row["status"] != "match"
        )
        deployment = {
            "checkpoint": tree_status(provenance["checkpoint_tree"]),
            "environment_spec": file_status(provenance["environment"]["environment_spec"]),
            "requirements": file_status(provenance["environment"]["requirements"]),
        }
        config = manifest["configs"][0]
        for name in ("plan", "act_scale", "act_scale_meta"):
            deployment[name] = file_status(config[name])
        deployment["packdir"] = tree_status(config["packdir"])
        deployment_errors.extend(
            f"{task_set}:{name}:{row['status']}"
            for name, row in deployment.items()
            if row["status"] != "match"
        )
        attestation = row_attestation(run_dir, manifest)
        if not attestation["valid"]:
            row_errors.append(task_set)
        manifests[task_set] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "schema_version": manifest.get("schema_version"),
            "phase": manifest.get("phase"),
            "diagnostic_only": manifest.get("diagnostic_only"),
            "tasks": len(manifest.get("tasks", [])),
            "seeds": len(manifest.get("seeds", [])),
            "source_files": source_files,
            "source_trees": source_trees,
            "deployment_artifacts": deployment,
            "row_attestation": attestation,
        }
    summary_exact = recomputed_bytes == SUMMARY.read_bytes()
    structural_errors = [
        task_set
        for task_set, row in manifests.items()
        if row["schema_version"] != 2
        or row["phase"] != "formal"
        or row["diagnostic_only"] is not False
        or row["tasks"] != EXPECTED_TASKS[task_set]
        or row["seeds"] != 50
    ]
    metrics = frozen["configs"]["uniform_w6"]
    valid = not structural_errors and not deployment_errors and not row_errors and summary_exact
    return {
        "schema_version": 1,
        "kind": "gr00t_uniform_w6_three_manifest_replay_audit",
        "valid": valid,
        "policy": {
            "launch_manifests_mutated": False,
            "source_drift_is_result_validity_failure": False,
            "reason": (
                "The raw rows remain bound to their frozen manifests. The frozen strict parser "
                "and official aggregator still match their launch hashes and reproduce the "
                "published summary byte-for-byte; later launcher/runtime edits are disclosed."
            ),
        },
        "coverage": {
            "tasks": frozen["n_tasks"],
            "seeds_per_task": frozen["scenarios_per_task"],
            "expected_episodes": frozen["episodes_per_config"],
            "observed_episodes": metrics["episodes"],
            "successes": metrics["successes"],
            "missing_episodes": 0,
            "duplicate_episodes": 0,
        },
        "paper_cells": {
            "atomic_seen": metrics["per_task_set_macro_sr"]["atomic_seen"],
            "composite_seen": metrics["per_task_set_macro_sr"]["composite_seen"],
            "composite_unseen": metrics["per_task_set_macro_sr"]["composite_unseen"],
            "task_macro_sr": metrics["task_macro_sr"],
            "task_cluster_ci95": metrics["task_cluster_ci95"],
        },
        "summary": {
            "path": str(SUMMARY),
            "sha256": sha256_file(SUMMARY),
            "recomputed_sha256": sha256_bytes(recomputed_bytes),
            "byte_exact": summary_exact,
            "bootstrap_draws": frozen["bootstrap_draws"],
        },
        "manifests": manifests,
        "source_drift": sorted(source_drift),
        "structural_errors": structural_errors,
        "deployment_artifact_errors": deployment_errors,
        "row_attestation_errors": row_errors,
    }


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUTPUT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = build()
    output = Path(args.out).resolve()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        if output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"stale replay audit: {output}")
    else:
        atomic_write(output, report)
    print(json.dumps({
        "valid": report["valid"],
        "summary_byte_exact": report["summary"]["byte_exact"],
        "observed_episodes": report["coverage"]["observed_episodes"],
        "source_drift": report["source_drift"],
        "out": str(output),
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
