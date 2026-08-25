#!/usr/bin/env python3
"""Audit an exact two-task x two-seed pi0.5 week-1 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PAIRED_NOISE = "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--runtime-audit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", required=True, help="Exactly two comma-separated task names")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    results_path = Path(args.results).resolve()
    audit_path = Path(args.runtime_audit).resolve()
    require(results_path.is_file(), f"missing preflight results: {results_path}")
    require(audit_path.is_file(), f"missing runtime audit: {audit_path}")
    runtime_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(runtime_audit.get("valid") is True, "runtime audit is invalid")
    require(runtime_audit.get("config_id") == args.config, "runtime audit config mismatch")
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    require(len(tasks) == 2 and len(set(tasks)) == 2, "preflight requires exactly two unique tasks")
    expected = {(task, seed) for task in tasks for seed in (0, 1)}
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed JSON at {results_path}:{line_number}") from error
        key = (str(row.get("task")), int(row.get("seed", -1)))
        checks = {
            "key": key in expected,
            "status": row.get("status") == "complete",
            "config": row.get("config") == args.config,
            "task_set": row.get("task_set") == "atomic_seen",
            "split": row.get("split") == "target",
            "n_action_steps": row.get("n_action_steps") == 16,
            "replan_steps": row.get("replan_steps") == 16,
            "flow_steps": row.get("flow_steps") == 4,
            "action_horizon": row.get("action_horizon") == 50,
            "paired_noise": row.get("paired_action_noise") is True,
            "noise_protocol": row.get("action_noise_protocol") == PAIRED_NOISE,
            "fresh_environment": row.get("fresh_environment") is True,
            "render": row.get("render_enabled") is True,
            "runtime_hash": row.get("server_metadata_sha256")
            == runtime_audit.get("server_metadata_sha256"),
            "selector_disabled": row.get("runtime_selector_enabled") is False,
            "finite_steps": isinstance(row.get("steps"), int) and row["steps"] >= 0,
        }
        failed = [name for name, valid in checks.items() if not valid]
        require(not failed, f"preflight row {line_number} failed checks: {failed}")
        require(key not in rows, f"duplicate preflight key: {key}")
        rows[key] = row
    require(set(rows) == expected, f"preflight coverage mismatch: missing={sorted(expected-set(rows))}")

    payload = {
        "schema_version": 1,
        "kind": "pi05_week1_two_task_two_seed_preflight",
        "valid": True,
        "diagnostic_only": True,
        "paper_claim_enabled": False,
        "config_id": args.config,
        "tasks": tasks,
        "seeds": [0, 1],
        "expected_episodes": 4,
        "observed_episodes": len(rows),
        "successes_not_for_claim": sum(bool(row["success"]) for row in rows.values()),
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
        "runtime_audit_path": str(audit_path),
        "runtime_audit_sha256": sha256_file(audit_path),
        "server_metadata_sha256": runtime_audit["server_metadata_sha256"],
        "protocol": runtime_audit["protocol"],
    }
    output = Path(args.out).resolve()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.exists():
        require(output.read_text(encoding="utf-8") == rendered, "preflight audit drift")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
