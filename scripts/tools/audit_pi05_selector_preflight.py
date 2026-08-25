#!/usr/bin/env python3
"""Audit the result-blind 2-task x 2-seed pi0.5 selector preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SELECTOR_SHA256 = "0f3178726c2b784898f18dfde248d9a9bae152da6ffcfdc299e8bdc02f0bd871"
RULE_NAME = "v8_no_oracle_absolute_mechanism_gate_aligned_runtime_rule"
TASKS = ("OpenDrawer", "TurnOnMicrowave")
SEEDS = (0, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result_path = Path(args.results).resolve()
    runtime_path = Path(args.runtime).resolve()
    output = Path(args.out).resolve()
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_hash = canonical_hash(runtime_payload)
    selector = ((runtime_payload.get("openpi_runtime") or {}).get("runtime_selector") or {})
    require(selector.get("enabled") is True, "runtime selector is disabled")
    require(selector.get("selector_sha256") == SELECTOR_SHA256, "selector SHA drift")
    require(selector.get("rule_name") == RULE_NAME, "selector rule drift")
    require(selector.get("uses_task_metadata_for_selection") is False, "task-dependent selector")

    rows = []
    for line_number, line in enumerate(result_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        checks = {
            "status": row.get("status") == "complete",
            "config": row.get("config") == "gdsq_vla_runtime_selector",
            "task": row.get("task") in TASKS,
            "seed": int(row.get("seed", -1)) in SEEDS,
            "selector_enabled": row.get("runtime_selector_enabled") is True,
            "variant": row.get("selected_variant") == "ohb",
            "selected_config": row.get("selected_config_id") == "gdsq_vla_ohb_only",
            "selector_sha": row.get("selector_sha256") == SELECTOR_SHA256,
            "selector_rule": row.get("selector_rule_name") == RULE_NAME,
            "selector_model": row.get("selector_model_id") == "pi05",
            "selector_task": row.get("selector_task_name") == row.get("task"),
            "server": row.get("server_metadata_sha256") == runtime_hash,
            "paired_noise": row.get("paired_action_noise") is True,
            "flow_steps": row.get("flow_steps") == 4,
            "execute16": row.get("replan_steps") == 16,
        }
        failed = [name for name, valid in checks.items() if not valid]
        require(not failed, f"row {line_number} failed checks: {failed}")
        rows.append(row)
    keys = {(str(row["task"]), int(row["seed"])) for row in rows}
    expected = {(task, seed) for task in TASKS for seed in SEEDS}
    require(len(keys) == len(rows), "duplicate preflight keys")
    require(keys == expected, f"preflight coverage mismatch: {sorted(expected - keys)}")
    payload = {
        "schema_version": 1,
        "kind": "pi05_selector_2task_2seed_preflight",
        "valid": True,
        "diagnostic_only": True,
        "paper_claim_enabled": False,
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "expected_episodes": 4,
        "observed_episodes": len(rows),
        "missing_episodes": 0,
        "duplicate_episodes": 0,
        "selector_sha256": SELECTOR_SHA256,
        "selector_rule_name": RULE_NAME,
        "selected_variant": "ohb",
        "selected_config_id": "gdsq_vla_ohb_only",
        "runtime_file": str(runtime_path),
        "runtime_file_sha256": sha256_file(runtime_path),
        "server_metadata_sha256": runtime_hash,
        "results_file": str(result_path),
        "results_file_sha256": sha256_file(result_path),
    }
    if output.exists():
        require(
            canonical_hash(json.loads(output.read_text(encoding="utf-8")))
            == canonical_hash(payload),
            "preflight audit drift",
        )
        print(f"preflight audit verified: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, output)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    print(f"preflight audit created: {output}")
    print(f"preflight audit sha256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
