#!/usr/bin/env python3
"""Freeze or verify the result-blind Omega-QVLA Table 1 reproduction protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "runs/gdsq_extension_preregistered_v1/omega_qvla_table1"
PROTOCOL = ROOT / "protocol_v1.json"
DOWNLOAD = REPO_ROOT / "checkpoints/omega_qvla/download_manifest.json"
SOURCE = ROOT / "source_record.json"
EXTENSION_PLAN = REPO_ROOT / "runs/gdsq_extension_preregistered_v1/plan.json"
OFFICIAL_CODE = REPO_ROOT / "external/Omega-QVLA"
MODELS = ("gr00t", "pi05")
CONFIGS = ("fp16", "omega_qvla_w4a4")
SUITES = ("goal", "spatial", "object", "long")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_download_manifest() -> dict[str, Any]:
    require(DOWNLOAD.is_file(), f"official pack download is incomplete: {DOWNLOAD}")
    value = json.loads(DOWNLOAD.read_text(encoding="utf-8"))
    require(set(value.get("models", {})) == set(MODELS), "both official model releases are required")
    for model in MODELS:
        rows = value["models"][model].get("files") or []
        require({row.get("suite") for row in rows} == set(SUITES), f"pack suite drift: {model}")
        for row in rows:
            path = REPO_ROOT / row["path"]
            require(path.is_file(), f"missing downloaded pack: {path}")
            require(path.stat().st_size == int(row["bytes"]), f"pack size drift: {path}")
            require(sha256_file(path) == row["sha256"], f"pack SHA drift: {path}")
    return value


def ensure_result_blind() -> None:
    result_root = ROOT / "results"
    observed = list(result_root.glob("**/merged_summary.json")) if result_root.exists() else []
    require(not observed, f"formal results predate protocol freeze: {observed[:3]}")


def code_commit() -> str:
    require((OFFICIAL_CODE / ".git").is_dir(), f"missing official code clone: {OFFICIAL_CODE}")
    return subprocess.check_output(["git", "-C", str(OFFICIAL_CODE), "rev-parse", "HEAD"], text=True).strip()


def build(created_utc: str | None = None) -> dict[str, Any]:
    download = verify_download_manifest()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    expected_commit = source["official_release"]["code"]["commit"]
    require(code_commit() == expected_commit, "official source commit drift")
    require(download["official_code"]["commit"] == expected_commit, "download/code commit mismatch")
    cells = [
        {
            "model": model,
            "config": config,
            "suite": suite,
            "tasks": list(range(10)),
            "trials_per_task": 10,
            "held_out_init_offset": 10,
            "expected_episodes": 100,
            "output": str((ROOT / "results" / model / config / suite).relative_to(REPO_ROOT)),
        }
        for model in MODELS
        for config in CONFIGS
        for suite in SUITES
    ]
    return {
        "schema_version": 1,
        "kind": "omega_qvla_table1_result_blind_protocol",
        "immutable": True,
        "result_blind": True,
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "launch_after": "gdsq_week1_preregistered_v1 phase=complete",
        "paper_source": artifact(SOURCE),
        "official_pack_download": artifact(DOWNLOAD),
        "official_code": {
            "path": str(OFFICIAL_CODE.relative_to(REPO_ROOT)),
            "commit": expected_commit,
            "remote": source["official_release"]["code"]["url"],
        },
        "base_checkpoints": {
            "attested_by": artifact(EXTENSION_PLAN),
            "gr00t": [f"checkpoints/gr00t/libero-{suite}" for suite in SUITES],
            "pi05": "code/pi05/checkpoints/pi05_libero_pytorch",
        },
        "release_revisions": {
            model: download["models"][model]["revision"] for model in MODELS
        },
        "protocol": {
            "benchmark": "LIBERO",
            "suite_order": list(SUITES),
            "models": list(MODELS),
            "configs": list(CONFIGS),
            "tasks_per_suite": 10,
            "trials_per_task": 10,
            "held_out_init_offset": 10,
            "denoising_steps": 8,
            "num_steps_wait": 10,
            "pi05_replan_steps": 5,
            "expected_episodes": 1600,
            "failed_episode_policy": "retain as failure; do not drop or replace",
            "test_result_feedback_allowed": False,
            "pack_rebuild_allowed": False,
            "threshold_or_recipe_retuning_allowed": False,
        },
        "preflight": {
            "models": list(MODELS),
            "config": "omega_qvla_w4a4",
            "suite": "object",
            "task_id": 0,
            "held_out_init_offset": 10,
            "trials": 1,
            "diagnostic_only": True,
        },
        "cells": cells,
        "artifacts": {
            "runner": artifact(REPO_ROOT / "scripts/run_omega_qvla_table1.sh"),
            "queue": artifact(REPO_ROOT / "scripts/run_omega_qvla_table1_queue.sh"),
            "aggregator": artifact(REPO_ROOT / "scripts/tools/aggregate_omega_qvla_table1.py"),
            "finalizer": artifact(REPO_ROOT / "scripts/tools/finalize_omega_qvla_table1.py"),
            "gr00t_launcher": artifact(OFFICIAL_CODE / "scripts/run_groot_benchmark.sh"),
            "pi05_launcher": artifact(OFFICIAL_CODE / "scripts/run_pi05_libero_benchmark.sh"),
            "multi_gpu_harness": artifact(OFFICIAL_CODE / "scripts/run_libero_duquant_benchmark_multi_gpu.py"),
        },
    }


def frozen_write(path: Path, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == rendered, f"frozen protocol drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("materialize", "verify", "status"))
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps({
            "materialized": PROTOCOL.is_file(),
            "protocol": str(PROTOCOL),
            "sha256": sha256_file(PROTOCOL) if PROTOCOL.is_file() else None,
            "download_complete": DOWNLOAD.is_file(),
        }, indent=2))
        return
    if args.command == "materialize":
        require(not PROTOCOL.exists(), f"refusing to overwrite frozen protocol: {PROTOCOL}")
        ensure_result_blind()
        value = build()
        frozen_write(PROTOCOL, value)
        status = "created"
    else:
        require(PROTOCOL.is_file(), f"missing protocol: {PROTOCOL}")
        existing = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        require(existing == build(existing["created_utc"]), "protocol or referenced artifact drift")
        status = "verified"
    print(json.dumps({"status": status, "protocol": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)}, indent=2))


if __name__ == "__main__":
    main()
