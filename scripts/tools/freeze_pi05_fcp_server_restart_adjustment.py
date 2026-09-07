#!/usr/bin/env python3
"""Freeze the outcome-blind restart of CUDA contexts poisoned by prior OOMs."""

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
OUT = ROOT / "server_restart_adjustment.json"
CONFIG = "full_context_w4a8_dynamic_profile"
ARMS = ("transferred_initializer", "single_best", "two_best")
RESTARTED = (
    "pi05_fcp_single_best_s2",
    "pi05_fcp_single_best_s3",
    "pi05_fcp_two_best_s1",
    "pi05_fcp_two_best_s3",
)


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def completed_counts() -> dict[str, int]:
    counts = {arm: 0 for arm in ARMS}
    for arm in ARMS:
        seen = set()
        result_dir = ROOT / "rollouts" / arm / "results" / CONFIG
        for path in sorted(result_dir.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("config") != CONFIG or row.get("status") != "complete":
                    continue
                key = (str(row["task_set"]), str(row["task"]), int(row["seed"]))
                if key in seen:
                    raise ValueError(f"duplicate row before server restart: {arm}/{key}")
                seen.add(key)
        counts[arm] = len(seen)
    return counts


def runtime_semantic_hash(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value["openpi_runtime"]["semantic_metadata_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"invalid semantic runtime hash: {path}")
    return digest


def main() -> None:
    audit = ROOT / "operational_audit" / "pre_poisoned_restart"
    control = ROOT / "control"
    restart_records = []
    for instance in RESTARTED:
        old_runtime = audit / f"{instance}.runtime.json"
        old_log = audit / f"{instance}.server.log"
        new_runtime = control / f"{instance}.runtime.json"
        new_log = control / f"{instance}.server.log"
        old_semantic = runtime_semantic_hash(old_runtime)
        new_semantic = runtime_semantic_hash(new_runtime)
        if old_semantic != new_semantic:
            raise ValueError(f"semantic runtime changed while restarting {instance}")
        restart_records.append(
            {
                "instance": instance,
                "semantic_metadata_sha256": new_semantic,
                "pre_restart_runtime": artifact(old_runtime),
                "pre_restart_server_log": artifact(old_log),
                "post_restart_runtime": artifact(new_runtime),
                "post_restart_server_log": artifact(new_log),
            }
        )

    failed_logs = []
    workers = control / "workers"
    for path in sorted(workers.glob("serial1_*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(
            marker in text
            for marker in (
                "CUDNN_STATUS_INTERNAL_ERROR",
                "CUBLAS_STATUS_NOT_INITIALIZED",
                "unable to find an engine",
                "CUDA out of memory",
            )
        ):
            failed_logs.append(artifact(path))
    if not failed_logs:
        raise ValueError("server restart requires recorded post-adjustment CUDA failures")

    smoke_logs = [artifact(path) for path in sorted((ROOT / "preflight" / "post_restart").glob("*.log"))]
    if len(smoke_logs) != 7:
        raise ValueError(f"expected 7 post-restart smoke logs, found {len(smoke_logs)}")

    launcher = REPO / "scripts/run_pi05_fcp_after_server_restart.sh"
    payload = {
        "schema_version": 1,
        "kind": "pi05_fcp_outcome_blind_server_restart_adjustment",
        "immutable": True,
        "scientific_protocol_changed": False,
        "model_or_artifact_changed": False,
        "runtime_semantics_changed": False,
        "statistics_changed": False,
        "success_values_read_by_adjustment": False,
        "trigger": (
            "One-worker-per-server retries failed only on retained servers whose CUDA "
            "contexts had previously encountered workspace allocation errors."
        ),
        "completed_row_counts_before_adjustment_without_success_values": completed_counts(),
        "failed_attempt_logs": failed_logs,
        "execution_manifest": artifact(ROOT / "execution_manifest.json"),
        "parent_concurrency_adjustment": artifact(ROOT / "concurrency_adjustment.json"),
        "recovery_launcher": artifact(launcher),
        "restarted_servers": restart_records,
        "post_restart_smoke_logs": smoke_logs,
        "resource_change": {
            "operation": "restart four retained model-server processes with unchanged frozen inputs",
            "formal_concurrency": "one evaluator per server; 16 jobs in stage 1 and 11 in stage 2",
            "gpu_coverage": "both formal stages use every GPU in 2--7",
            "resume_rule": (
                "All jobs use the frozen global task-seed resume index; committed keys are "
                "skipped and incomplete keys are retried."
            ),
        },
    }
    if OUT.is_file():
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("immutable server-restart adjustment drift")
    else:
        atomic_json(OUT, payload)
    print(json.dumps({"path": str(OUT), "sha256": sha256_file(OUT)}, indent=2))


if __name__ == "__main__":
    main()
