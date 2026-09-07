#!/usr/bin/env python3
"""Freeze the outcome-blind one-worker-per-server recovery amendment."""

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
OUT = ROOT / "concurrency_adjustment.json"
CONFIG = "full_context_w4a8_dynamic_profile"
ARMS = ("transferred_initializer", "single_best", "two_best")


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
                    raise ValueError(f"duplicate row before concurrency adjustment: {arm}/{key}")
                seen.add(key)
        counts[arm] = len(seen)
    return counts


def main() -> None:
    workers = ROOT / "control/workers"
    failed_logs = []
    for path in sorted(workers.glob("safe1_*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(
            marker in text
            for marker in (
                "CUDA out of memory",
                "CUDNN_STATUS_INTERNAL_ERROR",
                "CUBLAS_STATUS_NOT_INITIALIZED",
                "unable to find an engine",
                "Segmentation fault",
            )
        ):
            failed_logs.append(artifact(path))
    if not failed_logs:
        raise ValueError("concurrency adjustment requires recorded stage-1 failures")

    launcher = REPO / "scripts/recover_pi05_fcp_one_worker_per_server.sh"
    payload = {
        "schema_version": 1,
        "kind": "pi05_fcp_outcome_blind_concurrency_adjustment",
        "immutable": True,
        "scientific_protocol_changed": False,
        "model_or_artifact_changed": False,
        "statistics_changed": False,
        "success_values_read_by_adjustment": False,
        "trigger": (
            "The 42-worker recovery produced three cuDNN/cuBLAS workspace failures "
            "on GPUs 5--6 at simultaneous-request peaks."
        ),
        "completed_row_counts_before_adjustment_without_success_values": completed_counts(),
        "failed_attempt_logs": failed_logs,
        "execution_manifest": artifact(ROOT / "execution_manifest.json"),
        "parent_resource_adjustment": artifact(ROOT / "resource_adjustment.json"),
        "recovery_launcher": artifact(launcher),
        "resource_change": {
            "retained_servers": 16,
            "failed_stage_workers": 42,
            "serial_stage_1_workers": 16,
            "serial_stage_2_workers": 11,
            "concurrency_rule": (
                "At most one formal evaluator may issue requests to each retained model "
                "server. GPU 2 runs one evaluator; GPUs 3--7 run at most three, one per server."
            ),
            "gpu_coverage": "Both stages retain active work on every GPU in 2--7.",
            "resume_rule": (
                "All jobs use the frozen global task-seed resume index; committed keys are "
                "skipped and incomplete keys are retried."
            ),
        },
    }
    if OUT.is_file():
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("immutable concurrency-adjustment drift")
    else:
        atomic_json(OUT, payload)
    print(
        json.dumps(
            {
                "path": str(OUT),
                "sha256": sha256_file(OUT),
                "failed_attempt_logs": len(failed_logs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
