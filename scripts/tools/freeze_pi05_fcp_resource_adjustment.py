#!/usr/bin/env python3
"""Freeze the outcome-blind resource-pressure recovery schedule."""

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
OUT = ROOT / "resource_adjustment.json"
CONFIG = "full_context_w4a8_dynamic_profile"
ARMS = ("transferred_initializer", "single_best", "two_best")
STOPPED_SERVERS = (
    "pi05_fcp_two_best_s4",
    "pi05_fcp_transferred_initializer_s5",
    "pi05_fcp_single_best_s4",
    "pi05_fcp_transferred_initializer_s6",
    "pi05_fcp_single_best_s5",
    "pi05_fcp_two_best_s5",
    "pi05_fcp_transferred_initializer_s7",
    "pi05_fcp_single_best_s6",
    "pi05_fcp_two_best_s6",
    "pi05_fcp_single_best_s8",
    "pi05_fcp_two_best_s8",
)


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def completed_counts() -> dict[str, int]:
    counts = {arm: 0 for arm in ARMS}
    for arm in ARMS:
        seen = set()
        for path in sorted((ROOT / "rollouts" / arm / "results" / CONFIG).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("config") != CONFIG or row.get("status") != "complete":
                    continue
                key = (str(row["task_set"]), str(row["task"]), int(row["seed"]))
                if key in seen:
                    raise ValueError(f"duplicate row before resource adjustment: {arm}/{key}")
                seen.add(key)
        counts[arm] = len(seen)
    return counts


def main() -> None:
    failure_logs = []
    workers = ROOT / "control/workers"
    for path in sorted(workers.glob("*.log")):
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
            failure_logs.append(artifact(path))
    if not failure_logs:
        raise ValueError("resource adjustment requires recorded memory-pressure failures")
    launcher = REPO / "scripts/recover_pi05_fcp_resource_pressure.sh"
    payload = {
        "schema_version": 1,
        "kind": "pi05_fcp_outcome_blind_resource_adjustment",
        "immutable": True,
        "scientific_protocol_changed": False,
        "model_or_artifact_changed": False,
        "statistics_changed": False,
        "success_values_read_by_adjustment": False,
        "trigger": (
            "Initial 54-worker saturation left 0.04--1.5 GiB on GPUs 5--7 and "
            "caused cuDNN/cuBLAS workspace failures or EGL process termination."
        ),
        "completed_row_counts_before_adjustment_without_success_values": completed_counts(),
        "failed_attempt_logs": failure_logs,
        "execution_manifest": artifact(ROOT / "execution_manifest.json"),
        "recovery_launcher": artifact(launcher),
        "resource_change": {
            "initial_servers": 27,
            "retained_servers": 16,
            "stopped_redundant_servers": list(STOPPED_SERVERS),
            "initial_workers": 54,
            "stage_1_workers": 42,
            "stage_2_workers": 12,
            "gpu_rule": (
                "GPU 2 retains one model plus two EGL workers; GPUs 3--7 retain "
                "three models and at most eight EGL workers, preserving roughly 5 GiB "
                "for inference workspaces."
            ),
            "resume_rule": (
                "Every recovery worker uses the frozen global task-seed resume index; "
                "committed keys are skipped and incomplete keys are retried."
            ),
        },
    }
    if OUT.is_file():
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("immutable resource-adjustment drift")
    else:
        atomic_json(OUT, payload)
    print(json.dumps({"path": str(OUT), "sha256": sha256_file(OUT), "failed_attempt_logs": len(failure_logs)}, indent=2))


if __name__ == "__main__":
    main()
