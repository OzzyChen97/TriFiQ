#!/usr/bin/env python3
"""Freeze an outcome-blind schedule amendment for stage-tail imbalance."""

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
OUT = ROOT / "tail_schedule_adjustment.json"
CONFIG = "full_context_w4a8_dynamic_profile"
ARMS = ("transferred_initializer", "single_best", "two_best")


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


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
                    raise ValueError(f"duplicate row before tail adjustment: {arm}/{key}")
                seen.add(key)
        counts[arm] = len(seen)
    return counts


def main() -> None:
    workers = ROOT / "control" / "workers"
    stage1_logs = sorted(workers.glob("clean1_*.log"))
    completed = [
        path for path in stage1_logs
        if "formal seeded worker complete:" in path.read_text(encoding="utf-8", errors="replace")
    ]
    if len(stage1_logs) != 16 or len(completed) < 14:
        raise ValueError(
            f"tail adjustment requires 16 stage-1 logs and at least 14 complete; "
            f"found {len(stage1_logs)}/{len(completed)}"
        )
    launcher = REPO / "scripts/run_pi05_fcp_tail_schedule.sh"
    payload = {
        "schema_version": 1,
        "kind": "pi05_fcp_outcome_blind_tail_schedule_adjustment",
        "immutable": True,
        "scientific_protocol_changed": False,
        "model_or_artifact_changed": False,
        "runtime_semantics_changed": False,
        "statistics_changed": False,
        "success_values_used_for_scheduling": False,
        "trigger": (
            "Fourteen of sixteen first-stage shards completed while two long-horizon "
            "shard-0 jobs kept the fixed stage barrier active and left GPUs 4--7 idle."
        ),
        "completed_row_counts_before_adjustment_without_success_values": completed_counts(),
        "completed_stage_1_logs": [artifact(path) for path in completed],
        "execution_manifest": artifact(ROOT / "execution_manifest.json"),
        "parent_server_restart_adjustment": artifact(ROOT / "server_restart_adjustment.json"),
        "recovery_launcher": artifact(launcher),
        "resource_change": {
            "old_supervisor_stopped": True,
            "immediate_stage_2_jobs": 9,
            "dependency_delayed_stage_2_jobs": 2,
            "concurrency_rule": (
                "Launch a stage-2 shard only when its assigned server has no stage-1 "
                "evaluator; at most one formal evaluator per retained server."
            ),
            "gpu_coverage": "Immediate jobs restore formal work on every GPU in 2--7.",
            "resume_rule": (
                "All jobs use the frozen global task-seed resume index; committed keys are "
                "skipped and incomplete keys are retried."
            ),
        },
    }
    if OUT.is_file():
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("immutable tail-schedule adjustment drift")
    else:
        atomic_json(OUT, payload)
    print(json.dumps({"path": str(OUT), "sha256": sha256_file(OUT)}, indent=2))


if __name__ == "__main__":
    main()
