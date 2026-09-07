#!/usr/bin/env python3
"""Freeze the outcome-blind correction of the tail-supervisor shell bug."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tools"))

from quantvla_cross_model_protocol import sha256_file  # noqa: E402
from quantvla_outputimpact import atomic_json  # noqa: E402


ROOT = REPO / "runs/full_context_v2/pi05_fcp_diagnostic"
OUT = ROOT / "tail_supervisor_erratum.json"


def artifact(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> None:
    workers = ROOT / "control" / "workers"
    immediate_logs = sorted(workers.glob("tail2_*.log"))
    if len(immediate_logs) != 9:
        raise ValueError(f"expected 9 immediate tail logs, found {len(immediate_logs)}")
    for worker_id in ("tail2_single_best_s8", "tail2_two_best_s8"):
        if (workers / f"{worker_id}.pid").exists() or (workers / f"{worker_id}.log").exists():
            raise ValueError(f"delayed job unexpectedly launched before erratum: {worker_id}")
    execution = json.loads((ROOT / "execution_manifest.json").read_text(encoding="utf-8"))
    extra_servers = []
    for arm, instance in (
        ("single_best", "pi05_fcp_single_best_x1"),
        ("two_best", "pi05_fcp_two_best_x1"),
    ):
        runtime_path = ROOT / "control" / f"{instance}.runtime.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        semantic = runtime["openpi_runtime"]["semantic_metadata_sha256"]
        allowed = {row["server_metadata_sha256"] for row in execution["servers"][arm]}
        if semantic not in allowed:
            raise ValueError(f"extra server semantic hash is not frozen for {arm}")
        extra_servers.append(
            {
                "arm": arm,
                "instance": instance,
                "gpu": 1,
                "semantic_metadata_sha256": semantic,
                "runtime": artifact(runtime_path),
                "server_log": artifact(ROOT / "control" / f"{instance}.server.log"),
            }
        )
    smoke_logs = sorted((ROOT / "preflight" / "tail_extra_gpu1").glob("*.log"))
    if len(smoke_logs) != 2:
        raise ValueError(f"expected two extra-server smoke logs, found {len(smoke_logs)}")
    payload = {
        "schema_version": 1,
        "kind": "pi05_fcp_outcome_blind_tail_supervisor_erratum",
        "immutable": True,
        "scientific_protocol_changed": False,
        "model_or_artifact_changed": False,
        "runtime_semantics_changed": False,
        "statistics_changed": False,
        "success_values_used": False,
        "cause": (
            "Bash expanded worker_id from the caller while constructing another local "
            "variable in the same local declaration, so the supervisor looked for a "
            "not-yet-created tail log instead of the stage-1 predecessor log."
        ),
        "effect": (
            "The first supervisor exited before launching either dependency-delayed job; "
            "all nine immediate jobs continued unchanged."
        ),
        "tail_schedule_adjustment": artifact(ROOT / "tail_schedule_adjustment.json"),
        "failed_supervisor_log": artifact(ROOT / "control" / "tail_schedule_supervisor.log"),
        "immediate_job_logs_at_correction": [artifact(path) for path in immediate_logs],
        "extra_gpu1_servers": extra_servers,
        "extra_server_smoke_logs": [artifact(path) for path in smoke_logs],
        "corrected_schedule": {
            "single_best_shard_8": "extra GPU-1 single_best server; EGL on GPU 0",
            "two_best_shard_8": "extra GPU-1 two_best server; EGL on GPU 1",
            "single_server_request_concurrency": 1,
            "gpu_scope": "GPU 0--7",
        },
        "corrected_supervisor": artifact(REPO / "scripts/supervise_pi05_fcp_tail_v2.sh"),
    }
    if OUT.is_file():
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("immutable tail-supervisor erratum drift")
    else:
        atomic_json(OUT, payload)
    print(json.dumps({"path": str(OUT), "sha256": sha256_file(OUT)}, indent=2))


if __name__ == "__main__":
    main()
