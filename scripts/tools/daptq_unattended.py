#!/usr/bin/env python3
"""Wait for frozen teacher trajectories and run the four DA-PTQ calibrations."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "daptq_table1"
SHARED = ROOT / "runs" / "qvla_actquant_table1"
PROTOCOL = ROOT / "scripts" / "daptq_table1_protocol.json"
GROOT_PY = "/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
OPENPI_PY = "/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
DEFAULT_ELIGIBLE_GPUS = (3, 4, 5, 6)
UNITS = {
    "gr00t_atomic_seen": {
        "model": "gr00t",
        "frozen_model_unit": "atomic_seen",
        "checkpoint": ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000",
    },
    "gr00t_composite_seen": {
        "model": "gr00t",
        "frozen_model_unit": "composite_seen",
        "checkpoint": ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000",
    },
    "gr00t_composite_unseen": {
        "model": "gr00t",
        "frozen_model_unit": "composite_unseen",
        "checkpoint": ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_unseen/checkpoint-60000",
    },
    "pi05_all_target": {
        "model": "pi05",
        "frozen_model_unit": "all_target",
        "checkpoint": ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_event(value: dict[str, Any]) -> None:
    path = RUN / "control" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": now(), **value}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def frozen_ready() -> dict[str, Any]:
    result = {}
    for unit in UNITS:
        path = SHARED / "frozen" / f"{unit}.json"
        ready = False
        error = None
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                ready = (
                    value.get("kind") == "qvla_actquant_frozen_calibration"
                    and len(value.get("episodes") or []) == 512
                    and value.get("model_unit") == UNITS[unit]["frozen_model_unit"]
                    and value.get("model") == UNITS[unit]["model"]
                    and value.get("test_results_used") is False
                )
                if not ready:
                    error = "identity_or_coverage_mismatch"
            except Exception as exc:  # fail-closed status only
                error = repr(exc)
        result[unit] = {"path": str(path), "ready": ready, "error": error}
    return result


def build_provenance() -> dict[str, Any]:
    shared = json.loads((SHARED / "provenance.json").read_text(encoding="utf-8"))
    source_checkout = Path("/home1/gyy/vla/DA-PTQ")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_checkout, text=True
    ).strip()
    if head != "94476e3a9fd1ad892bbdf44f979dd15373365c2a":
        raise ValueError(f"DA-PTQ checkout commit drift: {head}")
    source_files = []
    for relative in ("README.md", "scripts/quantize_quantvla.py", "vla/quantvla_ptq.py"):
        path = source_checkout / relative
        source_files.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    implementation_files = [
        ROOT / "code/daptq/core.py",
        ROOT / "code/daptq/runtime.py",
        ROOT / "code/qvla_actquant/model_adapters.py",
        ROOT / "scripts/tools/run_daptq_calibration.py",
        ROOT / "scripts/inference_service.py",
        ROOT / "code/pi05/openpi/scripts/serve_pi05_quant_policy.py",
    ]
    record = {
        "schema_version": 1,
        "kind": "daptq_robocasa365_provenance",
        "created_at": now(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "source": {
            "repository": "https://github.com/Killerxuuuuu/DA-PTQ",
            "checkout": str(source_checkout),
            "commit": head,
            "license_declared": False,
            "files": source_files,
        },
        "implementation": {
            "policy": "clean_room_from_paper_with_source_behavior_audit",
            "files": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in implementation_files
            ],
        },
        "checkpoints": shared["checkpoints"],
        "shared_teacher_provenance": str((SHARED / "provenance.json").resolve()),
        "shared_teacher_provenance_sha256": sha256_file(SHARED / "provenance.json"),
        "test_results_used": False,
    }
    record["record_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return record


def build_jobs(provenance: dict[str, Any]) -> dict[str, Any]:
    jobs = []
    # Validate the distinct OpenPI training/backward path first.  This avoids
    # spending three GR00T calibrations before surfacing a pi0.5-only failure.
    unit_order = (
        "pi05_all_target",
        "gr00t_atomic_seen",
        "gr00t_composite_seen",
        "gr00t_composite_unseen",
    )
    for unit in unit_order:
        config = UNITS[unit]
        model = config["model"]
        python = GROOT_PY if model == "gr00t" else OPENPI_PY
        output = RUN / "artifacts" / unit
        frozen = SHARED / "frozen" / f"{unit}.json"
        checkpoint_sha = provenance["checkpoints"][unit]["checkpoint_manifest_sha256"]
        env = {
            "GR00T_MODEL_DTYPE" if model == "gr00t" else "OPENPI_MODEL_DTYPE": "bfloat16",
            # GPU jobs are independent.  Bound host math pools so concurrent
            # calibration units do not each create a machine-wide pool.
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
            # NPZ members are independently compressed; ordered thread-map
            # parallelism removes the single-core decompression bottleneck.
            "DAPTQ_IO_WORKERS": "8",
            "PYTHONPATH": ":".join(
                [
                    str(ROOT / "code"),
                    str(ROOT / "scripts/tools"),
                    str(ROOT / "code/pi05/openpi/src"),
                ]
            ),
            "TORCHDYNAMO_DISABLE": "1",
        }
        jobs.append(
            {
                "id": f"calibrate_{unit}",
                "argv": [
                    python,
                    str(ROOT / "scripts/tools/run_daptq_calibration.py"),
                    "--model-family",
                    model,
                    "--checkpoint",
                    str(config["checkpoint"]),
                    "--checkpoint-sha256",
                    checkpoint_sha,
                    "--frozen-manifest",
                    str(frozen),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cuda:0",
                ],
                "cwd": str(ROOT),
                "env": env,
                # pi0.5 peaks below 15 GiB in the audited backward preflight;
                # the scheduler adds a separate 3 GiB device reserve.
                "required_mib": 20000 if model == "pi05" else 22000,
                "expected_outputs": [
                    str(output / "manifest.json"),
                    str(output / "pack_arrays.npz"),
                ],
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "daptq_gpu_jobs_v1",
        "eligible_gpus": list(DEFAULT_ELIGIBLE_GPUS),
        "protocol_sha256": sha256_file(PROTOCOL),
        "provenance_sha256": provenance["record_sha256"],
        "jobs": jobs,
    }
    payload["jobs_sha256"] = hashlib.sha256(
        json.dumps(jobs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def verify_artifacts() -> dict[str, Any]:
    records = {}
    for unit, config in UNITS.items():
        path = RUN / "artifacts" / unit / "manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        arrays = path.parent / value["arrays_file"]
        expected = {
            "kind": "daptq_robocasa365_pack_v1",
            "method": "DA-PTQ",
            "model_family": config["model"],
            "model_unit": config["frozen_model_unit"],
            "upstream_commit": "94476e3a9fd1ad892bbdf44f979dd15373365c2a",
            "test_results_used": False,
            "flow_steps": 4,
            "trajectory_count": 512,
        }
        mismatches = {
            key: (value.get(key), wanted)
            for key, wanted in expected.items()
            if value.get(key) != wanted
        }
        if mismatches:
            raise ValueError(f"{path}: DA-PTQ manifest mismatch: {mismatches}")
        if not arrays.is_file() or sha256_file(arrays) != value["arrays_sha256"]:
            raise ValueError(f"{path}: DA-PTQ array bundle mismatch")
        if int(value["storage"]["total_static_bytes"]) >= int(
            value["storage"]["fp16_baseline_bytes"]
        ):
            raise ValueError(f"{path}: DA-PTQ pack did not compress its Table-1 scope")
        records[unit] = {
            "manifest": str(path),
            "manifest_sha256": sha256_file(path),
            "arrays_sha256": value["arrays_sha256"],
            "storage": value["storage"],
        }
    result = {"verified_at": now(), "artifacts": records, "test_results_used": False}
    atomic_json(RUN / "artifact_inventory.json", result)
    return result


def execute(poll_seconds: int) -> None:
    lock_path = RUN / "control" / "controller.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        append_event({"event": "controller_started", "pid": os.getpid()})
        while True:
            state = frozen_ready()
            atomic_json(RUN / "control" / "status.json", {"stage": "waiting_for_frozen", "state": state})
            if all(row["ready"] for row in state.values()):
                break
            time.sleep(max(10, poll_seconds))
        provenance = build_provenance()
        atomic_json(RUN / "provenance.json", provenance)
        jobs = build_jobs(provenance)
        jobs_path = RUN / "jobs.json"
        atomic_json(jobs_path, jobs)
        atomic_json(RUN / "control" / "status.json", {"stage": "calibrating", "jobs": jobs["jobs_sha256"]})
        completed = subprocess.run(
            [
                GROOT_PY,
                str(ROOT / "scripts/tools/qvla_actquant_gpu_scheduler.py"),
                "--manifest",
                str(jobs_path),
                "--run-dir",
                str(RUN / "scheduler"),
                "--reserve-mib",
                "3072",
                "--poll-seconds",
                "15",
                "--launch-settle-seconds",
                "45",
            ],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"DA-PTQ calibration scheduler failed: {completed.returncode}")
        inventory = verify_artifacts()
        atomic_json(
            RUN / "control" / "status.json",
            {"stage": "calibration_complete", "artifact_inventory": inventory},
        )
        append_event({"event": "calibration_complete"})
        atomic_json(
            RUN / "control" / "status.json",
            {"stage": "formal_running", "artifact_inventory": inventory},
        )
        formal = subprocess.run(
            [
                GROOT_PY,
                str(ROOT / "scripts/tools/daptq_formal_pipeline.py"),
                "run",
                "--max-workers",
                "32",
                "--workers-per-gpu",
                "4",
                "--replicas",
                "2",
                "--poll-seconds",
                "60",
            ],
            cwd=ROOT,
            check=False,
        )
        if formal.returncode:
            raise RuntimeError(f"DA-PTQ formal pipeline failed: {formal.returncode}")
        complete_path = RUN / "formal" / "complete.json"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "complete" or int(complete.get("new_formal_episodes", 0)) != 5000:
            raise ValueError(f"DA-PTQ formal completion drift: {complete}")
        atomic_json(
            RUN / "control" / "status.json",
            {
                "stage": "complete",
                "artifact_inventory": inventory,
                "formal": complete,
            },
        )
        append_event({"event": "formal_complete", "episodes": 5000})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.command == "status":
        status_path = RUN / "control" / "status.json"
        print(
            status_path.read_text(encoding="utf-8")
            if status_path.is_file()
            else json.dumps({"stage": "not_started", "frozen": frozen_ready()}, indent=2)
        )
        return
    try:
        execute(args.poll_seconds)
    except Exception as error:
        atomic_json(RUN / "control" / "status.json", {"stage": "failed", "error": repr(error)})
        append_event({"event": "controller_failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
