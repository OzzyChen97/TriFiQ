#!/usr/bin/env python3
"""Dynamic, non-evicting GPU scheduler for QVLA/ActQuant deterministic shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_event(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_snapshot() -> dict[int, dict[str, float]]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True,
    ).splitlines()
    result = {}
    for row in rows:
        index, used, free, utilization = [field.strip() for field in row.split(",")]
        result[int(index)] = {
            "used_mib": float(used), "free_mib": float(free), "utilization_pct": float(utilization),
        }
    return result


def gpu_process_snapshot() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
         "--format=csv,noheader,nounits"], text=True, stdout=subprocess.PIPE, check=False,
    )
    return [{"raw": row} for row in completed.stdout.splitlines() if row.strip()]


def normalize_job(job: dict[str, Any], root: Path) -> dict[str, Any]:
    required = {"id", "argv", "required_mib", "expected_outputs"}
    if not required <= set(job):
        raise ValueError(f"job lacks fields {sorted(required - set(job))}: {job}")
    if not isinstance(job["argv"], list) or not job["argv"] or not all(isinstance(v, str) for v in job["argv"]):
        raise ValueError(f"job argv must be a nonempty string list: {job['id']}")
    value = dict(job)
    value["cwd"] = str(Path(job.get("cwd", root)).resolve())
    value["expected_outputs"] = [str(Path(path).resolve()) for path in job["expected_outputs"]]
    value["required_mib"] = int(job["required_mib"])
    value["env"] = {str(key): str(item) for key, item in job.get("env", {}).items()}
    value["job_sha256"] = canonical_hash({key: value[key] for key in ("id", "argv", "cwd", "env", "required_mib", "expected_outputs")})
    return value


def completed_receipt(job: dict[str, Any], receipt: Path) -> bool:
    if not receipt.is_file():
        return False
    value = json.loads(receipt.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or value.get("job_sha256") != job["job_sha256"]:
        raise ValueError(f"scheduler receipt drift: {receipt}")
    records = value.get("outputs", [])
    if [record["path"] for record in records] != job["expected_outputs"]:
        raise ValueError(f"scheduler output inventory drift: {receipt}")
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"scheduler completed output drift: {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reserve-mib", type=int, default=4096)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--launch-settle-seconds", type=float, default=30.0)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "qvla_actquant_gpu_jobs_v1":
        raise ValueError("not a QVLA/ActQuant GPU job manifest")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = [normalize_job(job, manifest_path.parent) for job in manifest["jobs"]]
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate scheduler job id")
    eligible = [int(gpu) for gpu in manifest.get("eligible_gpus", sorted(gpu_snapshot()))]
    if len(eligible) != len(set(eligible)):
        raise ValueError("duplicate eligible GPU")
    receipts = run_dir / "receipts"
    logs = run_dir / "logs"
    receipts.mkdir(exist_ok=True); logs.mkdir(exist_ok=True)
    preflight = {
        "sampled_at": now(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dirty_worktree": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
        "disk": subprocess.check_output(["df", "-B1", str(run_dir)], text=True).splitlines(),
        "gpus": gpu_snapshot(),
        "existing_gpu_processes": gpu_process_snapshot(),
        "policy": "never evict; preserve reserve; at most one new launch per GPU per settle interval",
    }
    preflight_path = run_dir / "preflight.json"
    if preflight_path.exists():
        old = json.loads(preflight_path.read_text(encoding="utf-8"))
        if old["manifest_sha256"] != preflight["manifest_sha256"]:
            raise ValueError("scheduler manifest differs from existing run")
    else:
        atomic_json(preflight_path, preflight)
    complete_ids = {
        job["id"] for job in jobs if completed_receipt(job, receipts / f"{job['id']}.json")
    }
    pending = [job for job in jobs if job["id"] not in complete_ids]
    running: dict[str, dict[str, Any]] = {}
    last_launch = {gpu: 0.0 for gpu in eligible}
    events = run_dir / "events.jsonl"
    while pending or running:
        for job_id, item in list(running.items()):
            returncode = item["process"].poll()
            if returncode is None:
                continue
            item["handle"].close()
            job = item["job"]
            if returncode != 0:
                receipt = {
                    "status": "failed", "job_sha256": job["job_sha256"], "job_id": job_id,
                    "gpu": item["gpu"], "returncode": returncode, "ended_at": now(), "log": item["log"],
                }
                atomic_json(receipts / f"{job_id}.failed.json", receipt)
                append_event(events, receipt)
                raise RuntimeError(f"GPU job failed: {job_id}; see {item['log']}")
            outputs = []
            for output in job["expected_outputs"]:
                path = Path(output)
                if not path.is_file():
                    raise RuntimeError(f"GPU job {job_id} omitted expected output {path}")
                outputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
            receipt = {
                "status": "complete", "job_sha256": job["job_sha256"], "job_id": job_id,
                "gpu": item["gpu"], "returncode": 0, "ended_at": now(), "log": item["log"],
                "outputs": outputs,
            }
            atomic_json(receipts / f"{job_id}.json", receipt)
            append_event(events, receipt)
            complete_ids.add(job_id)
            del running[job_id]

        snapshot = gpu_snapshot()
        launched_gpus: set[int] = set()
        for job in list(pending):
            candidates = [
                gpu for gpu in eligible
                if gpu not in launched_gpus
                and time.monotonic() - last_launch[gpu] >= args.launch_settle_seconds
                and snapshot.get(gpu, {}).get("free_mib", 0) >= job["required_mib"] + args.reserve_mib
            ]
            if not candidates:
                continue
            gpu = max(candidates, key=lambda value: (snapshot[value]["free_mib"], -value))
            env = {**os.environ, **job["env"], "CUDA_VISIBLE_DEVICES": str(gpu)}
            log_path = logs / f"{job['id']}.log"
            handle = log_path.open("a", encoding="utf-8", buffering=1)
            process = subprocess.Popen(
                job["argv"], cwd=job["cwd"], env=env, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[job["id"]] = {
                "process": process, "handle": handle, "job": job, "gpu": gpu, "log": str(log_path),
            }
            pending.remove(job)
            launched_gpus.add(gpu); last_launch[gpu] = time.monotonic()
            event = {
                "status": "started", "job_id": job["id"], "job_sha256": job["job_sha256"],
                "gpu": gpu, "pid": process.pid, "required_mib": job["required_mib"],
                "free_mib_before_launch": snapshot[gpu]["free_mib"], "started_at": now(),
            }
            append_event(events, event)
            print(json.dumps(event), flush=True)
        atomic_json(run_dir / "status.json", {
            "sampled_at": now(), "pending": [job["id"] for job in pending],
            "running": {job_id: {"gpu": item["gpu"], "pid": item["process"].pid}
                        for job_id, item in running.items()},
            "complete": sorted(complete_ids),
            "gpus": snapshot,
        })
        if pending or running:
            time.sleep(max(1.0, args.poll_seconds))
    atomic_json(run_dir / "complete.json", {
        "status": "complete", "completed_at": now(), "jobs": len(jobs),
        "manifest_sha256": sha256_file(manifest_path),
    })


if __name__ == "__main__":
    main()
