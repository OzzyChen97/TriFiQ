#!/usr/bin/env python3
"""Freeze the max-free-VRAM resource amendment for pi0.5 DyPAC-VLA Table 1."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any


CONFIG_ID = "full_context_w4a8_dynamic_profile"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_server(spec: str, allowed_hashes: set[str]) -> dict[str, Any]:
    instance, gpu, port, runtime_text = spec.split(",", 3)
    runtime_path = Path(runtime_text).expanduser().resolve()
    metadata = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_hash = canonical_hash(metadata)
    if (metadata.get("openpi_runtime") or {}).get("config_id") != CONFIG_ID:
        raise ValueError(f"runtime/config mismatch: {instance}")
    if runtime_hash not in allowed_hashes:
        raise ValueError(f"runtime is outside the frozen base manifest: {instance}")
    return {
        "instance": instance,
        "gpu": int(gpu),
        "port": int(port),
        "runtime_path": str(runtime_path),
        "runtime_file_sha256": sha256_file(runtime_path),
        "server_metadata_sha256": runtime_hash,
    }


def parse_worker(spec: str) -> dict[str, Any]:
    worker_id, instance, egl_gpu, shard, shard_count, seeds = spec.split(",", 5)
    start, end = map(int, seeds.split("-", 1))
    row = {
        "worker_id": worker_id,
        "server_instance": instance,
        "egl_gpu": int(egl_gpu),
        "task_shard_index": int(shard),
        "task_shard_count": int(shard_count),
        "trial_seed_start": start,
        "trial_seed_end": end,
    }
    if not 0 <= row["task_shard_index"] < row["task_shard_count"]:
        raise ValueError(f"invalid task shard: {spec}")
    if not 0 <= start <= end <= 49:
        raise ValueError(f"invalid seed shard: {spec}")
    return row


def validate_layout(
    servers: list[dict[str, Any]], workers: list[dict[str, Any]], base: dict[str, Any]
) -> dict[str, Any]:
    instances = {row["instance"]: row for row in servers}
    if len(instances) != len(servers):
        raise ValueError("duplicate server instance")
    if len({row["port"] for row in servers}) != len(servers):
        raise ValueError("duplicate server port")
    worker_ids = [row["worker_id"] for row in workers]
    if len(set(worker_ids)) != len(worker_ids):
        raise ValueError("duplicate worker id")
    for worker in workers:
        server = instances.get(worker["server_instance"])
        if server is None or server["gpu"] != worker["egl_gpu"]:
            raise ValueError(f"worker/server GPU mismatch: {worker['worker_id']}")

    tasks = base["table1_protocol"]["tasks"]
    expected = {
        (split, task, seed)
        for split, names in tasks.items()
        for task in names
        for seed in range(50)
    }
    actual: set[tuple[str, str, int]] = set()
    duplicates = 0
    for worker in workers:
        shard = worker["task_shard_index"]
        count = worker["task_shard_count"]
        for split, names in tasks.items():
            selected = [task for index, task in enumerate(names) if index % count == shard]
            if not selected:
                raise ValueError(f"empty task-set shard: {worker['worker_id']}/{split}")
            for task in selected:
                for seed in range(worker["trial_seed_start"], worker["trial_seed_end"] + 1):
                    key = (split, task, seed)
                    duplicates += int(key in actual)
                    actual.add(key)
    if duplicates or actual != expected:
        raise ValueError(
            f"coverage drift: duplicates={duplicates}, missing={len(expected-actual)}, "
            f"extra={len(actual-expected)}"
        )
    return {"keys": len(actual), "keyset_sha256": canonical_hash(sorted(actual))}


def committed_rows(run_dir: Path, allowed_hashes: set[str]) -> dict[str, Any]:
    keys = set()
    for name in glob.glob(str(run_dir / "results" / CONFIG_ID / "*.jsonl")):
        for line_number, line in enumerate(open(name, encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["task_set"], row["task"], int(row["seed"]))
            if row.get("status") != "complete" or row.get("config") != CONFIG_ID:
                raise ValueError(f"invalid committed row: {name}:{line_number}")
            if row.get("server_metadata_sha256") not in allowed_hashes:
                raise ValueError(f"unfrozen runtime row: {name}:{line_number}")
            if key in keys:
                raise ValueError(f"duplicate committed key: {key}")
            keys.add(key)
    return {"completed": len(keys), "keyset_sha256": canonical_hash(sorted(keys))}


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_path = Path(args.base_manifest).expanduser().resolve()
    parent_path = Path(args.parent_schedule).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if base.get("immutable") is not True or base.get("config_id") != CONFIG_ID:
        raise ValueError("invalid immutable base manifest")
    if parent.get("immutable") is not True or parent.get("protocol_change") is not False:
        raise ValueError("invalid parent resource schedule")
    if parent.get("base_manifest_sha256") != sha256_file(base_path):
        raise ValueError("parent/base manifest mismatch")
    allowed_hashes = {row["server_metadata_sha256"] for row in base["servers"]}
    servers = [parse_server(value, allowed_hashes) for value in args.server]
    workers = [parse_worker(value) for value in args.worker]
    coverage = validate_layout(servers, workers, base)
    scheduler = Path(args.scheduler).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    return {
        "schema_version": 2,
        "kind": "dypac_vla_pi05_table1_max_free_vram_schedule_amendment",
        "immutable": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": (
            "user-authorized saturation of all GPUs with enough current VRAM; "
            "16 shards is the maximum non-empty shard count for every Table-1 split"
        ),
        "protocol_change": False,
        "base_manifest": str(base_path),
        "base_manifest_sha256": sha256_file(base_path),
        "parent_schedule": str(parent_path),
        "parent_schedule_sha256": sha256_file(parent_path),
        "frozen_protocol": base["table1_protocol"],
        "scheduler": str(scheduler),
        "scheduler_sha256": sha256_file(scheduler),
        "schedule_tool": str(Path(__file__).resolve()),
        "schedule_tool_sha256": sha256_file(Path(__file__).resolve()),
        "pre_switch_committed": committed_rows(run_dir, allowed_hashes),
        "servers": servers,
        "workers": workers,
        "coverage": coverage,
        "resume_semantics": (
            "global committed key (config,task_set,task,seed); interrupted in-flight "
            "episodes are discarded and deterministically reconstructed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--parent-schedule", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--server", action="append", required=True)
    parser.add_argument("--worker", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = Path(args.out).expanduser().resolve()
    candidate = build(args)
    if args.verify:
        existing = json.loads(output.read_text(encoding="utf-8"))
        candidate["created_utc"] = existing["created_utc"]
        candidate["pre_switch_committed"] = existing["pre_switch_committed"]
        if canonical_hash(candidate) != canonical_hash(existing):
            raise ValueError("immutable max-free-VRAM schedule drift")
        print(f"schedule verified: {output}")
        return
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable schedule: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{output.name}.", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(candidate, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)
    print(f"schedule created: {output}")
    print(f"schedule sha256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
