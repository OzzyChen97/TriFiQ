#!/usr/bin/env python3
"""Freeze or verify a resource-only DyPAC-VLA pi0.5 Table-1 schedule amendment."""

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
    runtime = metadata.get("openpi_runtime") or {}
    metadata_hash = canonical_hash(metadata)
    if runtime.get("config_id") != CONFIG_ID or metadata_hash not in allowed_hashes:
        raise ValueError(f"unfrozen runtime for {instance}: {metadata_hash}")
    return {
        "instance": instance,
        "gpu": int(gpu),
        "port": int(port),
        "runtime_path": str(runtime_path),
        "runtime_file_sha256": sha256_file(runtime_path),
        "server_metadata_sha256": metadata_hash,
    }


def parse_worker(spec: str) -> dict[str, Any]:
    worker_id, instance, gpu, port, shard, shard_count, seeds = spec.split(",", 6)
    seed_start, seed_end = map(int, seeds.split("-", 1))
    row = {
        "worker_id": worker_id,
        "server_instance": instance,
        "gpu": int(gpu),
        "port": int(port),
        "task_shard_index": int(shard),
        "task_shard_count": int(shard_count),
        "trial_seed_start": seed_start,
        "trial_seed_end": seed_end,
    }
    if (
        not 0 <= row["task_shard_index"] < row["task_shard_count"]
        or not 0 <= seed_start <= seed_end <= 49
    ):
        raise ValueError(f"invalid worker spec: {spec}")
    return row


def validate_coverage(workers: list[dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
    task_sets = base["table1_protocol"]["tasks"]
    expected = {
        (split, task, seed)
        for split, tasks in task_sets.items()
        for task in tasks
        for seed in range(50)
    }
    actual: set[tuple[str, str, int]] = set()
    duplicates = 0
    for worker in workers:
        shard = worker["task_shard_index"]
        count = worker["task_shard_count"]
        for split, tasks in task_sets.items():
            selected = [task for index, task in enumerate(tasks) if index % count == shard]
            for task in selected:
                for seed in range(worker["trial_seed_start"], worker["trial_seed_end"] + 1):
                    key = (split, task, seed)
                    duplicates += int(key in actual)
                    actual.add(key)
    if duplicates or actual != expected:
        raise ValueError(
            f"schedule coverage drift: duplicates={duplicates}, "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
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
                raise ValueError(f"unfrozen committed runtime: {name}:{line_number}")
            if key in keys:
                raise ValueError(f"duplicate committed key: {key}")
            keys.add(key)
    return {"completed": len(keys), "keyset_sha256": canonical_hash(sorted(keys))}


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_path = Path(args.base_manifest).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("immutable") is not True or base.get("config_id") != CONFIG_ID:
        raise ValueError("invalid base manifest")
    allowed_hashes = {row["server_metadata_sha256"] for row in base["servers"]}
    servers = [parse_server(value, allowed_hashes) for value in args.server]
    workers = [parse_worker(value) for value in args.worker]
    if len({row["instance"] for row in servers}) != len(servers):
        raise ValueError("duplicate server instance")
    if len({row["gpu"] for row in servers}) != len(servers):
        raise ValueError("one formal server is required per GPU")
    server_by_instance = {row["instance"]: row for row in servers}
    for worker in workers:
        server = server_by_instance.get(worker["server_instance"])
        if not server or server["gpu"] != worker["gpu"] or server["port"] != worker["port"]:
            raise ValueError(f"worker/server mismatch: {worker['worker_id']}")
    coverage = validate_coverage(workers, base)
    run_dir = Path(args.run_dir).expanduser().resolve()
    scheduler = Path(args.scheduler).expanduser().resolve()
    return {
        "schema_version": 1,
        "kind": "dypac_vla_pi05_table1_resource_schedule_amendment",
        "immutable": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": "user-authorized use of every GPU with sufficient free VRAM",
        "protocol_change": False,
        "base_manifest": str(base_path),
        "base_manifest_sha256": sha256_file(base_path),
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
            "global committed key (config,task_set,task,seed); an interrupted in-flight "
            "episode is discarded and reconstructed with a fresh deterministic environment"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
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
            raise ValueError("immutable resource schedule drift")
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
