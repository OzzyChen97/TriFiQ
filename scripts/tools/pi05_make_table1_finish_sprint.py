#!/usr/bin/env python3
"""Freeze exact remaining Table-1 keys into a one-worker-per-server finish sprint."""

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


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_committed(run_dir: Path, allowed_hashes: set[str]) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for name in glob.glob(str(run_dir / "results" / CONFIG_ID / "*.jsonl")):
        with open(name, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
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
    return keys


def parse_server(spec: str, allowed_hashes: set[str]) -> dict[str, Any]:
    instance, gpu, port, runtime_text = spec.split(",", 3)
    runtime = Path(runtime_text).expanduser().resolve()
    metadata = json.loads(runtime.read_text(encoding="utf-8"))
    metadata_hash = canonical_hash(metadata)
    if (metadata.get("openpi_runtime") or {}).get("config_id") != CONFIG_ID:
        raise ValueError(f"runtime/config mismatch: {instance}")
    if metadata_hash not in allowed_hashes:
        raise ValueError(f"runtime outside frozen base manifest: {instance}")
    return {
        "instance": instance,
        "gpu": int(gpu),
        "port": int(port),
        "runtime_path": str(runtime),
        "runtime_file_sha256": sha256_file(runtime),
        "server_metadata_sha256": metadata_hash,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_path = Path(args.base_manifest).expanduser().resolve()
    parent_path = Path(args.parent_schedule).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    scheduler = Path(args.scheduler).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if base.get("immutable") is not True or base.get("config_id") != CONFIG_ID:
        raise ValueError("invalid immutable base manifest")
    if parent.get("immutable") is not True or parent.get("protocol_change") is not False:
        raise ValueError("invalid immutable parent schedule")
    if parent.get("base_manifest_sha256") != sha256_file(base_path):
        raise ValueError("parent/base mismatch")

    allowed_hashes = {row["server_metadata_sha256"] for row in base["servers"]}
    servers = [parse_server(value, allowed_hashes) for value in args.server]
    if len({row["instance"] for row in servers}) != len(servers):
        raise ValueError("duplicate server instance")
    if len({row["port"] for row in servers}) != len(servers):
        raise ValueError("duplicate server port")

    expected = {
        (task_set, task, seed)
        for task_set, tasks in base["table1_protocol"]["tasks"].items()
        for task in tasks
        for seed in range(50)
    }
    committed = read_committed(run_dir, allowed_hashes)
    if not committed <= expected:
        raise ValueError(f"committed keys outside protocol: {len(committed - expected)}")
    missing = sorted(expected - committed)
    assignments: list[dict[str, Any]] = []
    for index, server in enumerate(servers):
        keys = missing[index:: len(servers)]
        if not keys:
            continue
        assignments.append(
            {
                "worker_id": f"finish_sprint_{index:02d}",
                "server_instance": server["instance"],
                "gpu": server["gpu"],
                "port": server["port"],
                "server_metadata_sha256": server["server_metadata_sha256"],
                "keys": [
                    {"task_set": task_set, "task": task, "seed": seed}
                    for task_set, task, seed in keys
                ],
            }
        )
    assigned = {
        (key["task_set"], key["task"], key["seed"])
        for worker in assignments
        for key in worker["keys"]
    }
    if assigned != set(missing):
        raise ValueError("finish-sprint assignment coverage drift")

    return {
        "schema_version": 1,
        "kind": "dypac_vla_pi05_table1_exact_finish_sprint",
        "immutable": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_change": False,
        "reason": "user-authorized use of remaining VRAM and finer exact-key evaluator queues",
        "base_manifest": str(base_path),
        "base_manifest_sha256": sha256_file(base_path),
        "parent_schedule": str(parent_path),
        "parent_schedule_sha256": sha256_file(parent_path),
        "scheduler": str(scheduler),
        "scheduler_sha256": sha256_file(scheduler),
        "schedule_tool": str(Path(__file__).resolve()),
        "schedule_tool_sha256": sha256_file(Path(__file__).resolve()),
        "pre_switch_committed": {
            "count": len(committed),
            "keyset_sha256": canonical_hash(sorted(committed)),
        },
        "remaining": {
            "count": len(missing),
            "keyset_sha256": canonical_hash(missing),
        },
        "coverage": {
            "expected": len(expected),
            "committed_plus_assigned": len(committed | assigned),
            "keyset_sha256": canonical_hash(sorted(committed | assigned)),
        },
        "servers": servers,
        "assignments": assignments,
        "resume_semantics": "global committed key; stopped in-flight episodes are reconstructed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--parent-schedule", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--server", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable schedule: {output}")
    candidate = build(args)
    if candidate["coverage"]["committed_plus_assigned"] != candidate["coverage"]["expected"]:
        raise ValueError("incomplete finish-sprint coverage")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{output.name}.", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(candidate, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)
    print(f"finish schedule created: {output}")
    print(f"schedule sha256: {sha256_file(output)}")
    print(f"committed={candidate['pre_switch_committed']['count']}")
    print(f"assigned={candidate['remaining']['count']}")
    print(f"workers={len(candidate['assignments'])}")


if __name__ == "__main__":
    main()
