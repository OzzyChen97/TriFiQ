#!/usr/bin/env python3
"""Create or verify an immutable four-config parallel schedule amendment."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
from pathlib import Path
from typing import Any


CONFIGS = (
    "fp16",
    "quantvla_w4a8_atmohb",
    "gdsq_vla_atmohb",
    "gdsq_vla",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_server(spec: str) -> dict[str, Any]:
    instance, config, gpu, port, runtime = spec.split(",", 4)
    if config not in CONFIGS:
        raise ValueError(f"unknown config in server spec: {config}")
    runtime_path = Path(runtime).expanduser().resolve()
    metadata = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime = metadata.get("openpi_runtime") or {}
    if runtime.get("config_id") != config:
        raise ValueError(f"runtime/config mismatch for {instance}")
    return {
        "instance": instance,
        "config_id": config,
        "gpu": int(gpu),
        "port": int(port),
        "runtime_path": str(runtime_path),
        "runtime_file_sha256": sha256_file(runtime_path),
        "server_metadata_sha256": canonical_hash(metadata),
    }


def parse_worker(spec: str) -> dict[str, Any]:
    worker_id, config, instance, shard, shard_count, seeds = spec.split(",", 5)
    start_text, end_text = seeds.split("-", 1)
    start, end = int(start_text), int(end_text)
    shard_index, shard_total = int(shard), int(shard_count)
    if (
        config not in CONFIGS
        or start < 0
        or end > 49
        or start > end
        or shard_total <= 0
        or not 0 <= shard_index < shard_total
    ):
        raise ValueError(f"invalid worker spec: {spec}")
    return {
        "worker_id": worker_id,
        "config_id": config,
        "server_instance": instance,
        "task_shard_index": shard_index,
        "task_shard_count": shard_total,
        "trial_seed_start": start,
        "trial_seed_end": end,
    }


def validate_layout(servers: list[dict[str, Any]], workers: list[dict[str, Any]]) -> None:
    instances = {row["instance"]: row for row in servers}
    if len(instances) != len(servers):
        raise ValueError("duplicate parallel server instances")
    if {row["gpu"] for row in servers} != set(range(1, 8)):
        raise ValueError("parallel schedule must use GPUs 1-7 exactly once")
    if set(CONFIGS) != {row["config_id"] for row in servers}:
        raise ValueError("parallel schedule does not cover all four configs")
    seen_worker_ids: set[str] = set()
    for worker in workers:
        if worker["worker_id"] in seen_worker_ids:
            raise ValueError(f"duplicate worker id: {worker['worker_id']}")
        seen_worker_ids.add(worker["worker_id"])
        server = instances.get(worker["server_instance"])
        if server is None or server["config_id"] != worker["config_id"]:
            raise ValueError(f"worker/server mismatch: {worker['worker_id']}")

    for config in CONFIGS:
        rows = [row for row in workers if row["config_id"] == config]
        if not rows:
            raise ValueError(f"no workers for {config}")
        counts = {row["task_shard_count"] for row in rows}
        if len(counts) != 1:
            raise ValueError(f"mixed shard counts for {config}")
        shard_count = counts.pop()
        for shard in range(shard_count):
            intervals = sorted(
                (row["trial_seed_start"], row["trial_seed_end"])
                for row in rows
                if row["task_shard_index"] == shard
            )
            cursor = 0
            for start, end in intervals:
                if start != cursor:
                    raise ValueError(
                        f"worker seed coverage has gap/overlap for {config} "
                        f"shard={shard}: expected_start={cursor}, interval={start}-{end}"
                    )
                cursor = end + 1
            if cursor != 50:
                raise ValueError(
                    f"worker seed coverage is incomplete for {config} "
                    f"shard={shard}: covered_through={cursor - 1}"
                )


def committed_rows(run_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    all_keys: list[tuple[str, str, str, int]] = []
    for config in CONFIGS:
        rows: list[dict[str, Any]] = []
        for name in glob.glob(str(run_dir / "results" / config / "*.jsonl")):
            with open(name, encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        keys = [
            (row["config"], row["task_set"], row["task"], int(row["seed"]))
            for row in rows
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate committed keys before parallel switch: {config}")
        if any(row.get("status") != "complete" for row in rows):
            raise ValueError(f"non-complete row before parallel switch: {config}")
        if any(row.get("config") != config for row in rows):
            raise ValueError(f"wrong config row before parallel switch: {config}")
        all_keys.extend(keys)
        summary[config] = {
            "completed": len(rows),
            "successes": sum(bool(row.get("success")) for row in rows),
            "keyset_sha256": canonical_hash(sorted(keys)),
        }
    summary["all_configs_keyset_sha256"] = canonical_hash(sorted(all_keys))
    summary["total_completed"] = len(all_keys)
    return summary


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_path = Path(args.base_manifest).expanduser().resolve()
    scheduler_path = Path(args.scheduler).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("immutable") is not True:
        raise ValueError("base manifest is not immutable")
    servers = [parse_server(spec) for spec in args.server]
    workers = [parse_worker(spec) for spec in args.worker]
    validate_layout(servers, workers)

    allowed: dict[str, set[str]] = {config: set() for config in CONFIGS}
    for server in base.get("servers", []):
        allowed[server["config_id"]].add(server["server_metadata_sha256"])
    for server in servers:
        if server["server_metadata_sha256"] not in allowed[server["config_id"]]:
            raise ValueError(
                f"parallel runtime differs from frozen config: {server['instance']}"
            )

    output = {
        "schema_version": 2,
        "kind": "pi05_table1_parallel_schedule_amendment",
        "immutable": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": (
            "user-authorized worker saturation; resource-only repartition with "
            "model and Table-1 protocol unchanged"
        ),
        "base_manifest": str(base_path),
        "base_manifest_sha256": sha256_file(base_path),
        "scheduler": str(scheduler_path),
        "scheduler_sha256": sha256_file(scheduler_path),
        "schedule_tool": str(Path(__file__).resolve()),
        "schedule_tool_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_change": False,
        "frozen_protocol": base["table_1_protocol"],
        "pre_switch_committed": committed_rows(run_dir),
        "servers": servers,
        "workers": workers,
        "resume_semantics": (
            "global committed key (config,task_set,task,seed); incomplete trial discarded; "
            "fresh environment reconstructed"
        ),
    }
    if args.supersedes:
        superseded = Path(args.supersedes).expanduser().resolve()
        output["supersedes_schedule"] = str(superseded)
        output["supersedes_schedule_sha256"] = sha256_file(superseded)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--server", action="append", default=[])
    parser.add_argument("--worker", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--supersedes")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = Path(args.out).expanduser().resolve()
    candidate = build(args)
    if args.verify:
        existing = json.loads(output.read_text(encoding="utf-8"))
        for volatile in ("created_utc", "pre_switch_committed"):
            candidate[volatile] = existing[volatile]
        if canonical_hash(candidate) != canonical_hash(existing):
            raise ValueError("parallel schedule amendment no longer matches runtime/source")
        print(f"parallel schedule verified: {output}")
        return
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable schedule: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    temporary.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(f"parallel schedule created: {output}")
    print(f"parallel schedule sha256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
