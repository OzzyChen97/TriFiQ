#!/usr/bin/env python3
"""Safely reshard incomplete pi0.5 teacher rollouts across all seven servers."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "qvla_actquant_table1"
PYTHON = "/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
UNIT = "pi05_all_target"
SHARDS = 9
PORTS = tuple(range(23100, 23107))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def alive(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return len(fields) > 2 and fields[2] != "Z"
    except (FileNotFoundError, PermissionError):
        return False


def command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return ""


def process_cwd(pid: int) -> str:
    try:
        return str(Path(f"/proc/{pid}/cwd").resolve())
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def committed_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    directory = RUN / "calibration" / UNIT
    for journal in sorted(directory.glob("worker_*.jsonl")):
        for line_number, line in enumerate(
            journal.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["episode_key"])
            if key in rows and rows[key] != row:
                raise ValueError(f"conflicting row {key} at {journal}:{line_number}")
            rows[key] = row
    return rows


def expected_keys() -> set[str]:
    manifest = json.loads(
        (RUN / "calibration_manifests" / f"{UNIT}.json").read_text(encoding="utf-8")
    )
    return {str(row["episode_key"]) for row in manifest["qvla_episodes"]}


def owned_live_collectors() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    owned, stale = [], []
    for pid_file in sorted((RUN / "collectors").glob(f"{UNIT}_s*.pid")):
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if not alive(pid):
            continue
        cmd = command(pid)
        row = {"pid": pid, "pid_file": str(pid_file.resolve()), "command": cmd}
        valid_out_dir = (
            "--out-dir runs/qvla_actquant_table1/calibration/pi05_all_target" in cmd
            or f"--out-dir {RUN / 'calibration' / UNIT}" in cmd
        )
        valid = (
            "collect_qvla_actquant_teacher.py" in cmd
            and "--model pi05" in cmd
            and valid_out_dir
            and process_cwd(pid) == str(ROOT)
        )
        if valid:
            owned.append(row)
        else:
            # PID files can outlive their processes and later refer to a reused
            # PID.  Record that state, but never signal the unrelated process.
            stale.append(row)
    unique = {int(row["pid"]): row for row in owned}
    return sorted(unique.values(), key=lambda row: int(row["pid"])), stale


def stop_collectors(records: list[dict[str, Any]]) -> None:
    for row in records:
        pid = int(row["pid"])
        if command(pid) != row["command"]:
            raise RuntimeError(f"collector PID identity changed before stop: {pid}")
        group = os.getpgid(pid)
        os.killpg(group, signal.SIGTERM) if group == pid else os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 120
    while time.time() < deadline and any(alive(int(row["pid"])) for row in records):
        time.sleep(1)
    remaining = [row["pid"] for row in records if alive(int(row["pid"]))]
    if remaining:
        raise RuntimeError(f"owned pi0.5 collectors did not stop after SIGTERM: {remaining}")


def launch_missing_shards() -> list[dict[str, Any]]:
    completed = set(committed_rows())
    expected = expected_keys()
    extra = completed - expected
    if extra:
        raise RuntimeError(f"pi0.5 calibration contains {len(extra)} extra keys")
    records = []
    directory = RUN / "collectors"
    for shard in range(SHARDS):
        shard_keys = {
            key
            for key in expected
            if int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % SHARDS
            == shard
        }
        missing = shard_keys - completed
        if not missing:
            records.append(
                {"shard": shard, "missing_before_launch": 0, "skipped_complete": True}
            )
            continue
        port = PORTS[shard % len(PORTS)]
        argv = [
            PYTHON,
            str(ROOT / "scripts/tools/collect_qvla_actquant_teacher.py"),
            "--manifest",
            str(RUN / "calibration_manifests" / f"{UNIT}.json"),
            "--model",
            "pi05",
            "--port",
            str(port),
            "--out-dir",
            str(RUN / "calibration" / UNIT),
            "--shard-index",
            str(shard),
            "--shard-count",
            str(SHARDS),
            "--egl-device",
            str(shard % 8),
        ]
        stem = f"{UNIT}_s{shard:02d}_of{SHARDS:02d}"
        log = directory / f"{stem}.log"
        with log.open("a", encoding="utf-8", buffering=1) as handle:
            process = subprocess.Popen(
                argv,
                cwd=ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_file = directory / f"{stem}.pid"
        pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
        records.append(
            {
                "shard": shard,
                "port": port,
                "egl_gpu": shard % 8,
                "pid": process.pid,
                "pid_file": str(pid_file.resolve()),
                "log": str(log.resolve()),
                "missing_before_launch": len(missing),
                "skipped_complete": False,
            }
        )
    time.sleep(10)
    dead_incomplete = []
    after = set(committed_rows())
    for row in records:
        if row.get("pid") is None or alive(int(row["pid"])):
            continue
        shard = int(row["shard"])
        still_missing = {
            key
            for key in expected
            if int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % SHARDS
            == shard
        } - after
        if still_missing:
            dead_incomplete.append({**row, "still_missing": len(still_missing)})
    if dead_incomplete:
        raise RuntimeError(f"resharded collectors exited immediately: {dead_incomplete}")
    return records


def main() -> None:
    before = set(committed_rows())
    expected = expected_keys()
    if not before <= expected:
        raise RuntimeError("pi0.5 calibration has out-of-manifest keys")
    owned, stale = owned_live_collectors()
    if before != expected and not owned:
        raise RuntimeError("pi0.5 calibration incomplete and no owned collector is live")
    stop_collectors(owned)
    records = launch_missing_shards()
    output = RUN / "unattended" / "pi05_collector_scale_out.json"
    atomic_json(
        output,
        {
            "schema_version": 1,
            "kind": "pi05_teacher_collector_reshard_v1",
            "scaled_at": now(),
            "completed_before": len(before),
            "expected": len(expected),
            "stopped_owned_collectors": owned,
            "ignored_stale_pid_files": stale,
            "new_shard_count": SHARDS,
            "collectors": records,
        },
    )
    print(json.dumps({"output": str(output), "collectors": records}, indent=2))


if __name__ == "__main__":
    main()
