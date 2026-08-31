#!/usr/bin/env python3
"""Recover dead calibration shards without touching unrelated processes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "qvla_actquant_table1"
PYTHON = "/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
GR_UNITS = (
    "gr00t_atomic_seen",
    "gr00t_composite_seen",
    "gr00t_composite_unseen",
)


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
    except (FileNotFoundError, PermissionError, OSError):
        return False


def command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return ""


def manifest_keys(unit: str) -> set[str]:
    manifest = json.loads(
        (RUN / "calibration_manifests" / f"{unit}.json").read_text(encoding="utf-8")
    )
    return {str(row["episode_key"]) for row in manifest["qvla_episodes"]}


def committed_keys(unit: str) -> set[str]:
    root = "calibration" if unit == "pi05_all_target" else "calibration_fp16"
    rows: dict[str, dict[str, Any]] = {}
    for journal in sorted((RUN / root / unit).glob("worker_*.jsonl")):
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
    expected = manifest_keys(unit)
    extra = set(rows) - expected
    if extra:
        raise ValueError(f"{unit}: {len(extra)} out-of-manifest calibration keys")
    return set(rows)


def shard_keys(keys: set[str], shard: int, count: int) -> set[str]:
    return {
        key
        for key in keys
        if int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % count
        == shard
    }


def pid_from(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def gr00t_rebalance_record() -> dict[str, Any]:
    path = RUN / "unattended" / "gr00t_composite_rebalance.json"
    if not path.is_file():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("stage") != "complete":
        return {}
    count = int(record.get("shard_count", 0))
    if count < 1:
        raise ValueError(f"invalid GR00T rebalance shard count: {count}")
    return record


def pi05_unhealthy_shards() -> list[dict[str, Any]]:
    record_path = RUN / "unattended" / "pi05_collector_scale_out.json"
    if not record_path.is_file():
        return []
    record = json.loads(record_path.read_text(encoding="utf-8"))
    count = int(record["new_shard_count"])
    expected, completed = manifest_keys("pi05_all_target"), committed_keys("pi05_all_target")
    unhealthy = []
    for row in record["collectors"]:
        shard = int(row["shard"])
        missing = shard_keys(expected, shard, count) - completed
        if not missing:
            continue
        pid = row.get("pid")
        cmd = command(int(pid)) if pid is not None and alive(int(pid)) else ""
        if (
            not cmd
            or "collect_qvla_actquant_teacher.py" not in cmd
            or "--model pi05" not in cmd
            or f"--shard-index {shard}" not in cmd
            or f"--shard-count {count}" not in cmd
        ):
            unhealthy.append(
                {"shard": shard, "pid": pid, "missing": len(missing), "command": cmd}
            )
    return unhealthy


def gr00t_unhealthy_shards() -> list[dict[str, Any]]:
    rebalance = gr00t_rebalance_record()
    if rebalance:
        count = int(rebalance["shard_count"])
        collectors = {
            (str(row["unit"]), int(row["shard"])): row
            for row in rebalance["collectors"]
        }
        unhealthy = []
        for unit in GR_UNITS:
            expected, completed = manifest_keys(unit), committed_keys(unit)
            if completed == expected:
                continue
            for shard in range(count):
                missing = shard_keys(expected, shard, count) - completed
                if not missing:
                    continue
                row = collectors.get((unit, shard), {})
                pid = row.get("pid")
                cmd = command(int(pid)) if pid is not None and alive(int(pid)) else ""
                if (
                    not cmd
                    or "collect_qvla_actquant_teacher.py" not in cmd
                    or "--model gr00t" not in cmd
                    or f"--shard-index {shard}" not in cmd
                    or f"--shard-count {count}" not in cmd
                    or str(RUN / "calibration_fp16") not in cmd
                ):
                    unhealthy.append(
                        {
                            "unit": unit,
                            "shard": shard,
                            "pid": pid,
                            "missing": len(missing),
                            "command": cmd,
                            "layout": f"rebalanced_{count}",
                        }
                    )
        return unhealthy

    unhealthy = []
    for unit in GR_UNITS:
        expected, completed = manifest_keys(unit), committed_keys(unit)
        for shard in range(8):
            missing = shard_keys(expected, shard, 8) - completed
            if not missing:
                continue
            pid_file = RUN / "collectors_fp16_strict" / f"{unit}_s{shard:02d}.pid"
            pid = pid_from(pid_file)
            cmd = command(pid) if pid is not None and alive(pid) else ""
            if (
                not cmd
                or "collect_qvla_actquant_teacher.py" not in cmd
                or "--model gr00t" not in cmd
                or f"--shard-index {shard}" not in cmd
                or "--shard-count 8" not in cmd
                or str(RUN / "calibration_fp16") not in cmd
            ):
                unhealthy.append(
                    {
                        "unit": unit,
                        "shard": shard,
                        "pid": pid,
                        "missing": len(missing),
                        "command": cmd,
                    }
                )
    return unhealthy


def run_recovery(script: str, log_name: str) -> None:
    log = RUN / "unattended" / log_name
    with log.open("a", encoding="utf-8", buffering=1) as handle:
        completed = subprocess.run(
            [PYTHON, str(ROOT / "scripts" / "tools" / script)],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"calibration recovery failed; see {log}")


def all_complete() -> bool:
    return all(
        committed_keys(unit) == manifest_keys(unit)
        for unit in (*GR_UNITS, "pi05_all_target")
    )


def main() -> None:
    lock_path = RUN / "unattended" / "calibration_watchdog.lock"
    status_path = RUN / "unattended" / "calibration_watchdog_status.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recoveries = 0
        while not all_complete():
            pi_expected = manifest_keys("pi05_all_target")
            pi_completed = committed_keys("pi05_all_target")
            rebalance_path = RUN / "unattended" / "pi05_rebalance_status.json"
            rebalance = (
                json.loads(rebalance_path.read_text(encoding="utf-8"))
                if rebalance_path.is_file()
                else {}
            )
            pi_unhealthy = []
            gr_unhealthy = []
            if pi_completed != pi_expected:
                pi_unhealthy = pi05_unhealthy_shards()
                if pi_unhealthy:
                    run_recovery(
                        "scale_qvla_actquant_pi05_collectors.py",
                        "watchdog_recover_pi05.log",
                    )
                    recoveries += 1
            elif rebalance.get("stage") == "complete":
                gr_unhealthy = gr00t_unhealthy_shards()
                if gr_unhealthy:
                    recovery_script = (
                        "rebalance_qvla_actquant_gr00t_composites.py"
                        if gr00t_rebalance_record()
                        else "scale_qvla_actquant_gr00t_teachers.py"
                    )
                    run_recovery(
                        recovery_script,
                        "watchdog_recover_gr00t.log",
                    )
                    recoveries += 1
            atomic_json(
                status_path,
                {
                    "stage": "monitoring",
                    "updated_at": now(),
                    "recoveries": recoveries,
                    "pi05_completed": len(pi_completed),
                    "pi05_expected": len(pi_expected),
                    "pi05_unhealthy_shards": pi_unhealthy,
                    "gr00t_unhealthy_shards": gr_unhealthy,
                    "pi05_rebalance_stage": rebalance.get("stage"),
                },
            )
            time.sleep(60)
        atomic_json(
            status_path,
            {
                "stage": "complete",
                "updated_at": now(),
                "recoveries": recoveries,
            },
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic_json(
            RUN / "unattended" / "calibration_watchdog_status.json",
            {"stage": "failed", "updated_at": now(), "error": repr(error)},
        )
        raise
