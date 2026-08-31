#!/usr/bin/env python3
"""Release completed pi0.5 teachers and scale GR00T into their freed VRAM."""

from __future__ import annotations

import fcntl
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
PYTHON = "/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"


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


def pi05_progress() -> tuple[int, int, int]:
    manifest = json.loads(
        (RUN / "calibration_manifests" / "pi05_all_target.json").read_text(encoding="utf-8")
    )
    expected = {row["episode_key"] for row in manifest["qvla_episodes"]}
    rows: dict[str, dict[str, Any]] = {}
    directory = RUN / "calibration" / "pi05_all_target"
    for journal in sorted(directory.glob("worker_*.jsonl")):
        for line_number, line in enumerate(journal.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["episode_key"])
            if key in rows and rows[key] != row:
                raise ValueError(f"conflicting pi0.5 row {key} at {journal}:{line_number}")
            rows[key] = row
    return len(rows), len(expected), len(set(rows) - expected)


def owned_pi05_servers() -> list[dict[str, Any]]:
    records = []
    for pid_file in sorted((RUN / "teacher_servers").glob("*pi05*.pid")):
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if not alive(pid):
            continue
        cmd = command(pid)
        if (
            "serve_pi05_quant_policy.py" not in cmd
            or str(ROOT) not in cmd
            or not any(f"--port {port}" in cmd for port in range(23100, 23107))
        ):
            raise RuntimeError(f"refusing non-owned pi0.5 PID {pid}: {cmd}")
        records.append(
            {"pid": pid, "pid_file": str(pid_file.resolve()), "command": cmd}
        )
    if len(records) != 7:
        raise RuntimeError(f"expected seven live owned pi0.5 teachers, found {len(records)}")
    return records


def stop_servers(records: list[dict[str, Any]]) -> None:
    for row in records:
        pid = int(row["pid"])
        if command(pid) != row["command"]:
            raise RuntimeError(f"pi0.5 PID identity changed before stop: {pid}")
        group = os.getpgid(pid)
        os.killpg(group, signal.SIGTERM) if group == pid else os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 120
    while time.time() < deadline and any(alive(int(row["pid"])) for row in records):
        time.sleep(1)
    remaining = [row["pid"] for row in records if alive(int(row["pid"]))]
    if remaining:
        raise RuntimeError(f"owned pi0.5 teachers did not stop after SIGTERM: {remaining}")


def main() -> None:
    lock_path = RUN / "unattended" / "pi05_rebalance.lock"
    status_path = RUN / "unattended" / "pi05_rebalance_status.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            completed, expected, extra = pi05_progress()
            atomic_json(
                status_path,
                {
                    "stage": "waiting_for_pi05_calibration",
                    "updated_at": now(),
                    "completed": completed,
                    "expected": expected,
                    "extra": extra,
                },
            )
            if completed == expected and extra == 0:
                break
            time.sleep(60)
        # Give collectors one polling interval to journal, close environments,
        # and release their EGL contexts before measuring placement capacity.
        time.sleep(60)
        records = owned_pi05_servers()
        stop_servers(records)
        atomic_json(
            status_path,
            {
                "stage": "pi05_teachers_stopped",
                "updated_at": now(),
                "completed": completed,
                "expected": expected,
                "stopped_servers": records,
            },
        )
        log_path = RUN / "unattended" / "rebalance_scale_gr00t.log"
        with log_path.open("a", encoding="utf-8", buffering=1) as handle:
            result = subprocess.run(
                [PYTHON, str(ROOT / "scripts/tools/scale_qvla_actquant_gr00t_teachers.py")],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"GR00T post-pi0.5 scale-out failed; see {log_path}")
        atomic_json(
            status_path,
            {
                "stage": "complete",
                "updated_at": now(),
                "completed": completed,
                "expected": expected,
                "stopped_servers": records,
                "scale_log": str(log_path),
            },
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic_json(
            RUN / "unattended" / "pi05_rebalance_status.json",
            {"stage": "failed", "updated_at": now(), "error": repr(error)},
        )
        raise
