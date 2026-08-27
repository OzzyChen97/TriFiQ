#!/usr/bin/env python3
"""Adopt an orphaned ErrorFold-v3 GR00T matrix, then resume the runner.

The matrix deliberately starts each server and client driver in a new process
session.  If an outer interactive launcher disappears, those workers continue
and must be allowed to finish before a resumable runner can bind the same
ports.  This supervisor watches only workers whose output lives below the
requested run root, stops only the v3 GR00T ports 22100--22107 after all such
drivers exit, and then launches the ordinary hash-aware runner.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time


V3_GR00T_PORTS = {str(port) for port in range(22100, 22108)}


def process_args(pid: int) -> list[str]:
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [part.decode(errors="replace") for part in data.split(b"\0") if part]


def all_processes() -> list[tuple[int, list[str]]]:
    result = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and int(entry.name) != os.getpid():
            args = process_args(int(entry.name))
            if args:
                result.append((int(entry.name), args))
    return result


def matching_drivers(run_root: Path) -> list[int]:
    output_root = str((run_root / "closed_loop/gr00t").resolve())
    return [
        pid
        for pid, args in all_processes()
        if any(arg.endswith("/run_crit4_trial_driver.py") for arg in args)
        and any(arg.startswith(output_root + "/") for arg in args)
    ]


def matching_servers() -> list[int]:
    result = []
    for pid, args in all_processes():
        if not any(arg.endswith("/inference_service.py") or arg == "scripts/inference_service.py" for arg in args):
            continue
        try:
            port = args[args.index("--port") + 1]
            model_path = args[args.index("--model-path") + 1]
        except (ValueError, IndexError):
            continue
        if (
            port in V3_GR00T_PORTS
            and "/gr00t_n1-5/foundation_model_learning/target_posttraining/" in model_path
        ):
            result.append(pid)
    return result


def stop_servers(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        alive = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
        if not alive:
            return
        time.sleep(1.0)
    for pid in pids:
        if not Path(f"/proc/{pid}").exists():
            continue
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    runner = Path(args.runner).expanduser().resolve()

    while True:
        drivers = matching_drivers(run_root)
        if not drivers:
            break
        print(
            f"[errorfold-v3-recover] waiting for {len(drivers)} orphan drivers",
            flush=True,
        )
        time.sleep(max(5.0, args.poll_seconds))

    servers = matching_servers()
    print(
        f"[errorfold-v3-recover] current matrix drained; stopping {len(servers)} v3 servers",
        flush=True,
    )
    stop_servers(servers)

    while True:
        print("[errorfold-v3-recover] launching resumable runner", flush=True)
        completed = subprocess.run(
            ["bash", str(runner), "run"],
            cwd=runner.parents[1],
            check=False,
        )
        if completed.returncode == 0:
            return
        print(
            f"[errorfold-v3-recover] runner exit={completed.returncode}; auditing orphans before retry",
            flush=True,
        )
        while matching_drivers(run_root):
            time.sleep(max(5.0, args.poll_seconds))
        stop_servers(matching_servers())
        time.sleep(20.0)


if __name__ == "__main__":
    main()
