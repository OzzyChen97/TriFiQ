#!/usr/bin/env python3
"""Durably resume ActQuant or DA-PTQ formal evaluation on physical GPUs 3--6."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
GPU_ALLOWLIST = "3,4,5,6"
CONFIGS = {
    "actquant": {
        "formal": ROOT / "runs" / "qvla_actquant_table1" / "formal",
        # DA-PTQ is complete, so ActQuant can use the released host-memory
        # budget. Rollout workers are mainly CPU/RAM consumers; eight per GPU
        # keeps more requests queued behind the resident model services.
        "command": [
            PYTHON,
            "scripts/tools/qvla_actquant_formal_pipeline.py",
            "run",
            "--max-workers",
            "32",
            "--workers-per-gpu",
            "8",
            "--replicas",
            "1",
            "--eligible-gpus",
            GPU_ALLOWLIST,
            "--methods",
            "actquant",
        ],
    },
    "daptq": {
        "formal": ROOT / "runs" / "daptq_table1" / "formal",
        # DA-PTQ keeps two server replicas per model unit resident.  Two
        # rollout workers per GPU leave enough host RAM for both pipelines.
        "command": [
            PYTHON,
            "scripts/tools/daptq_formal_pipeline.py",
            "run",
            "--skip-smoke50",
            "--max-workers",
            "8",
            "--workers-per-gpu",
            "2",
            "--replicas",
            "2",
            "--poll-seconds",
            "30",
            "--eligible-gpus",
            GPU_ALLOWLIST,
        ],
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_event(control: Path, value: dict[str, Any]) -> None:
    path = control / "supervisor_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": now(), **value}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def process_command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
    except OSError:
        return ""


def registered_worker_matches(
    row: dict[str, Any], command: str, process_group: int, formal: Path
) -> bool:
    """Fail closed before signaling a PID recovered from a stale registry."""
    try:
        output = Path(str(row["output"])).resolve()
        expected_group = int(row["process_group"])
        pid = int(row["pid"])
    except (KeyError, TypeError, ValueError, OSError):
        return False
    run_root = formal.parent.resolve()
    if not output.is_relative_to(run_root):
        return False
    if expected_group != pid or process_group != pid:
        return False
    return (
        str(output) in command
        and str(ROOT) in command
        and (
            "scripts/run_robocasa365_gr00t_eval.py" in command
            or "scripts/run_robocasa365_pi05_eval.py" in command
        )
    )


def legacy_worker_matches(pid: int, command: str, process_group: int, formal: Path) -> bool:
    """Recognize pre-registry orphan workers by their exact formal output root."""
    formal_prefix = f"{formal.resolve()}/"
    return (
        process_group == pid
        and formal_prefix in command
        and str(ROOT) in command
        and any(f"--egl-device {gpu}" in command for gpu in (3, 4, 5, 6))
        and (
            "scripts/run_robocasa365_gr00t_eval.py" in command
            or "scripts/run_robocasa365_pi05_eval.py" in command
        )
    )


def pipeline_lock_is_free(formal: Path) -> bool:
    path = formal / "control" / "pipeline.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return True


def cleanup_legacy_workers(formal: Path, control: Path) -> list[int]:
    """Clean workers orphaned before the persistent registry was introduced."""
    if not pipeline_lock_is_free(formal):
        raise RuntimeError("refusing legacy cleanup while the formal pipeline lock is held")
    matched: dict[int, str] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            pid = int(proc.name)
            process_group = os.getpgid(pid)
        except (OSError, ProcessLookupError):
            continue
        command = process_command(pid)
        if legacy_worker_matches(pid, command, process_group, formal):
            matched[pid] = command
            try:
                os.killpg(process_group, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 30
    remaining = set(matched)
    while remaining and time.time() < deadline:
        remaining = {pid for pid in remaining if Path(f"/proc/{pid}").exists()}
        if remaining:
            time.sleep(0.5)
    force_killed = []
    refused = []
    for pid in sorted(remaining):
        command = process_command(pid)
        try:
            process_group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if not legacy_worker_matches(pid, command, process_group, formal):
            refused.append({"pid": pid, "command": command})
            continue
        try:
            os.killpg(process_group, signal.SIGKILL)
            force_killed.append(pid)
        except ProcessLookupError:
            pass
    if matched:
        append_event(
            control,
            {
                "event": "legacy_worker_cleanup",
                "terminated_pids": sorted(matched),
                "force_killed_pids": force_killed,
                "refused": refused,
            },
        )
    return sorted(matched)


def cleanup_registered_workers(formal: Path, control: Path, *, reason: str) -> list[int]:
    """Reap only exact worker identities recorded by the previous child scheduler."""
    matched: list[int] = []
    refused: list[dict[str, Any]] = []
    for registry in sorted(control.glob("*_running_workers.json")):
        try:
            rows = json.loads(registry.read_text(encoding="utf-8")).get("workers") or []
        except (OSError, ValueError, TypeError):
            continue
        for row in rows:
            try:
                pid = int(row["pid"])
                process_group = os.getpgid(pid)
            except (KeyError, TypeError, ValueError, ProcessLookupError):
                continue
            command = process_command(pid)
            if not registered_worker_matches(row, command, process_group, formal):
                refused.append({"pid": pid, "registry": str(registry), "command": command})
                continue
            try:
                os.killpg(process_group, signal.SIGTERM)
                matched.append(pid)
            except ProcessLookupError:
                pass
    deadline = time.time() + 30
    remaining = set(matched)
    while remaining and time.time() < deadline:
        remaining = {pid for pid in remaining if Path(f"/proc/{pid}").exists()}
        if remaining:
            time.sleep(0.5)
    force_killed = []
    for pid in sorted(remaining):
        command = process_command(pid)
        try:
            process_group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        # The original validated identity may have exited and the PID may have
        # been reused during the grace period.  Never SIGKILL unless the exact
        # command is still one of this run's evaluator workers.
        if not (
            str(formal.parent.resolve()) in command
            and str(ROOT) in command
            and process_group == pid
            and (
                "scripts/run_robocasa365_gr00t_eval.py" in command
                or "scripts/run_robocasa365_pi05_eval.py" in command
            )
        ):
            refused.append({"pid": pid, "phase": "force", "command": command})
            continue
        try:
            os.killpg(process_group, signal.SIGKILL)
            force_killed.append(pid)
        except ProcessLookupError:
            pass
    if matched or refused:
        append_event(
            control,
            {
                "event": "registered_worker_cleanup",
                "reason": reason,
                "terminated_pids": sorted(matched),
                "force_killed_pids": force_killed,
                "refused": refused,
            },
        )
    return sorted(matched)


def pipeline_command(name: str) -> list[str]:
    return list(CONFIGS[name]["command"])


def supervise(name: str, restart_delay: int) -> None:
    config = CONFIGS[name]
    formal = Path(config["formal"])
    control = formal / "control"
    control.mkdir(parents=True, exist_ok=True)
    lock_path = control / "supervisor.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"{name} supervisor is already running") from error

        child: subprocess.Popen[str] | None = None
        stopping = False

        def request_stop(signum: int, _frame: Any) -> None:
            nonlocal stopping
            stopping = True
            append_event(control, {"event": "supervisor_signal", "signal": signum})
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
        previous_sigint = signal.signal(signal.SIGINT, request_stop)
        attempt = 0
        try:
            cleanup_legacy_workers(formal, control)
            cleanup_registered_workers(formal, control, reason="supervisor_start")
            while not stopping:
                complete = formal / "complete.json"
                if complete.is_file():
                    atomic_json(
                        control / "supervisor_status.json",
                        {
                            "updated_at": now(),
                            "pipeline": name,
                            "state": "complete",
                            "complete": str(complete),
                        },
                    )
                    append_event(control, {"event": "supervisor_complete", "pipeline": name})
                    return
                attempt += 1
                command = pipeline_command(name)
                log_path = control / f"supervisor.pipeline.attempt{attempt:03d}.log"
                with log_path.open("a", encoding="utf-8", buffering=1) as handle:
                    env = dict(os.environ)
                    env["PYTHONUNBUFFERED"] = "1"
                    child = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        text=True,
                    )
                    atomic_json(
                        control / "supervisor_status.json",
                        {
                            "updated_at": now(),
                            "pipeline": name,
                            "state": "running",
                            "supervisor_pid": os.getpid(),
                            "child_pid": child.pid,
                            "attempt": attempt,
                            "command": command,
                            "gpu_allowlist": [3, 4, 5, 6],
                            "log": str(log_path),
                        },
                    )
                    append_event(
                        control,
                        {
                            "event": "pipeline_started",
                            "pipeline": name,
                            "attempt": attempt,
                            "pid": child.pid,
                            "log": str(log_path),
                        },
                    )
                    while child.poll() is None and not stopping:
                        time.sleep(5)
                    if stopping and child.poll() is None:
                        request_stop(signal.SIGTERM, None)
                        try:
                            child.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            if child.poll() is None:
                                os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                    return_code = child.wait()
                child = None
                cleanup_registered_workers(
                    formal, control, reason=f"pipeline_exit_{return_code}"
                )
                if stopping:
                    break
                complete_exists = complete.is_file()
                append_event(
                    control,
                    {
                        "event": "pipeline_exited",
                        "pipeline": name,
                        "attempt": attempt,
                        "return_code": return_code,
                        "complete_exists": complete_exists,
                        "restart_delay_seconds": 0 if complete_exists else restart_delay,
                    },
                )
                if complete_exists:
                    continue
                atomic_json(
                    control / "supervisor_status.json",
                    {
                        "updated_at": now(),
                        "pipeline": name,
                        "state": "restart_wait",
                        "supervisor_pid": os.getpid(),
                        "attempt": attempt,
                        "last_return_code": return_code,
                        "restart_delay_seconds": restart_delay,
                    },
                )
                deadline = time.time() + restart_delay
                while not stopping and time.time() < deadline:
                    time.sleep(min(5, max(0.1, deadline - time.time())))
        finally:
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=60)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            cleanup_registered_workers(formal, control, reason="supervisor_exit")
            complete = formal / "complete.json"
            atomic_json(
                control / "supervisor_status.json",
                {
                    "updated_at": now(),
                    "pipeline": name,
                    "state": "complete" if complete.is_file() else "stopped",
                    "supervisor_pid": os.getpid(),
                    "complete": str(complete) if complete.is_file() else None,
                },
            )
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGINT, previous_sigint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", choices=sorted(CONFIGS))
    parser.add_argument("--restart-delay", type=int, default=60)
    args = parser.parse_args()
    if args.restart_delay < 1:
        raise ValueError("restart delay must be positive")
    supervise(args.pipeline, args.restart_delay)


if __name__ == "__main__":
    main()
