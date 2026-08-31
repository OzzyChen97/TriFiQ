#!/usr/bin/env python3
"""Quarantine BF16 GR00T teacher data and restart only owned GR00T work in FP16."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import msgpack
import zmq


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "qvla_actquant_table1"
GROOT_PY = "/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
ROBOCASA_PY = "/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
SERVER_SPECS = (
    ("gr00t_atomic_seen", "atomic_seen", 23000, 1, "primary"),
    ("gr00t_atomic_seen", "atomic_seen", 23003, 4, "replica"),
    ("gr00t_composite_seen", "composite_seen", 23001, 2, "primary"),
    ("gr00t_composite_seen", "composite_seen", 23004, 6, "replica"),
    ("gr00t_composite_unseen", "composite_unseen", 23002, 3, "primary"),
    ("gr00t_composite_unseen", "composite_unseen", 23005, 5, "replica"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def process_command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return ""


def alive(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return len(fields) > 2 and fields[2] != "Z"
    except (FileNotFoundError, PermissionError):
        return False


def owned_targets() -> list[dict[str, Any]]:
    targets: dict[int, dict[str, Any]] = {}
    for proc in Path("/proc").glob("[0-9]*"):
        pid = int(proc.name)
        command = process_command(pid)
        if "qvla_actquant_unattended.py run" in command:
            cwd = Path(f"/proc/{pid}/cwd")
            if cwd.exists() and cwd.resolve() == ROOT:
                targets[pid] = {"pid": pid, "kind": "controller", "command": command}
    pid_files = list((RUN / "collectors").glob("gr00t_*.pid"))
    pid_files.extend((RUN / "collectors_fp16").glob("gr00t_*.pid"))
    pid_files.extend((RUN / "collectors_fp16_strict").glob("gr00t_*.pid"))
    for pid_file in sorted(pid_files):
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        command = process_command(pid)
        if alive(pid):
            if (
                "collect_qvla_actquant_teacher.py" not in command
                or "--model gr00t" not in command
                or str(RUN) not in str((ROOT / command.split("--out-dir ", 1)[-1].split()[0]).resolve())
            ):
                raise RuntimeError(f"refusing non-owned collector PID {pid}: {command}")
            targets[pid] = {
                "pid": pid, "kind": "gr00t_nonprotocol_collector", "command": command,
                "pid_file": str(pid_file.resolve()),
            }
    for pid_file in sorted((RUN / "teacher_servers").glob("gr00t*.pid")):
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        command = process_command(pid)
        if alive(pid):
            match = re.search(r"(?:^| )--port (\d+)(?: |$)", command)
            if (
                "scripts/inference_service.py" not in command
                or not match
                or int(match.group(1)) not in range(23000, 23006)
                or str(ROOT) not in command
            ):
                raise RuntimeError(f"refusing non-owned teacher PID {pid}: {command}")
            targets[pid] = {
                "pid": pid, "kind": "gr00t_nonprotocol_teacher", "command": command,
                "pid_file": str(pid_file.resolve()),
            }
    return sorted(targets.values(), key=lambda row: (row["kind"], row["pid"]))


def calibration_snapshot() -> dict[str, Any]:
    units = {}
    for unit in ("gr00t_atomic_seen", "gr00t_composite_seen", "gr00t_composite_unseen"):
        directory = RUN / "calibration" / unit
        journals = []
        keys = set()
        for path in sorted(directory.glob("worker_*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    keys.add(json.loads(line)["episode_key"])
            journals.append({
                "path": str(path.resolve()), "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        archives = list((directory / "episodes").glob("**/*.npz"))
        units[unit] = {
            "committed_episode_keys": len(keys),
            "journal_files": journals,
            "archive_files": len(archives),
            "archive_bytes": sum(path.stat().st_size for path in archives),
        }
    return units


def stop_targets(targets: list[dict[str, Any]]) -> None:
    order = {"controller": 0, "gr00t_nonprotocol_collector": 1, "gr00t_nonprotocol_teacher": 2}
    for row in sorted(targets, key=lambda value: (order[value["kind"]], value["pid"])):
        pid = int(row["pid"])
        if not alive(pid):
            continue
        if process_command(pid) != row["command"]:
            raise RuntimeError(f"PID command changed before stop: {pid}")
        group = os.getpgid(pid)
        if group == pid:
            os.killpg(group, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 120
    while time.time() < deadline and any(alive(int(row["pid"])) for row in targets):
        time.sleep(1)
    remaining = [row["pid"] for row in targets if alive(int(row["pid"]))]
    if remaining:
        raise RuntimeError(f"owned GR00T processes did not stop after SIGTERM: {remaining}")


def runtime_info(port: int) -> dict[str, Any]:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 5_000)
    socket.setsockopt(zmq.SNDTIMEO, 5_000)
    socket.connect(f"tcp://127.0.0.1:{port}")
    try:
        socket.send(msgpack.packb({"endpoint": "get_runtime_info", "data": {}}))
        return msgpack.unpackb(socket.recv(), raw=False)
    finally:
        socket.close(linger=0)
        context.term()


def launch_servers() -> list[dict[str, Any]]:
    records = []
    server_dir = RUN / "teacher_servers"
    server_dir.mkdir(parents=True, exist_ok=True)
    for unit, split, port, gpu, replica in SERVER_SPECS:
        checkpoint = ROOT / (
            "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
            f"target_posttraining/{split}/checkpoint-60000"
        )
        command = [
            GROOT_PY, str(ROOT / "scripts/inference_service.py"), "--server",
            "--model-path", str(checkpoint),
            "--data-config", "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig",
            "--embodiment-tag", "new_embodiment", "--port", str(port),
            "--denoising-steps", "4",
        ]
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(("GR00T_DUQUANT_", "GR00T_GPTQ", "GR00T_ATM_", "GR00T_OHB_")):
                env.pop(key)
        for key in ("QVLA_ACTQUANT_METHOD", "QVLA_ACTQUANT_PACK", "QVLA_ACTQUANT_PACK_SHA256"):
            env.pop(key, None)
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "GR00T_MODEL_DTYPE": "float16",
            "GR00T_CONFIG_ID": "fp16_teacher_proxy",
            "PYTHONUNBUFFERED": "1",
        })
        stem = f"{unit}_strict_fp16_{replica}"
        log = server_dir / f"{stem}.log"
        with log.open("a", encoding="utf-8", buffering=1) as handle:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_file = server_dir / f"{stem}.pid"
        pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
        records.append({
            "unit": unit, "port": port, "gpu": gpu, "replica": replica,
            "pid": process.pid, "pid_file": str(pid_file), "log": str(log),
            "command": command,
        })
    deadline = time.time() + 1800
    pending = {row["port"]: row for row in records}
    errors = {}
    while pending and time.time() < deadline:
        for port, row in list(pending.items()):
            if not alive(int(row["pid"])):
                raise RuntimeError(f"FP16 teacher exited during startup: {row}")
            try:
                metadata = runtime_info(port)
                precision = metadata.get("model_dtype") or {}
                linear_dtypes = precision.get("linear_conv_layers_by_weight_dtype") or {}
                if precision.get("resolved") != "float16" or set(linear_dtypes) != {"float16"}:
                    raise RuntimeError(f"precision mismatch: {precision}")
                row["server_metadata_sha256"] = str(metadata["metadata_sha256"])
                row["model_dtype"] = precision
                del pending[port]
            except Exception as error:  # server loading or semantic validation
                errors[port] = f"{type(error).__name__}: {error}"
        if pending:
            time.sleep(5)
    if pending:
        raise RuntimeError(f"FP16 teacher startup timeout: {sorted(pending)} {errors}")
    return records


def launch_collectors() -> list[dict[str, Any]]:
    records = []
    directory = RUN / "collectors_fp16_strict"
    directory.mkdir(parents=True, exist_ok=True)
    ports = {
        "gr00t_atomic_seen": (23000, 23003),
        "gr00t_composite_seen": (23001, 23004),
        "gr00t_composite_unseen": (23002, 23005),
    }
    for unit, unit_ports in ports.items():
        for shard in range(8):
            port = unit_ports[shard % len(unit_ports)]
            command = [
                ROBOCASA_PY, str(ROOT / "scripts/tools/collect_qvla_actquant_teacher.py"),
                "--manifest", str(RUN / "calibration_manifests" / f"{unit}.json"),
                "--model", "gr00t", "--port", str(port),
                "--out-dir", str(RUN / "calibration_fp16" / unit),
                "--shard-index", str(shard), "--shard-count", "8",
                "--egl-device", str(shard),
            ]
            stem = f"{unit}_s{shard:02d}"
            log = directory / f"{stem}.log"
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            with log.open("a", encoding="utf-8", buffering=1) as handle:
                process = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            pid_file = directory / f"{stem}.pid"
            pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
            records.append({
                "unit": unit, "shard": shard, "port": port, "egl_gpu": shard,
                "pid": process.pid, "pid_file": str(pid_file), "log": str(log),
            })
    time.sleep(10)
    dead = [row for row in records if not alive(int(row["pid"]))]
    if dead:
        raise RuntimeError(f"FP16 collectors exited immediately: {dead}")
    return records


def launch_controller() -> dict[str, Any]:
    log = RUN / "unattended" / "controller_strict_fp16.log"
    command = [
        GROOT_PY, str(ROOT / "scripts/tools/qvla_actquant_unattended.py"), "run",
        "--poll-seconds", "60", "--shards", "16",
    ]
    with log.open("a", encoding="utf-8", buffering=1) as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
    pid_file = RUN / "unattended" / "controller_strict_fp16.pid"
    pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    time.sleep(2)
    if not alive(process.pid):
        raise RuntimeError(f"replacement controller exited immediately; see {log}")
    return {"pid": process.pid, "pid_file": str(pid_file), "log": str(log), "command": command}


def main() -> None:
    targets = owned_targets()
    quarantine = {
        "schema_version": 1,
        "kind": "qvla_actquant_nonprotocol_calibration_quarantine_v1",
        "reason": (
            "GR00T teacher services contained BF16 Linear/Conv weights; protocol "
            "requires strict all-Linear/Conv fp16_teacher_proxy"
        ),
        "recoverable": True,
        "quarantined_at": now(),
        "calibration": calibration_snapshot(),
        "stopped_processes": targets,
        "precision_attestation": str((RUN / "teacher_precision_attestations.json").resolve()),
        "precision_attestation_sha256": sha256_file(RUN / "teacher_precision_attestations.json"),
    }
    quarantine_path = RUN / "quarantine" / f"gr00t_nonprotocol_teacher_{int(time.time())}.json"
    atomic_json(quarantine_path, quarantine)
    stop_targets(targets)
    servers = launch_servers()
    subprocess.run(
        [ROBOCASA_PY, str(ROOT / "scripts/tools/record_qvla_actquant_teacher_precision.py")],
        cwd=ROOT, check=True,
    )
    collectors = launch_collectors()
    controller = launch_controller()
    result = {
        "restarted_at": now(), "quarantine": str(quarantine_path),
        "servers": servers, "collectors": collectors, "controller": controller,
    }
    atomic_json(RUN / "unattended" / "gr00t_fp16_restart.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
