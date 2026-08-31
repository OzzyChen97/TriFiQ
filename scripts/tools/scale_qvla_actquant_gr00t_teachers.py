#!/usr/bin/env python3
"""Safely add strict-FP16 GR00T teacher replicas and reconnect owned workers."""

from __future__ import annotations

import json
import hashlib
import os
import re
import signal
import socket
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
EXTRA_SERVERS = (
    ("gr00t_atomic_seen", "atomic_seen", 23006, 3, "scaled0"),
    ("gr00t_atomic_seen", "atomic_seen", 23007, 5, "scaled1"),
    ("gr00t_composite_seen", "composite_seen", 23008, 5, "scaled0"),
    ("gr00t_composite_seen", "composite_seen", 23009, 6, "scaled1"),
    ("gr00t_composite_unseen", "composite_unseen", 23010, 7, "scaled0"),
    ("gr00t_composite_unseen", "composite_unseen", 23011, 7, "scaled1"),
    ("gr00t_composite_unseen", "composite_unseen", 23012, 7, "scaled2"),
    ("gr00t_atomic_seen", "atomic_seen", 23013, 1, "scaled2"),
    ("gr00t_atomic_seen", "atomic_seen", 23014, 2, "scaled3"),
    ("gr00t_composite_seen", "composite_seen", 23015, 4, "scaled2"),
    ("gr00t_composite_unseen", "composite_unseen", 23016, 6, "scaled3"),
    ("gr00t_atomic_seen", "atomic_seen", 23017, 1, "released_pi0"),
    ("gr00t_atomic_seen", "atomic_seen", 23018, 2, "released_pi1"),
    ("gr00t_composite_seen", "composite_seen", 23019, 3, "released_pi0"),
    ("gr00t_composite_seen", "composite_seen", 23020, 4, "released_pi1"),
    ("gr00t_composite_seen", "composite_seen", 23021, 5, "released_pi2"),
    ("gr00t_composite_unseen", "composite_unseen", 23022, 6, "released_pi0"),
    ("gr00t_composite_unseen", "composite_unseen", 23023, 7, "released_pi1"),
)
PORTS = {
    "gr00t_atomic_seen": (23000, 23003, 23006, 23007, 23013, 23014, 23017, 23018),
    "gr00t_composite_seen": (23001, 23004, 23008, 23009, 23015, 23019, 23020, 23021),
    "gr00t_composite_unseen": (23002, 23005, 23010, 23011, 23012, 23016, 23022, 23023),
}


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


def committed_keys(unit: str) -> set[str]:
    rows: dict[str, dict[str, Any]] = {}
    directory = RUN / "calibration_fp16" / unit
    for journal in sorted(directory.glob("worker_*.jsonl")):
        for line_number, line in enumerate(
            journal.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["episode_key"])
            if key in rows and rows[key] != row:
                raise ValueError(
                    f"conflicting calibration key {key} at {journal}:{line_number}"
                )
            rows[key] = row
    return set(rows)


def shard_missing_keys(unit: str, shard: int) -> set[str]:
    manifest = json.loads(
        (RUN / "calibration_manifests" / f"{unit}.json").read_text(encoding="utf-8")
    )
    expected = {
        str(row["episode_key"])
        for row in manifest["qvla_episodes"]
        if int.from_bytes(
            hashlib.sha256(str(row["episode_key"]).encode()).digest()[:8], "big"
        )
        % 8
        == shard
    }
    return expected - committed_keys(unit)


def gpu_snapshot() -> dict[int, dict[str, int]]:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    result = {}
    for row in rows:
        index, used, free, utilization = (int(value.strip()) for value in row.split(","))
        result[index] = {"used_mib": used, "free_mib": free, "utilization_pct": utilization}
    return result


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def runtime_info(port: int) -> dict[str, Any]:
    context = zmq.Context()
    request = context.socket(zmq.REQ)
    request.setsockopt(zmq.RCVTIMEO, 5_000)
    request.setsockopt(zmq.SNDTIMEO, 5_000)
    request.connect(f"tcp://127.0.0.1:{port}")
    try:
        request.send(msgpack.packb({"endpoint": "get_runtime_info", "data": {}}))
        return msgpack.unpackb(request.recv(), raw=False)
    finally:
        request.close(linger=0)
        context.term()


def launch_extra_servers() -> list[dict[str, Any]]:
    initial = gpu_snapshot()
    planned = {gpu: 0 for gpu in initial}
    reusable: dict[int, tuple[int, Path, Path]] = {}
    # A warmed strict teacher occupies about 7.5 GiB.  Check the complete
    # placement before launching anything and retain at least 4 GiB per card.
    for unit, _split, port, gpu, replica in EXTRA_SERVERS:
        stem = f"{unit}_strict_fp16_{replica}"
        pid_file = RUN / "teacher_servers" / f"{stem}.pid"
        log = RUN / "teacher_servers" / f"{stem}.log"
        if port_free(port):
            planned[gpu] += 7_500
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as error:
            raise RuntimeError(f"refusing unowned occupied scale-out port {port}") from error
        cmd = command(pid)
        if (
            not alive(pid)
            or "scripts/inference_service.py" not in cmd
            or f"--port {port}" not in cmd
            or str(ROOT) not in cmd
        ):
            raise RuntimeError(f"refusing unowned occupied scale-out port {port}: {cmd}")
        reusable[port] = (pid, pid_file, log)
    for gpu, added in planned.items():
        if added and initial[gpu]["free_mib"] - added < 4_096:
            raise RuntimeError(
                f"GPU {gpu} lacks the 4 GiB reserve for scale-out: "
                f"free={initial[gpu]['free_mib']} planned={added}"
            )

    records = []
    server_dir = RUN / "teacher_servers"
    for unit, split, port, gpu, replica in EXTRA_SERVERS:
        if port in reusable:
            pid, pid_file, log = reusable[port]
            records.append(
                {
                    "unit": unit,
                    "port": port,
                    "gpu": gpu,
                    "replica": replica,
                    "pid": pid,
                    "pid_file": str(pid_file),
                    "log": str(log),
                    "reused": True,
                }
            )
            continue
        checkpoint = ROOT / (
            "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
            f"target_posttraining/{split}/checkpoint-60000"
        )
        argv = [
            GROOT_PY,
            str(ROOT / "scripts/inference_service.py"),
            "--server",
            "--model-path",
            str(checkpoint),
            "--data-config",
            "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig",
            "--embodiment-tag",
            "new_embodiment",
            "--port",
            str(port),
            "--denoising-steps",
            "4",
        ]
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(("GR00T_DUQUANT_", "GR00T_GPTQ", "GR00T_ATM_", "GR00T_OHB_", "QVLA_ACTQUANT_")):
                env.pop(key)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "GR00T_MODEL_DTYPE": "float16",
                "GR00T_CONFIG_ID": "fp16_teacher_proxy",
                "PYTHONUNBUFFERED": "1",
            }
        )
        stem = f"{unit}_strict_fp16_{replica}"
        log = server_dir / f"{stem}.log"
        with log.open("a", encoding="utf-8", buffering=1) as handle:
            process = subprocess.Popen(
                argv,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_file = server_dir / f"{stem}.pid"
        pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
        records.append(
            {
                "unit": unit,
                "port": port,
                "gpu": gpu,
                "replica": replica,
                "pid": process.pid,
                "pid_file": str(pid_file),
                "log": str(log),
                "reused": False,
            }
        )

    deadline = time.time() + 1_800
    pending = {row["port"]: row for row in records}
    errors: dict[int, str] = {}
    while pending and time.time() < deadline:
        for port, row in list(pending.items()):
            if not alive(int(row["pid"])):
                raise RuntimeError(f"scaled teacher exited during startup: {row}")
            try:
                metadata = runtime_info(port)
                precision = metadata.get("model_dtype") or {}
                dtypes = precision.get("linear_conv_layers_by_weight_dtype") or {}
                if precision.get("resolved") != "float16" or set(dtypes) != {"float16"}:
                    raise RuntimeError(f"precision mismatch: {precision}")
                row["server_metadata_sha256"] = str(metadata["metadata_sha256"])
                row["model_dtype"] = precision
                del pending[port]
            except Exception as error:
                errors[port] = f"{type(error).__name__}: {error}"
        if pending:
            time.sleep(5)
    if pending:
        raise RuntimeError(f"scaled teacher startup timeout: {sorted(pending)} {errors}")
    return records


def owned_control_processes() -> list[dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    controller_pid = RUN / "unattended" / "controller_strict_fp16.pid"
    pid_files = [controller_pid, *sorted((RUN / "collectors_fp16_strict").glob("gr00t_*.pid"))]
    for path in pid_files:
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if not alive(pid):
            continue
        cmd = command(pid)
        if path == controller_pid:
            valid = "qvla_actquant_unattended.py run" in cmd and str(ROOT) in cmd
            kind = "controller"
        else:
            valid = (
                "collect_qvla_actquant_teacher.py" in cmd
                and "--model gr00t" in cmd
                and str(RUN / "calibration_fp16") in cmd
            )
            kind = "collector"
        if not valid:
            raise RuntimeError(f"refusing to stop non-owned PID {pid}: {cmd}")
        result[pid] = {"pid": pid, "kind": kind, "pid_file": str(path), "command": cmd}
    expected_collectors = sum(row["kind"] == "collector" for row in result.values())
    if not 0 <= expected_collectors <= 24 or sum(
        row["kind"] == "controller" for row in result.values()
    ) != 1:
        raise RuntimeError(f"owned control inventory drift: {list(result.values())}")
    return sorted(result.values(), key=lambda row: (row["kind"], row["pid"]))


def stop_control_processes(records: list[dict[str, Any]]) -> None:
    for row in records:
        pid = int(row["pid"])
        if command(pid) != row["command"]:
            raise RuntimeError(f"PID identity changed before stop: {pid}")
        group = os.getpgid(pid)
        os.killpg(group, signal.SIGTERM) if group == pid else os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 120
    while time.time() < deadline and any(alive(int(row["pid"])) for row in records):
        time.sleep(1)
    remaining = [row["pid"] for row in records if alive(int(row["pid"]))]
    if remaining:
        raise RuntimeError(f"owned control processes did not stop after SIGTERM: {remaining}")


def launch_collectors() -> list[dict[str, Any]]:
    records = []
    directory = RUN / "collectors_fp16_strict"
    for unit, ports in PORTS.items():
        for shard in range(8):
            missing_before = shard_missing_keys(unit, shard)
            if not missing_before:
                records.append(
                    {
                        "unit": unit,
                        "shard": shard,
                        "port": None,
                        "egl_gpu": shard,
                        "pid": None,
                        "skipped_complete": True,
                    }
                )
                continue
            port = ports[shard % len(ports)]
            argv = [
                ROBOCASA_PY,
                str(ROOT / "scripts/tools/collect_qvla_actquant_teacher.py"),
                "--manifest",
                str(RUN / "calibration_manifests" / f"{unit}.json"),
                "--model",
                "gr00t",
                "--port",
                str(port),
                "--out-dir",
                str(RUN / "calibration_fp16" / unit),
                "--shard-index",
                str(shard),
                "--shard-count",
                "8",
                "--egl-device",
                str(shard),
            ]
            stem = f"{unit}_s{shard:02d}"
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
                    "unit": unit,
                    "shard": shard,
                    "port": port,
                    "egl_gpu": shard,
                    "pid": process.pid,
                    "pid_file": str(pid_file),
                    "log": str(log),
                    "missing_before_launch": len(missing_before),
                    "skipped_complete": False,
                }
            )
    time.sleep(10)
    dead = [
        row
        for row in records
        if row["pid"] is not None
        and not alive(int(row["pid"]))
        and shard_missing_keys(str(row["unit"]), int(row["shard"]))
    ]
    if dead:
        raise RuntimeError(f"scaled collectors exited immediately: {dead}")
    return records


def launch_controller() -> dict[str, Any]:
    argv = [
        GROOT_PY,
        str(ROOT / "scripts/tools/qvla_actquant_unattended.py"),
        "run",
        "--poll-seconds",
        "60",
        "--shards",
        "16",
    ]
    log = RUN / "unattended" / "controller_strict_fp16.log"
    with log.open("a", encoding="utf-8", buffering=1) as handle:
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file = RUN / "unattended" / "controller_strict_fp16.pid"
    pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    time.sleep(2)
    if not alive(process.pid):
        raise RuntimeError(f"scaled controller exited immediately; see {log}")
    return {"pid": process.pid, "pid_file": str(pid_file), "log": str(log)}


def main() -> None:
    before = gpu_snapshot()
    added_servers = launch_extra_servers()
    stopped = owned_control_processes()
    stop_control_processes(stopped)
    collectors = launch_collectors()
    controller = launch_controller()
    result = {
        "schema_version": 1,
        "kind": "qvla_actquant_teacher_scale_out_v1",
        "scaled_at": now(),
        "policy": "preserve all committed archives; stop only validated owned control PIDs; retain >=4 GiB planned GPU reserve",
        "gpu_before": before,
        "gpu_after": gpu_snapshot(),
        "added_servers": added_servers,
        "stopped_control_processes": stopped,
        "collectors": collectors,
        "controller": controller,
    }
    output = RUN / "unattended" / "gr00t_teacher_scale_out.json"
    atomic_json(output, result)
    print(json.dumps({"output": str(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
