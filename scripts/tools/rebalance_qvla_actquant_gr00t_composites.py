#!/usr/bin/env python3
"""Recycle completed atomic GR00T teachers into composite calibration capacity.

The transition is deliberately conservative:

* only repository-owned PIDs whose complete command lines match are signalled;
* committed archives and legacy journals are never modified;
* replacement teachers are FP16-attested before any active collector is stopped;
* the two remaining units are deterministically repartitioned from 8 to 12
  workers, and the watchdog can invoke this file without arguments to recover
  a failed 12-way collector.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
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
STATUS = RUN / "unattended" / "gr00t_composite_rebalance.json"
ATOMIC_PORTS = (23000, 23003, 23006, 23007, 23013, 23014, 23017, 23018)
EXISTING_PORTS = {
    "gr00t_composite_seen": (23001, 23004, 23008, 23009, 23015, 23019, 23020, 23021),
    "gr00t_composite_unseen": (23002, 23005, 23010, 23011, 23012, 23016, 23022, 23023),
}
# Balance checkpoint types on every GPU while keeping the replacement count 4+4.
REPLACEMENTS = {
    23000: "gr00t_composite_seen",
    23013: "gr00t_composite_unseen",
    23017: "gr00t_composite_seen",
    23014: "gr00t_composite_unseen",
    23018: "gr00t_composite_unseen",
    23006: "gr00t_composite_seen",
    23003: "gr00t_composite_unseen",
    23007: "gr00t_composite_seen",
}
SPLIT = {
    "gr00t_composite_seen": "composite_seen",
    "gr00t_composite_unseen": "composite_unseen",
}
LAYOUT_COUNT = 12


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


def environment(pid: int) -> dict[str, str]:
    try:
        entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        return {
            key.decode(): value.decode()
            for item in entries
            if b"=" in item
            for key, value in (item.split(b"=", 1),)
        }
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return {}


def pid_from(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def manifest_keys(unit: str) -> set[str]:
    manifest = json.loads(
        (RUN / "calibration_manifests" / f"{unit}.json").read_text(encoding="utf-8")
    )
    return {str(row["episode_key"]) for row in manifest["qvla_episodes"]}


def committed_keys(unit: str) -> set[str]:
    rows: dict[str, dict[str, Any]] = {}
    directory = RUN / "calibration_fp16" / unit
    for journal in sorted(directory.glob("worker_*.jsonl")):
        for line_number, line in enumerate(journal.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["episode_key"])
            if key in rows and rows[key] != row:
                raise ValueError(f"conflicting committed row {key} at {journal}:{line_number}")
            rows[key] = row
    extra = set(rows) - manifest_keys(unit)
    if extra:
        raise ValueError(f"{unit}: {len(extra)} out-of-manifest keys")
    return set(rows)


def shard_keys(keys: set[str], shard: int, count: int) -> set[str]:
    return {
        key
        for key in keys
        if int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % count == shard
    }


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


def process_for_port(port: int) -> tuple[int, str]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmd = command(pid)
        if (
            "scripts/inference_service.py" in cmd
            and str(ROOT) in cmd
            and f"--port {port} " in cmd
        ):
            matches.append((pid, cmd))
    if len(matches) != 1:
        raise RuntimeError(f"port {port}: expected one repository service, found {matches}")
    return matches[0]


def matching_pid_files(pid: int) -> list[Path]:
    result = []
    for path in sorted((RUN / "teacher_servers").glob("*.pid")):
        if pid_from(path) == pid:
            result.append(path)
    return result


def validate_precision(metadata: dict[str, Any], unit: str) -> None:
    precision = metadata.get("model_dtype") or {}
    dtypes = precision.get("linear_conv_layers_by_weight_dtype") or {}
    if precision.get("resolved") != "float16" or set(dtypes) != {"float16"}:
        raise RuntimeError(f"{unit}: teacher is not strict FP16: {precision}")


def discover_servers() -> tuple[list[dict[str, Any]], dict[str, str]]:
    expected_sha: dict[str, str] = {}
    for unit, ports in EXISTING_PORTS.items():
        for port in ports:
            pid, cmd = process_for_port(port)
            if f"/{SPLIT[unit]}/checkpoint-60000" not in cmd:
                raise RuntimeError(f"port {port}: checkpoint drift: {cmd}")
            metadata = runtime_info(port)
            validate_precision(metadata, unit)
            sha = str(metadata["metadata_sha256"])
            if unit in expected_sha and expected_sha[unit] != sha:
                raise RuntimeError(f"{unit}: semantic metadata differs across replicas")
            expected_sha[unit] = sha
    atomic = []
    for port in ATOMIC_PORTS:
        pid, cmd = process_for_port(port)
        if "/atomic_seen/checkpoint-60000" not in cmd:
            raise RuntimeError(f"port {port}: expected atomic checkpoint: {cmd}")
        pid_files = matching_pid_files(pid)
        if not pid_files:
            raise RuntimeError(f"port {port}: service PID {pid} has no ownership file")
        gpu = environment(pid).get("CUDA_VISIBLE_DEVICES")
        if gpu is None or not gpu.isdigit():
            raise RuntimeError(f"port {port}: unresolved CUDA_VISIBLE_DEVICES for PID {pid}")
        atomic.append(
            {
                "port": port,
                "pid": pid,
                "gpu": int(gpu),
                "command": cmd,
                "pid_files": [str(path) for path in pid_files],
                "replacement_unit": REPLACEMENTS[port],
            }
        )
    return atomic, expected_sha


def validated_control_process(path: Path, needle: str) -> dict[str, Any] | None:
    pid = pid_from(path)
    if pid is None or not alive(pid):
        return None
    cmd = command(pid)
    if needle not in cmd or str(ROOT) not in cmd:
        raise RuntimeError(f"refusing non-owned PID {pid} from {path}: {cmd}")
    return {"pid": pid, "command": cmd, "pid_file": str(path)}


def active_composite_collectors() -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for unit in SPLIT:
        for path in sorted((RUN / "collectors_fp16_strict").glob(f"{unit}_s*.pid")):
            row = validated_control_process(path, "collect_qvla_actquant_teacher.py")
            if row is None:
                continue
            cmd = row["command"]
            if (
                "--model gr00t" not in cmd
                or str(RUN / "calibration_fp16" / unit) not in cmd
            ):
                raise RuntimeError(f"collector ownership drift: {row}")
            records[int(row["pid"])] = {**row, "unit": unit}
    return list(records.values())


def terminate(records: list[dict[str, Any]], *, timeout: int = 180) -> None:
    for row in records:
        pid = int(row["pid"])
        if not alive(pid):
            continue
        if command(pid) != row["command"]:
            raise RuntimeError(f"PID identity changed before SIGTERM: {pid}")
        group = os.getpgid(pid)
        os.killpg(group, signal.SIGTERM) if group == pid else os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline and any(alive(int(row["pid"])) for row in records):
        time.sleep(1)
    remaining = [int(row["pid"]) for row in records if alive(int(row["pid"]))]
    if remaining:
        raise RuntimeError(f"owned processes ignored SIGTERM: {remaining}")


def launch_replacements(
    atomic: list[dict[str, Any]], expected_sha: dict[str, str]
) -> list[dict[str, Any]]:
    terminate(atomic, timeout=300)
    records = []
    for old in atomic:
        unit = str(old["replacement_unit"])
        port, gpu = int(old["port"]), int(old["gpu"])
        checkpoint = ROOT / (
            "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
            f"target_posttraining/{SPLIT[unit]}/checkpoint-60000"
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
            if key.startswith(
                ("GR00T_DUQUANT_", "GR00T_GPTQ", "GR00T_ATM_", "GR00T_OHB_", "QVLA_ACTQUANT_")
            ):
                env.pop(key)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "GR00T_MODEL_DTYPE": "float16",
                "GR00T_CONFIG_ID": "fp16_teacher_proxy",
                "PYTHONUNBUFFERED": "1",
            }
        )
        log = RUN / "teacher_servers" / f"{unit}_rebalanced_p{port}.log"
        with log.open("a", encoding="utf-8", buffering=1) as handle:
            process = subprocess.Popen(
                argv,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        # Rebind every prior ownership file, avoiding stale-PID reuse hazards.
        for pid_file in old["pid_files"]:
            Path(pid_file).write_text(f"{process.pid}\n", encoding="utf-8")
        records.append(
            {
                "unit": unit,
                "port": port,
                "gpu": gpu,
                "pid": process.pid,
                "command": " ".join(argv) + " ",
                "pid_files": old["pid_files"],
                "log": str(log),
            }
        )
    deadline = time.time() + 1_800
    pending = {int(row["port"]): row for row in records}
    errors: dict[int, str] = {}
    while pending and time.time() < deadline:
        for port, row in list(pending.items()):
            if not alive(int(row["pid"])):
                raise RuntimeError(f"replacement teacher exited: {row}")
            try:
                metadata = runtime_info(port)
                validate_precision(metadata, str(row["unit"]))
                if str(metadata["metadata_sha256"]) != expected_sha[str(row["unit"])]:
                    raise RuntimeError("semantic metadata SHA differs from existing replica")
                row["server_metadata_sha256"] = str(metadata["metadata_sha256"])
                del pending[port]
            except Exception as error:  # startup polling records the last cause
                errors[port] = f"{type(error).__name__}: {error}"
        if pending:
            time.sleep(5)
    if pending:
        raise RuntimeError(f"replacement startup timeout: {sorted(pending)} {errors}")
    return records


def all_ports(replacements: list[dict[str, Any]]) -> dict[str, tuple[int, ...]]:
    result = {unit: list(ports) for unit, ports in EXISTING_PORTS.items()}
    for row in replacements:
        result[str(row["unit"])].append(int(row["port"]))
    for unit in result:
        result[unit] = sorted(result[unit])
        if len(result[unit]) != LAYOUT_COUNT or len(set(result[unit])) != LAYOUT_COUNT:
            raise RuntimeError(f"{unit}: invalid 12-port layout {result[unit]}")
    return {unit: tuple(ports) for unit, ports in result.items()}


def launch_collector(unit: str, shard: int, count: int, port: int) -> dict[str, Any]:
    missing = shard_keys(manifest_keys(unit), shard, count) - committed_keys(unit)
    if not missing:
        return {
            "unit": unit,
            "shard": shard,
            "shard_count": count,
            "port": port,
            "pid": None,
            "skipped_complete": True,
            "missing_before_launch": 0,
        }
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
        str(count),
        "--egl-device",
        str(shard % 8),
    ]
    directory = RUN / "collectors_fp16_strict"
    stem = f"{unit}_s{shard:02d}"
    log = directory / f"{stem}_n{count:02d}.log"
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
    return {
        "unit": unit,
        "shard": shard,
        "shard_count": count,
        "port": port,
        "egl_gpu": shard % 8,
        "pid": process.pid,
        "pid_file": str(pid_file),
        "log": str(log),
        "missing_before_launch": len(missing),
        "skipped_complete": False,
    }


def launch_layout(ports: dict[str, tuple[int, ...]], count: int) -> list[dict[str, Any]]:
    records = []
    for unit in SPLIT:
        for shard in range(count):
            records.append(launch_collector(unit, shard, count, ports[unit][shard]))
    time.sleep(10)
    dead = []
    for row in records:
        pid = row.get("pid")
        if pid is None or alive(int(pid)):
            continue
        missing = shard_keys(
            manifest_keys(str(row["unit"])), int(row["shard"]), count
        ) - committed_keys(str(row["unit"]))
        if missing:
            dead.append({**row, "remaining": len(missing)})
    if dead:
        raise RuntimeError(f"collectors exited immediately: {dead}")
    return records


def launch_control(script: str, pid_name: str, log_name: str, args: list[str]) -> dict[str, Any]:
    argv = [GROOT_PY, str(ROOT / "scripts/tools" / script), *args]
    log = RUN / "unattended" / log_name
    with log.open("a", encoding="utf-8", buffering=1) as handle:
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file = RUN / "unattended" / pid_name
    pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    time.sleep(2)
    if not alive(process.pid):
        raise RuntimeError(f"control process exited: {argv}; see {log}")
    return {"pid": process.pid, "pid_file": str(pid_file), "log": str(log)}


def controls() -> list[dict[str, Any]]:
    result = []
    for path, needle in (
        (RUN / "unattended" / "controller_strict_fp16.pid", "qvla_actquant_unattended.py run"),
        (RUN / "unattended" / "calibration_watchdog.pid", "qvla_actquant_calibration_watchdog.py"),
    ):
        row = validated_control_process(path, needle)
        if row is not None:
            result.append(row)
    return result


def start_controls() -> dict[str, Any]:
    controller = launch_control(
        "qvla_actquant_unattended.py",
        "controller_strict_fp16.pid",
        "controller_strict_fp16.log",
        ["run", "--poll-seconds", "60", "--shards", "16"],
    )
    watchdog = launch_control(
        "qvla_actquant_calibration_watchdog.py",
        "calibration_watchdog.pid",
        "calibration_watchdog.log",
        [],
    )
    return {"controller": controller, "watchdog": watchdog}


def preflight() -> dict[str, Any]:
    state = {
        unit: {"completed": len(committed_keys(unit)), "expected": len(manifest_keys(unit))}
        for unit in ("gr00t_atomic_seen", *SPLIT)
    }
    if state["gr00t_atomic_seen"]["completed"] != state["gr00t_atomic_seen"]["expected"]:
        raise RuntimeError(f"atomic calibration is not complete: {state['gr00t_atomic_seen']}")
    if all(state[unit]["completed"] == state[unit]["expected"] for unit in SPLIT):
        raise RuntimeError("both composite calibration units are already complete")
    atomic, expected_sha = discover_servers()
    return {
        "state": state,
        "atomic_servers": atomic,
        "expected_metadata_sha256": expected_sha,
        "active_composite_collectors": len(active_composite_collectors()),
        "controls": controls(),
    }


def apply_rebalance() -> None:
    before = preflight()
    atomic_json(
        STATUS,
        {"schema_version": 1, "stage": "starting_replacement_teachers", "updated_at": now(), "before": before},
    )
    replacements = launch_replacements(
        list(before["atomic_servers"]), dict(before["expected_metadata_sha256"])
    )
    ports = all_ports(replacements)
    atomic_json(
        STATUS,
        {
            "schema_version": 1,
            "stage": "replacement_teachers_ready",
            "updated_at": now(),
            "before": before,
            "replacements": replacements,
            "ports": ports,
        },
    )
    # The expensive/model-risky part is complete.  The active 8-way workers
    # are stopped only now, making the write-side transition short.
    stopped_controls = controls()
    stopped_collectors = active_composite_collectors()
    terminate([*stopped_controls, *stopped_collectors])
    try:
        collectors = launch_layout(ports, LAYOUT_COUNT)
        result = {
            "schema_version": 1,
            "kind": "gr00t_composite_calibration_rebalance_v1",
            "stage": "complete",
            "updated_at": now(),
            "shard_count": LAYOUT_COUNT,
            "before": before,
            "replacements": replacements,
            "ports": ports,
            "collectors": collectors,
            "stopped_controls": stopped_controls,
            "stopped_collectors": stopped_collectors,
        }
        # The watchdog reads this record to understand the new layout.
        atomic_json(STATUS, result)
        result["controls"] = start_controls()
        atomic_json(STATUS, result)
    except Exception as error:
        # Preserve progress with the original 8-way layout and original
        # composite ports if the short collector transition fails.
        terminate(controls())
        for row in active_composite_collectors():
            terminate([row])
        fallback = launch_layout(EXISTING_PORTS, 8)
        fallback_controls = start_controls()
        atomic_json(
            STATUS,
            {
                "schema_version": 1,
                "stage": "failed_rolled_back_collectors",
                "updated_at": now(),
                "error": repr(error),
                "replacements": replacements,
                "fallback_collectors": fallback,
                "fallback_controls": fallback_controls,
            },
        )
        raise
    print(json.dumps({"status": str(STATUS), **result}, indent=2, sort_keys=True))


def recover_collectors() -> None:
    if not STATUS.is_file():
        raise RuntimeError("no composite rebalance record; refusing recovery")
    record = json.loads(STATUS.read_text(encoding="utf-8"))
    if record.get("stage") != "complete" or int(record.get("shard_count", 0)) != LAYOUT_COUNT:
        raise RuntimeError(f"rebalance is not recoverable: stage={record.get('stage')}")
    by_key = {
        (str(row["unit"]), int(row["shard"])): row
        for row in record["collectors"]
    }
    recovered = []
    for unit in SPLIT:
        for shard in range(LAYOUT_COUNT):
            missing = shard_keys(manifest_keys(unit), shard, LAYOUT_COUNT) - committed_keys(unit)
            if not missing:
                continue
            old = by_key[(unit, shard)]
            pid = old.get("pid")
            cmd = command(int(pid)) if pid is not None and alive(int(pid)) else ""
            if (
                cmd
                and "collect_qvla_actquant_teacher.py" in cmd
                and f"--shard-index {shard}" in cmd
                and f"--shard-count {LAYOUT_COUNT}" in cmd
            ):
                continue
            row = launch_collector(unit, shard, LAYOUT_COUNT, int(old["port"]))
            by_key[(unit, shard)] = row
            recovered.append(row)
    record["collectors"] = [by_key[key] for key in sorted(by_key)]
    record["updated_at"] = now()
    record.setdefault("recoveries", []).append(
        {"at": now(), "collectors": recovered}
    )
    atomic_json(STATUS, record)
    print(json.dumps({"recovered": recovered, "status": str(STATUS)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    lock_path = RUN / "unattended" / "gr00t_composite_rebalance.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.check:
            print(json.dumps(preflight(), indent=2, sort_keys=True))
        elif args.apply:
            apply_rebalance()
        else:
            recover_collectors()


if __name__ == "__main__":
    main()
