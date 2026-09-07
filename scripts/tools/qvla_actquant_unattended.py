#!/usr/bin/env python3
"""Unattended calibration-to-pack controller for the Table-1 reproduction.

This controller owns no pre-existing process.  It waits for the already
launched teacher collectors, freezes their exact journals, stops only teacher
PIDs recorded under this run root, and then drives deterministic GPU shards
through the reusable free-memory scheduler.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "qvla_actquant_table1"
PYTHON = "/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
UNITS = (
    "gr00t_atomic_seen",
    "gr00t_composite_seen",
    "gr00t_composite_unseen",
    "pi05_all_target",
)
PHASES = (
    "qvla",
    "actquant-hsic",
    "actquant-fisher",
    "qvla-pack",
    "actquant-allocate",
    "actquant-pack",
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


def append_event(value: dict[str, Any]) -> None:
    path = RUN / "unattended" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": now(), **value}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def calibration_directory(unit: str) -> Path:
    root = "calibration" if unit == "pi05_all_target" else "calibration_fp16"
    return RUN / root / unit


def collector_pid_directory(unit: str) -> Path:
    root = "collectors" if unit == "pi05_all_target" else "collectors_fp16_strict"
    return RUN / root


def committed_keys(unit: str) -> set[str]:
    rows: dict[str, dict[str, Any]] = {}
    directory = calibration_directory(unit)
    for journal in sorted(directory.glob("worker_*.jsonl")):
        for line_number, line in enumerate(journal.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["episode_key"])
            if key in rows and rows[key] != row:
                raise ValueError(
                    f"conflicting calibration key {key} at {journal}:{line_number}"
                )
            # A crash-safe retry may journal the byte-identical committed row
            # under a new shard layout.  The final frozen inventory remains
            # unique by episode_key and binds one archive SHA.
            rows[key] = row
    return set(rows)


def collector_state() -> dict[str, Any]:
    result = {}
    for unit in UNITS:
        keys = committed_keys(unit)
        planned = json.loads(
            (RUN / "calibration_manifests" / f"{unit}.json").read_text(encoding="utf-8")
        )
        expected = {row["episode_key"] for row in planned["qvla_episodes"]}
        result[unit] = {
            "completed": len(keys),
            "expected": len(expected),
            "missing": len(expected - keys),
            "extra": len(keys - expected),
        }
    return result


def alive(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return len(fields) > 2 and fields[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False
    except (PermissionError, OSError):
        return True


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return ""


def assert_collectors_viable(state: dict[str, Any]) -> None:
    for unit, row in state.items():
        if row["completed"] >= row["expected"]:
            continue
        pid_files = sorted(collector_pid_directory(unit).glob(f"{unit}_s*.pid"))
        live = 0
        for pid_file in pid_files:
            try:
                pid = int(pid_file.read_text().strip())
            except (OSError, ValueError):
                continue
            command = cmdline(pid)
            if alive(pid) and "collect_qvla_actquant_teacher.py" in command:
                live += 1
        if live == 0:
            raise RuntimeError(
                f"{unit}: calibration incomplete ({row['completed']}/{row['expected']}) "
                "and no owned collector remains alive"
            )


def run_command(argv: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    append_event({"event": "command_started", "argv": argv, "log": str(log_path)})
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        completed = subprocess.run(
            argv,
            cwd=ROOT,
            env={**os.environ, **(env or {})},
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        append_event(
            {"event": "command_failed", "argv": argv, "returncode": completed.returncode}
        )
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")
    append_event({"event": "command_complete", "argv": argv})


def validate_export(unit: str) -> bool:
    metadata_path = RUN / "actquant_calibration" / unit / "metadata.json"
    frozen_path = RUN / "frozen" / f"{unit}.json"
    if not metadata_path.is_file() or not frozen_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("episode_count") != 60:
            return False
        if metadata.get("frozen_manifest_sha256") != sha256_file(frozen_path):
            return False
        for record in (metadata.get("files") or {}).values():
            path = Path(record["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(record["bytes"])
                or sha256_file(path) != record["sha256"]
            ):
                return False
        return True
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def freeze_all() -> None:
    finalizer = ROOT / "scripts" / "tools" / "finalize_qvla_actquant_calibration.py"
    for unit in UNITS:
        frozen = RUN / "frozen" / f"{unit}.json"
        if validate_export(unit):
            append_event({"event": "calibration_freeze_reused", "unit": unit})
            continue
        run_command(
            [
                PYTHON,
                str(finalizer),
                "freeze",
                "--manifest",
                str(RUN / "calibration_manifests" / f"{unit}.json"),
                "--calibration-dir",
                str(calibration_directory(unit)),
                "--output",
                str(frozen),
                "--precision-attestations",
                str(RUN / "teacher_precision_attestations.json"),
            ],
            RUN / "unattended" / "logs" / f"freeze_{unit}.log",
        )
        if not validate_export(unit):
            run_command(
                [
                    PYTHON,
                    str(finalizer),
                    "export-actquant",
                    "--frozen-manifest",
                    str(frozen),
                    "--output-dir",
                    str(RUN / "actquant_calibration" / unit),
                ],
                RUN / "unattended" / "logs" / f"export_actquant_{unit}.log",
            )
            if not validate_export(unit):
                raise RuntimeError(f"{unit}: ActQuant export failed post-write validation")


def stop_owned_teachers() -> None:
    pid_files = sorted((RUN / "teacher_servers").glob("*.pid"))
    stopped = []
    ignored_stale = []
    for pid_file in pid_files:
        try:
            pid = int(pid_file.read_text().strip())
        except (OSError, ValueError):
            continue
        if not alive(pid):
            continue
        command = cmdline(pid)
        allowed_script = (
            "scripts/inference_service.py" in command
            or "serve_pi05_quant_policy.py" in command
        )
        allowed_port = any(f"--port {port}" in command for port in range(23000, 23107))
        if not allowed_script or not allowed_port or str(ROOT) not in command:
            # A pid file can outlive its process and later name an unrelated
            # process after PID reuse. Never signal it; preserve an audit row.
            ignored_stale.append(
                {"pid": pid, "pid_file": str(pid_file), "command": command}
            )
            continue
        process_group = os.getpgid(pid)
        if process_group == pid:
            os.killpg(process_group, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
        stopped.append({"pid": pid, "pid_file": str(pid_file), "command": command})
    deadline = time.time() + 60
    while time.time() < deadline and any(alive(row["pid"]) for row in stopped):
        time.sleep(1)
    remaining = [row for row in stopped if alive(row["pid"])]
    if remaining:
        raise RuntimeError(
            "owned teacher servers did not terminate after SIGTERM; refusing destructive escalation: "
            f"{[row['pid'] for row in remaining]}"
        )
    atomic_json(
        RUN / "unattended" / "teachers_stopped.json",
        {
            "stopped_at": now(),
            "servers": stopped,
            "ignored_stale_pid_files": ignored_stale,
            "recoverable": False,
        },
    )
    append_event(
        {
            "event": "owned_teachers_stopped",
            "count": len(stopped),
            "ignored_stale_pid_files": len(ignored_stale),
        }
    )


def disk_available_bytes() -> int:
    return os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize


def run_phase(phase: str, shards: int) -> None:
    # Keep manifests and scheduler receipts from different layer-sharding
    # layouts separate.  A numerically equivalent throughput retune must not
    # overwrite the audit trail of an earlier layout.
    run_id = f"{phase}_s{shards:03d}"
    manifest = RUN / "jobs" / f"{run_id}.json"
    run_command(
        [
            PYTHON,
            str(ROOT / "scripts" / "tools" / "qvla_actquant_jobs.py"),
            "--phase",
            phase,
            "--frozen-dir",
            str(RUN / "frozen"),
            "--provenance",
            str(RUN / "provenance.json"),
            "--output-root",
            str(RUN / "artifacts"),
            "--manifest-output",
            str(manifest),
            "--shards",
            str(shards),
        ],
        RUN / "unattended" / "logs" / f"make_jobs_{phase}.log",
    )
    run_command(
        [
            PYTHON,
            str(ROOT / "scripts" / "tools" / "qvla_actquant_gpu_scheduler.py"),
            "--manifest",
            str(manifest),
            "--run-dir",
            str(RUN / "scheduler" / run_id),
            "--reserve-mib",
            "4096",
            "--poll-seconds",
            "15",
            "--launch-settle-seconds",
            "30",
        ],
        RUN / "unattended" / "logs" / f"scheduler_{phase}.log",
    )


def verify_packs() -> dict[str, Any]:
    records = {}
    for method in ("qvla", "actquant"):
        for unit in UNITS:
            manifest_path = RUN / "artifacts" / "packs" / method / unit / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("test_results_used") is not False:
                raise ValueError(f"{manifest_path}: test-feedback attestation drift")
            if manifest.get("source_protocol_equivalent") is not False:
                raise ValueError(f"{manifest_path}: teacher-proxy attestation drift")
            if method == "qvla":
                precision = manifest.get("model_precision") or {}
                if (
                    precision.get("resolved") != "bfloat16"
                    or precision.get("strict_all_linear_conv_bf16") is not True
                    or manifest.get("activation_compute_dtype") != "bfloat16"
                ):
                    raise ValueError(f"{manifest_path}: QVLA BF16 precision attestation drift")
                if float(manifest["actual_average_channel_bits"]) > 4.0:
                    raise ValueError(f"{manifest_path}: QVLA average bits exceed 4")
                pack = manifest["pack"]
                pack_path = Path(pack["path"])
                if (
                    not pack_path.is_file()
                    or sha256_file(pack_path) != pack["sha256"]
                    or int(manifest["full_component_static_bytes"]) != pack_path.stat().st_size
                    or int(manifest.get("fixed_precision_pack_payload_bytes", 0)) <= 0
                ):
                    raise ValueError(f"{manifest_path}: incomplete/corrupt complete QVLA pack")
            else:
                precision = manifest.get("model_precision") or {}
                if (
                    precision.get("resolved") != "float16"
                    or precision.get("strict_all_linear_conv_fp16") is not True
                    or manifest.get("activation_compute_dtype") != "float16"
                ):
                    raise ValueError(f"{manifest_path}: ActQuant FP16 precision attestation drift")
                if float(manifest["achieved_bpw"]) > 4.0 + 1e-6:
                    raise ValueError(f"{manifest_path}: ActQuant BPW exceeds 4")
                static_bytes = 0
                for file_record in manifest.get("files", []):
                    artifact = Path(file_record["path"])
                    if not artifact.is_file() or sha256_file(artifact) != file_record["sha256"]:
                        raise ValueError(f"{manifest_path}: corrupt ActQuant GGUF {artifact}")
                    static_bytes += artifact.stat().st_size
                expected_static = int(
                    manifest[
                        "gguf_static_bytes"
                        if unit == "pi05_all_target"
                        else "full_component_static_bytes"
                    ]
                )
                if static_bytes != expected_static:
                    raise ValueError(f"{manifest_path}: ActQuant static byte accounting drift")
                if unit != "pi05_all_target" and not manifest.get("fixed_tensors"):
                    raise ValueError(f"{manifest_path}: missing fixed-FP16 component")
            records[f"{method}_{unit}"] = {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            }
    result = {"verified_at": now(), "packs": records, "test_results_used": False}
    atomic_json(RUN / "artifacts" / "pack_inventory.json", result)
    return result


def status(stage: str, **extra: Any) -> None:
    atomic_json(
        RUN / "unattended" / "status.json",
        {"updated_at": now(), "stage": stage, **extra},
    )


def execute(
    *,
    poll_seconds: int,
    qvla_shards: int,
    hsic_shards: int,
    fisher_shards: int,
    run_formal: bool,
) -> None:
    lock_path = RUN / "unattended" / "controller.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another QVLA/ActQuant unattended controller owns the lock") from error
        append_event({"event": "controller_started", "pid": os.getpid()})
        while True:
            state = collector_state()
            status("waiting_for_calibration", calibration=state)
            assert_collectors_viable(state)
            if all(row["completed"] == row["expected"] and row["extra"] == 0 for row in state.values()):
                break
            time.sleep(max(10, poll_seconds))
        status("freezing_calibration", calibration=state)
        freeze_all()
        stop_owned_teachers()
        available = disk_available_bytes()
        if available < 150 * 1024**3:
            raise RuntimeError(
                f"less than 150 GiB free before real-pack stages: {available / 1024**3:.1f} GiB"
            )
        phase_shards = {
            "qvla": qvla_shards,
            "qvla-pack": qvla_shards,
            "actquant-hsic": hsic_shards,
            "actquant-allocate": hsic_shards,
            "actquant-fisher": fisher_shards,
            "actquant-pack": fisher_shards,
        }
        for phase in PHASES:
            shards = phase_shards[phase]
            status(
                "gpu_phase",
                phase=phase,
                shards=shards,
                disk_available_bytes=disk_available_bytes(),
            )
            run_phase(phase, shards)
        inventory = verify_packs()
        status("packs_complete", pack_inventory=inventory)
        if run_formal:
            formal_script = ROOT / "scripts" / "tools" / "qvla_actquant_formal_pipeline.py"
            if not formal_script.is_file():
                raise FileNotFoundError(formal_script)
            run_command(
                [PYTHON, str(formal_script), "run"],
                RUN / "unattended" / "logs" / "formal_pipeline.log",
            )
        status("complete", pack_inventory=inventory, formal_requested=run_formal)
        append_event({"event": "controller_complete"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--qvla-shards", type=int)
    parser.add_argument("--hsic-shards", type=int)
    parser.add_argument("--fisher-shards", type=int)
    parser.add_argument("--no-formal", action="store_true")
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps({"calibration": collector_state()}, indent=2, sort_keys=True))
        return
    qvla_shards = args.qvla_shards if args.qvla_shards is not None else args.shards
    hsic_shards = args.hsic_shards if args.hsic_shards is not None else args.shards
    fisher_shards = args.fisher_shards if args.fisher_shards is not None else args.shards
    if min(qvla_shards, hsic_shards, fisher_shards) < 1:
        raise ValueError("shards must be positive")
    try:
        execute(
            poll_seconds=args.poll_seconds,
            qvla_shards=qvla_shards,
            hsic_shards=hsic_shards,
            fisher_shards=fisher_shards,
            run_formal=not args.no_formal,
        )
    except Exception as error:
        status("failed", error=repr(error))
        append_event({"event": "controller_failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
