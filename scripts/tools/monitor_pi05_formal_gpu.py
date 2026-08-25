#!/usr/bin/env python3
"""Sample per-server and device-level GPU efficiency for the formal π0.5 run."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def rows(command: list[str]) -> list[list[str]]:
    output = subprocess.check_output(command, text=True)
    return [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(output))
        if row
    ]


def live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def read_pid(path: Path) -> int | None:
    try:
        value = int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    return value if live(value) else None


def sample(run_dir: Path, manifest: dict) -> list[dict]:
    gpu_rows = rows(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = {
        int(index): {
            "uuid": uuid,
            "device_memory_used_mib": float(memory),
            "device_utilization_gpu_pct": float(utilization),
            "device_power_draw_w": float(power),
        }
        for index, uuid, memory, utilization, power in gpu_rows
    }
    process_rows = rows(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    process_memory = {}
    for pid_text, uuid, memory_text in process_rows:
        if memory_text in {"[N/A]", "N/A"}:
            continue
        process_memory[(int(pid_text), uuid)] = float(memory_text)

    timestamp = time.time()
    output = []
    control = run_dir / "control"
    for server in manifest["servers"]:
        gpu = int(server["gpu"])
        pid = read_pid(control / f"{server['instance']}.pid")
        # A wave schedule lists every config/GPU replica in the immutable
        # manifest, while only one config wave is live at a time.  Sampling
        # stale instances would attribute another config's whole-device load
        # to the inactive config and corrupt the efficiency table.
        if pid is None:
            continue
        uuid = devices[gpu]["uuid"]
        output.append(
            {
                "timestamp": timestamp,
                "instance": server["instance"],
                "config": server["config_id"],
                "gpu": gpu,
                "server_pid": pid,
                "server_process_memory_mib": (
                    process_memory.get((pid, uuid)) if pid is not None else None
                ),
                **devices[gpu],
            }
        )
    return output


def workers_alive(run_dir: Path) -> int:
    directory = run_dir / "control/workers"
    return sum(read_pid(path) is not None for path in directory.glob("*.pid"))


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    output = run_dir / "gpu_efficiency.jsonl"
    seen_workers = False
    while True:
        alive = workers_alive(run_dir)
        seen_workers = seen_workers or alive > 0
        batch = sample(run_dir, manifest)
        with output.open("a", encoding="utf-8") as handle:
            for row in batch:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(f"[pi05 gpu monitor] rows={len(batch)} workers_alive={alive}", flush=True)
        if args.once or (seen_workers and alive == 0):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
