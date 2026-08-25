#!/usr/bin/env python3
"""Report strict unique-row progress and ETA for the latest π0.5 schedule."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import time


CONFIGS = ("fp16", "quantvla_w4a8_atmohb", "gdsq_vla_atmohb", "gdsq_vla")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional progress baseline JSON; auto-selects the latest schedule baseline.",
    )
    return parser.parse_args()


def select_baseline(run_dir: Path, requested: str | None) -> Path:
    if requested:
        path = Path(requested).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(run_dir.glob("progress_baseline.schedule_v*.json"))
    if not candidates:
        raise FileNotFoundError(f"no schedule progress baseline in {run_dir}")
    return candidates[-1]


def read_rows(directory: Path, config: str) -> dict[tuple[str, int], dict]:
    rows = {}
    for name in glob.glob(str(directory / "results" / config / "*.jsonl")):
        for line_number, line in enumerate(open(name, encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["task"]), int(row["seed"]))
            if key in rows:
                raise ValueError(f"{config}: duplicate {key} in {name}:{line_number}")
            rows[key] = row
    return rows


def live_workers(run_dir: Path, prefix: str | None = None) -> tuple[int, list[str]]:
    alive = 0
    stale = []
    pattern = f"{prefix}*.pid" if prefix else "*.pid"
    for path in (run_dir / "control/workers").glob(pattern):
        try:
            pid = int(path.read_text().strip())
            os.kill(pid, 0)
            alive += 1
        except (ValueError, ProcessLookupError, PermissionError):
            stale.append(path.stem)
    return alive, sorted(stale)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    baseline_path = select_baseline(run_dir, args.baseline)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    schedule = str(baseline.get("schedule", baseline_path.stem.rsplit(".", 1)[-1]))
    elapsed = max(time.time() - float(baseline["started_at_unix"]), 1.0)
    configs = {}
    eta_values = []
    for config in CONFIGS:
        rows = read_rows(run_dir, config)
        completed = len(rows)
        successes = sum(bool(row["success"]) for row in rows.values())
        delta = completed - int(baseline["completed_episodes"][config])
        rate = max(delta, 0) / elapsed * 3600
        eta = (2500 - completed) / rate if rate > 0 else None
        if eta is not None:
            eta_values.append(eta)
        configs[config] = {
            "completed": completed,
            "expected": 2500,
            "successes": successes,
            "observed_episode_sr": successes / completed if completed else None,
            "schedule_new_episodes": delta,
            "schedule_rate_episodes_per_hour": rate,
            "eta_hours_at_schedule_rate": eta,
        }
    worker_prefix = "v3_" if schedule == "schedule_v3" else None
    alive, stale = live_workers(run_dir, worker_prefix)
    print(
        json.dumps(
            {
                "schedule": schedule,
                "baseline": str(baseline_path),
                "schedule_elapsed_minutes": elapsed / 60,
                "workers_alive": alive,
                "stale_workers": stale,
                "estimated_matrix_eta_hours": max(eta_values) if eta_values else None,
                "configs": configs,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
