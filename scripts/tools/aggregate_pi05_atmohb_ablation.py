#!/usr/bin/env python3
"""Paired ATM-vs-OHB ablation summary for pi0.5 on the atomic task group.

Merges the new rows (gdsq_vla_atm_only / gdsq_vla_ohb_only) from
runs/pi05_atmohb_ablation_atomic with the completed Table-1 rows
(gdsq_vla / gdsq_vla_atmohb, plus fp16 / quantvla context) from
runs/pi05_gdsq_gr00t_aligned/official_target_paired50.  All rows share the
deterministic (task, seed, replan) paired action-noise protocol, so contrasts
across the two run directories are valid paired comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import robocasa  # noqa: F401
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

from openpi_client.paired_noise import PROTOCOL as NOISE_PROTOCOL

from atmohb_ablation_common import build_summary, write_markdown

REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_RUN = REPO_ROOT / "runs/pi05_atmohb_ablation_atomic"
BASE_RUN = REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/official_target_paired50"

ABLATION_CONFIGS = ["gdsq_vla_atm_only", "gdsq_vla_ohb_only"]
BASE_CONFIGS = ["gdsq_vla", "gdsq_vla_atmohb", "fp16", "quantvla_w4a8_atmohb"]
CONTRASTS = [
    ("gdsq_vla_atm_only", "gdsq_vla"),
    ("gdsq_vla_ohb_only", "gdsq_vla"),
    ("gdsq_vla_atm_only", "gdsq_vla_atmohb"),
    ("gdsq_vla_ohb_only", "gdsq_vla_atmohb"),
    ("gdsq_vla_atmohb", "gdsq_vla"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-run-dir", default=str(ABLATION_RUN))
    parser.add_argument("--base-run-dir", default=str(BASE_RUN))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def load_pi05_rows(run_dir: Path, config: str) -> dict[tuple[str, int], dict]:
    root = run_dir / "results" / config
    if not root.is_dir():
        raise SystemExit(f"missing results directory: {root}")
    rows = {}
    for path in sorted(root.glob("atomic_seen_*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from exc
            if row.get("status") != "complete":
                raise ValueError(f"{path}:{line_number}: non-complete row")
            if row.get("config") != config or row.get("task_set") != "atomic_seen":
                raise ValueError(f"{path}:{line_number}: config/task-set mismatch")
            if row.get("split") != "target":
                raise ValueError(f"{path}:{line_number}: split mismatch")
            if row.get("paired_action_noise") is not True:
                raise ValueError(f"{path}:{line_number}: paired-noise mismatch")
            if row.get("action_noise_protocol") != NOISE_PROTOCOL:
                raise ValueError(f"{path}:{line_number}: noise-protocol mismatch")
            key = (str(row["task"]), int(row["seed"]))
            if key in rows:
                raise ValueError(f"{path}:{line_number}: duplicate {key}")
            rows[key] = row
    return rows


def main() -> None:
    args = parse_args()
    ablation_run = Path(args.ablation_run_dir).resolve()
    base_run = Path(args.base_run_dir).resolve()
    tasks = list(TASK_SET_REGISTRY["atomic_seen"])
    seeds = list(range(50))

    rows_by_config: dict[str, dict[tuple[str, int], dict]] = {}
    for config in BASE_CONFIGS:
        rows_by_config[config] = load_pi05_rows(base_run, config)
    for config in ABLATION_CONFIGS:
        rows_by_config[config] = load_pi05_rows(ablation_run, config)

    summary = build_summary(rows_by_config, tasks, seeds, CONTRASTS, args.bootstrap)
    summary["kind"] = "pi05_gdsq_atmohb_component_ablation"
    summary["ablation_run_dir"] = str(ablation_run)
    summary["base_run_dir"] = str(base_run)
    summary["noise_protocol"] = NOISE_PROTOCOL

    out_dir = ablation_run / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out) if args.out else out_dir / "summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(write_markdown(summary, "pi0.5 GDSQ-VLA ATM/OHB ablation (atomic_seen)"), encoding="utf-8")
    print(json.dumps(
        {
            "out": str(out_json),
            "markdown": str(out_md),
            "configs": {
                config_id: {
                    "task_macro_sr": row["task_macro_sr"],
                    "task_cluster_ci95": row["task_cluster_ci95"],
                }
                for config_id, row in summary["configs"].items()
            },
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()