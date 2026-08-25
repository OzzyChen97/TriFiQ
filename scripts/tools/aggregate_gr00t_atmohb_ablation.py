#!/usr/bin/env python3
"""Paired ATM-vs-OHB ablation summary for GR00T N1.5 on the atomic task group.

Merges the new rows (cscka_final_atm / cscka_final_ohb) from
runs/robocasa365_atmohb_ablation_atomic with the completed official atomic
matrix rows (fp16 / w4a8_atmohb / cscka_final / cscka_final_atmohb) from
runs/robocasa365_official_full_atomic_paired50.  All rows share the
deterministic sha256(task,env_seed,replan_index) paired action-noise scheme,
so contrasts across the two run directories are valid paired comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import robocasa  # noqa: F401
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

from atmohb_ablation_common import build_summary, write_markdown

REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_RUN = REPO_ROOT / "runs/robocasa365_atmohb_ablation_atomic"
BASE_RUN = REPO_ROOT / "runs/robocasa365_official_full_atomic_paired50"

ABLATION_CONFIGS = ["cscka_final_atm", "cscka_final_ohb"]
BASE_CONFIGS = ["fp16", "w4a8_atmohb", "cscka_final", "cscka_final_atmohb"]
CONTRASTS = [
    ("cscka_final_atm", "cscka_final"),
    ("cscka_final_ohb", "cscka_final"),
    ("cscka_final_atm", "cscka_final_atmohb"),
    ("cscka_final_ohb", "cscka_final_atmohb"),
    ("cscka_final_atmohb", "cscka_final"),
]
NOISE_SCHEME = "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-run-dir", default=str(ABLATION_RUN))
    parser.add_argument("--base-run-dir", default=str(BASE_RUN))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def load_gr00t_rows(run_dir: Path, config: str) -> dict[tuple[str, int], dict]:
    root = run_dir
    rows = {}
    for path in sorted(root.glob(f"{config}_s*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from exc
            if row.get("config") != config:
                raise ValueError(f"{path}:{line_number}: config mismatch")
            if row.get("crashed") or row.get("success") is None:
                raise ValueError(f"{path}:{line_number}: crashed/incomplete row")
            if bool(row.get("paired_action_noise")) is not True:
                raise ValueError(f"{path}:{line_number}: paired-noise mismatch")
            if row.get("action_noise_scheme") != NOISE_SCHEME:
                raise ValueError(f"{path}:{line_number}: noise-scheme mismatch")
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
        rows_by_config[config] = load_gr00t_rows(base_run, config)
    for config in ABLATION_CONFIGS:
        rows_by_config[config] = load_gr00t_rows(ablation_run, config)

    summary = build_summary(rows_by_config, tasks, seeds, CONTRASTS, args.bootstrap)
    summary["kind"] = "gr00t_gdsq_atmohb_component_ablation"
    summary["ablation_run_dir"] = str(ablation_run)
    summary["base_run_dir"] = str(base_run)
    summary["noise_scheme"] = NOISE_SCHEME

    out_dir = ablation_run / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out) if args.out else out_dir / "summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(write_markdown(summary, "GR00T N1.5 GDSQ-VLA ATM/OHB ablation (atomic_seen)"), encoding="utf-8")
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