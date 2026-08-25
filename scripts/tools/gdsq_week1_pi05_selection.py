#!/usr/bin/env python3
"""Materialize and freeze result-blind pi0.5 same-budget control selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "runs/gdsq_week1_preregistered_v1"
EXECUTION = ROOT / "execution"
DEV4 = [
    "CoffeeSetupMug",
    "OpenCabinet",
    "OpenStandMixerHead",
    "PickPlaceDrawerToCounter",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def frozen_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == rendered, f"frozen artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def screen_path(family: str) -> Path:
    return EXECUTION / "screen" / f"pi05_{family}.json"


def materialize_dev() -> dict[str, Any]:
    entries = []
    screen_records = {}
    for family in ("random", "action_only"):
        path = screen_path(family)
        report = load(path)
        require(report.get("complete") is True, f"incomplete {family} screen")
        require(report.get("heldout_results_read") is False, f"tainted {family} screen")
        require(report.get("development_tasks") == DEV4, f"development task drift: {family}")
        require(len(report["development_representatives"]) == 7, f"wrong representatives: {family}")
        screen_records[family] = artifact(path)
        for row in report["development_representatives"]:
            entries.append(
                {
                    "config_id": f"{family}_{int(row['candidate_index']):02d}",
                    "family": family,
                    "candidate_index": int(row["candidate_index"]),
                    "plan_path": row["plan_path"],
                    "plan_sha256": row["plan_sha256"],
                    "a8_path": row["a8_path"],
                    "a8_sha256": row["a8_sha256"],
                    "wrapped_layers": int(row["wrapped_layers"]),
                    "d_func": float(row["d_func"]),
                }
            )
    waves = []
    for wave_index, start in enumerate(range(0, len(entries), 8), 1):
        values = entries[start : start + 8]
        for offset, row in enumerate(values):
            row["gpu"] = offset
            row["port"] = 20300 + (wave_index - 1) * 10 + offset
        waves.append({"wave": wave_index, "configs": values})
    payload = {
        "schema_version": 1,
        "kind": "gdsq_vla_pi05_control_dev_schedule",
        "result_blind": True,
        "development_tasks": DEV4,
        "heldout_tasks_read": False,
        "selection_rule": "max dev4 task-macro SR; ties: lower D_func then candidate index",
        "screen_reports": screen_records,
        "configurations": len(entries),
        "waves": waves,
    }
    output = EXECUTION / "selection/pi05_control_dev_schedule.json"
    frozen_json(output, payload)
    return {"schedule": str(output), "sha256": sha256_file(output), "waves": len(waves)}


def freeze_selection() -> dict[str, Any]:
    schedule_path = EXECUTION / "selection/pi05_control_dev_schedule.json"
    schedule = load(schedule_path)
    require(schedule.get("result_blind") is True, "development schedule is not result-blind")
    selected = {}
    dev_artifacts = []
    for family in ("random", "action_only"):
        candidates = [
            row
            for wave in schedule["waves"]
            for row in wave["configs"]
            if row["family"] == family
        ]
        scored = []
        for candidate in candidates:
            run_dir = EXECUTION / "runs/pi05_controls_dev4" / candidate["config_id"]
            manifest_path = run_dir / "manifest.json"
            summary_path = run_dir / "aggregate/summary.json"
            manifest, summary = load(manifest_path), load(summary_path)
            checks = {
                "manifest_immutable": manifest.get("immutable") is True,
                "manifest_blind": manifest.get("result_blind") is True,
                "scope": manifest.get("protocol", {}).get("scope") == "dev4",
                "tasks": manifest.get("protocol", {}).get("task_sets") == {"atomic_seen": DEV4},
                "config": manifest.get("config_id") == candidate["config_id"],
                "plan": manifest.get("quantization", {}).get("plan", {}).get("sha256")
                == candidate["plan_sha256"],
                "summary_complete": summary.get("complete") is True,
                "summary_formal": summary.get("formal_result") is True,
                "coverage": summary.get("completed_episodes")
                == summary.get("expected_episodes")
                == 200,
                "summary_manifest": summary.get("manifest_sha256") == sha256_file(manifest_path),
            }
            failed = [name for name, valid in checks.items() if not valid]
            require(not failed, f"invalid dev result {candidate['config_id']}: {failed}")
            macro = float(summary["configs"][candidate["config_id"]]["task_macro_sr"])
            scored.append({**candidate, "dev_task_macro_sr": macro})
            dev_artifacts.extend([artifact(manifest_path), artifact(summary_path)])
        winner = min(
            scored,
            key=lambda row: (-row["dev_task_macro_sr"], row["d_func"], row["candidate_index"]),
        )
        selected[family] = {"candidates": scored, "winner": winner}
    payload = {
        "schema_version": 1,
        "kind": "gdsq_vla_pi05_same_budget_dev_selection",
        "valid": True,
        "development_tasks": DEV4,
        "heldout_results_read": False,
        "formal_scope": "heldout46",
        "selection_rule": schedule["selection_rule"],
        "schedule": artifact(schedule_path),
        "development_artifacts": dev_artifacts,
        "selected": selected,
    }
    output = EXECUTION / "selection/pi05_controls_dev4.json"
    frozen_json(output, payload)
    return {"selection": str(output), "sha256": sha256_file(output), "selected": selected}


def formal_spec() -> dict[str, Any]:
    selection_path = EXECUTION / "selection/pi05_controls_dev4.json"
    selection = load(selection_path)
    require(selection.get("valid") is True, "invalid development selection")
    require(selection.get("heldout_results_read") is False, "held-out feedback leak")
    placements = {"random": [0, 1, 2, 3], "action_only": [4, 5, 6, 7]}
    ports = {
        "random": [20600, 20601, 20602, 20603],
        "action_only": [20604, 20605, 20606, 20607],
    }
    configs = []
    for family in ("random", "action_only"):
        winner = selection["selected"][family]["winner"]
        configs.append(
            {
                **winner,
                "formal_config_id": (
                    "search_matched_random" if family == "random" else "action_only"
                ),
                "gpus": placements[family],
                "ports": ports[family],
                "selection_sha256": sha256_file(selection_path),
                "heldout_feedback_used": False,
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "gdsq_vla_pi05_same_budget_formal_spec",
        "result_blind": True,
        "scope": "heldout46",
        "development_tasks_excluded": DEV4,
        "selection": artifact(selection_path),
        "configs": configs,
    }
    output = EXECUTION / "specs/pi05_same_budget_controls_heldout46.json"
    frozen_json(output, payload)
    return {"spec": str(output), "sha256": sha256_file(output), "configs": configs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("materialize-dev", "select", "formal-spec"))
    args = parser.parse_args()
    if args.command == "materialize-dev":
        result = materialize_dev()
    elif args.command == "select":
        result = freeze_selection()
    else:
        result = formal_spec()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
