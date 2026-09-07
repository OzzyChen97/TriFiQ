#!/usr/bin/env python3
"""Strict paired aggregation for the preregistered pi0.5 FCP diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tools"))

from quantvla_cross_model_protocol import (  # noqa: E402
    closed_loop_row_protocol,
    sha256_file,
    validate_closed_loop_row,
)
from quantvla_outputimpact import atomic_json  # noqa: E402


CONFIG_ID = "full_context_w4a8_dynamic_profile"
NEW_ARMS = ("transferred_initializer", "single_best", "two_best")


def exact_mcnemar(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    numerator = 2 * sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, float(Fraction(numerator, 2**discordant)))


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda key: (raw[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * raw[key]))
        adjusted[key] = running
    return adjusted


def hierarchical_bootstrap(
    left: dict[tuple[str, str, int], dict[str, Any]],
    right: dict[tuple[str, str, int], dict[str, Any]],
    *,
    draws: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    tasks = sorted({(split, task) for split, task, _seed in left})
    deltas: dict[tuple[str, str], np.ndarray] = {}
    for split, task in tasks:
        seeds = sorted(seed for row_split, row_task, seed in left if (row_split, row_task) == (split, task))
        deltas[(split, task)] = np.asarray(
            [
                float(bool(left[(split, task, value)]["success"]))
                - float(bool(right[(split, task, value)]["success"]))
                for value in seeds
            ],
            dtype=np.float64,
        )
    generator = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        task_indices = generator.integers(0, len(tasks), size=len(tasks))
        task_means = []
        for task_index in task_indices:
            values = deltas[tasks[int(task_index)]]
            seed_indices = generator.integers(0, len(values), size=len(values))
            task_means.append(float(values[seed_indices].mean()))
        samples[draw] = float(np.mean(task_means))
    observed = float(np.mean([values.mean() for values in deltas.values()]))
    return {
        "draws": draws,
        "seed": seed,
        "observed_task_macro_delta": observed,
        "bootstrap_mean_delta": float(samples.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def validate_source_hashes(paths: Iterable[Path], registered: list[dict[str, Any]]) -> None:
    observed = {str(path.resolve()): sha256_file(path) for path in paths}
    expected = {str(Path(row["path"]).resolve()): row["sha256"] for row in registered}
    if observed != expected:
        raise ValueError("projected-anchor source inventory or hash drift")


def load_rows(
    paths: Iterable[Path],
    expected: set[tuple[str, str, int]],
    *,
    allowed_metadata: set[str] | None,
    source_label: str,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    wanted_protocol = closed_loop_row_protocol()
    wanted_protocol["flow_steps"] = 4
    for path in sorted(paths):
        used = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config") != CONFIG_ID:
                continue
            key = (str(row.get("task_set")), str(row.get("task")), int(row.get("seed", -1)))
            if key not in expected:
                continue
            if key in rows:
                raise ValueError(f"duplicate {source_label} key {key}: {path}:{line_number}")
            if row.get("status") != "complete":
                raise ValueError(f"incomplete {source_label} row: {path}:{line_number}")
            if allowed_metadata is not None and row.get("server_metadata_sha256") not in allowed_metadata:
                raise ValueError(f"unregistered {source_label} runtime: {path}:{line_number}")
            mismatches = {
                field: (row.get(field), wanted)
                for field, wanted in wanted_protocol.items()
                if row.get(field) != wanted
            }
            if mismatches:
                raise ValueError(f"{source_label} protocol drift {path}:{line_number}: {mismatches}")
            validate_closed_loop_row(row, source=f"{path}:{line_number}")
            rows[key] = row
            used += 1
        if used:
            sources.append({"path": str(path.resolve()), "sha256": sha256_file(path), "episodes": used})
    missing = sorted(expected - set(rows))
    extra = sorted(set(rows) - expected)
    if missing or extra:
        raise ValueError(
            f"{source_label} coverage drift: observed={len(rows)} expected={len(expected)} "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    return rows, sources


def rates(
    rows: dict[tuple[str, str, int], dict[str, Any]],
    task_sets: dict[str, list[str]],
) -> dict[str, Any]:
    by_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (split, task, _seed), row in rows.items():
        by_task[(split, task)].append(float(bool(row["success"])))
    per_task = {
        f"{split}/{task}": float(np.mean(values))
        for (split, task), values in sorted(by_task.items())
    }
    split_macro = {
        split: float(np.mean([per_task[f"{split}/{task}"] for task in tasks]))
        for split, tasks in task_sets.items()
    }
    values = [float(bool(row["success"])) for row in rows.values()]
    return {
        "episodes": len(values),
        "successes": int(sum(values)),
        "micro_success_rate": float(np.mean(values)),
        "task_macro_success_rate": float(np.mean(list(per_task.values()))),
        "split_task_macro_success_rate": split_macro,
        "per_task_success_rate": per_task,
    }


def compare(
    left_id: str,
    left: dict[tuple[str, str, int], dict[str, Any]],
    right_id: str,
    right: dict[tuple[str, str, int], dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError(f"unpaired comparison: {left_id} vs {right_id}")
    wins = sum(bool(left[key]["success"]) and not bool(right[key]["success"]) for key in left)
    losses = sum(not bool(left[key]["success"]) and bool(right[key]["success"]) for key in left)
    ties = len(left) - wins - losses
    return {
        "orientation": f"{left_id} minus {right_id}",
        "task_macro_delta": summaries[left_id]["task_macro_success_rate"]
        - summaries[right_id]["task_macro_success_rate"],
        "micro_delta": summaries[left_id]["micro_success_rate"]
        - summaries[right_id]["micro_success_rate"],
        "paired_left_wins": int(wins),
        "paired_left_losses": int(losses),
        "paired_ties": int(ties),
        "discordant_pairs": int(wins + losses),
        "exact_two_sided_mcnemar_p": exact_mcnemar(int(wins), int(losses)),
        "hierarchical_bootstrap": hierarchical_bootstrap(left, right),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--execution-manifest", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--raw-out", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    execution = json.loads(args.execution_manifest.read_text(encoding="utf-8"))
    if prereg.get("kind") != "pi05_fcp_same_protocol_preregistration":
        raise ValueError("invalid pi0.5 FCP preregistration")
    if execution.get("kind") != "pi05_fcp_same_protocol_execution_manifest":
        raise ValueError("invalid pi0.5 FCP execution manifest")
    if execution["preregistration"]["sha256"] != sha256_file(args.preregistration):
        raise ValueError("execution/preregistration hash drift")
    tasks = {split: list(values) for split, values in prereg["evaluation"]["task_sets"].items()}
    seeds = [int(value) for value in prereg["evaluation"]["seeds"]]
    expected = {
        (split, task, seed)
        for split, split_tasks in tasks.items()
        for task in split_tasks
        for seed in seeds
    }
    if len(expected) != 500:
        raise ValueError("expected-key contract drift")

    anchor_paths = [Path(row["path"]) for row in prereg["evaluation"]["projected_anchor_source_files"]]
    validate_source_hashes(anchor_paths, prereg["evaluation"]["projected_anchor_source_files"])
    all_rows: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    sources: dict[str, Any] = {}
    all_rows["projected_anchor"], sources["projected_anchor"] = load_rows(
        anchor_paths, expected, allowed_metadata=None, source_label="projected_anchor"
    )
    if sum(bool(row["success"]) for row in all_rows["projected_anchor"].values()) != 141:
        raise ValueError("known anchor outcome drift")
    for arm in NEW_ARMS:
        allowed = {row["server_metadata_sha256"] for row in execution["servers"][arm]}
        paths = sorted(
            (args.results_root / arm / "results" / CONFIG_ID).glob("*.jsonl")
        )
        all_rows[arm], sources[arm] = load_rows(
            paths, expected, allowed_metadata=allowed, source_label=arm
        )

    summaries = {arm: rates(rows, tasks) for arm, rows in all_rows.items()}
    primary = compare(
        "projected_anchor",
        all_rows["projected_anchor"],
        "transferred_initializer",
        all_rows["transferred_initializer"],
        summaries,
    )
    secondary = {
        arm: compare(arm, all_rows[arm], "projected_anchor", all_rows["projected_anchor"], summaries)
        for arm in ("single_best", "two_best")
    }
    adjusted = holm_adjust(
        {arm: row["exact_two_sided_mcnemar_p"] for arm, row in secondary.items()}
    )
    for arm, value in adjusted.items():
        secondary[arm]["holm_adjusted_p"] = value

    raw = {
        "schema_version": 1,
        "kind": "pi05_fcp_same_protocol_raw_aggregate",
        "complete": True,
        "preregistration": {"path": str(args.preregistration.resolve()), "sha256": sha256_file(args.preregistration)},
        "execution_manifest": {"path": str(args.execution_manifest.resolve()), "sha256": sha256_file(args.execution_manifest)},
        "coverage": {
            "new_rollouts": sum(len(all_rows[arm]) for arm in NEW_ARMS),
            "reused_anchor_rollouts": len(all_rows["projected_anchor"]),
            "unique_keys_per_configuration": len(expected),
        },
        "outcomes": {
            arm: [
                {
                    "task_set": split,
                    "task": task,
                    "seed": seed,
                    "success": bool(rows[(split, task, seed)]["success"]),
                    "server_metadata_sha256": rows[(split, task, seed)].get("server_metadata_sha256"),
                }
                for split, task, seed in sorted(rows)
            ]
            for arm, rows in all_rows.items()
        },
        "sources": sources,
    }
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.raw_out, raw)
    summary = {
        "schema_version": 1,
        "kind": "pi05_fcp_same_protocol_experiment_summary",
        "complete": True,
        "raw_aggregate": {"path": str(args.raw_out.resolve()), "sha256": sha256_file(args.raw_out)},
        "configurations": {
            arm: {**prereg["configurations"][arm], "result": summaries[arm]}
            for arm in ("transferred_initializer", "projected_anchor", "single_best", "two_best")
        },
        "primary_comparison": primary,
        "secondary_comparisons": secondary,
        "multiplicity": {
            "primary_family": "singleton; no multiplicity adjustment",
            "secondary_family": ["single_best", "two_best"],
            "secondary_method": "Holm",
            "secondary_adjusted_p": adjusted,
        },
        "interpretation": (
            "Post-selection mechanism diagnostic only; it cannot revise the frozen pi0.5 choice. "
            "Non-significance does not establish equivalence. The transferred initializer violates "
            "the deployment byte budget."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, summary)
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "successes": {arm: summaries[arm]["successes"] for arm in summaries},
                "primary": primary,
                "secondary": secondary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
