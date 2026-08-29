#!/usr/bin/env python3
"""Aggregate paired RoboCasa365 Table-1 rows with registered statistics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from quantvla_cross_model_protocol import sha256_file
from quantvla_full_context import PROTOCOL, protocol_attestation, require_protocol_attestation
from quantvla_outputimpact import atomic_json


def named_dir(value: str) -> tuple[str, Path]:
    identifier, separator, raw = value.partition("=")
    if not separator or not identifier or not raw:
        raise argparse.ArgumentTypeError("baseline must be ID=DIRECTORY")
    return identifier, Path(raw).expanduser().resolve()


def load_rows(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    for path in sorted(root.glob("**/*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "complete":
                raise ValueError(f"{path}:{line_number}: incomplete formal episode")
            key = (str(row["task"]), int(row["seed"]))
            if key in rows:
                raise ValueError(f"{root}: duplicate formal key {key}")
            rows[key] = row
    expected = {
        (task, seed)
        for tasks in PROTOCOL["table1"]["tasks"].values()
        for task in tasks
        for seed in range(50)
    }
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise ValueError(f"{root}: formal coverage drift missing={missing[:3]} extra={extra[:3]}")
    return rows


def rates(rows: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    task_success: dict[str, list[float]] = defaultdict(list)
    for (task, _seed), row in rows.items():
        task_success[task].append(float(bool(row["success"])))
    per_task = {task: float(np.mean(values)) for task, values in sorted(task_success.items())}
    split_macro = {
        split: float(np.mean([per_task[task] for task in tasks]))
        for split, tasks in PROTOCOL["table1"]["tasks"].items()
    }
    return {
        "episodes": len(rows),
        "successes": sum(int(bool(row["success"])) for row in rows.values()),
        "micro_success_rate": float(np.mean([bool(row["success"]) for row in rows.values()])),
        "task_macro_success_rate": float(np.mean(list(per_task.values()))),
        "split_task_macro_success_rate": split_macro,
        "per_task_success_rate": per_task,
    }


def exact_mcnemar(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, float(Fraction(2 * probability, 2 ** discordant)))


def hierarchical_bootstrap(
    candidate: dict[tuple[str, int], dict[str, Any]],
    baseline: dict[tuple[str, int], dict[str, Any]],
    *,
    draws: int = 10000,
) -> dict[str, float]:
    tasks = sorted({task for task, _seed in candidate})
    deltas = np.asarray(
        [
            [
                float(bool(candidate[(task, seed)]["success"]))
                - float(bool(baseline[(task, seed)]["success"]))
                for seed in range(50)
            ]
            for task in tasks
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(0)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        task_indices = generator.integers(0, len(tasks), size=len(tasks))
        sampled = []
        for task_index in task_indices:
            seed_indices = generator.integers(0, 50, size=50)
            sampled.append(float(deltas[task_index, seed_indices].mean()))
        values[draw] = float(np.mean(sampled))
    return {
        "draws": draws,
        "seed": 0,
        "mean_delta": float(values.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (count - rank) * pvalues[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--baseline", action="append", type=named_dir, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_protocol_attestation(manifest, source=str(manifest_path))
    candidate = load_rows(Path(args.candidate_dir).expanduser().resolve())
    baselines = {identifier: load_rows(path) for identifier, path in args.baseline}
    if "gdsq_vla_main" not in baselines:
        raise ValueError("formal aggregation requires gdsq_vla_main")

    comparisons = {}
    raw_pvalues = {}
    candidate_rates = rates(candidate)
    for identifier, baseline in baselines.items():
        wins = sum(
            int(bool(candidate[key]["success"]) and not bool(baseline[key]["success"]))
            for key in candidate
        )
        losses = sum(
            int(bool(baseline[key]["success"]) and not bool(candidate[key]["success"]))
            for key in candidate
        )
        pvalue = exact_mcnemar(wins, losses)
        raw_pvalues[identifier] = pvalue
        comparisons[identifier] = {
            "baseline": rates(baseline),
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": len(candidate) - wins - losses,
            "exact_two_sided_mcnemar_p": pvalue,
            "task_then_seed_hierarchical_bootstrap": hierarchical_bootstrap(candidate, baseline),
        }
    adjusted = holm_adjust(raw_pvalues)
    for identifier, value in adjusted.items():
        comparisons[identifier]["holm_adjusted_mcnemar_p"] = value

    main = comparisons["gdsq_vla_main"]
    main_rates = main["baseline"]
    bootstrap = main["task_then_seed_hierarchical_bootstrap"]
    formal_win = bool(
        candidate_rates["task_macro_success_rate"] > main_rates["task_macro_success_rate"]
        and candidate_rates["micro_success_rate"] > main_rates["micro_success_rate"]
        and (
            bootstrap["ci95_low"] > 0.0
            or main["holm_adjusted_mcnemar_p"] < 0.05
        )
    )
    payload = {
        "schema_version": 1,
        "kind": "full_context_table1_aggregate",
        "full_context_protocol": protocol_attestation(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "candidate": candidate_rates,
        "comparisons": comparisons,
        "formal_superiority_over_gdsq_main": formal_win,
        "claim_status": "formal_superiority" if formal_win else "numerical_or_inconclusive",
        "selection_feedback_allowed": False,
    }
    atomic_json(args.out, payload)
    print(json.dumps({"out": str(Path(args.out).resolve()), "formal_win": formal_win}, indent=2))


if __name__ == "__main__":
    main()
