#!/usr/bin/env python3
"""Aggregate the paired 50-task by 10-seed GR00T Table-3 ablations."""

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
from quantvla_outputimpact import atomic_json


ARMS = ("static_a8", "local_mse_selection", "no_fullnet_check")


def load_rows(
    root: Path, config: str, expected: set[tuple[str, int]]
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    sources = []
    for path in sorted(root.glob("**/*.jsonl")):
        if path.name.startswith(("gpu_efficiency", "gpu_server_efficiency")):
            continue
        used = 0
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config") != config:
                continue
            key = (str(row["task"]), int(row["seed"]))
            if key not in expected:
                continue
            if row.get("status") != "complete":
                raise ValueError(f"{path}:{line_number}: incomplete episode")
            if key in rows:
                raise ValueError(f"duplicate {config} task-seed pair: {key}")
            if not bool(row.get("paired_action_noise", False)):
                raise ValueError(f"{path}:{line_number}: unpaired action noise")
            if int(row.get("flow_steps", -1)) != 4:
                raise ValueError(f"{path}:{line_number}: flow-step drift")
            if int(row.get("action_horizon", -1)) != 16:
                raise ValueError(f"{path}:{line_number}: action-horizon drift")
            rows[key] = row
            used += 1
        if used:
            sources.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "episodes": used,
                }
            )
    missing = sorted(expected - set(rows))
    extra = sorted(set(rows) - expected)
    if missing or extra:
        raise ValueError(
            f"{config} coverage drift: observed={len(rows)} expected={len(expected)} "
            f"missing={len(missing)} extra={len(extra)}"
        )
    return rows, sources


def rates(
    rows: dict[tuple[str, int], dict[str, Any]], task_to_split: dict[str, str]
) -> dict[str, Any]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for (task, _seed), row in rows.items():
        by_task[task].append(float(bool(row["success"])))
    per_task = {
        task: float(np.mean(values)) for task, values in sorted(by_task.items())
    }
    by_split: dict[str, list[float]] = defaultdict(list)
    for task, value in per_task.items():
        by_split[task_to_split[task]].append(value)
    values = [float(bool(row["success"])) for row in rows.values()]
    return {
        "episodes": len(rows),
        "successes": int(sum(values)),
        "micro_success_rate": float(np.mean(values)),
        "task_macro_success_rate": float(np.mean(list(per_task.values()))),
        "split_task_macro_success_rate": {
            split: float(np.mean(split_values))
            for split, split_values in sorted(by_split.items())
        },
        "per_task_success_rate": per_task,
    }


def exact_mcnemar(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    numerator = 2 * sum(
        math.comb(discordant, index) for index in range(tail + 1)
    )
    return min(1.0, float(Fraction(numerator, 2**discordant)))


def hierarchical_bootstrap(
    candidate: dict[tuple[str, int], dict[str, Any]],
    baseline: dict[tuple[str, int], dict[str, Any]],
    *,
    draws: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    tasks = sorted({task for task, _seed in baseline})
    seeds_by_task = {
        task: sorted(seed_value for candidate_task, seed_value in baseline if candidate_task == task)
        for task in tasks
    }
    deltas = {
        task: np.asarray(
            [
                float(bool(candidate[(task, seed_value)]["success"]))
                - float(bool(baseline[(task, seed_value)]["success"]))
                for seed_value in seeds_by_task[task]
            ],
            dtype=np.float64,
        )
        for task in tasks
    }
    generator = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        task_indices = generator.integers(0, len(tasks), size=len(tasks))
        task_values = []
        for task_index in task_indices:
            task = tasks[int(task_index)]
            task_delta = deltas[task]
            seed_indices = generator.integers(
                0, len(task_delta), size=len(task_delta)
            )
            task_values.append(float(np.mean(task_delta[seed_indices])))
        samples[draw] = float(np.mean(task_values))
    observed = float(np.mean([np.mean(value) for value in deltas.values()]))
    return {
        "draws": draws,
        "seed": seed,
        "observed_task_macro_delta": observed,
        "bootstrap_mean_delta": float(np.mean(samples)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda key: (raw[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        value = min(1.0, (total - index) * raw[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "gr00t_table3_reduced_closed_loop_preregistration":
        raise ValueError("unexpected Table-3 preregistration kind")
    seeds = [int(value) for value in manifest["evaluation"]["trial_seeds"]]
    task_sets = manifest["evaluation"]["task_sets"]
    task_to_split = {
        task: split for split, tasks in task_sets.items() for task in tasks
    }
    expected = {
        (task, seed)
        for task in task_to_split
        for seed in seeds
    }
    if len(expected) != 500:
        raise ValueError(f"Table-3 expected-pair drift: {len(expected)}")

    baseline, baseline_sources = load_rows(
        Path(args.baseline), "full_context_v2", expected
    )
    baseline_rates = rates(baseline, task_to_split)
    arm_rows = {}
    arm_sources = {}
    arm_rates = {}
    comparisons = {}
    raw_p = {}
    for arm in ARMS:
        rows, sources = load_rows(Path(args.results) / arm, arm, expected)
        arm_rows[arm] = rows
        arm_sources[arm] = sources
        arm_rates[arm] = rates(rows, task_to_split)
        wins = sum(
            int(bool(rows[key]["success"]) and not bool(baseline[key]["success"]))
            for key in expected
        )
        losses = sum(
            int(not bool(rows[key]["success"]) and bool(baseline[key]["success"]))
            for key in expected
        )
        ties = len(expected) - wins - losses
        p_value = exact_mcnemar(wins, losses)
        raw_p[arm] = p_value
        comparisons[arm] = {
            "orientation": f"{arm} minus full_context_v2",
            "task_macro_delta": arm_rates[arm]["task_macro_success_rate"]
            - baseline_rates["task_macro_success_rate"],
            "micro_delta": arm_rates[arm]["micro_success_rate"]
            - baseline_rates["micro_success_rate"],
            "paired_candidate_wins": wins,
            "paired_candidate_losses": losses,
            "paired_ties": ties,
            "discordant_pairs": wins + losses,
            "exact_mcnemar_p": p_value,
            "hierarchical_bootstrap": hierarchical_bootstrap(rows, baseline),
        }
    adjusted = holm_adjust(raw_p)
    for arm in ARMS:
        comparisons[arm]["holm_adjusted_p"] = adjusted[arm]

    payload = {
        "schema_version": 1,
        "kind": "gr00t_table3_reduced_closed_loop_aggregate",
        "complete": True,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "coverage": {
            "tasks": len(task_to_split),
            "seeds_per_task": len(seeds),
            "episodes_per_config": len(expected),
            "new_episode_configs": len(ARMS),
            "new_episodes": len(expected) * len(ARMS),
        },
        "baseline": baseline_rates,
        "arms": arm_rates,
        "comparisons": comparisons,
        "multiplicity": {
            "family": list(ARMS),
            "method": "Holm",
            "raw_p": raw_p,
            "adjusted_p": adjusted,
        },
        "sources": {
            "baseline": baseline_sources,
            **arm_sources,
        },
        "reporting_scope": (
            "Reduced 10-seed component ablation. Estimates are paired and "
            "directional; absence of significance is not an equivalence claim."
        ),
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "out": str(output),
                "baseline_sr": baseline_rates["task_macro_success_rate"],
                "arms": {
                    arm: {
                        "sr": arm_rates[arm]["task_macro_success_rate"],
                        "delta": comparisons[arm]["task_macro_delta"],
                        "ci": [
                            comparisons[arm]["hierarchical_bootstrap"]["ci95_low"],
                            comparisons[arm]["hierarchical_bootstrap"]["ci95_high"],
                        ],
                        "p_holm": comparisons[arm]["holm_adjusted_p"],
                    }
                    for arm in ARMS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
