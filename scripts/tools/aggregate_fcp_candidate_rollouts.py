#!/usr/bin/env python3
"""Aggregate paired seeds 0--9 closed-loop diagnostics for six FCP candidates."""

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


def load_rows(
    root: Path, config: str, expected: set[tuple[str, int]]
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/*.jsonl")):
        if path.name.startswith(("gpu_efficiency", "gpu_server_efficiency")):
            continue
        used = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
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
                {"path": str(path.resolve()), "sha256": sha256_file(path), "episodes": used}
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
    per_task = {task: float(np.mean(values)) for task, values in sorted(by_task.items())}
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
    numerator = 2 * sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, float(Fraction(numerator, 2**discordant)))


def hierarchical_bootstrap(
    candidate: dict[tuple[str, int], dict[str, Any]],
    baseline: dict[tuple[str, int], dict[str, Any]],
    *,
    draws: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    tasks = sorted({task for task, _ in baseline})
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
        sampled_tasks = generator.integers(0, len(tasks), size=len(tasks))
        task_values = []
        for task_index in sampled_tasks:
            task = tasks[int(task_index)]
            values = deltas[task]
            sampled_seeds = generator.integers(0, len(values), size=len(values))
            task_values.append(float(np.mean(values[sampled_seeds])))
        samples[draw] = float(np.mean(task_values))
    observed = float(np.mean([np.mean(values) for values in deltas.values()]))
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
        running = max(running, min(1.0, (total - index) * raw[key]))
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
    if manifest.get("kind") != "gr00t_fcp_candidate_reduced_closed_loop_preregistration":
        raise ValueError("unexpected FCP closed-loop preregistration kind")
    seeds = [int(value) for value in manifest["evaluation"]["trial_seeds"]]
    candidate_ids = list(manifest["evaluation"]["candidate_ids"])
    task_sets = manifest["evaluation"]["task_sets"]
    task_to_split = {task: split for split, tasks in task_sets.items() for task in tasks}
    expected = {(task, seed) for task in task_to_split for seed in seeds}
    if len(expected) != 500 or len(candidate_ids) != 6:
        raise ValueError("FCP aggregate coverage contract drift")

    baseline, baseline_sources = load_rows(Path(args.baseline), "full_context_v2", expected)
    baseline_rates = rates(baseline, task_to_split)
    candidate_rates: dict[str, Any] = {}
    candidate_sources: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for candidate_id in candidate_ids:
        rows, sources = load_rows(Path(args.results) / candidate_id, candidate_id, expected)
        candidate_sources[candidate_id] = sources
        candidate_rates[candidate_id] = rates(rows, task_to_split)
        wins = sum(
            int(bool(rows[key]["success"]) and not bool(baseline[key]["success"]))
            for key in expected
        )
        losses = sum(
            int(not bool(rows[key]["success"]) and bool(baseline[key]["success"]))
            for key in expected
        )
        p_value = exact_mcnemar(wins, losses)
        raw_p[candidate_id] = p_value
        comparisons[candidate_id] = {
            "orientation": f"{candidate_id} minus full_context_v2",
            "task_macro_delta": candidate_rates[candidate_id]["task_macro_success_rate"]
            - baseline_rates["task_macro_success_rate"],
            "micro_delta": candidate_rates[candidate_id]["micro_success_rate"]
            - baseline_rates["micro_success_rate"],
            "paired_candidate_wins": wins,
            "paired_candidate_losses": losses,
            "paired_ties": len(expected) - wins - losses,
            "discordant_pairs": wins + losses,
            "exact_mcnemar_p": p_value,
            "hierarchical_bootstrap": hierarchical_bootstrap(rows, baseline),
        }
    adjusted = holm_adjust(raw_p)
    for candidate_id in candidate_ids:
        comparisons[candidate_id]["holm_adjusted_p"] = adjusted[candidate_id]

    payload = {
        "schema_version": 1,
        "kind": "gr00t_fcp_candidate_reduced_closed_loop_aggregate",
        "complete": True,
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "coverage": {
            "tasks": len(task_to_split),
            "seeds_per_task": len(seeds),
            "episodes_per_config": len(expected),
            "new_episode_configs": len(candidate_ids),
            "new_episodes": len(expected) * len(candidate_ids),
        },
        "baseline": baseline_rates,
        "candidates": candidate_rates,
        "comparisons": comparisons,
        "multiplicity": {
            "family": candidate_ids,
            "method": "Holm",
            "raw_p": raw_p,
            "adjusted_p": adjusted,
        },
        "sources": {"baseline": baseline_sources, **candidate_sources},
        "reporting_scope": (
            "Post-selection 10-seed diagnostic only. Candidate success labels did not "
            "enter the already frozen FCP decision; non-significance is not equivalence."
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
                "candidates": {
                    candidate_id: {
                        "sr": candidate_rates[candidate_id]["task_macro_success_rate"],
                        "delta": comparisons[candidate_id]["task_macro_delta"],
                        "ci": [
                            comparisons[candidate_id]["hierarchical_bootstrap"]["ci95_low"],
                            comparisons[candidate_id]["hierarchical_bootstrap"]["ci95_high"],
                        ],
                        "p_holm": comparisons[candidate_id]["holm_adjusted_p"],
                    }
                    for candidate_id in candidate_ids
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
