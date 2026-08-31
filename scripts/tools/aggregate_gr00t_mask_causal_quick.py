#!/usr/bin/env python3
"""Aggregate the paired 50-episode GR00T mask causal quick test."""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np

from quantvla_cross_model_protocol import sha256_file
from quantvla_outputimpact import atomic_json


def load_rows(root: Path, config: str) -> dict[tuple[str, int], dict]:
    rows = {}
    for path in sorted(root.glob("**/*.jsonl")):
        if path.name.startswith(("gpu_efficiency", "gpu_server_efficiency")):
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config") != config:
                continue
            if row.get("status") != "complete":
                raise ValueError(f"{path}:{line_number}: incomplete quick episode")
            key = (str(row["task"]), int(row["seed"]))
            if key in rows:
                raise ValueError(f"duplicate quick key: {key}")
            rows[key] = row
    return rows


def rates(rows: dict[tuple[str, int], dict], task_to_split: dict[str, str]) -> dict:
    by_task = defaultdict(list)
    for (task, _seed), row in rows.items():
        by_task[task].append(float(bool(row["success"])))
    per_task = {task: float(np.mean(values)) for task, values in sorted(by_task.items())}
    by_split = defaultdict(list)
    for task, value in per_task.items():
        by_split[task_to_split[task]].append(value)
    return {
        "episodes": len(rows),
        "successes": sum(int(bool(row["success"])) for row in rows.values()),
        "micro_success_rate": float(np.mean([bool(row["success"]) for row in rows.values()])),
        "task_macro_success_rate": float(np.mean(list(per_task.values()))),
        "split_task_macro_success_rate": {
            split: float(np.mean(values)) for split, values in sorted(by_split.items())
        },
        "per_task_success_rate": per_task,
    }


def exact_mcnemar(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, float(Fraction(2 * probability, 2**discordant)))


def hierarchical_bootstrap(m0: dict, full_w4: dict, draws: int = 10000) -> dict:
    tasks = sorted({task for task, _seed in m0})
    seeds_by_task = {
        task: sorted(seed for candidate, seed in m0 if candidate == task) for task in tasks
    }
    deltas = {
        task: np.asarray(
            [
                float(bool(m0[(task, seed)]["success"]))
                - float(bool(full_w4[(task, seed)]["success"]))
                for seed in seeds_by_task[task]
            ],
            dtype=np.float64,
        )
        for task in tasks
    }
    generator = np.random.default_rng(0)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_tasks = generator.integers(0, len(tasks), size=len(tasks))
        task_values = []
        for task_index in sampled_tasks:
            task = tasks[int(task_index)]
            values_for_task = deltas[task]
            indices = generator.integers(0, len(values_for_task), size=len(values_for_task))
            task_values.append(float(values_for_task[indices].mean()))
        values[draw] = float(np.mean(task_values))
    return {
        "draws": draws,
        "seed": 0,
        "mean_delta": float(values.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--full-w4-dir", required=True)
    parser.add_argument("--m0-dir", required=True)
    parser.add_argument("--prior-attribution-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "gr00t_mask_causal_quick_preregistration":
        raise ValueError("wrong quick preregistration kind")
    task_to_split = {
        task: split
        for split, tasks in manifest["evaluation"]["tasks"].items()
        for task in tasks
    }
    expected = {
        (task, seed)
        for task in task_to_split
        for seed in manifest["evaluation"]["trial_seeds"]
    }
    full_w4 = load_rows(Path(args.full_w4_dir).expanduser().resolve(), "full_w4_dyrange")
    m0 = load_rows(Path(args.m0_dir).expanduser().resolve(), "full_context_v2")
    if set(full_w4) != expected or set(m0) != expected:
        raise ValueError(
            f"quick coverage drift: full_w4={len(full_w4)}, m0={len(m0)}, expected={len(expected)}"
        )

    full_rates = rates(full_w4, task_to_split)
    m0_rates = rates(m0, task_to_split)
    m0_wins = sum(
        int(bool(m0[key]["success"]) and not bool(full_w4[key]["success"]))
        for key in expected
    )
    m0_losses = sum(
        int(bool(full_w4[key]["success"]) and not bool(m0[key]["success"]))
        for key in expected
    )
    bootstrap = hierarchical_bootstrap(m0, full_w4)
    payload = {
        "schema_version": 1,
        "kind": "gr00t_mask_causal_quick_aggregate",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "diagnostic_only": True,
        "m0": m0_rates,
        "full_w4": full_rates,
        "effect_m0_minus_full_w4": {
            "micro_success_rate": float(
                m0_rates["micro_success_rate"] - full_rates["micro_success_rate"]
            ),
            "task_macro_success_rate": float(
                m0_rates["task_macro_success_rate"]
                - full_rates["task_macro_success_rate"]
            ),
            "paired_m0_wins": m0_wins,
            "paired_m0_losses": m0_losses,
            "paired_ties": len(expected) - m0_wins - m0_losses,
            "strict_pairwise_direction": (
                "m0_better" if m0_wins > m0_losses
                else "full_w4_better" if m0_losses > m0_wins
                else "tie"
            ),
            "exact_two_sided_mcnemar_p": exact_mcnemar(m0_wins, m0_losses),
            "task_then_seed_hierarchical_bootstrap": bootstrap,
        },
        "interpretation_scope": (
            "50-episode development diagnostic; direction and failure modes only, "
            "not a formal equivalence or superiority claim"
        ),
    }
    if args.prior_attribution_root:
        prior_root = Path(args.prior_attribution_root).expanduser().resolve()
        prior_m0 = load_rows(prior_root, "m")
        prior_full_w4 = load_rows(prior_root, "c")
        if set(prior_m0) != set(prior_full_w4) or len(prior_m0) != 50:
            raise ValueError(
                f"prior M/C attribution coverage drift: m={len(prior_m0)}, c={len(prior_full_w4)}"
            )
        if set(prior_m0) & set(m0):
            raise ValueError("prior and new quick waves must have disjoint task/seed keys")
        prior_task_to_split = {task: "prior_wave" for task, _seed in prior_m0}
        prior_m0_rates = rates(prior_m0, prior_task_to_split)
        prior_full_rates = rates(prior_full_w4, prior_task_to_split)
        prior_m0_wins = sum(
            int(bool(prior_m0[key]["success"]) and not bool(prior_full_w4[key]["success"]))
            for key in prior_m0
        )
        prior_m0_losses = sum(
            int(bool(prior_full_w4[key]["success"]) and not bool(prior_m0[key]["success"]))
            for key in prior_m0
        )
        prior_bootstrap = hierarchical_bootstrap(prior_m0, prior_full_w4)
        pooled_m0 = {**prior_m0, **m0}
        pooled_full_w4 = {**prior_full_w4, **full_w4}
        pooled_task_to_split = {
            **prior_task_to_split,
            **task_to_split,
        }
        pooled_m0_rates = rates(pooled_m0, pooled_task_to_split)
        pooled_full_rates = rates(pooled_full_w4, pooled_task_to_split)
        pooled_m0_wins = prior_m0_wins + m0_wins
        pooled_m0_losses = prior_m0_losses + m0_losses
        payload["independent_prior_wave"] = {
            "posthoc_context_only": True,
            "root": str(prior_root),
            "m0": prior_m0_rates,
            "full_w4": prior_full_rates,
            "effect_m0_minus_full_w4": {
                "micro_success_rate": float(
                    prior_m0_rates["micro_success_rate"]
                    - prior_full_rates["micro_success_rate"]
                ),
                "task_macro_success_rate": float(
                    prior_m0_rates["task_macro_success_rate"]
                    - prior_full_rates["task_macro_success_rate"]
                ),
                "paired_m0_wins": prior_m0_wins,
                "paired_m0_losses": prior_m0_losses,
                "paired_ties": 50 - prior_m0_wins - prior_m0_losses,
                "exact_two_sided_mcnemar_p": exact_mcnemar(
                    prior_m0_wins, prior_m0_losses
                ),
                "task_then_seed_hierarchical_bootstrap": prior_bootstrap,
            },
        }
        payload["pooled_exploratory_two_wave_diagnostic"] = {
            "posthoc_not_preregistered": True,
            "episodes": 100,
            "tasks": 10,
            "m0": pooled_m0_rates,
            "full_w4": pooled_full_rates,
            "effect_m0_minus_full_w4": {
                "micro_success_rate": float(
                    pooled_m0_rates["micro_success_rate"]
                    - pooled_full_rates["micro_success_rate"]
                ),
                "task_macro_success_rate": float(
                    pooled_m0_rates["task_macro_success_rate"]
                    - pooled_full_rates["task_macro_success_rate"]
                ),
                "paired_m0_wins": pooled_m0_wins,
                "paired_m0_losses": pooled_m0_losses,
                "paired_ties": 100 - pooled_m0_wins - pooled_m0_losses,
                "exact_two_sided_mcnemar_p": exact_mcnemar(
                    pooled_m0_wins, pooled_m0_losses
                ),
                "task_then_seed_hierarchical_bootstrap": hierarchical_bootstrap(
                    pooled_m0, pooled_full_w4
                ),
            },
            "direction_replicated": bool(
                (m0_wins - m0_losses) * (prior_m0_wins - prior_m0_losses) > 0
            ),
        }
    atomic_json(args.out, payload)
    print(json.dumps({"out": str(Path(args.out).resolve()), "m0_wins": m0_wins, "m0_losses": m0_losses}, indent=2))


if __name__ == "__main__":
    main()
