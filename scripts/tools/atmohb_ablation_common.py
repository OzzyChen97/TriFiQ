#!/usr/bin/env python3
"""Shared paired-ablation statistics for ATM/OHB component ablations.

Reuses the frozen statistical protocol of parse_robocasa_atomic_matrix:
task-macro success rate, 10,000-draw task-cluster bootstrap CI, exact paired
sign-flip p-values, Holm correction over predeclared contrasts, and an
episode-level McNemar check.  Tasks (not episodes) are the unit of
generalization, matching the paper protocol.
"""

from __future__ import annotations

import random
from typing import Any

from parse_robocasa_atomic_matrix import (
    cluster_ci,
    exact_sign_flip_p,
    holm_adjust,
    mcnemar,
    paired_delta_ci,
)

SEED_RNG = 20260817
N_BOOT = 10_000


def build_summary(
    rows_by_config: dict[str, dict[tuple[str, int], dict]],
    tasks: list[str],
    seeds: list[int],
    contrasts: list[tuple[str, str]],
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    for config_id, rows in rows_by_config.items():
        missing = {(task, seed) for task in tasks for seed in seeds} - set(rows)
        if missing:
            raise ValueError(
                f"{config_id}: missing {len(missing)} task/seed rows, e.g. {sorted(missing)[:5]}"
            )
        extra = set(rows) - {(task, seed) for task in tasks for seed in seeds}
        if extra:
            raise ValueError(f"{config_id}: unexpected task/seed rows {sorted(extra)[:5]}")

    rng = random.Random(SEED_RNG)
    configs: dict[str, dict[str, Any]] = {}
    for config_id, rows in rows_by_config.items():
        rates = {
            task: sum(bool(rows[(task, seed)]["success"]) for seed in seeds) / len(seeds)
            for task in tasks
        }
        per_task = {}
        for task in tasks:
            successes = sum(bool(rows[(task, seed)]["success"]) for seed in seeds)
            per_task[task] = {
                "sr": rates[task],
                "successes": successes,
                "episodes": len(seeds),
            }
        configs[config_id] = {
            "per_task_sr": rates,
            "per_task": per_task,
            "task_macro_sr": sum(rates.values()) / len(rates),
            "task_cluster_ci95": cluster_ci(rates, n_boot, rng),
            "episode_sr": sum(bool(row["success"]) for row in rows.values()) / len(rows),
            "successes": sum(bool(row["success"]) for row in rows.values()),
            "episodes": len(rows),
        }

    comparisons: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float] = {}
    for a, b in contrasts:
        if a not in configs or b not in configs:
            raise ValueError(f"contrast references missing config: {a} vs {b}")
        rates_a = {task: configs[a]["per_task_sr"][task] for task in tasks}
        rates_b = {task: configs[b]["per_task_sr"][task] for task in tasks}
        delta, ci, diffs = paired_delta_ci(rates_a, rates_b, n_boot, rng)
        name = f"{a}_vs_{b}"
        p = exact_sign_flip_p(diffs)
        raw_p[name] = p
        comparisons[name] = {
            "a": a,
            "b": b,
            "task_macro_delta": delta,
            "task_cluster_ci95": ci,
            "paired_sign_flip_p": p,
            "episode_mcnemar": mcnemar(rows_by_config[a], rows_by_config[b], tasks, seeds),
        }
    adjusted = holm_adjust(raw_p)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value

    return {
        "tasks": tasks,
        "seeds": seeds,
        "bootstrap_draws": n_boot,
        "rng_seed": SEED_RNG,
        "configs": configs,
        "comparisons": comparisons,
    }


def write_markdown(summary: dict[str, Any], title: str) -> str:
    config_ids = list(summary["configs"])
    lines = [f"# {title}", ""]
    lines += [
        "| Config | Task-macro SR | 95% task-cluster CI | Episode SR | Successes |",
        "|---|---:|---:|---:|---:|",
    ]
    for config_id, row in summary["configs"].items():
        ci = row["task_cluster_ci95"]
        lines.append(
            f"| {config_id} | {row['task_macro_sr']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | {row['episode_sr']:.4f} | "
            f"{row['successes']}/{row['episodes']} |"
        )
    lines += ["", "## Per-task success rate", "",
              "| Task | " + " | ".join(config_ids) + " |",
              "|---|" + "---:|" * len(config_ids)]
    for task in summary["tasks"]:
        values = [summary["configs"][cid]["per_task"][task]["sr"] for cid in config_ids]
        lines.append("| " + task + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "## Prespecified paired contrasts", "",
              "| Contrast | Delta | 95% task-cluster CI | Sign-flip p | Holm p | McNemar (b,c) |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, row in summary["comparisons"].items():
        ci = row["task_cluster_ci95"]
        mc = row["episode_mcnemar"]
        lines.append(
            f"| {name} | {row['task_macro_delta']:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {row['paired_sign_flip_p']:.4f} | "
            f"{row['holm_adjusted_p']:.4f} | ({mc['a_wins']},{mc['b_wins']}) |"
        )
    lines += [
        "",
        "Statistics treat the 18 tasks as the unit of generalization: task-macro "
        "SR, 10,000-draw task-cluster bootstrap CI, exact paired sign-flip test, "
        "and Holm correction over the predeclared contrasts.",
        "",
    ]
    return "\n".join(lines)