#!/usr/bin/env python3
"""Bootstrap and leave-one-task-out audit of the frozen GR00T ratio sweep."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = (
    REPO_ROOT / "runs/robocasa365_cs_tuning_official_core",
    REPO_ROOT / "runs/robocasa365_cs_tuning_large_ratio_official",
    REPO_ROOT / "runs/robocasa365_cs_tuning_cka_boundary_gpu1",
)
CONFIG_ORDER = (
    "cscka_8to1",
    "cscka_16to1",
    "cscka_32to1",
    "cscka_64to1",
    "cscka_128to1",
    "cscka_256to1",
    "ckaonly",
)
DEVELOPMENT_TASKS = (
    "CoffeeSetupMug",
    "OpenCabinet",
    "OpenStandMixerHead",
    "PickPlaceDrawerToCounter",
)
SEEDS = tuple(range(50))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty sequence")
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_rows(roots: list[Path]) -> tuple[dict[str, dict[tuple[str, int], bool]], list[dict[str, Any]]]:
    grouped = {config: {} for config in CONFIG_ORDER}
    sources: list[dict[str, Any]] = []
    for root in roots:
        manifest = root / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        sources.append({"path": str(manifest), "sha256": sha256_file(manifest)})
        for path in sorted(root.glob("*.jsonl")):
            if path.name.startswith("gpu_"):
                continue
            file_configs: set[str] = set()
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: malformed JSON") from error
                config = str(row.get("config"))
                if config not in grouped:
                    continue
                task, seed = str(row.get("task")), int(row.get("seed", -1))
                if task not in DEVELOPMENT_TASKS or seed not in SEEDS:
                    continue
                if row.get("success") is None or row.get("crashed") is True:
                    raise ValueError(f"{path}:{line_number}: invalid formal result")
                key = (task, seed)
                if key in grouped[config]:
                    raise ValueError(f"duplicate {config} result {key}")
                grouped[config][key] = bool(row["success"])
                file_configs.add(config)
            if file_configs:
                sources.append(
                    {"path": str(path), "sha256": sha256_file(path), "configs": sorted(file_configs)}
                )
    expected = {(task, seed) for task in DEVELOPMENT_TASKS for seed in SEEDS}
    for config, rows in grouped.items():
        missing = sorted(expected - set(rows))
        unexpected = sorted(set(rows) - expected)
        if missing or unexpected:
            raise ValueError(
                f"{config}: coverage mismatch missing={len(missing)} unexpected={len(unexpected)}"
            )
    return grouped, sources


def macro(
    rows: dict[tuple[str, int], bool], tasks: list[str], seeds_by_task: dict[str, list[int]]
) -> float:
    return sum(
        sum(rows[(task, seed)] for seed in seeds_by_task[task]) / len(seeds_by_task[task])
        for task in tasks
    ) / len(tasks)


def paired_hierarchical_bootstrap(
    rows: dict[str, dict[tuple[str, int], bool]], n_boot: int, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    estimates = {config: [] for config in CONFIG_ORDER}
    deltas = {config: [] for config in CONFIG_ORDER if config != "cscka_16to1"}
    winner_credit = {config: 0.0 for config in CONFIG_ORDER}
    winner_sets: defaultdict[str, int] = defaultdict(int)
    for _ in range(n_boot):
        sampled_tasks = [rng.choice(DEVELOPMENT_TASKS) for _ in DEVELOPMENT_TASKS]
        # Duplicate task draws must retain independent seed resamples.
        per_config_values = {config: [] for config in CONFIG_ORDER}
        for task in sampled_tasks:
            sampled_seeds = [rng.choice(SEEDS) for _ in SEEDS]
            for config in CONFIG_ORDER:
                per_config_values[config].append(
                    sum(rows[config][(task, seed)] for seed in sampled_seeds)
                    / len(sampled_seeds)
                )
        scores = {
            config: sum(values) / len(values)
            for config, values in per_config_values.items()
        }
        for config, value in scores.items():
            estimates[config].append(value)
            if config != "cscka_16to1":
                deltas[config].append(value - scores["cscka_16to1"])
        best = max(scores.values())
        winners = sorted(config for config, value in scores.items() if abs(value - best) < 1e-12)
        for config in winners:
            winner_credit[config] += 1.0 / len(winners)
        winner_sets["+".join(winners)] += 1
    return {
        "draws": n_boot,
        "seed": seed,
        "resampling_unit": "tasks with replacement, then paired seeds within each sampled task",
        "configs": {
            config: {
                "bootstrap_mean": sum(values) / len(values),
                "ci95": [percentile(values, 0.025), percentile(values, 0.975)],
                "fractional_best_probability": winner_credit[config] / n_boot,
                **(
                    {
                        "delta_vs_16to1_ci95": [
                            percentile(deltas[config], 0.025),
                            percentile(deltas[config], 0.975),
                        ],
                        "delta_vs_16to1_mean": sum(deltas[config]) / len(deltas[config]),
                    }
                    if config != "cscka_16to1"
                    else {"delta_vs_16to1_ci95": [0.0, 0.0], "delta_vs_16to1_mean": 0.0}
                ),
            }
            for config, values in estimates.items()
        },
        "most_common_winner_sets": [
            {"configs": name.split("+"), "draws": count, "fraction": count / n_boot}
            for name, count in sorted(winner_sets.items(), key=lambda row: (-row[1], row[0]))[:10]
        ],
    }


def analyze(roots: list[Path], n_boot: int) -> dict[str, Any]:
    rows, sources = load_rows(roots)
    observed: dict[str, Any] = {}
    for config in CONFIG_ORDER:
        per_task = {
            task: sum(rows[config][(task, seed)] for seed in SEEDS) / len(SEEDS)
            for task in DEVELOPMENT_TASKS
        }
        observed[config] = {
            "successes": sum(rows[config].values()),
            "episodes": len(rows[config]),
            "per_task_sr": per_task,
            "task_macro_sr": sum(per_task.values()) / len(per_task),
        }
    leave_one_out = {}
    for omitted in DEVELOPMENT_TASKS:
        kept = [task for task in DEVELOPMENT_TASKS if task != omitted]
        scores = {
            config: sum(observed[config]["per_task_sr"][task] for task in kept) / len(kept)
            for config in CONFIG_ORDER
        }
        best = max(scores.values())
        leave_one_out[omitted] = {
            "task_macro_sr": scores,
            "winners": [config for config, value in scores.items() if abs(value - best) < 1e-12],
        }
    bootstrap = paired_hierarchical_bootstrap(rows, n_boot, 20260823)
    plateau = [
        config
        for config, row in bootstrap["configs"].items()
        if row["delta_vs_16to1_ci95"][0] <= 0.0 <= row["delta_vs_16to1_ci95"][1]
    ]
    return {
        "schema_version": 1,
        "kind": "gdsq_ratio_stability_audit",
        "complete": True,
        "coverage": {
            "tasks": len(DEVELOPMENT_TASKS),
            "seeds_per_task": len(SEEDS),
            "episodes_per_config": len(DEVELOPMENT_TASKS) * len(SEEDS),
            "configs": len(CONFIG_ORDER),
            "missing": 0,
            "duplicates": 0,
        },
        "development_tasks": list(DEVELOPMENT_TASKS),
        "observed": observed,
        "hierarchical_bootstrap": bootstrap,
        "leave_one_task_out": leave_one_out,
        "wide_plateau_vs_16to1": plateau,
        "interpretation_policy": (
            "Treat 16:1 as stable only if it remains a leave-one-task-out winner and "
            "its paired hierarchical-bootstrap advantage excludes zero; otherwise describe "
            "a broad development plateau."
        ),
        "sources": sources,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# CKA:CS ratio stability audit",
        "",
        "All seven configurations have 4 development tasks × 50 paired seeds. Results below are diagnostic selection evidence, not held-out test results.",
        "",
        "| Config | Observed macro SR | Bootstrap 95% CI | Δ vs 16:1 95% CI | Fractional P(best) |",
        "|---|---:|---:|---:|---:|",
    ]
    for config in CONFIG_ORDER:
        observed = report["observed"][config]["task_macro_sr"]
        row = report["hierarchical_bootstrap"]["configs"][config]
        ci, delta = row["ci95"], row["delta_vs_16to1_ci95"]
        lines.append(
            f"| {config} | {observed:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"[{delta[0]:+.3f}, {delta[1]:+.3f}] | {row['fractional_best_probability']:.3f} |"
        )
    lines += ["", "## Leave-one-task-out winners", ""]
    for task, row in report["leave_one_task_out"].items():
        lines.append(f"- Omit `{task}`: {', '.join(row['winners'])}")
    lines += [
        "",
        "Configurations whose paired Δ-vs-16:1 interval includes zero: "
        + ", ".join(report["wide_plateau_vs_16to1"])
        + ".",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(value).resolve() for value in args.root] or list(DEFAULT_ROOTS)
    report = analyze(roots, args.bootstrap)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        write_markdown(Path(args.markdown_out).resolve(), report)
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "coverage": report["coverage"],
                "wide_plateau_vs_16to1": report["wide_plateau_vs_16to1"],
                "out": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
