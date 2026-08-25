#!/usr/bin/env python3
"""Compute the five preregistered GR00T component-ablation contrasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from parse_robocasa_atomic_matrix import exact_sign_flip_p, holm_adjust, paired_delta_ci


REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTION = REPO_ROOT / "runs/gdsq_week1_preregistered_v1/execution"
STATIC = REPO_ROOT / "runs/robocasa365_official_full_paired50/summary.json"
STATIC_SHA256 = "243162a79a57fa9bf85402364e2cb6bf50c69baa5320fc2256e5821dfd0ef810"
ABLATION = EXECUTION / "runs/gr00t_core_ablations_primary14/summary.json"
OUTPUT = EXECUTION / "aggregate/gr00t_core_ablation_statistics.json"
DEV4 = {"CoffeeSetupMug", "OpenCabinet", "OpenStandMixerHead", "PickPlaceDrawerToCounter"}
CONFIGS = {
    "cka_only": "ablation_cka_only",
    "cs_only": "ablation_cs_only",
    "weights_uniform": "ablation_weights_uniform",
    "no_guards": "ablation_no_guards",
    "no_functional_adjudication": "ablation_no_functional_adjudication",
}


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
    require(path.is_file(), f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def rates(row: dict[str, Any]) -> dict[str, float]:
    return {str(task): float(value) for task, value in row["per_task"].items()}


def build() -> dict[str, Any]:
    require(sha256_file(STATIC) == STATIC_SHA256, "static GR00T summary drift")
    static, ablation = load(STATIC), load(ABLATION)
    require(ablation.get("bootstrap_draws") == 10_000, "ablation bootstrap drift")
    require(not ablation.get("validation_errors"), "ablation matrix has validation errors")
    require(ablation.get("primary_scope", {}).get("n_tasks") == 14, "Primary14 drift")
    require(set(ablation.get("configs", {})) == set(CONFIGS.values()), "ablation set drift")
    baseline_all = rates(static["configs"]["cscka_final"])
    atomic_tasks = set(static["task_sets"]["atomic_seen"])
    baseline = {
        task: value
        for task, value in baseline_all.items()
        if task in atomic_tasks and task not in DEV4
    }
    formal_tasks = set(ablation["primary_scope"]["tasks"])
    require(set(baseline) == formal_tasks and len(baseline) == 14, "baseline task scope drift")

    rng = random.Random(20260823)
    contrasts: dict[str, dict[str, Any]] = {}
    raw_p = {}
    for name, config in CONFIGS.items():
        row = ablation["configs"][config]
        candidate = rates(row)
        require(set(candidate) == formal_tasks, f"{config}: task scope drift")
        require(row.get("episodes") == 700, f"{config}: coverage drift")
        delta, ci, diffs = paired_delta_ci(baseline, candidate, 10_000, rng)
        contrast_name = f"full_gdsq_vs_{name}"
        p_value = exact_sign_flip_p(diffs)
        raw_p[contrast_name] = p_value
        contrasts[contrast_name] = {
            "baseline": "cscka_final",
            "ablation": config,
            "scope": "Primary14",
            "tasks": 14,
            "seeds_per_task": 50,
            "baseline_task_macro_sr": sum(baseline.values()) / len(baseline),
            "ablation_task_macro_sr": sum(candidate.values()) / len(candidate),
            "delta_full_minus_ablation": delta,
            "task_cluster_ci95": ci,
            "paired_sign_flip_p": p_value,
            "ablation_task_cluster_ci95": row["primary_task_cluster_ci95"],
            "formal_failures": int(row.get("formal_failures", 0)),
            "formal_failure_rate": float(row.get("formal_failure_rate", 0.0)),
        }
    adjusted = holm_adjust(raw_p)
    for name, value in adjusted.items():
        contrasts[name]["holm_adjusted_p"] = value
        contrasts[name]["removal_hurts_significantly"] = (
            contrasts[name]["task_cluster_ci95"][0] > 0 and value <= 0.05
        )
    return {
        "schema_version": 1,
        "kind": "gdsq_vla_gr00t_core_ablation_statistics",
        "complete": True,
        "scope": "Primary14",
        "tasks": 14,
        "seeds_per_task": 50,
        "episodes_per_ablation": 700,
        "bootstrap_draws": 10_000,
        "bootstrap_unit": "task cluster",
        "paired_test": "task-level sign-flip",
        "multiplicity": "Holm over five preregistered component contrasts",
        "direction": "full GDSQ-VLA minus ablated configuration",
        "sources": {
            "static_gdsq": artifact(STATIC),
            "ablation_matrix": artifact(ABLATION),
        },
        "contrasts": contrasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args()
    payload = build()
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {"complete": True, "contrasts": 5, "out": str(output), "sha256": sha256_file(output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
