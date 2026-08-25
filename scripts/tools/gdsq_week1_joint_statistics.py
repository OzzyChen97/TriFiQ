#!/usr/bin/env python3
"""Recompute preregistered same-budget contrasts across both VLA families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from parse_robocasa_atomic_matrix import exact_sign_flip_p, holm_adjust, paired_delta_ci


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "runs/gdsq_week1_preregistered_v1"
EXECUTION = ROOT / "execution"
OUTPUT = EXECUTION / "aggregate/week1_joint_statistics.json"
DEV4 = {"CoffeeSetupMug", "OpenCabinet", "OpenStandMixerHead", "PickPlaceDrawerToCounter"}


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
    require(path.is_file(), f"missing summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def rates(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("per_task_sr") or row.get("per_task")
    require(isinstance(raw, dict) and raw, "summary has no per-task rates")
    result = {}
    for task, value in raw.items():
        if isinstance(value, dict):
            value = value.get("sr")
        result[str(task)] = float(value)
    return result


def paired(
    name: str,
    ours: dict[str, float],
    baseline: dict[str, float],
    scope: str,
    rng: random.Random,
) -> dict[str, Any]:
    require(set(ours) == set(baseline), f"task mismatch for {name}")
    delta, ci, diffs = paired_delta_ci(ours, baseline, 10_000, rng)
    return {
        "name": name,
        "scope": scope,
        "tasks": len(ours),
        "ours_task_macro_sr": sum(ours.values()) / len(ours),
        "baseline_task_macro_sr": sum(baseline.values()) / len(baseline),
        "task_macro_delta_ours_minus_baseline": delta,
        "task_cluster_ci95": ci,
        "paired_sign_flip_p": exact_sign_flip_p(diffs),
        "ci_excludes_zero_in_favor_of_ours": ci[0] > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args()
    paths = {
        "gr00t_ours": REPO_ROOT / "runs/robocasa365_official_full_paired50/summary.json",
        "gr00t_w6": EXECUTION / "aggregate/gr00t_uniform_w6/summary.json",
        "gr00t_controls": EXECUTION / "runs/gr00t_same_budget_controls_primary14/summary.json",
        "pi05_ours": ROOT / "pi05_selector_official50/aggregate/summary.json",
        "pi05_w6": EXECUTION / "runs/pi05_uniform_w6_official50/aggregate/summary.json",
        "pi05_random": EXECUTION / "runs/pi05_controls_heldout46/search_matched_random/aggregate/summary.json",
        "pi05_action": EXECUTION / "runs/pi05_controls_heldout46/action_only/aggregate/summary.json",
    }
    values = {name: load(path) for name, path in paths.items()}
    require(values["pi05_ours"].get("complete") is True, "pi0.5 selector incomplete")
    require(values["pi05_w6"].get("complete") is True, "pi0.5 W6 incomplete")
    require(values["pi05_random"].get("complete") is True, "pi0.5 random incomplete")
    require(values["pi05_action"].get("complete") is True, "pi0.5 action incomplete")

    gr_ours_all = rates(values["gr00t_ours"]["configs"]["cscka_final"])
    gr_w6_all = rates(values["gr00t_w6"]["configs"]["uniform_w6"])
    gr_primary = {task: value for task, value in gr_ours_all.items() if task not in DEV4 and task in rates(values["gr00t_controls"]["configs"]["search_matched_random"])}
    gr_random = rates(values["gr00t_controls"]["configs"]["search_matched_random"])
    gr_action = rates(values["gr00t_controls"]["configs"]["action_only"])

    pi_ours_all = rates(values["pi05_ours"]["configs"]["gdsq_vla_runtime_selector"])
    pi_w6_all = rates(values["pi05_w6"]["configs"]["uniform_w6"])
    pi_random = rates(values["pi05_random"]["configs"]["search_matched_random"])
    pi_action = rates(values["pi05_action"]["configs"]["action_only"])
    pi_heldout = {task: pi_ours_all[task] for task in pi_random}

    require(len(gr_ours_all) == len(gr_w6_all) == 50, "GR00T all-50 rate drift")
    require(len(gr_primary) == len(gr_random) == len(gr_action) == 14, "GR00T Primary14 drift")
    require(len(pi_ours_all) == len(pi_w6_all) == 50, "pi0.5 all-50 rate drift")
    require(len(pi_heldout) == len(pi_random) == len(pi_action) == 46, "pi0.5 heldout46 drift")

    rng = random.Random(20260823)
    contrasts = [
        paired("gr00t_ours_vs_uniform_w6", gr_ours_all, gr_w6_all, "all50", rng),
        paired("gr00t_ours_vs_random", gr_primary, gr_random, "Primary14", rng),
        paired("gr00t_ours_vs_action_only", gr_primary, gr_action, "Primary14", rng),
        paired("pi05_ours_vs_uniform_w6", pi_ours_all, pi_w6_all, "all50", rng),
        paired("pi05_ours_vs_random", pi_heldout, pi_random, "heldout46", rng),
        paired("pi05_ours_vs_action_only", pi_heldout, pi_action, "heldout46", rng),
    ]
    adjusted = holm_adjust({row["name"]: row["paired_sign_flip_p"] for row in contrasts})
    for row in contrasts:
        row["holm_adjusted_p"] = adjusted[row["name"]]
        row["holm_significant_in_favor_of_ours"] = (
            row["ci_excludes_zero_in_favor_of_ours"] and row["holm_adjusted_p"] <= 0.05
        )

    by_name = {row["name"]: row for row in contrasts}
    groups = {}
    for model in ("gr00t", "pi05"):
        full = by_name[f"{model}_ours_vs_uniform_w6"]
        heldout_candidates = [
            by_name[f"{model}_ours_vs_random"],
            by_name[f"{model}_ours_vs_action_only"],
        ]
        strongest = max(
            enumerate(heldout_candidates),
            key=lambda item: (item[1]["baseline_task_macro_sr"], -item[0]),
        )[1]
        enabled = full["holm_significant_in_favor_of_ours"] and strongest[
            "holm_significant_in_favor_of_ours"
        ]
        groups[model] = {
            "all50_baseline": full["name"],
            "heldout_strongest_baseline": strongest["name"],
            "superiority_claim_enabled": enabled,
            "reporting_mode": "superior" if enabled else "competitive_or_mechanistic",
            "architecture_boundary_required": model == "pi05" and not enabled,
        }
    overall = all(row["superiority_claim_enabled"] for row in groups.values())
    payload = {
        "schema_version": 1,
        "kind": "gdsq_vla_week1_joint_preregistered_statistics",
        "complete": True,
        "bootstrap_draws": 10_000,
        "bootstrap_unit": "task cluster",
        "paired_test": "task-level sign-flip",
        "multiplicity": "Holm over six preregistered same-budget contrasts",
        "scope_note": "All-50 W6 and held-out random/action contrasts are reported separately and never pooled.",
        "strongest_baseline_rule": "maximum held-out task-macro SR; exact tie: random then action-only",
        "sources": {name: artifact(path) for name, path in paths.items()},
        "contrasts": {row["name"]: row for row in contrasts},
        "strongest_baseline_gates": groups,
        "same_budget_superiority_claim_enabled_across_architectures": overall,
        "paper_rule": (
            "Use superiority wording only when ours minus the strongest same-budget baseline "
            "has task-cluster CI lower > 0 and Holm-adjusted sign-flip p <= 0.05; "
            "otherwise report competitive results/mechanism analysis."
        ),
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"complete": True, "out": str(output), "sha256": sha256_file(output), "claim_enabled": overall}, indent=2))


if __name__ == "__main__":
    main()
