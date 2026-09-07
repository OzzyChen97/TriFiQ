#!/usr/bin/env python3
"""Aggregate the preregistered CS / CKA-DiT / CS+CKA-DiT byte-ablation matrix.

Requires exactly 1,500 unique, complete task-seed-config units (three arms x
50 tasks x 10 seeds; a row counts as complete when it carries a boolean
``success`` and did not crash before writing).  Crashes, timeouts, NaN and
out-of-range actions are all formal failures on the arm (they surface as
``success=false`` rows written by the trial driver under
``--terminal-crash-as-failure``).

Primary estimand: unweighted task-macro SR over Primary-46 (all 50 tasks minus
the four dev tasks used for ratio selection: OpenCabinet, OpenStandMixerHead,
PickPlaceDrawerToCounter, CoffeeSetupMug).  The two preregistered comparisons
are cs_cka minus cka_only and cs_cka minus cs_only, each reported with paired
wins/losses/ties, an exact two-sided McNemar p on discordant paired task-seed
units, and Holm correction over the two comparisons.  Secondary outputs:
All-50 SR, per-split SR, formal failure rates, and mask overlap from the frozen
masks manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scipy.stats import binomtest
except Exception:  # pragma: no cover - fallback for envs without scipy
    binomtest = None

ARMS = ("cs_cka", "cka_only", "cs_only")
DEV_TASKS = ("OpenCabinet", "OpenStandMixerHead", "PickPlaceDrawerToCounter",
             "CoffeeSetupMug")
COMPARISONS = (("cs_cka", "cka_only"), ("cs_cka", "cs_only"))
SPLIT_TASKS = {
    "atomic_seen": [
        "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug",
        "NavigateKitchen", "OpenCabinet", "OpenDrawer", "OpenStandMixerHead",
        "PickPlaceCounterToCabinet", "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter",
        "PickPlaceToasterToCounter", "SlideDishwasherRack", "TurnOffStove",
        "TurnOnElectricKettle", "TurnOnMicrowave", "TurnOnSinkFaucet",
    ],
    "composite_seen": [
        "DeliverStraw", "GetToastedBread", "KettleBoiling", "LoadDishwasher",
        "PackIdenticalLunches", "PreSoakPan", "PrepareCoffee", "RinseSinkBasin",
        "ScrubCuttingBoard", "SearingMeat", "SetUpCuttingStation",
        "StackBowlsCabinet", "SteamInMicrowave", "StirVegetables",
        "StoreLeftoversInBowl", "WashLettuce",
    ],
    "composite_unseen": [
        "ArrangeBreadBasket", "ArrangeTea", "BreadSelection", "CategorizeCondiments",
        "CuttingToolSelection", "GarnishPancake", "GatherTableware",
        "HeatKebabSandwich", "MakeIceLemonade", "PanTransfer", "PortionHotDogs",
        "RecycleBottlesByType", "SeparateFreezerRack", "WaffleReheat",
        "WashFruitColander", "WeighIngredients",
    ],
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(
    manifest: dict[str, Any], manifest_sha: str
) -> dict[tuple[str, str, int], bool]:
    """One validated row per (config, task, seed) unit."""
    rows: dict[tuple[str, str, int], bool] = {}
    counts: Counter = Counter()
    for config in manifest["configs"]:
        for out_path in config["result_files"]:
            path = Path(out_path)
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    row.get("config") != config["id"]
                    or row.get("manifest_sha256") != manifest_sha
                    or row.get("config_sha256") != config["config_sha256"]
                    or not isinstance(row.get("success"), bool)
                    or row.get("crashed")
                ):
                    continue
                key = (str(row.get("config")), str(row.get("task")), int(row.get("seed")))
                rows[key] = bool(row["success"])
                counts[key] += 1
    return rows, counts


def task_macro_sr_for(
    rows: dict[tuple[str, str, int], bool],
    arm: str,
    tasks: list[str],
    seeds: list[int],
) -> float:
    per_task = []
    for task in tasks:
        values = [rows[(arm, task, seed)] for seed in seeds]
        per_task.append(sum(values) / len(values))
    return sum(per_task) / len(per_task) if per_task else float("nan")


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if binomtest is not None:
        return float(binomtest(k, n, 0.5).pvalue)
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * p)


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [float(p) for p in p_values]
    for rank, index in enumerate(order):
        adjusted[index] = min(1.0, p_values[index] * (len(p_values) - rank))
    for rank in range(1, len(order)):
        adjusted[order[rank]] = max(adjusted[order[rank]], adjusted[order[rank - 1]])
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--paper-table", default=None, help="Optional .md table path.")
    args = parser.parse_args()

    root = Path(args.run_root).expanduser().resolve()
    prereg = json.loads((root / "preregistration.json").read_text(encoding="utf-8"))
    masks = json.loads(
        (root / "masks" / "masks_manifest.json").read_text(encoding="utf-8")
    )

    rows: dict[tuple[str, str, int], bool] = {}
    unit_counts: Counter = Counter()
    manifests: dict[str, dict[str, Any]] = {}
    for split in ("atomic_seen", "composite_seen", "composite_unseen"):
        run_dir = root / "formal" / split
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"missing formal manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_sha = sha256_file(manifest_path)
        manifests[split] = manifest
        split_rows, counts = load_rows(manifest, manifest_sha)
        unit_counts.update(counts)
        rows.update(split_rows)

    # Coverage: 1,500 unique complete units, no missing, no duplicates.
    seeds = list(range(10))
    expected = {
        (arm, task, seed)
        for split, tasks in SPLIT_TASKS.items()
        for task in tasks
        for arm in ARMS
        for seed in seeds
    }
    missing = sorted(expected - set(rows))
    unexpected = sorted(set(rows) - expected)
    duplicates = sorted(key for key, count in unit_counts.items() if count != 1)
    if missing or duplicates or unexpected:
        raise SystemExit(
            "aggregation requires 1,500 unique complete task-seed-config units; "
            f"missing={len(missing)} duplicates={len(duplicates)} "
            f"unexpected={len(unexpected)}; first missing={missing[:5]}"
        )

    # Frozen-artifact audit: every deployed plan/subset must match prereg.
    frozen_plans = prereg["masks"]["plans"]
    for split in SPLIT_TASKS:
        manifest = manifests[split]
        for config in manifest["configs"]:
            arm = config["id"]
            if config["plan"]["sha256"] != frozen_plans[arm]["sha256"]:
                raise SystemExit(f"{split}/{arm}: deployed plan differs from frozen mask")
            subset = prereg["hessian_subsets"][f"{split}/{arm}"]
            if config["hessian_w4"]["sha256"] != subset["sha256"]:
                raise SystemExit(f"{split}/{arm}: deployed Hessian subset differs from frozen")

    all_tasks = [t for tasks in SPLIT_TASKS.values() for t in tasks]
    primary_tasks = [t for t in all_tasks if t not in DEV_TASKS]

    def sr(arm: str, tasks: list[str]) -> float:
        return task_macro_sr_for(rows, arm, tasks, seeds)

    results: dict[str, Any] = {
        "coverage": {
            "units": len(rows),
            "missing": len(missing),
            "duplicates": len(duplicates),
            "complete": True,
        },
        "arms": {},
        "comparisons": {},
        "secondary": {"splits": {}, "mask_overlap": {}},
    }
    for arm in ARMS:
        primary = sr(arm, primary_tasks)
        all50 = sr(arm, all_tasks)
        results["arms"][arm] = {
            "primary46_task_macro_sr": primary,
            "all50_task_macro_sr": all50,
            "formal_failure_rate_primary46": 1.0 - primary,
            "formal_failure_rate_all50": 1.0 - all50,
        }
        results["secondary"]["splits"][arm] = {
            split: sr(arm, SPLIT_TASKS[split]) for split in SPLIT_TASKS
        }

    raw_p = []
    for fused, other in COMPARISONS:
        diff = results["arms"][fused]["primary46_task_macro_sr"] - results["arms"][other][
            "primary46_task_macro_sr"
        ]
        b = c = 0
        for task in primary_tasks:
            for seed in seeds:
                f = rows[(fused, task, seed)]
                o = rows[(other, task, seed)]
                if f and not o:
                    c += 1
                elif not f and o:
                    b += 1
        wins = losses = ties = 0
        for task in primary_tasks:
            f = sum(rows[(fused, task, seed)] for seed in seeds)
            o = sum(rows[(other, task, seed)] for seed in seeds)
            if f > o:
                wins += 1
            elif f < o:
                losses += 1
            else:
                ties += 1
        p = exact_mcnemar(b, c)
        raw_p.append(p)
        results["comparisons"][f"{fused}_vs_{other}"] = {
            "delta_task_macro_sr": diff,
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": ties,
            "discordant_other_success_fused_fail": b,
            "discordant_fused_success_other_fail": c,
            "exact_two_sided_mcnemar_p": p,
        }
    adjusted = holm(raw_p)
    for index, (fused, other) in enumerate(COMPARISONS):
        results["comparisons"][f"{fused}_vs_{other}"]["holm_p"] = adjusted[index]

    results["decision"] = {
        "deltas": {
            f"{f}_vs_{o}": results["comparisons"][f"{f}_vs_{o}"]["delta_task_macro_sr"]
            for f, o in COMPARISONS
        },
        "holm_p": {
            f"{f}_vs_{o}": results["comparisons"][f"{f}_vs_{o}"]["holm_p"]
            for f, o in COMPARISONS
        },
    }
    deltas = results["decision"]["deltas"]
    holms = results["decision"]["holm_p"]
    both_positive = all(value > 0 for value in deltas.values())
    both_significant = all(value < 0.05 for value in holms.values())
    if both_positive and both_significant:
        results["decision"]["conclusion"] = "complementary_signals"
    elif both_positive:
        results["decision"]["conclusion"] = "directional_evidence_only"
    else:
        results["decision"]["conclusion"] = "no_complementarity_claim"

    # Mask overlap from the frozen masks manifest.
    protected = {arm: set(masks["arms"][arm]["protected_layers"]) for arm in ARMS}
    overlap = {}
    for left, right in (("cs_cka", "cka_only"), ("cs_cka", "cs_only"),
                        ("cka_only", "cs_only")):
        inter = len(protected[left] & protected[right])
        union = len(protected[left] | protected[right])
        overlap[f"{left}_vs_{right}"] = {
            "hamming": len(protected[left] ^ protected[right]),
            "intersection": inter,
            "jaccard": inter / union if union else 1.0,
        }
    results["secondary"]["mask_overlap"] = overlap

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if out.exists() and out.read_text(encoding="utf-8") != payload:
        raise SystemExit(f"aggregate changed on re-run: {out}")
    out.write_text(payload)

    if args.paper_table:
        rows_out = []
        for arm in ARMS:
            entry = results["arms"][arm]
            delta = (
                results["arms"]["cs_cka"]["primary46_task_macro_sr"]
                - entry["primary46_task_macro_sr"]
            )
            holm_p = (
                results["comparisons"][f"cs_cka_vs_{arm}"]["holm_p"]
                if arm != "cs_cka" else None
            )
            counts = masks["arms"][arm]
            rows_out.append(
                f"| {arm} | {counts['w4_count']}/{counts['fp16_count']} | "
                f"{masks['total_static_bytes']} | {masks['compression']:.4f}x | "
                f"{entry['primary46_task_macro_sr']:.4f} | "
                f"{delta:+.4f} | {'' if holm_p is None else f'{holm_p:.4f}'} | "
                f"{entry['all50_task_macro_sr']:.4f} |"
            )
        table = "\n".join(
            [
                "| signal | W4/FP16 | static size | compression | Primary-46 SR | fused delta | Holm p | All-50 SR |",
                "|---|---|---|---|---|---|---|---|",
                *rows_out,
            ]
        )
        Path(args.paper_table).write_text(table + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
