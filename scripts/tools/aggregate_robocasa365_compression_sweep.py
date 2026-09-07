#!/usr/bin/env python3
"""Aggregate the compression-rate sweep (descriptive), six arms including max.

Requires exactly 3,000 unique, complete task-seed-config units (6 rates x 50
tasks x 10 seeds).  Crashes/timeouts/NaN/out-of-range actions surface as
``success=false`` rows and count as formal failures.  Primary output: per-rate
unweighted task-macro SR over all 50 tasks plus per-split SRs and the
static-bytes/compression columns; the frozen DyPAC headline (2.2239x static,
54.0% All-50) is cited as an existing reference row, not re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ARMS = ("rate12", "rate16", "rate20", "rate24", "rate28", "ratemax")
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
REFERENCE_OURS = {
    "rate": None,
    "w4": 100,
    "fp16": 16,
    "static_bytes": 962_068_480,
    "static_compression": 2.2239,
    "all50_task_macro_sr": 0.540,
    "source": "FINAL_VERSIONS.md frozen DyPAC Table-1 row (cited, not re-run)",
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


def task_macro_sr(
    rows: dict[tuple[str, str, int], bool], arm: str, tasks: list[str], seeds: list[int]
) -> float:
    per_task = []
    for task in tasks:
        values = [rows[(arm, task, seed)] for seed in seeds]
        per_task.append(sum(values) / len(values))
    return sum(per_task) / len(per_task) if per_task else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--paper-table", default=None)
    args = parser.parse_args()

    root = Path(args.run_root).expanduser().resolve()
    prereg = json.loads((root / "preregistration.json").read_text(encoding="utf-8"))
    masks = json.loads((root / "masks" / "masks_manifest.json").read_text(encoding="utf-8"))

    rows: dict[tuple[str, str, int], bool] = {}
    unit_counts: Counter = Counter()
    for split in SPLIT_TASKS:
        run_dir = root / "formal" / split
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"missing formal manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_sha = sha256_file(manifest_path)
        split_rows, counts = load_rows(manifest, manifest_sha)
        unit_counts.update(counts)
        rows.update(split_rows)

    seeds = list(range(10))
    expected = {
        (arm, task, seed)
        for split, tasks in SPLIT_TASKS.items()
        for task in tasks
        for arm in ARMS
        for seed in seeds
    }
    missing = sorted(expected - set(rows))
    duplicates = sorted(key for key, count in unit_counts.items() if count != 1)
    unexpected = sorted(set(rows) - expected)
    if missing or duplicates or unexpected:
        raise SystemExit(
            "aggregation requires 3,000 unique complete task-seed-config units; "
            f"missing={len(missing)} duplicates={len(duplicates)} "
            f"unexpected={len(unexpected)}; first missing={missing[:5]}"
        )

    frozen_plans = prereg["masks"]["rates"]
    for split in SPLIT_TASKS:
        manifest = json.loads(
            (root / "formal" / split / "manifest.json").read_text(encoding="utf-8")
        )
        for config in manifest["configs"]:
            arm = config["id"]
            if config["plan"]["sha256"] != frozen_plans[arm]["plan_sha256"]:
                raise SystemExit(f"{split}/{arm}: deployed plan differs from frozen mask")
            subset = prereg["hessian_subsets"][f"{split}/{arm}"]
            if config["hessian_w4"]["sha256"] != subset["sha256"]:
                raise SystemExit(f"{split}/{arm}: deployed Hessian subset differs from frozen")

    all_tasks = [t for tasks in SPLIT_TASKS.values() for t in tasks]
    results: dict[str, Any] = {
        "coverage": {
            "units": len(rows), "missing": len(missing),
            "duplicates": len(duplicates), "complete": True,
        },
        "rates": {},
        "reference_ours": REFERENCE_OURS,
    }
    for arm in ARMS:
        entry = masks["rates_manifest"][arm]
        all50 = task_macro_sr(rows, arm, all_tasks, seeds)
        results["rates"][arm] = {
            "rate": entry["rate"],
            "w4_layers": entry["w4_layers"],
            "fp16_layers": entry["fp16_layers"],
            "static_bytes": entry["static_bytes"],
            "static_compression": entry["static_compression"],
            "achieved_candidate_compression": entry["achieved_candidate_compression"],
            "all50_task_macro_sr": all50,
            "formal_failure_rate_all50": 1.0 - all50,
            "splits": {
                split: task_macro_sr(rows, arm, SPLIT_TASKS[split], seeds)
                for split in SPLIT_TASKS
            },
        }

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if out.exists() and out.read_text(encoding="utf-8") != payload:
        raise SystemExit(f"aggregate changed on re-run: {out}")
    out.write_text(payload)

    if args.paper_table:
        lines = [
            "| target_compression | W4/FP16 | static bytes | static compression | All-50 SR | Atomic | C-Seen | C-Unseen |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for arm in ARMS:
            entry = results["rates"][arm]
            lines.append(
                f"| {entry['rate']:g} | {entry['w4_layers']}/{entry['fp16_layers']} | "
                f"{entry['static_bytes']} | {entry['static_compression']:.4f}x | "
                f"{entry['all50_task_macro_sr']:.4f} | {entry['splits']['atomic_seen']:.4f} | "
                f"{entry['splits']['composite_seen']:.4f} | {entry['splits']['composite_unseen']:.4f} |"
            )
        ref = REFERENCE_OURS
        lines.append(
            f"| ours (frozen, cited) | {ref['w4']}/{ref['fp16']} | {ref['static_bytes']} | "
            f"{ref['static_compression']}x | {ref['all50_task_macro_sr']:.4f} | - | - | - |"
        )
        Path(args.paper_table).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
