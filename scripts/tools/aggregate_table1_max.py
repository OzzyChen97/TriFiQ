#!/usr/bin/env python3
"""Aggregate the GR00T + pi0.5 Table-1 max (all-W4) runs (descriptive).

Requires 2,500 unique complete units per model (5,000 total).  Primary:
All-50 unweighted task-macro SR per model with per-split breakdowns and the
static bytes/compression columns.  Secondary: cross-run paired (task-seed)
McNemar against the frozen Table-1 "ours" rows (GR00T 54.0% @2.2239x, pi0.5
27.7% @2.702x, both 50 seeds, same paired-noise protocol) - explicitly
flagged as cross-run descriptive, not a significance claim.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None

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
OURS = {
    "gr00t": {"all50": 0.540, "static_bytes": 962_068_480, "compression": 2.2239},
    "pi05": {"all50": 0.277, "static_bytes": 1_634_828_288, "compression": 2.7016},
}
OURS_ROW_DIRS = {
    "gr00t": "/home1/gyy/vla/QuantVLA/runs/full_context_v2/table1/results/full_context_v2",
    "pi05": "/home1/gyy/vla/QuantVLA/runs/full_context_v2/pi05_table1/results/"
    "full_context_w4a8_dynamic_profile",
}


def macro_sr(
    rows: dict[tuple[str, str, int], bool], tasks: list[str], seeds: list[int]
) -> float:
    per_task = []
    for task in tasks:
        values = [rows[(task, seed)] for seed in seeds]
        per_task.append(sum(values) / len(values))
    return sum(per_task) / len(per_task) if per_task else float("nan")


def load_gr00t(root: Path) -> dict[tuple[str, str, int], bool]:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from run_robocasa_atomic_matrix import sha256_file
    rows: dict[tuple[str, str, int], bool] = {}
    counts: Counter = Counter()
    for split in SPLIT_TASKS:
        run_dir = root / "gr00t" / split
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest_sha = sha256_file(run_dir / "manifest.json")
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
                    key = (str(row.get("task")), int(row.get("seed")))
                    rows[key] = bool(row["success"])
                    counts[key] += 1
    return rows, counts


def load_pi05(results_dir: Path) -> dict[tuple[str, str, int], bool]:
    rows: dict[tuple[str, str, int], bool] = {}
    counts: Counter = Counter()
    for path in sorted(results_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("success"), bool) or row.get("crashed"):
                continue
            key = (str(row.get("task")), int(row.get("seed")))
            rows[key] = bool(row["success"])
            counts[key] += 1
    return rows, counts


def load_ours(dir_path: Path) -> dict[tuple[str, str, int], bool]:
    rows: dict[tuple[str, str, int], bool] = {}
    for path in sorted(Path(dir_path).rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("success"), bool):
                continue
            key = (str(row.get("task")), int(row.get("seed")))
            if key not in rows:
                rows[key] = bool(row["success"])
    return rows


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if binomtest is not None:
        return float(binomtest(k, n, 0.5).pvalue)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--paper-table", default=None)
    parser.add_argument(
        "--gr00t-only",
        action="store_true",
        help="aggregate the GR00T phase only (pi0.5 rows not yet complete)",
    )
    args = parser.parse_args()

    root = Path(args.run_root).expanduser().resolve()
    seeds = list(range(50))
    all_tasks = [t for ts in SPLIT_TASKS.values() for t in ts]

    results: dict[str, Any] = {
        "coverage": {"complete": True, "models": {}},
        "models": {},
        "cross_run_comparisons": {},
        "reference_ours": OURS,
    }

    gr00t_rows, gr00t_counts = load_gr00t(root)
    expected = {(task, seed) for task in all_tasks for seed in seeds}
    missing = sorted(expected - set(gr00t_rows))
    duplicates = sorted(k for k, c in gr00t_counts.items() if c != 1)
    if missing or duplicates or len(gr00t_rows) != 2500:
        raise SystemExit(
            f"GR00T max coverage incomplete: rows={len(gr00t_rows)} "
            f"missing={len(missing)} duplicates={len(duplicates)}"
        )
    results["coverage"]["models"]["gr00t"] = {"units": len(gr00t_rows), "complete": True}

    pi05_rows: dict[Any, Any] = {}
    if not args.gr00t_only:
        pi05_results = root / "pi05" / "results" / "full_context_w4a8_dynamic_profile"
        pi05_rows, pi05_counts = load_pi05(pi05_results)
        missing = sorted(expected - set(pi05_rows))
        duplicates = sorted(k for k, c in pi05_counts.items() if c != 1)
        if missing or duplicates or len(pi05_rows) != 2500:
            raise SystemExit(
                f"pi05 max coverage incomplete: rows={len(pi05_rows)} "
                f"missing={len(missing)} duplicates={len(duplicates)}"
            )
        results["coverage"]["models"]["pi05"] = {"units": len(pi05_rows), "complete": True}

    model_rows = [("gr00t", gr00t_rows)]
    if not args.gr00t_only:
        model_rows.append(("pi05", pi05_rows))
    for model, rows in model_rows:
        all50 = macro_sr(rows, all_tasks, seeds)
        results["models"][model] = {
            "all50_task_macro_sr": all50,
            "formal_failure_rate_all50": 1.0 - all50,
            "splits": {
                split: macro_sr(rows, SPLIT_TASKS[split], seeds)
                for split in SPLIT_TASKS
            },
        }
        ours_rows = load_ours(Path(OURS_ROW_DIRS[model]))
        common = set(rows) & set(ours_rows)
        b = c = 0
        for key in common:
            if rows[key] and not ours_rows[key]:
                c += 1
            elif not rows[key] and ours_rows[key]:
                b += 1
        results["cross_run_comparisons"][f"max_vs_ours_{model}"] = {
            "paired_units": len(common),
            "discordant_ours_success_max_fail": b,
            "discordant_max_success_ours_fail": c,
            "exact_two_sided_mcnemar_p": exact_mcnemar(b, c),
            "flag": "cross-run descriptive; same 50 seeds and paired-noise protocol",
        }

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if out.exists() and out.read_text(encoding="utf-8") != payload:
        raise SystemExit(f"aggregate changed on re-run: {out}")
    out.write_text(payload)

    if args.paper_table:
        lines = [
            "| model | config | W4/FP16 | static bytes | static compression | All-50 SR | Atomic | C-Seen | C-Unseen |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        max_info = {
            "gr00t": {"w4": 116, "fp16": 0, "static_bytes": 836_960_256, "compression": 2.5563},
            "pi05": {"w4": 180, "fp16": 0, "static_bytes": 1_242_169_344, "compression": 3.5556},
        }
        for model in ("gr00t", "pi05"):
            if model not in results["models"]:
                continue
            info = max_info[model]
            entry = results["models"][model]
            lines.append(
                f"| {model} | max (all-W4) | {info['w4']}/{info['fp16']} | "
                f"{info['static_bytes']} | {info['compression']:.4f}x | "
                f"{entry['all50_task_macro_sr']:.4f} | {entry['splits']['atomic_seen']:.4f} | "
                f"{entry['splits']['composite_seen']:.4f} | {entry['splits']['composite_unseen']:.4f} |"
            )
            ours = OURS[model]
            lines.append(
                f"| {model} | ours (cited) | - | {ours['static_bytes']} | "
                f"{ours['compression']:.4f}x | {ours['all50']:.4f} | - | - | - |"
            )
        Path(args.paper_table).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
