#!/usr/bin/env python3
"""Build non-oracle task features for the task-conditioned ATM/OHB selector.

The feature block intentionally contains only task metadata plus already-public
baseline context (GDSQ, FP16, and low-bit baseline success rates).  Retrospective
ATM/OHB ablation outcomes are kept under ``analysis_only`` so downstream tools
can reject accidental oracle use.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_PROVENANCE = REPO_ROOT / "runs/atmohb_seeded_atomic_10x10/final/provenance"
DEFAULT_GR00T_SUMMARY = FINAL_PROVENANCE / "gr00t_development_ablation_summary.json"
DEFAULT_PI05_SUMMARY = FINAL_PROVENANCE / "pi05_development_ablation_summary.json"

DEVELOPMENT_TASKS = {
    "OpenCabinet",
    "OpenStandMixerHead",
    "PickPlaceDrawerToCounter",
    "CoffeeSetupMug",
}

TASK_HORIZONS = {
    "CloseBlenderLid": 900,
    "CloseFridge": 900,
    "CloseToasterOvenDoor": 450,
    "CoffeeSetupMug": 600,
    "NavigateKitchen": 450,
    "OpenCabinet": 1050,
    "OpenDrawer": 750,
    "OpenStandMixerHead": 450,
    "PickPlaceCounterToCabinet": 750,
    "PickPlaceCounterToStove": 600,
    "PickPlaceDrawerToCounter": 750,
    "PickPlaceSinkToCounter": 900,
    "PickPlaceToasterToCounter": 600,
    "SlideDishwasherRack": 450,
    "TurnOffStove": 750,
    "TurnOnElectricKettle": 450,
    "TurnOnMicrowave": 450,
    "TurnOnSinkFaucet": 600,
    "DeliverStraw": 2550,
    "GetToastedBread": 3000,
    "KettleBoiling": 1500,
    "LoadDishwasher": 1800,
    "PackIdenticalLunches": 3900,
    "PreSoakPan": 2400,
    "PrepareCoffee": 1800,
    "RinseSinkBasin": 1350,
    "ScrubCuttingBoard": 1200,
    "SearingMeat": 4350,
    "SetUpCuttingStation": 2400,
    "StackBowlsCabinet": 2100,
    "SteamInMicrowave": 2100,
    "StirVegetables": 2400,
    "StoreLeftoversInBowl": 2550,
    "WashLettuce": 1650,
    "ArrangeBreadBasket": 4350,
    "ArrangeTea": 2250,
    "BreadSelection": 1950,
    "CategorizeCondiments": 1650,
    "CuttingToolSelection": 1200,
    "GarnishPancake": 2700,
    "GatherTableware": 2250,
    "HeatKebabSandwich": 2700,
    "MakeIceLemonade": 3000,
    "PanTransfer": 1800,
    "PortionHotDogs": 2250,
    "RecycleBottlesByType": 2850,
    "SeparateFreezerRack": 2400,
    "WaffleReheat": 4050,
    "WashFruitColander": 3150,
    "WeighIngredients": 3000,
}

MODEL_CONFIGS = {
    "gr00t": {
        "display_name": "GR00T N1.5 RoboCasa365 atomic",
        "summary_arg": "gr00t_summary",
        "baseline_config_id": "cscka_final",
        "fp16_config_id": "fp16",
        "lowbit_context_config_id": "w4a8_atmohb",
        "variant_config_ids": {
            "baseline": "cscka_final",
            "atm": "cscka_final_atm",
            "ohb": "cscka_final_ohb",
            "atmohb": "cscka_final_atmohb",
        },
    },
    "pi05": {
        "display_name": "pi0.5 RoboCasa365 atomic",
        "summary_arg": "pi05_summary",
        "baseline_config_id": "gdsq_vla",
        "fp16_config_id": "fp16",
        "lowbit_context_config_id": "quantvla_w4a8_atmohb",
        "variant_config_ids": {
            "baseline": "gdsq_vla",
            "atm": "gdsq_vla_atm_only",
            "ohb": "gdsq_vla_ohb_only",
            "atmohb": "gdsq_vla_atmohb",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-summary", default=str(DEFAULT_GR00T_SUMMARY))
    parser.add_argument("--pi05-summary", default=str(DEFAULT_PI05_SUMMARY))
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"missing summary file: {path}") from exc


def split_task_tokens(task: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", task)]


def action_family(task: str, tokens: list[str]) -> str:
    if task.startswith("PickPlace"):
        return "pick_place"
    if task.startswith("Open"):
        return "open"
    if task.startswith("Close"):
        return "close"
    if task.startswith("TurnOn") or task.startswith("TurnOff"):
        return "turn"
    if task.startswith("Slide"):
        return "slide"
    if task.startswith("Navigate"):
        return "navigate"
    if "setup" in tokens or task.startswith("CoffeeSetup"):
        return "setup"
    return tokens[0] if tokens else "unknown"


def object_family(task: str, tokens: list[str]) -> str:
    text = "_".join(tokens)
    if "pick" in tokens and "place" in tokens:
        if "sink" in tokens:
            return "sink_transfer"
        if "toaster" in tokens:
            return "appliance_transfer"
        if "stove" in tokens:
            return "appliance_transfer"
        if "cabinet" in tokens or "drawer" in tokens:
            return "storage_transfer"
        return "object_transfer"
    if any(word in tokens for word in ("fridge", "microwave", "toaster", "kettle", "stove", "blender")):
        return "appliance"
    if "stand" in tokens and "mixer" in tokens:
        return "appliance"
    if any(word in tokens for word in ("cabinet", "drawer")):
        return "storage"
    if "sink" in tokens or "faucet" in tokens or "dishwasher" in tokens:
        return "fixture"
    if "mug" in tokens or "coffee" in tokens:
        return "beverage"
    if "kitchen" in tokens:
        return "navigation"
    return text or "unknown"


def task_metadata(task: str) -> dict[str, Any]:
    tokens = split_task_tokens(task)
    is_dev = task in DEVELOPMENT_TASKS
    return {
        "task_name": task,
        "task_name_tokens": tokens,
        "action_family": action_family(task, tokens),
        "object_family": object_family(task, tokens),
        "official_horizon": TASK_HORIZONS.get(task),
        "is_development_task": is_dev,
        "is_heldout": not is_dev,
        "task_set": "atomic_seen" if task in TASK_HORIZONS and TASK_HORIZONS[task] <= 1050 else "unknown",
    }


def config_sr(summary: dict[str, Any], config_id: str, task: str) -> float | None:
    config = (summary.get("configs") or {}).get(config_id) or {}
    per_task = config.get("per_task") or {}
    if task in per_task and isinstance(per_task[task], dict):
        return float(per_task[task]["sr"])
    per_task_sr = config.get("per_task_sr") or {}
    if task in per_task_sr:
        return float(per_task_sr[task])
    per_task_details = config.get("per_task_details") or {}
    if task in per_task_details:
        return float(per_task_details[task]["sr"])
    return None


def build_model_features(model_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    cfg = MODEL_CONFIGS[model_id]
    tasks = list(summary.get("tasks") or [])
    if not tasks:
        configs = summary.get("configs") or {}
        baseline = configs.get(cfg["baseline_config_id"], {})
        tasks = sorted((baseline.get("per_task") or baseline.get("per_task_sr") or {}).keys())
    if not tasks:
        raise ValueError(f"{model_id}: no tasks found in summary")

    model_tasks: dict[str, Any] = {}
    for task in tasks:
        meta = task_metadata(task)
        baseline_sr = config_sr(summary, cfg["baseline_config_id"], task)
        fp16_sr = config_sr(summary, cfg["fp16_config_id"], task)
        lowbit_sr = config_sr(summary, cfg["lowbit_context_config_id"], task)
        if baseline_sr is None:
            raise ValueError(f"{model_id}: missing baseline SR for {task}")
        features = dict(meta)
        features.update(
            {
                "baseline_gdsq_sr": baseline_sr,
                "fp16_gap": None if fp16_sr is None else baseline_sr - fp16_sr,
                "lowbit_gap": None if lowbit_sr is None else baseline_sr - lowbit_sr,
            }
        )

        variant_srs: dict[str, float] = {}
        deltas: dict[str, float] = {}
        for variant, config_id in cfg["variant_config_ids"].items():
            sr = config_sr(summary, config_id, task)
            if sr is not None:
                variant_srs[variant] = sr
                deltas[variant] = sr - baseline_sr
        best_variant = None
        if variant_srs:
            best_variant = sorted(variant_srs, key=lambda v: (-variant_srs[v], v))[0]
        model_tasks[task] = {
            "features": features,
            "analysis_only": {
                "retrospective_variant_success_rates": variant_srs,
                "retrospective_deltas_vs_baseline": deltas,
                "retrospective_best_variant": best_variant,
                "source_note": "diagnostic only; not permitted for held-out selector threshold fitting",
            },
        }

    return {
        "display_name": cfg["display_name"],
        "baseline_config_id": cfg["baseline_config_id"],
        "fp16_config_id": cfg["fp16_config_id"],
        "lowbit_context_config_id": cfg["lowbit_context_config_id"],
        "variant_config_ids": cfg["variant_config_ids"],
        "tasks": model_tasks,
    }


def build_features(gr00t_summary: Path, pi05_summary: Path) -> dict[str, Any]:
    summaries = {
        "gr00t": read_json(gr00t_summary),
        "pi05": read_json(pi05_summary),
    }
    model_blocks = {model: build_model_features(model, summary) for model, summary in summaries.items()}
    all_tasks = sorted({task for block in model_blocks.values() for task in block["tasks"]})
    return {
        "schema_version": 1,
        "kind": "atmohb_dynamic_selector_features",
        "no_oracle_feature_contract": {
            "feature_fields_exclude_ablation_outcomes": True,
            "ablation_outcomes_location": "models.<model>.tasks.<task>.analysis_only",
            "allowed_feature_context": ["task metadata", "baseline GDSQ SR", "FP16 gap", "low-bit gap"],
        },
        "source_summaries": {
            "gr00t": str(gr00t_summary),
            "pi05": str(pi05_summary),
        },
        "development_tasks": sorted(DEVELOPMENT_TASKS),
        "tasks": {task: task_metadata(task) for task in all_tasks},
        "models": model_blocks,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()
    payload = build_features(Path(args.gr00t_summary).resolve(), Path(args.pi05_summary).resolve())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "models": sorted(payload["models"]), "tasks": len(payload["tasks"])}))


if __name__ == "__main__":
    main()