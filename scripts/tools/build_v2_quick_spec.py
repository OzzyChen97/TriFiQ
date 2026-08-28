#!/usr/bin/env python3
"""Build the frozen v2 quick development spec.

Tasks are drawn deterministically from the v2 protocol Table-1 lists:
2 per split (2/2/1 episodes-per-task weighting means 2 atomic / 2 seen /
1 unseen tasks), EXCLUDING every task already consumed by the v2
selection buffer and the v1 quick gate (seeds 50-59). Seeds are 60-69,
fresh for confirmation. The spec is frozen before launch: tasks, seeds,
source SHAs and the byte rule are all recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from quantvla_full_context import PROTOCOL_V2, PROTOCOL_V2_SHA256
from quantvla_outputimpact import atomic_json

SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")
QUICK_TASKS_PER_SPLIT = {"atomic_seen": 2, "composite_seen": 2, "composite_unseen": 1}
SEEDS = list(range(60, 70))


def hash_order(key: bytes, values: list[str]) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(key + value.encode("utf-8")).hexdigest(),
    )


def draw_quick_tasks(
    protocol_sha: str, excluded_tasks: set[str]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split in SPLITS:
        candidates = [
            str(task)
            for task in PROTOCOL_V2["table1"]["tasks"][split]
            if str(task) not in excluded_tasks
        ]
        key = hashlib.sha256(
            f"{protocol_sha}|v2_quick_tasks|{split}".encode("utf-8")
        ).digest()
        count = QUICK_TASKS_PER_SPLIT[split]
        if len(candidates) < count:
            raise ValueError(f"{split}: only {len(candidates)} unused tasks, need {count}")
        result[split] = hash_order(key, candidates)[:count]
    return result


def build_spec(
    protocol_sha: str,
    selection_spec: dict[str, Any] | None,
    v1_quick_tasks: list[str] | None,
) -> dict[str, Any]:
    excluded: set[str] = set()
    if selection_spec is not None:
        excluded.update(entry["task"] for entry in selection_spec["entries"])
    if v1_quick_tasks:
        excluded.update(v1_quick_tasks)
    tasks = draw_quick_tasks(protocol_sha, excluded)
    return {
        "schema_version": 1,
        "kind": "full_context_v2_quick_spec",
        "protocol_sha256": protocol_sha,
        "method_id": PROTOCOL_V2["method_id"],
        "seeds": SEEDS,
        "episodes_per_config": 50,
        "tasks": tasks,
        "per_split_task_counts": QUICK_TASKS_PER_SPLIT,
        "excluded_tasks": sorted(excluded),
        "advance_if": [
            "candidate_micro_success_strictly_greater_than_gdsq_main",
            "candidate_task_macro_not_less_than_gdsq_main",
            "paired_wins_strictly_greater_than_paired_losses",
            "candidate_total_static_bytes_not_greater_than_1.10_times_table1_quantvla_bytes",
        ],
        "pi05_baseline_requirement": (
            "runtime_selector_main_or_bitwise_equivalence_artifact_for_gdsq_vla_ohb_only"
        ),
        "freeze_before_launch": [
            "tasks",
            "seeds",
            "mask_plans",
            "activation_mode",
            "hessian_artifacts",
            "byte_counter_source_sha256",
        ],
        "table1_held_out": True,
        "total_tasks": sum(len(value) for value in tasks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-spec",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v2/selection_context_spec.json",
    )
    parser.add_argument(
        "--v1-quick-tasks",
        default="CloseFridge,OpenDrawer,LoadDishwasher,PrepareCoffee,MakeIceLemonade",
    )
    parser.add_argument(
        "--out",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v2/quick_spec.json",
    )
    args = parser.parse_args()
    selection_spec = None
    selection_path = Path(args.selection_spec).expanduser().resolve()
    if selection_path.is_file():
        selection_spec = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_spec.get("kind") != "full_context_v2_selection_context_spec":
            raise ValueError(f"{selection_path}: wrong spec kind")
    v1_tasks = [item.strip() for item in args.v1_quick_tasks.split(",") if item.strip()]
    spec = build_spec(PROTOCOL_V2_SHA256, selection_spec, v1_tasks)
    atomic_json(Path(args.out).expanduser().resolve(), spec)
    print(json.dumps({"out": args.out, "tasks": spec["tasks"]}, indent=2))


if __name__ == "__main__":
    main()