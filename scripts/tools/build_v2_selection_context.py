#!/usr/bin/env python3
"""Build the frozen v2 cross-context selection buffer specification.

All choices are deterministic, seeded by the v2 protocol SHA-256:

* 4 tasks per split (atomic_seen / composite_seen / composite_unseen) from
  the v2 protocol Table-1 lists;
* 3 distinct seeds per task drawn from 0..49;
* one replan window per task-seed: early/middle/late = replans
  [0..3] / [4..7] / [8..11] of a 12-replan on-policy FP16 rollout;
* the GR00T scoring checkpoint per split.

The emitted spec is frozen: the capture and pooling tools fail closed if
it changes after buffers were collected.
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
TASKS_PER_SPLIT = 4
SEEDS_PER_TASK = 3
SEED_POOL = list(range(50))
REPLANS_PER_SEQUENCE = 4
MAX_REPLANS = 12
WINDOW_NAMES = ("early", "middle", "late")
WINDOW_STARTS = {"early": 0, "middle": 4, "late": 8}

GR00T_CHECKPOINTS = {
    "atomic_seen": (
        "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
        "target_posttraining/atomic_seen/checkpoint-60000"
    ),
    "composite_seen": (
        "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
        "target_posttraining/composite_seen/checkpoint-60000"
    ),
    "composite_unseen": (
        "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
        "target_posttraining/composite_unseen/checkpoint-60000"
    ),
}


def hash_order(key: bytes, values: list[str]) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(key + value.encode("utf-8")).hexdigest(),
    )


def select_tasks(protocol_sha: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split in SPLITS:
        candidates = [str(task) for task in PROTOCOL_V2["table1"]["tasks"][split]]
        key = hashlib.sha256(
            f"{protocol_sha}|v2_selection_tasks|{split}".encode("utf-8")
        ).digest()
        result[split] = hash_order(key, candidates)[:TASKS_PER_SPLIT]
    return result


def select_seeds(protocol_sha: str, task: str) -> list[int]:
    key = hashlib.sha256(
        f"{protocol_sha}|v2_selection_seeds|{task}".encode("utf-8")
    ).digest()
    order = hash_order(key, [str(seed) for seed in SEED_POOL])
    return [int(seed) for seed in order[:SEEDS_PER_TASK]]


def window_for(protocol_sha: str, task: str, seed: int) -> tuple[str, int]:
    digest = hashlib.sha256(
        f"{protocol_sha}|v2_selection_window|{task}|{seed}".encode("utf-8")
    ).digest()
    name = WINDOW_NAMES[digest[0] % 3]
    return name, WINDOW_STARTS[name]


def build_spec(protocol_sha: str = PROTOCOL_V2_SHA256) -> dict[str, Any]:
    tasks = select_tasks(protocol_sha)
    entries = []
    for split in SPLITS:
        for task in tasks[split]:
            entries.append(
                {
                    "split": split,
                    "task": task,
                    "gr00t_checkpoint": GR00T_CHECKPOINTS[split],
                    "seeds": [
                        {
                            "seed": seed,
                            "window": window_for(protocol_sha, task, seed)[0],
                            "window_start": window_for(protocol_sha, task, seed)[1],
                        }
                        for seed in select_seeds(protocol_sha, task)
                    ],
                }
            )
    return {
        "schema_version": 1,
        "kind": "full_context_v2_selection_context_spec",
        "protocol_sha256": protocol_sha,
        "method_id": PROTOCOL_V2["method_id"],
        "tasks_per_split": TASKS_PER_SPLIT,
        "seeds_per_task": SEEDS_PER_TASK,
        "replans_per_sequence": REPLANS_PER_SEQUENCE,
        "max_replans": MAX_REPLANS,
        "windows": {"early": [0, 4], "middle": [4, 8], "late": [8, 12]},
        "teacher": "fp16_onpolicy_server",
        "pi05_checkpoint_note": "pi0.5 uses the single FP16 server checkpoint",
        "entries": entries,
        "total_tasks": len(entries),
        "total_sequences": len(entries) * SEEDS_PER_TASK,
        "total_observations": len(entries) * SEEDS_PER_TASK * REPLANS_PER_SEQUENCE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v2/selection_context_spec.json",
    )
    args = parser.parse_args()
    spec = build_spec()
    output = Path(args.out).expanduser().resolve()
    atomic_json(output, spec)
    print(
        json.dumps(
            {
                "out": str(output),
                "tasks": spec["total_tasks"],
                "sequences": spec["total_sequences"],
                "observations": spec["total_observations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()