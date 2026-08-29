#!/usr/bin/env python3
"""Aggregate the frozen v2 quick gate (seeds 60-69, hash-drawn 2/2/1 tasks).

The v1 aggregator is pinned by the full-context protocol attestation
(``quick_statistics_sha256``), so the v2 gate lives in this separate tool:
it reads the frozen ``full_context_v2_quick_spec``, pairs the gdsq_main and
candidate rows, and applies the identical advancement arithmetic
(micro strictly greater, task macro not lower, W > L, bytes <= budget).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from quantvla_cross_model_protocol import sha256_file
from quantvla_full_context import (
    protocol_attestation,
    require_protocol_attestation,
)
from quantvla_outputimpact import atomic_json
from quantvla_table1_bytes import (
    TABLE1_QUANTVLA_BYTES,
    fixed_bytes,
    table1_total_static_budget,
    table1_total_static_bytes,
    table1_variable_budget,
)


def load_rows(root: Path, config: str | None) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for path in sorted(root.glob("**/*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "status" not in row:
                # GPU/server monitoring journals live beside the episode rows.
                continue
            if config is not None and row.get("config") != config:
                continue
            if row.get("status") != "complete":
                raise ValueError(f"{path}:{line_number}: incomplete quick episode")
            key = (str(row["task"]), int(row["seed"]))
            if key in result:
                raise ValueError(f"duplicate quick episode key: {key}")
            result[key] = row
    if not result:
        raise ValueError(f"no quick rows found under {root}")
    return result


def quick_gate(
    rows: list[dict[str, Any]],
    *,
    expected_tasks: set[str],
    expected_seeds: set[int],
    episodes_per_config: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("quick gate requires paired rows")
    tasks: dict[str, list[tuple[bool, bool]]] = {}
    wins = losses = 0
    for row in rows:
        task = str(row["task"])
        main = bool(row["main_success"])
        candidate = bool(row["candidate_success"])
        tasks.setdefault(task, []).append((main, candidate))
        wins += int(candidate and not main)
        losses += int(main and not candidate)
    observed_tasks = set(tasks)
    observed_seeds = {int(row["seed"]) for row in rows}
    if observed_tasks != expected_tasks or observed_seeds != expected_seeds:
        raise ValueError("quick development task/seed coverage drift")
    if len(rows) != episodes_per_config:
        raise ValueError("quick development episode count drift")
    observed_keys = {(str(row["task"]), int(row["seed"])) for row in rows}
    expected_keys = {(task, seed) for task in expected_tasks for seed in expected_seeds}
    if observed_keys != expected_keys or len(observed_keys) != len(rows):
        raise ValueError("quick development paired-key coverage drift")
    main_total = sum(int(bool(row["main_success"])) for row in rows)
    candidate_total = sum(int(bool(row["candidate_success"])) for row in rows)
    main_macro = float(np.mean([np.mean([m for m, _ in values]) for values in tasks.values()]))
    candidate_macro = float(np.mean([np.mean([c for _, c in values]) for values in tasks.values()]))
    return {
        "main_successes": main_total,
        "candidate_successes": candidate_total,
        "main_task_macro": main_macro,
        "candidate_task_macro": candidate_macro,
        "paired_wins": wins,
        "paired_losses": losses,
        "passes_success_gate": bool(
            candidate_total > main_total and candidate_macro >= main_macro and wins > losses
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick-spec", required=True)
    parser.add_argument("--main-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--main-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    quick_spec_path = Path(args.quick_spec).expanduser().resolve()
    quick_spec = json.loads(quick_spec_path.read_text(encoding="utf-8"))
    if quick_spec.get("kind") != "full_context_v2_quick_spec":
        raise ValueError(f"{quick_spec_path}: not a frozen v2 quick spec")
    main_dir = Path(args.main_dir).expanduser().resolve()
    candidate_dir = Path(args.candidate_dir).expanduser().resolve()
    plan_path = Path(args.candidate_plan).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    require_protocol_attestation(plan.get("meta") or {}, source=str(plan_path))

    static_budget = table1_total_static_budget("gr00t")
    static_total = table1_total_static_bytes("gr00t", int(plan["total_bytes"]))
    bytes_pass = static_total <= static_budget
    main_rows = load_rows(main_dir, args.main_config)
    candidate_rows = load_rows(candidate_dir, args.candidate_config)
    if set(main_rows) != set(candidate_rows):
        raise ValueError("main/candidate quick episode inventories differ")
    paired = [
        {
            "task": task,
            "seed": seed,
            "main_success": bool(main_rows[(task, seed)]["success"]),
            "candidate_success": bool(candidate_rows[(task, seed)]["success"]),
        }
        for task, seed in sorted(main_rows)
    ]
    expected_tasks = (
        set(quick_spec["tasks"]["atomic_seen"])
        | set(quick_spec["tasks"]["composite_seen"])
        | set(quick_spec["tasks"]["composite_unseen"])
    )
    result = quick_gate(
        paired,
        expected_tasks=expected_tasks,
        expected_seeds=set(quick_spec["seeds"]),
        episodes_per_config=int(quick_spec["episodes_per_config"]),
    )
    payload = {
        "schema_version": 1,
        "kind": "full_context_v2_quick_development_gate",
        "full_context_protocol": protocol_attestation(),
        "model_adapter": "gr00t",
        "main_dir": str(main_dir),
        "candidate_dir": str(candidate_dir),
        "candidate_plan": str(plan_path),
        "candidate_plan_sha256": sha256_file(plan_path),
        "quick_spec": str(quick_spec_path),
        "quick_spec_sha256": sha256_file(quick_spec_path),
        "advance_if": quick_spec.get("advance_if"),
        "quantvla_table1_bytes": TABLE1_QUANTVLA_BYTES["gr00t"],
        "fixed_bytes": fixed_bytes("gr00t"),
        "table1_total_static_budget_bytes": static_budget,
        "table1_total_static_bytes": static_total,
        "maximum_candidate_bytes": table1_variable_budget("gr00t"),
        "candidate_bytes": int(plan["total_bytes"]),
        "bytes_pass": bytes_pass,
        "paired_rows": paired,
        **result,
        "advance_to_table1": bool(bytes_pass and result["passes_success_gate"]),
    }
    output = Path(args.out).expanduser().resolve()
    atomic_json(output, payload)
    print(
        json.dumps(
            {"out": str(output), "advance": payload["advance_to_table1"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
