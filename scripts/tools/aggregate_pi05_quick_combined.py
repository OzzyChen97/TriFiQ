#!/usr/bin/env python3
"""Combine the pi0.5 quick waves (seeds 50-59 and the 60-69 expansion) into
one parity report.

The frozen v1 gate alone (5 tasks x 10 seeds) has low power; the combined
100-episode paired report is the registered evidence for the compression-
parity claim.  The output keeps the ``full_context_quick_development_gate``
kind so ``register_pi05_non_inferiority_anchor`` can freeze it (that tool
refuses a superior candidate, which would instead open the advance path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from quantvla_cross_model_protocol import sha256_file
from quantvla_full_context import protocol_attestation, require_protocol_attestation
from quantvla_outputimpact import atomic_json
from quantvla_table1_bytes import (
    TABLE1_QUANTVLA_BYTES,
    fixed_bytes,
    table1_total_static_budget,
    table1_total_static_bytes,
)


def load_rows(root: Path, config: str) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(root.glob("**/*.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if "status" not in row:
                continue  # monitoring journals
            if row.get("config") != config:
                continue
            if row.get("status") != "complete":
                raise ValueError(f"{path}:{line_number}: incomplete quick episode")
            key = (str(row["task"]), int(row["seed"]))
            if key in rows:
                raise ValueError(f"duplicate quick episode key: {key}")
            rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-root", action="append", required=True)
    parser.add_argument("--main-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    plan_path = Path(args.candidate_plan).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    require_protocol_attestation(plan.get("meta") or {}, source=str(plan_path))
    static_budget = table1_total_static_budget("pi05")
    static_total = table1_total_static_bytes("pi05", int(plan["total_bytes"]))
    bytes_pass = static_total <= static_budget

    main_rows: dict[tuple[str, int], dict[str, Any]] = {}
    candidate_rows: dict[tuple[str, int], dict[str, Any]] = {}
    wave_seeds: dict[str, list[int]] = {}
    for root_value in args.wave_root:
        root = Path(root_value).expanduser().resolve()
        label = root.name
        main_part = load_rows(root, args.main_config)
        candidate_part = load_rows(root, args.candidate_config)
        overlap = set(main_part) & set(main_rows)
        if overlap:
            raise ValueError(f"{root}: seed overlap across waves: {sorted(overlap)[:3]}")
        if set(main_part) != set(candidate_part):
            raise ValueError(f"{root}: main/candidate inventories differ")
        wave_seeds[label] = sorted({seed for _task, seed in main_part})
        main_rows.update(main_part)
        candidate_rows.update(candidate_part)
    if set(main_rows) != set(candidate_rows):
        raise ValueError("combined main/candidate inventories differ")
    paired = [
        {
            "task": task,
            "seed": seed,
            "main_success": bool(main_rows[(task, seed)]["success"]),
            "candidate_success": bool(candidate_rows[(task, seed)]["success"]),
        }
        for task, seed in sorted(main_rows)
    ]
    tasks: dict[str, list[tuple[bool, bool]]] = {}
    wins = losses = 0
    for row in paired:
        tasks.setdefault(str(row["task"]), []).append(
            (bool(row["main_success"]), bool(row["candidate_success"]))
        )
        wins += int(row["candidate_success"] and not row["main_success"])
        losses += int(row["main_success"] and not row["candidate_success"])
    main_total = sum(int(bool(row["main_success"])) for row in paired)
    candidate_total = sum(int(bool(row["candidate_success"])) for row in paired)
    main_macro = float(
        np.mean([np.mean([m for m, _ in values]) for values in tasks.values()])
    )
    candidate_macro = float(
        np.mean([np.mean([c for _, c in values]) for values in tasks.values()])
    )
    payload = {
        "schema_version": 1,
        "kind": "full_context_quick_development_gate",
        "full_context_protocol": protocol_attestation(),
        "model_adapter": "pi05",
        "combined_waves": True,
        "wave_seeds": wave_seeds,
        "wave_roots": [str(Path(value).expanduser().resolve()) for value in args.wave_root],
        "candidate_plan": str(plan_path),
        "candidate_plan_sha256": sha256_file(plan_path),
        "quantvla_table1_bytes": TABLE1_QUANTVLA_BYTES["pi05"],
        "fixed_bytes": fixed_bytes("pi05"),
        "table1_total_static_budget_bytes": static_budget,
        "table1_total_static_bytes": static_total,
        "bytes_pass": bytes_pass,
        "paired_rows": paired,
        "main_successes": main_total,
        "candidate_successes": candidate_total,
        "main_task_macro": main_macro,
        "candidate_task_macro": candidate_macro,
        "paired_wins": wins,
        "paired_losses": losses,
        "passes_success_gate": bool(
            candidate_total > main_total and candidate_macro >= main_macro and wins > losses
        ),
        "advance_to_table1": False,
        "parity_statement": (
            "compression parity: candidate matches main at much higher "
            "compression (172 vs 80 W4 layers); not a success-rate win"
            if candidate_total <= main_total and wins <= losses
            else "candidate shows a success edge over main on the expanded quick"
        ),
    }
    output = Path(args.out).expanduser().resolve()
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "out": str(output),
                "episodes_per_config": len(paired),
                "main": main_total,
                "candidate": candidate_total,
                "W": wins,
                "L": losses,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
