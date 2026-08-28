#!/usr/bin/env python3
"""Aggregate the frozen seeds-50..59 development advancement gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantvla_cross_model_protocol import sha256_file
from quantvla_full_context import (
    PROTOCOL,
    protocol_attestation,
    quick_advancement,
    require_protocol_attestation,
)
from quantvla_outputimpact import atomic_json


def load_rows(root: Path, config: str | None) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for path in sorted(root.glob("**/*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("gr00t", "pi05"))
    parser.add_argument("--main-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--main-config")
    parser.add_argument("--candidate-config")
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--quantvla-bytes", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    main_dir = Path(args.main_dir).expanduser().resolve()
    candidate_dir = Path(args.candidate_dir).expanduser().resolve()
    plan_path = Path(args.candidate_plan).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    require_protocol_attestation(plan.get("meta") or {}, source=str(plan_path))
    byte_limit = int(
        PROTOCOL["byte_budget"]["maximum_quantvla_byte_multiplier"]
        * args.quantvla_bytes
    )
    bytes_pass = int(plan["total_bytes"]) <= byte_limit
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
    result = quick_advancement(paired)
    payload = {
        "schema_version": 1,
        "kind": "full_context_quick_development_gate",
        "full_context_protocol": protocol_attestation(),
        "model_adapter": args.model,
        "main_dir": str(main_dir),
        "candidate_dir": str(candidate_dir),
        "candidate_plan": str(plan_path),
        "candidate_plan_sha256": sha256_file(plan_path),
        "quantvla_bytes": args.quantvla_bytes,
        "maximum_candidate_bytes": byte_limit,
        "candidate_bytes": int(plan["total_bytes"]),
        "bytes_pass": bytes_pass,
        "paired_rows": paired,
        **result,
        "advance_to_table1": bool(bytes_pass and result["passes_success_gate"]),
    }
    atomic_json(args.out, payload)
    print(json.dumps({"out": str(Path(args.out).resolve()), "advance": payload["advance_to_table1"]}, indent=2))


if __name__ == "__main__":
    main()
