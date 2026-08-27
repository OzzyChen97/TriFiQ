#!/usr/bin/env python3
"""Merge disjoint SoftFold score shards and require exact 9x9 coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from fit_softfold_compensation import GRID
from quantvla_cross_model_protocol import require_protocol_attestation
from quantvla_dynamic_a8_protocol import (
    require_protocol_attestation as require_dynamic_a8_protocol_attestation,
)


def canonical_gate(row: dict[str, Any]) -> tuple[float, float]:
    gate = row.get("gate") or {}
    return round(float(gate["atm"]), 6), round(float(gate["errorfold"]), 6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    merged: dict[str, Any] = {}
    provenance = []
    invariant_keys = (
        "checkpoint_sha256",
        "buffer_sha256",
        "artifact_calibration_buffer_sha256",
        "selection_metric",
        "plan_sha256",
        "quant_plan_sha256",
        "a8_sha256",
        "hessian_w4_sha256",
        "raw_correction_sha256",
        "pack_dir_sha256",
        "source_sha256",
        "cross_model_protocol",
        "dynamic_a8_protocol",
        "activation_mode",
        "quantization_selection",
    )
    invariants: dict[str, Any] | None = None
    seen_gates: dict[tuple[float, float], str] = {}
    scores: dict[str, Any] = {}
    for raw_path in args.input:
        path = Path(raw_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        require_protocol_attestation(payload, source=str(path))
        if payload.get("activation_mode") == "dynamic_a8":
            require_dynamic_a8_protocol_attestation(payload, source=str(path))
        current = {key: payload.get(key) for key in invariant_keys}
        if invariants is None:
            invariants = current
            merged = {key: payload.get(key) for key in payload if key not in {"scores", "best"}}
        elif current != invariants:
            raise ValueError(f"score-shard provenance mismatch: {path}")
        provenance.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "shard": payload.get("softfold_grid_shard"),
            }
        )
        for config_id, row in (payload.get("scores") or {}).items():
            gate = canonical_gate(row)
            if config_id in scores or gate in seen_gates:
                raise ValueError(f"duplicate SoftFold candidate {config_id}/{gate}")
            scores[config_id] = row
            seen_gates[gate] = config_id

    expected = {(round(a, 6), round(b, 6)) for a in GRID for b in GRID}
    if set(seen_gates) != expected:
        missing = sorted(expected - set(seen_gates))
        extra = sorted(set(seen_gates) - expected)
        raise ValueError(f"incomplete SoftFold grid: missing={missing[:8]} extra={extra[:8]}")

    metric = str(merged.get("selection_metric") or "d_pac")
    best_id = min(scores, key=lambda key: float(scores[key][metric]))
    merged.update(
        {
            "kind": "softfold_grid_score_merged",
            "softfold_grid": True,
            "softfold_grid_size": 81,
            "softfold_grid_shard": None,
            "source_shards": provenance,
            "scores": dict(sorted(scores.items())),
            "best": {
                "config_id": best_id,
                "metric": metric,
                "value": float(scores[best_id][metric]),
                "d_func": float(scores[best_id]["d_func"]),
                "d_pac": float(scores[best_id]["d_pac"]),
                "selection_rule": "diagnostic_argmin_only",
                "final_selection_rule": "paired_one_standard_error_then_minimum_correction_norm_gate_sum_interaction",
            },
        }
    )
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "candidates": len(scores),
                "best": merged["best"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
