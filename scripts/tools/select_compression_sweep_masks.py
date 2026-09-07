#!/usr/bin/env python3
"""Compression-rate masks for the GR00T DyPAC sweep (candidate scope).

The frozen DyPAC/FCP allocation rule is inverted from the v2 prune tool: every
adapter-bound layer carries a protection damage from the frozen cross-split
single-flip scores (W4->FP16 restores measured against the full-W4 context
with the frozen D_PAC objective), and layers are protected from all-W4 in
descending damage-per-extra-byte order until the rate budget is spent::

    budget = fp16_total_bytes / target_compression   (candidate scope)
    fp16_total_bytes = 1,811,939,328, all-W4 = 509,607,936

All five requested rates (1.2, 1.6, 2.0, 2.4, 2.8) are feasible (candidate
maximum 3.5556x).  Outputs are DyPAC-format deployment plans (schema_version
5) plus a frozen manifest with byte totals, mask counts, and input hashes.
Re-running with the same frozen inputs must reproduce identical files; the
tool fails loudly otherwise.  Byte totals are strictly monotonic across the
rates by construction and asserted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from quantvla_cross_model_protocol import sha256_file, validate_quant_plan
from quantvla_full_context import paired_candidate_summary
from quantvla_table1_bytes import (
    TABLE1_FP16_BYTES,
    fixed_bytes,
    table1_total_static_bytes,
    table1_total_static_compression,
)

RATES = (1.2, 1.6, 2.0, 2.4, 2.8)
FP16_TOTAL_BYTES = 1_811_939_328
ALL_W4_BYTES = 509_607_936
FIXED_BYTES = 327_352_320
MAXIMUM_CANDIDATE_COMPRESSION = FP16_TOTAL_BYTES / ALL_W4_BYTES


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def build_damages(
    byte_rows: dict[str, dict[str, int]],
    flip_manifest: list[dict[str, Any]],
    flip_scores: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Per-layer protection damage from the frozen cross-split flip scores.

    gr00t round-1 flips measured as W4->FP16 restore: the more negative the
    paired objective, the more FP16 helps -> damage = -objective.  Flips
    measured as FP16->W4 removal: the more positive the objective, the more
    the removal hurts -> damage = +objective.  (Identical to the v2 prune
    tool's damage mapping.)
    """
    baseline = flip_scores["context_base"]
    layer_to_flip = {
        candidate["flip"]["layer"]: candidate
        for candidate in flip_manifest
        if candidate.get("flip")
    }
    damages: dict[str, float] = {}
    for name in byte_rows:
        candidate = layer_to_flip.get(name)
        if candidate is None:
            raise ValueError(f"no counterfactual score for {name}")
        score = flip_scores[candidate["candidate_id"]]
        objective = float(paired_candidate_summary(score, baseline)["objective"])
        damages[name] = (
            objective if candidate["flip"]["from"] == "fp16" else -objective
        )
    return damages


def select_protected(
    byte_rows: dict[str, dict[str, int]], damages: dict[str, float], budget: float
) -> tuple[list[str], int]:
    """Greedily spend the FP16 budget on maximum damage removed per byte."""
    total = sum(int(row["w4_bytes"]) for row in byte_rows.values())
    if total > float(budget) + 1e-6:
        raise ValueError("target compression is infeasible below the all-W4 bytes")
    protected: list[str] = []
    ranked = sorted(
        byte_rows.items(),
        key=lambda item: (
            -float(damages[item[0]])
            / max(float(item[1]["fp16_bytes"]) - float(item[1]["w4_bytes"]), 1.0),
            -float(damages[item[0]]),
            item[0],
        ),
    )
    for name, row in ranked:
        delta = float(row["fp16_bytes"]) - float(row["w4_bytes"])
        if total + delta <= float(budget) + 1e-6:
            protected.append(name)
            total += int(delta)
    return sorted(protected), int(total)


def build_plan(
    *,
    rate: float,
    base: dict[str, Any],
    byte_rows: dict[str, dict[str, int]],
    damages: dict[str, float],
    protected: list[str],
    total: int,
    budget: float,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    plan = copy.deepcopy(base)
    protected_set = set(protected)
    for name, entry in plan["layers"].items():
        if name in protected_set:
            entry.update(
                {"bits": None, "skip": True, "reason": "compression_sweep_rate_fp16"}
            )
        else:
            entry.update(
                {"bits": 4, "group": 64, "skip": False, "reason": "compression_sweep_rate_w4"}
            )
        entry["sweep"] = {
            "damage": float(damages[name]),
            "fp16_bytes": int(byte_rows[name]["fp16_bytes"]),
            "w4_bytes": int(byte_rows[name]["w4_bytes"]),
        }
    meta = {
        "kind": "compression_sweep_rate_mask",
        "model_adapter": "gr00t",
        "activation_mode": "dynamic_a8",
        "flow_steps": 4,
        "budget_anchor": "candidate_scope_target_compression",
        "fixed_bytes": FIXED_BYTES,
        "target_compression": float(rate),
        "target_compression_scope": "candidate",
        "runtime_selector": False,
        "runtime_correction": False,
        "uses_cka": False,
        "uses_cs": False,
        "uses_task_ids_for_stratified_statistics": True,
        "uses_task_labels_for_routing_or_task_specific_mask": False,
        "uses_success_labels": False,
        "uses_gradients": False,
        "uses_extra_training_data": False,
        "selection_rule": (
            "from all-W4, protect descending damage/extra_byte where damage is "
            "the frozen cross-split single-flip D_PAC objective mapped exactly "
            "like the v2 prune tool; fill the budget (ceiling semantics)"
        ),
        "damage_source": input_hashes,
    }
    plan["schema_version"] = max(int(plan.get("schema_version", 1)), 5)
    plan["meta"] = meta
    plan.update(
        {
            "target_compression": float(rate),
            "budget_bytes": float(budget),
            "fp16_total_bytes": float(FP16_TOTAL_BYTES),
            "all_w4_total_bytes": float(ALL_W4_BYTES),
            "total_bytes": int(total),
            "fixed_bytes": FIXED_BYTES,
            "table1_total_static_bytes": int(table1_total_static_bytes("gr00t", total)),
            "table1_total_static_compression": float(
                table1_total_static_compression("gr00t", total)
            ),
            "achieved_candidate_compression": float(FP16_TOTAL_BYTES / total),
            "quantized_w4_layers": len(byte_rows) - len(protected),
            "retained_fp16_layers": len(protected),
            "protected_layers": sorted(protected),
        }
    )
    return plan


def rate_id(rate: float) -> str:
    if abs(rate - MAXIMUM_CANDIDATE_COMPRESSION) < 1e-9:
        return "ratemax"
    return f"rate{int(rate * 10)}"  # 1.2 -> rate12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-plan",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v2/mask_causal/"
        "full_w4_current_protocol.json",
    )
    parser.add_argument(
        "--byte-rows-manifest",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v2/p2/"
        "interventions_dynamic/manifest.json",
    )
    parser.add_argument(
        "--flip-scores",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v2/p2/flip_scores/"
        "flip_scores_cross_split.json",
    )
    parser.add_argument(
        "--rates", default="1.2,1.6,2.0,2.4,2.8",
        help=(
            "Comma-separated candidate-scope target compressions; 'max' adds "
            "the all-W4 extreme point (candidate maximum)."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rate_values: list[float] = []
    has_max = False
    for value in args.rates.split(","):
        value = value.strip()
        if not value:
            continue
        if value.lower() == "max":
            has_max = True
        else:
            rate_values.append(float(value))
    if has_max:
        rate_values.append(MAXIMUM_CANDIDATE_COMPRESSION)
    rates = rate_values
    if not rates or len(rates) != len(set(rates)):
        raise SystemExit("--rates must be unique positive values")
    for rate in rates:
        if not 1.0 < rate <= MAXIMUM_CANDIDATE_COMPRESSION + 1e-9:
            raise SystemExit(
                f"rate {rate:g} outside feasible range (1, {MAXIMUM_CANDIDATE_COMPRESSION:.6g}]"
            )

    base = load_json(args.base_plan)
    manifest = load_json(args.byte_rows_manifest)
    byte_rows = manifest["byte_rows"]
    flip_scores = load_json(args.flip_scores)["scores"]
    flip_manifest = manifest["candidates"]
    damages = build_damages(byte_rows, flip_manifest, flip_scores)

    input_hashes = {
        "base_plan_sha256": sha256_file(args.base_plan),
        "byte_rows_manifest_sha256": sha256_file(args.byte_rows_manifest),
        "flip_scores_sha256": sha256_file(args.flip_scores),
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_out: dict[str, Any] = {
        "schema_version": 1,
        "kind": "compression_sweep_masks",
        "rates": rates,
        "fp16_total_bytes": FP16_TOTAL_BYTES,
        "all_w4_bytes": ALL_W4_BYTES,
        "fixed_bytes": FIXED_BYTES,
        "maximum_candidate_compression": MAXIMUM_CANDIDATE_COMPRESSION,
        "input_hashes": input_hashes,
        "rates_manifest": {},
    }
    previous_total = None
    previous_budget = None
    for rate in rates:
        budget = FP16_TOTAL_BYTES / rate
        protected, total = select_protected(byte_rows, damages, budget)
        # Higher rate -> smaller budget -> strictly fewer total bytes.
        if previous_total is not None and total >= previous_total:
            raise SystemExit(
                f"non-monotonic bytes at rate {rate:g} "
                f"({total} >= {previous_total})"
            )
        if previous_budget is not None and budget >= previous_budget:
            raise SystemExit(f"non-monotonic budget at rate {rate:g}")
        previous_total, previous_budget = total, budget
        plan = build_plan(
            rate=rate, base=base, byte_rows=byte_rows, damages=damages,
            protected=protected, total=total, budget=budget,
            input_hashes=input_hashes,
        )
        validation = validate_quant_plan(
            plan, model="gr00t", source=f"compression sweep rate {rate:g}"
        )
        if validation["quantized_w4_layers"] != plan["quantized_w4_layers"]:
            raise SystemExit(f"rate {rate:g}: plan validation count mismatch")
        identifier = rate_id(rate)
        path = out_dir / f"{identifier}.plan.json"
        payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise SystemExit(f"mask changed on re-materialization: {path}")
        else:
            path.write_text(payload, encoding="utf-8")
        manifest_out["rates_manifest"][identifier] = {
            "rate": rate,
            "budget_bytes": float(budget),
            "total_bytes": int(total),
            "static_bytes": int(table1_total_static_bytes("gr00t", total)),
            "static_compression": float(
                table1_total_static_compression("gr00t", total)
            ),
            "achieved_candidate_compression": float(FP16_TOTAL_BYTES / total),
            "w4_layers": len(byte_rows) - len(protected),
            "fp16_layers": len(protected),
            "protected_layers": sorted(protected),
            "plan_sha256": sha256_file(path),
        }

    manifest_path = out_dir / "masks_manifest.json"
    payload = json.dumps(manifest_out, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != payload:
            raise SystemExit(f"masks manifest changed on re-materialization: {manifest_path}")
    else:
        manifest_path.write_text(payload, encoding="utf-8")

    print(json.dumps(
        {
            identifier: {
                "rate": entry["rate"],
                "w4": entry["w4_layers"],
                "fp16": entry["fp16_layers"],
                "static_compression": round(entry["static_compression"], 4),
                "total_bytes": entry["total_bytes"],
            }
            for identifier, entry in manifest_out["rates_manifest"].items()
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
