#!/usr/bin/env python3
"""Aggregate the preregistered GR00T M0-vs-full-W4 causal ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aggregate_full_context_table1 import (
    exact_mcnemar,
    hierarchical_bootstrap,
    load_rows,
    rates,
)
from quantvla_cross_model_protocol import sha256_file
from quantvla_outputimpact import atomic_json


def require_config(rows: dict[tuple[str, int], dict[str, Any]], expected: str) -> None:
    observed = {str(row.get("config")) for row in rows.values()}
    if observed != {expected}:
        raise ValueError(f"expected only config={expected}, got {sorted(observed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--full-w4-dir", required=True)
    parser.add_argument("--m0-dir", required=True)
    parser.add_argument("--equivalence-margin", type=float, default=0.02)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "gr00t_mask_causal_preregistration":
        raise ValueError("wrong causal-ablation manifest kind")
    if manifest.get("result_feedback_allowed") is not False:
        raise ValueError("formal causal ablation must forbid result feedback")
    registered_margin = float(manifest["statistics"]["practical_equivalence_margin"])
    if float(args.equivalence_margin) != registered_margin:
        raise ValueError("equivalence margin differs from preregistration")

    full_w4 = load_rows(Path(args.full_w4_dir).expanduser().resolve())
    m0 = load_rows(Path(args.m0_dir).expanduser().resolve())
    require_config(full_w4, "full_w4_dyrange")
    require_config(m0, "full_context_v2")

    full_rates = rates(full_w4)
    m0_rates = rates(m0)
    m0_wins = sum(
        int(bool(m0[key]["success"]) and not bool(full_w4[key]["success"]))
        for key in m0
    )
    m0_losses = sum(
        int(bool(full_w4[key]["success"]) and not bool(m0[key]["success"]))
        for key in m0
    )
    pvalue = exact_mcnemar(m0_wins, m0_losses)
    bootstrap = hierarchical_bootstrap(m0, full_w4)
    delta_micro = float(m0_rates["micro_success_rate"] - full_rates["micro_success_rate"])
    delta_macro = float(
        m0_rates["task_macro_success_rate"] - full_rates["task_macro_success_rate"]
    )
    directional_positive = delta_micro > 0.0 and delta_macro > 0.0
    directional_negative = delta_micro < 0.0 and delta_macro < 0.0
    significant = bool(
        bootstrap["ci95_low"] > 0.0
        or bootstrap["ci95_high"] < 0.0
        or pvalue < 0.05
    )
    equivalent = bool(
        bootstrap["ci95_low"] >= -registered_margin
        and bootstrap["ci95_high"] <= registered_margin
    )
    if directional_positive and significant:
        conclusion = "m0_mask_positive_causal_effect"
    elif directional_negative and significant:
        conclusion = "m0_mask_negative_causal_effect"
    elif equivalent:
        conclusion = "practically_equivalent_within_registered_margin"
    else:
        conclusion = "inconclusive"

    payload = {
        "schema_version": 1,
        "kind": "gr00t_mask_causal_aggregate",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "result_feedback_allowed": False,
        "hypothesis": (
            "closed-loop causal effect of the inherited 100-W4/16-FP16 mask "
            "relative to 116-W4 under the same Hessian-W4 and DyRange-A8 runtime"
        ),
        "m0": m0_rates,
        "full_w4": full_rates,
        "effect_m0_minus_full_w4": {
            "micro_success_rate": delta_micro,
            "task_macro_success_rate": delta_macro,
            "paired_m0_wins": m0_wins,
            "paired_m0_losses": m0_losses,
            "paired_ties": len(m0) - m0_wins - m0_losses,
            "exact_two_sided_mcnemar_p": pvalue,
            "task_then_seed_hierarchical_bootstrap": bootstrap,
            "practical_equivalence_margin": registered_margin,
        },
        "registered_decision": {
            "significant": significant,
            "practically_equivalent": equivalent,
            "conclusion": conclusion,
        },
    }
    atomic_json(args.out, payload)
    print(json.dumps({"out": str(Path(args.out).resolve()), "conclusion": conclusion}, indent=2))


if __name__ == "__main__":
    main()
