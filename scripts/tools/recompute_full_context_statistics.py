#!/usr/bin/env python3
"""Recompute selection-only full-context statistics after the SE correction.

The source score documents contain per-sequence D_func/D_PAC measurements and
are therefore reusable: the corrected function changes only the task-level
uncertainty calculation applied after scoring.  This script accepts only the
content-attested legacy selection-core hash and records every reused source.
It never rewrites a frozen plan or a closed-loop result.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from quantvla_cross_model_protocol import (
    require_protocol_attestation as require_cross_model_attestation,
)
from quantvla_full_context import (
    BudgetItem,
    conservative_fp16_benefit,
    paired_candidate_summary,
    protocol_attestation,
    sha256_file,
    v2_coordinate_candidates,
)
from quantvla_outputimpact import atomic_json
from select_full_context_protection import finalize_candidate_plan, set_plan_mask


ROOT = Path(__file__).resolve().parents[2]
LEGACY_SELECTION_CORE_SHA256 = (
    "18e9131b85cb08182dada7255daaf0924964dc7b157bdfc613a97fc8b156e985"
)
LEGACY_FULL_CONTEXT_ATTESTATION = {
    "method_id": "full_context_fp16_protection_v1",
    "protocol_sha256": "71319d8076317f4027649961e87973909e5863c68bd37c27fda5d60194254635",
    "protocol_file": str(ROOT / "scripts/quantvla_full_context_protocol.json"),
    "protocol_file_sha256": "0da2356fca7883d64ff60c20425f9198d09c4b5b9345cb2bd44375ff713f2e00",
    "selection_core_sha256": LEGACY_SELECTION_CORE_SHA256,
    "metric_core_sha256": "89cf552a572d4c4d0657ddafa67ba29b5ef855413bdc45c0a94218f0d703a4d1",
    "quick_statistics_sha256": "15cdc55116f73c3a2170918acc30d564dcc9dc8c086cab6145f9fd106e884d88",
    "formal_statistics_sha256": "7dba6c94398c1c9702f6a5b38309a598c5226e022f9cb7622edb0bd6d806eb40",
}


def load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def require_reanalysis_source(value: Mapping[str, Any], *, source: str) -> str:
    """Accept current evidence or the one known pre-fix selection core."""
    require_cross_model_attestation(value, source=source)
    actual = dict(value.get("full_context_protocol") or {})
    expected = protocol_attestation()
    if actual == expected:
        return "current_corrected_core"
    if actual == LEGACY_FULL_CONTEXT_ATTESTATION:
        return "legacy_pre_fix_core_reanalyzed"
    drift = {
        key: (actual.get(key), expected.get(key), LEGACY_FULL_CONTEXT_ATTESTATION.get(key))
        for key in sorted(set(actual) | set(expected) | set(LEGACY_FULL_CONTEXT_ATTESTATION))
        if actual.get(key) not in (expected.get(key), LEGACY_FULL_CONTEXT_ATTESTATION.get(key))
    }
    raise ValueError(f"{source}: unrecognized full-context provenance: {drift}")


def require_score_document(value: Mapping[str, Any], *, source: str) -> str:
    lineage = require_reanalysis_source(value, source=source)
    if value.get("complete") is not True:
        raise ValueError(f"{source}: score document is incomplete")
    if value.get("selection_noise", "A") != "A":
        raise ValueError(f"{source}: only noise-A evidence may be reanalyzed")
    if any(bool(value.get(key, False)) for key in ("uses_success_labels", "uses_task_success")):
        raise ValueError(f"{source}: success-label leakage")
    return lineage


def source_row(path: Path, lineage: str) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "lineage": lineage}


def candidate_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "objective": float(summary["objective"]),
        "eligible": bool(summary["eligible"]),
        "component_constraints_pass": bool(summary["component_constraints_pass"]),
        "components": {
            key: {
                "mean": float(row["mean"]),
                "se": float(row["se"]),
                "passes": bool(row["passes"]),
            }
            for key, row in summary["components"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interventions",
        default=ROOT / "runs/full_context_v2/p2/interventions_dynamic/manifest.json",
    )
    parser.add_argument(
        "--flip-scores",
        default=ROOT / "runs/full_context_v2/p2/flip_scores/flip_scores_cross_split.json",
    )
    parser.add_argument(
        "--fullnet-scores",
        default=ROOT / "runs/full_context_v2/p2/fullnet_scores/fullnet_scores_cross_split.json",
    )
    parser.add_argument(
        "--static-scores",
        default=ROOT / "runs/full_context_v2/p2/a8_scores/scores_static_a8_cross_split.json",
    )
    parser.add_argument(
        "--dynamic-scores",
        default=ROOT / "runs/full_context_v2/p2/a8_scores/scores_dynamic_a8_cross_split.json",
    )
    parser.add_argument(
        "--a16-scores",
        default=ROOT / "runs/full_context_v2/p2/a8_scores/scores_fp16_cross_split.json",
    )
    parser.add_argument(
        "--out",
        default=ROOT / "runs/full_context_v2/statistics_correction/corrected_statistics.json",
    )
    parser.add_argument(
        "--proposal-out-dir",
        default=ROOT / "runs/full_context_v2/statistics_correction/proposals",
    )
    args = parser.parse_args()

    paths: dict[str, Path] = {}
    documents: dict[str, dict[str, Any]] = {}
    lineage: dict[str, str] = {}
    for key, raw_path in {
        "interventions": args.interventions,
        "flip_scores": args.flip_scores,
        "fullnet_scores": args.fullnet_scores,
        "static_scores": args.static_scores,
        "dynamic_scores": args.dynamic_scores,
        "a16_scores": args.a16_scores,
    }.items():
        path, document = load_json(raw_path)
        paths[key], documents[key] = path, document
        if key == "interventions":
            lineage[key] = require_reanalysis_source(document, source=str(path))
        else:
            lineage[key] = require_score_document(document, source=str(path))

    manifest = documents["interventions"]
    flip_scores = documents["flip_scores"]["scores"]
    candidate_rows = {row["candidate_id"]: row for row in manifest["candidates"]}
    if set(candidate_rows) != set(flip_scores):
        raise ValueError("flip score/candidate inventory mismatch")
    baseline = flip_scores["context_base"]
    current_protected = set(candidate_rows["context_base"]["protected_layers"])
    items: list[BudgetItem] = []
    positive: list[dict[str, Any]] = []
    for identifier, row in sorted(candidate_rows.items()):
        flip = row.get("flip")
        if not flip:
            continue
        name = str(flip["layer"])
        benefit = conservative_fp16_benefit(
            current_is_fp16=name in current_protected,
            flip=flip_scores[identifier],
            baseline=baseline,
        )
        item = BudgetItem(
            name=name,
            extra_bytes=int(manifest["byte_rows"][name]["extra_fp16_bytes"]),
            benefit_d_func=float(benefit["d_func"]),
            benefit_d_pac=float(benefit["d_pac"]),
        )
        items.append(item)
        if item.benefit_d_func > 0.0 and item.benefit_d_pac > 0.0:
            positive.append(
                {
                    "candidate_id": identifier,
                    "layer": name,
                    "flip": flip,
                    "benefit_d_func": item.benefit_d_func,
                    "benefit_d_pac": item.benefit_d_pac,
                    "benefit_sum": item.benefit_d_func + item.benefit_d_pac,
                }
            )
    positive.sort(key=lambda row: (-row["benefit_sum"], row["layer"]))
    capacity = int(manifest["budget_bytes"]) - int(manifest["all_w4_total_bytes"])
    proposals = [
        {
            "candidate_id": identifier,
            "retained_fp16_layers": len(protected),
            "protected_layers": list(protected),
            "predicted_utility": float(utility),
        }
        for identifier, protected, utility in v2_coordinate_candidates(
            items, capacity=capacity, historical_protected=[]
        )
    ]

    base_plan_path, base_plan = load_json(manifest["base_full_w4_plan"])
    proposal_out_dir = Path(args.proposal_out_dir).expanduser().resolve()
    proposal_out_dir.mkdir(parents=True, exist_ok=True)
    proposal_artifacts: list[dict[str, Any]] = []
    candidate_specs = [
        ("context_base", tuple(sorted(current_protected)), 0.0),
        *[
            (
                str(row["candidate_id"]),
                tuple(str(name) for name in row["protected_layers"]),
                float(row["predicted_utility"]),
            )
            for row in proposals
        ],
    ]
    for identifier, protected_tuple, utility in candidate_specs:
        protected = set(protected_tuple)
        plan = set_plan_mask(
            copy.deepcopy(base_plan),
            protected,
            reason_prefix="corrected_full_context_protection",
        )
        plan = finalize_candidate_plan(
            plan,
            identifier=identifier,
            model=str(manifest["model_adapter"]),
            activation_mode=str(manifest["activation_mode"]),
            round_index=int(manifest["round"]),
            protected=protected,
            byte_rows=manifest["byte_rows"],
            manifest_sha=sha256_file(paths["interventions"]),
            flow_steps=int(manifest["flow_steps"]),
        )
        plan["meta"].update(
            {
                "statistics_correction": "correct_jackknife_se_of_task_mean",
                "legacy_score_reanalysis": True,
                "predicted_utility": utility,
            }
        )
        plan_path = proposal_out_dir / f"{identifier}.json"
        atomic_json(plan_path, plan)
        proposal_artifacts.append(
            {
                "candidate_id": identifier,
                "path": str(plan_path),
                "sha256": sha256_file(plan_path),
                "predicted_utility": utility,
                "retained_fp16_layers": len(protected),
                "protected_layers": sorted(protected),
                "requires_complete_policy_scoring": identifier != "context_base",
            }
        )
    proposal_manifest_path = proposal_out_dir / "manifest.json"
    atomic_json(
        proposal_manifest_path,
        {
            "schema_version": 1,
            "kind": "corrected_full_context_budgeted_proposal_manifest",
            "model_adapter": manifest["model_adapter"],
            "activation_mode": manifest["activation_mode"],
            "flow_steps": manifest["flow_steps"],
            "full_context_protocol": protocol_attestation(),
            "legacy_intervention_manifest": source_row(
                paths["interventions"], lineage["interventions"]
            ),
            "legacy_flip_scores": source_row(
                paths["flip_scores"], lineage["flip_scores"]
            ),
            "base_full_w4_plan": {
                "path": str(base_plan_path),
                "sha256": sha256_file(base_plan_path),
            },
            "statistics_correction": "correct_jackknife_se_of_task_mean",
            "uses_success_labels": False,
            "selection_noise": "A",
            "candidates": proposal_artifacts,
        },
    )

    fullnet_scores = documents["fullnet_scores"]["scores"]
    fullnet_baseline = fullnet_scores["context_base"]
    fullnet = {
        identifier: candidate_summary(
            paired_candidate_summary(score, fullnet_baseline)
        )
        for identifier, score in sorted(fullnet_scores.items())
        if identifier != "context_base"
    }

    activation_scores = {
        key: documents[f"{key}_scores"]["scores"]["frozen"]
        for key in ("static", "dynamic", "a16")
    }
    static_vs_a16 = paired_candidate_summary(
        activation_scores["static"], activation_scores["a16"]
    )
    dynamic_vs_a16 = paired_candidate_summary(
        activation_scores["dynamic"], activation_scores["a16"]
    )
    dynamic_vs_static = paired_candidate_summary(
        activation_scores["dynamic"], activation_scores["static"]
    )

    payload = {
        "schema_version": 1,
        "kind": "full_context_statistics_correction_v1",
        "correction_scope": "selection_statistics_only",
        "closed_loop_results_changed": False,
        "frozen_plan_changed": False,
        "old_erroneous_se": "sqrt((n-1)/n * sum((x_i-mean(x))^2))",
        "corrected_jackknife_se": "sqrt(sum((x_i-mean(x))^2)/(n*(n-1)))",
        "inflation_factor_of_old_formula": "n_tasks-1",
        "active_full_context_protocol": protocol_attestation(),
        "sources": {
            key: source_row(paths[key], lineage[key]) for key in sorted(paths)
        },
        "local_interventions": {
            "baseline_id": "context_base",
            "n_flips": len(items),
            "n_positive_both_metrics": len(positive),
            "positive_both_metrics": positive,
            "corrected_proposal_candidates": proposals,
            "new_candidates_require_complete_policy_scoring": True,
            "corrected_proposal_manifest": {
                "path": str(proposal_manifest_path),
                "sha256": sha256_file(proposal_manifest_path),
            },
        },
        "existing_complete_policy_candidates": {
            "n_candidates": len(fullnet),
            "n_eligible": sum(int(row["eligible"]) for row in fullnet.values()),
            "summaries": fullnet,
        },
        "activation_attribution": {
            "static_vs_a16": candidate_summary(static_vs_a16),
            "dynamic_vs_a16": candidate_summary(dynamic_vs_a16),
            "dynamic_vs_static": candidate_summary(dynamic_vs_static),
            "interpretation": (
                "The registered static table remains rejected. Dynamic A8 is "
                "much closer to A16 and strictly improves over the registered "
                "static table, but is not equivalent to A16 under the strict gate."
            ),
        },
    }
    output = Path(args.out).expanduser().resolve()
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "out": str(output),
                "positive_flips": len(positive),
                "corrected_proposals": len(proposals),
                "eligible_existing_fullnet": payload[
                    "existing_complete_policy_candidates"
                ]["n_eligible"],
                "static_vs_a16_j": static_vs_a16["objective"],
                "dynamic_vs_a16_j": dynamic_vs_a16["objective"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
