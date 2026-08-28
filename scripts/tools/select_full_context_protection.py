#!/usr/bin/env python3
"""Propose and freeze byte-bounded masks from full-context counterfactuals."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from quantvla_cross_model_protocol import (
    protocol_attestation as cross_model_attestation,
    require_protocol_attestation as require_cross_model_attestation,
    sha256_file,
    validate_quant_plan,
)
from quantvla_dynamic_a8_protocol import (
    protocol_attestation as dynamic_a8_protocol_attestation,
)
from quantvla_full_context import (
    BudgetItem,
    PROTOCOL,
    conservative_fp16_benefit,
    exact_weighted_knapsack,
    protocol_attestation as full_context_attestation,
    require_protocol_attestation as require_full_context_attestation,
    select_activation_mode,
    select_frozen_candidate,
)
from quantvla_outputimpact import atomic_json


def load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def score_payload(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved, value = load_json(path)
    require_cross_model_attestation(value, source=str(resolved))
    require_full_context_attestation(value, source=str(resolved))
    if value.get("complete") is not True:
        raise ValueError(f"{resolved}: score artifact is incomplete")
    if value.get("selection_noise", "A") != "A":
        raise ValueError(f"{resolved}: only noise-A may enter selection")
    if any(bool(value.get(key, False)) for key in ("uses_success_labels", "uses_task_success")):
        raise ValueError(f"{resolved}: success-label leakage")
    return resolved, value


def set_plan_mask(
    base: dict[str, Any], protected: set[str], *, reason_prefix: str
) -> dict[str, Any]:
    plan = copy.deepcopy(base)
    for name, row in plan["layers"].items():
        if name in protected:
            row.update({"bits": None, "skip": True, "reason": f"{reason_prefix}_fp16"})
        else:
            row.update({"bits": 4, "group": 64, "skip": False, "reason": f"{reason_prefix}_w4"})
        for forbidden in ("errorfold", "blocksoftfold", "atm", "ohb"):
            row.pop(forbidden, None)
    return plan


def finalize_candidate_plan(
    plan: dict[str, Any],
    *,
    identifier: str,
    model: str,
    activation_mode: str,
    round_index: int,
    protected: set[str],
    byte_rows: dict[str, dict[str, int]],
    manifest_sha: str,
    flow_steps: int,
) -> dict[str, Any]:
    all_w4 = sum(row["w4_bytes"] for row in byte_rows.values())
    fp16_total = sum(row["fp16_bytes"] for row in byte_rows.values())
    total = all_w4 + sum(byte_rows[name]["extra_fp16_bytes"] for name in protected)
    budget = int(PROTOCOL["byte_budget"]["maximum_quantvla_byte_multiplier"] * all_w4)
    if total > budget:
        raise ValueError(f"{identifier}: exact byte budget exceeded")
    meta = dict(plan.get("meta") or {})
    for forbidden in ("errorfold", "atm", "ohb", "blocksoftfold", "runtime_selector"):
        meta.pop(forbidden, None)
    meta.update(
        {
            "kind": "full_context_fp16_protection_candidate",
            "method_id": PROTOCOL["method_id"],
            "candidate_id": identifier,
            "round": round_index,
            "model_adapter": model,
            "activation_mode": activation_mode,
            "flow_steps": int(flow_steps),
            "cross_model_protocol": cross_model_attestation(),
            "full_context_protocol": full_context_attestation(),
            "dynamic_a8_protocol": (
                dynamic_a8_protocol_attestation()
                if activation_mode == "dynamic_a8"
                else None
            ),
            "counterfactual_manifest_sha256": manifest_sha,
            "target_compression_scope": "candidate",
            "target_compression": float(fp16_total / budget),
            "runtime_selector": False,
            "runtime_correction": False,
            "uses_cka": False,
            "uses_cs": False,
            "uses_task_labels": False,
            "uses_success_labels": False,
            "uses_gradients": False,
            "uses_extra_training_data": False,
        }
    )
    plan["meta"] = meta
    plan["schema_version"] = max(int(plan.get("schema_version", 1)), 5)
    plan.update(
        {
            "budget_bytes": budget,
            "all_w4_total_bytes": all_w4,
            "fp16_total_bytes": fp16_total,
            "total_bytes": total,
            "achieved_candidate_compression": float(fp16_total / total),
            "achieved_compression": float(fp16_total / total),
            "quantized_w4_layers": len(byte_rows) - len(protected),
            "retained_fp16_layers": len(protected),
            "protected_layers": sorted(protected),
        }
    )
    validate_quant_plan(plan, model=model, source=identifier)
    return plan


def propose(args: argparse.Namespace) -> None:
    manifest_path, manifest = load_json(args.interventions)
    require_cross_model_attestation(manifest, source=str(manifest_path))
    require_full_context_attestation(manifest, source=str(manifest_path))
    scores_path, score_document = score_payload(args.scores)
    if score_document.get("model_adapter") != manifest.get("model_adapter"):
        raise ValueError("counterfactual score/model mismatch")
    candidate_rows = {row["candidate_id"]: row for row in manifest["candidates"]}
    scores = score_document["scores"]
    if set(scores) != set(candidate_rows):
        raise ValueError("counterfactual score/candidate inventory mismatch")
    baseline = scores["context_base"]
    byte_rows = manifest["byte_rows"]
    current_protected = set(candidate_rows["context_base"]["protected_layers"])
    items = []
    for identifier, row in candidate_rows.items():
        flip = row.get("flip")
        if not flip:
            continue
        name = flip["layer"]
        benefit = conservative_fp16_benefit(
            current_is_fp16=name in current_protected,
            flip=scores[identifier],
            baseline=baseline,
        )
        items.append(
            BudgetItem(
                name=name,
                extra_bytes=int(byte_rows[name]["extra_fp16_bytes"]),
                benefit_d_func=benefit["d_func"],
                benefit_d_pac=benefit["d_pac"],
            )
        )
    all_w4 = int(manifest["all_w4_total_bytes"])
    capacity = int(manifest["budget_bytes"]) - all_w4
    base_path, base_plan = load_json(manifest["base_full_w4_plan"])
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha256_file(manifest_path)
    candidates = []
    seen_masks: dict[tuple[str, ...], str] = {}

    def materialize(identifier: str, protected_tuple: tuple[str, ...], utility: float, lam: float | None) -> None:
        if protected_tuple in seen_masks:
            return
        seen_masks[protected_tuple] = identifier
        protected = set(protected_tuple)
        plan = set_plan_mask(base_plan, protected, reason_prefix="full_context_protection")
        plan = finalize_candidate_plan(
            plan,
            identifier=identifier,
            model=manifest["model_adapter"],
            activation_mode=manifest["activation_mode"],
            round_index=int(manifest["round"]),
            protected=protected,
            byte_rows=byte_rows,
            manifest_sha=manifest_sha,
            flow_steps=int(manifest["flow_steps"]),
        )
        path = out_dir / f"{identifier}.json"
        atomic_json(path, plan)
        candidates.append(
            {
                "candidate_id": identifier,
                "path": str(path),
                "sha256": sha256_file(path),
                "lambda_d_func": lam,
                "predicted_utility": utility,
                "total_bytes": plan["total_bytes"],
                "retained_fp16_layers": plan["retained_fp16_layers"],
                "protected_layers": plan["protected_layers"],
            }
        )

    materialize("context_base", tuple(sorted(current_protected)), 0.0, None)
    for lam in PROTOCOL["counterfactual_search"]["scalarization_lambdas"]:
        result = exact_weighted_knapsack(items, budget_bytes=capacity, lambda_d_func=float(lam))
        identifier = f"lambda_{str(lam).replace('.', 'p')}"
        materialize(identifier, result.protected, result.utility, float(lam))

    proposal = {
        "schema_version": 1,
        "kind": "full_context_budgeted_proposal_manifest",
        "method_id": PROTOCOL["method_id"],
        "cross_model_protocol": cross_model_attestation(),
        "full_context_protocol": full_context_attestation(),
        "model_adapter": manifest["model_adapter"],
        "activation_mode": manifest["activation_mode"],
        "flow_steps": manifest["flow_steps"],
        "round": manifest["round"],
        "intervention_manifest": str(manifest_path),
        "intervention_manifest_sha256": manifest_sha,
        "intervention_scores": str(scores_path),
        "intervention_scores_sha256": sha256_file(scores_path),
        "base_full_w4_plan": str(base_path),
        "base_full_w4_plan_sha256": sha256_file(base_path),
        "budget_bytes": manifest["budget_bytes"],
        "all_w4_total_bytes": all_w4,
        "exact_solver": "sparse_pareto_01_dynamic_programming",
        "candidates": candidates,
        "uses_success_labels": False,
        "selection_noise": "A",
    }
    output = out_dir / "manifest.json"
    atomic_json(output, proposal)
    print(json.dumps({"manifest": str(output), "candidates": len(candidates)}, indent=2))


def freeze(args: argparse.Namespace) -> None:
    manifest_path, manifest = load_json(args.proposals)
    require_cross_model_attestation(manifest, source=str(manifest_path))
    require_full_context_attestation(manifest, source=str(manifest_path))
    scores_path, document = score_payload(args.scores)
    rows = {row["candidate_id"]: row for row in manifest["candidates"]}
    scores = document["scores"]
    selection = select_frozen_candidate(
        scores=scores, baseline_id="context_base", plan_rows=rows
    )
    selected_id = selection["selected_id"]
    selected_path, selected_plan = load_json(rows[selected_id]["path"])
    meta = dict(selected_plan.get("meta") or {})
    meta.update(
        {
            "kind": "full_context_fp16_protection_frozen",
            "frozen": True,
            "selection_noise": "A",
            "noise_b_used_for_selection": False,
            "proposal_manifest_sha256": sha256_file(manifest_path),
            "full_network_scores_sha256": sha256_file(scores_path),
            "selected_candidate_id": selected_id,
            "selection_result": selection,
        }
    )
    selected_plan["meta"] = meta
    output = Path(args.out).expanduser().resolve()
    atomic_json(output, selected_plan)
    report = {
        "schema_version": 1,
        "kind": "full_context_frozen_selection_report",
        "cross_model_protocol": cross_model_attestation(),
        "full_context_protocol": full_context_attestation(),
        "model_adapter": manifest["model_adapter"],
        "activation_mode": manifest["activation_mode"],
        "round": manifest["round"],
        "proposal_manifest": str(manifest_path),
        "proposal_manifest_sha256": sha256_file(manifest_path),
        "full_network_scores": str(scores_path),
        "full_network_scores_sha256": sha256_file(scores_path),
        "selected_source": str(selected_path),
        "selected_source_sha256": sha256_file(selected_path),
        "frozen_plan": str(output),
        "frozen_plan_sha256": sha256_file(output),
        "selection": selection,
    }
    report_path = output.with_suffix(output.suffix + ".selection.json")
    atomic_json(report_path, report)
    print(json.dumps({"out": str(output), "report": str(report_path), "selected": selected_id}, indent=2))


def a8(args: argparse.Namespace) -> None:
    paths_documents = {
        key: score_payload(path)
        for key, path in {
            "static": args.static_scores,
            "dynamic": args.dynamic_scores,
            "a16": args.a16_scores,
        }.items()
    }
    documents = {key: value[1] for key, value in paths_documents.items()}
    model = documents["static"].get("model_adapter")
    if any(value.get("model_adapter") != model for value in documents.values()):
        raise ValueError("A8 attribution model mismatch")
    identifier = args.candidate_id
    if any(identifier not in value["scores"] for value in documents.values()):
        raise ValueError("A8 attribution candidate missing")
    result = select_activation_mode(
        static=documents["static"]["scores"][identifier],
        dynamic=documents["dynamic"]["scores"][identifier],
        a16=documents["a16"]["scores"][identifier],
    )
    payload = {
        "schema_version": 1,
        "kind": "full_context_activation_attribution",
        "cross_model_protocol": cross_model_attestation(),
        "full_context_protocol": full_context_attestation(),
        "model_adapter": model,
        "candidate_id": identifier,
        "selection_noise": "A",
        "noise_b_used_for_selection": False,
        "sources": {
            key: {"path": str(value[0]), "sha256": sha256_file(value[0])}
            for key, value in paths_documents.items()
        },
        **result,
    }
    atomic_json(args.out, payload)
    print(json.dumps({"out": str(Path(args.out).resolve()), **result}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    propose_parser = sub.add_parser("propose")
    propose_parser.add_argument("--interventions", required=True)
    propose_parser.add_argument("--scores", required=True)
    propose_parser.add_argument("--out-dir", required=True)
    propose_parser.set_defaults(handler=propose)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--proposals", required=True)
    freeze_parser.add_argument("--scores", required=True)
    freeze_parser.add_argument("--out", required=True)
    freeze_parser.set_defaults(handler=freeze)
    a8_parser = sub.add_parser("a8")
    a8_parser.add_argument("--static-scores", required=True)
    a8_parser.add_argument("--dynamic-scores", required=True)
    a8_parser.add_argument("--a16-scores", required=True)
    a8_parser.add_argument("--candidate-id", default="context_base")
    a8_parser.add_argument("--out", required=True)
    a8_parser.set_defaults(handler=a8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
