#!/usr/bin/env python3
"""Rank FCP proposals, freeze Top-3, and fail closed after held-out audits."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file


def merge_scores(paths: list[str], expected_noise: str) -> tuple[dict, list[dict]]:
    scores = {}; sources = []
    for raw in paths:
        path = Path(raw).resolve(); value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("complete") is not True or value.get("noise") != expected_noise:
            raise ValueError(f"invalid noise-{expected_noise} score shard: {path}")
        if set(scores) & set(value["scores"]):
            raise ValueError("duplicate proposal score")
        scores.update(value["scores"]); sources.append({"path": str(path), "sha256": sha256_file(path)})
    return scores, sources


def aggregate(values: np.ndarray) -> float:
    count = max(1, int(np.ceil(0.1 * len(values))))
    return float(values.mean() + np.sort(values)[-count:].mean())


def robust_gate(baseline: dict, candidate: dict) -> dict:
    base = np.asarray(baseline["d_pac_summary"]["per_sequence"], dtype=np.float64)
    value = np.asarray(candidate["d_pac_summary"]["per_sequence"], dtype=np.float64)
    improvement = base - value
    one_se = float(improvement.std(ddof=1) / math.sqrt(len(improvement))) if len(improvement) > 1 else 0.0
    sequences = baseline["d_pac_summary"]["sequences"]
    task_keys = [(row["suite"], int(row["task_index"])) for row in sequences]
    suites = sorted(set(key[0] for key in task_keys))
    suite_pass = {}
    for suite in suites:
        selected = np.asarray([key[0] == suite for key in task_keys])
        suite_pass[suite] = aggregate(value[selected]) <= aggregate(base[selected]) + 1e-12
    jackknife = {}
    for key in sorted(set(task_keys)):
        selected = np.asarray([other != key for other in task_keys])
        jackknife[f"{key[0]}:{key[1]}"] = aggregate(value[selected]) <= aggregate(base[selected]) + 1e-12
    return {
        "mean_improvement": float(improvement.mean()),
        "one_se": one_se,
        "one_se_pass": float(improvement.mean()) > one_se,
        "suite_minimax_pass": all(suite_pass.values()),
        "suite_checks": suite_pass,
        "delete_one_task_pass": all(jackknife.values()),
        "delete_one_task_checks": jackknife,
        "eligible": float(improvement.mean()) > one_se and all(suite_pass.values()) and all(jackknife.values()),
    }


def rank(args: argparse.Namespace) -> None:
    manifest_path = Path(args.proposals).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scores, sources = merge_scores(args.scores_a, "A")
    if set(scores) != set(manifest["candidates"]):
        raise ValueError("proposal noise-A score coverage mismatch")
    baseline = scores["context_base"]
    gates = {identifier: ({"eligible": True, "baseline": True} if identifier == "context_base" else robust_gate(baseline, score)) for identifier, score in scores.items()}
    eligible = [identifier for identifier in scores if gates[identifier]["eligible"]]
    ranking = sorted(eligible, key=lambda identifier: (scores[identifier]["d_pac"], scores[identifier]["d_func"], identifier))
    top3 = ranking[:3]
    if "context_base" not in top3:
        top3 = (top3[:2] + ["context_base"])[:3]
    candidates = {identifier: manifest["candidates"][identifier] for identifier in top3}
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_fcp_top3_manifest",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "proposal_manifest": str(manifest_path),
        "proposal_manifest_sha256": sha256_file(manifest_path),
        "noise_a_sources": sources,
        "ranking_frozen_before_noise_b": True,
        "ranking": ranking,
        "top3": top3,
        "gates": gates,
        "candidates": candidates,
        "uses_success_labels": False,
    }
    atomic_json(args.out, payload)
    print(json.dumps({"out": str(Path(args.out).resolve()), "top3": top3, "eligible": ranking}, indent=2))


def freeze(args: argparse.Namespace) -> None:
    top3_path = Path(args.top3).resolve(); top3 = json.loads(top3_path.read_text(encoding="utf-8"))
    scores_b, sources_b = merge_scores(args.scores_b, "B")
    if set(scores_b) != set(top3["candidates"]):
        raise ValueError("Top-3 noise-B score coverage mismatch")
    audit_path = Path(args.candidate_state_audit).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("protocol_id") != PROTOCOL["protocol_id"] or set(audit.get("candidates", {})) != set(top3["candidates"]):
        raise ValueError("candidate-state audit coverage/protocol mismatch")
    preferred = top3["ranking"][0] if top3["ranking"] else "context_base"
    baseline_b = scores_b["context_base"]
    selected = preferred
    failures = []
    if preferred != "context_base":
        gate_b = robust_gate(baseline_b, scores_b[preferred])
        if not gate_b["suite_minimax_pass"]:
            failures.append("noise_b_suite_minimax")
        if not bool(audit["candidates"][preferred].get("eligible", False)):
            failures.append("candidate_state_audit")
    if failures:
        selected = "context_base"
    selected_plan_path = Path(top3["candidates"][selected]["path"]).resolve()
    plan = copy.deepcopy(json.loads(selected_plan_path.read_text(encoding="utf-8")))
    plan["meta"].update({
        "kind": "dypac_libero_fcp_frozen",
        "frozen": True,
        "selected_candidate_id": selected,
        "fcp_changed_initial_mask": selected != "context_base",
        "failure_reasons": failures,
        "top3_manifest_sha256": sha256_file(top3_path),
        "candidate_state_audit_sha256": sha256_file(audit_path),
        "noise_b_used_only_for_fail_closed_audit": True,
        "uses_success_labels": False,
    })
    atomic_json(args.out, plan)
    report = {
        "schema_version": 1,
        "kind": "dypac_libero_fcp_frozen_report",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "top3_manifest": str(top3_path),
        "top3_manifest_sha256": sha256_file(top3_path),
        "noise_b_sources": sources_b,
        "candidate_state_audit": str(audit_path),
        "candidate_state_audit_sha256": sha256_file(audit_path),
        "preferred_candidate": preferred,
        "selected_candidate": selected,
        "failure_reasons": failures,
        "fcp_changed_initial_mask": selected != "context_base",
        "frozen_plan": str(Path(args.out).resolve()),
        "frozen_plan_sha256": sha256_file(args.out),
    }
    atomic_json(str(args.out) + ".selection.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rank_parser = commands.add_parser("rank")
    rank_parser.add_argument("--proposals", required=True)
    rank_parser.add_argument("--scores-a", required=True, nargs="+")
    rank_parser.add_argument("--out", required=True)
    rank_parser.set_defaults(handler=rank)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--top3", required=True)
    freeze_parser.add_argument("--scores-b", required=True, nargs="+")
    freeze_parser.add_argument("--candidate-state-audit", required=True)
    freeze_parser.add_argument("--out", required=True)
    freeze_parser.set_defaults(handler=freeze)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); args.handler(args)


if __name__ == "__main__":
    main()
