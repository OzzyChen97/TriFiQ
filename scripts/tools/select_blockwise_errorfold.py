#!/usr/bin/env python3
"""Select one static blockwise ErrorFold coordinate using D_func and D_PAC.

The two losses are put on a common, dimensionless scale using the identity and
full-ATM endpoints from the already-complete shared 9x9 SoftFold evaluation.
The selected coordinate minimizes the worse endpoint-normalized regret.  No
rollout result, task label, fitted weighting coefficient, or runtime selector
is accepted by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def objectives(row: Mapping[str, Any]) -> tuple[float, float]:
    d_func_values = (row.get("d_func_summary") or {}).get("per_sequence")
    if not d_func_values:
        raise ValueError("candidate lacks sequence-aligned D_func values")
    local = statistics.fmean(float(value) for value in d_func_values)
    long_horizon = float(row["d_pac"])
    if not math.isfinite(local) or not math.isfinite(long_horizon):
        raise ValueError("candidate objectives must be finite")
    return local, long_horizon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", action="append", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--identity-id", default="softfold_a00_e00")
    parser.add_argument("--full-atm-id", default="softfold_a08_e00")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    anchor_path = Path(args.anchors).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates") or {}
    scores: dict[str, Any] = {}
    score_provenance = []
    for raw_score_path in args.scores:
        score_path = Path(raw_score_path).expanduser().resolve()
        score_payload = json.loads(score_path.read_text(encoding="utf-8"))
        if score_payload.get("uses_task_labels_for_selection") is not False:
            raise ValueError("score artifact must attest task-label-free selection")
        if score_payload.get("uses_rollout_success_for_selection") is not False:
            raise ValueError("score artifact must attest success-free selection")
        if score_payload.get("errorfold_candidate_manifest_sha256") != sha256_file(
            manifest_path
        ):
            raise ValueError("score/manifest provenance mismatch")
        overlap = set(scores) & set(score_payload.get("scores") or {})
        if overlap:
            raise ValueError(f"duplicate candidate scores: {sorted(overlap)[:3]}")
        scores.update(score_payload.get("scores") or {})
        score_provenance.append(
            {"path": str(score_path), "sha256": sha256_file(score_path)}
        )
    if set(scores) != set(candidates):
        raise ValueError("candidate score inventory is incomplete")

    anchor_scores = anchors.get("scores") or {}
    endpoint_rows = [anchor_scores[args.identity_id], anchor_scores[args.full_atm_id]]
    endpoint_values = [objectives(row) for row in endpoint_rows]
    func_best_index = min(range(2), key=lambda index: endpoint_values[index][0])
    pac_best_index = min(range(2), key=lambda index: endpoint_values[index][1])
    best_values = {
        "d_func": endpoint_values[func_best_index][0],
        "d_pac": endpoint_values[pac_best_index][1],
    }
    spans = {
        "d_func": endpoint_values[pac_best_index][0] - best_values["d_func"],
        "d_pac": endpoint_values[func_best_index][1] - best_values["d_pac"],
    }
    if any(value <= 0.0 for value in spans.values()):
        raise ValueError(f"identity/full-ATM endpoints do not bracket both metrics: {spans}")

    ranked = []
    for config_id, row in sorted(scores.items()):
        local, long_horizon = objectives(row)
        regrets = {
            "d_func": (local - best_values["d_func"]) / spans["d_func"],
            "d_pac": (long_horizon - best_values["d_pac"]) / spans["d_pac"],
        }
        ranked.append(
            {
                "config_id": config_id,
                "d_func_objective": local,
                "d_pac_objective": long_horizon,
                "normalized_endpoint_regret": regrets,
                "worst_normalized_endpoint_regret": max(regrets.values()),
                "mean_normalized_endpoint_regret": statistics.fmean(regrets.values()),
                "correction_norm": float(candidates[config_id]["correction_norm"]),
                "profile": candidates[config_id]["gate_profile"],
                "profile_sha256": candidates[config_id]["gate_profile_sha256"],
                "errorfold": candidates[config_id]["errorfold"],
                "errorfold_sha256": candidates[config_id]["errorfold_sha256"],
            }
        )
    selected = min(
        ranked,
        key=lambda row: (
            row["worst_normalized_endpoint_regret"],
            row["mean_normalized_endpoint_regret"],
            row["correction_norm"],
            row["profile_sha256"],
        ),
    )
    base_id = f"block_r{int(manifest['round']):02d}_base"
    base = next(row for row in ranked if row["config_id"] == base_id)
    tolerance = max(
        1e-12,
        1e-9 * max(1.0, abs(base["worst_normalized_endpoint_regret"])),
    )
    improved = (
        selected["worst_normalized_endpoint_regret"]
        < base["worst_normalized_endpoint_regret"] - tolerance
    )
    if not improved:
        selected = base

    payload = {
        "schema_version": 1,
        "kind": "blockwise_errorfold_coordinate_selection",
        "method_id": "blocksoftfold-dual-coordinate-v1",
        "round": int(manifest["round"]),
        "profile": selected["profile"],
        "profile_sha256": selected["profile_sha256"],
        "selected": selected,
        "base": base,
        "coordinate_improved": improved,
        "converged": not improved,
        "anchors": {
            "path": str(anchor_path),
            "sha256": sha256_file(anchor_path),
            "identity_id": args.identity_id,
            "full_atm_id": args.full_atm_id,
            "best_objectives": best_values,
            "endpoint_spans": spans,
        },
        "scores": score_provenance,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "selection": {
            "rule": "minimum_worst_endpoint_normalized_regret_d_func_d_pac",
            "tie_break": "mean_regret_then_correction_norm_then_profile_hash",
            "uses_task_labels": False,
            "uses_rollout_success": False,
            "sample_unit": "paired_shared_buffer_sequence",
            "runtime_selector": False,
        },
        "candidates": ranked,
    }
    output = Path(args.out).expanduser().resolve()
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": sha256_file(output),
                "selected": selected["config_id"],
                "coordinate_improved": improved,
                "converged": not improved,
                "d_func": selected["d_func_objective"],
                "d_pac": selected["d_pac_objective"],
                "worst_regret": selected["worst_normalized_endpoint_regret"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
