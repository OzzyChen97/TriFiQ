#!/usr/bin/env python3
"""Finalize the statistics-corrected FCP candidate manifest.

The correction artifact predates the top-level cross-model attestation field
required by ``select_full_context_protection.py``.  This narrow finalizer keeps
that immutable manifest intact, validates its full-context attestation and all
candidate hashes, and delegates the actual decision to the same frozen
``select_frozen_candidate`` implementation used by the ordinary finalizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from quantvla_cross_model_protocol import (  # noqa: E402
    protocol_attestation as cross_model_attestation,
    require_protocol_attestation as require_cross_model_attestation,
    sha256_file,
)
from quantvla_full_context import (  # noqa: E402
    protocol_attestation as full_context_attestation,
    require_protocol_attestation as require_full_context_attestation,
    select_frozen_candidate,
)
from quantvla_outputimpact import atomic_json  # noqa: E402


def load(path: str) -> tuple[Path, dict]:
    resolved = Path(path).expanduser().resolve()
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_path, manifest = load(args.proposals)
    scores_path, document = load(args.scores)
    if manifest.get("kind") != "corrected_full_context_budgeted_proposal_manifest":
        raise ValueError("this finalizer only accepts the corrected FCP manifest")
    require_full_context_attestation(manifest, source=str(manifest_path))
    require_cross_model_attestation(document, source=str(scores_path))
    require_full_context_attestation(document, source=str(scores_path))
    if document.get("complete") is not True or document.get("selection_noise") != "A":
        raise ValueError("complete noise-A score document required")
    if document.get("uses_success_labels") is not False:
        raise ValueError("success-label leakage in complete-policy scores")

    rows = {str(row["candidate_id"]): row for row in manifest["candidates"]}
    if set(rows) != set(document["scores"]):
        raise ValueError("candidate manifest and score inventories differ")
    for identifier, row in rows.items():
        candidate_path = Path(row["path"]).expanduser().resolve()
        if sha256_file(candidate_path) != row["sha256"]:
            raise ValueError(f"candidate artifact drift: {identifier}")
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        require_cross_model_attestation(candidate.get("meta") or {}, source=str(candidate_path))
        require_full_context_attestation(candidate.get("meta") or {}, source=str(candidate_path))

    selection = select_frozen_candidate(
        scores=document["scores"],
        baseline_id="context_base",
        plan_rows=rows,
    )
    selected_id = selection["selected_id"]
    selected_path, selected_plan = load(rows[selected_id]["path"])
    meta = dict(selected_plan.get("meta") or {})
    meta.update(
        {
            "kind": "corrected_full_context_fp16_protection_frozen",
            "frozen": True,
            "selection_noise": "A",
            "noise_b_used_for_selection": False,
            "proposal_manifest_sha256": sha256_file(manifest_path),
            "full_network_scores_sha256": sha256_file(scores_path),
            "selected_candidate_id": selected_id,
            "selection_result": selection,
            "corrected_manifest_compatibility_finalizer": sha256_file(Path(__file__)),
        }
    )
    selected_plan["meta"] = meta
    output = Path(args.out).expanduser().resolve()
    atomic_json(output, selected_plan)
    report = {
        "schema_version": 1,
        "kind": "corrected_full_context_frozen_selection_report",
        "cross_model_protocol": cross_model_attestation(),
        "full_context_protocol": full_context_attestation(),
        "proposal_manifest": str(manifest_path),
        "proposal_manifest_sha256": sha256_file(manifest_path),
        "full_network_scores": str(scores_path),
        "full_network_scores_sha256": sha256_file(scores_path),
        "selected_source": str(selected_path),
        "selected_source_sha256": sha256_file(selected_path),
        "frozen_plan": str(output),
        "frozen_plan_sha256": sha256_file(output),
        "finalizer": str(Path(__file__).resolve()),
        "finalizer_sha256": sha256_file(Path(__file__)),
        "selection": selection,
    }
    report_path = output.with_suffix(output.suffix + ".selection.json")
    atomic_json(report_path, report)
    print(json.dumps({"out": str(output), "report": str(report_path), "selected": selected_id}, indent=2))


if __name__ == "__main__":
    main()
