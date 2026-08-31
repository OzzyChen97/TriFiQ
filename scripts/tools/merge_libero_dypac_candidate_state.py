#!/usr/bin/env python3
"""Merge Top-3 candidate-state audits and apply the fail-closed gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top3", required=True)
    parser.add_argument("--scores", required=True, nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    top3_path = Path(args.top3).resolve(); top3 = json.loads(top3_path.read_text(encoding="utf-8"))
    rows = {}; sources = []
    for raw in args.scores:
        path = Path(raw).resolve(); value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("protocol_id") != PROTOCOL["protocol_id"] or value.get("uses_success_labels") is not False:
            raise ValueError(f"invalid candidate-state score: {path}")
        rows[value["candidate_id"]] = value; sources.append({"path": str(path), "sha256": sha256_file(path)})
    if set(rows) != set(top3["candidates"]):
        raise ValueError("candidate-state Top-3 coverage mismatch")
    baseline = rows["context_base"]
    base_values = np.asarray(baseline["d_pac_summary"]["per_sequence"], dtype=np.float64)
    candidates = {}
    for identifier, row in rows.items():
        values = np.asarray(row["d_pac_summary"]["per_sequence"], dtype=np.float64)
        difference = values - base_values
        one_se = float(difference.std(ddof=1) / math.sqrt(len(difference))) if len(difference) > 1 else 0.0
        eligible = identifier == "context_base" or float(difference.mean()) <= one_se
        candidates[identifier] = {
            "d_pac": row["d_pac"],
            "d_func": row["d_func"],
            "mean_excess_over_initial": float(difference.mean()),
            "one_se": one_se,
            "eligible": eligible,
        }
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_candidate_state_audit",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "top3_manifest": str(top3_path),
        "top3_manifest_sha256": sha256_file(top3_path),
        "candidates": candidates,
        "sources": sources,
        "uses_success_labels": False,
    }
    atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
