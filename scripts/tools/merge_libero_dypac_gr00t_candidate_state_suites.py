#!/usr/bin/env python3
"""Merge four suite actions into one global GR00T candidate-state score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantvla_libero_dypac import (
    PROTOCOL,
    PROTOCOL_PATH,
    atomic_json,
    physical_scale,
    sha256_file,
    summarize_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    values = []
    sources = []
    for raw in args.inputs:
        path = Path(raw).resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("model") != "gr00t"
            or value.get("protocol_id") != PROTOCOL["protocol_id"]
            or value.get("uses_success_labels") is not False
        ):
            raise ValueError(f"invalid GR00T candidate-state suite file: {path}")
        values.append(value)
        sources.append({"path": str(path), "sha256": sha256_file(path)})
    suites = set(PROTOCOL["benchmark"]["suites"])
    if len(values) != 4 or {value["suite"] for value in values} != suites:
        raise ValueError("GR00T candidate-state merge requires exactly four suites")
    identifiers = {value["candidate_id"] for value in values}
    if len(identifiers) != 1:
        raise ValueError("cannot merge different GR00T candidate ids")
    teacher = np.concatenate(
        [np.asarray(value["teacher_actions"], dtype=np.float32) for value in values], axis=0
    )
    candidate = np.concatenate(
        [np.asarray(value["candidate_actions"], dtype=np.float32) for value in values], axis=0
    )
    records = [record for value in values for record in value["records"]]
    if teacher.shape != (48, 5, 7) or candidate.shape != (48, 5, 7):
        raise ValueError("GR00T candidate-state global action coverage drift")
    summary = summarize_pair(teacher, candidate, records, scale=physical_scale(teacher))
    identifier = next(iter(identifiers))
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_candidate_state_score",
        "model": "gr00t",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_id": identifier,
        "sequences": 12,
        "rows": 48,
        "initial_state_indices": [0],
        "formal_initial_state_overlap": False,
        "uses_success_labels": False,
        "uses_test_rollout_feedback": False,
        "d_pac": summary["d_pac_summary"]["d_pac"],
        "d_func": summary["d_func_summary"]["d_func"],
        "d_pac_summary": summary["d_pac_summary"],
        "d_func_summary": summary["d_func_summary"],
        "sources": sources,
        "elapsed_s": sum(float(value.get("elapsed_s", 0.0)) for value in values),
    }
    atomic_json(args.out, payload)
    print(json.dumps({"candidate_id": identifier, "d_pac": payload["d_pac"], "rows": 48}, indent=2))


if __name__ == "__main__":
    main()
