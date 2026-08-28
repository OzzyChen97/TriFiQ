#!/usr/bin/env python3
"""Strictly merge disjoint full-context candidate-score GPU shards."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from quantvla_cross_model_protocol import (
    require_protocol_attestation as require_cross_model_attestation,
    sha256_file,
)
from quantvla_full_context import require_protocol_attestation
from quantvla_outputimpact import atomic_json


def load(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    require_cross_model_attestation(value, source=str(resolved))
    require_protocol_attestation(value, source=str(resolved))
    return resolved, value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--score", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_path, manifest = load(args.manifest)
    expected_rows = {row["candidate_id"]: row for row in manifest.get("candidates") or []}
    if not expected_rows:
        raise ValueError("candidate manifest is empty")

    loaded = [load(path) for path in args.score]
    documents = [value for _path, value in loaded]
    first = documents[0]
    ignored = {
        "scores", "complete", "best_noise_a", "candidate_plans", "candidate_shard",
        # Each shard independently executes the same frozen FP16 teacher.
        # Wall-clock latency is a runtime measurement, not provenance, and is
        # expected to differ slightly across GPUs.
        "teacher_latency_mean_s",
    }
    invariant_keys = set(first) - ignored
    for value in documents[1:]:
        if set(value) - ignored != invariant_keys:
            raise ValueError("score shard schema drift")
        drift = [key for key in invariant_keys if value.get(key) != first.get(key)]
        if drift:
            raise ValueError(f"score shard provenance drift: {drift}")

    scores: dict[str, Any] = {}
    candidate_plans: dict[str, Any] = {}
    shard_indices = set()
    shard_counts = set()
    for path, value in loaded:
        shard = value.get("candidate_shard") or {}
        shard_indices.add(int(shard.get("index", -1)))
        shard_counts.add(int(shard.get("count", -1)))
        if value.get("complete") is not True:
            raise ValueError(f"{path}: incomplete score shard")
        if set(value.get("scores") or {}) != set(value.get("candidate_plans") or {}):
            raise ValueError(f"{path}: score/candidate inventory mismatch")
        for identifier, row in (value.get("candidate_plans") or {}).items():
            if identifier in scores:
                raise ValueError(f"duplicate candidate across shards: {identifier}")
            expected = expected_rows.get(identifier)
            if expected is None or row.get("sha256") != expected.get("sha256"):
                raise ValueError(f"{path}: candidate manifest mismatch for {identifier}")
            candidate_plans[identifier] = row
            scores[identifier] = value["scores"][identifier]
    if len(shard_counts) != 1:
        raise ValueError("inconsistent candidate shard counts")
    shard_count = next(iter(shard_counts))
    if shard_indices != set(range(shard_count)):
        raise ValueError(f"incomplete shard indices: {sorted(shard_indices)} of {shard_count}")
    if set(scores) != set(expected_rows):
        raise ValueError("merged score inventory does not cover the candidate manifest")

    merged = copy.deepcopy(first)
    merged.pop("candidate_shard", None)
    merged["candidate_manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "kind": manifest.get("kind"),
    }
    merged["candidate_plans"] = dict(sorted(candidate_plans.items()))
    merged["scores"] = dict(sorted(scores.items()))
    merged["complete"] = True
    merged["merged_shards"] = [
        {"path": str(path), "sha256": sha256_file(path)} for path, _value in loaded
    ]
    merged["teacher_latency_mean_s_by_shard"] = [
        float(value["teacher_latency_mean_s"]) for value in documents
    ]
    merged["teacher_latency_mean_s"] = sum(
        merged["teacher_latency_mean_s_by_shard"]
    ) / len(documents)
    if merged["scores"]:
        merged["best_noise_a"] = min(
            merged["scores"], key=lambda key: merged["scores"][key]["d_pac"]
        )
    output = Path(args.out).expanduser().resolve()
    atomic_json(output, merged)
    print(json.dumps({"out": str(output), "candidates": len(scores)}, indent=2))


if __name__ == "__main__":
    main()
