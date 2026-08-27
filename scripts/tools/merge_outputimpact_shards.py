#!/usr/bin/env python3
"""Merge disjoint GR00T/pi0.5 OutputImpact shards with strict provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantvla_cross_model_protocol import require_protocol_attestation, sha256_file
from quantvla_dynamic_a8_protocol import (
    require_protocol_attestation as require_dynamic_a8_protocol_attestation,
)
from quantvla_outputimpact import ARTIFACT_KIND, atomic_json


INVARIANTS = (
    "cross_model_protocol",
    "dynamic_a8_protocol",
    "model_adapter",
    "teacher",
    "selection_metric",
    "base_full_w4_plan_sha256",
    "hessian_w4_sha256",
    "activation_mode",
    "a8_sha256",
    "selection_buffer_sha256",
    "calibration_buffer_sha256",
    "checkpoint_sha256",
    "n_obs",
    "noise",
    "intervention",
    "uses_cka",
    "uses_cs",
    "uses_task_success",
    "uses_success_labels",
    "uses_gradients",
    "uses_extra_training_data",
    "source_sha256",
)


def merge(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one OutputImpact shard is required")
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = documents[0]
    require_protocol_attestation(reference, source=str(paths[0]))
    if reference.get("activation_mode") == "dynamic_a8":
        require_dynamic_a8_protocol_attestation(reference, source=str(paths[0]))
    if reference.get("kind") != ARTIFACT_KIND:
        raise ValueError("wrong OutputImpact artifact kind")
    count = int((reference.get("shard") or {}).get("count", 0))
    if count != len(paths):
        raise ValueError(f"expected {count} shards, received {len(paths)}")
    indices: set[int] = set()
    layers: dict[str, Any] = {}
    shard_records = []
    expected_union: set[str] = set()
    for path, document in zip(paths, documents):
        require_protocol_attestation(document, source=str(path))
        if document.get("activation_mode") == "dynamic_a8":
            require_dynamic_a8_protocol_attestation(document, source=str(path))
        if document.get("kind") != ARTIFACT_KIND or document.get("complete") is not True:
            raise ValueError(f"incomplete OutputImpact shard: {path}")
        if any(document.get(key) != reference.get(key) for key in INVARIANTS):
            raise ValueError(f"OutputImpact shard provenance drift: {path}")
        shard = document.get("shard") or {}
        index = int(shard.get("index", -1))
        if int(shard.get("count", 0)) != count or index in indices:
            raise ValueError(f"invalid or duplicate OutputImpact shard index: {path}")
        indices.add(index)
        declared = set(str(name) for name in shard.get("layer_names") or [])
        observed = set(document.get("layers") or {})
        if declared != observed:
            raise ValueError(f"OutputImpact shard layer set mismatch: {path}")
        overlap = set(layers) & observed
        if overlap:
            raise ValueError(f"duplicate OutputImpact layers: {sorted(overlap)[:3]}")
        layers.update(document["layers"])
        expected_union.update(declared)
        shard_records.append(
            {
                "index": index,
                "path": str(path),
                "sha256": sha256_file(path),
                "layers": len(observed),
            }
        )
    if indices != set(range(count)):
        raise ValueError(f"OutputImpact shards are incomplete: {sorted(indices)}")
    plan_path = Path(reference["base_full_w4_plan"]).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected_plan = {
        name
        for name, row in (plan.get("layers") or {}).items()
        if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
    }
    if expected_union != expected_plan:
        raise ValueError("merged OutputImpact inventory does not cover the full W4 plan")
    payload = {key: value for key, value in reference.items() if key not in ("shard", "layers")}
    payload["shards"] = sorted(shard_records, key=lambda row: row["index"])
    payload["layers"] = dict(sorted(layers.items()))
    payload["completed_layers"] = len(layers)
    payload["complete"] = True
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = [Path(value).expanduser().resolve() for value in args.input]
    payload = merge(paths)
    atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "out": str(Path(args.out).expanduser().resolve()),
                "model_adapter": payload["model_adapter"],
                "layers": len(payload["layers"]),
                "sha256": sha256_file(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
