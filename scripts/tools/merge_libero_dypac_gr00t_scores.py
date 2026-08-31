#!/usr/bin/env python3
"""Merge four suite-specific GR00T DyPAC score shards into global scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file


SUITE_ORDER = tuple(PROTOCOL["benchmark"]["suites"])
SUITES = set(SUITE_ORDER)


def combine_summary(rows: list[dict]) -> dict:
    pac_sequences = [sequence for row in rows for sequence in row["d_pac_summary"]["sequences"]]
    pac_values = np.asarray(
        [value for row in rows for value in row["d_pac_summary"]["per_sequence"]],
        dtype=np.float64,
    )
    func_sequences = [sequence for row in rows for sequence in row["d_func_summary"]["sequences"]]
    func_values = np.asarray(
        [value for row in rows for value in row["d_func_summary"]["per_sequence"]],
        dtype=np.float64,
    )
    if len(pac_values) != 36 or len(func_values) != 36:
        raise ValueError("global GR00T summary must contain 36 task-state sequences")
    scales = [np.asarray(row["d_pac_summary"]["dimension_scale"]) for row in rows]
    if any(not np.array_equal(scales[0], value) for value in scales[1:]):
        raise ValueError("suite-specific GR00T scorers used different physical scales")
    tail = max(1, int(np.ceil(0.1 * len(pac_values))))
    mean = float(pac_values.mean())
    cvar = float(np.sort(pac_values)[-tail:].mean())
    suite_mean = {
        suite: float(
            np.mean(
                [
                    value
                    for value, sequence in zip(pac_values, pac_sequences, strict=True)
                    if sequence["suite"] == suite
                ]
            )
        )
        for suite in sorted(SUITES)
    }
    return {
        "d_pac": mean + cvar,
        "d_func": float(func_values.mean()),
        "d_pac_summary": {
            "d_pac": mean + cvar,
            "mean": mean,
            "cvar90": cvar,
            "suite_mean": suite_mean,
            "suite_minimax": max(suite_mean.values()),
            "per_sequence": pac_values.tolist(),
            "sequences": pac_sequences,
            "dimension_scale": scales[0].tolist(),
        },
        "d_func_summary": {
            "d_func": float(func_values.mean()),
            "per_sequence": func_values.tolist(),
            "sequences": func_sequences,
        },
        "elapsed_s": float(sum(float(row.get("elapsed_s", 0.0)) for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("outputimpact", "mask"), required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    values = []
    sources = []
    for raw in args.inputs:
        path = Path(raw).resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("model") != "gr00t" or value.get("complete") is not True:
            raise ValueError(f"invalid GR00T suite score shard: {path}")
        if value.get("protocol_id") != PROTOCOL["protocol_id"]:
            raise ValueError(f"GR00T suite score protocol drift: {path}")
        values.append(value)
        sources.append({"path": str(path), "sha256": sha256_file(path)})
    if {value.get("suite") for value in values} != SUITES:
        raise ValueError("GR00T global merge requires exactly the four LIBERO suites")

    key = "layers" if args.kind == "outputimpact" else "scores"
    suite_values = {}
    for suite in SUITE_ORDER:
        partitions = [value for value in values if value.get("suite") == suite]
        merged = {}
        for value in partitions:
            overlap = set(merged) & set(value[key])
            if overlap:
                raise ValueError(
                    f"overlapping {suite} GR00T score partitions: {sorted(overlap)[:3]}"
                )
            merged.update(value[key])
        suite_values[suite] = merged
    identifiers = set(next(iter(suite_values.values())))
    if any(set(value) != identifiers for value in suite_values.values()):
        raise ValueError("suite-specific GR00T score coverage differs")
    combined = {
        identifier: combine_summary(
            [suite_values[suite][identifier] for suite in SUITE_ORDER]
        )
        for identifier in sorted(identifiers)
    }
    first = values[0]
    one_partition_per_suite = len(values) == len(SUITES)
    shard_index = first["shard_index"] if one_partition_per_suite else 0
    shard_count = first["shard_count"] if one_partition_per_suite else 1
    if args.kind == "outputimpact":
        payload = {
            "schema_version": 1,
            "kind": "dypac_libero_gr00t_single_layer_outputimpact",
            "model": "gr00t",
            "protocol_id": PROTOCOL["protocol_id"],
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "selection_rows": 144,
            "intervention": "one_hessian_group64_w4_dynamic_a8_layer_all_others_native_fp16",
            "teacher": "four_suite_specific_original_fp16_checkpoints",
            "teacher_identity_by_suite": {
                suite: next(
                    value["teacher_identity"]
                    for value in values
                    if value["suite"] == suite
                )
                for suite in SUITE_ORDER
            },
            "dimension_scale": next(iter(combined.values()))["d_pac_summary"]["dimension_scale"],
            "uses_success_labels": False,
            "uses_test_rollout_feedback": False,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "assigned_layers": sorted(identifiers),
            "layers": combined,
            "sources": sources,
            "complete": True,
        }
    else:
        noises = {value.get("noise") for value in values}
        if len(noises) != 1:
            raise ValueError("cannot merge GR00T mask scores from different noise streams")
        payload = {
            "schema_version": 1,
            "kind": "dypac_libero_gr00t_complete_mask_scores",
            "model": "gr00t",
            "protocol_id": PROTOCOL["protocol_id"],
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "manifest": first["manifest"],
            "manifest_sha256": first["manifest_sha256"],
            "selection_buffer_sha256": first["selection_buffer_sha256"],
            "noise": next(iter(noises)),
            "selection_noise": next(iter(noises)) == "A",
            "uses_success_labels": False,
            "uses_test_rollout_feedback": False,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "assigned_candidates": sorted(identifiers),
            "scores": combined,
            "sources": sources,
            "complete": True,
        }
    atomic_json(args.out, payload)
    print(json.dumps({"out": str(Path(args.out).resolve()), "kind": args.kind, "rows": len(combined)}, indent=2))


if __name__ == "__main__":
    main()
