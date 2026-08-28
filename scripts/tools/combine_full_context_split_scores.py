#!/usr/bin/env python3
"""Combine per-split full-context score documents into one cross-context document.

Each split (atomic_seen / composite_seen / composite_unseen) scores the same
candidate inventory on its own checkpoint and selection buffer.  The proposal
and freeze stages operate on a single cross-context score document, so the
per-(task, seed) sequence rows are concatenated across splits (tasks are
unique per split) and the aggregate scalars are recomputed from the combined
rows with the exact frozen ``aggregate_d_pac_sequences`` formula.  The
task-macro scalar fields (``d_pac`` / ``d_func``) follow the same
task-cluster estimand used by the v2 anchor ranking.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from quantvla_cross_model_protocol import (
    protocol_attestation as cross_model_attestation,
    require_protocol_attestation as require_cross_model_attestation,
    sha256_file,
)
from quantvla_full_context import (
    require_protocol_attestation as require_full_context_attestation,
    task_scalars,
)
from quantvla_metric_protocol import aggregate_d_pac_sequences
from quantvla_outputimpact import atomic_json

SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")


def load_document(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    require_cross_model_attestation(value, source=str(resolved))
    require_full_context_attestation(value, source=str(resolved))
    if value.get("complete") is not True:
        raise ValueError(f"{resolved}: score artifact is incomplete")
    if value.get("selection_noise", "A") != "A":
        raise ValueError(f"{resolved}: only noise-A may enter selection")
    if any(bool(value.get(key, False)) for key in ("uses_success_labels", "uses_task_success")):
        raise ValueError(f"{resolved}: success-label leakage")
    return resolved, value


def macro_mean(score: dict[str, Any]) -> dict[str, float]:
    scalars = task_scalars(score)
    return {
        key: float(
            np.mean([value for (metric, _task), value in scalars.items() if metric == key])
        )
        for key in ("d_func", "d_pac")
    }


def combine(
    payloads: dict[str, dict[str, Any]], paths: dict[str, Path]
) -> dict[str, Any]:
    inventories = [set(payloads[split]["scores"]) for split in SPLITS]
    if len({frozenset(value) for value in inventories}) != 1:
        raise ValueError("per-split candidate inventories differ")
    first = payloads[SPLITS[0]]
    template = copy.deepcopy(first)
    scores: dict[str, Any] = {}
    for identifier in sorted(inventories[0]):
        pac_rows: list[dict[str, Any]] = []
        func_rows: list[dict[str, Any]] = []
        for split in SPLITS:
            row = payloads[split]["scores"][identifier]
            pac = row["d_pac_summary"]
            func = row["d_func_summary"]
            if len(pac["per_sequence"]) != len(pac["sequences"]):
                raise ValueError(f"{split}/{identifier}: D_PAC sequence inventory drift")
            if len(func["per_sequence"]) != len(func["sequences"]):
                raise ValueError(f"{split}/{identifier}: D_func sequence inventory drift")
            pac_rows.extend(list(pac["sequences"]))
            func_rows.extend(list(func["sequences"]))
        combined = copy.deepcopy(first["scores"][identifier])
        pac_summary = aggregate_d_pac_sequences(pac_rows)
        pac_summary.update(
            {
                "teacher": "original_native_fp16_checkpoint",
                "sequence_grouping": combined["d_pac_summary"].get("sequence_grouping"),
                "dimension_scale": combined["d_pac_summary"].get("dimension_scale"),
                "sequences": pac_rows,
                "combined_across_splits": True,
            }
        )
        functional = dict(combined["d_func_summary"])
        functional.pop("per_obs", None)
        functional.pop("per_dim", None)
        functional.update(
            {
                "per_sequence": [row["d_func_sequence"] for row in func_rows],
                "sequences": func_rows,
                "n_sequences": len(func_rows),
                "selection_sample_unit": "paired_task_seed_sequence",
                "combined_across_splits": True,
            }
        )
        macros = macro_mean(
            {
                "d_func_summary": functional,
                "d_pac_summary": pac_summary,
            }
        )
        combined.update(
            {
                "d_pac": macros["d_pac"],
                "d_pac_summary": pac_summary,
                "d_func": macros["d_func"],
                "d_func_summary": functional,
                "elapsed_s": float(
                    sum(payloads[split]["scores"][identifier]["elapsed_s"] for split in SPLITS)
                ),
                "combined_across_splits": True,
            }
        )
        scores[identifier] = combined
    template["scores"] = scores
    template["n_obs_per_split"] = template.pop("n_obs")
    template["n_obs"] = sum(
        int(payloads[split].get("n_obs", template["n_obs_per_split"])) for split in SPLITS
    )
    template["complete"] = True
    template.pop("candidate_shard", None)
    template["source_shard_manifests"] = {
        split: payloads[split].get("merged_shards")
        for split in SPLITS
    }
    template.pop("merged_shards", None)
    template["combined_splits"] = {
        split: {
            "path": str(paths[split]),
            "sha256": sha256_file(paths[split]),
            "checkpoint_sha256": payloads[split].get("checkpoint_sha256"),
            "selection_buffer_sha256": payloads[split].get("selection_buffer_sha256"),
        }
        for split in SPLITS
    }
    template["cross_model_protocol"] = cross_model_attestation()
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        metavar="split=/path.json",
        help="repeat for each of the three splits",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    provided: dict[str, str] = {}
    for chunk in args.score:
        split, separator, path = chunk.partition("=")
        if not separator or split not in SPLITS or not path:
            raise ValueError(f"score must be split={SPLITS}, got {chunk!r}")
        provided[split] = path
    if set(provided) != set(SPLITS):
        raise ValueError(f"scores must cover exactly {SPLITS}")
    paths, payloads = {}, {}
    for split in SPLITS:
        resolved, document = load_document(provided[split])
        paths[split] = resolved
        payloads[split] = document
    output = Path(args.out).expanduser().resolve()
    atomic_json(output, combine(payloads, paths))
    print(
        json.dumps(
            {"out": str(output), "candidates": len(payloads[SPLITS[0]]["scores"])},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
