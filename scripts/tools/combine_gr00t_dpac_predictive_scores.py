#!/usr/bin/env python3
"""Strictly combine the three preregistered GR00T predictive score splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from combine_full_context_split_scores import SPLITS, combine, load_document
from quantvla_outputimpact import atomic_json
from quantvla_predictive_validity import (
    artifact,
    combine_mse_rows,
    protocol_attestation,
    require_protocol_attestation,
    sha256_file,
)


def load_library(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    require_protocol_attestation(value, source=str(resolved))
    if value.get("kind") != "gr00t_dpac_predictive_mask_library":
        raise ValueError(f"{resolved}: wrong mask-library kind")
    if int(value.get("candidate_count", -1)) != 60:
        raise ValueError(f"{resolved}: mask library must contain exactly 60 masks")
    return resolved, value


def combine_predictive(
    payloads: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    library_path: Path,
    library: dict[str, Any],
) -> dict[str, Any]:
    expected = {str(row["candidate_id"]) for row in library["candidates"]}
    library_hash = sha256_file(library_path)
    for split in SPLITS:
        document = payloads[split]
        candidate_manifest = document.get("candidate_manifest") or {}
        if candidate_manifest.get("sha256") != library_hash:
            raise ValueError(f"{split}: score mask-library hash drift")
        if set(document.get("scores") or {}) != expected:
            raise ValueError(f"{split}: score inventory is not the 60 preregistered masks")
        if int(document.get("n_obs", -1)) != 48:
            raise ValueError(f"{split}: expected exactly 48 selection observations")
        if document.get("activation_mode") != "dynamic_a8":
            raise ValueError(f"{split}: predictive score activation mode drift")
        if int(document.get("flow_steps", -1)) != 4:
            raise ValueError(f"{split}: predictive score flow-step drift")

    result = combine(payloads, paths)
    result.update(
        {
            "kind": "gr00t_dpac_predictive_offline_scores",
            "schema_version": 1,
            "predictive_validity_protocol": protocol_attestation(),
            "mask_library": artifact(library_path),
            "offline_outcomes_withheld": True,
            "metric_names": ["mse", "d_func", "d_pac"],
        }
    )
    for identifier in sorted(expected):
        mse_rows: list[dict[str, Any]] = []
        split_scores: dict[str, dict[str, float]] = {}
        for split in SPLITS:
            score = payloads[split]["scores"][identifier]
            summary = score.get("mse_summary") or {}
            sequences = summary.get("sequences") or []
            if len(sequences) != len(summary.get("per_sequence") or []):
                raise ValueError(f"{split}/{identifier}: MSE sequence inventory drift")
            mse_rows.extend(sequences)
            split_scores[split] = {
                "mse": float(score["mse"]),
                "d_func": float(score["d_func"]),
                "d_pac": float(score["d_pac"]),
            }
        mse = combine_mse_rows(mse_rows)
        result["scores"][identifier].update(
            {
                "mse": float(mse["mse"]),
                "mse_summary": mse,
                "split_scores": split_scores,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--score", action="append", required=True, metavar="split=PATH")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    provided: dict[str, str] = {}
    for chunk in args.score:
        split, separator, path = chunk.partition("=")
        if not separator or split not in SPLITS or not path or split in provided:
            raise ValueError(f"invalid or repeated --score: {chunk!r}")
        provided[split] = path
    if set(provided) != set(SPLITS):
        raise ValueError(f"scores must cover exactly {SPLITS}")
    paths: dict[str, Path] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        paths[split], payloads[split] = load_document(provided[split])
    library_path, library = load_library(args.manifest)
    result = combine_predictive(payloads, paths, library_path, library)
    output = Path(args.out).expanduser().resolve()
    atomic_json(output, result)
    print(json.dumps({"out": str(output), "candidates": len(result["scores"])}, indent=2))


if __name__ == "__main__":
    main()
