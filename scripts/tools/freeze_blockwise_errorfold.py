#!/usr/bin/env python3
"""Freeze a selected BlockSoftFold profile and its one-candidate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--artifact-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    args = parser.parse_args()

    selection_path = Path(args.selection).expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("kind") != "blockwise_errorfold_coordinate_selection":
        raise ValueError("unsupported blockwise selection")
    selected = selection["selected"]
    source_artifact = Path(selected["errorfold"]).expanduser().resolve()
    if sha256_file(source_artifact) != selected["errorfold_sha256"]:
        raise ValueError("selected ErrorFold artifact hash drift")
    source_manifest_path = Path(selection["manifest"]["path"]).resolve()
    if sha256_file(source_manifest_path) != selection["manifest"]["sha256"]:
        raise ValueError("selected manifest hash drift")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    artifact = json.loads(source_artifact.read_text(encoding="utf-8"))
    artifact["selection"] = {
        "status": "frozen_before_noise_b_and_closed_loop",
        "method_id": "blocksoftfold-dual-coordinate-v1",
        "round": int(selection["round"]),
        "rule": selection["selection"]["rule"],
        "tie_break": selection["selection"]["tie_break"],
        "sample_unit": selection["selection"]["sample_unit"],
        "uses_task_labels": False,
        "uses_rollout_success": False,
        "runtime_selector": False,
        "selection_artifact": str(selection_path),
        "selection_artifact_sha256": sha256_file(selection_path),
        "selected_config_id": selected["config_id"],
        "d_func_objective": selected["d_func_objective"],
        "d_pac_objective": selected["d_pac_objective"],
        "normalized_endpoint_regret": selected["normalized_endpoint_regret"],
        "worst_normalized_endpoint_regret": selected[
            "worst_normalized_endpoint_regret"
        ],
    }
    artifact["meta"] = {
        **artifact["meta"],
        "frozen_selection_sha256": sha256_file(selection_path),
        "noise_b_used_for_selection": False,
        "closed_loop_used_for_selection": False,
    }
    artifact_out = Path(args.artifact_out).expanduser().resolve()
    atomic_json(artifact_out, artifact)
    artifact_hash = sha256_file(artifact_out)

    config_id = "blocksoftfold_frozen"
    manifest = {
        "schema_version": 1,
        "kind": "blockwise_errorfold_candidate_manifest",
        "method_id": "blocksoftfold-dual-coordinate-v1",
        "round": int(selection["round"]),
        "raw_correction": source_manifest["raw_correction"],
        "raw_correction_sha256": source_manifest["raw_correction_sha256"],
        "base_profile": selected["profile"],
        "base_profile_sha256": selected["profile_sha256"],
        "gate_levels": source_manifest["gate_levels"],
        "common": source_manifest["common"],
        "candidates": {
            config_id: {
                "errorfold": str(artifact_out),
                "errorfold_sha256": artifact_hash,
                "gate": {"type": "static_block_profile", "errorfold": 0.0},
                "gate_profile": selected["profile"],
                "gate_profile_sha256": selected["profile_sha256"],
                "correction_norm": selected["correction_norm"],
            }
        },
        "selection": artifact["selection"],
    }
    manifest_out = Path(args.manifest_out).expanduser().resolve()
    atomic_json(manifest_out, manifest)
    print(
        json.dumps(
            {
                "artifact": str(artifact_out),
                "artifact_sha256": artifact_hash,
                "manifest": str(manifest_out),
                "manifest_sha256": sha256_file(manifest_out),
                "active_blocks": sum(
                    float(value) != 0.0 for value in selected["profile"].values()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
