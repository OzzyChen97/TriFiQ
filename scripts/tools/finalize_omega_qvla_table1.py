#!/usr/bin/env python3
"""Register the complete FP16/Omega subset without enabling Table 2 claims."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "runs/gdsq_extension_preregistered_v1/omega_qvla_table1"
SUMMARY = ROOT / "aggregate/summary.json"
PROTOCOL = ROOT / "protocol_v1.json"
DOWNLOAD = REPO_ROOT / "checkpoints/omega_qvla/download_manifest.json"
REGISTRY = REPO_ROOT / "docs/gdsq_vla_cvpr2026/experiment_registry.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    for path in (SUMMARY, PROTOCOL, DOWNLOAD, REGISTRY):
        require(path.is_file(), f"missing finalization input: {path}")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    coverage = summary.get("coverage") or {}
    require(summary.get("complete") is True and summary.get("formal_result") is True, "summary is not formal/complete")
    require(coverage.get("expected_episodes") == coverage.get("observed_episodes") == 1600, "coverage is not exact")
    require(not coverage.get("missing_cells") and not coverage.get("invalid_cells"), "summary contains missing or invalid cells")

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    experiment = registry["experiments"]["omega_qvla_table1_reproduction"]
    require(experiment.get("status") in {"queued_post_week1", "running", "complete"}, "unexpected experiment state")
    protocol_record = {
        "path": str(PROTOCOL.relative_to(REPO_ROOT)),
        "sha256": sha256_file(PROTOCOL),
        "result_blind": True,
        "cells": 16,
    }
    registry["preregistrations"]["omega_qvla_table1_v1"] = protocol_record
    experiment.update({
        "status": "complete",
        "preregistration": "omega_qvla_table1_v1",
        "pack_download_manifest": {
            "path": str(DOWNLOAD.relative_to(REPO_ROOT)),
            "sha256": sha256_file(DOWNLOAD),
        },
        "summary": {
            "path": str(SUMMARY.relative_to(REPO_ROOT)),
            "sha256": sha256_file(SUMMARY),
        },
        "coverage": {
            "expected_episodes": 1600,
            "observed_episodes": 1600,
            "missing_episodes": 0,
            "duplicate_episodes": 0,
        },
        "main_claim_enabled": False,
        "notes": "Pinned official FP16/W4A4 subset completed with exact 1,600-episode coverage. It is precursor evidence only and cannot populate the five-configuration Table 2 by itself.",
    })
    joint = registry["experiments"]["libero_table2_five_config_local"]
    joint["precursor_summaries"] = {
        "fp16_omega_subset": {
            "path": str(SUMMARY.relative_to(REPO_ROOT)),
            "sha256": sha256_file(SUMMARY),
            "episodes": 1600,
        }
    }
    registry["paper_claims"]["omega_qvla_local_reproduction"] = {
        "experiment": "libero_table2_five_config_local",
        "enabled": False,
        "reason": "The FP16/Omega precursor covers only 1,600 of the 4,000 jointly preregistered five-configuration episodes.",
    }
    rendered = json.dumps(registry, indent=2) + "\n"
    temporary = REGISTRY.with_name(f".{REGISTRY.name}.tmp.{os.getpid()}")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(REGISTRY)
    print(json.dumps({"status": "complete", "summary_sha256": sha256_file(SUMMARY), "protocol_sha256": sha256_file(PROTOCOL)}, indent=2))


if __name__ == "__main__":
    main()
