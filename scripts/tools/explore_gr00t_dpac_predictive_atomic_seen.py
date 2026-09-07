#!/usr/bin/env python3
"""User-requested exploratory unblinding of the complete atomic_seen split only."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from aggregate_gr00t_dpac_predictive_validity import (
    METRICS,
    correlations,
    delta,
    stratified_bootstrap,
    write_plot,
)
from quantvla_predictive_validity import (
    artifact,
    atomic_json,
    protocol_attestation,
    require_protocol_attestation,
    sha256_file,
)


SPLIT = "atomic_seen"


def read_rows(root: Path) -> list[tuple[Path, int, dict[str, Any]]]:
    rows = []
    for path in sorted(root.glob("*.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.strip():
                rows.append((path, line_number, json.loads(line)))
    return rows


def validate_and_summarize(
    manifest_path: Path,
    score_path: Path,
    mask_root: Path,
    fp16_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    library = json.loads(manifest_path.read_text(encoding="utf-8"))
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    require_protocol_attestation(library, source=str(manifest_path))
    require_protocol_attestation(scores, source=str(score_path))
    if (scores.get("mask_library") or {}).get("sha256") != sha256_file(
        manifest_path
    ):
        raise ValueError("offline score/library hash drift")
    candidates = {row["candidate_id"]: row for row in library["candidates"]}
    tasks = list(library["closed_loop"]["tasks"][SPLIT])
    if len(candidates) != 60 or len(tasks) != 18:
        raise ValueError("atomic_seen inventory drift")
    seed = int(library["closed_loop"]["environment_seed"])
    library_hash = sha256_file(manifest_path)

    masks: dict[tuple[str, str], dict[str, Any]] = {}
    mask_files: set[Path] = set()
    for path, line_number, row in read_rows(mask_root):
        source = f"{path}:{line_number}"
        identifier = row.get("predictive_mask_id")
        task = row.get("task")
        if row.get("status") != "complete" or not isinstance(
            row.get("success"), bool
        ):
            raise ValueError(f"{source}: incomplete outcome")
        if identifier not in candidates or task not in tasks:
            raise ValueError(f"{source}: non-preregistered mask/task")
        if int(row.get("seed", -1)) != seed:
            raise ValueError(f"{source}: seed drift")
        if row.get("predictive_library_sha256") != library_hash:
            raise ValueError(f"{source}: library hash drift")
        if row.get("predictive_plan_sha256") != candidates[identifier]["sha256"]:
            raise ValueError(f"{source}: plan hash drift")
        if row.get("predictive_validity_protocol") != protocol_attestation():
            raise ValueError(f"{source}: protocol drift")
        key = (identifier, task)
        if key in masks:
            raise ValueError(f"duplicate mask outcome: {key}")
        masks[key] = row
        mask_files.add(path.resolve())
    expected = {(identifier, task) for identifier in candidates for task in tasks}
    if set(masks) != expected:
        raise ValueError(
            f"atomic_seen mask coverage must be exactly 1080, got {len(masks)}"
        )

    fp16: dict[str, dict[str, Any]] = {}
    fp16_files: set[Path] = set()
    for path, line_number, row in read_rows(fp16_root):
        source = f"{path}:{line_number}"
        task = row.get("task")
        if row.get("status") != "complete" or not isinstance(
            row.get("success"), bool
        ):
            raise ValueError(f"{source}: incomplete FP16 outcome")
        if task not in tasks or int(row.get("seed", -1)) != seed:
            raise ValueError(f"{source}: FP16 task/seed drift")
        if row.get("predictive_mask_id") is not None or task in fp16:
            raise ValueError(f"{source}: duplicate or masked FP16 outcome")
        fp16[task] = row
        fp16_files.add(path.resolve())
    if set(fp16) != set(tasks):
        raise ValueError(f"atomic_seen FP16 coverage must be exactly 18, got {len(fp16)}")

    fp16_sr = float(np.mean([fp16[task]["success"] for task in tasks]))
    summaries = []
    for identifier, candidate in sorted(candidates.items()):
        mask_sr = float(np.mean([masks[(identifier, task)]["success"] for task in tasks]))
        metric = scores["scores"][identifier]["split_scores"][SPLIT]
        summaries.append(
            {
                "predictive_mask_id": identifier,
                "swap_count": int(candidate["swap_count"]),
                "hamming_distance": int(candidate["hamming_distance"]),
                "mse": float(metric["mse"]),
                "d_func": float(metric["d_func"]),
                "d_pac": float(metric["d_pac"]),
                "fp16_success_rate": fp16_sr,
                "mask_success_rate": mask_sr,
                "success_drop": fp16_sr - mask_sr,
            }
        )
    provenance = {
        "manifest": artifact(manifest_path),
        "offline_scores": artifact(score_path),
        "mask_files": [artifact(path) for path in sorted(mask_files)],
        "fp16_files": [artifact(path) for path in sorted(fp16_files)],
    }
    return summaries, provenance


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = {metric: correlations(rows, metric) for metric in METRICS}
    gaps = {
        baseline: {
            "delta_spearman_rho": delta(
                primary["d_pac"], primary[baseline], "spearman_rho"
            ),
            "delta_kendall_tau_b": delta(
                primary["d_pac"], primary[baseline], "kendall_tau_b"
            ),
        }
        for baseline in ("mse", "d_func")
    }
    by_distance = {}
    for distance in sorted({row["swap_count"] for row in rows}):
        selected = [row for row in rows if row["swap_count"] == distance]
        by_distance[str(distance)] = {
            metric: correlations(selected, metric) for metric in METRICS
        }
    return {
        "primary": primary,
        "primary_gaps": gaps,
        "within_swap_count": by_distance,
        "stratified_bootstrap": stratified_bootstrap(rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "predictive_mask_id",
        "swap_count",
        "hamming_distance",
        *METRICS,
        "fp16_success_rate",
        "mask_success_rate",
        "success_drop",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--offline-scores", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--fp16-root", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    out = Path(args.out_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, provenance = validate_and_summarize(
        Path(args.manifest).resolve(),
        Path(args.offline_scores).resolve(),
        Path(args.mask_root).resolve(),
        Path(args.fp16_root).resolve(),
    )
    result = analyze(rows)
    payload = {
        "schema_version": 1,
        "kind": "gr00t_dpac_predictive_atomic_seen_exploratory_unblinding",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "user_requested_interim_unblinding": True,
        "scope": {
            "split": SPLIT,
            "candidate_count": 60,
            "mask_rows": 1080,
            "fp16_rows": 18,
            "complete_balanced_split": True,
            "composite_outcomes_read": False,
            "primary_full_study_status": "still sealed and running",
        },
        "analysis": result,
        "mask_results": rows,
        "provenance": provenance,
    }
    atomic_json(out / "audit.json", payload)
    write_csv(out / "mask_results.csv", rows)
    write_plot(out / "metric_vs_success_drop.png", rows, result)
    print(json.dumps({"out": str(out), "analysis": result}, indent=2))


if __name__ == "__main__":
    main()
