#!/usr/bin/env python3
"""Strictly aggregate the frozen 15-task x 20-seed SoftFold experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "scripts/dpac_softfold_15x20_manifest.json"
TASK_SET_ALIASES = {"composite_unseen_long": "composite_unseen"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def expected_layout(manifest: dict[str, Any]) -> tuple[dict[str, str], set[tuple[str, int]]]:
    task_to_set: dict[str, str] = {}
    for raw_set, tasks in manifest["tasks"].items():
        task_set = TASK_SET_ALIASES.get(raw_set, raw_set)
        for task in tasks:
            if task in task_to_set:
                raise ValueError(f"duplicate preregistered task: {task}")
            task_to_set[str(task)] = task_set
    seeds = [int(seed) for seed in manifest["seeds"]]
    expected = {(task, seed) for task in task_to_set for seed in seeds}
    if len(task_to_set) != 15 or len(seeds) != 20 or len(expected) != 300:
        raise ValueError("manifest is not exactly 15 tasks x 20 seeds")
    return task_to_set, expected


def read_rows(
    paths: Iterable[Path],
    *,
    task_to_set: dict[str, str],
    expected: set[tuple[str, int]],
    config: str,
    model: str,
    allow_attestation: bool = False,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    provenance = []
    for path in sorted(set(path.resolve() for path in paths)):
        if not path.is_file():
            continue
        used = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task = str(row.get("task"))
            try:
                seed = int(row.get("seed"))
            except (TypeError, ValueError):
                continue
            key = (task, seed)
            if key not in expected:
                continue
            if row.get("success") is None:
                raise ValueError(f"{path}:{line_number}: selected row lacks success")
            if not allow_attestation and row.get("status") not in (None, "complete"):
                raise ValueError(f"{path}:{line_number}: selected row is not complete")
            normalized = dict(row)
            normalized.update(
                {
                    "model": model,
                    "config": config,
                    "task": task,
                    "task_set": task_to_set[task],
                    "seed": seed,
                    "success": bool(row["success"]),
                    "source_path": str(path),
                    "source_line": line_number,
                }
            )
            if key in rows:
                previous = rows[key]
                if (
                    previous["success"] != normalized["success"]
                    or previous.get("steps") != normalized.get("steps")
                ):
                    raise ValueError(f"conflicting duplicate result for {model}/{config}/{key}")
                continue
            rows[key] = normalized
            used += 1
        if used:
            provenance.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "selected_rows": used,
                }
            )
    missing = sorted(expected - set(rows))
    extra = sorted(set(rows) - expected)
    if missing or extra:
        raise ValueError(
            f"{model}/{config}: incomplete keyset; rows={len(rows)} "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    return rows, provenance


def exact_mcnemar_pvalue(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, candidate_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / 2**discordant
    return min(1.0, 2.0 * probability)


def summarize_config(
    rows: dict[tuple[str, int], dict[str, Any]], task_to_set: dict[str, str]
) -> dict[str, Any]:
    successes = sum(row["success"] for row in rows.values())
    task_sets = sorted(set(task_to_set.values()))
    by_task = {}
    for task in sorted(task_to_set):
        selected = [row for (name, _), row in rows.items() if name == task]
        count = sum(row["success"] for row in selected)
        by_task[task] = {"successes": count, "episodes": len(selected), "success_rate": count / len(selected)}
    by_task_set = {}
    for task_set in task_sets:
        selected = [row for (task, _), row in rows.items() if task_to_set[task] == task_set]
        count = sum(row["success"] for row in selected)
        by_task_set[task_set] = {
            "successes": count,
            "episodes": len(selected),
            "success_rate": count / len(selected),
        }
    return {
        "successes": successes,
        "episodes": len(rows),
        "success_rate": successes / len(rows),
        "by_task_set": by_task_set,
        "by_task": by_task,
    }


def comparison(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    candidate_only = sum(not baseline[key]["success"] and candidate[key]["success"] for key in baseline)
    baseline_only = sum(baseline[key]["success"] and not candidate[key]["success"] for key in baseline)
    both = sum(baseline[key]["success"] and candidate[key]["success"] for key in baseline)
    neither = len(baseline) - candidate_only - baseline_only - both
    return {
        "paired_episodes": len(baseline),
        "candidate_only_success": candidate_only,
        "baseline_only_success": baseline_only,
        "both_success": both,
        "neither_success": neither,
        "success_delta_episodes": candidate_only - baseline_only,
        "success_rate_delta": (candidate_only - baseline_only) / len(baseline),
        "mcnemar_exact_two_sided_p": exact_mcnemar_pvalue(baseline_only, candidate_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO / "runs/dpac_softfold_15x20_v1"))
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    task_to_set, expected = expected_layout(manifest)

    gr00t_control_roots = {
        "atomic_seen": REPO / "runs/robocasa365_official_full_atomic_paired50",
        "composite_seen": REPO / "runs/robocasa365_official_full_composite_seen_paired50",
        "composite_unseen": REPO / "runs/robocasa365_official_full_composite_unseen_paired50",
    }
    sources: dict[tuple[str, str], tuple[list[Path], bool]] = {
        ("gr00t", "current_gdsq"): (
            [path for directory in gr00t_control_roots.values() for path in directory.glob("cscka_final_s*.jsonl")],
            False,
        ),
        ("gr00t", "hard_selector_v8"): (
            [REPO / "runs/gdsq_week1_preregistered_v1/gr00t_selector_reuse/episode_attestation.jsonl"],
            True,
        ),
        ("gr00t", "softfold_dfunc"): (
            list((root / "closed_loop/gr00t").glob("*/softfold_dfunc_s*.jsonl")),
            False,
        ),
        ("gr00t", "softfold_dpac"): (
            list((root / "closed_loop/gr00t").glob("*/softfold_dpac_s*.jsonl")),
            False,
        ),
        ("pi05", "current_gdsq"): (
            list((REPO / "runs/pi05_gdsq_gr00t_aligned/official_target_paired50/results/gdsq_vla").glob("*.jsonl")),
            False,
        ),
        ("pi05", "hard_selector_v8"): (
            list((REPO / "runs/gdsq_week1_preregistered_v1/pi05_selector_official50/results/gdsq_vla_runtime_selector").glob("*.jsonl")),
            False,
        ),
        ("pi05", "softfold_dfunc"): (
            list((root / "closed_loop/pi05/results/gdsq_vla_softfold_dfunc").glob("*.jsonl")),
            False,
        ),
        ("pi05", "softfold_dpac"): (
            list((root / "closed_loop/pi05/results/gdsq_vla_softfold_dpac").glob("*.jsonl")),
            False,
        ),
    }

    all_rows: dict[str, dict[str, dict[tuple[str, int], dict[str, Any]]]] = {}
    provenance: dict[str, Any] = {
        "manifest": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST)},
        "sources": {},
    }
    for (model, config), (paths, allow_attestation) in sources.items():
        rows, artifacts = read_rows(
            paths,
            task_to_set=task_to_set,
            expected=expected,
            config=config,
            model=model,
            allow_attestation=allow_attestation,
        )
        all_rows.setdefault(model, {})[config] = rows
        provenance["sources"][f"{model}/{config}"] = artifacts
        output = root / "selected_rows" / model / f"{config}.jsonl"
        rendered = "".join(json.dumps(rows[key], sort_keys=True) + "\n" for key in sorted(rows))
        atomic_text(output, rendered)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "dpac_softfold_15x20_closed_loop_summary",
        "manifest_sha256": provenance["manifest"]["sha256"],
        "tasks": len(task_to_set),
        "seeds": len(manifest["seeds"]),
        "episodes_per_config": len(expected),
        "models": {},
    }
    for model, configs in sorted(all_rows.items()):
        model_summary = {
            "configs": {name: summarize_config(rows, task_to_set) for name, rows in sorted(configs.items())},
            "paired_vs_current_gdsq": {},
        }
        baseline = configs["current_gdsq"]
        for config, rows in sorted(configs.items()):
            if config != "current_gdsq":
                model_summary["paired_vs_current_gdsq"][config] = comparison(baseline, rows)
        model_summary["paired_dpac_vs_dfunc"] = comparison(
            configs["softfold_dfunc"], configs["softfold_dpac"]
        )
        summary["models"][model] = model_summary

    atomic_text(root / "source_provenance.json", json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    atomic_text(root / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown = ["# D_PAC + SoftFold 15 x 20 closed-loop", ""]
    markdown.append("| Model | Config | Success | Episodes | SR |")
    markdown.append("|---|---|---:|---:|---:|")
    for model, model_summary in summary["models"].items():
        for config, row in model_summary["configs"].items():
            markdown.append(
                f"| {model} | {config} | {row['successes']} | {row['episodes']} | {row['success_rate']:.4f} |"
            )
    markdown.extend(["", "All comparisons use the identical frozen 300 (task, seed) keys.", ""])
    atomic_text(root / "summary.md", "\n".join(markdown))
    print(json.dumps({"summary": str(root / "summary.json"), "models": sorted(summary["models"]), "complete": True}, indent=2))


if __name__ == "__main__":
    main()
