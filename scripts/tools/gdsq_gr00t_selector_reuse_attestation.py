#!/usr/bin/env python3
"""Attest every reused GR00T episode as a frozen v8-selector baseline row.

The original 2,500 static rows remain byte-for-byte untouched.  This tool
validates their three immutable manifests, the 256-observation bitwise
equivalence gate, and the model-level v8 decision, then emits a derived row
index carrying the selector fields required by the paper protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "runs/gdsq_week1_preregistered_v1"
SELECTOR = REPO_ROOT / "runs/atmohb_dynamic_selector_v8/selector.json"
SELECTOR_SHA256 = "0f3178726c2b784898f18dfde248d9a9bae152da6ffcfdc299e8bdc02f0bd871"
SELECTOR_RULE = "v8_no_oracle_absolute_mechanism_gate_aligned_runtime_rule"
EQUIVALENCE = ROOT / "selector_equivalence/summary.json"
STATIC_SUMMARY = REPO_ROOT / "runs/robocasa365_official_full_paired50/summary.json"
STATIC_SUMMARY_SHA256 = "243162a79a57fa9bf85402364e2cb6bf50c69baa5320fc2256e5821dfd0ef810"
OUTPUT_DIR = ROOT / "gr00t_selector_reuse"
ROWS = OUTPUT_DIR / "episode_attestation.jsonl"
SUMMARY = OUTPUT_DIR / "summary.json"
CONFIG = "cscka_final"
NOISE = "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"
TASK_COUNTS = {"atomic_seen": 18, "composite_seen": 16, "composite_unseen": 16}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def frozen_text(path: Path, rendered: str) -> None:
    if path.exists():
        require(path.read_text(encoding="utf-8") == rendered, f"frozen artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def selector_contract() -> dict[str, Any]:
    require(sha256_file(SELECTOR) == SELECTOR_SHA256, "v8 selector SHA drift")
    selector = load(SELECTOR)
    require(selector.get("rule_name") == SELECTOR_RULE, "selector rule drift")
    require(
        selector.get("fit_metadata", {}).get("atmohb_combination_disabled") is True,
        "ATM+OHB must remain disabled",
    )
    require(
        selector.get("no_oracle_contract", {}).get("uses_runtime_success_feedback") is False,
        "selector uses runtime feedback",
    )
    model = selector["models"]["gr00t"]
    decisions = {
        (row.get("selected_variant"), row.get("selected_config_id"))
        for row in model["tasks"].values()
    }
    require(decisions == {("baseline", CONFIG)}, f"non-model-level GR00T decision: {decisions}")
    fit = model.get("fit") or {}
    require(fit.get("selection_scope") == "model_level_absolute_mechanism_gate", "scope drift")
    require(fit.get("rollout_labels_used") is False, "rollout labels entered selector")
    require(fit.get("runtime_success_feedback_used") is False, "runtime feedback entered selector")
    require(
        fit.get("task_metadata_used_for_variant_selection") is False,
        "task metadata entered selector",
    )
    return {
        "selector_sha256": SELECTOR_SHA256,
        "selector_rule_name": SELECTOR_RULE,
        "selector_model_id": "gr00t",
        "selection_scope": "model_level_absolute_mechanism_gate",
        "selected_variant": "baseline",
        "selected_config_id": CONFIG,
        "atm_enabled": False,
        "ohb_enabled": False,
    }


def equivalence_contract() -> dict[str, Any]:
    summary = load(EQUIVALENCE)
    report = summary.get("reports", {}).get("gr00t", {}).get("result", {})
    comparison = report.get("comparison") or {}
    checks = {
        "complete": summary.get("complete") is True and report.get("complete") is True,
        "observations": report.get("observations") == 256,
        "denoising": report.get("denoising_steps") == 4,
        "bitwise": comparison.get("bitwise_equal_observations") == 256,
        "mismatch": comparison.get("mismatched_observations") == 0,
        "max_abs": float(comparison.get("max_abs", -1.0)) == 0.0,
    }
    failed = [name for name, valid in checks.items() if valid is not True]
    require(not failed, f"GR00T selector equivalence failed: {failed}")
    return artifact(EQUIVALENCE)


def config_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    matches = [row for row in manifest.get("configs", []) if row.get("id") == CONFIG]
    require(len(matches) == 1, f"manifest must contain exactly one {CONFIG} config")
    return matches[0]


def source_rows(
    source: dict[str, Any], selector_fields: dict[str, Any], equivalence_sha: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_set = str(source["task_set"])
    require(task_set in TASK_COUNTS, f"unknown task set: {task_set}")
    run_dir = Path(source["run_dir"]).resolve()
    manifest_path = run_dir / "manifest.json"
    require(sha256_file(manifest_path) == source["manifest_sha256"], f"manifest drift: {task_set}")
    manifest = load(manifest_path)
    config = config_from_manifest(manifest)
    tasks = [str(task) for task in manifest["tasks"]]
    seeds = [int(seed) for seed in manifest["seeds"]]
    require(len(tasks) == TASK_COUNTS[task_set], f"task count drift: {task_set}")
    require(seeds == list(range(50)), f"seed drift: {task_set}")
    require(manifest.get("task_set") == task_set, f"task-set manifest mismatch: {task_set}")
    manifest_sha = source["manifest_sha256"]
    config_sha = config["config_sha256"]
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    result_files = []
    for path_text in config["result_files"]:
        path = Path(path_text).resolve()
        require(path.is_file(), f"missing static result: {path}")
        result_files.append(artifact(path))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("task")), int(row.get("seed", -1)))
            checks = {
                "config": row.get("config") == CONFIG,
                "manifest": row.get("manifest_sha256") == manifest_sha,
                "config_sha": row.get("config_sha256") == config_sha,
                "task": key[0] in tasks,
                "seed": key[1] in seeds,
                "paired_noise": row.get("paired_action_noise") is True,
                "noise_scheme": row.get("action_noise_scheme") == NOISE,
                "success": isinstance(row.get("success"), bool),
            }
            failed = [name for name, valid in checks.items() if not valid]
            require(not failed, f"{path}:{line_number}: invalid static row {failed}")
            require(key not in observed, f"duplicate static key {task_set}/{key}")
            observed[key] = row
    expected = {(task, seed) for task in tasks for seed in seeds}
    require(set(observed) == expected, f"incomplete static matrix: {task_set}")
    derived = []
    for task in tasks:
        for seed in seeds:
            row = observed[(task, seed)]
            derived.append(
                {
                    "schema_version": 1,
                    "kind": "gdsq_vla_selector_reuse_episode_attestation",
                    "model": "gr00t",
                    "task_set": task_set,
                    "task": task,
                    "seed": seed,
                    "success": row["success"],
                    "source_config_id": CONFIG,
                    "source_manifest_sha256": manifest_sha,
                    "source_config_sha256": config_sha,
                    "source_row_sha256": canonical_sha(row),
                    "equivalence_summary_sha256": equivalence_sha,
                    "reuse_static_result": True,
                    "runtime_selector_enabled": True,
                    **selector_fields,
                }
            )
    return derived, {
        "task_set": task_set,
        "manifest": artifact(manifest_path),
        "result_files": result_files,
        "tasks": len(tasks),
        "episodes": len(derived),
    }


def build() -> tuple[str, dict[str, Any]]:
    require(sha256_file(STATIC_SUMMARY) == STATIC_SUMMARY_SHA256, "static summary drift")
    static_summary = load(STATIC_SUMMARY)
    require(static_summary.get("n_tasks") == 50, "static summary task count drift")
    require(static_summary.get("episodes_per_config") == 2500, "static coverage drift")
    selector_fields = selector_contract()
    equivalence = equivalence_contract()
    derived: list[dict[str, Any]] = []
    sources = []
    seen_task_sets = set()
    for source in static_summary["sources"]:
        rows, record = source_rows(source, selector_fields, equivalence["sha256"])
        require(record["task_set"] not in seen_task_sets, "duplicate task set")
        seen_task_sets.add(record["task_set"])
        derived.extend(rows)
        sources.append(record)
    require(seen_task_sets == set(TASK_COUNTS), "static task-set coverage drift")
    keys = {(row["task_set"], row["task"], row["seed"]) for row in derived}
    require(len(derived) == len(keys) == 2500, "derived selector coverage drift")
    rows_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in derived)
    rows_sha = hashlib.sha256(rows_text.encode()).hexdigest()
    summary = {
        "schema_version": 1,
        "kind": "gdsq_vla_gr00t_selector_static_reuse_attestation",
        "valid": True,
        "result_blind": True,
        "raw_results_modified": False,
        "reuse_static_result": True,
        "selector": artifact(SELECTOR),
        "selector_contract": selector_fields,
        "equivalence": equivalence,
        "static_summary": artifact(STATIC_SUMMARY),
        "sources": sources,
        "coverage": {
            "tasks": 50,
            "seeds_per_task": 50,
            "expected_episodes": 2500,
            "observed_episodes": 2500,
            "missing_episodes": 0,
            "duplicate_episodes": 0,
        },
        "episode_attestation": {
            "path": str(ROWS.resolve().relative_to(REPO_ROOT)),
            "sha256": rows_sha,
            "bytes": len(rows_text.encode()),
            "rows": 2500,
        },
    }
    return rows_text, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("materialize", "verify"), nargs="?", default="verify")
    args = parser.parse_args()
    rows_text, summary = build()
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.command == "materialize":
        frozen_text(ROWS, rows_text)
        frozen_text(SUMMARY, summary_text)
    else:
        require(ROWS.is_file() and SUMMARY.is_file(), "reuse attestation is not materialized")
        require(ROWS.read_text(encoding="utf-8") == rows_text, "episode attestation drift")
        require(SUMMARY.read_text(encoding="utf-8") == summary_text, "reuse summary drift")
    print(
        json.dumps(
            {
                "valid": True,
                "command": args.command,
                "rows": 2500,
                "summary": str(SUMMARY),
                "summary_sha256": sha256_file(SUMMARY),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
