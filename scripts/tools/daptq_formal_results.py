#!/usr/bin/env python3
"""Strictly validate and aggregate the DA-PTQ RoboCasa365 formal rollouts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "scripts" / "daptq_table1_protocol.json"
FULL_CONTEXT_PATH = ROOT / "scripts" / "quantvla_full_context_protocol_v2.json"
FORMAL = ROOT / "runs" / "daptq_table1" / "formal"
BASELINE_GLOBS = {
    "gr00t": str(
        ROOT / "runs/full_context_v2/table1/results/full_context_v2/**/*.jsonl"
    ),
    "pi05": str(
        ROOT
        / "runs/full_context_v2/pi05_table1/results/full_context_w4a8_dynamic_profile/*.jsonl"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def inventory() -> tuple[dict[str, list[str]], dict[str, str]]:
    value = json.loads(FULL_CONTEXT_PATH.read_text(encoding="utf-8"))
    splits = value["table1"]["tasks"]
    task_to_split = {task: split for split, tasks in splits.items() for task in tasks}
    if len(task_to_split) != 50 or sum(map(len, splits.values())) != 50:
        raise ValueError("Table-1 task inventory is not 50 unique tasks")
    return splits, task_to_split


def expected_keys() -> set[tuple[str, str, int]]:
    splits, _ = inventory()
    return {
        (split, task, seed)
        for split, tasks in splits.items()
        for task in tasks
        for seed in range(50)
    }


def validate_arm(path: Path, model: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "kind": "daptq_formal_arm",
        "immutable": True,
        "method": "daptq",
        "model": model,
        "flow_steps": 4,
        "episodes": 2500,
        "test_feedback_allowed": False,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
    }
    mismatches = {
        key: (value.get(key), wanted)
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"{path}: DA-PTQ arm identity mismatch: {mismatches}")
    provenance = Path(value["source_provenance"])
    if not provenance.is_file() or sha256_file(provenance) != value["source_provenance_sha256"]:
        raise ValueError(f"{path}: provenance is missing or changed")
    artifacts = value.get("artifacts") or []
    expected_units = (
        {"gr00t_atomic_seen", "gr00t_composite_seen", "gr00t_composite_unseen"}
        if model == "gr00t"
        else {"pi05_all_target"}
    )
    if {row.get("model_unit") for row in artifacts} != expected_units:
        raise ValueError(f"{path}: artifact model-unit coverage drift")
    for record in artifacts:
        manifest_path = Path(record["pack_manifest"])
        arrays_path = Path(record["arrays_path"])
        if (
            not manifest_path.is_file()
            or sha256_file(manifest_path) != record["pack_manifest_sha256"]
            or str(manifest_path.resolve()) != str(Path(record["runtime_path"]).resolve())
            or record["runtime_sha256"] != record["pack_manifest_sha256"]
        ):
            raise ValueError(f"{path}: frozen DA-PTQ manifest changed")
        if not arrays_path.is_file() or sha256_file(arrays_path) != record["arrays_sha256"]:
            raise ValueError(f"{path}: frozen DA-PTQ arrays changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("method") != "DA-PTQ"
            or manifest.get("model_family") != model
            or manifest.get("test_results_used") is not False
            or manifest.get("arrays_sha256") != record["arrays_sha256"]
            or int(manifest.get("flow_steps", -1)) != 4
        ):
            raise ValueError(f"{path}: artifact semantic identity drift")
    allowed = value.get("allowed_server_metadata_sha256") or []
    if not allowed or len(allowed) != len(set(allowed)) or any(len(x) != 64 for x in allowed):
        raise ValueError(f"{path}: invalid server metadata allow-list")
    return value


def expand(expressions: list[str]) -> list[Path]:
    paths = sorted({Path(path).resolve() for expression in expressions for path in glob.glob(expression, recursive=True)})
    if not paths:
        raise ValueError(f"result globs matched no files: {expressions}")
    return paths


def validate_row(
    row: dict[str, Any], *, model: str, split: str, task: str, seed: int, source: str
) -> None:
    expected = {
        "status": "complete",
        "split": "target",
        "flow_steps": 4,
        "n_action_steps": 16,
        "replan_steps": 16,
        "paired_action_noise": True,
        "fresh_environment": True,
        "official_task_horizon": True,
    }
    mismatches = {
        key: (row.get(key), value)
        for key, value in expected.items()
        if row.get(key) != value
    }
    if row.get("render") is not True and row.get("render_enabled") is not True:
        mismatches["render"] = (row.get("render", row.get("render_enabled")), True)
    if row.get("task") != task or int(row.get("seed", -1)) != seed:
        mismatches["key"] = ((row.get("task"), row.get("seed")), (task, seed))
    if model == "pi05" and row.get("task_set") != split:
        mismatches["task_set"] = (row.get("task_set"), split)
    if mismatches:
        raise ValueError(f"{source}: formal protocol mismatch: {mismatches}")
    if not isinstance(row.get("success"), bool):
        raise ValueError(f"{source}: success is not Boolean")
    if row.get("runtime_selector_enabled") is not False:
        raise ValueError(f"{source}: runtime selector was enabled")


def load_rows(
    paths: list[Path],
    *,
    method: str,
    model: str,
    allowed_metadata: set[str] | None,
    required_keys: set[tuple[str, str, int]] | None = None,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    _splits, task_to_split = inventory()
    expected = expected_keys() if required_keys is None else required_keys
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            source = f"{path}:{line_number}"
            row = json.loads(line)
            task = str(row.get("task"))
            if task not in task_to_split:
                raise ValueError(f"{source}: task is outside Table 1")
            split = task_to_split[task]
            seed = int(row.get("seed", -1))
            key = (split, task, seed)
            if key not in expected:
                raise ValueError(f"{source}: key is outside formal inventory: {key}")
            if key in rows:
                raise ValueError(f"{source}: duplicate formal key: {key}")
            validate_row(row, model=model, split=split, task=task, seed=seed, source=source)
            metadata = str(row.get("server_metadata_sha256", ""))
            if allowed_metadata is not None and metadata not in allowed_metadata:
                raise ValueError(f"{source}: unregistered server metadata SHA")
            canonical = {
                "schema_version": 1,
                "method": method,
                "model": model,
                "task_split": split,
                "task": task,
                "seed": seed,
                "success": row["success"],
                "server_metadata_sha256": metadata,
                "flow_steps": 4,
                "n_action_steps": 16,
                "replan_steps": 16,
                "source_file": str(path),
                "source_line": line_number,
            }
            canonical["record_sha256"] = canonical_hash(canonical)
            rows[key] = canonical
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        raise ValueError(
            f"{method}/{model} is incomplete: {len(rows)}/{len(expected)}; missing={missing[:3]}"
        )
    return rows


def receipted_split_paths(model: str, split: str) -> list[Path]:
    manifest_path = FORMAL / "formal_jobs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs") or []
    if (
        manifest.get("kind") != "qvla_actquant_rollout_jobs"
        or manifest.get("stage") != "formal"
        or manifest.get("jobs_sha256") != canonical_hash(jobs)
    ):
        raise ValueError(f"{manifest_path}: formal job manifest identity drift")
    selected = [
        job for job in jobs if job.get("model") == model and job.get("split") == split
    ]
    expected_jobs = len(inventory()[0][split]) * 5
    if len(selected) != expected_jobs:
        raise ValueError(
            f"{model}/{split}: job coverage drift {len(selected)}/{expected_jobs}"
        )
    paths: list[Path] = []
    receipts = FORMAL / "control" / "receipts" / "formal"
    for job in selected:
        receipt_path = receipts / f"{job['id']}.json"
        if not receipt_path.is_file():
            raise ValueError(f"{model}/{split}: missing receipt {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_receipt = {
            "status": "complete",
            "job_id": job["id"],
            "job_sha256": canonical_hash(job),
            "rows": len(job["seeds"]),
        }
        mismatches = {
            key: (receipt.get(key), wanted)
            for key, wanted in expected_receipt.items()
            if receipt.get(key) != wanted
        }
        if mismatches:
            raise ValueError(f"{receipt_path}: receipt identity drift {mismatches}")
        output = Path(receipt["output"])
        if not output.is_file() or sha256_file(output) != receipt["output_sha256"]:
            raise ValueError(f"{receipt_path}: receipted output is missing or changed")
        paths.append(output.resolve())
    if len(paths) != len(set(paths)):
        raise ValueError(f"{model}/{split}: duplicate receipted output path")
    return sorted(paths)


def split_rates(
    rows: dict[tuple[str, str, int], dict[str, Any]], split: str
) -> dict[str, Any]:
    tasks = inventory()[0][split]
    per_task = {
        task: sum(int(rows[(split, task, seed)]["success"]) for seed in range(50)) / 50
        for task in tasks
    }
    successes = sum(int(row["success"]) for row in rows.values())
    return {
        "split": split,
        "tasks": len(tasks),
        "episodes": len(rows),
        "successes": successes,
        "micro_success_rate": successes / len(rows),
        "task_macro_success_rate": float(np.mean(list(per_task.values()))),
        "per_task_success_rate": dict(sorted(per_task.items())),
    }


def rates(rows: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    splits, _ = inventory()
    task_values: dict[str, list[float]] = defaultdict(list)
    for (_split, task, _seed), row in rows.items():
        task_values[task].append(float(row["success"]))
    per_task = {task: float(np.mean(values)) for task, values in sorted(task_values.items())}
    successes = sum(int(row["success"]) for row in rows.values())
    return {
        "episodes": len(rows),
        "successes": successes,
        "micro_success_rate": successes / len(rows),
        "task_macro_success_rate": float(np.mean(list(per_task.values()))),
        "split_task_macro_success_rate": {
            split: float(np.mean([per_task[task] for task in tasks]))
            for split, tasks in splits.items()
        },
        "per_task_success_rate": per_task,
    }


def exact_mcnemar(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, float(Fraction(2 * probability, 2**discordant)))


def hierarchical_bootstrap(
    candidate: dict[tuple[str, str, int], dict[str, Any]],
    baseline: dict[tuple[str, str, int], dict[str, Any]],
    *,
    draws: int = 10_000,
) -> dict[str, Any]:
    splits, _ = inventory()
    tasks = [task for values in splits.values() for task in values]
    task_to_split = {task: split for split, values in splits.items() for task in values}
    deltas = np.asarray(
        [
            [
                float(candidate[(task_to_split[task], task, seed)]["success"])
                - float(baseline[(task_to_split[task], task, seed)]["success"])
                for seed in range(50)
            ]
            for task in tasks
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(0)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        task_indices = generator.integers(0, len(tasks), size=len(tasks))
        task_means = np.empty(len(tasks), dtype=np.float64)
        for index, task_index in enumerate(task_indices):
            seed_indices = generator.integers(0, 50, size=50)
            task_means[index] = deltas[task_index, seed_indices].mean()
        samples[draw] = task_means.mean()
    return {
        "draws": draws,
        "seed": 0,
        "observed_task_macro_delta": float(deltas.mean()),
        "bootstrap_mean_delta": float(samples.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (count - rank) * pvalues[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def aggregate(gr00t_arm: Path, pi05_arm: Path, output: Path) -> dict[str, Any]:
    arm_paths = {"gr00t": gr00t_arm, "pi05": pi05_arm}
    candidates = {}
    baselines = {}
    arms = {}
    canonical_dir = output.parent / "canonical"
    for model, arm_path in arm_paths.items():
        arm = validate_arm(arm_path, model)
        candidate = load_rows(
            expand(arm["result_globs"]),
            method="daptq",
            model=model,
            allowed_metadata=set(arm["allowed_server_metadata_sha256"]),
        )
        baseline = load_rows(
            expand([BASELINE_GLOBS[model]]),
            method="dypac",
            model=model,
            allowed_metadata=None,
        )
        candidate_path = canonical_dir / f"daptq_{model}.jsonl"
        baseline_path = canonical_dir / f"dypac_{model}.jsonl"
        atomic_jsonl(candidate_path, [candidate[key] for key in sorted(candidate)])
        atomic_jsonl(baseline_path, [baseline[key] for key in sorted(baseline)])
        candidates[model] = candidate
        baselines[model] = baseline
        arms[f"daptq_{model}"] = {
            **rates(candidate),
            "arm_manifest": str(arm_path.resolve()),
            "arm_manifest_sha256": sha256_file(arm_path),
            "canonical_jsonl": str(candidate_path.resolve()),
            "canonical_jsonl_sha256": sha256_file(candidate_path),
            "storage": arm["storage"],
            "artifacts": arm["artifacts"],
        }
        arms[f"dypac_{model}"] = {
            **rates(baseline),
            "canonical_jsonl": str(baseline_path.resolve()),
            "canonical_jsonl_sha256": sha256_file(baseline_path),
        }
    comparisons = {}
    pvalues = {}
    for model in ("gr00t", "pi05"):
        candidate, baseline = candidates[model], baselines[model]
        wins = sum(
            int(candidate[key]["success"] and not baseline[key]["success"])
            for key in candidate
        )
        losses = sum(
            int(baseline[key]["success"] and not candidate[key]["success"])
            for key in candidate
        )
        identifier = f"daptq_{model}_vs_dypac_{model}"
        pvalues[identifier] = exact_mcnemar(wins, losses)
        comparisons[identifier] = {
            "candidate": f"daptq_{model}",
            "baseline": f"dypac_{model}",
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": 2500 - wins - losses,
            "exact_two_sided_mcnemar_p": pvalues[identifier],
            "task_then_seed_hierarchical_bootstrap": hierarchical_bootstrap(
                candidate, baseline
            ),
        }
    for identifier, adjusted in holm_adjust(pvalues).items():
        comparisons[identifier]["holm_adjusted_mcnemar_p"] = adjusted
    payload = {
        "schema_version": 1,
        "kind": "daptq_robocasa365_table1_aggregate",
        "complete": True,
        "ready_for_table_update": True,
        "protocol": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "formal_episode_count_new": 5000,
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 0,
        "holm_family": sorted(comparisons),
        "arms": arms,
        "comparisons": comparisons,
        "selection_feedback_allowed": False,
        "comparison_scope": {
            "gr00t": "paired to complete GR00T DyPAC four-flow-step episodes",
            "pi05": "paired to complete pi0.5 DyPAC four-flow-step episodes",
        },
    }
    atomic_json(output, payload)
    return {
        "out": str(output.resolve()),
        "out_sha256": sha256_file(output),
        "ready_for_table_update": True,
    }


def aggregate_split(
    gr00t_arm: Path, pi05_arm: Path, output: Path, split: str
) -> dict[str, Any]:
    split_tasks = inventory()[0][split]
    required = {
        (split, task, seed) for task in split_tasks for seed in range(50)
    }
    canonical_dir = output.parent / "canonical"
    arms: dict[str, Any] = {}
    for model, arm_path in (("gr00t", gr00t_arm), ("pi05", pi05_arm)):
        arm = validate_arm(arm_path, model)
        rows = load_rows(
            receipted_split_paths(model, split),
            method="daptq",
            model=model,
            allowed_metadata=set(arm["allowed_server_metadata_sha256"]),
            required_keys=required,
        )
        canonical_path = canonical_dir / f"daptq_{model}_{split}.jsonl"
        atomic_jsonl(canonical_path, [rows[key] for key in sorted(rows)])
        artifacts = arm["artifacts"]
        selected_artifacts = (
            [row for row in artifacts if row["model_unit"] == f"gr00t_{split}"]
            if model == "gr00t"
            else artifacts
        )
        if len(selected_artifacts) != 1:
            raise ValueError(f"{model}/{split}: artifact selection drift")
        artifact = selected_artifacts[0]
        pack = json.loads(Path(artifact["pack_manifest"]).read_text(encoding="utf-8"))
        if pack.get("source_protocol_equivalent") is not False:
            raise ValueError(f"{model}/{split}: source-equivalence label drift")
        arms[f"daptq_{model}"] = {
            **split_rates(rows, split),
            "arm_manifest": str(arm_path.resolve()),
            "arm_manifest_sha256": sha256_file(arm_path),
            "canonical_jsonl": str(canonical_path.resolve()),
            "canonical_jsonl_sha256": sha256_file(canonical_path),
            "storage": {
                "static_bytes": int(artifact["static_bytes"]),
                "static_gib": int(artifact["static_bytes"]) / 1024**3,
                "fp16_baseline_bytes": int(artifact["fp16_baseline_bytes"]),
                "compression_ratio": float(artifact["compression_ratio"]),
            },
            "allocation": {
                "w4_layers": int(artifact["w4_layers"]),
                "bf16_layers": int(artifact["bf16_layers"]),
            },
            "artifact": artifact,
            "source_protocol_equivalent": False,
            "implementation_policy": pack["implementation_policy"],
        }
    payload = {
        "schema_version": 1,
        "kind": "daptq_robocasa365_table1_split_aggregate",
        "complete": True,
        "ready_for_table_update": True,
        "split": split,
        "tasks_per_model": len(split_tasks),
        "episodes_per_model": len(required),
        "formal_episode_count_new": 2 * len(required),
        "protocol": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "selection_feedback_allowed": False,
        "source_protocol_equivalent": False,
        "arms": arms,
    }
    atomic_json(output, payload)
    return {
        "out": str(output.resolve()),
        "out_sha256": sha256_file(output),
        "ready_for_table_update": True,
        "split": split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-arm", type=Path, required=True)
    parser.add_argument("--pi05-arm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("atomic_seen", "composite_seen", "composite_unseen"),
        help="strictly aggregate one complete split from validated receipts",
    )
    args = parser.parse_args()
    if args.split:
        result = aggregate_split(
            args.gr00t_arm.resolve(),
            args.pi05_arm.resolve(),
            args.out.resolve(),
            args.split,
        )
    else:
        result = aggregate(
            args.gr00t_arm.resolve(), args.pi05_arm.resolve(), args.out.resolve()
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
