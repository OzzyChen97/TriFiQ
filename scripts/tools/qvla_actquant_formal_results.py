#!/usr/bin/env python3
"""Canonicalize and strictly aggregate QVLA/ActQuant Table-1 episodes.

The canonicalization step binds every raw episode to an immutable arm manifest.
The aggregation step refuses partial, duplicate, or protocol-incompatible rows.
It is intentionally the only producer of ``ready_for_table_update=true`` for
the QVLA/ActQuant extension.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "scripts" / "qvla_actquant_table1_protocol.json"
FULL_CONTEXT_PATH = REPO_ROOT / "scripts" / "quantvla_full_context_protocol_v2.json"
METHODS = {"qvla", "actquant", "dypac"}
MODELS = {"gr00t", "pi05"}
CANDIDATE_KEYS = {
    ("qvla", "gr00t"),
    ("actquant", "gr00t"),
    ("qvla", "pi05"),
    ("actquant", "pi05"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def load_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    full_context = json.loads(FULL_CONTEXT_PATH.read_text(encoding="utf-8"))
    expected = protocol["benchmark"]["task_source_sha256"]
    actual = sha256_file(FULL_CONTEXT_PATH)
    if actual != expected:
        raise ValueError(f"frozen task protocol SHA drift: {actual} != {expected}")
    return protocol, full_context


def task_inventory() -> tuple[dict[str, list[str]], dict[str, str]]:
    _protocol, full_context = load_protocol()
    splits = full_context["table1"]["tasks"]
    task_to_split: dict[str, str] = {}
    for split, tasks in splits.items():
        for task in tasks:
            if task in task_to_split:
                raise ValueError(f"task appears in multiple splits: {task}")
            task_to_split[task] = split
    if len(task_to_split) != 50:
        raise ValueError(f"Table-1 task inventory must contain 50 tasks, got {len(task_to_split)}")
    return splits, task_to_split


def expected_episode_keys() -> set[tuple[str, str, int]]:
    splits, _ = task_inventory()
    return {
        (split, task, seed)
        for split, tasks in splits.items()
        for task in tasks
        for seed in range(50)
    }


def load_arm_manifest(path: Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported arm manifest schema")
    if manifest.get("kind") != "qvla_actquant_formal_arm" or manifest.get("immutable") is not True:
        raise ValueError(f"{path}: arm manifest is not immutable")
    method = str(manifest.get("method", "")).lower()
    model = str(manifest.get("model", "")).lower()
    if method not in METHODS or model not in MODELS:
        raise ValueError(f"{path}: unsupported method/model {method}/{model}")
    if manifest.get("test_feedback_allowed") is not False:
        raise ValueError(f"{path}: test-feedback prohibition is missing")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise ValueError(f"{path}: QVLA/ActQuant protocol SHA mismatch")
    expected_flow = int(protocol["models"][model]["flow_steps"])
    if int(manifest.get("flow_steps", -1)) != expected_flow:
        raise ValueError(f"{path}: flow_steps drift")
    allowed = manifest.get("allowed_server_metadata_sha256") or []
    if not allowed or len(allowed) != len(set(allowed)):
        raise ValueError(f"{path}: server metadata allow-list is empty or duplicated")
    if any(len(str(value)) != 64 for value in allowed):
        raise ValueError(f"{path}: malformed server metadata SHA")
    if method in {"qvla", "actquant"}:
        provenance_path = Path(manifest.get("source_provenance", "")).expanduser()
        if not provenance_path.is_absolute():
            provenance_path = (REPO_ROOT / provenance_path).resolve()
        if (
            not provenance_path.is_file()
            or sha256_file(provenance_path) != manifest.get("source_provenance_sha256")
        ):
            raise ValueError(f"{path}: source provenance is missing or changed")
        artifacts = manifest.get("artifacts")
        if artifacts is None and manifest.get("artifact"):
            artifacts = [manifest["artifact"]]
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"{path}: candidate arm does not bind its frozen artifacts")
        required = ("path", "sha256", "pack_manifest_sha256", "model_unit")
        units = set()
        for artifact in artifacts:
            if any(not artifact.get(key) for key in required):
                raise ValueError(f"{path}: incomplete frozen artifact record")
            artifact_path = Path(artifact["path"]).expanduser()
            if not artifact_path.is_absolute():
                artifact_path = (REPO_ROOT / artifact_path).resolve()
            if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
                raise ValueError(f"{path}: frozen artifact is missing or changed")
            pack_manifest_path = Path(artifact.get("pack_manifest", "")).expanduser()
            if not pack_manifest_path.is_absolute():
                pack_manifest_path = (REPO_ROOT / pack_manifest_path).resolve()
            if (
                not pack_manifest_path.is_file()
                or sha256_file(pack_manifest_path) != artifact["pack_manifest_sha256"]
            ):
                raise ValueError(f"{path}: pack manifest is missing or changed")
            pack_manifest = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
            if (
                pack_manifest.get("method") != method
                or pack_manifest.get("model_unit") != artifact["model_unit"]
                or pack_manifest.get("source_protocol_equivalent") is not False
                or not pack_manifest.get("target_inventory_sha256")
                or not pack_manifest.get("exclusion_inventory_sha256")
                or not pack_manifest.get("precision_semantics")
                or not (pack_manifest.get("implementation") or {}).get("sha256")
            ):
                raise ValueError(f"{path}: pack manifest semantic identity drift")
            implementation = pack_manifest["implementation"]
            implementation_records = implementation.get("files") or []
            if canonical_sha256(implementation_records) != implementation["sha256"]:
                raise ValueError(f"{path}: implementation inventory hash drift")
            for source_record in implementation_records:
                source_path = (REPO_ROOT / source_record["path"]).resolve()
                if not source_path.is_file() or sha256_file(source_path) != source_record["sha256"]:
                    raise ValueError(f"{path}: frozen implementation changed: {source_path}")
            if method == "qvla":
                packed = pack_manifest.get("pack") or {}
                packed_path = Path(packed.get("path", "")).expanduser().resolve()
                if (
                    packed_path != artifact_path
                    or packed.get("sha256") != artifact["sha256"]
                    or not packed_path.is_file()
                ):
                    raise ValueError(f"{path}: QVLA payload binding drift")
            else:
                if artifact_path != pack_manifest_path:
                    raise ValueError(f"{path}: ActQuant runtime must be its bundle manifest")
                for file_record in pack_manifest.get("files") or []:
                    gguf_path = Path(file_record["path"]).expanduser()
                    if not gguf_path.is_absolute():
                        gguf_path = (pack_manifest_path.parent / gguf_path).resolve()
                    if not gguf_path.is_file() or sha256_file(gguf_path) != file_record["sha256"]:
                        raise ValueError(f"{path}: ActQuant GGUF payload changed: {gguf_path}")
            units.add(str(artifact["model_unit"]))
        expected_units = (
            {
                "gr00t_atomic_seen",
                "gr00t_composite_seen",
                "gr00t_composite_unseen",
            }
            if model == "gr00t"
            else {"pi05_all_target"}
        )
        if units != expected_units:
            raise ValueError(f"{path}: artifact model-unit coverage drift: {sorted(units)}")
    return manifest, sha256_file(path)


def expand_result_files(manifest: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for expression in manifest.get("result_globs") or []:
        raw = Path(expression).expanduser()
        if raw.is_absolute():
            root = raw.anchor
            pattern = str(raw)[len(root) :].lstrip("/")
            matches = list(Path(root).glob(pattern))
        else:
            matches = list(REPO_ROOT.glob(str(raw)))
        paths.extend(path.resolve() for path in matches if path.is_file())
    unique = sorted(set(paths))
    if not unique:
        raise ValueError("arm manifest result_globs matched no files")
    return unique


def validate_raw_protocol(
    row: dict[str, Any], *, model: str, split: str, task: str, seed: int, source: str
) -> None:
    protocol, _ = load_protocol()
    expected_flow = int(protocol["models"][model]["flow_steps"])
    expected = {
        "status": "complete",
        "split": "target",
        "flow_steps": expected_flow,
        "n_action_steps": 16,
        "replan_steps": 16,
        "paired_action_noise": True,
        "fresh_environment": True,
        "official_task_horizon": True,
    }
    mismatches = {key: (row.get(key), value) for key, value in expected.items() if row.get(key) != value}
    if row.get("render") is not True and row.get("render_enabled") is not True:
        mismatches["render"] = (row.get("render", row.get("render_enabled")), True)
    if model == "pi05" and row.get("task_set") != split:
        mismatches["task_set"] = (row.get("task_set"), split)
    if int(row.get("seed", -1)) != seed or row.get("task") != task:
        mismatches["key"] = ((row.get("task"), row.get("seed")), (task, seed))
    if mismatches:
        raise ValueError(f"{source}: formal protocol drift: {mismatches}")
    if not isinstance(row.get("success"), bool):
        raise ValueError(f"{source}: success must be boolean")
    if row.get("runtime_selector_enabled") is not False:
        raise ValueError(f"{source}: DyPAC/ATM/OHB selector must be disabled")


def canonicalize(manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest, manifest_sha = load_arm_manifest(manifest_path)
    method = str(manifest["method"]).lower()
    model = str(manifest["model"]).lower()
    allowed_metadata = set(manifest["allowed_server_metadata_sha256"])
    _splits, task_to_split = task_inventory()
    expected = expected_episode_keys()
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    source_files = expand_result_files(manifest)
    for path in source_files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            source = f"{path}:{line_number}"
            row = json.loads(line)
            task = str(row.get("task"))
            seed = int(row.get("seed", -1))
            if task not in task_to_split:
                raise ValueError(f"{source}: task is outside frozen Table-1 inventory: {task}")
            split = task_to_split[task]
            key = (split, task, seed)
            if key not in expected:
                raise ValueError(f"{source}: episode is outside formal seed/task set: {key}")
            if key in records:
                raise ValueError(f"duplicate formal episode key {key}: {source}")
            metadata_sha = row.get("server_metadata_sha256")
            if metadata_sha not in allowed_metadata:
                raise ValueError(f"{source}: server metadata SHA is not registered")
            validate_raw_protocol(row, model=model, split=split, task=task, seed=seed, source=source)
            record = {
                "schema_version": 1,
                "method": method,
                "model": model,
                "task_split": split,
                "task": task,
                "seed": seed,
                "success": row["success"],
                "status": "complete",
                "server_metadata_sha256": metadata_sha,
                "arm_manifest_path": str(manifest_path),
                "arm_manifest_sha256": manifest_sha,
                "protocol_sha256": manifest["protocol_sha256"],
                "flow_steps": int(row["flow_steps"]),
                "n_action_steps": int(row["n_action_steps"]),
                "paired_action_noise": True,
                "source_file": str(path),
                "source_line": line_number,
            }
            for timing_key in ("episode_wall_seconds", "inference_seconds", "env_step_seconds"):
                if row.get(timing_key) is not None:
                    record[timing_key] = float(row[timing_key])
            record["record_sha256"] = canonical_sha256(record)
            records[key] = record
    actual = set(records)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"formal arm incomplete: completed={len(actual)}/2500, "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    ordered = [records[key] for key in sorted(records)]
    atomic_jsonl(output, ordered)
    return {
        "method": method,
        "model": model,
        "episodes": len(ordered),
        "canonical_jsonl": str(output.resolve()),
        "canonical_jsonl_sha256": sha256_file(output),
        "arm_manifest_sha256": manifest_sha,
    }


def load_canonical(path: Path) -> tuple[tuple[str, str], dict[tuple[str, str, int], dict[str, Any]]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    identity: tuple[str, str] | None = None
    manifest_shas: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        source = f"{path}:{line_number}"
        if row.get("status") != "complete" or row.get("schema_version") != 1:
            raise ValueError(f"{source}: incomplete/unsupported canonical row")
        row_identity = (str(row.get("method")), str(row.get("model")))
        if identity is None:
            identity = row_identity
        elif identity != row_identity:
            raise ValueError(f"{source}: multiple arm identities in one canonical file")
        key = (str(row["task_split"]), str(row["task"]), int(row["seed"]))
        if key in rows:
            raise ValueError(f"{source}: duplicate canonical key {key}")
        check = dict(row)
        declared_sha = check.pop("record_sha256", None)
        if declared_sha != canonical_sha256(check):
            raise ValueError(f"{source}: canonical record SHA mismatch")
        manifest_shas.add(str(row.get("arm_manifest_sha256")))
        rows[key] = row
    if identity is None or len(manifest_shas) != 1:
        raise ValueError(f"{path}: missing rows or mixed arm manifests")
    expected = expected_episode_keys()
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise ValueError(
            f"{path}: canonical arm incomplete: completed={len(rows)}/2500, "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return identity, rows


def rates(rows: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    splits, _ = task_inventory()
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
    seed: int = 0,
) -> dict[str, Any]:
    splits, _ = task_inventory()
    tasks = [task for split_tasks in splits.values() for task in split_tasks]
    task_to_split = {task: split for split, values in splits.items() for task in values}
    deltas = np.asarray(
        [
            [
                float(candidate[(task_to_split[task], task, episode_seed)]["success"])
                - float(baseline[(task_to_split[task], task, episode_seed)]["success"])
                for episode_seed in range(50)
            ]
            for task in tasks
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
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
        "seed": seed,
        "observed_task_macro_delta": float(deltas.mean()),
        "bootstrap_mean_delta": float(samples.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    result: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (count - rank) * pvalues[key])
        result[key] = min(1.0, running)
    return result


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name.lower(), Path(raw_path).expanduser().resolve()


def aggregate(
    candidate_paths: list[Path], baseline_paths: dict[str, Path], output: Path, draws: int
) -> dict[str, Any]:
    candidates: dict[tuple[str, str], tuple[Path, dict]] = {}
    for path in candidate_paths:
        identity, rows = load_canonical(path)
        if identity in candidates:
            raise ValueError(f"duplicate candidate arm {identity}")
        candidates[identity] = (path, rows)
    if set(candidates) != CANDIDATE_KEYS:
        raise ValueError(f"candidate family drift: {sorted(candidates)} != {sorted(CANDIDATE_KEYS)}")
    if set(baseline_paths) != MODELS:
        raise ValueError("aggregation requires exactly gr00t and pi05 DyPAC baselines")
    baselines: dict[str, tuple[Path, dict]] = {}
    for model, path in baseline_paths.items():
        identity, rows = load_canonical(path)
        if identity != ("dypac", model):
            raise ValueError(f"{path}: expected dypac/{model}, got {identity}")
        baselines[model] = (path, rows)

    comparisons: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    arm_rates: dict[str, Any] = {}
    for (method, model), (path, rows) in sorted(candidates.items()):
        identifier = f"{method}_{model}_vs_dypac_{model}"
        baseline_rows = baselines[model][1]
        wins = sum(int(rows[key]["success"] and not baseline_rows[key]["success"]) for key in rows)
        losses = sum(int(baseline_rows[key]["success"] and not rows[key]["success"]) for key in rows)
        pvalue = exact_mcnemar(wins, losses)
        raw_pvalues[identifier] = pvalue
        candidate_rates = rates(rows)
        first_row = next(iter(rows.values()))
        arm_manifest_path = Path(first_row["arm_manifest_path"])
        if sha256_file(arm_manifest_path) != first_row["arm_manifest_sha256"]:
            raise ValueError(f"candidate arm manifest changed: {arm_manifest_path}")
        arm_manifest = json.loads(arm_manifest_path.read_text(encoding="utf-8"))
        arm_rates[f"{method}_{model}"] = {
            "canonical_jsonl": str(path),
            "canonical_jsonl_sha256": sha256_file(path),
            "arm_manifest": str(arm_manifest_path),
            "arm_manifest_sha256": first_row["arm_manifest_sha256"],
            "storage": arm_manifest.get("storage"),
            "artifacts": arm_manifest.get("artifacts"),
            **candidate_rates,
        }
        comparisons[identifier] = {
            "candidate": f"{method}_{model}",
            "baseline": f"dypac_{model}",
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": 2500 - wins - losses,
            "exact_two_sided_mcnemar_p": pvalue,
            "task_then_seed_hierarchical_bootstrap": hierarchical_bootstrap(
                rows, baseline_rows, draws=draws, seed=0
            ),
        }
    adjusted = holm_adjust(raw_pvalues)
    for identifier, value in adjusted.items():
        comparisons[identifier]["holm_adjusted_mcnemar_p"] = value
    for model, (path, rows) in sorted(baselines.items()):
        arm_rates[f"dypac_{model}"] = {
            "canonical_jsonl": str(path),
            "canonical_jsonl_sha256": sha256_file(path),
            **rates(rows),
        }
    payload = {
        "schema_version": 1,
        "kind": "qvla_actquant_robocasa365_table1_aggregate",
        "complete": True,
        "ready_for_table_update": True,
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "formal_episode_count_new": 10_000,
        "bootstrap_draws": draws,
        "bootstrap_seed": 0,
        "holm_family": sorted(comparisons),
        "arms": arm_rates,
        "comparisons": comparisons,
        "selection_feedback_allowed": False,
        "comparison_scope": {
            "gr00t": "paired to complete compatible GR00T DyPAC flow-4 episodes",
            "pi05": "paired to complete compatible pi0.5 DyPAC four-flow-step episodes",
            "pi05_legacy_flow4": "descriptive only; protocol-incompatible and excluded from tests",
        },
    }
    atomic_json(output, payload)
    return {
        "out": str(output.resolve()),
        "out_sha256": sha256_file(output),
        "ready_for_table_update": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    canonical_parser = subparsers.add_parser("canonicalize")
    canonical_parser.add_argument("--arm-manifest", required=True)
    canonical_parser.add_argument("--out", required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--candidate", action="append", required=True)
    aggregate_parser.add_argument("--baseline", action="append", type=parse_named_path, required=True)
    aggregate_parser.add_argument("--bootstrap", type=int, default=10_000)
    aggregate_parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "canonicalize":
        result = canonicalize(
            Path(args.arm_manifest).expanduser().resolve(), Path(args.out).expanduser().resolve()
        )
    else:
        if args.bootstrap != 10_000:
            raise ValueError("formal protocol requires exactly 10,000 bootstrap draws")
        baselines = dict(args.baseline)
        if len(baselines) != len(args.baseline):
            raise ValueError("duplicate baseline model")
        result = aggregate(
            [Path(path).expanduser().resolve() for path in args.candidate],
            baselines,
            Path(args.out).expanduser().resolve(),
            args.bootstrap,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
