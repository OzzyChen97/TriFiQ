#!/usr/bin/env python3
"""Freeze and aggregate an explicitly interim QVLA/ActQuant Table-1 snapshot.

The formal aggregator remains fail-closed until all four 2,500-episode arms
are complete.  This tool is deliberately separate: it takes a content-bound
cut of the currently completed JSONL rows, validates every row against the
frozen protocol and arm manifest, and emits descriptive partial task-macro
rates without paired inference.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

import qvla_actquant_formal_results as formal  # noqa: E402


FORMAL = ROOT / "runs" / "qvla_actquant_table1" / "formal"
ARM_KEYS = (
    ("qvla", "gr00t"),
    ("qvla", "pi05"),
    ("actquant", "gr00t"),
    ("actquant", "pi05"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def partial_rates(
    records: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    splits, _ = formal.task_inventory()
    task_values: dict[str, list[float]] = defaultdict(list)
    for (_split, task, _seed), row in records.items():
        task_values[task].append(float(row["success"]))
    per_task = {
        task: sum(values) / len(values)
        for task, values in sorted(task_values.items())
    }
    successes = sum(int(row["success"]) for row in records.values())
    split_rates: dict[str, float | None] = {}
    split_coverage: dict[str, dict[str, Any]] = {}
    for split, tasks in splits.items():
        observed = [task for task in tasks if task in per_task]
        counts = {task: len(task_values[task]) for task in observed}
        split_rates[split] = (
            sum(per_task[task] for task in observed) / len(observed)
            if observed
            else None
        )
        split_coverage[split] = {
            "episodes_observed": sum(counts.values()),
            "episodes_expected": len(tasks) * 50,
            "tasks_observed": len(observed),
            "tasks_complete": sum(count == 50 for count in counts.values()),
            "tasks_expected": len(tasks),
            "observed_seed_counts_by_task": counts,
        }
    return {
        "episodes": len(records),
        "episodes_expected": 2500,
        "successes": successes,
        "micro_success_rate": successes / len(records),
        "task_macro_success_rate": sum(per_task.values()) / len(per_task),
        "split_task_macro_success_rate": split_rates,
        "per_task_observed_success_rate": per_task,
        "tasks_observed": len(per_task),
        "tasks_complete": sum(len(task_values[task]) == 50 for task in per_task),
        "tasks_expected": 50,
        "split_coverage": split_coverage,
        "metric_scope": (
            "descriptive_task_macro_over_observed_completed_episode_rows; "
            "tasks_equal_weighted; incomplete_tasks_use_their_observed_seeds"
        ),
    }


def snapshot_arm(
    method: str,
    model: str,
    canonical_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = FORMAL / "arm_manifests" / f"{method}_{model}.json"
    manifest, manifest_sha = formal.load_arm_manifest(manifest_path)
    allowed_metadata = set(manifest["allowed_server_metadata_sha256"])
    _splits, task_to_split = formal.task_inventory()
    expected = formal.expected_episode_keys()
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    sources = []
    for path in formal.expand_result_files(manifest):
        raw = path.read_bytes()
        sources.append(
            {
                "path": str(path),
                "bytes_at_snapshot": len(raw),
                "sha256_at_snapshot": content_sha256(raw),
            }
        )
        text = raw.decode("utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            source = f"{path}:{line_number}"
            row = json.loads(line)
            task = str(row.get("task"))
            seed = int(row.get("seed", -1))
            if task not in task_to_split:
                raise ValueError(f"{source}: task outside frozen inventory")
            split = task_to_split[task]
            key = (split, task, seed)
            if key not in expected:
                raise ValueError(f"{source}: episode outside formal task/seed set: {key}")
            if key in records:
                raise ValueError(f"duplicate partial episode key {key}: {source}")
            metadata_sha = row.get("server_metadata_sha256")
            if metadata_sha not in allowed_metadata:
                raise ValueError(f"{source}: unregistered server metadata SHA")
            formal.validate_raw_protocol(
                row,
                model=model,
                split=split,
                task=task,
                seed=seed,
                source=source,
            )
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
                "arm_manifest_path": str(manifest_path.resolve()),
                "arm_manifest_sha256": manifest_sha,
                "protocol_sha256": manifest["protocol_sha256"],
                "flow_steps": int(row["flow_steps"]),
                "n_action_steps": int(row["n_action_steps"]),
                "paired_action_noise": True,
                "source_file": str(path),
                "source_file_sha256_at_snapshot": sources[-1]["sha256_at_snapshot"],
                "source_line": line_number,
            }
            record["record_sha256"] = formal.canonical_sha256(record)
            records[key] = record
    if not records:
        raise ValueError(f"{method}/{model}: no completed formal rows")
    ordered = [records[key] for key in sorted(records)]
    canonical_path = canonical_dir / f"{method}_{model}.jsonl"
    formal.atomic_jsonl(canonical_path, ordered)
    result = {
        **partial_rates(records),
        "canonical_jsonl": str(canonical_path.resolve()),
        "canonical_jsonl_sha256": formal.sha256_file(canonical_path),
        "arm_manifest": str(manifest_path.resolve()),
        "arm_manifest_sha256": manifest_sha,
        "storage": manifest["storage"],
        "artifacts": manifest["artifacts"],
        "source_files": sources,
    }
    return result, ordered


def snapshot(output: Path, canonical_dir: Path) -> dict[str, Any]:
    started = now()
    arms: dict[str, Any] = {}
    record_shas = []
    for method, model in ARM_KEYS:
        arm, records = snapshot_arm(method, model, canonical_dir)
        arms[f"{method}_{model}"] = arm
        record_shas.extend(row["record_sha256"] for row in records)
    observed = sum(row["episodes"] for row in arms.values())
    if observed >= 10_000:
        raise ValueError("partial snapshot refuses complete coverage; use the formal aggregator")
    payload = {
        "schema_version": 1,
        "kind": "qvla_actquant_robocasa365_table1_partial_snapshot",
        "complete": False,
        "ready_for_final_table_update": False,
        "ready_for_partial_table_update": True,
        "snapshot_started_at": started,
        "snapshot_completed_at": now(),
        "snapshot_cut_semantics": "content_bound_non_atomic_cross_file_cut",
        "protocol": str(formal.PROTOCOL_PATH.resolve()),
        "protocol_sha256": formal.sha256_file(formal.PROTOCOL_PATH),
        "formal_episode_count_observed": observed,
        "formal_episode_count_expected": 10_000,
        "arms": arms,
        "canonical_record_set_sha256": formal.canonical_sha256(sorted(record_shas)),
        "selection_feedback_allowed": False,
        "test_results_used_for_pack_or_method_selection": False,
        "paired_inference_reported": False,
        "reporting_scope": (
            "interim_descriptive_snapshot_only; uneven task/seed coverage; "
            "must_be_replaced_by_complete_formal_aggregate"
        ),
    }
    formal.atomic_json(output, payload)
    return {
        "out": str(output.resolve()),
        "out_sha256": formal.sha256_file(output),
        "episodes": observed,
        "ready_for_partial_table_update": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(FORMAL / "partial" / "table1_partial_snapshot.json"),
    )
    parser.add_argument(
        "--canonical-dir",
        default=str(FORMAL / "partial" / "canonical"),
    )
    args = parser.parse_args()
    result = snapshot(
        Path(args.out).expanduser().resolve(),
        Path(args.canonical_dir).expanduser().resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
