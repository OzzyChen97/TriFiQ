#!/usr/bin/env python3
"""Strict standalone aggregation for the DyPAC-VLA pi0.5 RoboCasa365 row."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from quantvla_cross_model_protocol import (  # noqa: E402
    closed_loop_row_protocol,
    validate_closed_loop_row,
)
from quantvla_full_context import (  # noqa: E402
    PROTOCOL,
    protocol_attestation,
    require_protocol_attestation,
)


CONFIG_ID = "full_context_w4a8_dynamic_profile"
EXPECTED_PLAN_SHA256 = "e502f7cd7d126517c000f8b5b8e7c6e5537b31226910a2dd6834caaebf736c83"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def expected_keys() -> set[tuple[str, str, int]]:
    return {
        (split, task, seed)
        for split, tasks in PROTOCOL["table1"]["tasks"].items()
        for task in tasks
        for seed in range(50)
    }


def load_rows(run_dir: Path, manifest: dict[str, Any]) -> dict[tuple[str, str, int], dict]:
    allowed_metadata = {
        row["server_metadata_sha256"] for row in manifest.get("servers") or []
    }
    if not allowed_metadata:
        raise ValueError("formal manifest has no runtime-attested server")
    rows: dict[tuple[str, str, int], dict] = {}
    result_dir = run_dir / "results" / CONFIG_ID
    for path in sorted(result_dir.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "complete":
                raise ValueError(f"{path}:{line_number}: incomplete episode row")
            if row.get("config") != CONFIG_ID:
                raise ValueError(f"{path}:{line_number}: wrong config")
            split = str(row.get("task_set"))
            task = str(row.get("task"))
            seed = int(row.get("seed"))
            key = (split, task, seed)
            if key in rows:
                raise ValueError(f"duplicate formal key {key} in {path}:{line_number}")
            if row.get("server_metadata_sha256") not in allowed_metadata:
                raise ValueError(f"{path}:{line_number}: unregistered server runtime")
            row_protocol = closed_loop_row_protocol()
            row_protocol["flow_steps"] = 4
            protocol_mismatches = {
                name: (row.get(name), expected)
                for name, expected in row_protocol.items()
                if row.get(name) != expected
            }
            if protocol_mismatches:
                raise ValueError(
                    f"{path}:{line_number}: closed-loop protocol drift: {protocol_mismatches}"
                )
            validate_closed_loop_row(row, source=f"{path}:{line_number}")
            rows[key] = row
    return rows


def task_then_seed_bootstrap(
    rows: dict[tuple[str, str, int], dict], *, draws: int
) -> dict[str, Any]:
    tasks = [task for values in PROTOCOL["table1"]["tasks"].values() for task in values]
    task_to_split = {
        task: split for split, values in PROTOCOL["table1"]["tasks"].items() for task in values
    }
    values = np.asarray(
        [
            [float(bool(rows[(task_to_split[task], task, seed)]["success"])) for seed in range(50)]
            for task in tasks
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(0)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        task_indices = generator.integers(0, len(tasks), size=len(tasks))
        task_means = []
        for task_index in task_indices:
            seed_indices = generator.integers(0, 50, size=50)
            task_means.append(float(values[task_index, seed_indices].mean()))
        samples[draw] = float(np.mean(task_means))
    return {
        "draws": draws,
        "rng_seed": 0,
        "mean": float(samples.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def rates(rows: dict[tuple[str, str, int], dict], *, bootstrap: int) -> dict[str, Any]:
    per_task_values: dict[str, list[float]] = defaultdict(list)
    episode_wall = []
    inference_seconds = []
    for (_split, task, _seed), row in rows.items():
        per_task_values[task].append(float(bool(row["success"])))
        episode_wall.append(float(row["episode_wall_seconds"]))
        inference_seconds.append(float(row["inference_seconds"]))
    per_task = {
        task: float(np.mean(values)) for task, values in sorted(per_task_values.items())
    }
    split_macro = {
        split: float(np.mean([per_task[task] for task in tasks]))
        for split, tasks in PROTOCOL["table1"]["tasks"].items()
    }
    successes = sum(int(bool(row["success"])) for row in rows.values())
    return {
        "episodes": len(rows),
        "successes": successes,
        "micro_success_rate": successes / len(rows),
        "task_macro_success_rate": float(np.mean(list(per_task.values()))),
        "split_task_macro_success_rate": split_macro,
        "per_task_success_rate": per_task,
        "task_then_seed_hierarchical_bootstrap": task_then_seed_bootstrap(
            rows, draws=bootstrap
        ),
        "efficiency_observations": {
            "episode_wall_seconds_mean": float(np.mean(episode_wall)),
            "inference_seconds_mean": float(np.mean(inference_seconds)),
            "scope": "queue-loaded formal rollout observations; not a latency speedup claim",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("bootstrap draws must be positive")
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("immutable") is not True or manifest.get("config_id") != CONFIG_ID:
        raise ValueError("invalid DyPAC-VLA pi0.5 formal manifest")
    if manifest.get("frozen_plan_sha256") != EXPECTED_PLAN_SHA256:
        raise ValueError("formal manifest does not bind the frozen DyPAC-VLA plan")
    require_protocol_attestation(manifest, source=str(manifest_path))
    if manifest.get("table1_protocol", {}).get("flow_steps") != 4:
        raise ValueError("pi0.5 Table-1 flow-step drift")
    if manifest.get("result_feedback_allowed_before_completion") is not False:
        raise ValueError("formal result-feedback policy drift")

    rows = load_rows(run_dir, manifest)
    expected = expected_keys()
    actual = set(rows)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"formal coverage incomplete: completed={len(actual)}/2500, "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    payload = {
        "schema_version": 1,
        "kind": "dypac_vla_pi05_robocasa365_table1_standalone_aggregate",
        "complete": True,
        "model": "pi0.5",
        "method": "DyPAC-VLA",
        "config_id": CONFIG_ID,
        "full_context_protocol": protocol_attestation(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_plan_sha256": EXPECTED_PLAN_SHA256,
        "result": rates(rows, bootstrap=args.bootstrap),
        "comparison_scope": (
            "protocol-matched four-flow-step formal row; cross-row differences remain "
            "descriptive until the registered paired significance analysis is recorded"
        ),
        "selection_feedback_allowed": False,
    }
    output = Path(args.out).expanduser().resolve()
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "complete": True,
                "episodes": payload["result"]["episodes"],
                "out": str(output),
                "out_sha256": sha256_file(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
