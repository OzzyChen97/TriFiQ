#!/usr/bin/env python3
"""Strict aggregation for the four-task, 30-seed paper QuantVLA sanity matrix."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import random


HISTORICAL = {
    "fp16": {
        "OpenCabinet": 17,
        "OpenStandMixerHead": 21,
        "PickPlaceDrawerToCounter": 16,
        "CoffeeSetupMug": 7,
    },
    "historical_w4a8": {
        "OpenCabinet": 9,
        "OpenStandMixerHead": 13,
        "PickPlaceDrawerToCounter": 7,
        "CoffeeSetupMug": 1,
    },
}
COMPARISONS = (
    ("quantvla_w4a8_atmohb", "fp16"),
    ("gdsq_vla_atmohb", "gdsq_vla"),
    ("gdsq_vla", "fp16"),
    ("gdsq_vla_atmohb", "quantvla_w4a8_atmohb"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def load_rows(run_dir: Path, manifest: dict) -> dict[str, dict[tuple[str, int], dict]]:
    expected_tasks = set(manifest["tasks"])
    expected_seeds = set(map(int, manifest["trial_seeds"]))
    expected_noise = manifest["noise_mode"]
    allowed_hash = {row["config"]: row["server_metadata_sha256"] for row in manifest["servers"]}
    result = {config: {} for config in manifest["configs"]}
    for config in manifest["configs"]:
        root = run_dir / "results" / config
        for path in sorted(root.glob("*.jsonl")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row.get("task")), int(row.get("seed", -1)))
                if row.get("status") != "complete" or row.get("config") != config:
                    raise ValueError(f"invalid committed row: {path}:{number}")
                if key[0] not in expected_tasks or key[1] not in expected_seeds:
                    raise ValueError(f"unexpected task/seed: {path}:{number}: {key}")
                if row.get("server_metadata_sha256") != allowed_hash[config]:
                    raise ValueError(f"runtime hash mismatch: {path}:{number}")
                if bool(row.get("paired_action_noise")) != (expected_noise == "paired"):
                    raise ValueError(f"noise mode mismatch: {path}:{number}")
                if key in result[config]:
                    raise ValueError(f"duplicate result key: {config}/{key}")
                result[config][key] = row
    expected = set(itertools.product(expected_tasks, expected_seeds))
    for config, rows in result.items():
        missing = expected - set(rows)
        extra = set(rows) - expected
        if missing or extra:
            raise ValueError(f"incomplete sanity matrix for {config}: missing={len(missing)} extra={len(extra)}")
    return result


def exact_mcnemar(a: dict, b: dict) -> dict:
    a_only = sum(bool(a[key]["success"]) and not bool(b[key]["success"]) for key in a)
    b_only = sum(bool(b[key]["success"]) and not bool(a[key]["success"]) for key in a)
    discordant = a_only + b_only
    if discordant == 0:
        p = 1.0
    else:
        tail = sum(math.comb(discordant, value) for value in range(min(a_only, b_only) + 1))
        p = min(1.0, 2.0 * tail / (2**discordant))
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant, "exact_p": p}


def task_bootstrap(diffs: dict[str, float], draws: int) -> list[float]:
    rng = random.Random(20260819)
    tasks = sorted(diffs)
    values = []
    for _ in range(draws):
        sampled = [rng.choice(tasks) for _ in tasks]
        values.append(sum(diffs[task] for task in sampled) / len(sampled))
    values.sort()
    return [values[int(0.025 * draws)], values[min(draws - 1, int(0.975 * draws))]]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = load_rows(run_dir, manifest)
    tasks = manifest["tasks"]
    seeds = list(map(int, manifest["trial_seeds"]))
    summary = {
        "schema_version": 1,
        "complete": True,
        "noise_mode": manifest["noise_mode"],
        "episodes_per_config": manifest["episodes_per_config"],
        "configs": {},
        "comparisons": {},
        "historical_reference": HISTORICAL,
        "historical_protocol_recoverable": False,
    }
    for config, config_rows in rows.items():
        per_task = {
            task: sum(bool(config_rows[(task, seed)]["success"]) for seed in seeds) / len(seeds)
            for task in tasks
        }
        successes = sum(bool(row["success"]) for row in config_rows.values())
        summary["configs"][config] = {
            "successes": successes,
            "episodes": len(config_rows),
            "episode_sr": successes / len(config_rows),
            "task_macro_sr": sum(per_task.values()) / len(per_task),
            "per_task_sr": per_task,
        }
    for a, b in COMPARISONS:
        diffs = {
            task: summary["configs"][a]["per_task_sr"][task]
            - summary["configs"][b]["per_task_sr"][task]
            for task in tasks
        }
        row = {
            "a": a,
            "b": b,
            "task_macro_delta": sum(diffs.values()) / len(diffs),
            "task_cluster_bootstrap_ci95": task_bootstrap(diffs, args.bootstrap),
            "per_task_delta": diffs,
        }
        if manifest["noise_mode"] == "paired":
            row["episode_paired_mcnemar"] = exact_mcnemar(rows[a], rows[b])
        summary["comparisons"][f"{a}_vs_{b}"] = row
    output = run_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
