#!/usr/bin/env python3
"""Strict aggregation for the preregistered pi0.5 v8 selector run.

The selector run contains one configuration.  Its paired reference rows are
read from the immutable four-configuration static run, but only after both the
new selector attestation and the frozen static artifact hashes are verified.
Incomplete output is progress-only and is never promoted to a paper result.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from typing import Any

import robocasa  # noqa: F401 -- registers RoboCasa environments
from robocasa.utils.dataset_registry_utils import get_task_horizon

from parse_robocasa_atomic_matrix import (
    cluster_ci,
    exact_sign_flip_p,
    holm_adjust,
    mcnemar,
    paired_delta_ci,
    percentile,
)


CONFIG = "gdsq_vla_runtime_selector"
REFERENCE_CONFIGS = (
    "fp16",
    "gdsq_vla",
    "gdsq_vla_atmohb",
    "quantvla_w4a8_atmohb",
)
CONTRASTS = tuple((CONFIG, value) for value in REFERENCE_CONFIGS)
DEVELOPMENT_TASKS = {
    "OpenCabinet",
    "OpenStandMixerHead",
    "PickPlaceDrawerToCounter",
    "CoffeeSetupMug",
}
RNG_SEED = 20260823


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def expected_layout(manifest: dict[str, Any]) -> tuple[dict[str, str], list[str], list[int]]:
    protocol = manifest["protocol"]
    task_sets = protocol["task_sets"]
    task_to_set = {
        task: task_set for task_set, values in task_sets.items() for task in values
    }
    tasks = [task for values in task_sets.values() for task in values]
    seeds = [int(value) for value in protocol["trial_seeds"]]
    require(len(task_to_set) == 50 and len(tasks) == 50, "manifest must contain 50 unique tasks")
    require(seeds == list(range(50)), "manifest must contain exactly trial seeds 0..49")
    return task_to_set, tasks, seeds


def validate_common_row(
    *,
    row: dict[str, Any],
    config: str,
    task_to_set: dict[str, str],
    expected_seeds: set[int],
    noise_protocol: str,
    server_hashes: set[str],
    source: str,
) -> tuple[str, int]:
    task = str(row.get("task"))
    seed = int(row.get("seed", -1))
    checks = {
        "status": row.get("status") == "complete",
        "config": row.get("config") == config,
        "task": task in task_to_set,
        "task_set": row.get("task_set") == task_to_set.get(task),
        "seed": seed in expected_seeds,
        "split": row.get("split") == "target",
        "n_action_steps": row.get("n_action_steps") == 16,
        "replan_steps": row.get("replan_steps") == 16,
        "flow_steps": row.get("flow_steps") == 4,
        "action_horizon": row.get("action_horizon") == 50,
        "paired_noise": row.get("paired_action_noise") is True,
        "noise_protocol": row.get("action_noise_protocol") == noise_protocol,
        "fresh_environment": row.get("fresh_environment") is True,
        "render": row.get("render_enabled") is True,
        "horizon": task in task_to_set
        and int(row.get("max_steps", -1)) == int(get_task_horizon(task)),
        "server_metadata": row.get("server_metadata_sha256") in server_hashes,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"{source}: protocol checks failed: {failed}")
    return task, seed


def load_selector_rows(
    run_dir: Path, manifest: dict[str, Any]
) -> dict[tuple[str, int], dict[str, Any]]:
    task_to_set, _tasks, seeds = expected_layout(manifest)
    protocol = manifest["protocol"]
    expected_selector = manifest["selector"]
    server_hashes = {row["server_metadata_sha256"] for row in manifest["servers"]}
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    directory = run_dir / "results" / CONFIG
    for path in sorted(directory.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from error
            source = f"{path}:{line_number}"
            key = validate_common_row(
                row=row,
                config=CONFIG,
                task_to_set=task_to_set,
                expected_seeds=set(seeds),
                noise_protocol=protocol["paired_action_noise_protocol"],
                server_hashes=server_hashes,
                source=source,
            )
            selector_checks = {
                "enabled": row.get("runtime_selector_enabled") is True,
                "variant": row.get("selected_variant") == "ohb",
                "selected_config": row.get("selected_config_id") == "gdsq_vla_ohb_only",
                "selector_sha": row.get("selector_sha256") == expected_selector["sha256"],
                "rule": row.get("selector_rule_name") == expected_selector["rule_name"],
                "model": row.get("selector_model_id") == "pi05",
                "task_attestation": row.get("selector_task_name") == key[0],
            }
            failed = [name for name, valid in selector_checks.items() if not valid]
            if failed:
                raise ValueError(f"{source}: selector attestation checks failed: {failed}")
            if key in rows:
                raise ValueError(f"duplicate selector result key: {key}")
            rows[key] = row
    return rows


def verify_reference_chain(manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    reference = manifest["static_reference"]
    for name in ("manifest", "summary", "replay_audit"):
        record = reference[name]
        path = Path(record["path"]).resolve()
        require(path.is_file(), f"missing static reference {name}: {path}")
        require(sha256_file(path) == record["sha256"], f"static reference drift: {name}")
    audit = json.loads(Path(reference["replay_audit"]["path"]).read_text(encoding="utf-8"))
    require(audit.get("valid") is True, "static replay audit is not valid")
    static_manifest_path = Path(reference["manifest"]["path"]).resolve()
    return static_manifest_path.parent, json.loads(static_manifest_path.read_text(encoding="utf-8"))


def load_reference_rows(
    manifest: dict[str, Any],
) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    static_run, static_manifest = verify_reference_chain(manifest)
    task_to_set, tasks, seeds = expected_layout(manifest)
    static_task_sets = static_manifest["table_1_protocol"]["task_sets"]
    require(static_task_sets == manifest["protocol"]["task_sets"], "static task list drift")
    require(
        static_manifest["table_1_protocol"]["trial_seeds"] == seeds,
        "static trial-seed drift",
    )
    expected = set(itertools.product(tasks, seeds))
    grouped: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for config in REFERENCE_CONFIGS:
        allowed_hashes = {
            row["server_metadata_sha256"]
            for row in static_manifest["servers"]
            if row["config_id"] == config
        }
        require(bool(allowed_hashes), f"static manifest has no server for {config}")
        config_rows: dict[tuple[str, int], dict[str, Any]] = {}
        for path in sorted((static_run / "results" / config).glob("*.jsonl")):
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = validate_common_row(
                    row=row,
                    config=config,
                    task_to_set=task_to_set,
                    expected_seeds=set(seeds),
                    noise_protocol=manifest["protocol"]["paired_action_noise_protocol"],
                    server_hashes=allowed_hashes,
                    source=f"{path}:{line_number}",
                )
                if key in config_rows:
                    raise ValueError(f"duplicate static key: {config}/{key}")
                config_rows[key] = row
        require(set(config_rows) == expected, f"static reference coverage mismatch: {config}")
        grouped[config] = config_rows
    return grouped


def gpu_metrics(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "gpu_efficiency.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("config") == CONFIG]
    if not rows:
        return None
    process_memory = [
        float(row["server_process_memory_mib"])
        for row in rows
        if row.get("server_process_memory_mib") is not None
    ]
    return {
        "samples": len(rows),
        "server_instances": sorted({str(row["instance"]) for row in rows}),
        "peak_server_process_memory_mib": max(process_memory) if process_memory else None,
        "mean_server_process_memory_mib": mean(process_memory),
        "peak_device_memory_used_mib": max(float(row["device_memory_used_mib"]) for row in rows),
        "mean_device_utilization_gpu_pct": mean(
            [float(row["device_utilization_gpu_pct"]) for row in rows]
        ),
        "p95_device_utilization_gpu_pct": percentile(
            [float(row["device_utilization_gpu_pct"]) for row in rows], 0.95
        ),
    }


def efficiency(rows: dict[tuple[str, int], dict[str, Any]], gpu: dict | None) -> dict[str, Any]:
    values = list(rows.values())
    replans = sum(int(row["replans"]) for row in values)
    return {
        "mean_episode_wall_seconds": mean([float(row["episode_wall_seconds"]) for row in values]),
        "p50_episode_wall_seconds": percentile(
            [float(row["episode_wall_seconds"]) for row in values], 0.50
        ) if values else None,
        "p95_episode_wall_seconds": percentile(
            [float(row["episode_wall_seconds"]) for row in values], 0.95
        ) if values else None,
        "mean_inference_seconds_per_replan": (
            sum(float(row["inference_seconds"]) for row in values) / replans
            if replans else None
        ),
        "mean_server_infer_ms": mean(
            [
                float(row["server_infer_ms_mean"])
                for row in values
                if row.get("server_infer_ms_mean") is not None
            ]
        ),
        "gpu": gpu,
    }


def summarize_complete_config(
    rows: dict[tuple[str, int], dict[str, Any]],
    tasks: list[str],
    seeds: list[int],
    task_sets: dict[str, list[str]],
    n_boot: int,
    rng: random.Random,
) -> dict[str, Any]:
    rates = {
        task: sum(bool(rows[(task, seed)]["success"]) for seed in seeds) / len(seeds)
        for task in tasks
    }
    heldout = [task for task in tasks if task not in DEVELOPMENT_TASKS]
    return {
        "completed_episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows.values()),
        "episode_sr": sum(bool(row["success"]) for row in rows.values()) / len(rows),
        "per_task_sr": rates,
        "per_task": {
            task: {
                "successes": sum(bool(rows[(task, seed)]["success"]) for seed in seeds),
                "episodes": len(seeds),
                "sr": rates[task],
            }
            for task in tasks
        },
        "task_macro_sr": sum(rates.values()) / len(rates),
        "task_cluster_ci95": cluster_ci(rates, n_boot, rng),
        "heldout46_task_macro_sr": sum(rates[task] for task in heldout) / len(heldout),
        "heldout46_task_cluster_ci95": cluster_ci(
            {task: rates[task] for task in heldout}, n_boot, rng
        ),
        "task_set_macro_sr": {
            name: sum(rates[task] for task in values) / len(values)
            for name, values in task_sets.items()
        },
    }


def aggregate(run_dir: Path, n_boot: int, allow_incomplete: bool) -> dict[str, Any]:
    require(n_boot == 10_000, "formal selector aggregation requires 10,000 bootstrap draws")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("immutable") is True, "selector manifest is not immutable")
    task_to_set, tasks, seeds = expected_layout(manifest)
    task_sets = manifest["protocol"]["task_sets"]
    expected = set(itertools.product(tasks, seeds))
    selector_rows = load_selector_rows(run_dir, manifest)
    missing = sorted(expected - set(selector_rows))
    extra = sorted(set(selector_rows) - expected)
    require(not extra, f"selector output has {len(extra)} unexpected keys")
    if missing and not allow_incomplete:
        raise ValueError(f"selector run is incomplete: {len(missing)} missing episodes")

    result: dict[str, Any] = {
        "schema_version": 1,
        "complete": not missing,
        "formal_result": not missing,
        "manifest_sha256": sha256_file(manifest_path),
        "expected_episodes": len(expected),
        "completed_episodes": len(selector_rows),
        "missing_episodes": len(missing),
        "duplicate_episodes": 0,
        "selector": manifest["selector"],
        "protocol": {
            "primary_metric": "50-task macro success rate",
            "bootstrap": "10,000 task-cluster resamples",
            "test": "paired task-level sign-flip",
            "multiplicity": "Holm over four preregistered selector contrasts",
        },
        "selection_partition": {
            "development_tasks": [task for task in tasks if task in DEVELOPMENT_TASKS],
            "heldout_tasks": [task for task in tasks if task not in DEVELOPMENT_TASKS],
            "primary_scope": "all50_task_macro",
            "heldout46_is_secondary": True,
        },
        "selector_attestation": {
            "validated_rows": len(selector_rows),
            "selected_variant_counts": {
                "ohb": sum(row.get("selected_variant") == "ohb" for row in selector_rows.values())
            },
            "selector_sha256": manifest["selector"]["sha256"],
            "selector_rule_name": manifest["selector"]["rule_name"],
            "uses_task_metadata_for_selection": False,
            "runtime_success_feedback_used": False,
            "atmohb_output_allowed": False,
        },
        "progress_only": None,
        "configs": {},
        "comparisons": {},
        "claim_gate": {
            "superiority_claim_enabled": False,
            "reason": "formal selector coverage incomplete" if missing else "same-budget P0 baselines pending",
        },
    }
    if missing:
        per_set = {name: {"episodes": 0, "successes": 0} for name in task_sets}
        for (task, _seed), row in selector_rows.items():
            bucket = per_set[task_to_set[task]]
            bucket["episodes"] += 1
            bucket["successes"] += int(bool(row["success"]))
        for bucket in per_set.values():
            bucket["observed_sr_not_formal"] = (
                bucket["successes"] / bucket["episodes"] if bucket["episodes"] else None
            )
        result["progress_only"] = {
            "warning": "Incomplete, task-imbalanced progress; not a paper result.",
            "per_task_set": per_set,
        }
        return result

    reference_rows = load_reference_rows(manifest)
    rng = random.Random(RNG_SEED)
    all_rows = {CONFIG: selector_rows, **reference_rows}
    for config, rows in all_rows.items():
        config_summary = summarize_complete_config(rows, tasks, seeds, task_sets, n_boot, rng)
        if config == CONFIG:
            config_summary["efficiency"] = efficiency(rows, gpu_metrics(run_dir))
        result["configs"][config] = config_summary

    raw_p: dict[str, float] = {}
    heldout = [task for task in tasks if task not in DEVELOPMENT_TASKS]
    for a, b in CONTRASTS:
        all_delta, all_ci, all_diffs = paired_delta_ci(
            result["configs"][a]["per_task_sr"],
            result["configs"][b]["per_task_sr"],
            n_boot,
            rng,
        )
        heldout_delta, heldout_ci, _ = paired_delta_ci(
            {task: result["configs"][a]["per_task_sr"][task] for task in heldout},
            {task: result["configs"][b]["per_task_sr"][task] for task in heldout},
            n_boot,
            rng,
        )
        name = f"{a}_vs_{b}"
        p_value = exact_sign_flip_p(all_diffs)
        raw_p[name] = p_value
        result["comparisons"][name] = {
            "a": a,
            "b": b,
            "scope": "all50_primary",
            "all50_task_macro_delta": all_delta,
            "all50_task_cluster_ci95": all_ci,
            "heldout46_task_macro_delta": heldout_delta,
            "heldout46_task_cluster_ci95": heldout_ci,
            "paired_sign_flip_p": p_value,
            "episode_mcnemar": mcnemar(all_rows[a], all_rows[b], tasks, seeds),
        }
    for name, adjusted in holm_adjust(raw_p).items():
        result["comparisons"][name]["holm_adjusted_p"] = adjusted
    return result


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# pi0.5 GDSQ-VLA + v8 selector official evaluation",
        "",
        f"Complete: **{summary['complete']}**; coverage: "
        f"{summary['completed_episodes']}/{summary['expected_episodes']}.",
        "",
    ]
    if not summary["complete"]:
        lines += [
            "> Incomplete progress only. The observed rates below are task-imbalanced and must not be used in the paper.",
            "",
            "| Task set | Episodes | Successes | Observed SR (not formal) |",
            "|---|---:|---:|---:|",
        ]
        for name, row in summary["progress_only"]["per_task_set"].items():
            value = row["observed_sr_not_formal"]
            lines.append(
                f"| {name} | {row['episodes']} | {row['successes']} | "
                + (f"{value:.4f} |" if value is not None else "-- |")
            )
    else:
        order = (CONFIG,) + REFERENCE_CONFIGS
        lines += [
            "| Configuration | Atomic | Composite seen | Composite unseen | All-50 macro SR | 95% task CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for config in order:
            row = summary["configs"][config]
            task_set = row["task_set_macro_sr"]
            ci = row["task_cluster_ci95"]
            lines.append(
                f"| {config} | {task_set['atomic_seen']:.4f} | "
                f"{task_set['composite_seen']:.4f} | {task_set['composite_unseen']:.4f} | "
                f"{row['task_macro_sr']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |"
            )
        lines += [
            "",
            "## Preregistered paired contrasts (all 50 tasks)",
            "",
            "| Contrast | Delta | 95% task CI | Sign-flip p | Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, row in summary["comparisons"].items():
            ci = row["all50_task_cluster_ci95"]
            lines.append(
                f"| {name} | {row['all50_task_macro_delta']:+.4f} | "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {row['paired_sign_flip_p']:.4g} | "
                f"{row['holm_adjusted_p']:.4g} |"
            )
        lines += [
            "",
            "The superiority claim remains disabled until the preregistered same-budget P0 baselines are complete.",
        ]
    lines += [
        "",
        f"Selector SHA: `{summary['selector']['sha256']}`; rule: "
        f"`{summary['selector']['rule_name']}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(run_dir, args.bootstrap, args.allow_incomplete)
    atomic_write(out_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown = out_dir / "summary.md"
    temporary_markdown = markdown.with_name(f".{markdown.name}.tmp")
    write_markdown(temporary_markdown, summary)
    temporary_markdown.replace(markdown)
    print(
        json.dumps(
            {
                "complete": summary["complete"],
                "completed": summary["completed_episodes"],
                "missing": summary["missing_episodes"],
                "selector_rows_attested": summary["selector_attestation"]["validated_rows"],
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
