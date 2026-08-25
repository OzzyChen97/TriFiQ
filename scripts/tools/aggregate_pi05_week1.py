#!/usr/bin/env python3
"""Strict aggregation for a frozen pi0.5 week-1 control run.

Incomplete rows are summarized only as coverage progress.  A formal summary is
emitted only after exact manifest coverage and every runtime/protocol check
passes; paired reference rows are accepted only through frozen manifest hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any

import robocasa  # noqa: F401 -- register task horizons before validation
from robocasa.utils.dataset_registry_utils import get_task_horizon

from parse_robocasa_atomic_matrix import (
    cluster_ci,
    exact_sign_flip_p,
    holm_adjust,
    mcnemar,
    paired_delta_ci,
    percentile,
)


RNG_SEED = 20260823


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def expected_layout(
    manifest: dict[str, Any],
) -> tuple[dict[str, str], list[str], list[int]]:
    protocol = manifest["protocol"]
    task_sets = protocol["task_sets"]
    tasks = [task for values in task_sets.values() for task in values]
    task_to_set = {
        task: task_set for task_set, values in task_sets.items() for task in values
    }
    seeds = [int(seed) for seed in protocol["trial_seeds"]]
    require(len(tasks) == len(task_to_set), "duplicate task in manifest")
    require(len(tasks) in (4, 46, 50), "unexpected week-1 task scope")
    require(seeds == list(range(50)), "formal seeds must be exactly 0..49")
    require(
        manifest["schedule"]["coverage"]["expected_episode_keys"]
        == len(tasks) * len(seeds),
        "manifest schedule count drift",
    )
    return task_to_set, tasks, seeds


def validate_common_row(
    *,
    row: dict[str, Any],
    config: str,
    task_to_set: dict[str, str],
    seeds: set[int],
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
        "seed": seed in seeds,
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
    require(not failed, f"{source}: protocol checks failed: {failed}")
    return task, seed


def read_config_rows(
    *,
    run_dir: Path,
    config: str,
    task_to_set: dict[str, str],
    seeds: list[int],
    noise_protocol: str,
    server_hashes: set[str],
    current_control: bool,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    files: list[dict[str, Any]] = []
    directory = run_dir / "results" / config
    for path in sorted(directory.glob("*.jsonl")):
        if not path.stat().st_size:
            continue
        files.append({"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size})
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from error
            task, seed = str(row.get("task")), int(row.get("seed", -1))
            # A full-50 reference may be narrowed to dev4 or heldout46.
            if task not in task_to_set or seed not in set(seeds):
                continue
            key = validate_common_row(
                row=row,
                config=config,
                task_to_set=task_to_set,
                seeds=set(seeds),
                noise_protocol=noise_protocol,
                server_hashes=server_hashes,
                source=f"{path}:{line_number}",
            )
            if current_control:
                checks = {
                    "selector_disabled": row.get("runtime_selector_enabled") is False,
                    "no_selected_variant": row.get("selected_variant") is None,
                    "no_selected_config": row.get("selected_config_id") is None,
                }
                failed = [name for name, valid in checks.items() if not valid]
                require(not failed, f"{path}:{line_number}: correction checks failed: {failed}")
            require(key not in rows, f"duplicate result key {config}/{key}")
            rows[key] = row
    return rows, files


def reference_protocol(manifest: dict[str, Any]) -> dict[str, Any]:
    if "protocol" in manifest:
        return manifest["protocol"]
    require("table_1_protocol" in manifest, "unknown reference manifest schema")
    return manifest["table_1_protocol"]


def reference_server_hashes(manifest: dict[str, Any], config: str) -> set[str]:
    return {
        str(row["server_metadata_sha256"])
        for row in manifest.get("servers", [])
        if row.get("config_id") == config and row.get("server_metadata_sha256")
    }


def load_references(
    manifest: dict[str, Any],
    task_to_set: dict[str, str],
    seeds: list[int],
) -> tuple[
    dict[str, dict[tuple[str, int], dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    expected_tasks = set(task_to_set)
    grouped: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    files: dict[str, list[dict[str, Any]]] = {}
    for reference in manifest.get("references", []):
        name = reference["name"]
        manifest_path = Path(reference["manifest"]["path"]).resolve()
        require(manifest_path.is_file(), f"missing reference manifest: {manifest_path}")
        require(
            sha256_file(manifest_path) == reference["manifest"]["sha256"],
            f"reference manifest drift: {name}",
        )
        reference_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocol = reference_protocol(reference_manifest)
        reference_tasks = {
            task for values in protocol["task_sets"].values() for task in values
        }
        reference_seeds = [int(seed) for seed in protocol["trial_seeds"]]
        reference_noise = protocol.get("paired_action_noise_protocol")
        require(expected_tasks <= reference_tasks, f"reference task scope mismatch: {name}")
        require(reference_seeds == seeds, f"reference seed mismatch: {name}")
        require(
            reference_noise == manifest["protocol"]["paired_action_noise_protocol"],
            f"reference noise mismatch: {name}",
        )
        config = reference["config_id"]
        hashes = reference_server_hashes(reference_manifest, config)
        require(hashes, f"reference server hashes missing: {name}")
        rows, result_files = read_config_rows(
            run_dir=Path(reference["run_dir"]).resolve(),
            config=config,
            task_to_set=task_to_set,
            seeds=seeds,
            noise_protocol=manifest["protocol"]["paired_action_noise_protocol"],
            server_hashes=hashes,
            current_control=False,
        )
        expected = set(itertools.product(task_to_set, seeds))
        require(set(rows) == expected, f"reference coverage mismatch: {name}")
        require(name not in grouped, f"duplicate reference name: {name}")
        grouped[name] = rows
        files[name] = result_files
    return grouped, files


def summarize_config(
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
    values = list(rows.values())
    replans = sum(int(row.get("replans", 0)) for row in values)
    return {
        "completed_episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in values),
        "episode_sr": sum(bool(row["success"]) for row in values) / len(values),
        "task_macro_sr": sum(rates.values()) / len(rates),
        "task_cluster_ci95": cluster_ci(rates, n_boot, rng),
        "per_task_sr": rates,
        "per_task": {
            task: {
                "successes": sum(bool(rows[(task, seed)]["success"]) for seed in seeds),
                "episodes": len(seeds),
                "sr": rates[task],
            }
            for task in tasks
        },
        "task_set_macro_sr": {
            name: sum(rates[task] for task in members) / len(members)
            for name, members in task_sets.items()
        },
        "efficiency": {
            "mean_episode_wall_seconds": mean(
                [float(row["episode_wall_seconds"]) for row in values]
            ),
            "p50_episode_wall_seconds": percentile(
                [float(row["episode_wall_seconds"]) for row in values], 0.50
            ),
            "p95_episode_wall_seconds": percentile(
                [float(row["episode_wall_seconds"]) for row in values], 0.95
            ),
            "mean_inference_seconds_per_replan": (
                sum(float(row["inference_seconds"]) for row in values) / replans
                if replans
                else None
            ),
        },
    }


def aggregate(run_dir: Path, n_boot: int, allow_incomplete: bool) -> dict[str, Any]:
    require(n_boot == 10_000, "formal aggregation requires 10,000 bootstrap draws")
    manifest_path = run_dir / "manifest.json"
    require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("immutable") is True, "manifest is not immutable")
    config = manifest["config_id"]
    task_to_set, tasks, seeds = expected_layout(manifest)
    server_hashes = {row["server_metadata_sha256"] for row in manifest["servers"]}
    rows, result_files = read_config_rows(
        run_dir=run_dir,
        config=config,
        task_to_set=task_to_set,
        seeds=seeds,
        noise_protocol=manifest["protocol"]["paired_action_noise_protocol"],
        server_hashes=server_hashes,
        current_control=True,
    )
    expected = set(itertools.product(tasks, seeds))
    missing, extra = sorted(expected - set(rows)), sorted(set(rows) - expected)
    require(not extra, f"control output has {len(extra)} unexpected keys")
    if missing and not allow_incomplete:
        raise ValueError(f"control run is incomplete: {len(missing)} missing episodes")
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pi05_gdsq_vla_week1_control_summary",
        "complete": not missing,
        "formal_result": not missing,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "config_id": config,
        "scope": manifest["protocol"]["scope"],
        "expected_episodes": len(expected),
        "completed_episodes": len(rows),
        "missing_episodes": len(missing),
        "duplicate_episodes": 0,
        "quantization": manifest["quantization"],
        "protocol": manifest["protocol"],
        "result_files": result_files,
        "configs": {},
        "comparisons": {},
        "progress_only": None,
        "claim_gate": {
            "paper_result_enabled": not missing,
            "superiority_claim_enabled": False,
            "reason": "formal coverage incomplete" if missing else "evaluated by paired contrasts below",
        },
    }
    if missing:
        counts = {name: {"episodes": 0, "successes": 0} for name in manifest["protocol"]["task_sets"]}
        for (task, _seed), row in rows.items():
            bucket = counts[task_to_set[task]]
            bucket["episodes"] += 1
            bucket["successes"] += int(bool(row["success"]))
        result["progress_only"] = {
            "warning": "Incomplete and task-imbalanced; no success rate is a paper result.",
            "per_task_set_counts": counts,
        }
        return result

    reference_rows, reference_files = load_references(manifest, task_to_set, seeds)
    rng = random.Random(RNG_SEED)
    result["configs"][config] = summarize_config(
        rows, tasks, seeds, manifest["protocol"]["task_sets"], n_boot, rng
    )
    raw_p: dict[str, float] = {}
    for name, paired_rows in reference_rows.items():
        reference_config = next(
            row["config_id"] for row in manifest["references"] if row["name"] == name
        )
        result["configs"][name] = summarize_config(
            paired_rows, tasks, seeds, manifest["protocol"]["task_sets"], n_boot, rng
        )
        delta, ci, diffs = paired_delta_ci(
            result["configs"][config]["per_task_sr"],
            result["configs"][name]["per_task_sr"],
            n_boot,
            rng,
        )
        contrast = f"{config}_vs_{name}"
        p_value = exact_sign_flip_p(diffs)
        raw_p[contrast] = p_value
        result["comparisons"][contrast] = {
            "a": config,
            "b": name,
            "b_config_id": reference_config,
            "scope": manifest["protocol"]["scope"],
            "task_macro_delta_a_minus_b": delta,
            "task_cluster_ci95": ci,
            "paired_sign_flip_p": p_value,
            "episode_mcnemar": mcnemar(rows, paired_rows, tasks, seeds),
            "reference_result_files": reference_files[name],
            "a_superior_ci_excludes_zero": ci[0] > 0,
            "b_superior_ci_excludes_zero": ci[1] < 0,
        }
    adjusted = holm_adjust(raw_p)
    for name, value in adjusted.items():
        result["comparisons"][name]["holm_adjusted_p"] = value
    result["claim_gate"] = {
        "paper_result_enabled": True,
        "superiority_claim_enabled": any(
            row["a_superior_ci_excludes_zero"] and row["holm_adjusted_p"] <= 0.05
            for row in result["comparisons"].values()
        ),
        "rule": "task-cluster CI lower > 0 and Holm-adjusted paired sign-flip p <= 0.05",
        "note": "Direction is current control minus frozen reference; final ours-minus-baseline claims are generated jointly.",
    }
    return result


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# pi0.5 week-1 control: {summary['config_id']}",
        "",
        f"Complete: **{summary['complete']}**; coverage: {summary['completed_episodes']}/{summary['expected_episodes']}.",
        "",
    ]
    if not summary["complete"]:
        lines += [
            "> Incomplete progress only. No observed success rate may enter the paper.",
            "",
            "| Task set | Episodes | Successes |",
            "|---|---:|---:|",
        ]
        for name, row in summary["progress_only"]["per_task_set_counts"].items():
            lines.append(f"| {name} | {row['episodes']} | {row['successes']} |")
    else:
        lines += [
            "| Configuration | Task-macro SR | 95% task-cluster CI |",
            "|---|---:|---:|",
        ]
        for name, row in summary["configs"].items():
            ci = row["task_cluster_ci95"]
            lines.append(f"| {name} | {row['task_macro_sr']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        if summary["comparisons"]:
            lines += [
                "",
                "| Paired contrast (control minus reference) | Delta | 95% task CI | Sign-flip p | Holm p |",
                "|---|---:|---:|---:|---:|",
            ]
            for name, row in summary["comparisons"].items():
                ci = row["task_cluster_ci95"]
                lines.append(
                    f"| {name} | {row['task_macro_delta_a_minus_b']:+.4f} | "
                    f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {row['paired_sign_flip_p']:.4g} | "
                    f"{row['holm_adjusted_p']:.4g} |"
                )
    lines += ["", f"Manifest SHA: `{summary['manifest_sha256']}`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(Path(args.run_dir).resolve(), args.bootstrap, args.allow_incomplete)
    atomic_write(out_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown = out_dir / "summary.md"
    temporary = markdown.with_name(f".{markdown.name}.tmp")
    write_markdown(temporary, summary)
    temporary.replace(markdown)
    print(
        json.dumps(
            {
                "complete": summary["complete"],
                "completed": summary["completed_episodes"],
                "missing": summary["missing_episodes"],
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
