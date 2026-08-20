#!/usr/bin/env python3
"""Strict Table-1 aggregation for the formal π0.5 RoboCasa365 matrix."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from typing import Any

import robocasa  # noqa: F401
from robocasa.utils.dataset_registry_utils import get_task_horizon

from parse_robocasa_atomic_matrix import (
    cluster_ci,
    exact_sign_flip_p,
    holm_adjust,
    mcnemar,
    paired_delta_ci,
    percentile,
)


CONFIG_ORDER = [
    "fp16",
    "quantvla_w4a8_atmohb",
    "gdsq_vla_atmohb",
    "gdsq_vla",
]
CONTRASTS = [
    ("gdsq_vla_atmohb", "gdsq_vla"),
    ("gdsq_vla", "fp16"),
    ("gdsq_vla", "quantvla_w4a8_atmohb"),
    ("gdsq_vla_atmohb", "quantvla_w4a8_atmohb"),
]
DEVELOPMENT_TASKS = {
    "OpenCabinet",
    "OpenStandMixerHead",
    "PickPlaceDrawerToCounter",
    "CoffeeSetupMug",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def quant_bytes(out_features: int, in_features: int, bits: int = 4, group: int = 64) -> int:
    return int(
        out_features * in_features * bits / 8
        + out_features * 2
        + math.ceil(in_features / group) * group * group * 4
        + math.ceil(out_features / group) * group * group * 4
    )


def load_rows(run_dir: Path, manifest: dict) -> dict[str, dict[tuple[str, int], dict]]:
    task_sets = manifest["table_1_protocol"]["task_sets"]
    task_to_set = {
        task: task_set for task_set, tasks in task_sets.items() for task in tasks
    }
    expected_seeds = set(int(value) for value in manifest["table_1_protocol"]["trial_seeds"])
    noise_protocol = manifest["table_1_protocol"]["paired_action_noise_protocol"]
    metadata_hashes = {
        config: {
            server["server_metadata_sha256"]
            for server in manifest["servers"]
            if server["config_id"] == config
        }
        for config in CONFIG_ORDER
    }
    grouped = {config: {} for config in CONFIG_ORDER}
    for config in CONFIG_ORDER:
        directory = run_dir / "results" / config
        for path in sorted(directory.glob("*.jsonl")):
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: malformed JSON") from error
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
                    "horizon": int(row.get("max_steps", -1)) == int(get_task_horizon(task)),
                    "server_metadata": row.get("server_metadata_sha256") in metadata_hashes[config],
                }
                failed = [name for name, valid in checks.items() if not valid]
                if failed:
                    raise ValueError(f"{path}:{line_number}: protocol checks failed: {failed}")
                key = (task, seed)
                if key in grouped[config]:
                    raise ValueError(f"{config}: duplicate result key {key}")
                grouped[config][key] = row
    return grouped


def paper_memory(run_dir: Path, manifest: dict) -> dict[str, Any]:
    inventory = json.loads(
        Path(manifest["artifacts"]["inventory"]["path"]).read_text(encoding="utf-8")
    )
    gdsq = json.loads(
        Path(manifest["artifacts"]["gdsq_plan"]["path"]).read_text(encoding="utf-8")
    )
    shapes = {
        row["name"]: (int(row["out_features"]), int(row["in_features"]))
        for row in inventory["layers"]
    }
    params = {name: out_features * in_features for name, (out_features, in_features) in shapes.items()}
    fp16_bytes = sum(value * 2 for value in params.values())
    full_w4_bytes = sum(quant_bytes(out, inn) for out, inn in shapes.values())
    gdsq_w4 = {name for name, entry in gdsq["layers"].items() if int(entry["bits"]) == 4}
    w4_params = sum(params[name] for name in gdsq_w4)
    total_params = sum(params.values())
    by_config = {
        "fp16": fp16_bytes,
        "quantvla_w4a8_atmohb": full_w4_bytes,
        "gdsq_vla_atmohb": int(gdsq.get("total_bytes", gdsq["meta"].get("total_bytes"))),
        "gdsq_vla": int(gdsq.get("total_bytes", gdsq["meta"].get("total_bytes"))),
    }
    return {
        "scope": "180 PaliGemma-language + Gemma-action-expert candidate Linear weights",
        "note": "theoretical tightly packed weights/scales/rotation artifacts; not eager CUDA residency",
        "candidate_parameters": total_params,
        "gdsq_w4_parameters": w4_params,
        "gdsq_w4_parameter_fraction": w4_params / total_params,
        "gdsq_w4_layers": len(gdsq_w4),
        "gdsq_fp16_layers": len(shapes) - len(gdsq_w4),
        "by_config": {
            config: {
                "bytes": value,
                "gib": value / 2**30,
                "relative_to_fp16": value / fp16_bytes,
                "compression_vs_fp16": fp16_bytes / value,
            }
            for config, value in by_config.items()
        },
    }


def gpu_metrics(run_dir: Path) -> dict[str, dict[str, Any] | None]:
    path = run_dir / "gpu_efficiency.jsonl"
    grouped = {config: [] for config in CONFIG_ORDER}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("config") in grouped:
                    grouped[row["config"]].append(row)
    result = {}
    for config, rows in grouped.items():
        process_memory = [
            float(row["server_process_memory_mib"])
            for row in rows
            if row.get("server_process_memory_mib") is not None
        ]
        result[config] = (
            {
                "samples": len(rows),
                "server_instances": sorted({row["instance"] for row in rows}),
                "peak_server_process_memory_mib": max(process_memory) if process_memory else None,
                "mean_server_process_memory_mib": mean(process_memory),
                "peak_device_memory_used_mib": max(float(row["device_memory_used_mib"]) for row in rows),
                "mean_device_utilization_gpu_pct": mean(
                    [float(row["device_utilization_gpu_pct"]) for row in rows]
                ),
                "p95_device_utilization_gpu_pct": percentile(
                    [float(row["device_utilization_gpu_pct"]) for row in rows], 0.95
                ),
                "mean_device_power_draw_w": mean(
                    [float(row["device_power_draw_w"]) for row in rows]
                ),
            }
            if rows
            else None
        )
    return result


def summarize_efficiency(rows: dict[tuple[str, int], dict], gpu: dict | None) -> dict:
    values = list(rows.values())
    replans = sum(int(row["replans"]) for row in values)
    steps = sum(int(row["steps"]) for row in values)
    success_steps = [float(row["steps"]) for row in values if row["success"]]
    failure_steps = [float(row["steps"]) for row in values if not row["success"]]
    return {
        "mean_episode_wall_seconds": mean([float(row["episode_wall_seconds"]) for row in values]),
        "p50_episode_wall_seconds": percentile(
            [float(row["episode_wall_seconds"]) for row in values], 0.50
        ) if values else None,
        "p95_episode_wall_seconds": percentile(
            [float(row["episode_wall_seconds"]) for row in values], 0.95
        ) if values else None,
        "mean_env_construct_seconds": mean([float(row["env_construct_seconds"]) for row in values]),
        "mean_inference_seconds_per_replan": (
            sum(float(row["inference_seconds"]) for row in values) / replans if replans else None
        ),
        "mean_server_infer_ms": mean(
            [float(row["server_infer_ms_mean"]) for row in values if row.get("server_infer_ms_mean") is not None]
        ),
        "mean_env_step_seconds": (
            sum(float(row["env_step_seconds"]) for row in values) / steps if steps else None
        ),
        "mean_success_steps": mean(success_steps),
        "mean_failure_steps": mean(failure_steps),
        "gpu": gpu,
    }


def aggregate(run_dir: Path, n_boot: int, allow_incomplete: bool) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_rows(run_dir, manifest)
    task_sets = manifest["table_1_protocol"]["task_sets"]
    tasks = [task for group in task_sets.values() for task in group]
    if not DEVELOPMENT_TASKS <= set(tasks):
        raise ValueError("manifest does not contain the four preregistered development tasks")
    heldout_tasks = [task for task in tasks if task not in DEVELOPMENT_TASKS]
    development_tasks = [task for task in tasks if task in DEVELOPMENT_TASKS]
    if len(heldout_tasks) != 46 or len(development_tasks) != 4:
        raise ValueError("expected a 46-task held-out / 4-task development partition")
    seeds = [int(value) for value in manifest["table_1_protocol"]["trial_seeds"]]
    expected = set(itertools.product(tasks, seeds))
    missing = {config: sorted(expected - set(config_rows)) for config, config_rows in rows.items()}
    if not allow_incomplete and any(missing.values()):
        raise ValueError(
            "formal matrix is incomplete: "
            + ", ".join(f"{config}={len(values)} missing" for config, values in missing.items())
        )
    gpu = gpu_metrics(run_dir)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "complete": not any(missing.values()),
        "expected_episodes_per_config": len(expected),
        "manifest_sha256": sha256_file(manifest_path),
        "paper_style_memory": paper_memory(run_dir, manifest),
        "configs": {},
        "comparisons": {},
        "selection_partition": {
            "development_tasks": development_tasks,
            "heldout_tasks": heldout_tasks,
            "primary_scope": "46 held-out tasks not used for ratio selection",
            "secondary_scope": "all 50 tasks",
        },
        "libero_context": {
            "v1.4_macro_avg": 0.892,
            "uniform_w6_macro_avg": 0.882,
            "v1.3_macro_avg": 0.852,
            "note": "cross-benchmark context only; never pooled with RoboCasa365",
        },
    }
    rng = random.Random(20260819)
    for config in CONFIG_ORDER:
        config_rows = rows[config]
        config_summary: dict[str, Any] = {
            "completed_episodes": len(config_rows),
            "missing_episodes": len(missing[config]),
            "observed_successes": sum(bool(row["success"]) for row in config_rows.values()),
            "observed_episode_sr": (
                sum(bool(row["success"]) for row in config_rows.values()) / len(config_rows)
                if config_rows else None
            ),
            "efficiency": summarize_efficiency(config_rows, gpu[config]),
        }
        if not missing[config]:
            per_task = {
                task: sum(bool(config_rows[(task, seed)]["success"]) for seed in seeds) / len(seeds)
                for task in tasks
            }
            config_summary.update(
                {
                    "per_task_sr": per_task,
                    "per_task_details": {
                        task: {
                            "successes": sum(
                                bool(config_rows[(task, seed)]["success"]) for seed in seeds
                            ),
                            "episodes": len(seeds),
                            "sr": per_task[task],
                            "mean_success_steps": mean(
                                [
                                    float(config_rows[(task, seed)]["steps"])
                                    for seed in seeds
                                    if config_rows[(task, seed)]["success"]
                                ]
                            ),
                            "mean_failure_steps": mean(
                                [
                                    float(config_rows[(task, seed)]["steps"])
                                    for seed in seeds
                                    if not config_rows[(task, seed)]["success"]
                                ]
                            ),
                        }
                        for task in tasks
                    },
                    "task_macro_sr": mean(list(per_task.values())),
                    "task_cluster_ci95": cluster_ci(per_task, n_boot, rng),
                    "heldout46_task_macro_sr": mean([per_task[task] for task in heldout_tasks]),
                    "heldout46_task_cluster_ci95": cluster_ci(
                        {task: per_task[task] for task in heldout_tasks}, n_boot, rng
                    ),
                    "development4_task_macro_sr": mean(
                        [per_task[task] for task in development_tasks]
                    ),
                    "episode_sr": sum(bool(row["success"]) for row in config_rows.values()) / len(config_rows),
                    "task_set_macro_sr": {
                        task_set: mean([per_task[task] for task in group_tasks])
                        for task_set, group_tasks in task_sets.items()
                    },
                }
            )
        summary["configs"][config] = config_summary

    if not summary["complete"]:
        task_to_set = {
            task: task_set for task_set, group_tasks in task_sets.items() for task in group_tasks
        }
        per_set: dict[str, Any] = {}
        for config in CONFIG_ORDER:
            counts = {task_set: [0, 0] for task_set in task_sets}
            for (task, _seed), row in rows[config].items():
                bucket = counts[task_to_set[task]]
                bucket[0] += 1
                bucket[1] += int(bool(row["success"]))
            per_set[config] = {
                task_set: {
                    "episodes": episodes,
                    "successes": successes,
                    "sr": successes / episodes if episodes else None,
                }
                for task_set, (episodes, successes) in counts.items()
            }
        common_keys = set.intersection(*(set(rows[config]) for config in CONFIG_ORDER))
        summary["progress_diagnostics"] = {
            "note": (
                "Incomplete matrix: configs advance through task sets at different rates, so "
                "pooled Episode SR mixes unequal task difficulty and must never be compared "
                "across configs. Use the stratified and paired-common-key views for progress "
                "monitoring only; neither is a formal result."
            ),
            "per_task_set_observed": per_set,
            "paired_common_keys": {
                "n_common_keys": len(common_keys),
                "per_task_set_n": {
                    task_set: sum(
                        1 for task, _seed in common_keys if task_to_set[task] == task_set
                    )
                    for task_set in task_sets
                },
                "sr": {
                    config: (
                        sum(bool(rows[config][key]["success"]) for key in common_keys)
                        / len(common_keys)
                        if common_keys
                        else None
                    )
                    for config in CONFIG_ORDER
                },
            },
        }

    if summary["complete"]:
        raw_p = {}
        for a, b in CONTRASTS:
            delta, ci, diffs = paired_delta_ci(
                {
                    task: summary["configs"][a]["per_task_sr"][task]
                    for task in heldout_tasks
                },
                {
                    task: summary["configs"][b]["per_task_sr"][task]
                    for task in heldout_tasks
                },
                n_boot,
                rng,
            )
            name = f"{a}_vs_{b}"
            p_value = exact_sign_flip_p(diffs)
            raw_p[name] = p_value
            summary["comparisons"][name] = {
                "a": a,
                "b": b,
                "scope": "heldout46_primary",
                "heldout46_task_macro_delta": delta,
                "heldout46_task_cluster_ci95": ci,
                "paired_permutation_p": p_value,
                "episode_mcnemar": mcnemar(rows[a], rows[b], heldout_tasks, seeds),
                "all50_task_macro_delta": (
                    summary["configs"][a]["task_macro_sr"]
                    - summary["configs"][b]["task_macro_sr"]
                ),
            }
        for name, adjusted in holm_adjust(raw_p).items():
            summary["comparisons"][name]["holm_adjusted_p"] = adjusted
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# π0.5 RoboCasa365 Table 1",
        "",
        f"Matrix complete: **{summary['complete']}**. Expected episodes/config: "
        f"{summary['expected_episodes_per_config']}.",
        "",
        "| Config | Complete | Successes | Episode SR | Held-out 46 macro SR | All 50 macro SR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIG_ORDER:
        row = summary["configs"][config]
        macro = row.get("task_macro_sr")
        lines.append(
            f"| {config} | {row['completed_episodes']}/{summary['expected_episodes_per_config']} | "
            f"{row['observed_successes']} | {row['observed_episode_sr'] or 0:.3f} | "
            f"{row['heldout46_task_macro_sr']:.3f} | {macro:.3f} |" if macro is not None else
            f"| {config} | {row['completed_episodes']}/{summary['expected_episodes_per_config']} | "
            f"{row['observed_successes']} | {row['observed_episode_sr'] or 0:.3f} | pending | pending |"
        )
    if not summary["complete"] and "progress_diagnostics" in summary:
        diagnostics = summary["progress_diagnostics"]
        lines += ["", "## Progress diagnostics (incomplete matrix, not comparable results)", "", diagnostics["note"], ""]
        observed = diagnostics["per_task_set_observed"]
        task_set_names = list(next(iter(observed.values())))
        lines += [
            "| Config | " + " | ".join(task_set_names) + " |",
            "|---|" + "---:|" * len(task_set_names),
        ]
        for config in CONFIG_ORDER:
            cells = []
            for task_set in task_set_names:
                cell = observed[config][task_set]
                if cell["episodes"]:
                    cells.append(f"{cell['successes']}/{cell['episodes']} = {cell['sr']:.3f}")
                else:
                    cells.append("-")
            lines.append("| " + config + " | " + " | ".join(cells) + " |")
        paired = diagnostics["paired_common_keys"]
        composition = ", ".join(
            f"{task_set}={count}" for task_set, count in paired["per_task_set_n"].items() if count
        )
        lines += [
            "",
            f"Paired SR on the {paired['n_common_keys']} (task, seed) keys completed by all "
            f"four configs ({composition or 'no common keys'}):",
            "",
            "| Config | Paired common-key SR |",
            "|---|---:|",
        ]
        for config in CONFIG_ORDER:
            value = paired["sr"][config]
            lines.append(f"| {config} | {value:.3f} |" if value is not None else f"| {config} | - |")
    if summary["complete"]:
        task_sets = next(iter(summary["configs"].values()))["task_set_macro_sr"]
        lines += ["", "## Task-set macro SR", "", "| Task set | " + " | ".join(CONFIG_ORDER) + " |", "|---|" + "---:|" * len(CONFIG_ORDER)]
        for task_set in task_sets:
            values = [summary["configs"][config]["task_set_macro_sr"][task_set] for config in CONFIG_ORDER]
            lines.append("| " + task_set + " | " + " | ".join(f"{value:.3f}" for value in values) + " |")
        lines += ["", "## Prespecified paired comparisons", "", "| Comparison | Delta | 95% CI | Permutation p | Holm p |", "|---|---:|---:|---:|---:|"]
        for name, row in summary["comparisons"].items():
            ci = row["heldout46_task_cluster_ci95"]
            lines.append(f"| {name} | {row['heldout46_task_macro_delta']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {row['paired_permutation_p']:.4g} | {row['holm_adjusted_p']:.4g} |")
    lines += ["", "## Efficiency", "", "| Config | Episode wall (s) | Inference/replan (s) | Server infer (ms) | Peak server CUDA MiB | Mean GPU util |", "|---|---:|---:|---:|---:|---:|"]
    for config in CONFIG_ORDER:
        efficiency = summary["configs"][config]["efficiency"]
        gpu = efficiency.get("gpu") or {}
        lines.append(
            f"| {config} | {efficiency.get('mean_episode_wall_seconds') or float('nan'):.1f} | "
            f"{efficiency.get('mean_inference_seconds_per_replan') or float('nan'):.3f} | "
            f"{efficiency.get('mean_server_infer_ms') or float('nan'):.1f} | "
            f"{gpu.get('peak_server_process_memory_mib') or float('nan'):.0f} | "
            f"{gpu.get('mean_device_utilization_gpu_pct') or float('nan'):.1f}% |"
        )
    memory = summary["paper_style_memory"]
    lines += ["", "## Paper-style candidate-component memory", "", "| Config | GiB | Compression vs FP16 |", "|---|---:|---:|"]
    for config in CONFIG_ORDER:
        row = memory["by_config"][config]
        lines.append(f"| {config} | {row['gib']:.3f} | {row['compression_vs_fp16']:.2f}× |")
    lines += [
        "",
        f"GDSQ quantizes {memory['gdsq_w4_parameters']:,}/{memory['candidate_parameters']:,} "
        f"candidate parameters ({100 * memory['gdsq_w4_parameter_fraction']:.1f}%) across "
        f"{memory['gdsq_w4_layers']} W4 + {memory['gdsq_fp16_layers']} FP16 layers.",
        "",
        "Paper-style memory is tightly packed theoretical component storage; eager fake-quant CUDA "
        "residency is reported separately and is not presented as deployment compression.",
        "",
        "LIBERO context only: v1.4 89.2%, uniform W6 88.2%, v1.3 85.2%; not pooled with RoboCasa365.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(run_dir, args.bootstrap, args.allow_incomplete)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(out_dir / "summary.md", summary)
    print(
        json.dumps(
            {
                "complete": summary["complete"],
                "configs": {
                    config: {
                        "completed": row["completed_episodes"],
                        "successes": row["observed_successes"],
                        "observed_sr": row["observed_episode_sr"],
                    }
                    for config, row in summary["configs"].items()
                },
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
