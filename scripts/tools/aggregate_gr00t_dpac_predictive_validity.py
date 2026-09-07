#!/usr/bin/env python3
"""Gate, analyze, and report the preregistered GR00T predictive-validity run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import kendalltau, spearmanr

from quantvla_predictive_validity import (
    PROTOCOL,
    artifact,
    atomic_json,
    protocol_attestation,
    require_protocol_attestation,
    sha256_file,
)


SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")
METRICS = ("mse", "d_func", "d_pac")


def _read_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def _read_jsonl_files(root: str | Path) -> list[tuple[Path, int, dict[str, Any]]]:
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    rows = []
    for path in sorted(resolved.rglob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                rows.append((path, line_number, json.loads(line)))
    return rows


def _finite(value: Any, *, source: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{source}: non-finite value")
    return result


def correlation(x: Iterable[float], y: Iterable[float]) -> dict[str, Any]:
    lhs = np.asarray(list(x), dtype=np.float64)
    rhs = np.asarray(list(y), dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim != 1 or lhs.size < 2:
        return {"value": None, "status": "undefined: insufficient paired observations"}
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        return {"value": None, "status": "undefined: non-finite input"}
    if np.all(lhs == lhs[0]):
        return {"value": None, "status": "undefined: metric is constant"}
    if np.all(rhs == rhs[0]):
        return {"value": None, "status": "undefined: success drop is constant"}
    return {
        "value": float(spearmanr(lhs, rhs).statistic),
        "status": "defined",
    }


def kendall(x: Iterable[float], y: Iterable[float]) -> dict[str, Any]:
    lhs = np.asarray(list(x), dtype=np.float64)
    rhs = np.asarray(list(y), dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim != 1 or lhs.size < 2:
        return {"value": None, "status": "undefined: insufficient paired observations"}
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        return {"value": None, "status": "undefined: non-finite input"}
    if np.all(lhs == lhs[0]):
        return {"value": None, "status": "undefined: metric is constant"}
    if np.all(rhs == rhs[0]):
        return {"value": None, "status": "undefined: success drop is constant"}
    value = kendalltau(lhs, rhs, variant="b").statistic
    if not np.isfinite(value):
        return {"value": None, "status": "undefined: Kendall tau-b returned non-finite"}
    return {"value": float(value), "status": "defined"}


def correlations(rows: list[dict[str, Any]], metric_key: str) -> dict[str, Any]:
    x = [_finite(row[metric_key], source=metric_key) for row in rows]
    y = [_finite(row["success_drop"], source="success_drop") for row in rows]
    return {"spearman_rho": correlation(x, y), "kendall_tau_b": kendall(x, y)}


def delta(lhs: dict[str, Any], rhs: dict[str, Any], key: str) -> dict[str, Any]:
    left = lhs[key]["value"]
    right = rhs[key]["value"]
    if left is None or right is None:
        return {"value": None, "status": "undefined: component correlation is undefined"}
    return {"value": float(left - right), "status": "defined"}


def validate_inputs(
    library_path: Path,
    library: dict[str, Any],
    score_path: Path,
    scores: dict[str, Any],
    mask_root: str | Path,
    fp16_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    require_protocol_attestation(library, source=str(library_path))
    require_protocol_attestation(scores, source=str(score_path))
    if library.get("kind") != "gr00t_dpac_predictive_mask_library":
        raise ValueError("wrong predictive mask-library kind")
    if scores.get("kind") != "gr00t_dpac_predictive_offline_scores":
        raise ValueError("wrong predictive offline-score kind")
    if (scores.get("mask_library") or {}).get("sha256") != sha256_file(library_path):
        raise ValueError("offline score/library hash drift")
    candidates = {str(row["candidate_id"]): row for row in library["candidates"]}
    expected_tasks = {
        task: split
        for split, tasks in library["closed_loop"]["tasks"].items()
        for task in tasks
    }
    if len(candidates) != 60 or len(expected_tasks) != 50:
        raise ValueError("preregistered candidate/task inventory drift")
    if set(scores.get("scores") or {}) != set(candidates):
        raise ValueError("offline score inventory drift")
    expected_seed = int(library["closed_loop"]["environment_seed"])
    library_hash = sha256_file(library_path)

    mask_rows: dict[tuple[str, str], dict[str, Any]] = {}
    mask_files: set[Path] = set()
    for path, line_number, row in _read_jsonl_files(mask_root):
        source = f"{path}:{line_number}"
        if row.get("status") != "complete":
            raise ValueError(f"{source}: incomplete mask row")
        identifier, task = str(row.get("predictive_mask_id")), str(row.get("task"))
        if identifier not in candidates or task not in expected_tasks:
            raise ValueError(f"{source}: non-preregistered mask/task")
        if int(row.get("seed", -1)) != expected_seed:
            raise ValueError(f"{source}: seed drift")
        if row.get("predictive_library_sha256") != library_hash:
            raise ValueError(f"{source}: mask-library hash drift")
        if row.get("predictive_plan_sha256") != candidates[identifier]["sha256"]:
            raise ValueError(f"{source}: mask-plan hash drift")
        if row.get("predictive_validity_protocol") != protocol_attestation():
            raise ValueError(f"{source}: predictive protocol hash drift")
        if row.get("predictive_evaluation_only") is not True:
            raise ValueError(f"{source}: evaluation-only attestation missing")
        key = (identifier, task)
        if key in mask_rows:
            raise ValueError(f"duplicate mask row: {key}")
        mask_rows[key] = row
        mask_files.add(path.resolve())
    expected_mask_keys = {
        (identifier, task) for identifier in candidates for task in expected_tasks
    }
    if set(mask_rows) != expected_mask_keys:
        missing = sorted(expected_mask_keys - set(mask_rows))
        extra = sorted(set(mask_rows) - expected_mask_keys)
        raise ValueError(
            f"closed-loop mask coverage is not exactly 3000; "
            f"observed={len(mask_rows)} missing={missing[:3]} extra={extra[:3]}"
        )

    fp16_rows: dict[str, dict[str, Any]] = {}
    fp16_files: set[Path] = set()
    for path, line_number, row in _read_jsonl_files(fp16_root):
        source = f"{path}:{line_number}"
        if row.get("status") != "complete":
            raise ValueError(f"{source}: incomplete FP16 row")
        task = str(row.get("task"))
        if task not in expected_tasks or int(row.get("seed", -1)) != expected_seed:
            raise ValueError(f"{source}: non-preregistered FP16 task/seed")
        if row.get("predictive_mask_id") is not None:
            raise ValueError(f"{source}: FP16 row unexpectedly carries a mask")
        if task in fp16_rows:
            raise ValueError(f"duplicate FP16 row: {task}")
        fp16_rows[task] = row
        fp16_files.add(path.resolve())
    if set(fp16_rows) != set(expected_tasks):
        missing = sorted(set(expected_tasks) - set(fp16_rows))
        raise ValueError(
            f"native FP16 coverage is not exactly 50; observed={len(fp16_rows)} "
            f"missing={missing[:3]}"
        )

    mask_summaries = []
    for identifier, candidate in sorted(candidates.items()):
        closed = [mask_rows[(identifier, task)] for task in expected_tasks]
        base = [fp16_rows[task] for task in expected_tasks]
        mask_sr = float(np.mean([bool(row["success"]) for row in closed]))
        fp16_sr = float(np.mean([bool(row["success"]) for row in base]))
        score = scores["scores"][identifier]
        item = {
            "predictive_mask_id": identifier,
            "swap_count": int(candidate["swap_count"]),
            "hamming_distance": int(candidate["hamming_distance"]),
            "mse": _finite(score["mse"], source=f"{identifier}/mse"),
            "d_func": _finite(score["d_func"], source=f"{identifier}/d_func"),
            "d_pac": _finite(score["d_pac"], source=f"{identifier}/d_pac"),
            "fp16_success_rate": fp16_sr,
            "mask_success_rate": mask_sr,
            "success_drop": fp16_sr - mask_sr,
            "splits": {},
        }
        for split in SPLITS:
            tasks = list(library["closed_loop"]["tasks"][split])
            split_mask_sr = float(
                np.mean([bool(mask_rows[(identifier, task)]["success"]) for task in tasks])
            )
            split_fp16_sr = float(np.mean([bool(fp16_rows[task]["success"]) for task in tasks]))
            split_metric = score["split_scores"][split]
            item["splits"][split] = {
                "mse": _finite(split_metric["mse"], source=f"{identifier}/{split}/mse"),
                "d_func": _finite(split_metric["d_func"], source=f"{identifier}/{split}/d_func"),
                "d_pac": _finite(split_metric["d_pac"], source=f"{identifier}/{split}/d_pac"),
                "fp16_success_rate": split_fp16_sr,
                "mask_success_rate": split_mask_sr,
                "success_drop": split_fp16_sr - split_mask_sr,
            }
        mask_summaries.append(item)
    provenance = {
        "mask_library": artifact(library_path),
        "offline_scores": artifact(score_path),
        "mask_result_files": [artifact(path) for path in sorted(mask_files)],
        "fp16_result_files": [artifact(path) for path in sorted(fp16_files)],
    }
    return mask_summaries, list(fp16_rows.values()), provenance


def stratified_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    iterations = int(PROTOCOL["statistics"]["stratified_bootstrap_draws"])
    rng = np.random.default_rng(
        int(PROTOCOL["statistics"]["stratified_bootstrap_seed"])
    )
    strata = {
        value: [row for row in rows if int(row["swap_count"]) == value]
        for value in sorted({int(row["swap_count"]) for row in rows})
    }
    counts = {
        method: {baseline: {kind: 0 for kind in ("spearman_rho", "kendall_tau_b")}
                 for baseline in ("mse", "d_func")}
        for method in ("d_pac",)
    }
    defined = {
        method: {baseline: {kind: 0 for kind in ("spearman_rho", "kendall_tau_b")}
                 for baseline in ("mse", "d_func")}
        for method in ("d_pac",)
    }
    for _ in range(iterations):
        sample = []
        for stratum in strata.values():
            indices = rng.integers(0, len(stratum), size=len(stratum))
            sample.extend(stratum[int(index)] for index in indices)
        values = {metric: correlations(sample, metric) for metric in METRICS}
        for baseline in ("mse", "d_func"):
            for kind in ("spearman_rho", "kendall_tau_b"):
                lhs = values["d_pac"][kind]["value"]
                rhs = values[baseline][kind]["value"]
                if lhs is not None and rhs is not None:
                    defined["d_pac"][baseline][kind] += 1
                    if lhs > rhs:
                        counts["d_pac"][baseline][kind] += 1
    comparisons = {}
    for baseline in ("mse", "d_func"):
        comparisons[baseline] = {}
        for kind in ("spearman_rho", "kendall_tau_b"):
            denominator = defined["d_pac"][baseline][kind]
            comparisons[baseline][kind] = {
                "proportion_dpac_greater": (
                    counts["d_pac"][baseline][kind] / denominator if denominator else None
                ),
                "defined_resamples": denominator,
                "undefined_resamples": iterations - denominator,
            }
    return {
        "method": "within-swap-count stratified nonparametric resampling",
        "iterations": iterations,
        "seed": int(PROTOCOL["statistics"]["stratified_bootstrap_seed"]),
        "comparisons": comparisons,
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = {metric: correlations(rows, metric) for metric in METRICS}
    gaps = {
        baseline: {
            "delta_spearman_rho": delta(primary["d_pac"], primary[baseline], "spearman_rho"),
            "delta_kendall_tau_b": delta(primary["d_pac"], primary[baseline], "kendall_tau_b"),
        }
        for baseline in ("mse", "d_func")
    }
    by_split = {}
    for split in SPLITS:
        split_rows = [
            {**row["splits"][split], "predictive_mask_id": row["predictive_mask_id"]}
            for row in rows
        ]
        by_split[split] = {metric: correlations(split_rows, metric) for metric in METRICS}
    by_distance = {}
    for distance in sorted({int(row["swap_count"]) for row in rows}):
        selected = [row for row in rows if int(row["swap_count"]) == distance]
        by_distance[str(distance)] = {
            metric: correlations(selected, metric) for metric in METRICS
        }
    return {
        "primary": primary,
        "primary_gaps": gaps,
        "by_split": by_split,
        "within_swap_count": by_distance,
        "stratified_bootstrap": stratified_bootstrap(rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "predictive_mask_id", "swap_count", "hamming_distance", *METRICS,
        "fp16_success_rate", "mask_success_rate", "success_drop",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def write_plot(path: Path, rows: list[dict[str, Any]], analysis: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"mse": "Raw physical-action MSE", "d_func": "Dfunc", "d_pac": "DPAC"}
    distances = sorted({int(row["swap_count"]) for row in rows})
    colors = plt.get_cmap("viridis", len(distances))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for axis, metric in zip(axes, METRICS):
        for index, distance in enumerate(distances):
            selected = [row for row in rows if int(row["swap_count"]) == distance]
            axis.scatter(
                [row[metric] for row in selected],
                [row["success_drop"] for row in selected],
                s=34, alpha=0.85, color=colors(index), label=f"k={distance}",
            )
        rho = analysis["primary"][metric]["spearman_rho"]["value"]
        tau = analysis["primary"][metric]["kendall_tau_b"]["value"]
        annotation = (
            f"Spearman ρ={rho:.3f}\nKendall τ-b={tau:.3f}"
            if rho is not None and tau is not None
            else "correlation undefined"
        )
        axis.text(0.04, 0.96, annotation, va="top", transform=axis.transAxes)
        axis.set_xlabel(labels[metric])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Signed success drop (SR_FP16 − SR_mask)")
    axes[-1].legend(title="matched swaps", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _fmt(stat: dict[str, Any]) -> str:
    return "undefined" if stat["value"] is None else f"{stat['value']:.4f}"


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    lines = [
        "# GR00T DPAC Predictive-Validity Experiment",
        "",
        "## Protocol and audit outcome",
        "",
        "The preregistered evaluation completed with exactly 60 masks × 50 RoboCasa365 "
        "target tasks (3,000 mask episodes) and 50 independent native-FP16 episodes. "
        "All masks retain 100 W4 / 16 FP16 layers at the same exact static byte budget; "
        "offline metrics were frozen before closed-loop outcomes were launched.",
        "",
        "## Primary analysis",
        "",
        "| Metric | Spearman ρ | Kendall τ-b |",
        "|---|---:|---:|",
    ]
    for metric in METRICS:
        value = analysis["primary"][metric]
        lines.append(
            f"| {metric} | {_fmt(value['spearman_rho'])} | {_fmt(value['kendall_tau_b'])} |"
        )
    lines.extend(["", "| Comparison | Δρ | Δτ-b |", "|---|---:|---:|"])
    for baseline in ("mse", "d_func"):
        gap = analysis["primary_gaps"][baseline]
        lines.append(
            f"| DPAC − {baseline} | {_fmt(gap['delta_spearman_rho'])} | "
            f"{_fmt(gap['delta_kendall_tau_b'])} |"
        )
    lines.extend(
        [
            "",
            "These are descriptive rank associations over the fixed 60-mask library. "
            "A positive value means a larger offline distortion is associated with a "
            "larger signed closed-loop success degradation. Negative and undefined results "
            "are retained without modifying the mask library.",
            "",
            "## Secondary analysis",
            "",
            "### RoboCasa365 splits",
            "",
            "| Split | Metric | ρ | τ-b |",
            "|---|---|---:|---:|",
        ]
    )
    for split in SPLITS:
        for metric in METRICS:
            value = analysis["by_split"][split][metric]
            lines.append(
                f"| {split} | {metric} | {_fmt(value['spearman_rho'])} | "
                f"{_fmt(value['kendall_tau_b'])} |"
            )
    lines.extend(
        [
            "",
            "### Within matched-swap distance",
            "",
            "| k | Metric | ρ | τ-b |",
            "|---:|---|---:|---:|",
        ]
    )
    for distance, values in analysis["within_swap_count"].items():
        for metric in METRICS:
            value = values[metric]
            lines.append(
                f"| {distance} | {metric} | {_fmt(value['spearman_rho'])} | "
                f"{_fmt(value['kendall_tau_b'])} |"
            )
    boot = analysis["stratified_bootstrap"]
    lines.extend(
        [
            "",
            "### Distance-stratified resampling",
            "",
            f"Using {boot['iterations']:,} fixed-seed within-distance resamples:",
            "",
            "| Baseline | Statistic | P(DPAC > baseline) | Defined resamples |",
            "|---|---|---:|---:|",
        ]
    )
    for baseline, values in boot["comparisons"].items():
        for kind, value in values.items():
            probability = value["proportion_dpac_greater"]
            rendered = "undefined" if probability is None else f"{probability:.4f}"
            lines.append(
                f"| {baseline} | {kind} | {rendered} | {value['defined_resamples']} |"
            )
    undefined = []
    for scope, values in [("primary", analysis["primary"])]:
        for metric, pair in values.items():
            for kind, stat in pair.items():
                if stat["value"] is None:
                    undefined.append(f"{scope}/{metric}/{kind}: {stat['status']}")
    for split, values in analysis["by_split"].items():
        for metric, pair in values.items():
            for kind, stat in pair.items():
                if stat["value"] is None:
                    undefined.append(f"split/{split}/{metric}/{kind}: {stat['status']}")
    for distance, values in analysis["within_swap_count"].items():
        for metric, pair in values.items():
            for kind, stat in pair.items():
                if stat["value"] is None:
                    undefined.append(f"k={distance}/{metric}/{kind}: {stat['status']}")
    lines.extend(["", "## Ties and undefined statistics", ""])
    if undefined:
        lines.extend(f"- {value}" for value in undefined)
    else:
        lines.append("All requested correlations were defined; tau-b accounts for ties.")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "The result applies to the frozen GR00T N1.5 checkpoints, RoboCasa365 target "
            "tasks, seed 70, DyRange-A8, Hessian group-64 W4, four flow steps, execute-16, "
            "and this exact matched-budget mask distribution. It does not establish causal "
            "superiority, cross-model validity, or validity under activation variants.",
            "",
            "## Artifacts",
            "",
            f"- Mask summaries: `{path.with_name('mask_results.csv').name}`",
            f"- Complete audit: `{path.with_name('audit.json').name}`",
            f"- Provenance manifest: `{path.with_name('coverage_provenance.json').name}`",
            f"- Three-panel figure: `{path.with_name('metric_vs_success_drop.png').name}`",
            "",
            f"Mask-library SHA-256: `{provenance['mask_library']['sha256']}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--offline-scores", required=True)
    parser.add_argument("--mask-results-root", required=True)
    parser.add_argument("--fp16-results-root", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    library_path, library = _read_json(args.manifest)
    score_path, scores = _read_json(args.offline_scores)
    rows, fp16_rows, provenance = validate_inputs(
        library_path, library, score_path, scores,
        args.mask_results_root, args.fp16_results_root,
    )
    result = analyze(rows)
    out = Path(args.out_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": 1,
        "kind": "gr00t_dpac_predictive_validity_audit",
        "predictive_validity_protocol": protocol_attestation(),
        "coverage_gate": {
            "passed": True,
            "mask_rows": len(rows) * 50,
            "fp16_rows": len(fp16_rows),
            "candidate_count": len(rows),
            "task_count": 50,
            "no_duplicate_failed_or_drifted_rows": True,
        },
        "mask_results": rows,
        "analysis": result,
        "provenance": provenance,
    }
    atomic_json(out / "audit.json", audit)
    atomic_json(
        out / "coverage_provenance.json",
        {
            "predictive_validity_protocol": protocol_attestation(),
            "coverage_gate": audit["coverage_gate"],
            "provenance": provenance,
        },
    )
    write_csv(out / "mask_results.csv", rows)
    write_plot(out / "metric_vs_success_drop.png", rows, result)
    write_report(out / "report.md", rows, result, provenance)
    print(json.dumps({"out": str(out), "coverage_gate": audit["coverage_gate"]}, indent=2))


if __name__ == "__main__":
    main()
