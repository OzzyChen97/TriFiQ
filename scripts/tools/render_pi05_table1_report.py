#!/usr/bin/env python3
"""Render the audited π0.5 RoboCasa365 Table-1 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONFIG_ORDER = (
    "fp16",
    "quantvla_w4a8_atmohb",
    "gdsq_vla_atmohb",
    "gdsq_vla",
)
LABELS = {
    "fp16": "FP16",
    "quantvla_w4a8_atmohb": "QuantVLA W4A8 + ATM/OHB",
    "gdsq_vla_atmohb": "GDSQ-VLA + ATM/OHB",
    "gdsq_vla": "GDSQ-VLA (Ours)",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def number(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def comparison_passes(row: dict[str, Any]) -> bool:
    return bool(
        row["heldout46_task_macro_delta"] > 0
        and row["heldout46_task_cluster_ci95"][0] > 0
        and row["holm_adjusted_p"] <= 0.05
    )


def render(
    summary: dict[str, Any], audit: dict[str, Any], manifest: dict[str, Any]
) -> str:
    require(summary.get("complete") is True, "summary is not complete")
    require(audit.get("complete") is True, "completion audit is not complete")
    require(summary["manifest_sha256"] == audit["manifest_sha256"], "manifest mismatch")
    require(audit["statistics"]["comparisons_ready"] is True, "comparisons are not audited")
    for config in CONFIG_ORDER:
        require(summary["configs"][config]["completed_episodes"] == 2500, f"incomplete {config}")

    protocol = manifest["table_1_protocol"]
    task_sets = protocol["task_sets"]
    comparisons = summary["comparisons"]
    ours_vs_w4 = comparisons["gdsq_vla_vs_quantvla_w4a8_atmohb"]
    ours_vs_fp16 = comparisons["gdsq_vla_vs_fp16"]
    atm_vs_ours = comparisons["gdsq_vla_atmohb_vs_gdsq_vla"]
    ours_passes = comparison_passes(ours_vs_w4)
    atm_passes = comparison_passes(atm_vs_ours)

    lines = [
        "# π0.5 RoboCasa365 Table 1 — Audited Final Report",
        "",
        "> Status: **complete and audited**. This report is generated only after the immutable ",
        "> 4-config × 50-task × 50-seed matrix and completion audit both pass.",
        "",
        "## Main result",
        "",
        "| Configuration | Atomic-Seen | Composite-Seen | Composite-Unseen | Held-out 46 | 50-task Macro | Episode SR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIG_ORDER:
        row = summary["configs"][config]
        task_set_sr = row["task_set_macro_sr"]
        lines.append(
            f"| {LABELS[config]} | {pct(task_set_sr['atomic_seen'])} | "
            f"{pct(task_set_sr['composite_seen'])} | "
            f"{pct(task_set_sr['composite_unseen'])} | "
            f"{pct(row['heldout46_task_macro_sr'])} | {pct(row['task_macro_sr'])} | "
            f"{pct(row['episode_sr'])} |"
        )

    lines += [
        "",
        "The four ratio-development tasks are excluded from the primary held-out column and included "
        "only in the secondary 50-task macro. Every task contributes equally to each macro average.",
        "",
        "## Prespecified paired comparisons (46-task held-out)",
        "",
        "| Comparison | Macro Δ | Task-cluster 95% CI | Permutation p | Holm p | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, row in comparisons.items():
        ci = row["heldout46_task_cluster_ci95"]
        lines.append(
            f"| {LABELS[row['a']]} vs {LABELS[row['b']]} | "
            f"{100 * row['heldout46_task_macro_delta']:+.2f}pp | "
            f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]pp | "
            f"{row['paired_permutation_p']:.4g} | {row['holm_adjusted_p']:.4g} | "
            f"{'positive' if comparison_passes(row) else 'not positive under the preregistered gate'} |"
        )

    lines += [
        "",
        "### Auxiliary episode-paired McNemar",
        "",
        "| Comparison | A wins | B wins | Two-sided p |",
        "|---|---:|---:|---:|",
    ]
    for row in comparisons.values():
        test = row["episode_mcnemar"]
        lines.append(
            f"| {LABELS[row['a']]} vs {LABELS[row['b']]} | "
            f"{test['a_wins']} | {test['b_wins']} | {test['two_sided_p']:.4g} |"
        )

    lines += [
        "",
        "## Frozen decisions",
        "",
        f"- GDSQ-VLA vs original W4A8 acceptance gate: **{'pass' if ours_passes else 'fail'}**.",
        f"- Plan-specific ATM/OHB vs GDSQ-only gate: **{'pass; enable by default' if atm_passes else 'fail; keep disabled by default'}**.",
        "- π0.5 ratio development selected CKA-only (`lambda_cs=0`); the 46 held-out tasks "
        "were not used to reopen CS tuning.",
        f"- Relative to FP16, GDSQ-VLA held-out delta is "
        f"`{100 * ours_vs_fp16['heldout46_task_macro_delta']:+.2f}pp` with Holm-adjusted "
        f"`p={ours_vs_fp16['holm_adjusted_p']:.4g}`.",
        "",
        "## Per-task success rate",
    ]
    for task_set, tasks in task_sets.items():
        lines += [
            "",
            f"### {task_set}",
            "",
            "| Task | " + " | ".join(LABELS[config] for config in CONFIG_ORDER) + " |",
            "|---|" + "---:|" * len(CONFIG_ORDER),
        ]
        for task in tasks:
            values = [summary["configs"][config]["per_task_sr"][task] for config in CONFIG_ORDER]
            lines.append("| " + task + " | " + " | ".join(pct(value) for value in values) + " |")

    lines += [
        "",
        "## Closed-loop efficiency",
        "",
        "| Configuration | Episode wall (s) | Server infer (ms) | Inference/replan (s) | Success steps | Failure steps | Peak server MiB | Peak device MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIG_ORDER:
        efficiency = summary["configs"][config]["efficiency"]
        gpu = efficiency.get("gpu") or {}
        lines.append(
            f"| {LABELS[config]} | {number(efficiency['mean_episode_wall_seconds'])} | "
            f"{number(efficiency['mean_server_infer_ms'])} | "
            f"{number(efficiency['mean_inference_seconds_per_replan'], 3)} | "
            f"{number(efficiency['mean_success_steps'])} | {number(efficiency['mean_failure_steps'])} | "
            f"{number(gpu.get('peak_server_process_memory_mib'), 0)} | "
            f"{number(gpu.get('peak_device_memory_used_mib'), 0)} |"
        )

    memory = summary["paper_style_memory"]
    lines += [
        "",
        "## QuantVLA paper-style candidate-component storage",
        "",
        "| Configuration | Theoretical GiB | Compression vs FP16 |",
        "|---|---:|---:|",
    ]
    for config in CONFIG_ORDER:
        row = memory["by_config"][config]
        lines.append(
            f"| {LABELS[config]} | {row['gib']:.3f} | {row['compression_vs_fp16']:.2f}× |"
        )
    lines += [
        "",
        f"GDSQ quantizes {memory['gdsq_w4_parameters']:,}/{memory['candidate_parameters']:,} "
        f"candidate parameters ({100 * memory['gdsq_w4_parameter_fraction']:.1f}%) across "
        f"{memory['gdsq_w4_layers']} W4 + {memory['gdsq_fp16_layers']} native-FP16 layers.",
        "",
        "Theoretical packed storage, eager CUDA residency, and shared whole-device memory are "
        "different scopes. Formal rollout latency includes simulator and multi-client queueing and "
        "is not presented as isolated single-request service-time acceleration.",
        "This implementation emulates W4/A8 numerical error but executes the matrix multiply with "
        "FP16 tensors (`fake_quant_fp16_gemm`). It does not reproduce the paper's packed int4/int8 "
        "kernel residency, memory footprint, or deployment speedup; the storage table is theoretical.",
        "",
        "## Cross-model context (not pooled)",
        "",
        "| GR00T N1.5 configuration | Atomic | Composite-Seen | Composite-Unseen | 50-task Macro |",
        "|---|---:|---:|---:|---:|",
        "| FP16 | 75.6% | 41.9% | 45.3% | 55.1% |",
        "| Original W4A8 + ATM/OHB | 49.6% | 19.6% | 19.8% | 30.4% |",
        "| GDSQ-VLA final | 69.0% | 40.6% | 40.4% | 50.8% |",
        "| GDSQ-VLA final + ATM/OHB | 66.7% | 39.1% | 40.3% | 49.4% |",
        "",
        "GR00T and π0.5 use architecture-native checkpoints and execution protocols; this table is "
        "background context rather than a pooled paired test. LIBERO context is v1.4 89.2%, uniform "
        "W6 88.2%, and v1.3 85.2%, also not pooled with RoboCasa365.",
        "",
        "## Provenance",
        "",
        f"- Immutable manifest SHA256: `{audit['manifest_sha256']}`",
        f"- Frozen aggregator SHA256: `{audit['frozen_aggregator_sha256']}`",
        f"- Completion auditor SHA256: `{audit['audit_tool_sha256']}`",
        f"- Bootstrap draws: `{audit['bootstrap_draws']}`",
        f"- Verified frozen artifacts: `{audit['artifacts']['verified']}`",
        f"- Resource schedule leaf SHA256: `{audit['schedule']['chain_leaf_to_root'][0]['sha256']}`",
        f"- Resource schedule chain depth: `{len(audit['schedule']['chain_leaf_to_root'])}`",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(render(summary, audit, manifest), encoding="utf-8")
    temporary.replace(output)
    print(f"wrote audited Table-1 report: {output}")


if __name__ == "__main__":
    main()
