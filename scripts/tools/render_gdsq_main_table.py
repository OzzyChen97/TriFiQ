#!/usr/bin/env python3
"""Render the GDSQ-VLA main table exclusively from audited evidence records."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "docs/gdsq_vla_cvpr2026/experiment_registry.json"
DEFAULT_TEX = REPO_ROOT / "docs/gdsq_vla_cvpr2026/tables/main_results.tex"
DEFAULT_AUDIT = REPO_ROOT / "docs/gdsq_vla_cvpr2026/tables/main_results.audit.json"
PENDING = r"\textit{pending}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verified_json(record: dict[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    path = resolve(str(record["path"]))
    require(path.is_file(), f"missing {label}: {path}")
    actual = sha256_file(path)
    require(actual == record["sha256"], f"{label} SHA drift: {actual}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def metrics(row: dict[str, Any], task_set_field: str) -> dict[str, float]:
    groups = row[task_set_field]
    return {
        "atomic": 100.0 * float(groups["atomic_seen"]),
        "composite_seen": 100.0 * float(groups["composite_seen"]),
        "composite_unseen": 100.0 * float(groups["composite_unseen"]),
        "mean": 100.0 * float(row["task_macro_sr"]),
    }


def format_metric(value: float | None) -> str:
    if value is None:
        return PENDING
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def result_cells(row: dict[str, Any]) -> list[str]:
    values = row.get("metrics") or {}
    return [
        format_metric(values.get("atomic")),
        format_metric(values.get("composite_seen")),
        format_metric(values.get("composite_unseen")),
        format_metric(values.get("mean")),
    ]


def latex_row(row: dict[str, Any]) -> str:
    values = result_cells(row)
    return (
        f"\\quad {row['label']} & {row['low_bit_plan']} & "
        + " & ".join(values)
        + f" & {row['storage']} & {row['compression']} \\\\"
    )


def pending_row(
    *,
    label: str,
    low_bit_plan: str,
    storage: str,
    compression: str,
    evidence: str,
    status: str = "pending",
) -> dict[str, Any]:
    return {
        "label": label,
        "low_bit_plan": low_bit_plan,
        "metrics": None,
        "storage": storage,
        "compression": compression,
        "status": status,
        "evidence": evidence,
        "paper_claim_enabled": False,
    }


def gated_summary(
    experiments: dict[str, Any], experiment_id: str, label: str
) -> tuple[dict[str, Any], Path | None, dict[str, Any] | None]:
    """Return a summary only after the registry's complete-evidence gate passes."""
    experiment = experiments[experiment_id]
    if (
        experiment.get("status") != "complete"
        or experiment.get("main_claim_enabled") is not True
        or not experiment.get("summary")
    ):
        return experiment, None, None
    coverage = experiment.get("coverage") or {}
    require(coverage.get("expected_episodes") == 2500, f"{label} expected coverage drift")
    require(coverage.get("observed_episodes") == 2500, f"{label} observed coverage drift")
    require(coverage.get("missing_episodes") == 0, f"{label} has missing episodes")
    require(coverage.get("duplicate_episodes") == 0, f"{label} has duplicate episodes")
    path, summary = verified_json(experiment["summary"], f"{label} summary")
    return experiment, path, summary


def omega_gr00t_partial_row(
    experiments: dict[str, Any], fp16_reference_bytes: int
) -> tuple[dict[str, Any], dict[str, str]]:
    """Enable only independently complete Omega-QVLA task-set/storage cells."""
    experiment = experiments["omega_qvla_robocasa365"]
    progress = experiment["model_progress"]["gr00t"]
    atomic = progress["task_sets"]["atomic_seen"]
    coverage = atomic["coverage"]
    require(atomic.get("status") == "complete", "Omega-QVLA Atomic status drift")
    require(atomic.get("main_claim_enabled") is True, "Omega-QVLA Atomic claim disabled")
    require(coverage.get("expected_episodes") == 900, "Omega-QVLA Atomic expected coverage drift")
    require(coverage.get("observed_episodes") == 900, "Omega-QVLA Atomic observed coverage drift")
    require(coverage.get("missing_episodes") == 0, "Omega-QVLA Atomic has missing episodes")
    require(coverage.get("duplicate_episodes") == 0, "Omega-QVLA Atomic has duplicates")
    require(coverage.get("tasks") == 18, "Omega-QVLA Atomic task-count drift")
    require(coverage.get("seeds_per_task") == 50, "Omega-QVLA Atomic seed-count drift")

    manifest_path, manifest = verified_json(atomic["manifest"], "Omega-QVLA Atomic manifest")
    summary_path, summary = verified_json(atomic["summary"], "Omega-QVLA Atomic summary")
    memory_path, memory = verified_json(progress["memory"], "Omega-QVLA memory audit")
    require(summary.get("validation_errors") == [], "Omega-QVLA Atomic validation errors")
    require(summary.get("task_set") == "atomic_seen", "Omega-QVLA Atomic task-set drift")
    require(summary.get("manifest_sha256") == sha256_file(manifest_path), "Omega manifest link drift")
    result = summary.get("configs", {}).get("omega_qvla_w4a4", {})
    require(result.get("episodes") == 900, "Omega-QVLA Atomic result coverage drift")
    require(result.get("formal_failures") == 0, "Omega-QVLA Atomic has formal failures")
    details = result.get("per_task_details") or {}
    require(len(details) == 18, "Omega-QVLA Atomic per-task coverage drift")
    require(all(row.get("episodes") == 50 for row in details.values()), "Omega task seed drift")
    require(len(manifest.get("configs") or []) == 1, "Omega-QVLA manifest config drift")

    require(memory.get("kind") == "omega_qvla_theoretical_packed_storage_audit", "Omega memory kind drift")
    require(memory.get("scope_linear_layers") == 180, "Omega memory layer-count drift")
    require(memory.get("quantized_layers") == 180, "Omega quantized-layer count drift")
    require(memory.get("representation", {}).get("weight_bits") == 4, "Omega memory is not W4")
    require(memory.get("representation", {}).get("activation_bits") == 4, "Omega memory is not A4")
    require(
        int(memory["fp16_reference"]["component_bytes"]) == fp16_reference_bytes,
        "Omega FP16 reference differs from the main-table scope",
    )
    require(
        memory.get("evidence", {}).get("summary", {}).get("sha256") == sha256_file(summary_path),
        "Omega memory-to-summary link drift",
    )
    packed_bytes = int(memory["packed"]["component_bytes"])
    row = {
        "label": r"$\Omega$-QVLA W4A4$^{\ddagger}$",
        "low_bit_plan": "180 W4",
        "metrics": {"atomic": 100.0 * float(result["task_macro_sr"])},
        "storage": f"{packed_bytes / 2**30:.3f}",
        "compression": f"{fp16_reference_bytes / packed_bytes:.2f}$\\times$",
        "status": progress["status"],
        "evidence": "omega_qvla_robocasa365",
        "paper_claim_enabled": False,
        "claim_enabled_cells": {
            "atomic": True,
            "composite_seen": False,
            "composite_unseen": False,
            "mean": False,
            "storage": True,
            "compression": True,
        },
    }
    sources = {
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "summary": str(summary_path.relative_to(REPO_ROOT)),
        "memory": str(memory_path.relative_to(REPO_ROOT)),
    }
    return row, sources


def build(registry_path: Path) -> tuple[str, dict[str, Any]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    experiments = registry["experiments"]
    gr_path, gr_summary = verified_json(
        experiments["gr00t_static_official50"]["summary"], "GR00T summary"
    )
    pi_path, pi_summary = verified_json(
        experiments["pi05_static_official50"]["summary"], "pi0.5 static summary"
    )
    prereg_path, prereg = verified_json(
        registry["preregistrations"]["week1_same_budget_v1"], "week-1 preregistration"
    )

    gr_fp16_bytes = int(
        gr_summary["configs"]["fp16"]["paper_style_memory"][
            "task_weighted_mean_component_bytes"
        ]
    )
    gr_gdsq_bytes = int(
        gr_summary["configs"]["cscka_final"]["paper_style_memory"][
            "task_weighted_mean_component_bytes"
        ]
    )
    pi_fp16_bytes = int(prereg["models"]["pi05"]["fp16_bytes"])
    pi_gdsq_bytes = int(prereg["models"]["pi05"]["final_plan"]["total_bytes_recomputed"])

    gr_candidate_fp16_bytes = int(prereg["models"]["gr00t"]["fp16_bytes"])
    gr_fixed_bytes = gr_fp16_bytes - gr_candidate_fp16_bytes
    require(gr_fixed_bytes >= 0, "GR00T fixed-byte accounting is negative")
    gr_w6_candidate_bytes = int(prereg["models"]["gr00t"]["uniform_w6_budget_bytes"])
    gr_w6_bytes = gr_fixed_bytes + gr_w6_candidate_bytes
    pi_w6_bytes = int(prereg["models"]["pi05"]["uniform_w6_budget_bytes"])
    require(
        int(prereg["models"]["gr00t"]["plans"]["uniform_w6"]["quantized_layers"]) == 116,
        "GR00T Uniform W6 layer-count drift",
    )
    require(
        int(prereg["models"]["pi05"]["plans"]["uniform_w6"]["quantized_layers"]) == 180,
        "pi0.5 Uniform W6 layer-count drift",
    )

    def memory_cells(value: int, reference: int) -> tuple[str, str]:
        return f"{value / 2**30:.3f}", f"{reference / value:.2f}$\\times$"

    gr_gdsq_storage, gr_gdsq_compression = memory_cells(gr_gdsq_bytes, gr_fp16_bytes)
    pi_gdsq_storage, pi_gdsq_compression = memory_cells(pi_gdsq_bytes, pi_fp16_bytes)
    gr_w6_storage, gr_w6_compression = memory_cells(gr_w6_bytes, gr_fp16_bytes)
    pi_w6_storage, pi_w6_compression = memory_cells(pi_w6_bytes, pi_fp16_bytes)

    omega_gr00t_row, omega_gr00t_sources = omega_gr00t_partial_row(
        experiments, gr_fp16_bytes
    )

    gr_w6_experiment, gr_w6_path, gr_w6_summary = gated_summary(
        experiments, "gr00t_uniform_w6_official50", "GR00T Uniform W6"
    )
    gr_w6_metrics = None
    if gr_w6_summary is not None:
        require(gr_w6_summary.get("n_tasks") == 50, "GR00T Uniform W6 task coverage drift")
        require(
            gr_w6_summary.get("episodes_per_config") == 2500,
            "GR00T Uniform W6 episode coverage drift",
        )
        require(
            gr_w6_summary.get("configs", {}).get("uniform_w6", {}).get("episodes") == 2500,
            "GR00T Uniform W6 config coverage drift",
        )
        gr_w6_metrics = metrics(
            gr_w6_summary["configs"]["uniform_w6"], "per_task_set_macro_sr"
        )

    pi_w6_experiment, pi_w6_path, pi_w6_summary = gated_summary(
        experiments, "pi05_uniform_w6_official50", "pi0.5 Uniform W6"
    )
    pi_w6_metrics = None
    if pi_w6_summary is not None:
        require(pi_w6_summary.get("complete") is True, "pi0.5 Uniform W6 summary incomplete")
        require(
            pi_w6_summary.get("completed_episodes") == 2500,
            "pi0.5 Uniform W6 episode coverage drift",
        )
        pi_w6_metrics = metrics(
            pi_w6_summary["configs"]["uniform_w6"], "task_set_macro_sr"
        )

    gr_rows = [
        {
            "label": "FP16",
            "low_bit_plan": "--",
            "metrics": metrics(gr_summary["configs"]["fp16"], "per_task_set_macro_sr"),
            "storage": f"{gr_fp16_bytes / 2**30:.3f}",
            "compression": "1.00$\\times$",
            "status": "complete",
            "evidence": "gr00t_static_official50",
            "paper_claim_enabled": True,
        },
        {
            "label": r"\quantvla W4A8",
            "low_bit_plan": "116 W4",
            "metrics": metrics(gr_summary["configs"]["w4a8_atmohb"], "per_task_set_macro_sr"),
            "storage": "0.898",
            "compression": "2.22$\\times$",
            "status": "complete",
            "evidence": "gr00t_static_official50",
            "paper_claim_enabled": True,
        },
        {
            "label": "Uniform W6",
            "low_bit_plan": "116 W6",
            "metrics": gr_w6_metrics,
            "storage": gr_w6_storage,
            "compression": gr_w6_compression,
            "status": gr_w6_experiment.get("status"),
            "evidence": "gr00t_uniform_w6_official50",
            "paper_claim_enabled": gr_w6_metrics is not None,
        },
        omega_gr00t_row,
        {
            "label": r"\textbf{\textsc{GDSQ-VLA}(Ours)}",
            "low_bit_plan": "100 W4",
            "metrics": metrics(gr_summary["configs"]["cscka_final"], "per_task_set_macro_sr"),
            "storage": gr_gdsq_storage,
            "compression": gr_gdsq_compression,
            "status": "complete_by_equivalence_reuse",
            "evidence": "gr00t_runtime_selector_official50",
            "paper_claim_enabled": True,
        },
    ]

    pi_selector = experiments["pi05_runtime_selector_official50"]
    pi_selector_metrics = None
    pi_selector_source = None
    if pi_selector.get("status") == "complete" and pi_selector.get("summary"):
        selector_path, selector_summary = verified_json(
            pi_selector["summary"], "pi0.5 selector summary"
        )
        require(selector_summary.get("complete") is True, "pi0.5 selector summary incomplete")
        require(selector_summary.get("completed_episodes") == 2500, "wrong selector coverage")
        pi_selector_metrics = metrics(
            selector_summary["configs"]["gdsq_vla_runtime_selector"],
            "task_set_macro_sr",
        )
        pi_selector_source = str(selector_path.relative_to(REPO_ROOT))

    pi_rows = [
        {
            "label": "FP16",
            "low_bit_plan": "--",
            "metrics": metrics(pi_summary["configs"]["fp16"], "task_set_macro_sr"),
            "storage": f"{pi_fp16_bytes / 2**30:.3f}",
            "compression": "1.00$\\times$",
            "status": "complete",
            "evidence": "pi05_static_official50",
            "paper_claim_enabled": True,
        },
        {
            "label": r"\quantvla W4A8",
            "low_bit_plan": "180 W4",
            "metrics": metrics(
                pi_summary["configs"]["quantvla_w4a8_atmohb"], "task_set_macro_sr"
            ),
            "storage": "1.388",
            "compression": "2.96$\\times$",
            "status": "complete",
            "evidence": "pi05_static_official50",
            "paper_claim_enabled": True,
        },
        {
            "label": "Uniform W6",
            "low_bit_plan": "180 W6",
            "metrics": pi_w6_metrics,
            "storage": pi_w6_storage,
            "compression": pi_w6_compression,
            "status": pi_w6_experiment.get("status"),
            "evidence": "pi05_uniform_w6_official50",
            "paper_claim_enabled": pi_w6_metrics is not None,
        },
        pending_row(
            label=r"$\Omega$-QVLA W4A4$^{\ddagger}$",
            low_bit_plan="W4A4",
            storage=PENDING,
            compression=PENDING,
            evidence="omega_qvla_robocasa365",
            status=experiments["omega_qvla_robocasa365"]
            .get("model_progress", {})
            .get("pi05", {})
            .get("status", "planned"),
        ),
        {
            "label": r"\textbf{\textsc{GDSQ-VLA}(Ours)}",
            "low_bit_plan": "80 W4",
            "metrics": pi_selector_metrics,
            "storage": pi_gdsq_storage,
            "compression": pi_gdsq_compression,
            "status": pi_selector.get("status"),
            "evidence": "pi05_runtime_selector_official50",
            "paper_claim_enabled": pi_selector.get("main_claim_enabled") is True,
        },
    ]

    lines = [
        "% AUTO-GENERATED by scripts/tools/render_gdsq_main_table.py; DO NOT EDIT.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{RoboCasa365 task-macro success rate (SR, \%) over 50 paired scenarios per task.}",
        r"\label{tab:main_results}",
        r"\small",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabular}{@{}lcrrrrrr@{}}",
        r"\toprule",
        r"Configuration & \shortstack{Low-bit\\layers} & \shortstack{Atomic\\SR $\uparrow$} & \shortstack{C-Seen\\SR $\uparrow$} & \shortstack{C-Unseen\\SR $\uparrow$} & \shortstack{All\\SR $\uparrow$} & \shortstack{Size\\(GiB) $\downarrow$} & \shortstack{Comp.\\$\uparrow$} \\",
        r"\midrule",
        r"\multicolumn{8}{@{}l}{\textbf{GR00T N1.5}\enspace\textit{(target split)}} \\",
    ]
    lines.extend(latex_row(row) for row in gr_rows)
    lines += [
        r"\midrule",
        r"\multicolumn{8}{@{}l}{\textbf{$\pi_{0.5}$}~\cite{intelligence2025pi05}\enspace\textit{(target split)}} \\",
    ]
    lines.extend(latex_row(row) for row in pi_rows)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\parbox{0.99\textwidth}{\footnotesize C-Seen/C-Unseen denote Composite-Seen/Composite-Unseen. Ours uses the frozen v8 selector. Uniform W6 is the direct byte-budget baseline; QuantVLA is a different-rate comparison. Size is theoretical tightly packed candidate-Linear storage, with FP16 shown as the reference. Pending cells are claim-disabled. $^{\ddagger}\Omega$-QVLA uses RoboCasa365-specific calibration; its GR00T Atomic cell is complete, while the remaining cells await exact coverage. Its displayed size includes packed W4 weights and required scales/rotation metadata, not the dequantized evaluation pack.}",
        r"\end{table*}",
        "",
    ]

    audit = {
        "schema_version": 1,
        "kind": "gdsq_vla_main_table_claim_evidence_map",
        "generated_by": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "registry": {
            "path": str(registry_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(registry_path),
        },
        "sources": {
            "gr00t_summary": {"path": str(gr_path.relative_to(REPO_ROOT)), "sha256": sha256_file(gr_path)},
            "pi05_static_summary": {"path": str(pi_path.relative_to(REPO_ROOT)), "sha256": sha256_file(pi_path)},
            "week1_preregistration": {
                "path": str(prereg_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(prereg_path),
            },
            "pi05_selector_summary": pi_selector_source,
            "gr00t_uniform_w6_summary": (
                str(gr_w6_path.relative_to(REPO_ROOT)) if gr_w6_path else None
            ),
            "pi05_uniform_w6_summary": (
                str(pi_w6_path.relative_to(REPO_ROOT)) if pi_w6_path else None
            ),
            "omega_qvla_gr00t": omega_gr00t_sources,
        },
        "rows": {"gr00t": gr_rows, "pi05": pi_rows},
        "forbidden_manual_result_sources": True,
        "pending_cells_are_claim_disabled": True,
    }
    return "\n".join(lines), audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--tex", default=str(DEFAULT_TEX))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    registry = resolve(args.registry)
    tex_path = resolve(args.tex)
    audit_path = resolve(args.audit)
    tex, audit = build(registry)
    audit_text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.check:
        require(tex_path.read_text(encoding="utf-8") == tex, "generated LaTeX table is stale")
        require(audit_path.read_text(encoding="utf-8") == audit_text, "table audit is stale")
        print(f"main table sources verified: {tex_path}")
        return
    tex_path.write_text(tex, encoding="utf-8")
    audit_path.write_text(audit_text, encoding="utf-8")
    print(f"main table rendered: {tex_path}")
    print(f"main table audit: {audit_path}")


if __name__ == "__main__":
    main()
