#!/usr/bin/env python3
"""Render the Omega-QVLA LIBERO comparison from audited evidence only."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "docs/gdsq_vla_iclr2027"
REGISTRY = PAPER_DIR / "experiment_registry.json"
SOURCE = REPO_ROOT / "runs/gdsq_extension_preregistered_v1/omega_qvla_table1/source_record.json"
OUTPUT = PAPER_DIR / "tables/omega_qvla_libero.tex"
COMPACT_OUTPUT = PAPER_DIR / "tables/libero_transfer_compact.tex"
AUDIT = PAPER_DIR / "tables/omega_qvla_libero.audit.json"
TABLE6_RESULTS = REPO_ROOT / "runs/table6_libero_v1/results"
DYPAC_RESULTS = REPO_ROOT / "runs/libero_dypac_v1/results"
DYPAC_GR00T_RESULTS = REPO_ROOT / "runs/libero_dypac_v1/results_v2"
ATTESTED_PI05_OMEGA = (
    REPO_ROOT
    / "runs/table6_libero_v1/external_attested/pi05_omega_qvla_w4a4.json"
)
STATIC_MEMORY = REPO_ROOT / "runs/table6_libero_v1/artifacts/table5_static_memory.json"
MODELS = ("gr00t", "pi05")
CONFIGS = (
    "fp16",
    "quantvla_w4a8",
    "uniform_w6",
    "omega_qvla_w4a4",
    "gdsq_vla_selector",
)
SUITES = ("goal", "spatial", "object", "long")
PENDING = r"\textit{pending}"
EXTERNAL_PI05_ROWS = {
    "qvla_source_4bpw": {
        "metrics": {"goal": 96.4, "spatial": 98.0, "object": 97.2, "long": 91.8, "average": 95.8},
        "memory_gb": 2.7,
    },
    "actquant_source_4bpw": {
        "metrics": {"goal": 96.8, "spatial": 98.4, "object": 99.4, "long": 91.8, "average": 96.6},
        "memory_gb": 2.7,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def metric(value: float | None) -> str:
    if value is None:
        return PENDING
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def row(
    config: str,
    values: dict[str, float] | None,
    memory_display: str | None = None,
) -> str:
    cells = [metric(values.get(name) if values else None) for name in (*SUITES, "average")]
    cells.append(memory_display if memory_display is not None else "--")
    return rf"\quad {config} & " + " & ".join(cells) + r" \\"


def load_attested_pi05_omega() -> tuple[dict[str, float], dict[str, Any]]:
    require(ATTESTED_PI05_OMEGA.is_file(), "missing attested pi0.5 Omega-QVLA result")
    value = json.loads(ATTESTED_PI05_OMEGA.read_text(encoding="utf-8"))
    require(value.get("kind") == "experiment_owner_attested_result", "attested result kind drift")
    require(value.get("model") == "pi05", "attested result model drift")
    require(value.get("configuration") == "omega_qvla_w4a4", "attested result config drift")
    require(value.get("benchmark") == "LIBERO", "attested result benchmark drift")
    require(value.get("display_authorized") is True, "attested result is not display-authorized")
    metrics = value.get("metrics") or {}
    require(
        metrics
        == {"goal": 100.0, "spatial": 99.0, "object": 97.0, "long": 96.0, "average": 98.0},
        "attested pi0.5 Omega-QVLA metrics drift",
    )
    require(
        metrics["average"] == sum(metrics[suite] for suite in SUITES) / len(SUITES),
        "attested pi0.5 Omega-QVLA average drift",
    )
    return metrics, {
        "path": str(ATTESTED_PI05_OMEGA.relative_to(REPO_ROOT)),
        "sha256": sha256_file(ATTESTED_PI05_OMEGA),
        "kind": value["kind"],
        "claim_enabled": True,
    }


def load_static_memory() -> tuple[dict[str, Any], dict[str, Any]]:
    require(STATIC_MEMORY.is_file(), "missing Table 5 static-memory audit")
    value = json.loads(STATIC_MEMORY.read_text(encoding="utf-8"))
    require(value.get("kind") == "table5_static_component_storage_audit", "memory audit kind drift")
    require(value.get("display_unit") == "GiB", "memory display unit drift")
    for name, record in (value.get("sources") or {}).items():
        source_path = REPO_ROOT / record["path"]
        require(source_path.is_file(), f"missing memory source {name}: {source_path}")
        require(sha256_file(source_path) == record["sha256"], f"memory source SHA drift: {name}")

    expected = {
        "gr00t": {
            "fp16": 2_139_537_408,
            "quantvla_w4a8": 963_772_416,
            "uniform_w6": 1_190_264_832,
            "omega_qvla_w4a4": 642_940_928,
            "gdsq_vla_selector": 1_018_204_672,
        },
        "pi05": {
            "fp16": 4_416_602_112,
            "quantvla_w4a8": 1_490_466_816,
            "uniform_w6": 2_042_542_080,
            "omega_qvla_w4a4": 1_402_970_112,
            "gdsq_vla_selector": 1_919_942_656,
        },
    }
    models = value.get("models") or {}
    for model, rows in expected.items():
        require(set(models.get(model, {})) == set(rows), f"memory row set drift: {model}")
        for config, expected_bytes in rows.items():
            record = models[model][config]
            require(int(record["component_bytes"]) == expected_bytes, f"memory byte drift: {model}/{config}")
            per_suite = record.get("per_suite_component_bytes")
            if per_suite is not None:
                require(set(per_suite) == set(SUITES), f"memory suite set drift: {model}/{config}")
                require(
                    sum(int(per_suite[suite]) for suite in SUITES) / len(SUITES) == expected_bytes,
                    f"memory suite mean drift: {model}/{config}",
                )
            record["display_gib"] = f"{expected_bytes / 2**30:.3f}"
    return models, {
        "path": str(STATIC_MEMORY.relative_to(REPO_ROOT)),
        "sha256": sha256_file(STATIC_MEMORY),
        "scope": value["scope"],
        "display_unit": value["display_unit"],
        "aggregation": value["aggregation"],
        "sources": value["sources"],
    }


def local_results(registry: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    experiment = registry["experiments"]["libero_table2_five_config_local"]
    preregistration_id = experiment["preregistration"]
    prereg_record = registry["preregistrations"][preregistration_id]
    prereg_path = REPO_ROOT / prereg_record["path"]
    require(prereg_path.is_file(), f"missing LIBERO Table 2 preregistration: {prereg_path}")
    require(sha256_file(prereg_path) == prereg_record["sha256"], "LIBERO Table 2 preregistration SHA drift")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    require(prereg["protocol"]["expected_total_episodes"] == 4000, "Table 2 preregistered coverage drift")
    require(
        prereg["configurations"]
        == [
            "fp16",
            "quantvla_w4a8",
            "uniform_w6",
            "omega_qvla_w4a4",
            "gdsq_vla_selector",
        ],
        "Table 2 configuration set drift",
    )
    audit = {
        "status": experiment["status"],
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_sha256": prereg_record["sha256"],
        "summary": None,
        "claim_enabled": False,
    }
    if experiment.get("status") != "complete" or not experiment.get("summary"):
        return None, audit
    record = experiment["summary"]
    path = REPO_ROOT / record["path"]
    require(path.is_file(), f"missing local Omega-QVLA summary: {path}")
    require(sha256_file(path) == record["sha256"], "local Omega-QVLA summary SHA drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    coverage = value.get("coverage") or {}
    require(value.get("complete") is True and value.get("formal_result") is True, "local summary incomplete")
    require(coverage.get("expected_episodes") == coverage.get("observed_episodes") == 4000, "local coverage drift")
    require(not coverage.get("missing_cells") and not coverage.get("invalid_cells"), "local cells invalid")
    audit.update({"summary": str(path.relative_to(REPO_ROOT)), "summary_sha256": record["sha256"], "claim_enabled": True})
    return value, audit


def load_complete_cell(model: str, config: str, suite: str) -> tuple[float | None, dict[str, Any]]:
    """Release one cell only after exact 10-task x 10-trial validation."""
    # The paper-method row is produced by the benchmark-specific DyPAC
    # calibration pipeline.  Keep the preregistered internal row key for the
    # 4,000-episode matrix, but never reuse the invalidated pre-calibration
    # gdsq_vla_selector rollouts from Table-6 staging.
    result_config = "dypac_vla" if config == "gdsq_vla_selector" else config
    if config == "gdsq_vla_selector" and model == "gr00t":
        result_root = DYPAC_GR00T_RESULTS
    elif config == "gdsq_vla_selector":
        result_root = DYPAC_RESULTS
    else:
        result_root = TABLE6_RESULTS
    path = result_root / model / result_config / suite / "merged_summary.json"
    relative = str(path.relative_to(REPO_ROOT))
    if not path.is_file():
        return None, {
            "model": model,
            "config": config,
            "result_config": result_config,
            "suite": suite,
            "path": relative,
            "status": "pending",
            "claim_enabled": False,
        }

    value = json.loads(path.read_text(encoding="utf-8"))
    episode_rows = value.get("episode_summaries") or []
    task_rows = value.get("task_summaries") or []
    keys = [(int(row["task_id"]), int(row["trial_id"])) for row in episode_rows]
    expected = {(task_id, trial_id) for task_id in range(10) for trial_id in range(10, 20)}
    counts = Counter(keys)
    require(len(episode_rows) == 100, f"{model}/{config}/{suite}: episode row count drift")
    require(set(keys) == expected, f"{model}/{config}/{suite}: episode key coverage drift")
    require(all(count == 1 for count in counts.values()), f"{model}/{config}/{suite}: duplicate episode keys")
    require(len(task_rows) == 10, f"{model}/{config}/{suite}: task row count drift")
    require({int(row["task_id"]) for row in task_rows} == set(range(10)), f"{model}/{config}/{suite}: task IDs drift")
    require(all(int(row["episodes"]) == 10 for row in task_rows), f"{model}/{config}/{suite}: task coverage drift")
    require(int(value.get("num_trials_per_task", -1)) == 10, f"{model}/{config}/{suite}: trial count drift")
    require(int(value.get("total_episodes", -1)) == 100, f"{model}/{config}/{suite}: total episode count drift")
    if config == "gdsq_vla_selector" and model == "gr00t":
        require(value.get("policy_backend") == "groot_zmq", f"{model}/{config}/{suite}: backend drift")
        require(int(value.get("executed_replan_steps", -1)) == 5, f"{model}/{config}/{suite}: replan drift")
        require(value.get("paired_action_noise") is True, f"{model}/{config}/{suite}: paired-noise drift")
        require(value.get("action_noise_suite_key") == suite, f"{model}/{config}/{suite}: noise namespace drift")
        require(
            value.get("action_noise_generator") == "torch-cpu-normal-v1",
            f"{model}/{config}/{suite}: noise generator drift",
        )
    successes = sum(bool(row["success"]) for row in episode_rows)
    require(int(value.get("total_successes", -1)) == successes, f"{model}/{config}/{suite}: success count drift")
    success_rate = 100.0 * successes / 100.0
    return success_rate, {
        "model": model,
        "config": config,
        "result_config": result_config,
        "suite": suite,
        "path": relative,
        "sha256": sha256_file(path),
        "episodes": 100,
        "successes": successes,
        "success_rate_percent": success_rate,
        "status": "complete",
        "claim_enabled": True,
    }


def interim_results() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a cell-audited snapshot while preserving the 4,000-episode joint gate."""
    models: dict[str, Any] = {}
    artifacts = []
    completed_cells = 0
    observed_episodes = 0
    for model in MODELS:
        config_rows = {}
        for config in CONFIGS:
            metrics: dict[str, float] = {}
            for suite in SUITES:
                value, artifact = load_complete_cell(model, config, suite)
                artifacts.append(artifact)
                if value is not None:
                    metrics[suite] = value
                    completed_cells += 1
                    observed_episodes += 100
            if all(suite in metrics for suite in SUITES):
                metrics["average"] = sum(metrics[suite] for suite in SUITES) / len(SUITES)
            config_rows[config] = {"metrics": metrics}
        models[model] = {"configs": config_rows}
    return {"models": models}, {
        "release_mode": "interim_complete_cells_only",
        "joint_claim_enabled": completed_cells == len(MODELS) * len(CONFIGS) * len(SUITES),
        "coverage": {
            "expected_cells": len(MODELS) * len(CONFIGS) * len(SUITES),
            "complete_cells": completed_cells,
            "pending_cells": len(MODELS) * len(CONFIGS) * len(SUITES) - completed_cells,
            "expected_episodes": 4000,
            "observed_episodes": observed_episodes,
            "missing_episodes": 4000 - observed_episodes,
        },
        "cell_artifacts": artifacts,
    }


def build() -> tuple[str, str, dict[str, Any]]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    joint, joint_audit = local_results(registry)
    interim, interim_audit = interim_results()
    attested_pi05_omega, attested_pi05_omega_audit = load_attested_pi05_omega()
    static_memory, static_memory_audit = load_static_memory()
    local = joint if joint is not None else interim
    if joint is not None:
        # A completed joint summary must agree with every independently
        # validated cell before the table can switch from interim to formal.
        for model in MODELS:
            for config in CONFIGS:
                require(
                    joint["models"][model]["configs"][config]["metrics"]
                    == interim["models"][model]["configs"][config]["metrics"],
                    f"joint/interim metric drift: {model}/{config}",
                )
    coverage = interim_audit["coverage"]
    if joint is not None:
        coverage_note = (
            "The complete local matrix contains 4,000 episodes across 40/40 "
            "exact-valid cells."
        )
        release_note = (
            "Every local suite value covers exactly ten tasks and held-out "
            "initial states 10--19."
        )
    else:
        coverage_note = (
            f"The ongoing local matrix contains {coverage['observed_episodes']}/4,000 "
            f"episodes across {coverage['complete_cells']}/40 complete cells."
        )
        release_note = (
            "A local suite value is released only after exact coverage of ten tasks "
            "and held-out initial states 10--19. Running or absent cells remain pending."
        )
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{LIBERO comparison in success rate (\%). Local size is tightly packed static model-component storage.}",
        r"\label{tab:omega_qvla_libero}",
        r"\small",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Configuration & Goal $\uparrow$ & Spatial $\uparrow$ & Object $\uparrow$ & Long $\uparrow$ & Avg. $\uparrow$ & \shortstack{Size\\(GiB) $\downarrow$} \\",
        r"\midrule",
    ]
    audit_rows: dict[str, list[dict[str, Any]]] = {}
    labels = {
        "fp16": "FP16",
        "quantvla_w4a8": r"\quantvla W4A8",
        "uniform_w6": "Uniform W6",
        "omega_qvla_w4a4": r"$\Omega$-QVLA W4A4",
        "gdsq_vla_selector": r"\textbf{\method (Ours)}",
    }
    model_labels = {"gr00t": "GR00T N1.5", "pi05": r"$\pi_{0.5}$"}
    for model in MODELS:
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textbf{{{model_labels[model]}}}}} \\")
        audit_rows[model] = []
        for config in CONFIGS:
            if model == "pi05" and config == "gdsq_vla_selector":
                for external_config, external in EXTERNAL_PI05_ROWS.items():
                    external_label = {
                        "qvla_source_4bpw": r"QVLA (4.0 BPW)$^{\dagger}$",
                        "actquant_source_4bpw": r"ActQuant (4.0 BPW)$^{\dagger}$",
                    }[external_config]
                    lines.append(row(external_label, external["metrics"], metric(external["memory_gb"])))
                    audit_rows[model].append({
                        "config": external_config,
                        "evidence": "source-reported ActQuant v3 Table 1",
                        "metrics": external["metrics"],
                        "memory_gb": external["memory_gb"],
                        "local_reproduction": False,
                        "included_in_local_4000_episode_gate": False,
                        "claim_enabled": True,
                    })
            local_values = local["models"][model]["configs"][config]["metrics"]
            evidence = (
                "benchmark-specific DyPAC-VLA reproduction"
                if config == "gdsq_vla_selector"
                else "local reproduction"
            )
            if model == "pi05" and config == "omega_qvla_w4a4":
                local_values = dict(attested_pi05_omega)
                evidence = "experiment-owner attested result"
            memory = static_memory[model][config]
            lines.append(row(labels[config], local_values, memory["display_gib"]))
            complete_suites = [suite for suite in SUITES if suite in local_values]
            audit_rows[model].append({
                "config": config,
                "evidence": evidence,
                "metrics": local_values,
                "static_memory": memory,
                "complete_suites": complete_suites,
                "row_average_enabled": "average" in local_values,
                "claim_enabled": bool(complete_suites),
            })
        if model == "gr00t":
            lines.append(r"\midrule")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        rf"\parbox{{0.99\textwidth}}{{\footnotesize {coverage_note} {release_note} The corrected GR00T ours row uses five executed actions per replan and paired suite-keyed action noise. Local size includes packed weights and quantization metadata. For suite-specific masks, it is the four-suite mean. $^{{\dagger}}$The $\pi_{{0.5}}$ QVLA and ActQuant entries are the source-reported 4.0 Vision+LLM-BPW rows from ActQuant~\cite{{akbari2026actquant}}; their 2.7-GB values follow that source's memory definition.}}",
        r"\end{table}",
        "",
    ])
    compact_index = {
        model: {
            item["config"]: item
            for item in audit_rows[model]
            if item["config"] in CONFIGS
        }
        for model in MODELS
    }
    compact_labels = {
        "fp16": "FP16",
        "quantvla_w4a8": r"\quantvla W4A8",
        "uniform_w6": "Uniform W6",
        "omega_qvla_w4a4": r"$\Omega$-QVLA W4A4$^{\dagger}$",
        "gdsq_vla_selector": r"\textbf{\method (Ours)}",
    }
    compact_lines = [
        "% AUTO-GENERATED by scripts/tools/render_omega_qvla_libero_table.py; DO NOT EDIT.",
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Cross-benchmark LIBERO transfer. Cells report four-suite mean success rate (\%) and tightly packed static component size.}",
        r"\label{tab:libero_transfer}",
        r"\small",
        r"\setlength{\tabcolsep}{7.5pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{GR00T N1.5}} & \multicolumn{2}{c}{\textbf{$\pi_{0.5}$}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"Configuration & Avg. SR $\uparrow$ & Size (GiB) $\downarrow$ & Avg. SR $\uparrow$ & Size (GiB) $\downarrow$ \\",
        r"\midrule",
    ]
    for config in CONFIGS:
        gr00t = compact_index["gr00t"][config]
        pi05 = compact_index["pi05"][config]
        cells = [
            metric(gr00t["metrics"].get("average")),
            gr00t["static_memory"]["display_gib"],
            metric(pi05["metrics"].get("average")),
            pi05["static_memory"]["display_gib"],
        ]
        if config == "gdsq_vla_selector":
            cells = [rf"\textbf{{{cell}}}" for cell in cells]
        compact_lines.append(
            compact_labels[config] + " & " + " & ".join(cells) + r" \\"
        )
    compact_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\parbox{0.99\linewidth}{\footnotesize The joint local gate covers 4,000 episodes across 40/40 exact-valid suite cells. $^{\dagger}$The displayed $\pi_{0.5}$ $\Omega$-QVLA success value is experiment-owner attested. Ours size is the arithmetic mean of four suite-specific static plans. Appendix~\ref{sec:libero_comparison} reports the full suite breakdown and protocol.}",
        r"\end{table}",
        "",
    ])
    audit = {
        "schema_version": 1,
        "kind": "omega_qvla_libero_table_audit",
        "source": str(SOURCE.relative_to(REPO_ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "paper_pdf_sha256": source["paper"]["pdf_sha256"],
        "official_code_commit": source["official_release"]["code"]["commit"],
        "external_source": {
            "paper": "ActQuant: Sub-4-bit Action-Guided Quantization for Vision-Language-Action Models",
            "arxiv": "2605.24011v3",
            "url": "https://arxiv.org/abs/2605.24011v3",
            "table": 1,
            "user_supplied_image_sha256": "279b79fb5fc2b8bce6bc82db89cdd9cd1404289d73c52fde9bfcf137035179a5",
            "scope": "pi0.5 4.0 Vision+LLM BPW rows only",
            "qvla_reproduction_note": "ActQuant authors' reproduction under the shared 60-episode calibration budget",
        },
        "local_reproduction": {
            "joint_gate": joint_audit,
            "interim_snapshot": interim_audit,
            "display_scope": "formal_joint_summary" if joint is not None else "interim_complete_cells_only",
        },
        "attested_pi05_omega": attested_pi05_omega_audit,
        "static_memory": static_memory_audit,
        "rows": audit_rows,
    }
    return "\n".join(lines), "\n".join(compact_lines), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tex, compact_tex, audit = build()
    rendered_audit = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.check:
        require(OUTPUT.is_file() and OUTPUT.read_text(encoding="utf-8") == tex, "Omega-QVLA table is stale")
        require(
            COMPACT_OUTPUT.is_file()
            and COMPACT_OUTPUT.read_text(encoding="utf-8") == compact_tex,
            "compact LIBERO transfer table is stale",
        )
        require(AUDIT.is_file() and AUDIT.read_text(encoding="utf-8") == rendered_audit, "Omega-QVLA table audit is stale")
        print("Omega-QVLA LIBERO tables are current")
        return
    OUTPUT.write_text(tex, encoding="utf-8")
    COMPACT_OUTPUT.write_text(compact_tex, encoding="utf-8")
    AUDIT.write_text(rendered_audit, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"wrote {COMPACT_OUTPUT}")
    print(f"wrote {AUDIT}")


if __name__ == "__main__":
    main()
