#!/usr/bin/env python3
"""Render the Omega-QVLA LIBERO comparison from audited evidence only."""

from __future__ import annotations

import argparse
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
AUDIT = PAPER_DIR / "tables/omega_qvla_libero.audit.json"
SUITES = ("goal", "spatial", "object", "long")
PENDING = r"\textit{pending}"


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


def row(config: str, values: dict[str, float] | None) -> str:
    cells = [metric(values.get(name) if values else None) for name in (*SUITES, "average")]
    return rf"\quad {config} & " + " & ".join(cells) + r" \\"


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


def build() -> tuple[str, dict[str, Any]]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    local, local_audit = local_results(registry)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{LIBERO local comparison under the $\Omega$-QVLA Table~1 protocol (success rate, \%).}",
        r"\label{tab:omega_qvla_libero}",
        r"\small",
        r"\setlength{\tabcolsep}{5.5pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Configuration & Goal $\uparrow$ & Spatial $\uparrow$ & Object $\uparrow$ & Long $\uparrow$ & Avg. $\uparrow$ \\",
        r"\midrule",
    ]
    audit_rows: dict[str, list[dict[str, Any]]] = {}
    labels = {
        "fp16": "FP16",
        "quantvla_w4a8": r"\quantvla W4A8",
        "uniform_w6": "Uniform W6",
        "omega_qvla_w4a4": r"$\Omega$-QVLA W4A4",
        "gdsq_vla_selector": r"\textbf{\textsc{GDSQ-VLA}(Ours)}",
    }
    model_labels = {"gr00t": "GR00T N1.5", "pi05": r"$\pi_{0.5}$"}
    for model in ("gr00t", "pi05"):
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{model_labels[model]}}}}} \\")
        audit_rows[model] = []
        for config in (
            "fp16",
            "quantvla_w4a8",
            "uniform_w6",
            "omega_qvla_w4a4",
            "gdsq_vla_selector",
        ):
            local_values = None
            if local is not None:
                local_values = local["models"][model]["configs"][config]["metrics"]
                require(local_values is not None, f"missing complete local row: {model}/{config}")
            lines.append(row(labels[config], local_values))
            audit_rows[model].append({"config": config, "evidence": "local reproduction", "metrics": local_values, "claim_enabled": local_values is not None})
        if model == "gr00t":
            lines.append(r"\midrule")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\parbox{0.99\textwidth}{\footnotesize Four official suites, ten tasks per suite, and ten held-out trials per task. All cells remain pending until the joint five-configuration manifest reaches exactly 4,000 valid local episodes; no author-reported values are imported. The official $\Omega$-QVLA W4A4 packs are pinned. QuantVLA, Uniform W6, and GDSQ-VLA are independently calibrated on LIBERO; the GDSQ selector is model-specific, with no RoboCasa artifact reuse or test-set retuning.}",
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
        "local_reproduction": local_audit,
        "rows": audit_rows,
    }
    return "\n".join(lines), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tex, audit = build()
    rendered_audit = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.check:
        require(OUTPUT.is_file() and OUTPUT.read_text(encoding="utf-8") == tex, "Omega-QVLA table is stale")
        require(AUDIT.is_file() and AUDIT.read_text(encoding="utf-8") == rendered_audit, "Omega-QVLA table audit is stale")
        print("Omega-QVLA LIBERO table is current")
        return
    OUTPUT.write_text(tex, encoding="utf-8")
    AUDIT.write_text(rendered_audit, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"wrote {AUDIT}")


if __name__ == "__main__":
    main()
