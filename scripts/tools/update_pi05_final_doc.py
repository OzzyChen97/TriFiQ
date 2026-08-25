#!/usr/bin/env python3
"""Atomically replace the audited Table-1 status block in the π0.5 final doc."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


BEGIN = "<!-- PI05_TABLE1_FINAL_BEGIN -->"
END = "<!-- PI05_TABLE1_FINAL_END -->"
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def replacement_block(
    summary: dict[str, Any], audit: dict[str, Any], report_path: Path
) -> str:
    require(summary.get("complete") is True, "summary is not complete")
    require(audit.get("complete") is True, "completion audit is not complete")
    require(audit["statistics"]["comparisons_ready"] is True, "comparisons are not ready")
    require(summary["manifest_sha256"] == audit["manifest_sha256"], "manifest mismatch")
    for config in CONFIG_ORDER:
        require(summary["configs"][config]["completed_episodes"] == 2500, f"incomplete {config}")
    require(report_path.is_file(), f"audited report is missing: {report_path}")

    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    lines = [
        BEGIN,
        "## 正式 Table 1 完成状态",
        "",
        f"状态：**complete and audited (`10,000/10,000`)**。生成时间：`{generated}`。",
        "",
        "| Configuration | Atomic-Seen | Composite-Seen | Composite-Unseen | Held-out 46 | 50-task Macro |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIG_ORDER:
        row = summary["configs"][config]
        task_sets = row["task_set_macro_sr"]
        lines.append(
            f"| {LABELS[config]} | {pct(task_sets['atomic_seen'])} | "
            f"{pct(task_sets['composite_seen'])} | {pct(task_sets['composite_unseen'])} | "
            f"{pct(row['heldout46_task_macro_sr'])} | {pct(row['task_macro_sr'])} |"
        )
    lines += [
        "",
        f"- Immutable manifest SHA256：`{audit['manifest_sha256']}`",
        f"- Frozen aggregator SHA256：`{audit['frozen_aggregator_sha256']}`",
        f"- Completion audit tool SHA256：`{audit['audit_tool_sha256']}`",
        f"- Completion audit output SHA256：`{sha256_file(Path(audit['_output_path']))}`",
        f"- Audited final report：`{report_path}`",
        f"- Audited final report SHA256：`{sha256_file(report_path)}`",
        "",
        "完整 per-task SR、配对 CI、置换检验、Holm、McNemar、步数与效率见 audited final report。",
        END,
    ]
    return "\n".join(lines)


def replace_block(text: str, block: str) -> str:
    require(text.count(BEGIN) == 1, "final-doc begin marker count is not one")
    require(text.count(END) == 1, "final-doc end marker count is not one")
    start = text.index(BEGIN)
    finish = text.index(END, start) + len(END)
    require(start < finish, "final-doc markers are reversed")
    return text[:start] + block + text[finish:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    doc_path = Path(args.doc).resolve()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    audit_path = Path(args.audit).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["_output_path"] = str(audit_path)
    report_path = Path(args.report).resolve()
    original = doc_path.read_text(encoding="utf-8")
    updated = replace_block(original, replacement_block(summary, audit, report_path))
    temporary = doc_path.with_name(f".{doc_path.name}.tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(doc_path)
    print(f"updated audited Table-1 block: {doc_path}")


if __name__ == "__main__":
    main()
