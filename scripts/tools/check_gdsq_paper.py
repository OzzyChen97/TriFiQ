#!/usr/bin/env python3
"""Fail when the GDSQ-VLA PDF violates source, warning, or page gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "docs/gdsq_vla_cvpr2026"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def label_page(aux_text: str, label: str) -> int:
    pattern = re.compile(
        rf"\\newlabel\{{{re.escape(label)}\}}\{{\{{[^}}]*\}}\{{(\d+)\}}"
    )
    match = pattern.search(aux_text)
    require(match is not None, f"missing page label in aux: {label}")
    return int(match.group(1))


def pdf_pages(path: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    match = re.search(r"^Pages:\s+(\d+)\s*$", output, flags=re.MULTILINE)
    require(match is not None, f"pdfinfo did not report pages for {path}")
    return int(match.group(1))


def audit(paper_dir: Path, main_page_limit: int) -> dict[str, Any]:
    build_dir = paper_dir / ".build"
    pdf_path = paper_dir / "main.pdf"
    build_pdf = build_dir / "main.pdf"
    aux_path = build_dir / "main.aux"
    log_path = build_dir / "main.log"
    for path in (pdf_path, build_pdf, aux_path, log_path):
        require(path.is_file(), f"missing paper build artifact: {path}")

    aux_text = aux_path.read_text(encoding="utf-8", errors="replace")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    main_end_page = label_page(aux_text, "sec:main-paper-end")
    supplement_page = label_page(aux_text, "sec:supplement")
    total_pages = pdf_pages(pdf_path)

    forbidden_warnings = {
        "overfull box": r"Overfull \\hbox",
        "undefined citation": r"Citation [`'][^\n]* undefined",
        "undefined reference": r"Reference [`'][^\n]* undefined",
        "undefined references summary": r"There were undefined references",
        "undefined citations summary": r"There were undefined citations",
    }
    found = [
        label
        for label, pattern in forbidden_warnings.items()
        if re.search(pattern, log_text, flags=re.IGNORECASE)
    ]
    require(not found, f"LaTeX warning gate failed: {found}")
    require(main_end_page <= main_page_limit, (
        f"main paper ends on page {main_end_page}, limit is {main_page_limit}"
    ))
    require(supplement_page > main_end_page, "supplement overlaps the main paper")
    require(total_pages >= supplement_page, "supplement page exceeds PDF length")
    require(sha256_file(pdf_path) == sha256_file(build_pdf), "copied PDF is stale")

    table_files = sorted((paper_dir / "tables").glob("*.tex"))
    checked_tables = []
    for table_path in table_files:
        table_text = table_path.read_text(encoding="utf-8")
        if r"\begin{tabular}" not in table_text:
            continue
        relative = str(table_path.relative_to(paper_dir))
        require(r"\caption{" in table_text, f"{relative}: missing caption")
        require(r"\label{" in table_text, f"{relative}: missing label")
        require(
            table_text.index(r"\caption{") < table_text.index(r"\label{"),
            f"{relative}: CVPR table caption must precede its label",
        )
        require(
            r"\small" in table_text or r"\footnotesize" in table_text,
            f"{relative}: missing readable table body size",
        )
        require(r"\scriptsize" not in table_text, f"{relative}: 7pt scriptsize is forbidden")
        require(r"\tiny" not in table_text, f"{relative}: tiny table text is forbidden")
        require(r"\resizebox" not in table_text, f"{relative}: scaled tables are forbidden")
        for command in (r"\toprule", r"\midrule", r"\bottomrule"):
            require(command in table_text, f"{relative}: missing booktabs rule {command}")
        tabular_lines = [
            line for line in table_text.splitlines() if r"\begin{tabular}" in line
        ]
        require(
            tabular_lines and all("|" not in line for line in tabular_lines),
            f"{relative}: vertical table rules are forbidden",
        )
        checked_tables.append(relative)

    return {
        "schema_version": 1,
        "kind": "gdsq_vla_latex_gate",
        "valid": True,
        "main_page_limit": main_page_limit,
        "main_end_page": main_end_page,
        "supplement_start_page": supplement_page,
        "total_pdf_pages": total_pages,
        "pdf": str(pdf_path.relative_to(REPO_ROOT)),
        "pdf_sha256": sha256_file(pdf_path),
        "forbidden_warnings": found,
        "table_style": {
            "valid": True,
            "checked_tables": checked_tables,
            "minimum_body_size": "footnotesize (8pt)",
            "scaled_tables_allowed": False,
            "vertical_rules_allowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", default=str(PAPER_DIR))
    parser.add_argument("--main-page-limit", type=int, default=8)
    parser.add_argument("--report")
    args = parser.parse_args()
    result = audit(Path(args.paper_dir).resolve(), args.main_page_limit)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        report = Path(args.report).resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
