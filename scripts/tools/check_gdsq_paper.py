#!/usr/bin/env python3
"""Fail when the DyPAC-VLA ICLR 2027 PDF violates submission gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "docs/gdsq_vla_iclr2027"
OFFICIAL_STYLE_SHA256 = "797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6"
OFFICIAL_BST_SHA256 = "2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5"


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


def pdf_page_size(path: Path) -> str:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    match = re.search(r"^Page size:\s+(.+?)\s*$", output, flags=re.MULTILINE)
    require(match is not None, f"pdfinfo did not report page size for {path}")
    return match.group(1)


def audit(paper_dir: Path, main_page_minimum: int, main_page_limit: int) -> dict[str, Any]:
    build_dir = paper_dir / ".build"
    pdf_path = paper_dir / "main.pdf"
    build_pdf = build_dir / "main.pdf"
    aux_path = build_dir / "main.aux"
    log_path = build_dir / "main.log"
    for path in (pdf_path, build_pdf, aux_path, log_path):
        require(path.is_file(), f"missing paper build artifact: {path}")

    main_tex = paper_dir / "main.tex"
    style_path = paper_dir / "iclr2027_conference.sty"
    bst_path = paper_dir / "iclr2027_conference.bst"
    for path in (main_tex, style_path, bst_path):
        require(path.is_file(), f"missing ICLR 2027 source asset: {path}")
    main_text = main_tex.read_text(encoding="utf-8")
    uncommented_main = "\n".join(
        line for line in main_text.splitlines() if not line.lstrip().startswith("%")
    )
    require(r"\usepackage{iclr2027_conference,times}" in main_text, "official ICLR package is not loaded")
    require(r"\documentclass{article}" in main_text, "ICLR article document class drift")
    require(r"\iclrfinalcopy" not in uncommented_main, "anonymous review build enables iclrfinalcopy")
    require(r"\bibliographystyle{iclr2027_conference}" in main_text, "official ICLR bibliography style is not loaded")
    require(r"\input{sections/6_statements}" in main_text, "required ICLR statements are not included")
    require(
        main_text.index(r"\input{sections/6_statements}")
        < main_text.index(r"\bibliography{main}")
        < main_text.index(r"\appendix"),
        "ICLR statements, references, and appendix are out of order",
    )
    require(sha256_file(style_path) == OFFICIAL_STYLE_SHA256, "official ICLR style SHA drift")
    require(sha256_file(bst_path) == OFFICIAL_BST_SHA256, "official ICLR BST SHA drift")
    require(not (paper_dir / "cvpr.sty").exists(), "obsolete cvpr.sty remains in paper tree")
    require(not (paper_dir / "ieeenat_fullname.bst").exists(), "obsolete CVPR BST remains in paper tree")

    statements_text = (paper_dir / "sections/6_statements.tex").read_text(encoding="utf-8")
    require(r"\subsection*{AI Use Statement}" in statements_text, "required AI use statement is missing")
    require(r"\subsection*{Reproducibility Statement}" in statements_text, "reproducibility statement is missing")

    aux_text = aux_path.read_text(encoding="utf-8", errors="replace")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    main_end_page = label_page(aux_text, "sec:main-paper-end")
    ai_statement_page = label_page(aux_text, "sec:ai-use-statement")
    reproducibility_statement_page = label_page(aux_text, "sec:reproducibility-statement")
    supplement_page = label_page(aux_text, "sec:supplement")
    total_pages = pdf_pages(pdf_path)
    page_size = pdf_page_size(pdf_path)

    forbidden_warnings = {
        "overfull box": r"Overfull \\hbox",
        "overfull vertical box": r"Overfull \\vbox",
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
    require(main_end_page >= main_page_minimum, (
        f"main paper ends on page {main_end_page}, minimum is {main_page_minimum}"
    ))
    require(main_end_page <= main_page_limit, (
        f"main paper ends on page {main_end_page}, limit is {main_page_limit}"
    ))
    require(ai_statement_page >= main_end_page, "AI statement precedes the main-text end")
    require(reproducibility_statement_page >= ai_statement_page, "reproducibility statement precedes AI statement")
    require(supplement_page > main_end_page, "supplement overlaps the main paper")
    require(total_pages >= supplement_page, "supplement page exceeds PDF length")
    require("612 x 792 pts" in page_size and "letter" in page_size.lower(), f"PDF is not US Letter: {page_size}")
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
            f"{relative}: ICLR table caption must precede its label",
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
        "schema_version": 2,
        "kind": "dypac_vla_iclr2027_latex_gate",
        "valid": True,
        "venue": "ICLR 2027",
        "anonymous_review": True,
        "main_page_minimum": main_page_minimum,
        "main_page_limit": main_page_limit,
        "main_end_page": main_end_page,
        "ai_use_statement_page": ai_statement_page,
        "reproducibility_statement_page": reproducibility_statement_page,
        "supplement_start_page": supplement_page,
        "total_pdf_pages": total_pages,
        "pdf_page_size": page_size,
        "pdf": str(pdf_path.relative_to(REPO_ROOT)),
        "pdf_sha256": sha256_file(pdf_path),
        "forbidden_warnings": found,
        "table_style": {
            "valid": True,
            "checked_tables": checked_tables,
            "minimum_body_size": "footnotesize (9pt in the ICLR 2027 style)",
            "scaled_tables_allowed": False,
            "vertical_rules_allowed": False,
        },
        "official_template": {
            "style": str(style_path.relative_to(REPO_ROOT)),
            "style_sha256": sha256_file(style_path),
            "bibliography_style": str(bst_path.relative_to(REPO_ROOT)),
            "bibliography_style_sha256": sha256_file(bst_path),
            "unmodified": True,
        },
        "statements": {
            "ai_use_required_and_present": True,
            "reproducibility_recommended_and_present": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", default=str(PAPER_DIR))
    parser.add_argument("--main-page-minimum", type=int, default=9)
    parser.add_argument("--main-page-limit", type=int, default=9)
    parser.add_argument("--report")
    args = parser.parse_args()
    result = audit(Path(args.paper_dir).resolve(), args.main_page_minimum, args.main_page_limit)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        report = Path(args.report).resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
