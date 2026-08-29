from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGG = load("aggregate_omega_qvla_table1", "scripts/tools/aggregate_omega_qvla_table1.py")
RENDER = load("render_omega_qvla_libero_table", "scripts/tools/render_omega_qvla_libero_table.py")


def test_table_contains_only_pending_local_rows_before_reproduction() -> None:
    tex, audit = RENDER.build()
    assert tex == (REPO_ROOT / "docs/gdsq_vla_iclr2027/tables/omega_qvla_libero.tex").read_text()
    assert audit["paper_pdf_sha256"] == "d0c2ede8a71a497e423c31a2431b0166592f61093acfcfeb6be1be62a7f716e7"
    assert "v1 reported" not in tex
    assert "91.0 & 86.0 & 92.0 & 82.0 & 87.8" not in tex
    assert "100.0 & 99.0 & 97.0 & 96.0 & 98.0" not in tex
    assert tex.count(RENDER.PENDING) == 50
    assert audit["local_reproduction"]["claim_enabled"] is False
    assert sum(len(rows) for rows in audit["rows"].values()) == 10
    assert tex.count(r"\quantvla W4A8") == 2
    assert tex.count("Uniform W6") >= 3  # two rows plus the table note
    assert tex.count(r"\textsc{GDSQ-VLA}(Ours)") == 2
    assert "Evidence &" not in tex
    assert "3,200" not in tex
    assert "4,000" in tex


def test_libero_table_uses_iclr_readable_booktabs_style() -> None:
    tex, _audit = RENDER.build()
    assert tex.index(r"\caption{") < tex.index(r"\label{")
    assert r"\small" in tex
    assert r"\scriptsize" not in tex
    assert r"\resizebox" not in tex
    assert r"\begin{tabular}{@{}lrrrrr@{}}" in tex
    assert "|" not in tex
    for rule in (r"\toprule", r"\midrule", r"\bottomrule"):
        assert rule in tex


def test_strict_aggregator_accepts_exact_table1_coverage(tmp_path: Path) -> None:
    root = tmp_path / "omega"
    for model in AGG.MODELS:
        for config in AGG.CONFIGS:
            for suite in AGG.SUITES:
                output = root / "results" / model / config / suite / "merged_summary.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                rows = [
                    {
                        "task_id": task_id,
                        "episodes": 10,
                        "successes": task_id % 3,
                        "success_rate": (task_id % 3) / 10.0,
                    }
                    for task_id in range(10)
                ]
                successes = sum(row["successes"] for row in rows)
                output.write_text(json.dumps({
                    "task_suite_name": AGG.TASK_SUITE_NAMES[suite],
                    "num_trials_per_task": 10,
                    "total_episodes": 100,
                    "total_successes": successes,
                    "total_success_rate": successes / 100.0,
                    "task_summaries": rows,
                }))
    value = AGG.aggregate(root)
    assert value["complete"] is True
    assert value["coverage"]["observed_episodes"] == 1600
    assert value["coverage"]["missing_cells"] == []


def test_aggregator_rejects_partial_cell(tmp_path: Path) -> None:
    value = AGG.aggregate(tmp_path / "omega")
    assert value["complete"] is False
    assert len(value["coverage"]["missing_cells"]) == 16
