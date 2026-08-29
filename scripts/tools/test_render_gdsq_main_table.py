from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts/tools/render_gdsq_main_table.py"
SPEC = importlib.util.spec_from_file_location("render_gdsq_main_table", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_generated_table_is_current_and_has_no_unsupported_ohb_row() -> None:
    tex, audit = MODULE.build(
        REPO_ROOT / "docs/gdsq_vla_iclr2027/experiment_registry.json"
    )
    assert tex == (REPO_ROOT / "docs/gdsq_vla_iclr2027/tables/main_results.tex").read_text()
    assert "27.6" not in tex
    assert r"\method + OHB" not in tex
    assert "pi05_runtime_selector_official50" in {
        row["evidence"] for row in audit["rows"]["pi05"]
    }
    assert tex.count(r"$\Omega$-QVLA W4A4") == 2
    assert r"\quantvla W4A8 + ATM/OHB (secondary)" not in tex
    assert tex.count(r"\textsc{GDSQ-VLA}(Ours)") == 2
    assert tex.count("Uniform W6") >= 3  # two rows plus the table note
    assert len(audit["rows"]["gr00t"]) == 5
    assert len(audit["rows"]["pi05"]) == 5
    for forbidden in ("Search-matched random", "Action-only allocator", "static-mask ablation"):
        assert forbidden not in tex


def test_main_table_uses_iclr_readable_booktabs_style() -> None:
    tex, _audit = MODULE.build(
        REPO_ROOT / "docs/gdsq_vla_iclr2027/experiment_registry.json"
    )
    assert r"\caption{" in tex and tex.index(r"\caption{") < tex.index(r"\label{")
    assert r"\small" in tex
    assert r"\scriptsize" not in tex
    assert r"\resizebox" not in tex
    assert r"\begin{tabular}{@{}lcrrrrrr@{}}" in tex
    assert "|" not in tex
    for rule in (r"\toprule", r"\midrule", r"\bottomrule"):
        assert rule in tex


def test_every_numeric_result_row_has_enabled_complete_evidence() -> None:
    _tex, audit = MODULE.build(
        REPO_ROOT / "docs/gdsq_vla_iclr2027/experiment_registry.json"
    )
    for model_rows in audit["rows"].values():
        for row in model_rows:
            if row["metrics"] is None:
                assert row["paper_claim_enabled"] is False
                assert row["status"] in {
                    "pending",
                    "planned",
                    "preregistered",
                    "preregistered_post_week1",
                    "queued",
                    "queued_deferred",
                    "running",
                }
            else:
                assert row["status"] in {
                    "complete",
                    "complete_by_equivalence_reuse",
                    "running_partial_taskset_complete",
                }
                if row["status"] == "running_partial_taskset_complete":
                    assert row["paper_claim_enabled"] is False
                    assert row["claim_enabled_cells"] == {
                        "atomic": True,
                        "composite_seen": True,
                        "composite_unseen": False,
                        "mean": False,
                        "storage": True,
                        "compression": True,
                    }

    for model in ("gr00t", "pi05"):
        omega_rows = [
            row for row in audit["rows"][model]
            if row["evidence"] == "omega_qvla_robocasa365"
        ]
        assert len(omega_rows) == 1
        if model == "gr00t":
            assert omega_rows[0]["paper_claim_enabled"] is True
            assert {
                key: round(value, 3)
                for key, value in omega_rows[0]["metrics"].items()
            } == {
                "atomic": 60.111,
                "composite_seen": 25.875,
                "composite_unseen": 27.5,
                "mean": 38.72,
            }
            assert omega_rows[0]["storage"] == "0.599"
            assert omega_rows[0]["compression"] == r"3.33$\times$"
        else:
            assert omega_rows[0]["paper_claim_enabled"] is True
            assert omega_rows[0]["metrics"] == {
                "atomic": 49.55555555555555,
                "composite_seen": 10.375,
                "composite_unseen": 1.0,
                "mean": 21.48,
            }
            assert omega_rows[0]["storage"] == "1.307"
            assert omega_rows[0]["compression"] == r"3.27$\times$"
