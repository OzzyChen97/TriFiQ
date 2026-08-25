from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("render_gdsq_component_ablation.py")
SPEC = importlib.util.spec_from_file_location("render_gdsq_component_ablation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_component_table_is_generated_and_claim_gated() -> None:
    tex, audit = MODULE.render()
    assert tex == MODULE.TEX.read_text(encoding="utf-8")
    assert "Full GDSQ-VLA & 65.9" in tex
    assert tex.count(MODULE.PENDING) == 20
    assert audit["status"] == "preregistered"
    assert audit["pending_cells_claim_disabled"] is True
    assert audit["ablation_statistics"] is None
