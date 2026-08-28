from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from prune_gr00t_main_to_budget import prune_main_plan  # noqa: E402


def score_doc(values: list[float]) -> dict:
    sequences = [
        {
            "task": "OpenDrawer" if index < 2 else "OpenCabinet",
            "seed": index % 2,
            "components": {"pose": 0.0, "stitch": 0.0, "grip": 0.0},
        }
        for index in range(len(values))
    ]
    return {
        "d_func": sum(values) / len(values),
        "d_pac": sum(values) / len(values),
        "d_func_summary": {"per_sequence": values},
        "d_pac_summary": {"per_sequence": values, "sequences": sequences},
    }


def main_plan(protected: set[str]) -> dict:
    layers = {}
    for name in ("a", "b", "c"):
        if name in protected:
            layers[name] = {"bits": None, "skip": True}
        else:
            layers[name] = {"bits": 4, "group": 64, "skip": False}
    return {"layers": layers, "meta": {}}


BYTE_ROWS = {
    "a": {"fp16_bytes": 200, "w4_bytes": 100, "extra_fp16_bytes": 100},
    "b": {"fp16_bytes": 300, "w4_bytes": 150, "extra_fp16_bytes": 150},
    "c": {"fp16_bytes": 400, "w4_bytes": 200, "extra_fp16_bytes": 200},
}


def flip_manifest() -> list[dict]:
    return [
        {"candidate_id": f"flip_{name}", "flip": {"layer": name, "from": "w4", "to": "fp16"}}
        for name in ("a", "b", "c")
    ]


def flip_scores() -> dict:
    baseline = score_doc([1.0] * 4)
    return {
        "context_base": baseline,
        "flip_a": score_doc([0.1] * 4),  # restore helps a lot -> keep
        "flip_b": score_doc([0.9] * 4),  # least valuable -> prune first
        "flip_c": score_doc([0.5] * 4),
    }


def test_already_within_budget_is_noop() -> None:
    plan, pruned, total = prune_main_plan(
        main_plan=main_plan({"a"}),
        byte_rows=BYTE_ROWS,
        flip_scores=flip_scores(),
        flip_manifest=flip_manifest(),
        budget=1_000_000,
    )
    assert pruned == []
    assert total == 450 + 100
    assert plan["layers"]["a"]["skip"] is True


def test_prunes_least_valuable_protection_first() -> None:
    # all_w4 = 450; protected a+b -> total = 700. Budget 600 forces one flip.
    plan, pruned, total = prune_main_plan(
        main_plan=main_plan({"a", "b"}),
        byte_rows=BYTE_ROWS,
        flip_scores=flip_scores(),
        flip_manifest=flip_manifest(),
        budget=600,
    )
    assert pruned == ["b"]
    assert plan["layers"]["b"]["skip"] is False
    assert plan["layers"]["a"]["skip"] is True
    assert total == 550


def test_missing_flip_score_raises() -> None:
    manifest = [
        {"candidate_id": "flip_a", "flip": {"layer": "a", "from": "w4", "to": "fp16"}}
    ]
    with pytest.raises(ValueError, match="no counterfactual score"):
        prune_main_plan(
            main_plan=main_plan({"a", "b"}),
            byte_rows=BYTE_ROWS,
            flip_scores=flip_scores(),
            flip_manifest=manifest,
            budget=600,
        )