from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from quantvla_full_context import BudgetItem, v2_coordinate_candidates  # noqa: E402


def item(name: str, extra: int, d_pac: float, d_func: float = 0.0) -> BudgetItem:
    return BudgetItem(
        name=name,
        extra_bytes=extra,
        benefit_d_func=d_func,
        benefit_d_pac=d_pac,
    )


NAMES = [
    "backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj",
    "backbone.eagle_model.language_model.model.layers.0.self_attn.k_proj",
    "backbone.eagle_model.language_model.model.layers.0.self_attn.v_proj",
    "backbone.eagle_model.language_model.model.layers.0.self_attn.o_proj",
    "backbone.eagle_model.language_model.model.layers.1.mlp.gate_proj",
    "backbone.eagle_model.language_model.model.layers.1.mlp.up_proj",
    "backbone.eagle_model.language_model.model.layers.1.mlp.down_proj",
    "action_head.model.transformer_blocks.0.ff.net.0.proj",
    "action_head.model.transformer_blocks.0.ff.net.2",
]


def standard_items() -> list[BudgetItem]:
    return [item(name, extra=100, d_pac=float(index)) for index, name in enumerate(NAMES)]


def mask_list(candidates):
    return [tuple(sorted(mask)) for _, mask, _ in candidates]


def test_single_and_pair_and_groups_present() -> None:
    candidates = v2_coordinate_candidates(standard_items(), capacity=1000)
    ids = [identifier for identifier, _, _ in candidates]
    assert "single_best" in ids
    assert "two_best" in ids
    assert "attention_0" in ids
    assert "mlp_1" in ids
    # The FF pair may be deduplicated by two_best (same mask); require that
    # the pair appears under some candidate.
    all_names = {name for _, mask, _ in candidates for name in mask}
    assert {NAMES[7], NAMES[8]} <= all_names
    assert "dp_half_budget" in ids
    assert any(identifier.startswith("dp_full_lambda") for identifier in ids)
    assert len(candidates) <= 8


def test_single_best_is_highest_benefit_layer() -> None:
    candidates = v2_coordinate_candidates(standard_items(), capacity=1000)
    single = next(c for c in candidates if c[0] == "single_best")
    assert single[1] == (NAMES[-1],)  # highest d_pac benefit


def test_byte_infeasible_candidates_are_skipped() -> None:
    items = standard_items()
    items[0] = item(NAMES[0], extra=10_000, d_pac=99.0)
    candidates = v2_coordinate_candidates(items, capacity=1000)
    produced = mask_list(candidates)
    assert (NAMES[0],) not in produced
    assert "attention_0" not in [c[0] for c in candidates]


def test_historical_control_included_when_feasible() -> None:
    historical = [NAMES[0], NAMES[1]]
    candidates = v2_coordinate_candidates(
        standard_items(), capacity=1000, historical_protected=historical
    )
    ids = [c[0] for c in candidates]
    assert "historical_main_control" in ids
    control = next(c for c in candidates if c[0] == "historical_main_control")
    assert set(control[1]) == set(historical)


def test_dedup_and_unknown_names_dropped() -> None:
    candidates = v2_coordinate_candidates(
        standard_items(),
        capacity=1000,
        historical_protected=[NAMES[0], "unknown.layer"],
    )
    assert len(candidates) == len({c[1] for c in candidates})
    for _, mask, _ in candidates:
        assert all(name in NAMES for name in mask)