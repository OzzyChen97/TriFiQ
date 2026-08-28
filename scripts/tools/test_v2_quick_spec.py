from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from build_v2_quick_spec import build_spec, draw_quick_tasks  # noqa: E402
from register_pi05_non_inferiority_anchor import register  # noqa: E402


SHA = "v2-quick-test-sha"
V1_QUICK = [
    "CloseFridge",
    "OpenDrawer",
    "LoadDishwasher",
    "PrepareCoffee",
    "MakeIceLemonade",
]


def test_draw_quick_tasks_2_2_1_and_excluded() -> None:
    tasks = draw_quick_tasks(SHA, excluded_tasks=set(V1_QUICK))
    assert [len(tasks[split]) for split in ("atomic_seen", "composite_seen", "composite_unseen")] == [2, 2, 1]
    drawn = [task for split in tasks for task in tasks[split]]
    assert len(set(drawn)) == 5
    assert set(drawn).isdisjoint(set(V1_QUICK))
    again = draw_quick_tasks(SHA, excluded_tasks=set(V1_QUICK))
    assert again == tasks


def test_build_spec_records_freeze_and_table1_holdout() -> None:
    selection_entries = [
        {"split": "atomic_seen", "task": "TurnOnMicrowave"},
        {"split": "atomic_seen", "task": "OpenStandMixerHead"},
        {"split": "atomic_seen", "task": "PickPlaceDrawerToCounter"},
        {"split": "atomic_seen", "task": "CloseBlenderLid"},
        {"split": "composite_seen", "task": "DeliverStraw"},
        {"split": "composite_seen", "task": "GetToastedBread"},
        {"split": "composite_seen", "task": "KettleBoiling"},
        {"split": "composite_seen", "task": "PackIdenticalLunches"},
        {"split": "composite_unseen", "task": "ArrangeBreadBasket"},
        {"split": "composite_unseen", "task": "ArrangeTea"},
        {"split": "composite_unseen", "task": "BreadSelection"},
        {"split": "composite_unseen", "task": "CategorizeCondiments"},
    ]
    spec = build_spec(
        SHA,
        selection_spec={"entries": selection_entries},
        v1_quick_tasks=V1_QUICK,
    )
    assert spec["total_tasks"] == 5
    assert spec["seeds"] == list(range(60, 70))
    assert spec["table1_held_out"] is True
    drawn = [task for split in spec["tasks"] for task in spec["tasks"][split]]
    assert set(drawn).isdisjoint(V1_QUICK)
    assert set(drawn).isdisjoint(entry["task"] for entry in selection_entries)
    assert "runtime_selector_main_or_bitwise_equivalence_artifact" in (
        spec["pi05_baseline_requirement"]
    )


def test_anchor_registration_accepts_parity_and_rejects_superiority() -> None:
    aggregate = {
        "kind": "full_context_quick_development_gate",
        "main_successes": 14,
        "candidate_successes": 14,
        "paired_wins": 4,
        "paired_losses": 4,
    }
    report = register(aggregate, source="unit")
    assert report["status"] == "not_superior_to_gdsq_main"
    assert "compression parity" in report["claim_guard"]
    aggregate["candidate_successes"] = 18
    with pytest.raises(ValueError, match="superior"):
        register(aggregate, source="unit")