from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from rank_v2_anchors import rank_anchors  # noqa: E402


def score_doc(d_pac: float) -> dict:
    sequences = [
        {
            "task": f"task-{index // 2}",
            "seed": index % 2,
            "components": {"pose": 0.0, "stitch": 0.0, "grip": 0.0},
        }
        for index in range(4)
    ]
    values = [d_pac / 2.0] * 4  # single-sequence seeds double under mean+CVaR
    return {
        "d_func": d_pac / 2.0,
        "d_pac": d_pac,
        "d_func_summary": {"per_sequence": values},
        "d_pac_summary": {"per_sequence": values, "sequences": sequences},
    }


def payloads(main_pac: dict[str, float], full_w4_pac: dict[str, float]) -> dict:
    result = {}
    for split in ("atomic_seen", "composite_seen", "composite_unseen"):
        result[split] = {
            "main_mask": score_doc(main_pac[split]),
            "full_w4": score_doc(full_w4_pac[split]),
        }
    return result


def test_proxy_passes_when_main_better_everywhere() -> None:
    report = rank_anchors(
        payloads(
            main_pac={"atomic_seen": 0.5, "composite_seen": 0.6, "composite_unseen": 0.7},
            full_w4_pac={"atomic_seen": 1.0, "composite_seen": 1.2, "composite_unseen": 1.4},
        )
    )
    assert report["stop_before_layer_flips"] is False
    assert report["decision"].startswith("proxy_passes")


def test_stop_when_full_w4_ranks_no_worse_on_any_split() -> None:
    report = rank_anchors(
        payloads(
            main_pac={"atomic_seen": 0.5, "composite_seen": 1.5, "composite_unseen": 0.7},
            full_w4_pac={"atomic_seen": 1.0, "composite_seen": 1.2, "composite_unseen": 1.4},
        )
    )
    assert report["stop_before_layer_flips"] is True
    assert report["contradicting_splits"] == ["composite_seen"]
    assert "STOP" in report["decision"]


def test_task_macro_uses_v2_seed_aggregated_scalar() -> None:
    # The single-sequence seed doubles under mean+CVaR, so the reported
    # per-split scalar equals the raw d_pac passed in.
    report = rank_anchors(
        payloads(
            main_pac={"atomic_seen": 0.8, "composite_seen": 0.8, "composite_unseen": 0.8},
            full_w4_pac={"atomic_seen": 1.6, "composite_seen": 1.6, "composite_unseen": 1.6},
        )
    )
    assert report["anchors"]["main_mask"]["atomic_seen_d_pac"] == pytest.approx(0.8)
    assert report["anchors"]["full_w4"]["composite_unseen_d_pac"] == pytest.approx(1.6)