from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from interpret_four_config_attribution import (  # noqa: E402
    CONFIG_IDS,
    interpret,
    paired_comparison,
    relation,
)

TASKS = ["t0", "t1", "t2", "t3", "t4"]
SEEDS = list(range(10))


def rows_by_success(successes: set[tuple[str, int]]) -> dict[tuple[str, int], bool]:
    return {
        (task, seed): (task, seed) in successes
        for task in TASKS
        for seed in SEEDS
    }


def build(
    h: set, m: set, c: set, c16: set
) -> dict[str, dict[tuple[str, int], bool]]:
    return {
        "h": rows_by_success(h),
        "m": rows_by_success(m),
        "c": rows_by_success(c),
        "c16": rows_by_success(c16),
    }


def all_keys() -> set[tuple[str, int]]:
    return {(task, seed) for task in TASKS for seed in SEEDS}


def has_finding(report, needle: str) -> bool:
    return any(needle in finding for finding in report["findings"])


def test_paired_comparison_counts() -> None:
    a = rows_by_success({("t0", 0), ("t1", 0)})
    b = rows_by_success({("t0", 0)})
    result = paired_comparison(a, b)
    assert result["pairs"] == 50
    assert result["wins"] == 1
    assert result["losses"] == 0
    assert relation(result) == "approx"
    c = rows_by_success({("t0", s) for s in range(6)})
    strong = paired_comparison(a, c)
    assert relation(strong) == "b_gt_a"


def test_mask_problem_scenario() -> None:
    # H ~= M and M >> C and C16 ~= C -> mask problem, A8 exonerated.
    h = {("t0", s) for s in range(8)}
    m = {("t0", s) for s in range(7)}
    c = {("t0", s) for s in range(3)}
    c16 = {("t0", s) for s in range(4)}
    report = interpret(build(h, m, c, c16))
    assert report["conclusions"]["h_vs_m"] == "approx"
    assert report["conclusions"]["m_vs_c"] == "a_gt_b"
    assert report["conclusions"]["c16_vs_c"] == "approx"
    assert "mask_problem" in report["findings"][0]
    assert has_finding(report, "a8_exonerated")


def test_historical_runtime_dominates_scenario() -> None:
    # H >> M: historical runtime carries main.
    h = {("t0", s) for s in range(9)}
    m = {("t0", s) for s in range(2)}
    c = {("t0", s) for s in range(2)}
    c16 = {("t0", s) for s in range(2)}
    report = interpret(build(h, m, c, c16))
    assert report["conclusions"]["h_vs_m"] == "a_gt_b"
    assert has_finding(report, "historical_runtime_dominates")


def test_a16_recovers_scenario() -> None:
    # C16 >> C: dynamic A8 unstable on candidate states.
    h = {("t0", s) for s in range(8)}
    m = {("t0", s) for s in range(8)}
    c = {("t0", s) for s in range(1)}
    c16 = {("t0", s) for s in range(7)}
    report = interpret(build(h, m, c, c16))
    assert report["conclusions"]["c16_vs_c"] == "a_gt_b"
    assert has_finding(report, "bounded dynamic A8")


def test_proxy_dead_contradiction_flag() -> None:
    # Offline prefers C, closed loop prefers M -> fixed teacher-state proxy dead.
    h = {("t0", s) for s in range(8)}
    m = {("t0", s) for s in range(8)}
    c = {("t0", s) for s in range(2)}
    c16 = {("t0", s) for s in range(3)}
    offline = {"m": 0.42, "c": 0.11}
    report = interpret(build(h, m, c, c16), offline=offline)
    assert report["proxy_dead"] is True
    assert has_finding(report, "proxy_dead")


def test_missing_config_rows_raises() -> None:
    rows = build(set(), set(), set(), set())
    rows.pop("c16")
    with pytest.raises(ValueError, match="missing attribution config rows"):
        interpret(rows)


def test_config_ids_are_the_four_expected() -> None:
    assert CONFIG_IDS == ("h", "m", "c", "c16")