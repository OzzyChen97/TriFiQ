#!/usr/bin/env python3
"""Unit tests for register_gr00t_non_inferiority_anchor.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from register_gr00t_non_inferiority_anchor import register  # noqa: E402


def base_aggregate() -> dict:
    return {
        "kind": "full_context_v2_quick_development_gate",
        "main_successes": 30,
        "candidate_successes": 30,
        "paired_wins": 4,
        "paired_losses": 4,
        "table1_total_static_bytes": 962068480,
        "table1_total_static_budget_bytes": 1060149657,
    }


def test_parity_registers_anchor() -> None:
    report = register(base_aggregate(), source="agg.json")
    assert report["status"] == "not_superior_to_gdsq_main"
    assert report["role"] == "runtime_portability_non_inferiority_anchor"
    assert "parity" in report["claim_guard"]


def test_superior_candidate_refused() -> None:
    aggregate = base_aggregate()
    aggregate["candidate_successes"] = 40
    try:
        register(aggregate, source="agg.json")
    except ValueError as error:
        assert "superior" in str(error)
    else:
        raise AssertionError("superior candidate must be refused")


def test_wrong_kind_refused() -> None:
    aggregate = base_aggregate()
    aggregate["kind"] = "other"
    try:
        register(aggregate, source="agg.json")
    except ValueError as error:
        assert "v2 quick gate" in str(error)
    else:
        raise AssertionError("wrong kind must be refused")


if __name__ == "__main__":
    test_parity_registers_anchor()
    test_superior_candidate_refused()
    test_wrong_kind_refused()
    print("[register-gr00t-non-inferiority-anchor] selftest OK")
