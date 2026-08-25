#!/usr/bin/env python3
"""Unit tests for the GR00T true runtime ATM/OHB selector."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from gr00t.atm.runtime_selector import (  # noqa: E402
    RuntimeSelector,
    atm_enabled_for_current_request,
    current_decision,
    ohb_enabled_for_current_request,
    runtime_selector_context,
    set_runtime_selector,
)


def make_selector() -> dict:
    return {
        "rule_name": "v4_test_rule",
        "models": {
            "gr00t": {
                "variant_config_ids": {
                    "baseline": "cscka_final",
                    "atm": "cscka_final_atm",
                    "ohb": "cscka_final_ohb",
                    "atmohb": "cscka_final_atmohb",
                },
                "tasks": {
                    "CloseFridge": {
                        "selected_variant": "baseline",
                        "selected_config_id": "cscka_final",
                    },
                    "PickPlaceSinkToCounter": {
                        "selected_variant": "atm",
                        "selected_config_id": "cscka_final_atm",
                    },
                    "OpenCabinet": {
                        "selected_variant": "ohb",
                        "selected_config_id": "cscka_final_ohb",
                    },
                },
            }
        }
    }


def test_context_switches_per_request() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "selector.json"
        path.write_text(json.dumps(make_selector()), encoding="utf-8")
        selector = RuntimeSelector(
            selector_path=path,
            selector=json.loads(path.read_text(encoding="utf-8")),
            model_id="gr00t",
            strict=True,
        )
        set_runtime_selector(selector)
        assert atm_enabled_for_current_request() is False
        assert ohb_enabled_for_current_request() is False
        with runtime_selector_context({"task_name": "CloseFridge"}) as decision:
            assert decision is not None
            assert decision.selected_variant == "baseline"
            assert decision.selector_rule_name == "v4_test_rule"
            assert selector.metadata()["rule_name"] == "v4_test_rule"
            assert atm_enabled_for_current_request() is False
            assert ohb_enabled_for_current_request() is False
            assert current_decision() == decision
        with runtime_selector_context({"task": "PickPlaceSinkToCounter"}) as decision:
            assert decision is not None
            assert decision.selected_variant == "atm"
            assert atm_enabled_for_current_request() is True
            assert ohb_enabled_for_current_request() is False
        with runtime_selector_context({"env_name": "OpenCabinet"}) as decision:
            assert decision is not None
            assert decision.selected_variant == "ohb"
            assert atm_enabled_for_current_request() is False
            assert ohb_enabled_for_current_request() is True
        try:
            with runtime_selector_context({"task_name": "UnknownTask"}):
                pass
        except ValueError as exc:
            assert "missing decision" in str(exc)
        else:
            raise AssertionError("strict selector accepted an unknown task")
        set_runtime_selector(None)
        assert atm_enabled_for_current_request() is True
        assert ohb_enabled_for_current_request() is True


def test_frozen_v8_is_model_level_and_ignores_task_metadata() -> None:
    path = REPO_ROOT / "runs/atmohb_dynamic_selector_v8/selector.json"
    selector = RuntimeSelector(
        selector_path=path,
        selector=json.loads(path.read_text(encoding="utf-8")),
        model_id="gr00t",
        strict=True,
    )
    assert selector.metadata()["selection_scope"] == "model_level_absolute_mechanism_gate"
    assert selector.metadata()["uses_task_metadata_for_selection"] is False
    decisions = [selector.select(None), selector.select({"task_name": "unknown"})]
    assert {row.selected_variant for row in decisions} == {"baseline"}
    assert {row.selected_config_id for row in decisions} == {"cscka_final"}


def main() -> None:
    test_context_switches_per_request()
    test_frozen_v8_is_model_level_and_ignores_task_metadata()
    print("gr00t runtime selector tests passed")


if __name__ == "__main__":
    main()
