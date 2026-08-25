#!/usr/bin/env python3
"""Unit tests for the π0.5 true runtime ATM/OHB selector."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/pi05/openpi/src"))

from openpi.quant.atm_runtime_selector import (  # noqa: E402
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
            "pi05": {
                "variant_config_ids": {
                    "baseline": "gdsq_vla",
                    "atm": "gdsq_vla_atm_only",
                    "ohb": "gdsq_vla_ohb_only",
                    "atmohb": "gdsq_vla_atmohb",
                },
                "tasks": {
                    "CloseFridge": {
                        "selected_variant": "baseline",
                        "selected_config_id": "gdsq_vla",
                    },
                    "CloseBlenderLid": {
                        "selected_variant": "ohb",
                        "selected_config_id": "gdsq_vla_ohb_only",
                    },
                    "PickPlaceSinkToCounter": {
                        "selected_variant": "atm",
                        "selected_config_id": "gdsq_vla_atm_only",
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
            model_id="pi05",
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
        with runtime_selector_context({"task_name": "CloseBlenderLid"}) as decision:
            assert decision is not None
            assert decision.selected_variant == "ohb"
            assert atm_enabled_for_current_request() is False
            assert ohb_enabled_for_current_request() is True
        with runtime_selector_context({"task_name": "PickPlaceSinkToCounter"}) as decision:
            assert decision is not None
            assert decision.selected_variant == "atm"
            assert atm_enabled_for_current_request() is True
            assert ohb_enabled_for_current_request() is False
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
        model_id="pi05",
        strict=True,
    )
    assert selector.metadata()["selection_scope"] == "model_level_absolute_mechanism_gate"
    assert selector.metadata()["uses_task_metadata_for_selection"] is False
    decisions = [selector.select(None), selector.select({"task_name": "unknown"})]
    assert {row.selected_variant for row in decisions} == {"ohb"}
    assert {row.selected_config_id for row in decisions} == {"gdsq_vla_ohb_only"}


def main() -> None:
    test_context_switches_per_request()
    test_frozen_v8_is_model_level_and_ignores_task_metadata()
    print("pi05 runtime selector tests passed")


if __name__ == "__main__":
    main()
