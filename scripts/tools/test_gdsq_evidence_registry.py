from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("gdsq_evidence_registry.py")
SPEC = importlib.util.spec_from_file_location("gdsq_evidence_registry", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_complete_experiment_rejects_missing_coverage() -> None:
    experiment = {
        "status": "complete",
        "coverage": {
            "expected_episodes": 10,
            "observed_episodes": 9,
            "missing_episodes": 1,
            "duplicate_episodes": 0,
        },
    }
    errors = MODULE.coverage_errors("demo", experiment)
    assert any("missing episodes" in error for error in errors)
    assert any("observed 9 != expected 10" in error for error in errors)


def test_enabled_claim_rejects_planned_experiment() -> None:
    registry = {
        "runtime_selector": {},
        "experiments": {
            "planned": {"status": "planned", "main_claim_enabled": False}
        },
        "paper_claims": {
            "bad": {"enabled": True, "experiment": "planned"}
        },
    }
    errors = MODULE.validate_registry(registry)
    assert any("non-complete" in error for error in errors)


def test_main_registry_artifacts_are_frozen() -> None:
    registry = MODULE.read_json(MODULE.DEFAULT_REGISTRY)
    errors = MODULE.validate_registry(registry)
    assert errors == []
