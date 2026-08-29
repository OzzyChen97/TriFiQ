from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "runs/gdsq_extension_preregistered_v1/plan.json"
REGISTRY = REPO_ROOT / "docs/gdsq_vla_iclr2027/experiment_registry.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extension_matrix_is_frozen_and_result_blind() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    record = registry["preregistrations"]["post_week1_extension_v1"]
    assert sha256_file(PLAN) == record["sha256"]
    assert plan["immutable"] is True
    assert plan["result_blind"] is True
    assert plan["libero"]["suites"] == [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    ]
    assert plan["libero"]["models"] == ["gr00t", "pi05"]
    assert len(plan["libero"]["suite_specs"]) == 8
    assert plan["libero"]["expected_formal_episodes"] == 16000
    assert plan["libero"]["robocasa_selector_artifact_reuse_allowed"] is False
    for record in plan["libero"]["suite_specs"].values():
        path = Path(record["path"])
        assert path.is_file()
        assert sha256_file(path) == record["sha256"]
        spec = json.loads(path.read_text(encoding="utf-8"))
        assert spec["result_blind"] is True
        assert spec["calibration"]["suite_specific"] is True
        assert spec["calibration"]["forbid_robocasa_artifacts"] is True
        assert spec["expected_episodes_per_config"] == 500


def test_robustness_pareto_and_deployment_protocols_are_explicit() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["calibration_robustness"]["observation_counts"] == [16, 64, 256]
    assert plan["calibration_robustness"]["selection_feedback_allowed"] is False
    assert plan["storage_sr_pareto"]["main_result_budget_ratio"] == 1.0
    assert plan["storage_sr_pareto"]["other_budgets_diagnostic_only"] is True
    deployment = plan["deployment_measurement"]
    assert deployment["batch_size"] == 1
    assert deployment["simulator_process_excluded_from_memory"] is True
    assert deployment["fused_int4_int8_required_for_end_to_end_claim"] is True
