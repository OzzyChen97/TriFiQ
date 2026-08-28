from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import pytest
import torch


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from quantvla_full_context import (  # noqa: E402
    BudgetItem,
    PROTOCOL,
    candidate_plan_mapping,
    exact_weighted_knapsack,
    paired_candidate_summary,
    quick_advancement,
    select_activation_mode,
    select_frozen_candidate,
    shard_candidate_mapping,
)
from aggregate_full_context_table1 import exact_mcnemar, holm_adjust  # noqa: E402
from quantvla_hessian_w4 import a8_scale_table  # noqa: E402


def score(d_func: list[float], d_pac: list[float], *, component: float = 0.1) -> dict:
    assert len(d_func) == len(d_pac)
    sequences = [
        {
            "task": f"task-{index // 2}",
            "seed": index,
            "components": {"pose": component, "stitch": component, "grip": component},
        }
        for index in range(len(d_pac))
    ]
    return {
        "d_func": sum(d_func) / len(d_func),
        "d_pac": sum(d_pac) / len(d_pac),
        "d_func_summary": {"per_sequence": d_func},
        "d_pac_summary": {"per_sequence": d_pac, "sequences": sequences},
    }


def test_paired_minimax_requires_both_metrics_and_component_safety() -> None:
    baseline = score([1.0] * 8, [2.0] * 8)
    improved = score([0.8] * 8, [1.5] * 8)
    summary = paired_candidate_summary(improved, baseline)
    assert summary["objective"] < 0.0
    assert summary["eligible"] is True

    pose_regression = score([0.8] * 8, [1.5] * 8, component=0.3)
    unsafe = paired_candidate_summary(pose_regression, baseline)
    assert unsafe["objective"] < 0.0
    assert unsafe["component_constraints_pass"] is False
    assert unsafe["eligible"] is False


def test_exact_sparse_knapsack_matches_brute_force() -> None:
    items = [
        BudgetItem("a", 3, 1.0, 0.2),
        BudgetItem("b", 4, 0.1, 1.2),
        BudgetItem("c", 5, 0.8, 0.8),
        BudgetItem("d", 7, 1.5, -0.1),
        BudgetItem("e", 2, -0.4, 0.9),
    ]
    for lam in PROTOCOL["counterfactual_search"]["scalarization_lambdas"]:
        actual = exact_weighted_knapsack(items, budget_bytes=10, lambda_d_func=lam)
        possibilities = []
        for count in range(len(items) + 1):
            for subset in itertools.combinations(items, count):
                cost = sum(item.extra_bytes for item in subset)
                if cost > 10:
                    continue
                utility = sum(
                    lam * item.benefit_d_func + (1.0 - lam) * item.benefit_d_pac
                    for item in subset
                )
                possibilities.append(
                    (utility, -cost, -len(subset), tuple(item.name for item in subset))
                )
        best = max(possibilities)
        assert actual.utility == pytest.approx(best[0])
        assert actual.extra_bytes == -best[1]
        assert actual.protected == best[3]


def test_frozen_one_se_prefers_smaller_plan() -> None:
    baseline = score([1.0] * 8, [1.0] * 8)
    larger = score([0.80] * 8, [0.80] * 8)
    smaller_values = [0.78, 0.84] * 4
    smaller = score(smaller_values, smaller_values)
    scores = {"context_base": baseline, "larger": larger, "smaller": smaller}
    plans = {
        "context_base": {"total_bytes": 100, "retained_fp16_layers": 0, "protected_layers": []},
        "larger": {"total_bytes": 110, "retained_fp16_layers": 2, "protected_layers": ["a", "b"]},
        "smaller": {"total_bytes": 105, "retained_fp16_layers": 1, "protected_layers": ["a"]},
    }
    result = select_frozen_candidate(scores=scores, baseline_id="context_base", plan_rows=plans)
    assert result["best_objective_id"] == "larger"
    assert result["selected_id"] == "smaller"


def test_activation_a16_is_diagnostic_only() -> None:
    static = score([1.0] * 8, [1.0] * 8)
    a16 = score([0.7] * 8, [0.7] * 8)
    dynamic = score([0.8] * 8, [0.8] * 8)
    result = select_activation_mode(static=static, dynamic=dynamic, a16=a16)
    assert result["a8_bottleneck"] is True
    assert result["selected_activation_mode"] == "dynamic_a8"
    assert result["a16_deployable"] is False


def test_quick_gate_uses_fresh_registered_keys_and_paired_wins() -> None:
    rows = []
    tasks = PROTOCOL["quick_development"]["tasks"]
    seeds = PROTOCOL["quick_development"]["seeds"]
    for task_index, task in enumerate(tasks):
        for seed in seeds:
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "main_success": seed % 2 == 0,
                    "candidate_success": seed % 2 == 0 or (task_index == 0 and seed == 51),
                }
            )
    result = quick_advancement(rows)
    assert result["candidate_successes"] == result["main_successes"] + 1
    assert result["paired_wins"] == 1
    assert result["paired_losses"] == 0
    assert result["passes_success_gate"] is True


def test_quick_gate_rejects_non_cartesian_coverage() -> None:
    rows = [
        {
            "task": task,
            "seed": seed,
            "main_success": False,
            "candidate_success": True,
        }
        for task in PROTOCOL["quick_development"]["tasks"]
        for seed in PROTOCOL["quick_development"]["seeds"]
    ]
    rows[-1] = dict(rows[-2])
    with pytest.raises(ValueError, match="paired-key"):
        quick_advancement(rows)


def test_candidate_manifest_and_gpu_shards_are_exact(tmp_path: Path) -> None:
    candidates = []
    for index in range(7):
        path = tmp_path / f"plan-{index}.json"
        path.write_text("{}", encoding="utf-8")
        candidates.append({"candidate_id": f"c{index}", "path": str(path)})
    from quantvla_full_context import protocol_attestation

    manifest = {
        "full_context_protocol": protocol_attestation(),
        "candidates": candidates,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    mapping, loaded = candidate_plan_mapping(manifest_path=manifest_path)
    assert loaded is not None
    shards = [
        shard_candidate_mapping(mapping, shard_index=index, shard_count=3)
        for index in range(3)
    ]
    assert set().union(*(set(shard) for shard in shards)) == set(mapping)
    assert sum(len(shard) for shard in shards) == len(mapping)


def test_registered_exact_statistics_are_stable_at_table1_scale() -> None:
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(10, 0) == pytest.approx(2.0 / 1024.0)
    value = exact_mcnemar(1300, 1200)
    assert 0.0 <= value <= 1.0
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_static_a8_supports_model_native_ten_flow_steps() -> None:
    generator = torch.Generator().manual_seed(11)
    activations = torch.randn(10, 6, 64, generator=generator)
    scales = a8_scale_table(activations, flow_steps=10)
    assert tuple(scales.shape) == (10, 64)
    assert torch.isfinite(scales).all()
    assert torch.all(scales > 0)
