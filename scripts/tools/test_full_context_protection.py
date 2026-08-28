from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np
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
from quantvla_table1_bytes import (  # noqa: E402
    CANDIDATE_FP16_BYTES,
    TABLE1_FP16_BYTES,
    TABLE1_QUANTVLA_BYTES,
    fixed_bytes,
    table1_display_gib,
    table1_total_static_budget,
    table1_variable_budget,
)
from select_full_context_protection import finalize_candidate_plan  # noqa: E402


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


def task_score(
    d_func_by_task: dict[str, list[float]],
    d_pac_by_task: dict[str, list[float]],
) -> dict:
    seq_d_func = []
    seq_d_pac = []
    sequences = []
    for task, values in d_func_by_task.items():
        for seed_index, value in enumerate(values):
            seq_d_func.append(value)
            seq_d_pac.append(d_pac_by_task[task][seed_index])
            sequences.append(
                {
                    "task": task,
                    "seed": seed_index,
                    "components": {"pose": 0.1, "stitch": 0.1, "grip": 0.1},
                }
            )
    return {
        "d_func": sum(seq_d_func) / len(seq_d_func),
        "d_pac": sum(seq_d_pac) / len(seq_d_pac),
        "d_func_summary": {"per_sequence": seq_d_func},
        "d_pac_summary": {"per_sequence": seq_d_pac, "sequences": sequences},
    }


def test_task_cluster_uniform_improvement_passes() -> None:
    atomic = ["OpenDrawer", "OpenCabinet"]
    composite = ["LoadDishwasher", "PrepareCoffee"]
    baseline = task_score(
        {t: [1.0, 1.0] for t in atomic + composite},
        {t: [1.0, 1.0] for t in atomic + composite},
    )
    candidate = task_score(
        {t: [0.0, 0.0] for t in atomic + composite},
        {t: [0.0, 0.0] for t in atomic + composite},
    )
    summary = paired_candidate_summary(candidate, baseline)
    assert summary["objective"] < 0.0
    assert summary["eligible"] is True
    assert {"all", "atomic_seen", "composite_seen"} <= {
        entry["cluster"] for entry in summary["metrics"]["d_func"]["clusters"]
    }


def test_task_cluster_redistribution_is_rejected() -> None:
    atomic = ["OpenDrawer", "OpenCabinet"]
    composite = ["LoadDishwasher", "PrepareCoffee"]
    baseline = task_score(
        {t: [1.0, 1.0] for t in atomic + composite},
        {t: [1.0, 1.0] for t in atomic + composite},
    )
    candidate = task_score(
        {t: [0.0, 0.0] for t in atomic} | {t: [1.5, 1.5] for t in composite},
        {t: [0.0, 0.0] for t in atomic} | {t: [1.5, 1.5] for t in composite},
    )
    summary = paired_candidate_summary(candidate, baseline)
    composite_entry = [
        entry
        for entry in summary["metrics"]["d_pac"]["clusters"]
        if entry["cluster"] == "composite_seen"
    ][0]
    assert composite_entry["delta_mean"] > 0.0
    assert summary["objective"] > 0.0
    assert summary["eligible"] is False


def test_jackknife_task_se_matches_closed_form() -> None:
    from quantvla_full_context import jackknife_task_se

    assert jackknife_task_se(np.asarray([1.0, 1.0, 1.0, 1.0])) == 0.0
    assert jackknife_task_se(np.asarray([0.0, 0.0, 2.0, 2.0])) == pytest.approx(
        (3.0 / 4.0 * 4.0) ** 0.5
    )


def test_task_scalars_aggregate_seeds_within_task() -> None:
    from quantvla_full_context import task_scalars

    score_doc = task_score(
        {"CloseFridge": [0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0]},
        {"CloseFridge": [0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0]},
    )
    scalars = task_scalars(score_doc)
    # Seed-aggregated mean+CVaR: seed 0 -> 0.0, seed 1 -> 8.0, task scalar 4.0.
    # The global-form mean+CVaR over all eight sequences would be 6.0.
    assert scalars[("d_pac", "CloseFridge")] == pytest.approx(4.0)


def _synthetic_byte_rows(count: int, fp16_bytes: int, w4_bytes: int) -> dict:
    return {
        f"l{index}": {
            "fp16_bytes": fp16_bytes,
            "w4_bytes": w4_bytes,
            "extra_fp16_bytes": fp16_bytes - w4_bytes,
        }
        for index in range(count)
    }


def _synthetic_plan(count: int) -> dict:
    return {
        "layers": {
            f"l{index}": {"bits": 4, "group": 64, "skip": False}
            for index in range(count)
        },
        "meta": {},
    }


def test_table1_byte_anchors_reproduce_table_display_cells() -> None:
    assert table1_display_gib(TABLE1_FP16_BYTES["gr00t"]) == "1.993"
    assert table1_display_gib(TABLE1_QUANTVLA_BYTES["gr00t"]) == "0.898"
    assert table1_display_gib(TABLE1_FP16_BYTES["pi05"]) == "4.113"
    assert table1_display_gib(TABLE1_QUANTVLA_BYTES["pi05"]) == "1.388"


def test_table1_variable_budget_exact_values() -> None:
    assert fixed_bytes("gr00t") == 327_352_320
    assert fixed_bytes("pi05") == 0
    assert table1_total_static_budget("gr00t") == 1_060_149_657
    assert table1_variable_budget("gr00t") == 732_797_337
    assert table1_variable_budget("pi05") == 1_639_513_497


def test_table1_anchors_match_frozen_official_artifacts() -> None:
    repo = Path(__file__).resolve().parents[2]
    registry = json.loads(
        (repo / "docs/gdsq_vla_cvpr2026/experiment_registry.json").read_text(
            encoding="utf-8"
        )
    )
    gr_summary = json.loads(
        (
            repo / registry["experiments"]["gr00t_static_official50"]["summary"]["path"]
        ).read_text(encoding="utf-8")
    )
    assert int(
        gr_summary["configs"]["fp16"]["paper_style_memory"][
            "task_weighted_mean_component_bytes"
        ]
    ) == TABLE1_FP16_BYTES["gr00t"]
    assert int(
        gr_summary["configs"]["w4a8_atmohb"]["paper_style_memory"][
            "task_weighted_mean_component_bytes"
        ]
    ) == TABLE1_QUANTVLA_BYTES["gr00t"]
    prereg = json.loads(
        (repo / "runs/gdsq_week1_preregistered_v1/preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    assert int(prereg["models"]["gr00t"]["fp16_bytes"]) == CANDIDATE_FP16_BYTES["gr00t"]
    assert int(prereg["models"]["pi05"]["fp16_bytes"]) == TABLE1_FP16_BYTES["pi05"]
    assert TABLE1_FP16_BYTES["pi05"] == CANDIDATE_FP16_BYTES["pi05"]
    pi05_doc = (repo / "docs/pi05_gdsq_formal_evaluation.md").read_text(encoding="utf-8")
    assert f"{TABLE1_QUANTVLA_BYTES['pi05']:,}" in pi05_doc


def test_finalize_candidate_plan_emits_table1_fields_and_enforces_static_ceiling() -> None:
    rows = _synthetic_byte_rows(4, fp16_bytes=250_000_000, w4_bytes=125_000_000)
    plan = _synthetic_plan(4)
    finalized = finalize_candidate_plan(
        plan,
        identifier="unit",
        model="gr00t",
        activation_mode="static_a8",
        round_index=1,
        protected={"l0"},
        byte_rows=rows,
        manifest_sha="0" * 64,
        flow_steps=4,
    )
    assert finalized["budget_bytes"] == table1_variable_budget("gr00t")
    assert finalized["fixed_bytes"] == fixed_bytes("gr00t")
    assert finalized["table1_total_static_budget_bytes"] == table1_total_static_budget(
        "gr00t"
    )
    assert finalized["table1_total_static_bytes"] == (
        fixed_bytes("gr00t") + finalized["total_bytes"]
    )
    assert finalized["achieved_target_matrix_compression"] == pytest.approx(
        finalized["fp16_total_bytes"] / finalized["total_bytes"]
    )
    assert finalized["table1_total_static_compression"] == pytest.approx(
        TABLE1_FP16_BYTES["gr00t"] / finalized["table1_total_static_bytes"]
    )
    over_plan = _synthetic_plan(4)
    with pytest.raises(ValueError, match="Table-1 total-static byte ceiling"):
        finalize_candidate_plan(
            over_plan,
            identifier="unit-over",
            model="gr00t",
            activation_mode="static_a8",
            round_index=1,
            protected={"l0", "l1", "l2"},
            byte_rows=rows,
            manifest_sha="0" * 64,
            flow_steps=4,
        )
