#!/usr/bin/env python3
"""Render and audit the DyPAC-VLA paper cells from frozen final-v2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/gdsq_vla_iclr2027"  # Legacy directory name; paper identity is DyPAC-VLA.
QVLA_ACTQUANT_AGGREGATE = ROOT / "runs/qvla_actquant_table1/formal/aggregate.json"
QVLA_ACTQUANT_PARTIAL = (
    ROOT
    / "runs/qvla_actquant_table1/formal/partial/table1_partial_snapshot.json"
)

SOURCES = {
    "gr00t_plan": (
        ROOT / "runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json",
        "969c5ae8bc84719229c81a88e8b6ace745cbb87cea551528ec8cf1fd3820e3a2",
    ),
    "gr00t_aggregate": (
        ROOT / "runs/full_context_v2/table1/aggregate.json",
        "cbb59547a6149456f9ae8fc0bac0fedea1267e5ff5f4d144448112f15b0f459e",
    ),
    "gr00t_ratio_selection": (
        ROOT / "runs/gdsq_week1_preregistered_v1/analysis/ratio_stability.json",
        "dc074814edf548c5b9f3a97af7c8240221da897177a935a30903a734568a7d33",
    ),
    "gr00t_gdsq_atomic_plan": (
        ROOT
        / "checkpoints/packs/robocasa365/gr00t_quant_plan_robocasa365_cscka_16to1_adjudicated.final_plan.json",
        "70b6b0d3190c3f4ebad9a5a874f9c0ac1044143a4912020aa9fb129f66a6eeb4",
    ),
    "gr00t_gdsq_composite_seen_plan": (
        ROOT
        / "checkpoints/packs/robocasa365/gr00t_quant_plan_robocasa365_cscka_16to1_composite_seen.final_plan.json",
        "abe7571b4dd1548fdf24042c7b508f34a3607a787cd92d7fb6f66f4855b46532",
    ),
    "gr00t_gdsq_composite_unseen_plan": (
        ROOT
        / "checkpoints/packs/robocasa365/gr00t_quant_plan_robocasa365_cscka_16to1_composite_unseen.final_plan.json",
        "970d6865b7fcd01b7848baa0daa2dfb7b98db79bbde8ea0022641d390bb71d47",
    ),
    "activation_attribution": (
        ROOT / "runs/full_context_v2/p2/activation_attribution.json",
        "3c59f27bf648b575bf87864ba9ed4a44590d077d8a57c298d841e6ff4bd879b9",
    ),
    "table3_closed_loop": (
        ROOT / "runs/full_context_v2/table3_quick/aggregate.json",
        "41a2d63af912111725bf3b547a7ee220d78135987cb914d0cd4b073c16a87f9b",
    ),
    "statistics_correction": (
        ROOT / "runs/full_context_v2/statistics_correction/corrected_statistics.json",
        "5ed6ce1efe409dccfc38af9e1473455563772c646941efcd913349717a01b3ec",
    ),
    "corrected_fcp_selection": (
        ROOT / "runs/full_context_v2/fcp_completion/gr00t_corrected_fcp_frozen.json.selection.json",
        "7e7830c749d501b516c82ead5e416437e4408ba519914a5b47d0f29613914987",
    ),
    "fcp_candidate_closed_loop": (
        ROOT / "runs/full_context_v2/fcp_completion/closed_loop/aggregate.json",
        "1cedce9f204bd3f265cbf1d604020253c78f89791655b914599582d8891f3c77",
    ),
    "gr00t_hardware": (
        ROOT / "runs/full_context_v2/fcp_completion/hardware/summary.json",
        "fa7c3e91ca9f678aae3a33882106ff7ab0bb3daaa0dd37a6faaa58715fdc45e5",
    ),
    "integer_w4a8_backend": (
        PAPER / "evidence/integer_w4a8_backend_audit.json",
        "8e12ed20550dff16f999af2383a5ae89c49e2d48f17b1deb04ad9a104447c219",
    ),
    "predictive_validity": (
        ROOT / "runs/full_context_v2/dpac_predictive_validity/report/audit.json",
        "51a56c68e319ee8bc77d8e2fb28592f15ccfd35c3195089289482e2c237b26bf",
    ),
    "signal_ablation": (
        ROOT / "runs/robocasa365_byte_ablation_v1/aggregate.json",
        "34ab7a8e1f3d75fbc2beef69e2108b52c9ab61e79884a41aa1404c60d640c09f",
    ),
    "signal_ablation_masks": (
        ROOT / "runs/robocasa365_byte_ablation_v1/masks/masks_manifest.json",
        "f233f7ca2fe0ed5b48c4a11402f4e4ff2320ae19472fae7e21abd1f9de8f39df",
    ),
    "signal_ablation_preregistration": (
        ROOT / "runs/robocasa365_byte_ablation_v1/preregistration.json",
        "0bd4917e2159af28aef9ed49c11db1ac4d7cebfe477a8a8a8e1d94fed23641bd",
    ),
    "compression_sweep": (
        ROOT / "runs/robocasa365_compression_sweep_v1/aggregate.json",
        "9c0381783c3521d6f333bf21ea1c1a3bc4bef3920fba734ac21864e46d3a288c",
    ),
    "compression_sweep_masks": (
        ROOT / "runs/robocasa365_compression_sweep_v1/masks/masks_manifest.json",
        "89bb1e8b0dd0ab3c09ba78ca6ab885a50505618ebc6150cae1a5184ab022aea6",
    ),
    "compression_sweep_preregistration": (
        ROOT / "runs/robocasa365_compression_sweep_v1/preregistration.json",
        "6726c7d7a59555759971ead429aabaa08e19e5ff0bc28d5c140f328e1f088d86",
    ),
    "pi05_plan": (
        ROOT / "runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json",
        "e502f7cd7d126517c000f8b5b8e7c6e5537b31226910a2dd6834caaebf736c83",
    ),
    "pi05_main_plan": (
        ROOT / "runs/full_context_v2/pi05_p2/interventions_round_main/context_base.json",
        "461531ea2c48cdd5e0f063bc944c882ede6a36ed73482beeb19134ec7e55e42b",
    ),
    "pi05_pruned_plan": (
        ROOT / "runs/full_context_v2/pi05_p2/pi05_main_pruned_to_budget.json",
        "e06870a2f83c6f957df1733dad0d461fcdd0fc66be929035699915d908ce6bcc",
    ),
    "pi05_dypac_formal": (
        ROOT / "runs/full_context_v2/pi05_table1/aggregate.json",
        "512480a2e0836423254217b5215489bcb218e17a4c7d0fff1cdf5aba9acf4f73",
    ),
    "pi05_protocol_correction": (
        ROOT / "runs/full_context_v2/pi05_table1/protocol_correction.json",
        "cd5e07baaaf7b40e53c1ace9a881eb49aabb9407d8d20189d46e3a11b2186770",
    ),
    "pi05_fcp_diagnostic_raw": (
        ROOT / "runs/full_context_v2/pi05_fcp_diagnostic/raw_aggregate.json",
        "7fb09f82d1fbe51ba4ee1af0f1a4d9671f2cce9255882951609242247630601c",
    ),
    "pi05_fcp_diagnostic_summary": (
        ROOT / "runs/full_context_v2/pi05_fcp_diagnostic/experiment_summary.json",
        "6fdcdf4c53f86ff45cf00cee4fe50abbc9c3ad8ceeab5005dbee531d1f19e87e",
    ),
    "pi05_static_official": (
        ROOT / "runs/pi05_gdsq_gr00t_aligned/official_target_paired50/aggregate/summary.json",
        "c7b6441c6ca0a49aa5cfe8e673d1ee465a5d101076b6ad866b149fb545d10cbb",
    ),
    "pi05_uniform_w6": (
        ROOT / "runs/gdsq_week1_preregistered_v1/execution/runs/pi05_uniform_w6_official50/aggregate/summary.json",
        "849e1e159daf6b515bcb1724a8862fea17e9c0c95053ed5fefc99b9c145e052a",
    ),
    "pi05_omega": (
        ROOT / "runs/gdsq_extension_preregistered_v1/omega_qvla_pi05_robocasa365_v1/aggregate/summary.json",
        "d898552b26fb68db253decfe9375977f1dbf38daf89bb94d73c6d62564114ae9",
    ),
    "daptq_formal": (
        ROOT / "runs/daptq_table1/formal/aggregate.json",
        "2c857eab542035650a22bfd54a890fed49afd084e999ee5f17c37adcddf9eeb8",
    ),
    "gr00t_max_plan": (
        ROOT / "runs/robocasa365_table1_max_sweep_v1/masks/gr00t_max.plan.json",
        "b1fe584404184bf6f5505af5ef40ebaeab759a94c1e0e952dd19b085e387179d",
    ),
    "gr00t_max_aggregate": (
        ROOT / "runs/robocasa365_table1_max_sweep_v1/gr00t_aggregate.json",
        "f1dd5829167dc0a9d5e999352acf673157445160633437323d5635faafe8b97b",
    ),
    "pi05_max_plan": (
        ROOT / "runs/robocasa365_table1_max_sweep_v1/masks/pi05_max.plan.json",
        "8b0aabc15e48970250e605ca689e08bde3c9b73616d2c31fd2fae88d01759dbd",
    ),
    "pi05_max_aggregate": (
        ROOT / "runs/robocasa365_table1_max_sweep_v1/aggregate.json",
        "93bb7b021ca83b073b314c6b18e97f3d7b4c4e327eeb190375d7d5fedb3070bc",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, tol: float = 1e-10) -> bool:
    return abs(actual - expected) <= tol


def task_direction_counts(
    candidate: dict[str, Any], baseline: dict[str, Any], tol: float = 1e-12
) -> dict[str, int]:
    """Count task-level directions from frozen per-task success rates."""
    candidate_rates = candidate["per_task_success_rate"]
    baseline_rates = baseline["per_task_success_rate"]
    require(candidate_rates.keys() == baseline_rates.keys(), "per-task key drift")
    counts = {"better": 0, "equal": 0, "worse": 0}
    for task in candidate_rates:
        delta = candidate_rates[task] - baseline_rates[task]
        direction = "equal" if abs(delta) <= tol else ("better" if delta > 0 else "worse")
        counts[direction] += 1
    return counts


def load_sources() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name, (path, expected_hash) in SOURCES.items():
        require(path.is_file(), f"missing frozen source: {path}")
        actual_hash = sha256(path)
        require(actual_hash == expected_hash, f"{name} SHA drift: {actual_hash}")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    if QVLA_ACTQUANT_AGGREGATE.is_file():
        loaded["qvla_actquant"] = json.loads(
            QVLA_ACTQUANT_AGGREGATE.read_text(encoding="utf-8")
        )
    elif QVLA_ACTQUANT_PARTIAL.is_file():
        loaded["qvla_actquant_partial"] = json.loads(
            QVLA_ACTQUANT_PARTIAL.read_text(encoding="utf-8")
        )
    return loaded


def audit_qvla_actquant(value: dict[str, Any]) -> dict[str, Any]:
    require(value.get("complete") is True, "QVLA/ActQuant aggregate incomplete")
    require(value.get("ready_for_table_update") is True, "QVLA/ActQuant table gate disabled")
    require(value.get("formal_episode_count_new") == 5_000, "ActQuant coverage drift")
    require(value.get("bootstrap_draws") == 10_000, "QVLA/ActQuant bootstrap drift")
    require(value.get("bootstrap_seed") == 0, "QVLA/ActQuant bootstrap seed drift")
    require(len(value.get("holm_family") or []) == 2, "ActQuant Holm family drift")
    expected = {
        "actquant_gr00t", "actquant_pi05",
        "dypac_gr00t", "dypac_pi05",
    }
    arms = value.get("arms") or {}
    require(set(arms) == expected, "QVLA/ActQuant arm inventory drift")
    candidate_records = {}
    for arm in sorted(expected - {"dypac_gr00t", "dypac_pi05"}):
        row = arms[arm]
        require(row.get("episodes") == 2500, f"{arm} coverage drift")
        require(len(row.get("per_task_success_rate") or {}) == 50, f"{arm} task drift")
        require(set((row.get("split_task_macro_success_rate") or {})) == {
            "atomic_seen", "composite_seen", "composite_unseen"
        }, f"{arm} split drift")
        artifacts = row.get("artifacts") or []
        expected_artifacts = 3 if arm.endswith("gr00t") else 1
        require(len(artifacts) == expected_artifacts, f"{arm} artifact coverage drift")
        for artifact in artifacts:
            manifest_path = Path(artifact["pack_manifest"])
            require(manifest_path.is_file(), f"{arm} pack manifest missing")
            require(sha256(manifest_path) == artifact["pack_manifest_sha256"], f"{arm} pack manifest SHA drift")
            pack = json.loads(manifest_path.read_text(encoding="utf-8"))
            require(pack.get("test_results_used") is False, f"{arm} used test feedback")
            require(pack.get("source_protocol_equivalent") is False, f"{arm} source label drift")
            precision = pack.get("model_precision") or {}
            require(float(pack["achieved_bpw"]) <= 4.0 + 1e-6, f"{arm} BPW drift")
            require(
                precision.get("resolved") == "float16"
                and precision.get("strict_all_linear_conv_fp16") is True
                and pack.get("activation_compute_dtype") == "float16",
                f"{arm} ActQuant precision drift",
            )
        candidate_records[arm] = {
            "episodes": row["episodes"],
            "successes": row["successes"],
            "task_macro_success_rate": row["task_macro_success_rate"],
            "micro_success_rate": row["micro_success_rate"],
            "split_task_macro_success_rate": row["split_task_macro_success_rate"],
            "storage": row["storage"],
        }
    for comparison in value.get("holm_family") or []:
        record = (value.get("comparisons") or {}).get(comparison) or {}
        require("exact_two_sided_mcnemar_p" in record, f"{comparison} McNemar missing")
        require("holm_adjusted_mcnemar_p" in record, f"{comparison} Holm p missing")
        bootstrap = record.get("task_then_seed_hierarchical_bootstrap") or {}
        require(bootstrap.get("draws") == 10_000 and bootstrap.get("seed") == 0,
                f"{comparison} bootstrap drift")
    return {
        "aggregate_path": str(QVLA_ACTQUANT_AGGREGATE.relative_to(ROOT)),
        "aggregate_sha256": sha256(QVLA_ACTQUANT_AGGREGATE),
        "calibration_source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "formal_episode_count_new": 5_000,
        "arms": candidate_records,
        "comparison_scope": value["comparison_scope"],
    }


def audit_qvla_actquant_partial(value: dict[str, Any]) -> dict[str, Any]:
    require(
        value.get("kind")
        == "qvla_actquant_robocasa365_table1_partial_snapshot",
        "QVLA/ActQuant partial kind drift",
    )
    require(value.get("complete") is False, "partial snapshot claims completion")
    require(
        value.get("ready_for_final_table_update") is False,
        "partial snapshot enabled the final table gate",
    )
    require(
        value.get("ready_for_partial_table_update") is True,
        "partial table gate disabled",
    )
    require(
        value.get("selection_feedback_allowed") is False
        and value.get("test_results_used_for_pack_or_method_selection") is False,
        "partial snapshot permits test feedback",
    )
    require(
        value.get("paired_inference_reported") is False,
        "partial snapshot must not report paired inference",
    )
    require(
        value.get("protocol_sha256")
        == sha256(ROOT / "scripts/qvla_actquant_table1_protocol.json"),
        "partial protocol SHA drift",
    )
    expected = {"qvla_gr00t", "actquant_gr00t", "qvla_pi05", "actquant_pi05"}
    arms = value.get("arms") or {}
    require(set(arms) == expected, "partial arm inventory drift")
    records = {}
    observed = 0
    for arm in sorted(expected):
        row = arms[arm]
        episodes = int(row.get("episodes", 0))
        require(0 < episodes < 2500, f"{arm} partial coverage drift")
        require(row.get("episodes_expected") == 2500, f"{arm} expected coverage drift")
        require(0 <= int(row.get("successes", -1)) <= episodes, f"{arm} success drift")
        require(0 < int(row.get("tasks_observed", 0)) <= 50, f"{arm} task drift")
        require(row.get("tasks_expected") == 50, f"{arm} task expectation drift")
        require(set(row.get("split_task_macro_success_rate") or {}) == {
            "atomic_seen", "composite_seen", "composite_unseen"
        }, f"{arm} split inventory drift")
        require(set(row.get("split_coverage") or {}) == {
            "atomic_seen", "composite_seen", "composite_unseen"
        }, f"{arm} split coverage drift")
        canonical = Path(row["canonical_jsonl"])
        require(
            canonical.is_file() and sha256(canonical) == row["canonical_jsonl_sha256"],
            f"{arm} partial canonical drift",
        )
        require(
            sum(1 for line in canonical.read_text(encoding="utf-8").splitlines() if line.strip())
            == episodes,
            f"{arm} partial canonical row-count drift",
        )
        manifest = Path(row["arm_manifest"])
        require(
            manifest.is_file() and sha256(manifest) == row["arm_manifest_sha256"],
            f"{arm} arm manifest drift",
        )
        require(row.get("storage") is not None, f"{arm} storage missing")
        observed += episodes
        records[arm] = {
            "episodes": episodes,
            "successes": row["successes"],
            "tasks_observed": row["tasks_observed"],
            "tasks_complete": row["tasks_complete"],
            "task_macro_success_rate": row["task_macro_success_rate"],
            "micro_success_rate": row["micro_success_rate"],
            "split_task_macro_success_rate": row["split_task_macro_success_rate"],
            "split_coverage": row["split_coverage"],
            "storage": row["storage"],
        }
    require(
        observed == value.get("formal_episode_count_observed")
        and value.get("formal_episode_count_expected") == 10_000,
        "partial total coverage drift",
    )
    return {
        "snapshot_path": str(QVLA_ACTQUANT_PARTIAL.relative_to(ROOT)),
        "snapshot_sha256": sha256(QVLA_ACTQUANT_PARTIAL),
        "snapshot_completed_at": value["snapshot_completed_at"],
        "formal_episode_count_observed": observed,
        "formal_episode_count_expected": 10_000,
        "metric_scope": "interim descriptive task-macro over observed rows",
        "paired_inference_reported": False,
        "calibration_source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "arms": records,
    }


def audit_daptq_formal(value: dict[str, Any]) -> dict[str, Any]:
    require(
        value.get("kind") == "daptq_robocasa365_table1_aggregate",
        "DA-PTQ aggregate kind drift",
    )
    require(value.get("complete") is True, "DA-PTQ aggregate incomplete")
    require(value.get("ready_for_table_update") is True, "DA-PTQ table gate disabled")
    require(value.get("formal_episode_count_new") == 5000, "DA-PTQ coverage drift")
    require(value.get("selection_feedback_allowed") is False, "DA-PTQ used formal feedback")
    require(
        value.get("protocol_sha256") == sha256(ROOT / "scripts/daptq_table1_protocol.json"),
        "DA-PTQ protocol SHA drift",
    )
    protocol = json.loads((ROOT / "scripts/daptq_table1_protocol.json").read_text(encoding="utf-8"))
    require(
        protocol["calibration"]["source_protocol_equivalent"] is False,
        "DA-PTQ source-equivalence label drift",
    )
    require(
        protocol["benchmark"]["flow_steps"] == 4
        and protocol["benchmark"]["episodes_per_model_row"] == 2500,
        "DA-PTQ benchmark protocol drift",
    )
    arms = value.get("arms") or {}
    require(
        set(arms)
        == {"daptq_gr00t", "daptq_pi05", "dypac_gr00t", "dypac_pi05"},
        "DA-PTQ arm inventory drift",
    )
    expected = {
        "daptq_gr00t": {
            "successes": 76,
            "w4": 55,
            "bf16": 9,
            "checkpoints": 3,
            "splits": {"atomic_seen": 73 / 900, "composite_seen": 0.0, "composite_unseen": 3 / 800},
        },
        "daptq_pi05": {
            "successes": 655,
            "w4": 87,
            "bf16": 15,
            "checkpoints": 1,
            "splits": {"atomic_seen": 527 / 900, "composite_seen": 98 / 800, "composite_unseen": 30 / 800},
        },
    }
    records = {}
    for name, wanted in expected.items():
        row = arms[name]
        require(row.get("episodes") == 2500, f"{name} coverage drift")
        require(row.get("successes") == wanted["successes"], f"{name} result drift")
        require(
            close(row.get("task_macro_success_rate"), wanted["successes"] / 2500),
            f"{name} macro drift",
        )
        require(len(row.get("per_task_success_rate") or {}) == 50, f"{name} task drift")
        for split, rate in wanted["splits"].items():
            require(
                close(row["split_task_macro_success_rate"][split], rate),
                f"{name} {split} result drift",
            )
        canonical = Path(row["canonical_jsonl"])
        require(
            canonical.is_file() and sha256(canonical) == row["canonical_jsonl_sha256"],
            f"{name} canonical result drift",
        )
        artifacts = row.get("artifacts") or []
        require(len(artifacts) == wanted["checkpoints"], f"{name} checkpoint-count drift")
        for artifact in artifacts:
            require(artifact.get("w4_layers") == wanted["w4"], f"{name} W4 count drift")
            require(artifact.get("bf16_layers") == wanted["bf16"], f"{name} BF16 count drift")
            manifest = Path(artifact["pack_manifest"])
            arrays = Path(artifact["arrays_path"])
            require(
                manifest.is_file() and sha256(manifest) == artifact["pack_manifest_sha256"],
                f"{name} artifact manifest drift",
            )
            require(
                arrays.is_file() and sha256(arrays) == artifact["arrays_sha256"],
                f"{name} artifact array drift",
            )
        storage = row.get("storage") or {}
        require(storage.get("checkpoint_count") == wanted["checkpoints"], f"{name} storage drift")
        records[name] = {
            "episodes": row["episodes"],
            "successes": row["successes"],
            "task_macro_success_rate": row["task_macro_success_rate"],
            "split_task_macro_success_rate": row["split_task_macro_success_rate"],
            "storage": storage,
            "allocation": {"w4_layers": wanted["w4"], "bf16_layers": wanted["bf16"]},
            "source_protocol_equivalent": False,
        }
    comparisons = value.get("comparisons") or {}
    expected_comparisons = {
        "daptq_gr00t_vs_dypac_gr00t": (8, 1282),
        "daptq_pi05_vs_dypac_pi05": (182, 220),
    }
    require(set(comparisons) == set(expected_comparisons), "DA-PTQ comparison inventory drift")
    for name, (wins, losses) in expected_comparisons.items():
        require(
            (comparisons[name].get("paired_wins"), comparisons[name].get("paired_losses"))
            == (wins, losses),
            f"{name} paired-count drift",
        )
    return {
        "formal_episode_count_new": 5000,
        "arms": records,
        "comparisons": {
            name: {
                "paired_wins": row["paired_wins"],
                "paired_losses": row["paired_losses"],
                "holm_adjusted_mcnemar_p": row["holm_adjusted_mcnemar_p"],
            }
            for name, row in comparisons.items()
        },
        "claim_status": "complete_local_cross_architecture_adaptation",
    }


def audit(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    plan = data["gr00t_plan"]
    agg = data["gr00t_aggregate"]
    candidate = agg["candidate"]
    comparisons = agg["comparisons"]
    layers = list(plan["layers"].values())

    require(sum(x["bits"] == 4 and not x["skip"] for x in layers) == 100, "GR00T W4 count drift")
    require(sum(bool(x["skip"]) for x in layers) == 16, "GR00T FP16 count drift")
    require(plan["meta"]["activation_mode"] == "dynamic_a8", "GR00T activation mode drift")
    require(plan["meta"]["runtime_correction"] is False, "runtime correction must remain disabled")
    require(plan["meta"]["runtime_selector"] is False, "runtime selector must remain disabled")
    require(plan["table1_total_static_bytes"] == 962_068_480, "GR00T byte total drift")
    require(close(plan["table1_total_static_compression"], 2.223893051771117), "GR00T compression drift")

    def precision_mask(value: dict[str, Any]) -> dict[str, tuple[Any, bool]]:
        return {
            name: (record.get("bits"), bool(record["skip"]))
            for name, record in value["layers"].items()
        }

    final_mask = precision_mask(plan)
    for key in (
        "gr00t_gdsq_atomic_plan",
        "gr00t_gdsq_composite_seen_plan",
        "gr00t_gdsq_composite_unseen_plan",
    ):
        require(
            precision_mask(data[key]) == final_mask,
            f"{key} no longer matches the frozen GR00T mask",
        )

    require(candidate["episodes"] == 2500 and candidate["successes"] == 1350, "GR00T coverage drift")
    require(close(candidate["task_macro_success_rate"], 0.54), "GR00T headline drift")
    fp16 = comparisons["fp16"]
    quantvla = comparisons["quantvla_w4a8"]
    gdsq = comparisons["gdsq_vla_main"]
    quant_task_direction = task_direction_counts(candidate, quantvla["baseline"])
    fp16_task_direction = task_direction_counts(candidate, fp16["baseline"])
    require(close(fp16["holm_adjusted_mcnemar_p"], 0.29286269346886834), "FP16 p-value drift")
    require(close(quantvla["baseline"]["task_macro_success_rate"], 0.3044), "QuantVLA result drift")
    require((quantvla["paired_wins"], quantvla["paired_losses"]) == (721, 132), "QuantVLA discordance drift")
    require(
        close(quantvla["holm_adjusted_mcnemar_p"], 1.8894854853781264e-98),
        "QuantVLA p-value drift",
    )
    require(quant_task_direction == {"better": 48, "equal": 1, "worse": 1}, "QuantVLA task-direction drift")
    require(fp16_task_direction == {"better": 18, "equal": 6, "worse": 26}, "FP16 task-direction drift")
    require(agg["formal_superiority_over_gdsq_main"] is True, "GDSQ comparison status drift")
    require(
        gdsq["baseline"]["episodes"] == 2500
        and gdsq["baseline"]["successes"] == 1269
        and close(gdsq["baseline"]["task_macro_success_rate"], 0.5076),
        "GDSQ predecessor result drift",
    )
    require((gdsq["paired_wins"], gdsq["paired_losses"]) == (370, 289), "GDSQ discordance drift")
    require(close(gdsq["holm_adjusted_mcnemar_p"], 0.003619369860097224), "GDSQ p-value drift")
    gdsq_bootstrap = gdsq["task_then_seed_hierarchical_bootstrap"]
    require(
        close(gdsq_bootstrap["ci95_low"], 0.0048)
        and close(gdsq_bootstrap["ci95_high"], 0.0604),
        "GDSQ bootstrap interval drift",
    )

    ratio_selection = data["gr00t_ratio_selection"]
    require(
        ratio_selection.get("complete") is True
        and ratio_selection.get("kind") == "gdsq_ratio_stability_audit"
        and ratio_selection["coverage"] == {
            "configs": 7,
            "duplicates": 0,
            "episodes_per_config": 200,
            "missing": 0,
            "seeds_per_task": 50,
            "tasks": 4,
        },
        "GR00T initializer development sweep drift",
    )
    development_tasks = set(ratio_selection["development_tasks"])
    require(len(development_tasks) == 4, "GR00T development-task inventory drift")
    require(
        ratio_selection["observed"]["cscka_16to1"]["successes"] == 161,
        "GR00T 16:1 development result drift",
    )

    def heldout46_rate(value: dict[str, Any]) -> float:
        per_task = value["per_task_success_rate"]
        require(development_tasks < set(per_task), "development tasks missing from Table-1 aggregate")
        heldout = [rate for task, rate in per_task.items() if task not in development_tasks]
        require(len(heldout) == 46, "held-out task count drift")
        return sum(heldout) / len(heldout)

    heldout46 = {
        "ours": heldout46_rate(candidate),
        "quantvla_w4a8": heldout46_rate(quantvla["baseline"]),
        "fp16": heldout46_rate(fp16["baseline"]),
    }
    require(
        close(heldout46["ours"], 0.5178260869565217)
        and close(heldout46["quantvla_w4a8"], 0.2756521739130434)
        and close(heldout46["fp16"], 0.5278260869565216),
        "GR00T held-out-46 recomputation drift",
    )

    table3 = data["table3_closed_loop"]
    require(
        table3.get("kind") == "gr00t_table3_reduced_closed_loop_aggregate"
        and table3.get("complete") is True,
        "Table-3 closed-loop aggregate incomplete",
    )
    require(
        table3["coverage"]
        == {
            "episodes_per_config": 500,
            "new_episode_configs": 3,
            "new_episodes": 1500,
            "seeds_per_task": 10,
            "tasks": 50,
        },
        "Table-3 coverage drift",
    )
    require(
        table3["baseline"]["episodes"] == 500
        and table3["baseline"]["successes"] == 269
        and close(table3["baseline"]["task_macro_success_rate"], 0.538),
        "Table-3 baseline drift",
    )
    expected_table3 = {
        "static_a8": (0, 0.0),
        "local_mse_selection": (265, 0.53),
        "no_fullnet_check": (265, 0.53),
    }
    require(set(table3["arms"]) == set(expected_table3), "Table-3 arm inventory drift")
    for arm, (successes, rate) in expected_table3.items():
        row = table3["arms"][arm]
        require(
            row["episodes"] == 500
            and row["successes"] == successes
            and close(row["task_macro_success_rate"], rate),
            f"Table-3 {arm} result drift",
        )

    activation = data["activation_attribution"]
    require(activation["selected_activation_mode"] == "dynamic_a8", "activation decision drift")
    correction = data["statistics_correction"]
    require(
        correction.get("kind") == "full_context_statistics_correction_v1"
        and correction.get("closed_loop_results_changed") is False
        and correction.get("frozen_plan_changed") is False,
        "statistics-correction scope drift",
    )
    local = correction["local_interventions"]
    require(local["n_flips"] == 116, "corrected flip coverage drift")
    require(local["n_positive_both_metrics"] == 2, "corrected positive-flip count drift")
    corrected_fullnet = correction["existing_complete_policy_candidates"]
    require(corrected_fullnet["n_candidates"] == 5, "corrected full-network coverage drift")
    require(corrected_fullnet["n_eligible"] == 0, "corrected full-network eligibility drift")
    corrected_activation = correction["activation_attribution"]
    require(close(corrected_activation["static_vs_a16"]["objective"], 264.93584915146585), "corrected static-A8 objective drift")
    require(close(corrected_activation["dynamic_vs_a16"]["objective"], 0.06592289711881642), "corrected DyRange/A16 objective drift")
    require(close(corrected_activation["dynamic_vs_static"]["objective"], -0.24143020495712125), "corrected DyRange/static objective drift")

    corrected_fcp = data["corrected_fcp_selection"]
    require(
        corrected_fcp.get("kind") == "corrected_full_context_frozen_selection_report",
        "corrected FCP selection kind drift",
    )
    fcp_selection = corrected_fcp["selection"]
    require(
        fcp_selection.get("selected_id") == "context_base"
        and fcp_selection.get("fallback_to_baseline") is True
        and fcp_selection.get("reason")
        == "no_candidate_has_negative_paired_minimax_and_component_safety",
        "corrected FCP abstention drift",
    )
    expected_fcp_candidates = {
        "single_best",
        "two_best",
        "attention_6",
        "mlp_2",
        "ff_pair_0",
        "dp_full_lambda_1p0",
    }
    fcp_summaries = fcp_selection["summaries"]
    require(set(fcp_summaries) == expected_fcp_candidates, "corrected FCP inventory drift")
    require(
        all(
            row.get("eligible") is False
            and row.get("component_constraints_pass") is False
            and float(row["objective"]) > 0.0
            for row in fcp_summaries.values()
        ),
        "corrected FCP failed-guard evidence drift",
    )
    require(
        close(min(row["objective"] for row in fcp_summaries.values()), 0.9483594348498502)
        and close(max(row["objective"] for row in fcp_summaries.values()), 2.36031518928543),
        "corrected FCP objective range drift",
    )
    fullnet_path = Path(corrected_fcp["full_network_scores"])
    require(
        fullnet_path.is_file()
        and sha256(fullnet_path) == corrected_fcp["full_network_scores_sha256"],
        "corrected FCP full-policy scores drift",
    )

    fcp_rollouts = data["fcp_candidate_closed_loop"]
    require(
        fcp_rollouts.get("kind")
        == "gr00t_fcp_candidate_reduced_closed_loop_aggregate"
        and fcp_rollouts.get("complete") is True,
        "FCP candidate closed-loop aggregate incomplete",
    )
    require(
        fcp_rollouts.get("coverage")
        == {
            "tasks": 50,
            "seeds_per_task": 10,
            "episodes_per_config": 500,
            "new_episode_configs": 6,
            "new_episodes": 3000,
        },
        "FCP candidate closed-loop coverage drift",
    )
    preregistration_path = Path(fcp_rollouts["manifest"]["path"])
    require(
        preregistration_path.is_file()
        and sha256(preregistration_path) == fcp_rollouts["manifest"]["sha256"],
        "FCP candidate preregistration drift",
    )
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    require(
        preregistration.get("immutable") is True
        and preregistration.get("result_feedback_allowed") is False
        and preregistration["selection"]
        == {
            "artifact": preregistration["selection"]["artifact"],
            "candidate_state_audit_triggered": False,
            "closed_loop_role": "post-selection diagnostic only",
            "result": "context_base",
        },
        "FCP candidate diagnostic leaked into frozen selection",
    )
    require(
        set(fcp_rollouts["candidates"]) == expected_fcp_candidates
        and fcp_rollouts["baseline"]["episodes"] == 500
        and fcp_rollouts["baseline"]["successes"] == 269
        and close(fcp_rollouts["baseline"]["task_macro_success_rate"], 0.538),
        "FCP candidate diagnostic inventory or baseline drift",
    )
    expected_candidate_rollouts = {
        "single_best": (274, 0.548, 59, 54, 1.0),
        "two_best": (253, 0.506, 46, 62, 0.8914580856633723),
        "attention_6": (278, 0.556, 56, 47, 1.0),
        "mlp_2": (265, 0.530, 53, 57, 1.0),
        "ff_pair_0": (281, 0.562, 59, 47, 1.0),
        "dp_full_lambda_1p0": (264, 0.528, 54, 59, 1.0),
    }
    for candidate_id, (successes, rate, wins, losses, holm_p) in expected_candidate_rollouts.items():
        row = fcp_rollouts["candidates"][candidate_id]
        comparison = fcp_rollouts["comparisons"][candidate_id]
        require(
            row["episodes"] == 500
            and row["successes"] == successes
            and close(row["task_macro_success_rate"], rate),
            f"FCP candidate {candidate_id} result drift",
        )
        require(
            comparison["paired_candidate_wins"] == wins
            and comparison["paired_candidate_losses"] == losses
            and comparison["paired_ties"] == 500 - wins - losses
            and close(comparison["task_macro_delta"], rate - 0.538)
            and close(comparison["holm_adjusted_p"], holm_p)
            and close(
                fcp_rollouts["multiplicity"]["adjusted_p"][candidate_id],
                holm_p,
            )
            and comparison["hierarchical_bootstrap"]["draws"] == 10_000
            and comparison["hierarchical_bootstrap"]["seed"] == 0,
            f"FCP candidate {candidate_id} comparison drift",
        )
    require(
        fcp_rollouts["multiplicity"]["method"] == "Holm"
        and fcp_rollouts["multiplicity"]["family"]
        == preregistration["evaluation"]["candidate_ids"]
        and fcp_rollouts["reporting_scope"]
        == (
            "Post-selection 10-seed diagnostic only. Candidate success labels did not "
            "enter the already frozen FCP decision; non-significance is not equivalence."
        ),
        "FCP candidate inference scope drift",
    )
    for sources in fcp_rollouts["sources"].values():
        for source in sources:
            path = Path(source["path"])
            require(
                path.is_file() and sha256(path) == source["sha256"],
                "FCP candidate rollout source drift",
            )

    hardware = data["gr00t_hardware"]
    require(
        hardware.get("kind") == "gr00t_real_hardware_summary"
        and hardware.get("complete") is True,
        "GR00T hardware aggregate incomplete",
    )
    hardware_protocol = ROOT / "scripts/quantvla_fcp_hardware_protocol.json"
    require(
        hardware["protocol"]["sha256"] == sha256(hardware_protocol),
        "GR00T hardware protocol drift",
    )
    expected_hardware_configs = expected_fcp_candidates | {
        "native_fp16", "quantvla_w4a8", "context_base"
    }
    hardware_summaries = hardware["summaries"]
    require(set(hardware_summaries) == expected_hardware_configs, "hardware inventory drift")
    for config_id, row in hardware_summaries.items():
        require(row["trials"] == 3 and row["requests"] == 180, f"{config_id} hardware coverage drift")
        sources = hardware["sources"][config_id]
        require([source["trial"] for source in sources] == [0, 1, 2], f"{config_id} trial order drift")
        for source in sources:
            path = Path(source["path"])
            require(path.is_file() and sha256(path) == source["sha256"], f"{config_id} trial hash drift")
    native_hardware = hardware_summaries["native_fp16"]
    context_hardware = hardware_summaries["context_base"]
    require(
        close(native_hardware["latency_ms"]["median_of_trial_p50"], 107.46477358043194)
        and close(native_hardware["peak_memory"]["allocated_gib_median"], 5.3858184814453125)
        and close(native_hardware["energy_joules_per_request"]["gross_board"]["mean"], 18.645160458035484),
        "native hardware anchor drift",
    )
    require(
        close(context_hardware["latency_ms"]["median_of_trial_p50"], 232.22207557410002)
        and close(context_hardware["peak_memory"]["allocated_gib_median"], 4.266719818115234)
        and close(context_hardware["energy_joules_per_request"]["gross_board"]["mean"], 39.80961975044194),
        "context-base hardware anchor drift",
    )

    integer_backend = data["integer_w4a8_backend"]
    require(
        integer_backend.get("kind")
        == "dypac_vla_integer_w4a8_backend_paper_audit"
        and integer_backend.get("complete") is True,
        "integer-W4A8 paper audit incomplete",
    )
    require(
        integer_backend["coverage"]
        == {
            "tasks": 50,
            "seeds_per_task": 10,
            "episodes_per_config": 500,
            "duplicates": 0,
            "missing": 0,
            "formal_terminal_failures": 0,
        },
        "integer-W4A8 coverage drift",
    )
    integer_control = integer_backend["control"]
    integer_candidate = integer_backend["candidate"]
    integer_paired = integer_backend["paired_comparison"]
    require(
        integer_control["successes"] == 269
        and integer_control["episodes"] == 500
        and close(integer_control["success_rate"], 0.538)
        and integer_candidate["successes"] == 251
        and integer_candidate["episodes"] == 500
        and close(integer_candidate["success_rate"], 0.502),
        "integer-W4A8 closed-loop result drift",
    )
    require(
        integer_candidate["integer_gemm"] is True
        and integer_candidate["packed_low_bit_residency"] is True
        and integer_candidate["execution_backend"]
        == "triton_packed_w4_dynamic_a8_int8_gemm_int32_accum"
        and integer_backend["runtime_contract_audit"]["passed"] is True,
        "integer-W4A8 runtime contract drift",
    )
    require(
        (integer_paired["candidate_wins"], integer_paired["candidate_losses"], integer_paired["ties"])
        == (36, 54, 410)
        and close(integer_paired["delta"], -0.036)
        and close(integer_paired["exact_two_sided_mcnemar_p"], 0.07254953219246181)
        and integer_paired["task_cluster_bootstrap"]
        == {
            "draws": 10000,
            "seed": 20260903,
            "unit": "task_cluster_with_seed_resampling",
            "ci95_low": -0.088,
            "ci95_high": 0.016,
        }
        and integer_paired["equivalence_established"] is False
        and integer_paired["noninferiority_established"] is False,
        "integer-W4A8 paired inference drift",
    )
    integer_hardware = integer_backend["hardware"]
    require(
        integer_hardware["quality_gate_passed"] is True
        and integer_hardware["trials_per_backend"] == 3
        and integer_hardware["requests_per_trial"] == 60
        and close(integer_hardware["reference"]["p50_ms"], 229.6646423637867)
        and close(integer_hardware["candidate"]["p50_ms"], 187.46585119515657)
        and close(integer_hardware["candidate_over_reference"]["p50_ratio"], 0.8162590865781263)
        and close(integer_hardware["candidate_over_reference"]["gross_energy_ratio"], 0.882471581434594)
        and close(integer_hardware["candidate_over_reference"]["idle_adjusted_energy_ratio"], 0.9261334141603386)
        and close(integer_hardware["candidate_over_reference"]["peak_cuda_allocated_ratio"], 1.0),
        "integer-W4A8 hardware result drift",
    )
    require(
        integer_backend["claim_boundary"]
        == {
            "paper_role": "separately_named_exploratory_real_integer_backend",
            "may_replace_frozen_ours": False,
            "may_reopen_frozen_fcp_selection": False,
            "reason": (
                "The integer backend changes dynamic-A8 scale granularity from "
                "per-input-channel to per-row group-64."
            ),
        },
        "integer-W4A8 claim boundary drift",
    )

    predictive = data["predictive_validity"]
    require(
        predictive.get("kind") == "gr00t_dpac_predictive_validity_audit",
        "predictive-validity audit kind drift",
    )
    coverage = predictive["coverage_gate"]
    require(
        coverage == {
            "candidate_count": 60,
            "fp16_rows": 50,
            "mask_rows": 3000,
            "no_duplicate_failed_or_drifted_rows": True,
            "passed": True,
            "task_count": 50,
        },
        "predictive-validity coverage drift",
    )
    predictive_primary = predictive["analysis"]["primary"]
    predictive_expected = {
        "mse": (-0.027902032002688378, -0.033636090705897105),
        "d_func": (-0.06534801861394221, -0.051655425012627695),
        "d_pac": (-0.11576255531095257, -0.0792850709496146),
    }
    for metric, (rho, tau) in predictive_expected.items():
        require(
            predictive_primary[metric]["spearman_rho"]["status"] == "defined"
            and predictive_primary[metric]["kendall_tau_b"]["status"] == "defined"
            and close(predictive_primary[metric]["spearman_rho"]["value"], rho)
            and close(predictive_primary[metric]["kendall_tau_b"]["value"], tau),
            f"{metric} predictive-validity drift",
        )
    predictive_gaps = predictive["analysis"]["primary_gaps"]
    require(
        close(
            predictive_gaps["mse"]["delta_spearman_rho"]["value"],
            -0.08786052330826419,
        )
        and close(
            predictive_gaps["d_func"]["delta_spearman_rho"]["value"],
            -0.05041453669701036,
        ),
        "predictive-validity gap drift",
    )

    signal_ablation = data["signal_ablation"]
    signal_masks = data["signal_ablation_masks"]
    signal_prereg = data["signal_ablation_preregistration"]
    require(
        signal_ablation["coverage"]
        == {
            "complete": True,
            "duplicates": 0,
            "missing": 0,
            "units": 1500,
        },
        "signal-ablation coverage drift",
    )
    require(
        signal_prereg.get("kind") == "robocasa365_byte_ablation_preregistration"
        and signal_prereg.get("experiment_id") == "robocasa365_byte_ablation_v1"
        and signal_prereg.get("model") == "gr00t_n1-5"
        and signal_prereg["protocol"]["seeds"] == "0-9"
        and signal_prereg["masks"]["manifest_sha256"]
        == SOURCES["signal_ablation_masks"][1],
        "signal-ablation preregistration drift",
    )
    require(
        signal_masks.get("kind") == "byte_ablation_masks"
        and signal_masks["total_static_bytes"] == 962_068_480
        and close(signal_masks["compression"], 2.223893051771117)
        and set(signal_masks["arms"]) == {"cs_cka", "cka_only", "cs_only"},
        "signal-ablation mask inventory or byte budget drift",
    )
    expected_signal_arms = {
        "cs_cka": (93, 23, 0.5326086956521738, 0.552),
        "cka_only": (94, 22, 0.5152173913043477, 0.532),
        "cs_only": (92, 24, 0.5065217391304347, 0.524),
    }
    for arm, (w4, fp16_count, primary46, all50) in expected_signal_arms.items():
        mask = signal_masks["arms"][arm]
        result_arm = signal_ablation["arms"][arm]
        require(
            mask["w4_count"] == w4
            and mask["fp16_count"] == fp16_count
            and mask["byte_total"] == signal_masks["budget_bytes"]
            and mask["matches_expected_counts"] is True
            and close(result_arm["primary46_task_macro_sr"], primary46)
            and close(result_arm["all50_task_macro_sr"], all50),
            f"signal-ablation {arm} drift",
        )
    expected_signal_comparisons = {
        "cs_cka_vs_cka_only": (0.017391304347826098, 16, 13, 17),
        "cs_cka_vs_cs_only": (0.02608695652173909, 21, 14, 11),
    }
    for comparison_id, (delta, wins, losses, ties) in expected_signal_comparisons.items():
        comparison = signal_ablation["comparisons"][comparison_id]
        require(
            close(comparison["delta_task_macro_sr"], delta)
            and close(comparison["holm_p"], 0.5795964308101232)
            and (
                comparison["paired_wins"],
                comparison["paired_losses"],
                comparison["paired_ties"],
            )
            == (wins, losses, ties),
            f"signal-ablation {comparison_id} drift",
        )
    require(
        signal_ablation["decision"]["conclusion"] == "directional_evidence_only",
        "signal-ablation claim boundary drift",
    )

    compression_sweep = data["compression_sweep"]
    compression_masks = data["compression_sweep_masks"]
    compression_prereg = data["compression_sweep_preregistration"]
    require(
        compression_sweep["coverage"]
        == {
            "complete": True,
            "duplicates": 0,
            "missing": 0,
            "units": 3000,
        },
        "compression-sweep coverage drift",
    )
    require(
        compression_prereg.get("kind")
        == "robocasa365_compression_sweep_preregistration"
        and compression_prereg.get("experiment_id")
        == "robocasa365_compression_sweep_v1"
        and compression_prereg.get("model") == "gr00t_n1-5"
        and compression_prereg["protocol"]["seeds"] == "0-9"
        and compression_prereg["protocol"]["tasks"]
        == "all 50 RoboCasa365 tasks"
        and compression_prereg["masks"]["manifest_sha256"]
        == SOURCES["compression_sweep_masks"][1]
        and len(compression_prereg["hessian_subsets"]) == 18
        and "no success labels" in compression_prereg["selection_rule"]
        and compression_prereg["statistics_plan"]["claims"]
        == "descriptive compression-SR curve; no new significance tests",
        "compression-sweep preregistration drift",
    )
    require(
        compression_masks.get("kind") == "compression_sweep_masks"
        and compression_masks["rates"]
        == [1.2, 1.6, 2.0, 2.4, 2.8, 3.5555555555555554]
        and close(
            compression_masks["maximum_candidate_compression"],
            3.5555555555555554,
        )
        and compression_masks["fixed_bytes"] == 327_352_320
        and set(compression_masks["rates_manifest"])
        == {"rate12", "rate16", "rate20", "rate24", "rate28", "ratemax"},
        "compression-sweep mask inventory drift",
    )
    expected_compression_arms = {
        "rate12": (
            1.2, 30, 86, 1_834_811_392, 1.1660802943172484,
            1.201982436309886, 0.554,
            {"atomic_seen": 0.7444444444444447, "composite_seen": 0.4625000000000001, "composite_unseen": 0.43125},
        ),
        "rate16": (
            1.6, 63, 53, 1_459_486_720, 1.4659519533004042,
            1.6004630969609261, 0.528,
            {"atomic_seen": 0.7166666666666667, "composite_seen": 0.4375, "composite_unseen": 0.40625000000000006},
        ),
        "rate20": (
            2.0, 80, 36, 1_230_372_864, 1.7389341642697347,
            2.006531678641411, 0.546,
            {"atomic_seen": 0.7444444444444445, "composite_seen": 0.41875, "composite_unseen": 0.45000000000000007},
        ),
        "rate24": (
            2.4, 91, 25, 1_081_147_392, 1.9789507183124204,
            2.403755868544601, 0.514,
            {"atomic_seen": 0.7166666666666666, "composite_seen": 0.42500000000000004, "composite_unseen": 0.3750000000000001},
        ),
        "rate28": (
            2.8, 98, 18, 974_127_104, 2.1963636975242196,
            2.801499645354139, 0.540,
            {"atomic_seen": 0.7388888888888889, "composite_seen": 0.44375000000000003, "composite_unseen": 0.41250000000000003},
        ),
        "ratemax": (
            3.5555555555555554, 116, 0, 836_960_256, 2.5563190039934227,
            3.5555555555555554, 0.554,
            {"atomic_seen": 0.7166666666666667, "composite_seen": 0.48125000000000007, "composite_unseen": 0.4437500000000001},
        ),
    }
    require(
        set(compression_sweep["rates"]) == set(expected_compression_arms),
        "compression-sweep aggregate arm inventory drift",
    )
    for arm, expected in expected_compression_arms.items():
        (
            target, w4, fp16_count, static_bytes, static_compression,
            achieved_candidate_compression, all50, splits,
        ) = expected
        aggregate_arm = compression_sweep["rates"][arm]
        mask_arm = compression_masks["rates_manifest"][arm]
        plan_path = (
            ROOT
            / "runs/robocasa365_compression_sweep_v1/masks"
            / f"{arm}.plan.json"
        )
        require(
            aggregate_arm["w4_layers"] == mask_arm["w4_layers"] == w4
            and aggregate_arm["fp16_layers"] == mask_arm["fp16_layers"] == fp16_count
            and aggregate_arm["static_bytes"] == mask_arm["static_bytes"] == static_bytes
            and close(aggregate_arm["rate"], target)
            and close(mask_arm["rate"], target)
            and close(aggregate_arm["static_compression"], static_compression)
            and close(mask_arm["static_compression"], static_compression)
            and close(
                aggregate_arm["achieved_candidate_compression"],
                achieved_candidate_compression,
            )
            and close(
                mask_arm["achieved_candidate_compression"],
                achieved_candidate_compression,
            )
            and close(aggregate_arm["all50_task_macro_sr"], all50)
            and all(close(aggregate_arm["splits"][split], value) for split, value in splits.items())
            and plan_path.is_file()
            and sha256(plan_path) == mask_arm["plan_sha256"],
            f"compression-sweep {arm} drift",
        )
    sweep_reference = compression_sweep["reference_ours"]
    require(
        sweep_reference["w4"] == 100
        and sweep_reference["fp16"] == 16
        and sweep_reference["static_bytes"] == plan["table1_total_static_bytes"]
        and close(sweep_reference["static_compression"], 2.2239, tol=1e-4)
        and close(sweep_reference["all50_task_macro_sr"], candidate["task_macro_success_rate"]),
        "compression-sweep frozen reference drift",
    )
    sweep_rows = [
        compression_sweep["rates"][arm]
        for arm in ("rate12", "rate16", "rate20", "rate24", "rate28", "ratemax")
    ]
    require(
        all(
            left["static_bytes"] > right["static_bytes"]
            for left, right in zip(sweep_rows, sweep_rows[1:])
        )
        and close(
            max(row["all50_task_macro_sr"] for row in sweep_rows)
            - min(row["all50_task_macro_sr"] for row in sweep_rows),
            0.04,
        )
        and close(sweep_rows[0]["all50_task_macro_sr"], sweep_rows[-1]["all50_task_macro_sr"]),
        "compression-sweep plateau drift",
    )

    pi_plan = data["pi05_plan"]
    pi_layers = list(pi_plan["layers"].values())
    require(sum(x["bits"] == 4 and not x["skip"] for x in pi_layers) == 121, "pi0.5 W4 count drift")
    require(sum(bool(x["skip"]) for x in pi_layers) == 59, "pi0.5 FP16 count drift")
    require(pi_plan["table1_total_static_bytes"] == 1_634_828_288, "pi0.5 byte total drift")
    require(close(pi_plan["table1_total_static_compression"], 2.701569421338518), "pi0.5 compression drift")

    pi_main = data["pi05_main_plan"]
    require(pi_main["meta"]["quantized_layers"] == 80 and pi_main["retained_fp16_layers"] == 100, "pi0.5 main mask drift")
    require(pi_main["table1_total_static_bytes"] == 1_845_100_544, "pi0.5 main byte drift")
    pi_pruned = data["pi05_pruned_plan"]
    pi_main_fp16 = {
        name for name, row in pi_main["layers"].items() if bool(row.get("skip"))
    }
    pi_pruned_fp16 = {
        name for name, row in pi_pruned["layers"].items() if bool(row.get("skip"))
    }
    pi_final_fp16 = {
        name for name, row in pi_plan["layers"].items() if bool(row.get("skip"))
    }
    removed_protections = pi_main_fp16 - pi_pruned_fp16
    require(
        pi_pruned_fp16 < pi_main_fp16
        and len(removed_protections) == 41
        and not (pi_pruned_fp16 - pi_main_fp16),
        "pi0.5 feasibility projection mask drift",
    )
    require(
        set(pi_pruned["meta"]["pruned_layers"]) == removed_protections
        and pi_pruned["meta"]["already_within_budget"] is False,
        "pi0.5 feasibility projection provenance drift",
    )
    require(
        pi_main["table1_total_static_bytes"]
        > pi_main["table1_total_static_budget_bytes"]
        and pi_pruned["table1_total_static_bytes"]
        <= pi_pruned["table1_total_static_budget_bytes"],
        "pi0.5 feasibility transition drift",
    )
    require(pi_final_fp16 == pi_pruned_fp16, "pi0.5 frozen anchor mask drift")
    pi_selection = pi_plan["meta"]["selection_result"]
    require(
        pi_selection.get("selected_id") == "context_base"
        and pi_selection.get("fallback_to_baseline") is True
        and pi_selection.get("reason")
        == "no_candidate_has_negative_paired_minimax_and_component_safety",
        "pi0.5 post-projection FCP abstention drift",
    )
    require(
        set(pi_selection["summaries"]) == {"single_best", "two_best"}
        and all(
            row.get("eligible") is False
            and row.get("component_constraints_pass") is False
            and float(row["objective"]) > 0.0
            for row in pi_selection["summaries"].values()
        ),
        "pi0.5 post-projection proposal evidence drift",
    )

    pi_diag_raw = data["pi05_fcp_diagnostic_raw"]
    pi_diag = data["pi05_fcp_diagnostic_summary"]
    require(
        pi_diag_raw.get("kind") == "pi05_fcp_same_protocol_raw_aggregate"
        and pi_diag_raw.get("complete") is True
        and pi_diag.get("kind") == "pi05_fcp_same_protocol_experiment_summary"
        and pi_diag.get("complete") is True,
        "pi0.5 FCP same-protocol diagnostic incomplete",
    )
    require(
        pi_diag_raw.get("coverage")
        == {
            "new_rollouts": 1500,
            "reused_anchor_rollouts": 500,
            "unique_keys_per_configuration": 500,
        }
        and pi_diag["raw_aggregate"]["sha256"]
        == SOURCES["pi05_fcp_diagnostic_raw"][1],
        "pi0.5 FCP diagnostic coverage or source drift",
    )
    expected_pi_diag = {
        "transferred_initializer": {
            "role": "pre_projection_initializer",
            "new_rollouts": True,
            "w4": 80,
            "fp16": 100,
            "bytes": 1_845_100_544,
            "budget": False,
            "successes": 130,
            "rate": 0.260,
        },
        "projected_anchor": {
            "role": "post_projection_frozen_anchor",
            "new_rollouts": False,
            "w4": 121,
            "fp16": 59,
            "bytes": 1_634_828_288,
            "budget": True,
            "successes": 141,
            "rate": 0.282,
        },
        "single_best": {
            "role": "post_selection_structural_candidate",
            "new_rollouts": True,
            "w4": 179,
            "fp16": 1,
            "bytes": 1_290_403_840,
            "budget": True,
            "successes": 124,
            "rate": 0.248,
        },
        "two_best": {
            "role": "post_selection_structural_candidate",
            "new_rollouts": True,
            "w4": 178,
            "fp16": 2,
            "bytes": 1_338_638_336,
            "budget": True,
            "successes": 125,
            "rate": 0.250,
        },
    }
    require(
        set(pi_diag["configurations"]) == set(expected_pi_diag)
        and set(pi_diag_raw["outcomes"]) == set(expected_pi_diag),
        "pi0.5 FCP diagnostic inventory drift",
    )
    for config_id, expected in expected_pi_diag.items():
        config = pi_diag["configurations"][config_id]
        result = config["result"]
        outcomes = pi_diag_raw["outcomes"][config_id]
        keys = {
            (row["task_set"], row["task"], int(row["seed"]))
            for row in outcomes
        }
        tasks = {(row["task_set"], row["task"]) for row in outcomes}
        require(
            config["role"] == expected["role"]
            and config["new_rollouts"] is expected["new_rollouts"]
            and config["w4_layers"] == expected["w4"]
            and config["fp16_layers"] == expected["fp16"]
            and config["table1_total_static_bytes"] == expected["bytes"]
            and config["deployment_budget_compliant"] is expected["budget"]
            and result["episodes"] == 500
            and result["successes"] == expected["successes"]
            and close(result["task_macro_success_rate"], expected["rate"]),
            f"pi0.5 FCP diagnostic {config_id} result drift",
        )
        require(
            len(outcomes) == len(keys) == 500
            and len(tasks) == 50
            and {int(row["seed"]) for row in outcomes} == set(range(10))
            and sum(bool(row["success"]) for row in outcomes)
            == expected["successes"]
            and len({row["server_metadata_sha256"] for row in outcomes}) == 1,
            f"pi0.5 FCP diagnostic {config_id} key/runtime drift",
        )

    pi_primary = pi_diag["primary_comparison"]
    require(
        pi_primary["orientation"]
        == "projected_anchor minus transferred_initializer"
        and (pi_primary["paired_left_wins"], pi_primary["paired_left_losses"], pi_primary["paired_ties"])
        == (41, 30, 429)
        and close(pi_primary["task_macro_delta"], 0.022)
        and close(pi_primary["exact_two_sided_mcnemar_p"], 0.23509756240834184),
        "pi0.5 FCP projection comparison drift",
    )
    expected_pi_secondary = {
        "single_best": (23, 40, 437, -0.034, 0.08591309104877842),
        "two_best": (25, 41, 434, -0.032, 0.08591309104877842),
    }
    for config_id, (wins, losses, ties, delta, holm_p) in expected_pi_secondary.items():
        comparison = pi_diag["secondary_comparisons"][config_id]
        require(
            (comparison["paired_left_wins"], comparison["paired_left_losses"], comparison["paired_ties"])
            == (wins, losses, ties)
            and close(comparison["task_macro_delta"], delta)
            and close(comparison["holm_adjusted_p"], holm_p),
            f"pi0.5 FCP diagnostic {config_id} comparison drift",
        )
    require(
        pi_diag["multiplicity"]["primary_family"]
        == "singleton; no multiplicity adjustment"
        and pi_diag["multiplicity"]["secondary_method"] == "Holm"
        and pi_diag["interpretation"]
        == (
            "Post-selection mechanism diagnostic only; it cannot revise the frozen pi0.5 choice. "
            "Non-significance does not establish equivalence. The transferred initializer "
            "violates the deployment byte budget."
        ),
        "pi0.5 FCP diagnostic claim boundary drift",
    )
    pi_diag_prereg_path = Path(pi_diag_raw["preregistration"]["path"])
    pi_diag_prereg = json.loads(pi_diag_prereg_path.read_text(encoding="utf-8"))
    require(
        pi_diag_prereg_path.is_file()
        and sha256(pi_diag_prereg_path) == pi_diag_raw["preregistration"]["sha256"]
        and pi_diag_prereg.get("immutable") is True
        and pi_diag_prereg.get("result_feedback_allowed") is False
        and pi_diag_prereg["acceptance"]["write_paper_only_after_complete_audit"] is True,
        "pi0.5 FCP diagnostic preregistration drift",
    )

    pi_dypac = data["pi05_dypac_formal"]
    pi_dypac_result = pi_dypac["result"]
    require(pi_dypac["complete"] is True, "pi0.5 DyPAC aggregate incomplete")
    require(
        pi_dypac["model"] == "pi0.5" and pi_dypac["method"] == "DyPAC-VLA",
        "pi0.5 DyPAC identity drift",
    )
    require(pi_dypac_result["episodes"] == 2500, "pi0.5 DyPAC coverage drift")
    require(pi_dypac_result["successes"] == 693, "pi0.5 DyPAC success-count drift")
    require(close(pi_dypac_result["task_macro_success_rate"], 0.2772), "pi0.5 DyPAC headline drift")
    require(
        pi_dypac_result["split_task_macro_success_rate"]
        == {
            "atomic_seen": 0.5955555555555555,
            "composite_seen": 0.15375,
            "composite_unseen": 0.042499999999999996,
        },
        "pi0.5 DyPAC split drift",
    )

    pi_static = data["pi05_static_official"]
    pi_fp16 = pi_static["configs"]["fp16"]
    pi_quant = pi_static["configs"]["quantvla_w4a8_atmohb"]
    pi_w6 = data["pi05_uniform_w6"]["configs"]["uniform_w6"]
    pi_omega = data["pi05_omega"]
    for name, row in (("FP16", pi_fp16), ("QuantVLA", pi_quant), ("Uniform W6", pi_w6)):
        require(row["completed_episodes"] == 2500, f"pi0.5 {name} coverage drift")
    require(pi_omega["complete"] is True and pi_omega["episodes"] == 2500, "pi0.5 Omega coverage drift")
    require(close(pi_fp16["task_macro_sr"], 0.2616), "pi0.5 FP16 result drift")
    require(close(pi_quant["task_macro_sr"], 0.2484), "pi0.5 QuantVLA result drift")
    require(close(pi_w6["task_macro_sr"], 0.2452), "pi0.5 W6 result drift")
    require(close(pi_omega["task_macro_sr"], 0.2148), "pi0.5 Omega result drift")
    pi_fp16_splits = pi_fp16["task_set_macro_sr"]
    require(
        all(
            pi_dypac_result["split_task_macro_success_rate"][split]
            > pi_fp16_splits[split]
            for split in ("atomic_seen", "composite_seen", "composite_unseen")
        ),
        "pi0.5 DyPAC no longer exceeds FP16 in every task group",
    )

    result = {
        "schema_version": 1,
        "kind": "dypac_vla_final_paper_evidence",
        "valid": True,
        "paper_identity": {
            "short_name": "DyPAC-VLA",
            "expanded_name": "Full-context PTQ under policy-induced deployment distributions",
            "title": "Quantization Changes the Data: Closed-Loop-Aware Quantization for Vision-Language-Action Policies",
        },
        "gr00t": {
            "ours_task_macro_success_rate": candidate["task_macro_success_rate"],
            "ours_successes": candidate["successes"],
            "episodes": candidate["episodes"],
            "split_task_macro_success_rate": candidate["split_task_macro_success_rate"],
            "w4_layers": 100,
            "fp16_layers": 16,
            "static_component_bytes": plan["table1_total_static_bytes"],
            "compression": plan["table1_total_static_compression"],
            "versus_quantvla": {
                "delta": candidate["task_macro_success_rate"] - quantvla["baseline"]["task_macro_success_rate"],
                "wins": quantvla["paired_wins"],
                "losses": quantvla["paired_losses"],
                "mcnemar_p": quantvla["exact_two_sided_mcnemar_p"],
                "holm_p": quantvla["holm_adjusted_mcnemar_p"],
                "status": "formal_superiority",
            },
            "versus_fp16": {
                "delta": candidate["task_macro_success_rate"] - fp16["baseline"]["task_macro_success_rate"],
                "wins": fp16["paired_wins"],
                "losses": fp16["paired_losses"],
                "holm_p": fp16["holm_adjusted_mcnemar_p"],
                "status": "difference_not_significant_no_equivalence_claim",
            },
            "versus_initializer_stage_gdsq": {
                "same_precision_mask": True,
                "gdsq_task_macro_success_rate": gdsq["baseline"]["task_macro_success_rate"],
                "delta": candidate["task_macro_success_rate"]
                - gdsq["baseline"]["task_macro_success_rate"],
                "wins": gdsq["paired_wins"],
                "losses": gdsq["paired_losses"],
                "holm_p": gdsq["holm_adjusted_mcnemar_p"],
                "hierarchical_bootstrap_ci95": [
                    gdsq_bootstrap["ci95_low"],
                    gdsq_bootstrap["ci95_high"],
                ],
                "scope": (
                    "combined post-allocation stack: Hessian-aware W4 plus "
                    "input-conditioned A8 versus original DuQuant W4 plus "
                    "split-calibrated static A8"
                ),
                "status": "formal_superiority",
            },
            "initializer_selection_disclosure": {
                "development_tasks": sorted(development_tasks),
                "selected_geometry_to_distribution_ratio": "16:1",
                "development_episodes_per_configuration": 200,
                "development_successes_selected_configuration": 161,
                "development_tasks_included_in_headline_aggregate": True,
                "descriptive_heldout46_task_macro_success_rate": heldout46,
            },
            "core_mechanism_evidence": {
                "compression_sweep": {
                    "scope": "GR00T six-arm frozen-rule sweep over all 50 tasks and seeds 0--9",
                    "episodes": compression_sweep["coverage"]["units"],
                    "candidate_target_range": [1.2, 3.5555555555555554],
                    "static_compression_range": [
                        sweep_rows[0]["static_compression"],
                        sweep_rows[-1]["static_compression"],
                    ],
                    "static_bytes_range": [
                        sweep_rows[-1]["static_bytes"],
                        sweep_rows[0]["static_bytes"],
                    ],
                    "task_macro_success_rate_range": [
                        min(row["all50_task_macro_sr"] for row in sweep_rows),
                        max(row["all50_task_macro_sr"] for row in sweep_rows),
                    ],
                    "endpoint_static_size_ratio": (
                        sweep_rows[0]["static_bytes"]
                        / sweep_rows[-1]["static_bytes"]
                    ),
                    "all_w4_matches_least_compressed_success_rate": True,
                    "rates": compression_sweep["rates"],
                    "frozen_reference": sweep_reference,
                    "claim": "broad_nonmonotonic_success_plateau",
                },
                "signal_ablation": {
                    "scope": "GR00T exact-byte Primary-46 directional evidence",
                    "episodes": signal_ablation["coverage"]["units"],
                    "static_component_bytes": signal_masks["total_static_bytes"],
                    "compression": signal_masks["compression"],
                    "arms": signal_ablation["arms"],
                    "comparisons": signal_ablation["comparisons"],
                    "claim": "directional_evidence_only",
                },
                "static_a8_vs_a16_objective": corrected_activation["static_vs_a16"]["objective"],
                "dynamic_a8_vs_a16_objective": corrected_activation["dynamic_vs_a16"]["objective"],
                "dynamic_a8_vs_static_objective": corrected_activation["dynamic_vs_static"]["objective"],
                "one_layer_flips_tested": local["n_flips"],
                "positive_fp16_retention_benefits": local["n_positive_both_metrics"],
                "positive_benefit_interpretation": "both coordinates already FP16; retention value, not improving removals",
                "new_corrected_candidates_require_scoring": False,
                "corrected_candidates_tested": len(fcp_summaries),
                "eligible_corrected_candidates": sum(
                    int(row["eligible"]) for row in fcp_summaries.values()
                ),
                "corrected_candidate_objective_min": min(
                    row["objective"] for row in fcp_summaries.values()
                ),
                "corrected_candidate_objective_max": max(
                    row["objective"] for row in fcp_summaries.values()
                ),
                "corrected_selection": fcp_selection["selected_id"],
                "candidate_state_audit_triggered": False,
                "existing_complete_policy_alternatives_tested": corrected_fullnet["n_candidates"],
                "eligible_existing_complete_policy_alternatives": corrected_fullnet["n_eligible"],
                "existing_complete_policy_objective_min": min(item["objective"] for item in corrected_fullnet["summaries"].values()),
                "existing_complete_policy_objective_max": max(item["objective"] for item in corrected_fullnet["summaries"].values()),
                "candidate_closed_loop_diagnostic": {
                    "role": "preregistered_post_selection_diagnostic",
                    "selection_feedback_allowed": False,
                    "candidates": 6,
                    "new_episodes": fcp_rollouts["coverage"]["new_episodes"],
                    "paired_control_task_macro_success_rate": fcp_rollouts["baseline"]["task_macro_success_rate"],
                    "candidate_task_macro_success_rate_min": min(
                        row["task_macro_success_rate"]
                        for row in fcp_rollouts["candidates"].values()
                    ),
                    "candidate_task_macro_success_rate_max": max(
                        row["task_macro_success_rate"]
                        for row in fcp_rollouts["candidates"].values()
                    ),
                    "best_observed_candidate": "ff_pair_0",
                    "best_observed_delta": fcp_rollouts["comparisons"]["ff_pair_0"]["task_macro_delta"],
                    "minimum_holm_adjusted_p": min(
                        fcp_rollouts["multiplicity"]["adjusted_p"].values()
                    ),
                    "claim": (
                        "no candidate differs significantly from the frozen anchor "
                        "after Holm correction; non-significance is not equivalence"
                    ),
                },
            },
            "real_hardware": {
                "device": "NVIDIA A40 48GB",
                "scope": hardware["scope"],
                "trials_per_configuration": 3,
                "requests_per_configuration": 180,
                "configurations": hardware_summaries,
                "context_base_relative_to_native_fp16": context_hardware[
                    "relative_to_native_fp16"
                ],
                "claim": (
                    "real packed W4 reduces measured CUDA peak allocation, but the "
                    "current generic packed-W4 path increases latency and GPU-board energy"
                ),
                "real_integer_backend_audit": integer_backend,
            },
            "offline_proxy_predictive_validity": {
                "scope": "frozen 60-mask exact-budget GR00T library",
                "mask_episodes": coverage["mask_rows"],
                "fp16_episodes": coverage["fp16_rows"],
                "coverage_gate_passed": coverage["passed"],
                "primary": predictive_primary,
                "primary_gaps": predictive_gaps,
                "registered_positive_direction": (
                    "larger offline distortion implies larger signed "
                    "closed-loop success degradation"
                ),
                "claim": (
                    "none of the three offline metrics provides a positive global "
                    "mask ordering in this library; DPAC is a behavioral guard, "
                    "not a success-ranking oracle"
                ),
            },
            "task_level_direction": {
                "versus_quantvla": quant_task_direction,
                "versus_fp16": fp16_task_direction,
            },
        },
        "pi05": {
            "role": "formal_target_split_result",
            "flow_steps": 4,
            "w4_layers": 121,
            "fp16_layers": 59,
            "static_component_bytes": pi_plan["table1_total_static_bytes"],
            "compression": pi_plan["table1_total_static_compression"],
            "successes": pi_dypac_result["successes"],
            "episodes_per_configuration": pi_dypac_result["episodes"],
            "task_macro_success_rate": pi_dypac_result["task_macro_success_rate"],
            "split_task_macro_success_rate": pi_dypac_result["split_task_macro_success_rate"],
            "claim_status": "formal_target_split_result_descriptive_cross_row_comparison",
            "fcp_mask_adaptation": {
                "initializer_w4_layers": 80,
                "initializer_fp16_layers": 100,
                "initializer_static_component_bytes": pi_main[
                    "table1_total_static_bytes"
                ],
                "initializer_exceeds_budget": True,
                "feasible_anchor_w4_layers": 121,
                "feasible_anchor_fp16_layers": 59,
                "fp16_protections_removed": len(removed_protections),
                "projection_basis": (
                    "paired complete-policy coordinate damage; remove the "
                    "least damaging FP16 protections until byte-feasible"
                ),
                "post_projection_candidates_tested": len(
                    pi_selection["summaries"]
                ),
                "post_projection_selected_id": pi_selection["selected_id"],
                "post_projection_abstained": pi_selection[
                    "fallback_to_baseline"
                ],
                "same_protocol_closed_loop_diagnostic": {
                    "role": "preregistered_post_selection_mechanism_diagnostic",
                    "selection_feedback_allowed": False,
                    "tasks": 50,
                    "seeds_per_task": 10,
                    "new_episodes": pi_diag_raw["coverage"]["new_rollouts"],
                    "reused_anchor_episodes": pi_diag_raw["coverage"][
                        "reused_anchor_rollouts"
                    ],
                    "transferred_initializer_success_rate": pi_diag[
                        "configurations"
                    ]["transferred_initializer"]["result"][
                        "task_macro_success_rate"
                    ],
                    "projected_anchor_success_rate": pi_diag[
                        "configurations"
                    ]["projected_anchor"]["result"][
                        "task_macro_success_rate"
                    ],
                    "projected_anchor_minus_initializer": pi_primary[
                        "task_macro_delta"
                    ],
                    "projection_exact_mcnemar_p": pi_primary[
                        "exact_two_sided_mcnemar_p"
                    ],
                    "post_projection_candidate_rates": {
                        candidate_id: pi_diag["configurations"][candidate_id][
                            "result"
                        ]["task_macro_success_rate"]
                        for candidate_id in ("single_best", "two_best")
                    },
                    "post_projection_minimum_holm_p": min(
                        row["holm_adjusted_p"]
                        for row in pi_diag["secondary_comparisons"].values()
                    ),
                    "claim": (
                        "the byte-feasibility projection has no observed success "
                        "drop in this diagnostic, but the positive estimate is not "
                        "significant; later proposals remain rejected"
                    ),
                },
            },
            "official_target_split_baselines": {
                "fp16": pi_fp16["task_macro_sr"],
                "quantvla_w4a8": pi_quant["task_macro_sr"],
                "uniform_w6": pi_w6["task_macro_sr"],
                "omega_qvla_w4a4": pi_omega["task_macro_sr"],
                "episodes_per_configuration": 2500,
            },
            "versus_fp16": {
                "delta": pi_dypac_result["task_macro_success_rate"]
                - pi_fp16["task_macro_sr"],
                "higher_in_all_task_groups": True,
                "split_delta": {
                    split: pi_dypac_result["split_task_macro_success_rate"][split]
                    - pi_fp16_splits[split]
                    for split in (
                        "atomic_seen",
                        "composite_seen",
                        "composite_unseen",
                    )
                },
                "status": "descriptive_cross_row_comparison",
            },
            "versus_quantvla": {
                "delta": pi_dypac_result["task_macro_success_rate"]
                - pi_quant["task_macro_sr"],
                "status": "descriptive_cross_row_comparison",
            },
        },
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": expected_hash}
            for name, (path, expected_hash) in SOURCES.items()
        },
    }
    if "qvla_actquant" in data:
        result["qvla_actquant_extension"] = audit_qvla_actquant(
            data["qvla_actquant"]
        )
    elif "qvla_actquant_partial" in data:
        result["qvla_actquant_partial_extension"] = audit_qvla_actquant_partial(
            data["qvla_actquant_partial"]
        )
    result["daptq_formal"] = audit_daptq_formal(data["daptq_formal"])

    max_plan = data["gr00t_max_plan"]
    max_layers = list(max_plan["layers"].values())
    require(
        sum(x["bits"] == 4 and not x["skip"] for x in max_layers) == 116
        and sum(bool(x["skip"]) for x in max_layers) == 0,
        "GR00T max mask drift",
    )
    require(
        max_plan["meta"]["activation_mode"] == "dynamic_a8",
        "GR00T max activation mode drift",
    )
    require(
        max_plan["table1_total_static_bytes"] == 836_960_256
        and close(max_plan["table1_total_static_compression"], 2.5563190039934227),
        "GR00T max byte/compression drift",
    )
    max_agg = data["gr00t_max_aggregate"]
    require(
        max_agg["coverage"]["models"]["gr00t"] == {"units": 2500, "complete": True},
        "GR00T max coverage drift",
    )
    max_model = max_agg["models"]["gr00t"]
    require(
        close(max_model["all50_task_macro_sr"], 0.5276)
        and close(max_model["splits"]["atomic_seen"], 0.7088888888888889)
        and close(max_model["splits"]["composite_seen"], 0.42875)
        and close(max_model["splits"]["composite_unseen"], 0.4225),
        "GR00T max success-rate drift",
    )
    max_cmp = max_agg["cross_run_comparisons"]["max_vs_ours_gr00t"]
    require(
        max_cmp["paired_units"] == 2500
        and (
            max_cmp["discordant_ours_success_max_fail"],
            max_cmp["discordant_max_success_ours_fail"],
        )
        == (319, 288)
        and close(max_cmp["exact_two_sided_mcnemar_p"], 0.22332170450216624),
        "GR00T max cross-run comparison drift",
    )
    result["gr00t_max_formal"] = {
        "claim_status": "descriptive_formal_operating_point_secondary_row",
        "mask": {"w4_layers": 116, "fp16_layers": 0, "activation_mode": "dynamic_a8"},
        "static_bytes": 836_960_256,
        "static_compression": 2.5563190039934227,
        "episodes": 2500,
        "successes": 1319,
        "task_macro_success_rate": 0.5276,
        "splits": {
            "atomic_seen": 0.7088888888888889,
            "composite_seen": 0.42875,
            "composite_unseen": 0.4225,
        },
        "cross_run_vs_audited_anchor": {
            "paired_units": 2500,
            "discordant_ours_success_max_fail": 319,
            "discordant_max_success_ours_fail": 288,
            "exact_two_sided_mcnemar_p": 0.22332170450216624,
            "flag": (
                "cross-run descriptive; same 50 seeds and paired-noise protocol; "
                "not a registered test; does not revise the frozen headline row"
            ),
        },
    }
    pi_max_plan = data["pi05_max_plan"]
    pi_max_layers = list(pi_max_plan["layers"].values())
    require(
        sum(x["bits"] == 4 and not x["skip"] for x in pi_max_layers) == 180
        and sum(bool(x["skip"]) for x in pi_max_layers) == 0,
        "pi0.5 max mask drift",
    )
    require(
        pi_max_plan["meta"]["activation_mode"] == "dynamic_a8",
        "pi0.5 max activation mode drift",
    )
    require(
        pi_max_plan["table1_total_static_bytes"] == 1_242_169_344
        and close(pi_max_plan["table1_total_static_compression"], 3.5555555555555554),
        "pi0.5 max byte/compression drift",
    )
    pi_max_agg = data["pi05_max_aggregate"]
    require(
        pi_max_agg["coverage"]["models"]["pi05"] == {"units": 2500, "complete": True},
        "pi0.5 max coverage drift",
    )
    pi_max_model = pi_max_agg["models"]["pi05"]
    require(
        close(pi_max_model["all50_task_macro_sr"], 0.268)
        and close(pi_max_model["splits"]["atomic_seen"], 0.5866666666666667)
        and close(pi_max_model["splits"]["composite_seen"], 0.145)
        and close(pi_max_model["splits"]["composite_unseen"], 0.0325),
        "pi0.5 max success-rate drift",
    )
    pi_max_cmp = pi_max_agg["cross_run_comparisons"]["max_vs_ours_pi05"]
    require(
        pi_max_cmp["paired_units"] == 2500
        and (
            pi_max_cmp["discordant_ours_success_max_fail"],
            pi_max_cmp["discordant_max_success_ours_fail"],
        )
        == (182, 159)
        and close(pi_max_cmp["exact_two_sided_mcnemar_p"], 0.23346206425855837),
        "pi0.5 max cross-run comparison drift",
    )
    result["pi05_max_formal"] = {
        "claim_status": "descriptive_formal_operating_point_secondary_row",
        "mask": {"w4_layers": 180, "fp16_layers": 0, "activation_mode": "dynamic_a8"},
        "static_bytes": 1_242_169_344,
        "static_compression": 3.5555555555555554,
        "episodes": 2500,
        "successes": 670,
        "task_macro_success_rate": 0.268,
        "splits": {
            "atomic_seen": 0.5866666666666667,
            "composite_seen": 0.145,
            "composite_unseen": 0.0325,
        },
        "cross_run_vs_audited_anchor": {
            "paired_units": 2500,
            "discordant_ours_success_max_fail": 182,
            "discordant_max_success_ours_fail": 159,
            "exact_two_sided_mcnemar_p": 0.23346206425855837,
            "flag": (
                "cross-run descriptive; same 50 seeds and paired-noise protocol; "
                "not a registered test; does not revise the frozen headline row"
            ),
        },
    }
    return result


def pct(value: float) -> str:
    # Use conventional half-up display rather than Python's ties-to-even formatting.
    # The frozen JSON may encode decimal ties a few ulps below their exact value.
    rounded = math.floor(1000 * value + 0.5 + 1e-9) / 10
    return f"{rounded:.1f}"


def reproduction_table_row(data: dict[str, dict[str, Any]], method: str, model: str) -> str:
    extension = data.get("qvla_actquant") or data.get("qvla_actquant_partial")
    partial = "qvla_actquant" not in data and extension is not None
    label = (
        r"QVLA-code Wavg4/A16$^{\S,*}$"
        if method == "qvla"
        else r"ActQuant 4.0 BPW/A16$^{\S,*}$"
    ) if partial else (
        r"QVLA-code Wavg4/A16$^{\S}$"
        if method == "qvla"
        else r"ActQuant 4.0 BPW/A16$^{\S}$"
    )
    allocation = (
        r"Ch. 0/2/4/8/16"
        if method == "qvla"
        else r"Tensor IQ2--Q4"
    )
    if extension is None:
        return (
            f"\\quad {label} & {allocation} & "
            "\\multicolumn{6}{c}{\\textit{Pending}} \\\\"
        )
    row = extension["arms"][f"{method}_{model}"]
    splits = row["split_task_macro_success_rate"]
    storage = row["storage"]
    static_gib = (
        float(storage["static_gib_total_deployed_checkpoint_set"])
        / int(storage["checkpoint_count"])
    )
    def display(value: float | None) -> str:
        if value is None:
            return r"\textit{--}"
        rendered = pct(value)
        return rf"\textit{{{rendered}}}" if partial else rendered

    all_cell = display(row["task_macro_success_rate"])
    if partial:
        all_cell += rf"$_{{{row['episodes']}}}$"

    return (
        f"\\quad {label} & {allocation} & {display(splits['atomic_seen'])} & "
        f"{display(splits['composite_seen'])} & {display(splits['composite_unseen'])} & "
        f"{all_cell} & {static_gib:.3f} & "
        f"{float(storage['compression_ratio_total_deployed_checkpoint_set']):.2f}$\\times$ \\\\"
    )


def daptq_table_row(data: dict[str, dict[str, Any]], model: str) -> str:
    row = data["daptq_formal"]["arms"][f"daptq_{model}"]
    artifacts = row["artifacts"]
    w4_layers = {artifact["w4_layers"] for artifact in artifacts}
    bf16_layers = {artifact["bf16_layers"] for artifact in artifacts}
    require(len(w4_layers) == len(bf16_layers) == 1, f"DA-PTQ {model} allocation drift")
    storage = row["storage"]
    static_gib = (
        float(storage["static_gib_total_deployed_checkpoint_set"])
        / int(storage["checkpoint_count"])
    )
    splits = row["split_task_macro_success_rate"]
    return (
        f"\\quad DA-PTQ W4A8$^{{\\parallel}}$ & "
        f"{next(iter(w4_layers))} W4/{next(iter(bf16_layers))} BF16 & "
        f"{pct(splits['atomic_seen'])} & {pct(splits['composite_seen'])} & "
        f"{pct(splits['composite_unseen'])} & {pct(row['task_macro_success_rate'])} & "
        f"{static_gib:.3f} & "
        f"{float(storage['compression_ratio_total_deployed_checkpoint_set']):.2f}$\\times$ \\\\"
    )


def main_table(data: dict[str, dict[str, Any]]) -> str:
    agg = data["gr00t_aggregate"]
    c = agg["candidate"]
    fp = agg["comparisons"]["fp16"]["baseline"]
    q = agg["comparisons"]["quantvla_w4a8"]["baseline"]
    cs = c["split_task_macro_success_rate"]
    fs = fp["split_task_macro_success_rate"]
    qs = q["split_task_macro_success_rate"]
    pi_static = data["pi05_static_official"]
    pi_fp = pi_static["configs"]["fp16"]
    pi_q = pi_static["configs"]["quantvla_w4a8_atmohb"]
    pi_w6 = data["pi05_uniform_w6"]["configs"]["uniform_w6"]
    pi_omega = data["pi05_omega"]
    pi_dypac = data["pi05_dypac_formal"]["result"]
    pifs = pi_fp["task_set_macro_sr"]
    piqs = pi_q["task_set_macro_sr"]
    piws = pi_w6["task_set_macro_sr"]
    pios = {name: row["task_macro_sr"] for name, row in pi_omega["task_sets"].items()}
    pids = pi_dypac["split_task_macro_success_rate"]
    gr_actquant = reproduction_table_row(data, "actquant", "gr00t")
    pi_actquant = reproduction_table_row(data, "actquant", "pi05")
    gr_daptq = daptq_table_row(data, "gr00t")
    pi_daptq = daptq_table_row(data, "pi05")
    max_model = data["gr00t_max_aggregate"]["models"]["gr00t"]
    max_splits = max_model["splits"]
    pi_max_model = data["pi05_max_aggregate"]["models"]["pi05"]
    pi_max_splits = pi_max_model["splits"]
    pi_max_cmp = data["pi05_max_aggregate"]["cross_run_comparisons"]["max_vs_ours_pi05"]
    allocation_header = "Allocation"
    table_column_separation = "1.5pt"
    if "qvla_actquant" in data:
        extension_note = (
            "$^{\\S}$ActQuant~\\cite{akbari2026actquant} uses local FP16-teacher proxy calibration "
            "(source-protocol-equivalent=false) and follows the pinned official GitHub implementation. Its size is a measured static pack; "
            "GR00T reports the mean across its three split checkpoints. "
        )
    elif "qvla_actquant_partial" in data:
        extension_note = (
            "$^{\\S}$ActQuant~\\cite{akbari2026actquant}: local FP16-teacher proxy calibration; "
            "the pinned official GitHub implementation is followed; static pack size (GR00T: three-checkpoint mean). "
            "$^{*}$Italic SRs are observed-only interim values; All-cell subscripts give $n/2500$; no final or inferential claim. "
        )
    else:
        extension_note = (
            "$^{\\S}$ActQuant~\\cite{akbari2026actquant} formal extensions are pending. "
        )
    return f"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\\begin{{table}}[t]
\\centering
\\caption{{RoboCasa365 success--compression operating points. On $\pi_{{0.5}}$, \method reaches 27.7\% success at 2.70$\\times$ compression, above FP16 (26.2\%) and every displayed compressed baseline.}}
\\label{{tab:main_results}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{{table_column_separation}}}
\\renewcommand{{\\arraystretch}}{{0.92}}
\\begin{{tabular}}{{@{{}}lcrrrrrr@{{}}}}
\\toprule
Configuration & {allocation_header} & \\shortstack{{Atomic\\\\SR $\\uparrow$}} & \\shortstack{{C-Seen\\\\SR $\\uparrow$}} & \\shortstack{{C-Unseen\\\\SR $\\uparrow$}} & \\shortstack{{All\\\\SR $\\uparrow$}} & \\shortstack{{Size\\\\(GiB) $\\downarrow$}} & \\shortstack{{Comp.\\\\$\\uparrow$}} \\\\
\\midrule
\\multicolumn{{8}}{{@{{}}l}}{{\\textbf{{$\\pi_{{0.5}}$}}~\\cite{{intelligence2025pi05}}\\enspace\\textit{{(formal target split)}}}} \\\\
\\quad FP16 & -- & {pct(pifs['atomic_seen'])} & {pct(pifs['composite_seen'])} & {pct(pifs['composite_unseen'])} & {pct(pi_fp['task_macro_sr'])} & 4.113 & 1.00$\\times$ \\\\
\\quad \\quantvla W4A8 & 180 W4 & {pct(piqs['atomic_seen'])} & {pct(piqs['composite_seen'])} & {pct(piqs['composite_unseen'])} & {pct(pi_q['task_macro_sr'])} & 1.388 & 2.96$\\times$ \\\\
\\quad Uniform W6 & 180 W6 & {pct(piws['atomic_seen'])} & {pct(piws['composite_seen'])} & {pct(piws['composite_unseen'])} & {pct(pi_w6['task_macro_sr'])} & 1.902 & 2.16$\\times$ \\\\
\\quad $\\Omega$-QVLA W4A4$^{{\\ddagger}}$ & 252 W4 & {pct(pios['atomic_seen'])} & {pct(pios['composite_seen'])} & {pct(pios['composite_unseen'])} & {pct(pi_omega['task_macro_sr'])} & 1.307 & 3.27$\\times$ \\\\
{pi_actquant}
{pi_daptq}
\\quad \\textbf{{\\method (Ours)}}$^{{\\dagger}}$ & 121 W4 & {pct(pids['atomic_seen'])} & {pct(pids['composite_seen'])} & {pct(pids['composite_unseen'])} & {pct(pi_dypac['task_macro_success_rate'])} & 1.523 & 2.70$\\times$ \\\\
\\quad \\textbf{{\\method (Ours, max rate)}}$^{{\\#}}$ & 180 W4 & {pct(pi_max_splits['atomic_seen'])} & {pct(pi_max_splits['composite_seen'])} & {pct(pi_max_splits['composite_unseen'])} & {pct(pi_max_model['all50_task_macro_sr'])} & 1.157 & 3.56$\\times$ \\\\
\\midrule
\\multicolumn{{8}}{{@{{}}l}}{{\\textbf{{GR00T N1.5}}\\enspace\\textit{{(high-success control)}}}} \\\\
\\quad FP16 & -- & {pct(fs['atomic_seen'])} & {pct(fs['composite_seen'])} & {pct(fs['composite_unseen'])} & {pct(fp['task_macro_success_rate'])} & 1.993 & 1.00$\\times$ \\\\
\\quad \\quantvla W4A8 & 116 W4 & {pct(qs['atomic_seen'])} & {pct(qs['composite_seen'])} & {pct(qs['composite_unseen'])} & {pct(q['task_macro_success_rate'])} & 0.898 & 2.22$\\times$ \\\\
\\quad Uniform W6 & 116 W6 & 68.4 & 42.9 & 41.8 & 51.7 & 1.109 & 1.80$\\times$ \\\\
\\quad $\\Omega$-QVLA W4A4$^{{\\ddagger}}$ & 180 W4 & 60.1 & 25.9 & 27.5 & 38.7 & 0.599 & 3.33$\\times$ \\\\
{gr_actquant}
{gr_daptq}
\\quad \\textbf{{\\method (Ours)}}$^{{\\dagger}}$ & 100 W4 & {pct(cs['atomic_seen'])} & {pct(cs['composite_seen'])} & {pct(cs['composite_unseen'])} & {pct(c['task_macro_success_rate'])} & 0.896 & 2.22$\\times$ \\\\
\\quad \\textbf{{\\method (Ours, max rate)}}$^{{\\#}}$ & 116 W4 & {pct(max_splits['atomic_seen'])} & {pct(max_splits['composite_seen'])} & {pct(max_splits['composite_unseen'])} & {pct(max_model['all50_task_macro_sr'])} & 0.779 & 2.56$\\times$ \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{0.99\\textwidth}}{{\\footnotesize C-Seen/C-Unseen denote Composite-Seen/Unseen; $\\pi_{{0.5}}$ uses four flow steps. Registered paired tests are reported in the text. $^{{\\dagger}}$Ours includes initializer-selection tasks; exact bytes are audited. $^{{\\#}}$Ours (max rate): all-W4 profiles at formal 2,500-episode scale (GR00T rate-family endpoint; $\pi_{{0.5}}$ 180-layer W4 profile); descriptive rows, cross-run vs.\ the audited anchors (GR00T 319/288, exact McNemar $p=0.223$; $\pi_{{0.5}}$ {pi_max_cmp['discordant_ours_success_max_fail']}/{pi_max_cmp['discordant_max_success_ours_fail']}, exact McNemar $p={pi_max_cmp['exact_two_sided_mcnemar_p']:.3f}$); not registered tests. $^{{\\ddagger}}\\Omega$-QVLA uses model/task-set-specific calibration. {extension_note}$^{{\\parallel}}$DA-PTQ~\\cite{{xu2026daptq}} is a complete local adaptation (2,500/model; source-protocol-equivalent=false); Appendix~\\ref{{tab:daptq_source}} gives source results. Compression is static storage, not runtime memory or latency.}}
\\end{{table}}
"""


def core_ablation_table(data: dict[str, dict[str, Any]]) -> str:
    correction = data["statistics_correction"]
    activation = correction["activation_attribution"]
    local = correction["local_interventions"]
    fcp = data["corrected_fcp_selection"]["selection"]
    objectives = [item["objective"] for item in fcp["summaries"].values()]
    eligible = sum(int(item["eligible"]) for item in fcp["summaries"].values())
    return f"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\\begin{{table}}[t]
\\centering
\\caption{{Core offline ablations on the frozen 12-task, 36-sequence context buffer; lower objective is better.}}
\\label{{tab:core_ablation}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{4.0pt}}
\\renewcommand{{\\arraystretch}}{{1.00}}
\\begin{{tabular}}{{@{{}}p{{0.16\\linewidth}}p{{0.33\\linewidth}}p{{0.24\\linewidth}}p{{0.18\\linewidth}}@{{}}}}
\\toprule
Factor & Controlled counterfactual & Audited evidence & Decision \\\\
\\midrule
Range & Static A8 vs. FP16-activation control & $J={activation['static_vs_a16']['objective']:.2f}$ & Reject static A8 \\\\
Range & Dynamic A8 vs. registered static A8 & $J={activation['dynamic_vs_static']['objective']:.3f}$ & Use \\dyrange \\\\
Proposal audit & FP16-retention value from state flips & {local['n_positive_both_metrics']}/{local['n_flips']} positive retention values & Rebuild protected sets \\\\
Policy audit & {len(objectives)} corrected complete policies & {eligible}/{len(objectives)} eligible; $J\\in[{min(objectives):.2f},{max(objectives):.2f}]$ & Retain $M_0$ \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{0.99\\textwidth}}{{\\footnotesize $J$ is the worst normalized one-standard-error upper bound over $\\dpac$, $\\dfunc$, and task clusters, using the corrected jackknife SE of the task mean. Dynamic-vs-A16 has $J={activation['dynamic_vs_a16']['objective']:.3f}$. All six corrected candidates have completed teacher-state scoring; none passes the minimax and component guards, so candidate-state scoring is not triggered.}}
\\end{{table}}
"""


def core_rollout_plan_table(data: dict[str, dict[str, Any]]) -> str:
    table3 = data["table3_closed_loop"]
    baseline = table3["baseline"]
    arms = table3["arms"]
    comparison = data["gr00t_aggregate"]["comparisons"]["gdsq_vla_main"]
    final = data["gr00t_aggregate"]["candidate"]
    fcp_rollouts = data["fcp_candidate_closed_loop"]
    fcp_rates = [
        row["task_macro_success_rate"] for row in fcp_rollouts["candidates"].values()
    ]
    fcp_deltas = [
        row["task_macro_delta"] for row in fcp_rollouts["comparisons"].values()
    ]
    minimum_holm_p = min(fcp_rollouts["multiplicity"]["adjusted_p"].values())
    def signed_pp(value: float) -> str:
        return f"{100.0 * value:+.1f}"

    return f"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\\begin{{table}}[H]
\\centering
\\caption{{GR00T post-initialization controls. The identical-mask stack gains $+3.2$ points; the outcome-blinded FCP diagnostic finds no Holm-significant proposal.}}
\\label{{tab:core_rollout_plan}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{2.8pt}}
\\renewcommand{{\\arraystretch}}{{0.88}}
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
Configuration & W4/FP16 & Episodes & SR $\\uparrow$ & $\\Delta$ (pp) \\\\
\\midrule
GDSQ-VLA (DuQuant/static A8) & 100/16 & {comparison['baseline']['episodes']} & {pct(comparison['baseline']['task_macro_success_rate'])} & reference \\\\
\\method (Hessian/\\dyrange) & 100/16 & {final['episodes']} & \\textbf{{{pct(final['task_macro_success_rate'])}}} & \\textbf{{{signed_pp(final['task_macro_success_rate'] - comparison['baseline']['task_macro_success_rate'])}}} \\\\
\\addlinespace[1pt]
$M_0$ + \\dyrange & 100/16 & {baseline['episodes']} & {pct(baseline['task_macro_success_rate'])} & reference \\\\
Frozen-range A8 & 100/16 & {arms['static_a8']['episodes']} & {pct(arms['static_a8']['task_macro_success_rate'])} & {signed_pp(arms['static_a8']['task_macro_success_rate'] - baseline['task_macro_success_rate'])} \\\\
FCP proposals (6; diagnostic) & 111--115/5--1 & {fcp_rollouts['coverage']['new_episodes']} & {pct(min(fcp_rates))}--{pct(max(fcp_rates))} & {signed_pp(min(fcp_deltas))}--{signed_pp(max(fcp_deltas))} \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{0.99\\linewidth}}{{\\footnotesize Candidate success labels were revealed only after FCP froze $M_0$. The smallest Holm-adjusted $p$ across the six paired tests is {minimum_holm_p:.3f}. Table~\\ref{{tab:fcp_candidate_rollouts}} reports every proposal.}}
\\end{{table}}
"""


def fcp_candidate_rollout_table(data: dict[str, dict[str, Any]]) -> str:
    fcp = data["fcp_candidate_closed_loop"]
    preregistration = json.loads(
        Path(fcp["manifest"]["path"]).read_text(encoding="utf-8")
    )
    order = (
        ("single_best", "Single-layer"),
        ("two_best", "Two-layer"),
        ("attention_6", "Attention block 6"),
        ("mlp_2", "MLP block 2"),
        ("ff_pair_0", "FF pair"),
        ("dp_full_lambda_1p0", r"DP-$\lambda$1.0"),
    )
    rows = []
    for candidate_id, label in order:
        plan = preregistration["candidates"][candidate_id]
        result = fcp["candidates"][candidate_id]
        comparison = fcp["comparisons"][candidate_id]
        rows.append(
            f"{label} & {plan['quantized_w4_layers']}/{plan['retained_fp16_layers']} & "
            f"{pct(result['task_macro_success_rate'])} & "
            f"{100.0 * comparison['task_macro_delta']:+.1f} & "
            f"{comparison['paired_candidate_wins']}/{comparison['paired_candidate_losses']} & "
            f"{comparison['exact_mcnemar_p']:.3f} & "
            f"{comparison['holm_adjusted_p']:.3f} \\\\"
        )
    body = "\n".join(rows)
    baseline = fcp["baseline"]
    return f"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\\begin{{table}}[t]
\\centering
\\caption{{Preregistered post-selection GR00T diagnostic. None of the six FCP proposals differs from the frozen $M_0$ after Holm correction.}}
\\label{{tab:fcp_candidate_rollouts}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.0pt}}
\\renewcommand{{\\arraystretch}}{{0.96}}
\\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\\toprule
Configuration & W4/FP16 & SR $\\uparrow$ & $\\Delta$ (pp) & W/L & Raw $p$ & Holm $p$ \\\\
\\midrule
$M_0$ (frozen) & 100/16 & {pct(baseline['task_macro_success_rate'])} & -- & -- & -- & -- \\\\
{body}
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{0.98\\linewidth}}{{\\footnotesize Each proposal uses 50 tasks and ten seeds per task (500 episodes); $\\Delta$ and paired wins/losses are relative to the same 500-episode $M_0$ control. The six tests were fixed before rollout and adjusted as one Holm family.}}
\\end{{table}}
"""


def pi05_fcp_diagnostic_table(data: dict[str, dict[str, Any]]) -> str:
    diagnostic = data["pi05_fcp_diagnostic_summary"]
    configurations = diagnostic["configurations"]
    primary = diagnostic["primary_comparison"]
    secondary = diagnostic["secondary_comparisons"]

    def size_gib(config_id: str) -> str:
        size_bytes = configurations[config_id]["table1_total_static_bytes"]
        return f"{size_bytes / (1024 ** 3):.3f}"

    def rate(config_id: str) -> str:
        value = configurations[config_id]["result"]["task_macro_success_rate"]
        return pct(value)

    return f"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\\begin{{table}}[H]
\\centering
\\caption{{Primary cross-architecture FCP result on $\\pi_{{0.5}}$: complete-policy projection changes 41 precision decisions and makes the transferred initializer byte-feasible.}}
\\label{{tab:pi05_fcp_diagnostic}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.4pt}}
\\renewcommand{{\\arraystretch}}{{0.96}}
\\begin{{tabular}}{{@{{}}lrrrrrl@{{}}}}
\\toprule
Configuration & W4/FP16 & Size (GiB) & SR $\\uparrow$ & $\\Delta$ (pp) & Paired $p$ & Status \\\\
\\midrule
$M_{{\\mathrm{{init}}}}$ (transferred) & 80/100 & {size_gib('transferred_initializer')} & {rate('transferred_initializer')} & $-2.2$ & {primary['exact_two_sided_mcnemar_p']:.3f} & Over budget \\\\
Projected $M_0$ & 121/59 & {size_gib('projected_anchor')} & \\textbf{{{rate('projected_anchor')}}} & reference & -- & Deployed \\\\
Single-best proposal & 179/1 & {size_gib('single_best')} & {rate('single_best')} & $-3.4$ & {secondary['single_best']['holm_adjusted_p']:.3f} & Rejected \\\\
Two-best proposal & 178/2 & {size_gib('two_best')} & {rate('two_best')} & $-3.2$ & {secondary['two_best']['holm_adjusted_p']:.3f} & Rejected \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{0.98\\linewidth}}{{\\footnotesize Each configuration uses the same 50 tasks and seeds 0--9 (500 paired episodes). Deltas are relative to projected $M_0$. The projection comparison is the preregistered singleton primary test ($p=0.235$); the two proposal values are Holm-adjusted. The frozen choice is projected $M_0$.}}
\\end{{table}}
"""


def hardware_results_table(data: dict[str, dict[str, Any]]) -> str:
    summaries = data["gr00t_hardware"]["summaries"]
    order = (
        ("native_fp16", "Native FP16"),
        ("quantvla_w4a8", r"\quantvla W4A8"),
        ("context_base", r"$M_0$ (ours)"),
        ("single_best", "Single-best"),
        ("two_best", "Two-best"),
        ("attention_6", "Attention-6"),
        ("mlp_2", "MLP-2"),
        ("ff_pair_0", "FF-pair-0"),
        ("dp_full_lambda_1p0", r"DP-$\lambda$1.0"),
    )
    rows = []
    for config_id, label in order:
        row = summaries[config_id]
        p50 = row["latency_ms"]["median_of_trial_p50"]
        p95 = row["latency_ms"]["pooled_p95"]
        peak = row["peak_memory"]["allocated_gib_median"]
        gross = row["energy_joules_per_request"]["gross_board"]
        idle = row["energy_joules_per_request"]["idle_adjusted_board"]["mean"]
        rows.append(
            f"{label} & {row['quantized_w4_layers']} & {p50:.1f} & {p95:.1f} & "
            f"{peak:.3f} & {gross['mean']:.2f}$\\pm${gross['sd']:.2f} & {idle:.2f} \\\\"
        )
    body = "\n".join(rows)
    integer = data["integer_w4a8_backend"]["hardware"]

    def integer_row(label: str, row: dict[str, Any]) -> str:
        gross = row["gross_joules_per_request"]
        idle = row["idle_adjusted_joules_per_request"]
        return (
            f"{label} & 100 & {row['p50_ms']:.1f} & {row['p95_ms']:.1f} & "
            f"{row['peak_cuda_allocated_gib']:.3f} & "
            f"{gross['mean']:.2f}$\\pm${gross['sd']:.2f} & "
            f"{idle['mean']:.2f} \\\\"
        )

    integer_body = "\n".join(
        (
            integer_row(r"$M_0$ paired reference", integer["reference"]),
            integer_row(r"$M_0$ real W4A8 integer", integer["candidate"]),
        )
    )
    return f"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\\begin{{table*}}[t]
\\centering
\\caption{{Real GR00T hardware measurements on an otherwise idle NVIDIA A40 at batch size one. The final two rows form a separate paired backend A/B; lower is better for every measured column.}}
\\label{{tab:hardware_results}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{5.0pt}}
\\renewcommand{{\\arraystretch}}{{1.00}}
\\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\\toprule
Configuration & W4 & p50 (ms) & p95 (ms) & Peak alloc. (GiB) & Gross J/req. & Idle-adj. J/req. \\\\
\\midrule
{body}
\\addlinespace[2pt]
\\midrule
{integer_body}
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{0.96\\textwidth}}{{\\footnotesize Each cell aggregates three trials of 60 measured requests after ten warmups (180 requests/configuration), with four flow steps and CUDA synchronization. p50 is the median of trial medians; p95 is pooled. Peak allocation is the median trial CUDA peak. Board energy is trapezoidal integration of 10-Hz NVML power samples; idle-adjusted values subtract a five-second pre-trial baseline. The paired reference and real-integer row must be compared with each other, not across measurement waves. Measurements exclude model loading, simulator, transport, host energy, and robot energy.}}
\\end{{table*}}
"""


def predictive_validity_table(data: dict[str, dict[str, Any]]) -> str:
    predictive = data["predictive_validity"]
    primary = predictive["analysis"]["primary"]
    gaps = predictive["analysis"]["primary_gaps"]

    def row(label: str, key: str) -> str:
        rho = primary[key]["spearman_rho"]["value"]
        tau = primary[key]["kendall_tau_b"]["value"]
        return f"{label} & {rho:.4f} & {tau:.4f} \\\\"

    rows = "\n".join((row("MSE", "mse"), row(r"\dfunc", "d_func"), row(r"\dpac", "d_pac")))
    return f"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\\begin{{table}}[t]
\\centering
\\caption{{Predictive validity of frozen offline scores over one exact-budget GR00T mask library. Positive association is the preregistered predictive direction.}}
\\label{{tab:predictive_validity}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{8.0pt}}
\\renewcommand{{\\arraystretch}}{{1.00}}
\\begin{{tabular}}{{@{{}}lrr@{{}}}}
\\toprule
Offline distortion & Spearman $\\rho$ $\\uparrow$ & Kendall $\\tau_b$ $\\uparrow$ \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{0.94\\linewidth}}{{\\footnotesize The coverage gate passes 60 masks $\\times$ 50 tasks (3,000 mask episodes) plus 50 independent FP16 references, with no failure, duplicate, or hash drift. All masks have 100 W4/16 FP16 layers and identical exact static bytes; offline scores were frozen before rollout. $\\Delta\\rho$ for \\dpac minus MSE is {gaps['mse']['delta_spearman_rho']['value']:.4f}, and for \\dpac minus \\dfunc is {gaps['d_func']['delta_spearman_rho']['value']:.4f}. Thus \\dpac is not a superior success-ranking surrogate in this library.}}
\\end{{table}}
"""


def signal_ablation_table(data: dict[str, dict[str, Any]]) -> str:
    result = data["signal_ablation"]
    masks = data["signal_ablation_masks"]
    labels = {
        "cs_cka": r"\textbf{CS+CKA-DiT (16:1)}",
        "cka_only": "CKA-DiT only",
        "cs_only": "CS only",
    }
    rows = []
    for arm in ("cs_cka", "cka_only", "cs_only"):
        outcome = result["arms"][arm]
        mask = masks["arms"][arm]
        if arm == "cs_cka":
            delta = "--"
            holm_p = "--"
        else:
            comparison = result["comparisons"][f"cs_cka_vs_{arm}"]
            delta = f"{100.0 * comparison['delta_task_macro_sr']:+.2f}"
            holm_p = f"{comparison['holm_p']:.4f}"
        primary = 100.0 * outcome["primary46_task_macro_sr"]
        all50 = 100.0 * outcome["all50_task_macro_sr"]
        primary_cell = f"\\textbf{{{primary:.2f}}}" if arm == "cs_cka" else f"{primary:.2f}"
        all50_cell = f"\\textbf{{{all50:.2f}}}" if arm == "cs_cka" else f"{all50:.2f}"
        rows.append(
            f"{labels[arm]} & {mask['w4_count']}/{mask['fp16_count']} & "
            f"{primary_cell} & {delta} & {holm_p} & {all50_cell} \\\\"
        )
    body = "\n".join(rows)
    return f"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\\begin{{table}}[H]
\\centering
\\caption{{Exact-byte GR00T signal ablation. Both primary comparisons favor CS+CKA-DiT, but the evidence is directional rather than significant.}}
\\label{{tab:signal_ablation}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{5.0pt}}
\\renewcommand{{\\arraystretch}}{{0.98}}
\\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\\toprule
Signal & W4/FP16 & Primary-46 SR (\\%) $\\uparrow$ & Fused $\\Delta$ (pp) & Holm $p$ & All-50 SR (\\%) $\\uparrow$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{0.97\\linewidth}}{{\\footnotesize All arms use exactly 962,068,480 static bytes (2.2239$\\times$ compression), deployment-conditioned A8, 50 tasks, and seeds 0--9, totaling 1,500 episodes. Primary-46 excludes the four tasks used to select the 16:1 ratio. Deltas are fused minus the single-signal arm; the two paired McNemar tests form one Holm family.}}
\\end{{table}}
"""


def compression_sweep_table(data: dict[str, dict[str, Any]]) -> str:
    result = data["compression_sweep"]
    order = (
        ("rate12", "1.20"),
        ("rate16", "1.60"),
        ("rate20", "2.00"),
        ("rate24", "2.40"),
        ("rate28", "2.80"),
        ("ratemax", r"\textbf{Max (3.56)}"),
    )
    rows = []
    for arm, target in order:
        outcome = result["rates"][arm]
        splits = outcome["splits"]
        all50 = pct(outcome["all50_task_macro_sr"])
        all50_cell = (
            rf"\textbf{{{all50}}}"
            if arm in {"rate12", "ratemax"}
            else all50
        )
        allocation = f"{outcome['w4_layers']}/{outcome['fp16_layers']}"
        if arm == "ratemax":
            allocation = rf"\textbf{{{allocation}}}"
        rows.append(
            f"{target} & {allocation} & "
            f"{outcome['static_bytes'] / (1024 ** 3):.3f} & "
            f"{outcome['static_compression']:.3f}$\\times$ & "
            f"{all50_cell} & {pct(splits['atomic_seen'])} & "
            f"{pct(splits['composite_seen'])} & "
            f"{pct(splits['composite_unseen'])} \\\\"
        )
    body = "\n".join(rows)
    return f"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\\begin{{table}}[H]
\\centering
\\caption{{Six-arm GR00T compression-rate sweep. Success remains on a 51.4--55.4\\% plateau while static compression increases from 1.166$\\times$ to 2.556$\\times$.}}
\\label{{tab:compression_sweep}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.4pt}}
\\renewcommand{{\\arraystretch}}{{0.96}}
\\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\\toprule
Target $r$ & W4/FP16 & Size (GiB) & Static comp. & All-50 SR (\\%) $\\uparrow$ & Atomic & C-Seen & C-Unseen \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{0.98\\linewidth}}{{\\footnotesize Every arm uses the frozen FCP ranking, Hessian-g64 W4, deployment-conditioned A8, four flow steps, all 50 tasks, and seeds 0--9 (500 episodes per arm). All 3,000 expected units are unique and complete. Target $r$ applies to candidate tensors; static compression includes fixed components and packing metadata. The frozen 100-W4/16-FP16 headline policy obtains 54.0\\% at 2.224$\\times$ and is cited rather than rerun.}}
\\end{{table}}
"""


def claim_status() -> str:
    return r"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\providecommand{\GRHeadlineStatus}{DyPAC-VLA obtains 54.0\% over 2,500 GR00T episodes at 2.22$\times$ static component compression.}
\providecommand{\GRComparisonStatus}{The gain over QuantVLA is 23.6 points with Holm-adjusted $p<10^{-4}$; the difference from FP16 is not significant ($p=0.2929$).}
\providecommand{\PiAnchorStatus}{DyPAC-VLA obtains 27.7\% over 2,500 $\pi_{0.5}$ episodes at 2.702$\times$ static component compression, 1.5 points above FP16 and 2.9 points above QuantVLA; the cross-row differences are descriptive.}
"""


def generated_files(data: dict[str, dict[str, Any]], registry: dict[str, Any]) -> dict[Path, str]:
    source_hashes = {name: digest for name, (_, digest) in SOURCES.items()}
    if "qvla_actquant" in data:
        source_hashes["qvla_actquant"] = sha256(QVLA_ACTQUANT_AGGREGATE)
    elif "qvla_actquant_partial" in data:
        source_hashes["qvla_actquant_partial"] = sha256(QVLA_ACTQUANT_PARTIAL)
    audit_payload = {
        "schema_version": 1,
        "kind": "dypac_vla_paper_render_audit",
        "valid": True,
        "source_hashes": source_hashes,
        "headline": registry["gr00t"],
        "pi05_claim_guard": registry["pi05"],
        "gr00t_max_formal": registry["gr00t_max_formal"],
        "pi05_max_formal": registry["pi05_max_formal"],
    }
    return {
        PAPER / "tables/main_results.tex": main_table(data),
        PAPER / "tables/core_ablation.tex": core_ablation_table(data),
        PAPER / "tables/core_rollout_plan.tex": core_rollout_plan_table(data),
        PAPER / "tables/fcp_candidate_rollouts.tex": fcp_candidate_rollout_table(data),
        PAPER / "tables/pi05_fcp_diagnostic.tex": pi05_fcp_diagnostic_table(data),
        PAPER / "tables/hardware_results.tex": hardware_results_table(data),
        PAPER / "tables/predictive_validity.tex": predictive_validity_table(data),
        PAPER / "tables/signal_ablation.tex": signal_ablation_table(data),
        PAPER / "tables/compression_sweep.tex": compression_sweep_table(data),
        PAPER / "tables/claim_status.tex": claim_status(),
        PAPER / "dypac_evidence_registry.json": json.dumps(registry, indent=2, sort_keys=True) + "\n",
        PAPER / "tables/dypac_paper.audit.json": json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args()
    data = load_sources()
    registry = audit(data)
    outputs = generated_files(data, registry)
    stale = []
    for path, content in outputs.items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    require(not stale, f"stale generated DyPAC paper files: {stale}")
    print(json.dumps({"valid": True, "checked": sorted(str(p.relative_to(ROOT)) for p in outputs)}, indent=2))


if __name__ == "__main__":
    main()
