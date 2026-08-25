#!/usr/bin/env python3
"""Self-contained tests for the audited π0.5 final report renderer."""

from __future__ import annotations

import copy
import unittest

import render_pi05_table1_report as report


TASK_SETS = {
    "atomic_seen": ["AtomicTask"],
    "composite_seen": ["SeenTask"],
    "composite_unseen": ["UnseenTask"],
}


def config_row(value: float) -> dict:
    return {
        "completed_episodes": 2500,
        "task_set_macro_sr": {task_set: value for task_set in TASK_SETS},
        "heldout46_task_macro_sr": value,
        "task_macro_sr": value,
        "episode_sr": value,
        "per_task_sr": {task: value for tasks in TASK_SETS.values() for task in tasks},
        "efficiency": {
            "mean_episode_wall_seconds": 10.0,
            "mean_server_infer_ms": 20.0,
            "mean_inference_seconds_per_replan": 0.02,
            "mean_success_steps": 100.0,
            "mean_failure_steps": 500.0,
            "gpu": {
                "peak_server_process_memory_mib": 1234.0,
                "peak_device_memory_used_mib": 2345.0,
            },
        },
    }


def comparison(a: str, b: str, delta: float) -> dict:
    return {
        "a": a,
        "b": b,
        "heldout46_task_macro_delta": delta,
        "heldout46_task_cluster_ci95": [delta - 0.01, delta + 0.01],
        "paired_permutation_p": 0.01,
        "holm_adjusted_p": 0.04,
        "episode_mcnemar": {"a_wins": 10, "b_wins": 5, "two_sided_p": 0.1},
    }


def fixtures() -> tuple[dict, dict, dict]:
    summary = {
        "complete": True,
        "manifest_sha256": "manifest",
        "configs": {
            config: config_row(0.4 + 0.01 * index)
            for index, config in enumerate(report.CONFIG_ORDER)
        },
        "comparisons": {
            "gdsq_vla_atmohb_vs_gdsq_vla": comparison(
                "gdsq_vla_atmohb", "gdsq_vla", 0.02
            ),
            "gdsq_vla_vs_fp16": comparison("gdsq_vla", "fp16", -0.01),
            "gdsq_vla_vs_quantvla_w4a8_atmohb": comparison(
                "gdsq_vla", "quantvla_w4a8_atmohb", 0.03
            ),
            "gdsq_vla_atmohb_vs_quantvla_w4a8_atmohb": comparison(
                "gdsq_vla_atmohb", "quantvla_w4a8_atmohb", 0.04
            ),
        },
        "paper_style_memory": {
            "candidate_parameters": 100,
            "gdsq_w4_parameters": 80,
            "gdsq_w4_parameter_fraction": 0.8,
            "gdsq_w4_layers": 69,
            "gdsq_fp16_layers": 111,
            "by_config": {
                config: {"gib": 1.0 + index, "compression_vs_fp16": 1.0 + index}
                for index, config in enumerate(report.CONFIG_ORDER)
            },
        },
    }
    audit = {
        "complete": True,
        "manifest_sha256": "manifest",
        "frozen_aggregator_sha256": "aggregator",
        "audit_tool_sha256": "auditor",
        "bootstrap_draws": 10_000,
        "artifacts": {"verified": 29},
        "statistics": {"comparisons_ready": True},
        "schedule": {"chain_leaf_to_root": [{"sha256": "schedule"}]},
    }
    manifest = {"table_1_protocol": {"task_sets": TASK_SETS}}
    return summary, audit, manifest


class FinalReportTest(unittest.TestCase):
    def test_complete_report_contains_required_sections(self) -> None:
        summary, audit, manifest = fixtures()
        text = report.render(summary, audit, manifest)
        for phrase in (
            "Main result",
            "Prespecified paired comparisons",
            "Auxiliary episode-paired McNemar",
            "Per-task success rate",
            "Closed-loop efficiency",
            "paper-style candidate-component storage",
            "complete and audited",
            "69 W4 + 111 native-FP16",
        ):
            self.assertIn(phrase, text)

    def test_incomplete_summary_is_rejected(self) -> None:
        summary, audit, manifest = fixtures()
        summary = copy.deepcopy(summary)
        summary["complete"] = False
        with self.assertRaisesRegex(ValueError, "summary is not complete"):
            report.render(summary, audit, manifest)

    def test_manifest_mismatch_is_rejected(self) -> None:
        summary, audit, manifest = fixtures()
        audit = copy.deepcopy(audit)
        audit["manifest_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            report.render(summary, audit, manifest)


if __name__ == "__main__":
    unittest.main()
