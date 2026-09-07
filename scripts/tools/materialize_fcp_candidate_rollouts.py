#!/usr/bin/env python3
"""Materialize the frozen reduced closed-loop FCP candidate evaluation.

This program intentionally reads no candidate closed-loop outcomes.  It binds
the already frozen FCP protocol and corrected proposal manifest to executable
GR00T specs, while proving exact paired-control coverage for seeds 0--9.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from quantvla_cross_model_protocol import sha256_file, validate_quant_plan
from quantvla_outputimpact import atomic_json


REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / "scripts/quantvla_fcp_hardware_protocol.json"
FULL_CONTEXT_PROTOCOL = REPO / "scripts/quantvla_full_context_protocol.json"
ROOT = REPO / "runs/full_context_v2/fcp_completion/closed_loop"
PROPOSALS = REPO / "runs/full_context_v2/statistics_correction/proposals/manifest.json"
TABLE1_SPECS = REPO / "runs/full_context_v2/table1/specs"
BASELINE_RESULTS = REPO / "runs/full_context_v2/table1/results/full_context_v2"
HESSIAN_ROOT = ROOT / "hessian"
PREREGISTRATION = ROOT / "preregistration.json"
SELECTION_REPORT = (
    REPO
    / "runs/full_context_v2/fcp_completion/gr00t_corrected_fcp_frozen.json.selection.json"
)
SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")


def artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def stable_write(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != payload:
            raise SystemExit(f"immutable FCP closed-loop artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)


def load_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("kind") != "fcp_candidate_and_hardware_completion_preregistration":
        raise SystemExit("unexpected FCP completion protocol kind")
    evaluation = protocol["closed_loop_evaluation"]
    if evaluation["seeds"] != list(range(10)):
        raise SystemExit("FCP closed-loop seed-set drift")
    if int(evaluation["tasks"]) != 50 or int(evaluation["episodes_per_candidate"]) != 500:
        raise SystemExit("FCP closed-loop episode-budget drift")

    proposals = json.loads(PROPOSALS.read_text(encoding="utf-8"))
    frozen = protocol["frozen_inputs"]["candidate_manifest"]
    if sha256_file(PROPOSALS) != frozen["sha256"]:
        raise SystemExit("corrected candidate-manifest hash drift")
    candidate_ids = list(evaluation["candidate_ids"])
    if candidate_ids != list(protocol["frozen_inputs"]["candidate_ids"]):
        raise SystemExit("candidate ordering drift between frozen protocol sections")
    rows = {row["candidate_id"]: row for row in proposals["candidates"]}
    if set(candidate_ids) - set(rows):
        raise SystemExit("frozen candidate missing from proposal manifest")
    return protocol, proposals, candidate_ids


def baseline_coverage(
    expected: set[tuple[str, int]], seeds: list[int]
) -> dict[str, Any]:
    observed: dict[tuple[str, int], Path] = {}
    successes = 0
    for path in sorted(BASELINE_RESULTS.glob("**/*.jsonl")):
        if path.name.startswith(("gpu_efficiency", "gpu_server_efficiency")):
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config") != "full_context_v2":
                continue
            key = (str(row["task"]), int(row["seed"]))
            if key not in expected:
                continue
            if row.get("status") != "complete":
                raise SystemExit(f"incomplete paired-control row: {path}:{line_number}")
            if not bool(row.get("paired_action_noise", False)):
                raise SystemExit(f"unpaired control action noise: {path}:{line_number}")
            if key in observed:
                raise SystemExit(f"duplicate paired-control task-seed: {key}")
            observed[key] = path
            successes += int(bool(row["success"]))
    if set(observed) != expected:
        raise SystemExit(
            f"paired-control coverage drift: observed={len(observed)} expected={len(expected)}"
        )
    return {
        "id": "full_context_v2",
        "root": str(BASELINE_RESULTS.resolve()),
        "tasks": len({task for task, _ in observed}),
        "seeds": seeds,
        "episodes": len(observed),
        "successes_observed_before_candidate_rollout": successes,
        "source_files": [artifact(path) for path in sorted(set(observed.values()))],
    }


def main() -> None:
    protocol, proposals, candidate_ids = load_frozen_inputs()
    seeds = [int(value) for value in protocol["closed_loop_evaluation"]["seeds"]]

    # Task identities are inherited from the frozen full-context protocol; the
    # completion protocol separately froze the corresponding split counts.
    full_protocol_attestation = protocol["frozen_inputs"]["parent_protocol"]
    if sha256_file(FULL_CONTEXT_PROTOCOL) != full_protocol_attestation["sha256"]:
        raise SystemExit("full-context parent-protocol hash drift")
    full_protocol = json.loads(FULL_CONTEXT_PROTOCOL.read_text(encoding="utf-8"))
    tasks: dict[str, list[str]] = {
        split: list(full_protocol["table1"]["tasks"][split]) for split in SPLITS
    }
    table_specs: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        document = json.loads((TABLE1_SPECS / f"{split}.json").read_text(encoding="utf-8"))
        table_specs[split] = document
    expected_counts = protocol["closed_loop_evaluation"]["task_sets"]
    if {split: len(values) for split, values in tasks.items()} != expected_counts:
        raise SystemExit("task-set counts drift from frozen completion protocol")
    if len({task for values in tasks.values() for task in values}) != 50:
        raise SystemExit("task identity overlap or count drift")
    expected = {
        (task, seed)
        for values in tasks.values()
        for task in values
        for seed in seeds
    }
    control = baseline_coverage(expected, seeds)

    selection_report = json.loads(SELECTION_REPORT.read_text(encoding="utf-8"))
    selection = selection_report["selection"]
    if (
        selection_report.get("kind") != "corrected_full_context_frozen_selection_report"
        or selection.get("selected_id") != "context_base"
        or selection.get("fallback_to_baseline") is not True
        or set(selection.get("summaries") or {}) != set(candidate_ids)
        or any(row.get("eligible") is not False for row in selection["summaries"].values())
    ):
        raise SystemExit("corrected teacher-state FCP decision drift")

    proposal_rows = {row["candidate_id"]: row for row in proposals["candidates"]}
    candidates: dict[str, Any] = {}
    execution_specs: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        proposal_row = proposal_rows[candidate_id]
        plan_path = Path(proposal_row["path"]).resolve()
        if sha256_file(plan_path) != proposal_row["sha256"]:
            raise SystemExit(f"{candidate_id}: proposal plan hash drift")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        attestation = validate_quant_plan(plan, model="gr00t", source=str(plan_path))
        if plan.get("meta", {}).get("candidate_id") != candidate_id:
            raise SystemExit(f"{candidate_id}: plan identity drift")
        if int(proposal_row["retained_fp16_layers"]) != int(attestation["retained_fp16_target_layers"]):
            raise SystemExit(f"{candidate_id}: FP16-count drift")
        candidates[candidate_id] = {
            "plan": artifact(plan_path),
            "protected_layers": list(plan["protected_layers"]),
            "quantized_w4_layers": int(attestation["quantized_w4_layers"]),
            "retained_fp16_layers": int(attestation["retained_fp16_target_layers"]),
            "table1_total_static_bytes": int(plan["table1_total_static_bytes"]),
        }

        for split in SPLITS:
            baseline_runtime = copy.deepcopy(
                next(
                    row
                    for row in table_specs[split]["configs"]
                    if row["id"] == "full_context_v2"
                )
            )
            hessian = HESSIAN_ROOT / candidate_id / split / "hessian_w4.npz"
            for field in ("gpu", "port", "egl_device", "replicas"):
                baseline_runtime.pop(field, None)
            baseline_runtime.update(
                {
                    "id": candidate_id,
                    "expected_wrapped": int(attestation["quantized_w4_layers"]),
                    "plan": str(plan_path),
                    "hessian_w4": str(hessian.resolve()),
                    "act_scale": None,
                    "activation_mode": "dynamic_a8",
                    "meta": {
                        "role": "post_selection_fcp_candidate_diagnostic",
                        "candidate_rejected_by_teacher_state_screen": True,
                        "result_feedback_allowed": False,
                        "runtime_correction": False,
                        "runtime_selector": False,
                    },
                }
            )
            if any(
                (
                    baseline_runtime.get("errorfold"),
                    baseline_runtime.get("atm"),
                    baseline_runtime.get("ohb"),
                )
            ):
                raise SystemExit(f"{candidate_id}/{split}: correction-state drift")
            spec = {
                "schema_version": 1,
                "kind": "gr00t_fcp_candidate_reduced_execution_spec",
                "purpose": "paired post-selection diagnostic for corrected FCP candidate",
                "task_set": split,
                "tasks": tasks[split],
                "seeds": "0-9",
                "configs": [baseline_runtime],
                "comparisons": {
                    "paired_external_control": "full_context_v2",
                    "unit": "paired_task_seed",
                },
                "decision": {
                    "teacher_state_selection_already_frozen": True,
                    "candidate_state_audit_triggered": False,
                    "result_feedback_allowed": False,
                    "reduced_rollout_budget": True,
                },
            }
            spec_path = ROOT / "specs" / candidate_id / f"{split}.json"
            if not hessian.is_file():
                raise FileNotFoundError(hessian)
            stable_write(spec_path, spec)
            execution_specs[f"{candidate_id}/{split}"] = {
                "spec": artifact(spec_path),
                "hessian_w4": artifact(hessian),
                "identity_pack": str(Path(baseline_runtime["packdir"]).resolve()),
            }

    preregistration = {
        "schema_version": 1,
        "kind": "gr00t_fcp_candidate_reduced_closed_loop_preregistration",
        "immutable": True,
        "result_feedback_allowed": False,
        "parent_completion_protocol": artifact(PROTOCOL),
        "corrected_proposal_manifest": artifact(PROPOSALS),
        "selection": {
            "artifact": artifact(
                REPO / "runs/full_context_v2/fcp_completion/gr00t_corrected_fcp_frozen.json"
            ),
            "result": "context_base",
            "candidate_state_audit_triggered": False,
            "closed_loop_role": "post-selection diagnostic only",
        },
        "runtime_contract": {
            "weight_quantization": "signed packed Hessian group-64 W4",
            "activation_quantization": "dynamic per-forward per-channel A8",
            "row_rotation": "identity",
            "flow_steps": 4,
            "action_horizon": 16,
            "paired_action_noise": True,
            "runtime_correction": False,
            "runtime_selector": False,
        },
        "evaluation": {
            "benchmark": "RoboCasa365 target-posttraining",
            "task_sets": tasks,
            "trial_seeds": seeds,
            "candidate_ids": candidate_ids,
            "episodes_per_candidate": len(expected),
            "total_new_episodes": len(expected) * len(candidate_ids),
            "paired_control": control,
        },
        "statistics": {
            "primary": "unweighted task-macro success-rate delta versus paired context_base",
            "secondary": [
                "micro success-rate delta",
                "split task-macro success rate",
                "paired wins/losses/ties",
                "exact two-sided McNemar p-value",
            ],
            "uncertainty": "task-then-seed hierarchical bootstrap, 10000 draws, seed 0",
            "multiplicity": "Holm adjustment over six candidate-control McNemar tests",
            "scope": "reduced 10-seed diagnostic; does not replace the 50-seed headline estimate",
        },
        "candidates": candidates,
        "execution_specs": execution_specs,
    }
    stable_write(PREREGISTRATION, preregistration)
    print(
        json.dumps(
            {
                "preregistration": str(PREREGISTRATION),
                "candidates": candidate_ids,
                "episodes_per_candidate": len(expected),
                "total_new_episodes": len(expected) * len(candidate_ids),
                "paired_control_successes": control["successes_observed_before_candidate_rollout"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
