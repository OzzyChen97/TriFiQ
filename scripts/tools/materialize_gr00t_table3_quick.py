#!/usr/bin/env python3
"""Freeze the reduced GR00T Table-3 ablations before closed-loop rollout.

The three contrasts are activation range, local-MSE mask selection, and the
complete-network verifier.  Closed-loop outcomes are never read here.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from quantvla_cross_model_protocol import (
    PROTOCOL as CROSS_MODEL_PROTOCOL,
    PROTOCOL_SHA256 as CROSS_MODEL_PROTOCOL_SHA256,
    sha256_file,
    validate_quant_plan,
)
from quantvla_full_context import (
    BudgetItem,
    PROTOCOL,
    exact_weighted_knapsack,
    protocol_attestation,
    require_protocol_attestation,
)
from quantvla_outputimpact import atomic_json
from quantvla_table1_bytes import table1_variable_budget
from select_full_context_protection import finalize_candidate_plan, set_plan_mask


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "runs/full_context_v2/table3_quick"
SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")
CONFIGS = ("static_a8", "local_mse_selection", "no_fullnet_check")
SEEDS = tuple(range(10))

M0_PLAN = REPO / "runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json"
INTERVENTIONS = REPO / "runs/full_context_v2/p2/interventions_dynamic/manifest.json"
PROPOSALS = REPO / "runs/full_context_v2/p2/proposals_dynamic/manifest.json"
SINGLE_BEST = REPO / "runs/full_context_v2/p2/proposals_dynamic/single_best.json"
FULLNET_SCORES = REPO / "runs/full_context_v2/p2/fullnet_scores/fullnet_scores_cross_split.json"
TABLE1_SPECS = REPO / "runs/full_context_v2/table1/specs"
ATTR_SPECS = REPO / "runs/full_context_v1/gr00t/attribution/specs"
BASELINE_RESULTS = REPO / "runs/full_context_v2/table1/results/full_context_v2"
LOCAL_PLAN = ROOT / "plans/local_mse_selection.json"
LOCAL_REPORT = ROOT / "plans/local_mse_selection.report.json"
PREREGISTRATION = ROOT / "preregistration.json"
HESSIAN_ROOT = ROOT / "hessian"


def artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def stable_write(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    if destination.is_file():
        saved = json.loads(destination.read_text(encoding="utf-8"))
        if saved != payload:
            raise SystemExit(f"immutable Table-3 artifact drift: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, payload)


def plan_modes(document: dict[str, Any]) -> dict[str, str]:
    return {
        name: "fp16" if bool(row.get("skip", False)) else "w4"
        for name, row in document["layers"].items()
    }


def materialize_local_mse_plan() -> dict[str, Any]:
    manifest = json.loads(INTERVENTIONS.read_text(encoding="utf-8"))
    require_protocol_attestation(manifest, source=str(INTERVENTIONS))
    fp16_path = Path(manifest["fp16_capture"]).resolve()
    hessian_path = Path(manifest["hessian_w4"]).resolve()
    base_path = Path(manifest["base_full_w4_plan"]).resolve()
    byte_rows = manifest["byte_rows"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    validate_quant_plan(base, model="gr00t", source=str(base_path))

    local_rows: dict[str, dict[str, float | int]] = {}
    items: list[BudgetItem] = []
    with np.load(fp16_path, allow_pickle=False) as fp16, np.load(
        hessian_path, allow_pickle=False
    ) as hessian:
        names = [str(value) for value in fp16["layer_names"].tolist()]
        hessian_names = [str(value) for value in hessian["layer_names"].tolist()]
        if names != hessian_names or set(names) != set(base["layers"]):
            raise SystemExit("local-MSE calibration inventory drift")
        for index, name in enumerate(names):
            output = np.asarray(fp16[f"outputs_{index:04d}"], dtype=np.float64)
            error = np.asarray(hessian[f"error_{index:04d}"], dtype=np.float64)
            teacher_energy = max(float(np.mean(np.square(output))), 1e-12)
            local_mse = float(np.mean(error))
            local_relative_mse = max(local_mse / teacher_energy, 0.0)
            if not math.isfinite(local_relative_mse):
                raise SystemExit(f"{name}: non-finite local relative MSE")
            extra_bytes = int(byte_rows[name]["extra_fp16_bytes"])
            local_rows[name] = {
                "local_mse": local_mse,
                "teacher_output_energy": teacher_energy,
                "local_relative_mse": local_relative_mse,
                "extra_fp16_bytes": extra_bytes,
            }
            items.append(
                BudgetItem(
                    name=name,
                    extra_bytes=extra_bytes,
                    benefit_d_func=local_relative_mse,
                    benefit_d_pac=local_relative_mse,
                )
            )

    all_w4 = int(manifest["all_w4_total_bytes"])
    capacity = table1_variable_budget("gr00t") - all_w4
    selection = exact_weighted_knapsack(
        items, budget_bytes=capacity, lambda_d_func=0.5
    )
    protected = set(selection.protected)
    plan = set_plan_mask(base, protected, reason_prefix="local_mse_selection")
    plan = finalize_candidate_plan(
        plan,
        identifier="local_mse_selection",
        model="gr00t",
        activation_mode="dynamic_a8",
        round_index=0,
        protected=protected,
        byte_rows=byte_rows,
        manifest_sha=sha256_file(INTERVENTIONS),
        flow_steps=4,
    )
    meta = dict(plan.get("meta") or {})
    meta.update(
        {
            "kind": "table3_local_mse_selection_ablation",
            "candidate_id": "local_mse_selection",
            "selection_formula": "mean(local Hessian W4 reconstruction error) / mean(FP16 teacher output squared)",
            "selection_solver": "exact sparse-frontier 0/1 knapsack",
            "selection_capacity_bytes": capacity,
            "local_metric_only": True,
            "uses_full_network_metric": False,
            "uses_complete_network_adjudication": False,
            "uses_closed_loop_results": False,
            "result_feedback_used": False,
        }
    )
    plan["meta"] = meta
    for name, row in plan["layers"].items():
        row["local_mse_selection"] = local_rows[name]
    validate_quant_plan(plan, model="gr00t", source="Table-3 local-MSE plan")
    stable_write(LOCAL_PLAN, plan)

    report = {
        "schema_version": 1,
        "kind": "table3_local_mse_selection_report",
        "immutable": True,
        "result_feedback_allowed": False,
        "selection_formula": meta["selection_formula"],
        "solver": meta["selection_solver"],
        "calibration": {
            "fp16_capture": artifact(fp16_path),
            "hessian_w4": artifact(hessian_path),
            "base_full_w4_plan": artifact(base_path),
            "intervention_manifest": artifact(INTERVENTIONS),
        },
        "capacity_bytes": capacity,
        "selected_extra_bytes": selection.extra_bytes,
        "unused_capacity_bytes": capacity - selection.extra_bytes,
        "objective": selection.utility,
        "protected_layers": list(selection.protected),
        "quantized_w4_layers": plan["quantized_w4_layers"],
        "retained_fp16_layers": plan["retained_fp16_layers"],
        "total_bytes": plan["total_bytes"],
        "table1_total_static_bytes": plan["table1_total_static_bytes"],
        "table1_total_static_budget_bytes": plan[
            "table1_total_static_budget_bytes"
        ],
        "local_rows": local_rows,
        "plan": artifact(LOCAL_PLAN),
    }
    stable_write(LOCAL_REPORT, report)
    return report


def validate_no_fullnet_arm() -> dict[str, Any]:
    proposals = json.loads(PROPOSALS.read_text(encoding="utf-8"))
    require_protocol_attestation(proposals, source=str(PROPOSALS))
    rows = proposals["candidates"]
    non_base = [row for row in rows if row["candidate_id"] != "context_base"]
    ranked = sorted(
        non_base,
        key=lambda row: (-float(row["predicted_utility"]), row["candidate_id"]),
    )
    if not ranked or ranked[0]["candidate_id"] != "single_best":
        raise SystemExit("highest-ranked non-base proposal drift")
    if Path(ranked[0]["path"]).resolve() != SINGLE_BEST.resolve():
        raise SystemExit("single_best proposal path drift")
    document = json.loads(SINGLE_BEST.read_text(encoding="utf-8"))
    attestation = validate_quant_plan(
        document, model="gr00t", source=str(SINGLE_BEST)
    )
    return {
        "selection_rule": "highest predicted-utility non-base proposal before complete-network adjudication",
        "proposal_rank": 1,
        "predicted_utility": float(ranked[0]["predicted_utility"]),
        "plan": artifact(SINGLE_BEST),
        "proposal_manifest": artifact(PROPOSALS),
        "full_network_scores_excluded_from_selection": artifact(FULLNET_SCORES),
        "quantized_w4_layers": attestation["quantized_w4_layers"],
        "retained_fp16_layers": attestation["retained_fp16_target_layers"],
        "protected_layers": document["protected_layers"],
        "total_bytes": document["total_bytes"],
        "table1_total_static_bytes": document["table1_total_static_bytes"],
    }


def baseline_coverage(expected: set[tuple[str, int]]) -> dict[str, Any]:
    observed: dict[tuple[str, int], Path] = {}
    for path in sorted(BASELINE_RESULTS.glob("**/*.jsonl")):
        if path.name.startswith(("gpu_efficiency", "gpu_server_efficiency")):
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config") != "full_context_v2":
                continue
            key = (str(row["task"]), int(row["seed"]))
            if key not in expected:
                continue
            if row.get("status") != "complete":
                raise SystemExit(f"incomplete baseline row: {path}:{line_number}")
            if key in observed:
                raise SystemExit(f"duplicate baseline pair: {key}")
            observed[key] = path
    if set(observed) != expected:
        raise SystemExit(f"baseline coverage drift: {len(observed)}/{len(expected)}")
    return {
        "root": str(BASELINE_RESULTS.resolve()),
        "episodes": len(observed),
        "tasks": len({task for task, _seed in observed}),
        "seeds": list(SEEDS),
        "source_files": sorted(str(path.resolve()) for path in set(observed.values())),
    }


def prepare() -> None:
    local_report = materialize_local_mse_plan()
    no_fullnet = validate_no_fullnet_arm()
    m0 = json.loads(M0_PLAN.read_text(encoding="utf-8"))
    m0_attestation = validate_quant_plan(m0, model="gr00t", source=str(M0_PLAN))
    require_protocol_attestation(m0.get("meta") or {}, source=str(M0_PLAN))
    local = json.loads(LOCAL_PLAN.read_text(encoding="utf-8"))
    single = json.loads(SINGLE_BEST.read_text(encoding="utf-8"))
    modes = {
        "m0": plan_modes(m0),
        "local_mse_selection": plan_modes(local),
        "no_fullnet_check": plan_modes(single),
    }
    if any(set(value) != set(modes["m0"]) for value in modes.values()):
        raise SystemExit("Table-3 target-layer inventory drift")

    tasks = PROTOCOL["table1"]["tasks"]
    expected = {
        (task, seed)
        for split in SPLITS
        for task in tasks[split]
        for seed in SEEDS
    }
    baseline = baseline_coverage(expected)
    specs: dict[str, dict[str, Any]] = {}
    runtime_audits: dict[str, Any] = {}
    for split in SPLITS:
        table = json.loads(
            (TABLE1_SPECS / f"{split}.json").read_text(encoding="utf-8")
        )
        attribution = json.loads(
            (ATTR_SPECS / f"{split}.json").read_text(encoding="utf-8")
        )
        m0_runtime = copy.deepcopy(
            next(row for row in table["configs"] if row["id"] == "full_context_v2")
        )
        full_runtime = copy.deepcopy(
            next(row for row in attribution["configs"] if row["id"] == "c")
        )
        scale = ROOT / "static_a8" / split / "m0_static_a8.npz"
        scale_meta_path = Path(str(scale) + ".meta.json")
        if not scale.is_file() or not scale_meta_path.is_file():
            raise FileNotFoundError(f"missing plan-specific Static-A8 artifact: {scale}")
        scale_meta = json.loads(scale_meta_path.read_text(encoding="utf-8"))
        if scale_meta.get("plan_sha256") != sha256_file(M0_PLAN):
            raise SystemExit(f"{split}: Static-A8 plan lineage drift")
        scale_checks = {
            "schema_version": int(scale_meta.get("schema_version", -1)) == 3,
            "kind": scale_meta.get("kind") == "v3_per_flow_step_a8",
            "protocol": scale_meta.get("protocol_sha256")
            == CROSS_MODEL_PROTOCOL_SHA256,
            "wrapped_layers": int(scale_meta.get("wrapped_layers", -1)) == 100,
            "calib_batches": int(scale_meta.get("calib_batches", -1)) == 32,
            "act_percentile": float(scale_meta.get("act_percentile", -1.0))
            == 99.9,
            "denoising_steps": int(scale_meta.get("denoising_steps", -1)) == 4,
            "prefix_tables": int(scale_meta.get("prefix_llm_tables", -1)) == 1,
            "dit_tables": int(scale_meta.get("dit_flow_step_tables", -1)) == 4,
            "calibration_buffer": scale_meta.get("source_buffer_sha256")
            == CROSS_MODEL_PROTOCOL["data"]["calibration_buffer"]["sha256"],
            "inventory_subset": scale_meta.get("inventory_subset") is True,
        }
        failed_scale_checks = [
            name for name, passed in scale_checks.items() if not passed
        ]
        if failed_scale_checks:
            raise SystemExit(
                f"{split}: Static-A8 metadata drift: {failed_scale_checks}"
            )
        configurations = {
            "static_a8": {
                **copy.deepcopy(m0_runtime),
                "id": "static_a8",
                "act_scale": str(scale.resolve()),
                "activation_mode": "static_a8",
                "meta": {
                    "role": "table3_activation_range_ablation",
                    "only_changed_decision": "frozen calibration range instead of DyRange-A8",
                    "runtime_correction": False,
                    "runtime_selector": False,
                },
            },
            "local_mse_selection": {
                **copy.deepcopy(full_runtime),
                "id": "local_mse_selection",
                "expected_wrapped": int(local["quantized_w4_layers"]),
                "plan": str(LOCAL_PLAN.resolve()),
                "hessian_w4": str(
                    (HESSIAN_ROOT / "local_mse_selection" / split / "hessian_w4.npz").resolve()
                ),
                "activation_mode": "dynamic_a8",
                "act_scale": None,
                "meta": {
                    "role": "table3_precision_metric_ablation",
                    "only_changed_decision": "local Hessian reconstruction MSE selects the mask",
                    "runtime_correction": False,
                    "runtime_selector": False,
                },
            },
            "no_fullnet_check": {
                **copy.deepcopy(full_runtime),
                "id": "no_fullnet_check",
                "expected_wrapped": int(single["quantized_w4_layers"]),
                "plan": str(SINGLE_BEST.resolve()),
                "hessian_w4": str(
                    (HESSIAN_ROOT / "no_fullnet_check" / split / "hessian_w4.npz").resolve()
                ),
                "activation_mode": "dynamic_a8",
                "act_scale": None,
                "meta": {
                    "role": "table3_complete_network_verifier_ablation",
                    "only_changed_decision": "deploy highest-ranked non-base proposal without complete-network adjudication",
                    "runtime_correction": False,
                    "runtime_selector": False,
                },
            },
        }
        for config in configurations.values():
            for field in ("gpu", "port", "egl_device", "replicas"):
                config.pop(field, None)
            if config.get("errorfold") or config.get("atm") or config.get("ohb"):
                raise SystemExit(f"{split}/{config['id']}: correction state drift")
        if configurations["static_a8"]["plan"] != str(M0_PLAN.resolve()):
            raise SystemExit(f"{split}: Static-A8 mask drift")
        if configurations["static_a8"]["hessian_w4"] != m0_runtime["hessian_w4"]:
            raise SystemExit(f"{split}: Static-A8 W4 payload drift")
        common_fields = ("packdir", "errorfold", "atm", "ohb", "ohb_only")
        for identifier, config in configurations.items():
            mismatch = {
                field: (config.get(field), m0_runtime.get(field))
                for field in common_fields
                if config.get(field) != m0_runtime.get(field)
            }
            if mismatch:
                raise SystemExit(f"{split}/{identifier}: runtime mismatch: {mismatch}")
            spec = {
                "schema_version": 1,
                "kind": "gr00t_table3_quick_execution_spec",
                "purpose": "paired reduced closed-loop Table-3 component ablation",
                "task_set": split,
                "tasks": list(tasks[split]),
                "seeds": "0-9",
                "configs": [config],
                "comparisons": {
                    "paired_external_baseline": "full_context_v2",
                    "unit": "paired_task_seed",
                },
                "decision": {
                    "role": "core_component_ablation",
                    "result_feedback_allowed": False,
                    "reduced_rollout_budget": True,
                },
            }
            path = ROOT / "specs" / identifier / f"{split}.json"
            stable_write(path, spec)
            specs[f"{identifier}/{split}"] = artifact(path)
        runtime_audits[split] = {
            "m0_hessian": artifact(m0_runtime["hessian_w4"]),
            "full_inventory_parent_hessian": artifact(full_runtime["hessian_w4"]),
            "local_mse_hessian_subset": artifact(
                HESSIAN_ROOT / "local_mse_selection" / split / "hessian_w4.npz"
            ),
            "no_fullnet_hessian_subset": artifact(
                HESSIAN_ROOT / "no_fullnet_check" / split / "hessian_w4.npz"
            ),
            "identity_pack": {
                "path": str(Path(m0_runtime["packdir"]).resolve()),
                "same_for_all_arms": True,
            },
            "static_a8": artifact(scale),
            "static_a8_meta": artifact(scale_meta_path),
        }

    manifest = {
        "schema_version": 1,
        "kind": "gr00t_table3_reduced_closed_loop_preregistration",
        "immutable": True,
        "result_feedback_allowed": False,
        "scope": "Table-3 component ablation, not a replacement for the 2500-episode headline estimate",
        "method": {
            "id": "full_context_v2",
            "plan": artifact(M0_PLAN),
            "quantized_w4_layers": m0_attestation["quantized_w4_layers"],
            "retained_fp16_layers": m0_attestation[
                "retained_fp16_target_layers"
            ],
        },
        "contrasts": {
            "static_a8": {
                "changed_decision": "activation range",
                "frozen_controls": "M0 precision mask, W4 payload, checkpoint, task-seed pairs, and runtime correction state",
            },
            "local_mse_selection": {
                "changed_decision": "precision metric",
                "frozen_controls": "Table-1 byte ceiling, full layer generator, calibration buffer, checkpoint, and DyRange-A8 runtime",
                "selection": artifact(LOCAL_REPORT),
            },
            "no_fullnet_check": {
                "changed_decision": "complete-network adjudication",
                "frozen_controls": "proposal generator, calibration buffers, checkpoint, task-seed pairs, and DyRange-A8 runtime",
                **no_fullnet,
            },
        },
        "runtime_contract": {
            "weight_quantization": "Hessian group-64 W4",
            "row_rotation": "identity",
            "flow_steps": 4,
            "action_horizon": 16,
            "paired_action_noise": True,
            "runtime_correction": False,
            "runtime_selector": False,
        },
        "full_context_protocol": protocol_attestation(),
        "evaluation": {
            "benchmark": PROTOCOL["table1"]["benchmark"],
            "task_sets": tasks,
            "trial_seeds": list(SEEDS),
            "episodes_per_config": len(expected),
            "total_new_episodes": len(expected) * len(CONFIGS),
            "paired_baseline": baseline,
        },
        "statistics": {
            "primary": "task-macro success-rate delta against paired M0",
            "secondary": ["micro success rate", "paired exact McNemar"],
            "uncertainty": "task-then-seed hierarchical bootstrap, 10000 draws, seed 0",
            "multiplicity": "Holm adjustment over the three paired McNemar tests",
            "scope": "reduced 10-seed ablation; directional evidence without equivalence claims",
        },
        "runtime_audits": runtime_audits,
        "execution_specs": specs,
    }
    stable_write(PREREGISTRATION, manifest)
    print(
        json.dumps(
            {
                "preregistration": str(PREREGISTRATION),
                "episodes_per_config": len(expected),
                "total_new_episodes": len(expected) * len(CONFIGS),
                "local_mse_w4_fp16": [
                    local_report["quantized_w4_layers"],
                    local_report["retained_fp16_layers"],
                ],
                "no_fullnet_w4_fp16": [
                    no_fullnet["quantized_w4_layers"],
                    no_fullnet["retained_fp16_layers"],
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "prepare"))
    args = parser.parse_args()
    if args.stage == "plan":
        report = materialize_local_mse_plan()
        print(
            json.dumps(
                {
                    "plan": str(LOCAL_PLAN),
                    "w4": report["quantized_w4_layers"],
                    "fp16": report["retained_fp16_layers"],
                    "total_bytes": report["total_bytes"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        prepare()


if __name__ == "__main__":
    main()
