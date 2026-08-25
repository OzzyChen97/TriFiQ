#!/usr/bin/env python3
"""Completion audit for the corrected paper-faithful π0.5 Table-1 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aggregate_pi05_robocasa365 as aggregator
from audit_pi05_table1_completion import (
    sha256_file,
    verify_artifacts,
    verify_schedule_chain,
    verify_statistical_schema,
)
from pi05_make_corrected_table1_manifest import (
    CHECKPOINT_SHA256,
    CONFIGS,
    validate_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def verify_corrected_manifest_invariants(manifest: dict, artifacts: dict) -> dict:
    protocol = manifest.get("table_1_protocol") or {}
    if set(manifest.get("configs") or {}) != set(CONFIGS):
        raise ValueError("corrected manifest does not contain exactly the four canonical configs")
    if protocol.get("trial_seeds") != list(range(50)):
        raise ValueError("corrected manifest trial seeds changed")
    task_sets = protocol.get("task_sets") or {}
    if [len(task_sets.get(name, [])) for name in (
        "atomic_seen", "composite_seen", "composite_unseen"
    )] != [18, 16, 16]:
        raise ValueError("corrected manifest task inventory changed")
    for key, value in {
        "split": "pretrain",
        "fresh_environment_per_trial": True,
        "render_enabled": True,
        "official_task_horizon": True,
        "action_horizon": 50,
        "replan_steps": 5,
        "flow_steps": 10,
        "paired_action_noise": True,
        "calibration_source": "real-on-policy-robocasa-pi05",
        "calibration_steps": 128,
        "max_calibration_trials_per_task": 5,
    }.items():
        if protocol.get(key) != value:
            raise ValueError(f"corrected manifest protocol {key} changed")
    quant = manifest.get("quantization_protocol") or {}
    for key, value in {
        "weight_bits": 4,
        "activation_bits": 8,
        "block_in": 64,
        "block_out": 64,
        "enable_permute": True,
        "lambda_smooth": 0.15,
        "activation_percentile": 99.9,
        "activation_calibration_batches": 32,
        "atm": "per-head alpha folded into action-expert q_proj",
        "ohb": "per-layer scalar beta folded into o_proj at the pre-residual interface",
        "execution_backend": "fake_quant_fp16_gemm",
        "accuracy_scope_only": True,
        "paper_integer_kernel_efficiency_reproduced": False,
    }.items():
        if quant.get(key) != value:
            raise ValueError(f"corrected manifest quantization protocol {key} changed")
    if artifacts["checkpoint"]["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("corrected manifest checkpoint is not the frozen pi0.5 checkpoint")
    inventory = json.loads(Path(artifacts["inventory"]["path"]).read_text(encoding="utf-8"))
    artifact_hashes = {name: row["sha256"] for name, row in artifacts.items()}
    artifact_hashes["candidate_inventory"] = inventory.get("candidate_inventory_sha256")
    servers = manifest.get("servers") or []
    if {row.get("config_id") for row in servers} != set(CONFIGS):
        raise ValueError("corrected manifest server configs changed")
    for server in servers:
        validate_runtime(server["config_id"], server.get("runtime") or {}, artifact_hashes)
    gdsq = [
        row["runtime"]["duquant"] for row in servers
        if row["config_id"] in ("gdsq_vla", "gdsq_vla_atmohb")
    ]
    if len(gdsq) != 2 or len({(row["plan_sha256"], row["act_scale_sha256"]) for row in gdsq}) != 1:
        raise ValueError("GDSQ rows do not share the exact same plan and A8 artifact")
    predecessor = manifest.get("invalidated_predecessor") or {}
    if predecessor.get("use") != "diagnostic only; rows must not be merged into corrected Table 1":
        raise ValueError("predecessor invalidation is missing")
    return {
        "configs": sorted(CONFIGS),
        "servers": len(servers),
        "paper_runtime_hashes_verified": True,
        "gdsq_shared_plan_and_a8": True,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("immutable") is not True or manifest.get("kind") != "pi05_corrected_paper_faithful_table1":
        raise ValueError("not a corrected immutable Table-1 manifest")
    artifacts = verify_artifacts(manifest)
    invariants = verify_corrected_manifest_invariants(manifest, artifacts)
    schedule = verify_schedule_chain(Path(args.schedule), manifest_path, manifest)
    summary = aggregator.aggregate(run_dir, args.bootstrap, allow_incomplete=False)
    statistics = verify_statistical_schema(summary, complete=True)
    result = {
        "schema_version": 2,
        "complete": True,
        "kind": "pi05_corrected_paper_faithful_table1_completion",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_aggregator_sha256": artifacts["aggregator"]["sha256"],
        "audit_tool_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap_draws": args.bootstrap,
        "artifacts": {"verified": len(artifacts), "records": artifacts},
        "schedule": schedule,
        "matrix": {
            config: {
                "completed": summary["configs"][config]["completed_episodes"],
                "missing": summary["configs"][config]["missing_episodes"],
            }
            for config in aggregator.CONFIG_ORDER
        },
        "statistics": statistics,
        "paper_faithful_invariants": {
            "enable_permute": True,
            "calibration_source": "real-on-policy-robocasa-pi05",
            "atm": "per-head folded q projection",
            "ohb": "per-layer scalar folded o_proj at the pre-residual interface",
            "gdsq_shared_plan_and_a8": True,
            "execution_backend": "fake_quant_fp16_gemm",
            "paper_integer_kernel_efficiency_reproduced": False,
            "verification": invariants,
        },
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"complete": True, "out": str(output), "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
