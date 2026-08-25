#!/usr/bin/env python3
"""Materialize or verify the post-week-1 GDSQ-VLA extension matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "runs/gdsq_extension_preregistered_v1"
PLAN = ROOT / "plan.json"
SUITES = {
    "libero_spatial": "libero-spatial",
    "libero_object": "libero-object",
    "libero_goal": "libero-goal",
    "libero_10": "libero-long",
}
CONFIGS = ["fp16", "quantvla_w4a8", "uniform_w6", "gdsq_vla_selector"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def checkpoint_record(path: Path) -> dict[str, Any]:
    require(path.is_dir(), f"missing checkpoint: {path}")
    weights = sorted(path.glob("*.safetensors"))
    require(weights, f"checkpoint has no safetensors: {path}")
    return {
        "path": str(path.resolve()),
        "config": artifact(path / "config.json"),
        "weights": [artifact(weight) for weight in weights],
        "weight_bytes": sum(weight.stat().st_size for weight in weights),
    }


def frozen_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == rendered, f"frozen artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def ensure_result_blind() -> None:
    offenders = [path for path in (ROOT / "results").glob("**/*.jsonl") if path.stat().st_size]
    require(not offenders, f"extension result rows predate plan freeze: {offenders[:3]}")


def build(created_utc: str | None = None) -> dict[str, Any]:
    pi_checkpoint = REPO_ROOT / "code/pi05/checkpoints/pi05_libero_pytorch"
    checkpoints = {
        "gr00t": {
            suite: checkpoint_record(REPO_ROOT / "checkpoints/gr00t" / directory)
            for suite, directory in SUITES.items()
        },
        "pi05": {"shared": checkpoint_record(pi_checkpoint)},
    }
    suite_specs = {}
    for model in ("gr00t", "pi05"):
        for suite in SUITES:
            spec = {
                "schema_version": 1,
                "kind": "gdsq_vla_libero_suite_spec",
                "result_blind": True,
                "model": model,
                "suite": suite,
                "official_tasks": 10,
                "seeds": list(range(50)),
                "expected_episodes_per_config": 500,
                "configs": CONFIGS,
                "protocol": {
                    "paired_action_noise": True,
                    "denoising_steps": 4,
                    "execute_actions": 16,
                    "official_horizons": True,
                    "fresh_environment_per_episode": True,
                    "task_macro_sr": True,
                    "bootstrap_draws": 10_000,
                    "bootstrap_unit": "task cluster",
                    "paired_test": "task-level sign-flip",
                    "holm_correction": True,
                },
                "calibration": {
                    "suite_specific": True,
                    "observations": 256,
                    "a8_percentile": 99.9,
                    "a8_batches": 32,
                    "a8_batch_size": 8,
                    "selector_refit": True,
                    "forbid_robocasa_artifacts": True,
                    "output_root": str((ROOT / "calibration" / model / suite).resolve()),
                },
                "checkpoint": (
                    checkpoints["gr00t"][suite]
                    if model == "gr00t"
                    else checkpoints["pi05"]["shared"]
                ),
                "claim_policy": {
                    "no_cross_suite_calibration_reuse": True,
                    "no_partial_coverage_claim": True,
                    "non_final_pareto_budgets_are_diagnostic": True,
                },
            }
            spec_path = ROOT / "specs/libero" / model / f"{suite}.json"
            frozen_json(spec_path, spec)
            suite_specs[f"{model}/{suite}"] = artifact(spec_path)
    return {
        "schema_version": 1,
        "kind": "gdsq_vla_post_week1_extension_preregistration",
        "immutable": True,
        "result_blind": True,
        "created_utc": created_utc or dt.datetime.now(dt.timezone.utc).isoformat(),
        "launch_after": "week-1 joint statistics and paper gates complete",
        "libero": {
            "suites": list(SUITES),
            "models": ["gr00t", "pi05"],
            "configs": CONFIGS,
            "suite_specs": suite_specs,
            "expected_formal_episodes": 2 * 4 * 4 * 500,
            "robocasa_selector_artifact_reuse_allowed": False,
        },
        "calibration_robustness": {
            "observation_counts": [16, 64, 256],
            "calibration_seeds": [0, 1, 2],
            "prompt_variants": ["canonical", "paraphrase_v1", "paraphrase_v2"],
            "reported_metrics": [
                "mask_jaccard",
                "selector_decision_agreement",
                "functional_d_func",
                "closed_loop_task_macro_sr",
            ],
            "closed_loop_scope": "10 suite tasks x 5 frozen seeds",
            "selection_feedback_allowed": False,
        },
        "storage_sr_pareto": {
            "budget_ratios_relative_to_uniform_w6": [
                {"ratio": 0.6666666667, "uniform_bit_equivalent": 4},
                {"ratio": 0.8333333333, "uniform_bit_equivalent": 5},
                {"ratio": 1.0, "uniform_bit_equivalent": 6},
                {"ratio": 1.3333333333, "uniform_bit_equivalent": 8},
            ],
            "same_selection_protocol_at_every_budget": True,
            "main_result_budget_ratio": 1.0,
            "other_budgets_diagnostic_only": True,
            "no_primary_result_reselection": True,
        },
        "deployment_measurement": {
            "reported_metrics": [
                "actual_packed_checkpoint_bytes",
                "isolated_policy_server_peak_process_memory_mib",
                "batch1_latency_ms_p50",
                "batch1_latency_ms_p95",
            ],
            "warmup_requests": 20,
            "measurement_requests": 200,
            "batch_size": 1,
            "simulator_process_excluded_from_memory": True,
            "fake_quant_claim": "theoretical static storage only",
            "fused_int4_int8_required_for_end_to_end_claim": True,
        },
        "external_baselines": {
            "ActQuant": {
                "paper": "https://arxiv.org/abs/2605.24011",
                "comparison": "common pi0.5/LIBERO protocol only if official implementation reproduces",
            },
            "Omega-QVLA": {
                "paper": "https://arxiv.org/abs/2605.28803",
                "comparison": "common pi0.5/LIBERO protocol only if official implementation reproduces",
            },
            "SQAP-VLA": {
                "paper": "https://arxiv.org/abs/2509.09090",
                "comparison": "common pi0.5/LIBERO protocol only if official implementation reproduces",
            },
            "fallback": "use internal action-only baseline and do not rank cross-paper numbers",
        },
        "excluded": ["real_robot_evaluation", "test_set_threshold_retuning"],
        "artifacts": {
            "materializer": artifact(Path(__file__).resolve()),
            "gr00t_environment": artifact(REPO_ROOT / "environments/groot_env.yml"),
            "openpi_lock": artifact(REPO_ROOT / "code/pi05/openpi/uv.lock"),
            "gr00t_libero_entrypoint": artifact(REPO_ROOT / "scripts/run_libero_eval.sh"),
            "pi05_libero_entrypoint": artifact(REPO_ROOT / "code/pi05/openpi/examples/libero/main.py"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("materialize", "verify", "status"))
    args = parser.parse_args()
    if args.command == "status":
        if not PLAN.exists():
            print(json.dumps({"materialized": False, "plan": str(PLAN)}, indent=2))
            return
        value = json.loads(PLAN.read_text(encoding="utf-8"))
        print(json.dumps({"materialized": True, "plan": str(PLAN), "sha256": sha256_file(PLAN), "suite_specs": len(value["libero"]["suite_specs"]), "result_blind": value["result_blind"]}, indent=2))
        return
    if args.command == "materialize":
        require(not PLAN.exists(), f"refusing to overwrite extension plan: {PLAN}")
        ensure_result_blind()
        payload = build()
        frozen_json(PLAN, payload)
        status = "created"
    else:
        require(PLAN.is_file(), f"missing extension plan: {PLAN}")
        existing = json.loads(PLAN.read_text(encoding="utf-8"))
        require(existing == build(existing.get("created_utc")), "extension plan drift")
        status = "verified"
    print(json.dumps({"status": status, "plan": str(PLAN), "sha256": sha256_file(PLAN)}, indent=2))


if __name__ == "__main__":
    main()
