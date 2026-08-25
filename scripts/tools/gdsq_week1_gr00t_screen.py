#!/usr/bin/env python3
"""Run the preregistered GR00T functional screen for same-budget controls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "runs/gdsq_week1_preregistered_v1"
PREREG = ROOT / "preregistration.json"
PREREG_SHA256 = "216f1b6267b5bc9ff67cb19f9e3502c30836b0aa7e81e10ade522fa7d6104541"
CHECKPOINT = REPO_ROOT / (
    "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
    "target_posttraining/atomic_seen/checkpoint-60000"
)
PACK = REPO_ROOT / (
    "checkpoints/packs/robocasa365/"
    "duquant_packed_robocasa365_protocolfix_d4_w4a8_b64c32ls015"
)
FUNCTIONAL_SEED = 2026082302
FUNCTIONAL_OBSERVATIONS = 16
CALIBRATION_OBSERVATIONS = 256
CALIBRATION_BATCH_SIZE = 8
DEVELOPMENT_REPRESENTATIVES = 7

import sys

sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts/tools"))

from gr00t_func_metrics import d_func  # noqa: E402
from gr00t_sensitivity_probe import run_rollouts  # noqa: E402
from gr00t_v2_common import (  # noqa: E402
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    ensure_a8_calibrated,
    ensure_flash_attn_rpath,
    fixed_calibration_buffer,
    load_policy,
    make_obs,
    set_quant_env,
    strip_quant_env,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def count_wrapped(plan: dict[str, Any]) -> int:
    return sum(
        not bool(row.get("skip", not int(row.get("bits", 0) or 0)))
        and int(row.get("bits", 0) or 0) > 0
        for row in (plan.get("layers") or {}).values()
    )


def ensure_no_development_results() -> None:
    for path in (ROOT / "execution/runs").glob("gr00t_controls_dev4_wave*/*.jsonl"):
        require(path.stat().st_size == 0, f"development results predate screen freeze: {path}")


def candidate_index_from_path(path: str | Path) -> int:
    """Parse ``*_NN.plan.json`` without treating ``.plan`` as part of NN."""
    name = Path(path).name
    suffix = ".plan.json"
    require(name.endswith(suffix), f"invalid candidate plan filename: {name}")
    token = name[: -len(suffix)].rsplit("_", 1)[-1]
    require(token.isdigit(), f"invalid candidate index in filename: {name}")
    return int(token)


def plan_records(family: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(sha256_file(PREREG) == PREREG_SHA256, "week-1 preregistration drift")
    prereg = read_json(PREREG)
    key = "search_matched_random" if family == "random" else "action_only"
    family_record = prereg["models"]["gr00t"]["plans"][key]
    indices = family_record["functional_candidate_indices"]
    by_index = {
        candidate_index_from_path(row["path"]): row
        for row in family_record["files"]
    }
    records = [by_index[index] for index in indices]
    require(
        len(records) == family_record["functional_discrimination_count"],
        "functional candidate count drift",
    )
    return family_record, records


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_no_development_results()
    family_record, candidates = plan_records(args.family)
    output = Path(args.out).resolve()
    journal_path = output.with_suffix(".journal.json")
    if output.exists():
        payload = read_json(output)
        require(payload.get("complete") is True, "existing screen report is incomplete")
        return payload

    journal = read_json(journal_path) if journal_path.exists() else {
        "schema_version": 1,
        "kind": "gdsq_vla_gr00t_functional_screen_journal",
        "family": args.family,
        "preregistration_sha256": PREREG_SHA256,
        "scores": {},
    }
    require(journal["preregistration_sha256"] == PREREG_SHA256, "screen journal drift")

    ensure_flash_attn_rpath()
    strip_quant_env()
    np_rng = np.random.default_rng(FUNCTIONAL_SEED)
    torch_generator = torch.Generator(device="cpu").manual_seed(FUNCTIONAL_SEED)
    fp16 = load_policy(
        str(CHECKPOINT),
        data_config="examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig",
        denoising_steps=4,
        device=args.device,
    )
    horizon = int(fp16.model.action_head.config.action_horizon)
    action_dim = int(fp16.model.action_head.config.action_dim)
    observations = [make_obs(np_rng, "robocasa365") for _ in range(FUNCTIONAL_OBSERVATIONS)]
    noises = [
        torch.randn(horizon, action_dim, generator=torch_generator)
        for _ in observations
    ]
    reference = run_rollouts(
        fp16.model, fp16, observations, noises, CALIBRATION_BATCH_SIZE
    )
    del fp16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    warm_obs, warm_noises, calibration_sha = fixed_calibration_buffer(
        0,
        CALIBRATION_OBSERVATIONS,
        horizon,
        action_dim,
        fmt="robocasa365",
    )
    for record in candidates:
        index = candidate_index_from_path(record["path"])
        key = f"{index:02d}"
        plan_path = Path(record["path"])
        require(sha256_file(plan_path) == record["sha256"], f"plan drift: {plan_path}")
        if key in journal["scores"]:
            require(
                journal["scores"][key]["plan_sha256"] == record["sha256"],
                f"journal plan drift at candidate {key}",
            )
            continue
        plan = read_json(plan_path)
        wrapped = count_wrapped(plan)
        a8_path = (
            ROOT
            / "execution/a8/gr00t"
            / f"{args.family}_{index:02d}_p999_b32x8.npz"
        )
        strip_quant_env()
        set_quant_env(
            DEFAULT_INCLUDE,
            DEFAULT_EXCLUDE,
            str(PACK),
            bits_default=4,
            group=64,
            ls=0.15,
            act_pct=99.9,
            calib_steps=32,
            row_rot="restore",
            act_dynamic=False,
        )
        os.environ["GR00T_DUQUANT_PLAN"] = str(plan_path.resolve())
        os.environ["GR00T_OBS_FORMAT"] = "robocasa365"
        os.environ["GR00T_DENOISING_STEPS"] = "4"
        os.environ["GR00T_ATM_ENABLE"] = "0"
        os.environ["GR00T_OHB_ENABLE"] = "0"
        started = time.time()
        policy = load_policy(
            str(CHECKPOINT),
            data_config="examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig",
            denoising_steps=4,
            device=args.device,
        )
        ensure_a8_calibrated(
            policy,
            warm_obs,
            warm_noises,
            CALIBRATION_BATCH_SIZE,
            expected_wrapped=wrapped,
            act_scale_path=str(a8_path),
            act_scale_meta={
                "buffer_sha256": calibration_sha,
                "calibration_seed": 0,
                "data_config": "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig",
                "obs_format": "robocasa365",
                "act_percentile": 99.9,
                "calib_batches": 32,
                "denoising_steps": 4,
                "plan_sha256": record["sha256"],
                "checkpoint_path": str(CHECKPOINT.resolve()),
                "wrapped_layers": wrapped,
            },
        )
        trajectory = run_rollouts(
            policy.model, policy, observations, noises, CALIBRATION_BATCH_SIZE
        )
        metrics = d_func(reference, trajectory, gamma=1.2)
        journal["scores"][key] = {
            "candidate_index": index,
            "plan_id": record["plan_id"],
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": record["sha256"],
            "a8_path": str(a8_path.resolve()),
            "a8_sha256": sha256_file(a8_path),
            "wrapped_layers": wrapped,
            "d_func": float(metrics["d_func"]),
            "d_solver": float(metrics["d_solver"]),
            "d_final": float(metrics["d_final"]),
            "d_kin": float(metrics["d_kin"]),
            "d_grip": float(metrics["d_grip"]),
            "tail_cvar90": float(metrics["tail"]["cvar90"]),
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(journal_path, journal)
        del policy, trajectory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores = [
        journal["scores"][f"{candidate_index_from_path(row['path']):02d}"]
        for row in candidates
    ]
    ranked = sorted(scores, key=lambda row: (row["d_func"], row["candidate_index"]))
    representatives = ranked[:DEVELOPMENT_REPRESENTATIVES]
    payload = {
        "schema_version": 1,
        "kind": "gdsq_vla_gr00t_functional_screen",
        "complete": True,
        "result_blind": True,
        "family": args.family,
        "preregistration_sha256": PREREG_SHA256,
        "candidate_count": family_record["candidate_count"],
        "functional_candidate_indices": family_record["functional_candidate_indices"],
        "functional_discrimination_count": family_record[
            "functional_discrimination_count"
        ],
        "functional_observations": FUNCTIONAL_OBSERVATIONS,
        "functional_subset_seed": FUNCTIONAL_SEED,
        "calibration_observations": CALIBRATION_OBSERVATIONS,
        "calibration_batches": 32,
        "calibration_batch_size": CALIBRATION_BATCH_SIZE,
        "denoising_steps": 4,
        "metric": "D_func",
        "scores": scores,
        "development_representatives": representatives,
        "development_representative_count": len(representatives),
        "development_tasks": [
            "CoffeeSetupMug",
            "OpenCabinet",
            "OpenStandMixerHead",
            "PickPlaceDrawerToCounter",
        ],
        "heldout_results_read": False,
    }
    atomic_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("random", "action_only"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "complete": payload["complete"],
                "family": payload["family"],
                "screened": len(payload["scores"]),
                "development_representatives": [
                    row["candidate_index"]
                    for row in payload["development_representatives"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
