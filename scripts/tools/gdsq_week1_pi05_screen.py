#!/usr/bin/env python3
"""Run the preregistered pi0.5 functional screen for same-budget controls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = REPO_ROOT / "code/pi05/openpi"
ROOT = REPO_ROOT / "runs/gdsq_week1_preregistered_v1"
PREREG = ROOT / "preregistration.json"
PREREG_SHA256 = "216f1b6267b5bc9ff67cb19f9e3502c30836b0aa7e81e10ade522fa7d6104541"
CHECKPOINT = REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch"
CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"
PACK = REPO_ROOT / "runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
BUFFER = REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
FUNCTIONAL_OBSERVATIONS = 16
CALIBRATION_OBSERVATIONS = 256
CALIBRATION_BATCH_SIZE = 8
DEVELOPMENT_REPRESENTATIVES = 7

sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages/openpi-client/src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/tools"))

from openpi.policies import policy_config  # noqa: E402
from openpi.quant import (  # noqa: E402
    enable_duquant_if_configured,
    save_act_scales,
    static_scales_ready,
)
from openpi.training import config  # noqa: E402
from pi05_batched_policy import iter_batches, load_records, sample_batch  # noqa: E402
from pi05_func_metrics import d_func  # noqa: E402


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def clear_quant_environment() -> None:
    for key in list(os.environ):
        if key.startswith(("OPENPI_DUQUANT_", "OPENPI_ATM_", "OPENPI_OHB_", "OPENPI_RUNTIME_SELECTOR_")):
            os.environ.pop(key, None)


def configure_base() -> None:
    clear_quant_environment()
    os.environ.update(
        {
            "TORCHDYNAMO_DISABLE": "1",
            "OPENPI_MODEL_DTYPE": "float16",
            "OPENPI_CHECKPOINT_SHA256": CHECKPOINT_SHA256,
        }
    )


def configure_quant(plan: Path, a8: Path, wrapped: int) -> None:
    configure_base()
    os.environ.update(
        {
            "OPENPI_DUQUANT_PLAN": str(plan.resolve()),
            "OPENPI_DUQUANT_PLAN_STRICT": "1",
            "OPENPI_DUQUANT_WBITS_DEFAULT": "4",
            "OPENPI_DUQUANT_ABITS": "8",
            "OPENPI_DUQUANT_BLOCK": "64",
            "OPENPI_DUQUANT_BLOCK_OUT": "64",
            "OPENPI_DUQUANT_EXPECT_BLOCK": "64",
            "OPENPI_DUQUANT_EXPECT_WRAPPED": str(wrapped),
            "OPENPI_DUQUANT_LS": "0.15",
            "OPENPI_DUQUANT_PERMUTE": "0",
            "OPENPI_DUQUANT_ROW_ROT": "restore",
            "OPENPI_DUQUANT_ACT_PCT": "99.9",
            "OPENPI_DUQUANT_CALIB_STEPS": "32",
            "OPENPI_DUQUANT_DENOISING_STEPS": "4",
            "OPENPI_DUQUANT_PACKDIR": str(PACK.resolve()),
            "OPENPI_DUQUANT_ACT_SCALE_PATH": str(a8.resolve()),
            "OPENPI_DUQUANT_CALIB_BUFFER_SHA256": sha256_file(BUFFER),
            "OPENPI_DUQUANT_STRICT_ARTIFACTS": "1",
            "OPENPI_DUQUANT_PRECACHE_WEIGHTS": "1",
            "OPENPI_DUQUANT_TRITON": "0",
            "OPENPI_DUQUANT_QUIET": "1",
            "OPENPI_ATM_ENABLE": "0",
            "OPENPI_OHB_ENABLE": "0",
        }
    )


def load_policy(device: str):
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_pretrain_human300"), CHECKPOINT, pytorch_device=device
    )
    policy._model.to(device)
    return policy


def trajectory(policy, records: list[dict[str, Any]], device: str) -> torch.Tensor:
    chunks = []
    for batch in iter_batches(records, CALIBRATION_BATCH_SIZE):
        _final, values = sample_batch(
            policy,
            batch,
            device,
            num_steps=4,
            return_trajectory=True,
        )
        chunks.append(values.detach().to(torch.float32).cpu())
    return torch.cat(chunks, dim=1)


def ensure_a8(
    policy,
    plan: Path,
    a8: Path,
    wrapped: int,
    calibration_records: list[dict[str, Any]],
    device: str,
) -> None:
    if a8.is_file():
        require(static_scales_ready(policy._model), f"existing A8 did not load: {a8}")
        sidecar = read_json(Path(str(a8) + ".json"))
        metadata = sidecar.get("metadata") or {}
        require(sidecar.get("npz_sha256") == sha256_file(a8), f"A8 hash drift: {a8}")
        require(metadata.get("plan_sha256") == sha256_file(plan), f"A8 plan drift: {a8}")
        require(int(metadata.get("wrapped_layers", -1)) == wrapped, f"A8 wrapped drift: {a8}")
        return
    for index, batch in enumerate(iter_batches(calibration_records, CALIBRATION_BATCH_SIZE), 1):
        actions = sample_batch(policy, batch, device, num_steps=4)
        require(tuple(actions.shape) == (8, 50, 32), f"invalid A8 batch {index}: {actions.shape}")
        require(bool(torch.isfinite(actions).all()), f"non-finite A8 batch {index}")
    require(static_scales_ready(policy._model), "A8 scales not ready after 32 batches")
    a8.parent.mkdir(parents=True, exist_ok=True)
    save_act_scales(
        policy._model,
        a8,
        {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "calibration_buffer_sha256": sha256_file(BUFFER),
            "calibration_buffer_path": str(BUFFER.resolve()),
            "plan_path": str(plan.resolve()),
            "plan_sha256": sha256_file(plan),
            "pack_dir": str(PACK.resolve()),
            "calibration_reference": "plan_true_mixed_deployment",
            "enable_permute": False,
            "calibration_seed": 0,
            "calibration_batch_size": 8,
            "calibration_observations": 256,
            "wrapped_layers": wrapped,
            "protocol": "GR00T-N1.5-aligned-target16-d4",
        },
    )


def count_wrapped(plan: dict[str, Any]) -> int:
    return sum(
        not bool(row.get("skip", not int(row.get("bits", 0) or 0)))
        and int(row.get("bits", 0) or 0) > 0
        for row in (plan.get("layers") or {}).values()
    )


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
    family_record = prereg["models"]["pi05"]["plans"][key]
    by_index = {
        candidate_index_from_path(row["path"]): row for row in family_record["files"]
    }
    records = [by_index[int(index)] for index in family_record["functional_candidate_indices"]]
    require(
        len(records) == family_record["functional_discrimination_count"] == 7,
        "functional candidate count drift",
    )
    return family_record, records


def ensure_result_blind() -> None:
    for path in (ROOT / "execution/runs").glob("pi05_controls_dev4*/results/**/*.jsonl"):
        require(path.stat().st_size == 0, f"development results predate screen: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_result_blind()
    family_record, candidates = plan_records(args.family)
    output = Path(args.out).resolve()
    journal_path = output.with_suffix(".journal.json")
    if output.exists():
        payload = read_json(output)
        require(payload.get("complete") is True, "existing screen report incomplete")
        return payload
    journal = (
        read_json(journal_path)
        if journal_path.exists()
        else {
            "schema_version": 1,
            "kind": "gdsq_vla_pi05_functional_screen_journal",
            "family": args.family,
            "preregistration_sha256": PREREG_SHA256,
            "scores": {},
        }
    )
    require(journal.get("preregistration_sha256") == PREREG_SHA256, "screen journal drift")
    require(sha256_file(CHECKPOINT / "model.safetensors") == CHECKPOINT_SHA256, "checkpoint drift")
    calibration_records = load_records(BUFFER, CALIBRATION_OBSERVATIONS)
    functional_records = calibration_records[:FUNCTIONAL_OBSERVATIONS]

    configure_base()
    fp16 = load_policy(args.device)
    reference = trajectory(fp16, functional_records, args.device)
    del fp16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for record in candidates:
        plan_path = Path(record["path"]).resolve()
        index = candidate_index_from_path(plan_path)
        key = f"{index:02d}"
        require(sha256_file(plan_path) == record["sha256"], f"plan drift: {plan_path}")
        if key in journal["scores"]:
            require(journal["scores"][key]["plan_sha256"] == record["sha256"], "journal plan drift")
            continue
        plan = read_json(plan_path)
        wrapped = count_wrapped(plan)
        a8_path = ROOT / "execution/a8/pi05" / f"{args.family}_{index:02d}_p999_b32x8.npz"
        configure_quant(plan_path, a8_path, wrapped)
        started = time.time()
        policy = load_policy(args.device)
        runtime = enable_duquant_if_configured(policy._model)
        require(int(runtime.get("wrapped_layers", -1)) == wrapped, "runtime wrapped drift")
        ensure_a8(policy, plan_path, a8_path, wrapped, calibration_records, args.device)
        candidate = trajectory(policy, functional_records, args.device)
        metrics = d_func(reference, candidate, gamma=1.2)
        journal["scores"][key] = {
            "candidate_index": index,
            "plan_id": record["plan_id"],
            "plan_path": str(plan_path),
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
        del policy, candidate
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
        "kind": "gdsq_vla_pi05_functional_screen",
        "complete": True,
        "result_blind": True,
        "family": args.family,
        "preregistration_sha256": PREREG_SHA256,
        "candidate_count": family_record["candidate_count"],
        "functional_candidate_indices": family_record["functional_candidate_indices"],
        "functional_discrimination_count": family_record["functional_discrimination_count"],
        "functional_observations": FUNCTIONAL_OBSERVATIONS,
        "functional_observation_indices": list(range(FUNCTIONAL_OBSERVATIONS)),
        "functional_buffer_path": str(BUFFER.resolve()),
        "functional_buffer_sha256": sha256_file(BUFFER),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("random", "action_only"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "complete": payload["complete"],
                "family": payload["family"],
                "screened": len(payload["scores"]),
                "development_representatives": [
                    row["candidate_index"] for row in payload["development_representatives"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
