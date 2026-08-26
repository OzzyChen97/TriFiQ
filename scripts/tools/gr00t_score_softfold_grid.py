#!/usr/bin/env python3
"""Score a sharded 9x9 GR00T SoftFold grid against the original FP16 teacher."""

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

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from fit_softfold_compensation import materialize_grid  # noqa: E402
from gr00t_sensitivity_probe import run_rollouts  # noqa: E402
from quantvla_cross_model_protocol import (  # noqa: E402
    PROTOCOL,
    PROTOCOL_SHA256,
    protocol_artifact,
    protocol_attestation,
    validate_quant_plan,
)
from quantvla_metric_protocol import summarize_pair  # noqa: E402
from quantvla_model_adapters import (  # noqa: E402
    gr00t_rollout_inputs,
    load_model_records,
    record_metadata,
    validate_calibration_artifact,
)
from gr00t_v2_common import (  # noqa: E402
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    ensure_a8_calibrated,
    ensure_flash_attn_rpath,
    load_policy,
    set_quant_env,
    strip_quant_env,
)
from gr00t.atm import (  # noqa: E402
    enable_dit_atm_if_configured,
    reset_errorfold_attention_folds,
)
from gr00t.quantization import apply_errorfold, finalize_real_quant  # noqa: E402
from gr00t.quantization.duquant_layers import DuQuantLinear  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / (
    "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
    "target_posttraining/atomic_seen/checkpoint-60000"
)
DEFAULT_PLAN = REPO_ROOT / (
    "checkpoints/packs/robocasa365/"
    "quantvla_v1_uniform_w4a8.json"
)
DEFAULT_PACK = REPO_ROOT / (
    "checkpoints/packs/robocasa365/"
    "duquant_packed_robocasa365_protocolfix_d4_w4a8_b64c32ls015"
)
DEFAULT_A8 = REPO_ROOT / (
    "checkpoints/packs/robocasa365/a8_scales_quantvla_v1_w4a8_protocolfix_d4.npz"
)
DEFAULT_RAW = REPO_ROOT / (
    "checkpoints/packs/robocasa365/"
    "atm_alpha_beta_static_quantvla_v1_w4a8_protocolfix_d4.json"
)
DEFAULT_DATA_CONFIG = "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig"
DEFAULT_BUFFER = protocol_artifact("selection_buffer", verify=False)
DEFAULT_CALIBRATION_BUFFER = protocol_artifact("calibration_buffer", verify=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"pack directory contains no files: {root}")
    for candidate in files:
        relative = candidate.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256_file(candidate).encode("ascii") + b"\n")
    return digest.hexdigest()


def sha256_checkpoint(path: str | Path) -> str:
    root = Path(path).expanduser().resolve()
    files = [root] if root.is_file() else sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no checkpoint safetensors under {root}")
    digest = hashlib.sha256()
    for candidate in files:
        digest.update(candidate.name.encode("utf-8") + b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--pack-dir", default=str(DEFAULT_PACK))
    parser.add_argument("--a8", default=str(DEFAULT_A8))
    parser.add_argument("--hessian-w4", required=True)
    parser.add_argument("--raw-correction", default=str(DEFAULT_RAW))
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--buffer", default=str(DEFAULT_BUFFER))
    parser.add_argument(
        "--artifact-calibration-buffer", default=str(DEFAULT_CALIBRATION_BUFFER)
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-obs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--grid-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def materialize_candidates(raw_path: Path, grid_dir: Path) -> dict[str, dict[str, Any]]:
    return materialize_grid(raw_path=raw_path, output_dir=grid_dir)


def score(args: argparse.Namespace) -> dict[str, Any]:
    if float(args.gamma) != float(PROTOCOL["metrics"]["d_func"]["gamma"]):
        raise ValueError("adapter-only protocol freezes gamma=1.2")
    if int(args.denoising_steps) != int(PROTOCOL["closed_loop"]["flow_steps"]):
        raise ValueError("adapter-only protocol freezes four denoising steps")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid SoftFold grid shard")
    if args.n_obs < 1 or args.batch_size < 1:
        raise ValueError("--n-obs and --batch-size must be positive")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    plan = Path(args.plan).expanduser().resolve()
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    a8 = Path(args.a8).expanduser().resolve()
    hessian_w4 = Path(args.hessian_w4).expanduser().resolve()
    raw = Path(args.raw_correction).expanduser().resolve()
    buffer_path = Path(args.buffer).expanduser().resolve()
    artifact_buffer_path = Path(args.artifact_calibration_buffer).expanduser().resolve()
    grid_dir = Path(args.grid_dir).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    for path in (
        checkpoint / "config.json",
        plan,
        a8,
        hessian_w4,
        Path(str(hessian_w4) + ".json"),
        raw,
        buffer_path,
        artifact_buffer_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not pack_dir.is_dir():
        raise FileNotFoundError(pack_dir)

    registry = materialize_candidates(raw, grid_dir)
    all_ids = sorted(registry)
    selected_ids = [
        identifier
        for index, identifier in enumerate(all_ids)
        if index % args.shard_count == args.shard_index
    ]
    if not selected_ids:
        raise ValueError("SoftFold grid shard is empty")

    a8_meta_path = Path(str(a8) + ".meta.json")
    a8_meta = json.loads(a8_meta_path.read_text(encoding="utf-8")) if a8_meta_path.is_file() else {}
    calibration_attestation = validate_calibration_artifact(
        a8,
        model="gr00t",
        expected_buffer_sha256=sha256_file(artifact_buffer_path),
    )
    quant_selection_attestation = validate_quant_plan(
        json.loads(plan.read_text(encoding="utf-8")),
        model="gr00t",
        source=str(plan),
    )
    payload: dict[str, Any] = {
        "schema_version": 3,
        "kind": "errorfold_v3_gr00t_softfold_grid_score",
        "cross_model_protocol": protocol_attestation(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_checkpoint(checkpoint),
        "plan": str(plan),
        "plan_sha256": sha256_file(plan),
        "pack_dir": str(pack_dir),
        "pack_dir_sha256": sha256_tree(pack_dir),
        "a8": str(a8),
        "a8_sha256": sha256_file(a8),
        "hessian_w4": str(hessian_w4),
        "hessian_w4_sha256": sha256_file(hessian_w4),
        "artifact_calibration_buffer_sha256": sha256_file(artifact_buffer_path),
        "artifact_declared_buffer_sha256": a8_meta.get("buffer_sha256"),
        "calibration_attestation": calibration_attestation,
        "quantization_selection": quant_selection_attestation,
        "raw_correction": str(raw),
        "raw_correction_sha256": sha256_file(raw),
        "selection_metric": "d_pac",
        "teacher": "original_fp16",
        "uses_task_labels_for_selection": False,
        "uses_rollout_success_for_selection": False,
        "n_obs": args.n_obs,
        "gamma": args.gamma,
        "softfold_grid": True,
        "softfold_grid_size": 81,
        "softfold_grid_shard": {
            "index": args.shard_index,
            "count": args.shard_count,
            "candidate_count": len(selected_ids),
            "config_ids": selected_ids,
        },
        "source_sha256": {
            "scorer": sha256_file(Path(__file__)),
            "metric_protocol": sha256_file(
                REPO_ROOT / "scripts/tools/quantvla_metric_protocol.py"
            ),
            "model_adapter": sha256_file(
                REPO_ROOT / "scripts/tools/quantvla_model_adapters.py"
            ),
            "softfold_fitter": sha256_file(
                REPO_ROOT / "scripts/tools/fit_softfold_compensation.py"
            ),
            "gr00t_atm_runtime": sha256_file(REPO_ROOT / "code/gr00t/atm/dit_atm.py"),
        },
        "scores": {},
    }
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        invariant_keys = (
            "checkpoint_sha256",
            "plan_sha256",
            "a8_sha256",
            "hessian_w4_sha256",
            "raw_correction_sha256",
            "pack_dir_sha256",
            "cross_model_protocol",
            "n_obs",
            "gamma",
            "softfold_grid_shard",
            "source_sha256",
        )
        if any(previous.get(key) != payload.get(key) for key in invariant_keys):
            raise ValueError(f"existing score shard provenance drift: {output}")
        payload["scores"] = previous.get("scores") or {}

    ensure_flash_attn_rpath()
    strip_quant_env()
    torch.manual_seed(0)
    fp16 = load_policy(
        str(checkpoint),
        data_config=args.data_config,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    horizon = int(fp16.model.action_head.config.action_horizon)
    action_dim = int(fp16.model.action_head.config.action_dim)
    if (horizon, action_dim) != (16, 32):
        raise ValueError(f"GR00T adapter shape drift: {(horizon, action_dim)}")
    records, buffer_provenance = load_model_records(
        buffer_path, args.n_obs, model="gr00t"
    )
    observations, noises = gr00t_rollout_inputs(records)
    details = record_metadata(records)
    payload["buffer"] = str(buffer_path)
    payload["buffer_sha256"] = buffer_provenance["sha256"]
    payload["records"] = details
    payload["model_adapter"] = buffer_provenance["adapter"]
    reference_trajectory, reference_actions = run_rollouts(
        fp16.model,
        fp16,
        observations,
        noises,
        args.batch_size,
        return_physical=True,
    )
    del fp16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    warm_records, warm_provenance = load_model_records(
        artifact_buffer_path, 32 * args.batch_size, model="gr00t"
    )
    warm_obs, warm_noises = gr00t_rollout_inputs(warm_records)
    payload["a8_reproduction_buffer_sha256"] = warm_provenance["sha256"]
    expected_wrapped = sum(
        not bool(row.get("skip", not int(row.get("bits", 0) or 0)))
        and int(row.get("bits", 0) or 0) > 0
        for row in (json.loads(plan.read_text(encoding="utf-8")).get("layers") or {}).values()
    )

    # Load the complete quantized network once.  Every offline candidate
    # restores the same Hessian scales/biases and native attention projections
    # before applying its static coefficients.  This preserves exact candidate
    # semantics while avoiding 81 checkpoint reloads.
    strip_quant_env()
    set_quant_env(DEFAULT_INCLUDE, DEFAULT_EXCLUDE, str(pack_dir), row_rot="0")
    os.environ.update(
        {
            "GR00T_DUQUANT_FUSED": "1",
            "GR00T_DUQUANT_PLAN": str(plan),
            "GR00T_DUQUANT_ACT_SCALE_PATH": str(a8),
            "GR00T_DUQUANT_HESSIAN_W4_PATH": str(hessian_w4),
            "GR00T_ATM_ENABLE": "0",
            "GR00T_OHB_ENABLE": "0",
        }
    )
    policy = load_policy(
        str(checkpoint),
        data_config=args.data_config,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    ensure_a8_calibrated(
        policy,
        warm_obs,
        warm_noises,
        args.batch_size,
        expected_wrapped=expected_wrapped,
        act_scale_path=str(a8),
    )
    quant_layers = [
        module for module in policy.model.modules() if isinstance(module, DuQuantLinear)
    ]
    if (
        len(quant_layers) != expected_wrapped
        or not all(module._hessian_w4_loaded and module._fused_ready for module in quant_layers)
    ):
        raise RuntimeError("shared GR00T grid model lacks complete packed Hessian W4")

    for identifier in selected_ids:
        if identifier in payload["scores"]:
            print(f"[gr00t-softfold] reuse {identifier}", flush=True)
            continue
        reset_errorfold_attention_folds(policy.model)
        apply_errorfold(policy.model, registry[identifier]["path"])
        os.environ.update(
            {
                "GR00T_DUQUANT_FUSED": "1",
                "GR00T_DUQUANT_PLAN": str(plan),
                "GR00T_DUQUANT_ACT_SCALE_PATH": str(a8),
                "GR00T_DUQUANT_HESSIAN_W4_PATH": str(hessian_w4),
                "GR00T_ERRORFOLD_PATH": str(registry[identifier]["path"]),
                "GR00T_ATM_ENABLE": "1",
                "GR00T_OHB_ENABLE": "1",
                "GR00T_ATM_ALPHA_PATH": str(registry[identifier]["path"]),
                "GR00T_ATM_SCOPE": "dit",
                "GR00T_OHB_SCOPE": "dit",
                "GR00T_ATM_PER_STEP": "0",
                "GR00T_ATM_APPLICATION": "fold_q_weight",
                "GR00T_OHB_APPLICATION": "fold_o_weight_perhead",
                "GR00T_ERRORFOLD_OFFLINE_REUSE": "1",
            }
        )
        started = time.time()
        enable_dit_atm_if_configured(policy.model)
        real_quant = {
            "layers": len(quant_layers),
            "packed_weight_bytes": int(
                sum(module._W_packed_u4.numel() for module in quant_layers)
            ),
            "fp_weight_sized_buffers": len(quant_layers),
            "packed_low_bit_residency": False,
            "true_nibble_kernel_active": True,
            "offline_model_reuse": True,
        }
        runtime = getattr(policy.model, "_gr00t_atm_runtime", {})
        if not runtime.get("enabled") or not runtime.get("selector_free"):
            raise RuntimeError(f"{identifier}: selector-free SoftFold was not loaded: {runtime}")
        trajectory, physical_actions = run_rollouts(
            policy.model,
            policy,
            observations,
            noises,
            args.batch_size,
            return_physical=True,
        )
        pair = summarize_pair(
            reference_actions,
            physical_actions,
            details,
        )
        func = pair["d_func_summary"]
        pac = pair["d_pac_summary"]
        payload["scores"][identifier] = {
            "gate": registry[identifier]["gate"],
            "correction_norm": float(registry[identifier]["correction_norm"]),
            "artifact": str(registry[identifier]["path"]),
            "artifact_sha256": sha256_file(registry[identifier]["path"]),
            "wrapped_layers": expected_wrapped,
            "atm_runtime": runtime,
            "real_quant_residency": real_quant,
            "d_func": float(func["d_func"]),
            "d_func_summary": func,
            "per_obs": [float(value) for value in func["per_obs"]],
            "d_pac": float(pac["d_pac"]),
            "d_pac_summary": pac,
            "cross_model_protocol_sha256": PROTOCOL_SHA256,
            "elapsed_s": time.time() - started,
        }
        atomic_json(output, payload)
        print(
            f"[gr00t-softfold] {identifier}: D_func={func['d_func']:.6g} "
            f"D_PAC={pac['d_pac']:.6g}",
            flush=True,
        )
        del trajectory, physical_actions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload["deployment_residency_preflight"] = finalize_real_quant(policy.model)
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    best = min(payload["scores"], key=lambda key: payload["scores"][key]["d_pac"])
    payload["best"] = {
        "config_id": best,
        "metric": "d_pac",
        "value": payload["scores"][best]["d_pac"],
        "d_func": payload["scores"][best]["d_func"],
        "d_pac": payload["scores"][best]["d_pac"],
        "selection_rule": "diagnostic_argmin_only",
        "final_selection_rule": "paired_one_standard_error_then_minimum_correction_norm_gate_sum_interaction",
        "status": "diagnostic_argmin_only_final_selection_uses_shared_one_standard_error_rule",
    }
    atomic_json(output, payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = score(args)
    output = Path(args.out).expanduser().resolve()
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": sha256_file(output),
                "candidates": len(payload["scores"]),
                "best": payload["best"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
