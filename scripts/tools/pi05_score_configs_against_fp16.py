#!/usr/bin/env python3
"""Score π0.5 quantization configs against the FP16 teacher on one buffer.

This is a diagnostic/selection tool, not an A8 calibration tool.  It answers:
given the same observations and paired denoising noise, which quantized config
best preserves the original FP16 policy actions under the paper D_func metric?
"""

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
OPENPI_ROOT = REPO_ROOT / "code" / "pi05" / "openpi"
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from openpi.policies import policy_config  # noqa: E402
from openpi.quant import (  # noqa: E402
    enable_duquant_if_configured,
    enable_pi05_atm_if_configured,
    finalize_real_quant,
)
from openpi.quant import sha256_file  # noqa: E402
from openpi.training import config  # noqa: E402
from pi05_sensitivity_probe import run_records  # noqa: E402
from fit_softfold_compensation import materialize_grid  # noqa: E402
from quantvla_cross_model_protocol import (  # noqa: E402
    PROTOCOL,
    PROTOCOL_SHA256,
    protocol_artifact,
    protocol_attestation,
    validate_quant_plan,
)
from quantvla_metric_protocol import summarize_pair  # noqa: E402
from quantvla_model_adapters import (  # noqa: E402
    canonical_trajectory,
    load_model_records,
    record_metadata,
    validate_calibration_artifact,
)


CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch"
DEFAULT_PACK = REPO_ROOT / "runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
DEFAULT_BUFFER = protocol_artifact("selection_buffer", verify=False)
DEFAULT_ORIGINAL_CALIBRATION_BUFFER = protocol_artifact(
    "calibration_buffer", verify=False
)
ALIGNED_ROOT = REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned"


DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "quantvla_w4a8_atmohb": {
        "plan": ALIGNED_ROOT / "plans/pi05_quantvla_uniform_w4a8_d4.plan.json",
        "a8": ALIGNED_ROOT / "a8/pi05_quantvla_uniform_w4a8_d4_p999_b32x8.npz",
        "atm": ALIGNED_ROOT / "atm_ohb/pi05_quantvla_uniform_w4a8_d4_static_perhead.json",
        "atm_enable": True,
        "ohb_enable": True,
        "wrapped": 180,
    },
    "gdsq_vla": {
        "plan": ALIGNED_ROOT / "plans/pi05_gdsq_cscka_16to1_d4.final_plan.json",
        "a8": ALIGNED_ROOT / "a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz",
        "atm": None,
        "wrapped": 80,
    },
    "gdsq_vla_atmohb": {
        "plan": ALIGNED_ROOT / "plans/pi05_gdsq_cscka_16to1_d4.final_plan.json",
        "a8": ALIGNED_ROOT / "a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz",
        "atm": ALIGNED_ROOT / "atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json",
        "atm_enable": True,
        "ohb_enable": True,
        "wrapped": 80,
    },
    "gdsq_vla_atm_only": {
        "plan": ALIGNED_ROOT / "plans/pi05_gdsq_cscka_16to1_d4.final_plan.json",
        "a8": ALIGNED_ROOT / "a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz",
        "atm": ALIGNED_ROOT / "atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json",
        "atm_enable": True,
        "ohb_enable": False,
        "wrapped": 80,
    },
    "gdsq_vla_ohb_only": {
        "plan": ALIGNED_ROOT / "plans/pi05_gdsq_cscka_16to1_d4.final_plan.json",
        "a8": ALIGNED_ROOT / "a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz",
        "atm": ALIGNED_ROOT / "atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json",
        "atm_enable": False,
        "ohb_enable": True,
        "wrapped": 80,
    },
    "expert_protect": {
        "plan": ALIGNED_ROOT / "diagnostics/expert_protect/pi05_gdsq_expert_mlp_protected.plan.json",
        "a8": ALIGNED_ROOT / "diagnostics/expert_protect/pi05_gdsq_expert_mlp_protected_p999_b32x8.npz",
        "atm": None,
        "wrapped": 60,
    },
    "expert_protect_atmohb": {
        "plan": ALIGNED_ROOT / "diagnostics/expert_protect/pi05_gdsq_expert_mlp_protected.plan.json",
        "a8": ALIGNED_ROOT / "diagnostics/expert_protect/pi05_gdsq_expert_mlp_protected_p999_b32x8.npz",
        "atm": ALIGNED_ROOT / "diagnostics/expert_protect/pi05_gdsq_expert_mlp_protected_static_perhead.json",
        "atm_enable": True,
        "ohb_enable": True,
        "wrapped": 60,
    },
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--buffer", default=str(DEFAULT_BUFFER))
    parser.add_argument("--artifact-calibration-buffer", default=str(DEFAULT_ORIGINAL_CALIBRATION_BUFFER))
    parser.add_argument("--pack-dir", default=str(DEFAULT_PACK))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-obs", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=1.2)
    parser.add_argument(
        "--selection-metric",
        choices=("d_pac", "d_func"),
        default="d_pac",
        help="FP16-relative frozen-buffer metric used to select the best config.",
    )
    parser.add_argument(
        "--pac-overlap-weight",
        type=float,
        default=0.0,
        help="Compatibility flag; the adapter-only protocol requires exactly 0.",
    )
    parser.add_argument(
        "--include",
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated config ids from the default registry.",
    )
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help="Require A8/ATM artifacts to match the scoring buffer. Off by default for FP16-guided selection buffers.",
    )
    parser.add_argument(
        "--out",
        default=str(ALIGNED_ROOT / "diagnostics/task_reset_probe/task_reset_action_probe_16obs.json"),
    )
    parser.add_argument(
        "--softfold-grid",
        action="store_true",
        help="Evaluate the complete selector-free 9x9 ATM/OHB gate grid.",
    )
    parser.add_argument(
        "--softfold-raw",
        default=None,
        help="Raw ATM/OHB artifact used to materialize --softfold-grid coefficients.",
    )
    parser.add_argument(
        "--softfold-base",
        choices=tuple(DEFAULT_CONFIGS),
        default="quantvla_w4a8_atmohb",
        help="Quant plan/A8 configuration held fixed during the SoftFold grid.",
    )
    parser.add_argument(
        "--softfold-grid-dir",
        default=None,
        help="Directory for deterministic effective-coefficient grid artifacts.",
    )
    parser.add_argument(
        "--softfold-grid-shard-index",
        type=int,
        default=0,
        help="Zero-based deterministic shard of the 9x9 SoftFold grid.",
    )
    parser.add_argument(
        "--softfold-grid-shard-count",
        type=int,
        default=1,
        help="Number of disjoint deterministic SoftFold grid shards.",
    )
    return parser.parse_args()


def materialize_softfold_grid(
    *,
    raw_path: Path,
    base_spec: dict[str, Any],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    return materialize_grid(
        raw_path=raw_path,
        output_dir=output_dir,
        base_spec=base_spec,
    )


def clear_quant_environment() -> None:
    for key in list(os.environ):
        if key.startswith(("OPENPI_DUQUANT_", "OPENPI_ATM_", "OPENPI_OHB_")):
            os.environ.pop(key, None)
    os.environ.pop("QUANTVLA_ADAPTER_ONLY", None)


def configure_base() -> None:
    clear_quant_environment()
    os.environ.update(
        {
            "TORCHDYNAMO_DISABLE": "1",
            "OPENPI_MODEL_DTYPE": "float16",
            "OPENPI_CHECKPOINT_SHA256": CHECKPOINT_SHA256,
        }
    )


def configure_quant(
    *,
    spec: dict[str, Any],
    pack_dir: Path,
    artifact_buffer_hash: str,
    strict_artifacts: bool,
) -> None:
    configure_base()
    plan = Path(spec["plan"]).expanduser().resolve()
    a8 = Path(spec["a8"]).expanduser().resolve()
    os.environ.update(
        {
            "OPENPI_DUQUANT_PLAN": str(plan),
            "OPENPI_DUQUANT_PLAN_STRICT": "1",
            "OPENPI_DUQUANT_WBITS_DEFAULT": "4",
            "OPENPI_DUQUANT_ABITS": "8",
            "OPENPI_DUQUANT_BLOCK": "64",
            "OPENPI_DUQUANT_BLOCK_OUT": "64",
            "OPENPI_DUQUANT_EXPECT_BLOCK": "64",
            "OPENPI_DUQUANT_EXPECT_WRAPPED": str(int(spec["wrapped"])),
            "OPENPI_DUQUANT_LS": "0.15",
            "OPENPI_DUQUANT_PERMUTE": "0",
            "OPENPI_DUQUANT_ROW_ROT": "restore",
            "OPENPI_DUQUANT_ACT_PCT": "99.9",
            "OPENPI_DUQUANT_CALIB_STEPS": "32",
            "OPENPI_DUQUANT_DENOISING_STEPS": "4",
            "OPENPI_DUQUANT_PACKDIR": str(pack_dir),
            "OPENPI_DUQUANT_ACT_SCALE_PATH": str(a8),
            "OPENPI_DUQUANT_REQUIRE_ACT_SCALE": "1",
            "OPENPI_DUQUANT_CALIB_BUFFER_SHA256": artifact_buffer_hash,
            "OPENPI_DUQUANT_STRICT_ARTIFACTS": "1" if strict_artifacts else "0",
            "OPENPI_DUQUANT_PRECACHE_WEIGHTS": "1",
            "OPENPI_DUQUANT_TRITON": "1",
            "OPENPI_DUQUANT_QUIET": "1",
        }
    )
    atm = spec.get("atm")
    if atm:
        atm_enable = bool(spec.get("atm_enable", True))
        ohb_enable = bool(spec.get("ohb_enable", True))
        os.environ.update(
            {
                "OPENPI_ATM_ENABLE": "1" if atm_enable else "0",
                "OPENPI_OHB_ENABLE": "1" if ohb_enable else "0",
                "OPENPI_ATM_ALPHA_PATH": str(Path(atm).expanduser().resolve()),
                "OPENPI_ATM_SCOPE": "expert",
                "OPENPI_OHB_SCOPE": "expert",
                "OPENPI_ATM_STRICT": "1" if strict_artifacts else "0",
                "OPENPI_ATM_EXPECT_LAYERS": "18" if atm_enable or ohb_enable else "0",
                "OPENPI_ATM_APPLICATION": str(
                    spec.get("atm_application", "runtime_query")
                ),
                "OPENPI_OHB_EXPECT_MODE": "per_head_pre_projection",
                "OPENPI_OHB_APPLICATION": str(
                    spec.get("ohb_application", "runtime_output")
                ),
                "OPENPI_ATM_EXPECT_PLAN_SHA256": sha256_file(plan),
                "OPENPI_ATM_EXPECT_BUFFER_SHA256": artifact_buffer_hash,
            }
        )


def load_policy(checkpoint_dir: Path, device: str):
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_pretrain_human300"),
        checkpoint_dir,
        pytorch_device=device,
    )
    policy._model.to(device)
    return policy


def digest_path(path: Path | None) -> str | None:
    return sha256_file(path) if path is not None and path.is_file() else None


def main() -> None:
    args = parse_args()
    if float(args.gamma) != float(PROTOCOL["metrics"]["d_func"]["gamma"]):
        raise ValueError("adapter-only protocol freezes gamma=1.2")
    if float(args.pac_overlap_weight) != 0.0:
        raise ValueError("pi0.5-only forecast overlap is forbidden")
    registry = dict(DEFAULT_CONFIGS)
    if args.softfold_grid:
        if not args.softfold_raw:
            raise ValueError("--softfold-grid requires --softfold-raw")
        grid_dir = (
            Path(args.softfold_grid_dir).expanduser().resolve()
            if args.softfold_grid_dir
            else Path(str(Path(args.out).expanduser().resolve()) + ".softfold_grid")
        )
        registry.update(
            materialize_softfold_grid(
                raw_path=Path(args.softfold_raw).expanduser().resolve(),
                base_spec=DEFAULT_CONFIGS[args.softfold_base],
                output_dir=grid_dir,
            )
        )
        all_grid_config_ids = sorted(key for key in registry if key.startswith("softfold_"))
        if args.softfold_grid_shard_count < 1:
            raise ValueError("--softfold-grid-shard-count must be positive")
        if not 0 <= args.softfold_grid_shard_index < args.softfold_grid_shard_count:
            raise ValueError("invalid --softfold-grid-shard-index")
        config_ids = [
            config_id
            for index, config_id in enumerate(all_grid_config_ids)
            if index % args.softfold_grid_shard_count == args.softfold_grid_shard_index
        ]
        if not config_ids:
            raise ValueError("SoftFold grid shard is empty")
    else:
        if args.softfold_grid_shard_index != 0 or args.softfold_grid_shard_count != 1:
            raise ValueError("SoftFold grid sharding requires --softfold-grid")
        config_ids = [item.strip() for item in args.include.split(",") if item.strip()]
    unknown = sorted(set(config_ids) - set(registry))
    if unknown:
        raise ValueError(f"unknown config ids: {unknown}")
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    buffer_path = Path(args.buffer).expanduser().resolve()
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    artifact_buffer_hash = sha256_file(Path(args.artifact_calibration_buffer).expanduser().resolve())
    calibration_attestation = validate_calibration_artifact(
        DEFAULT_CONFIGS[args.softfold_base]["a8"] if args.softfold_grid else registry[config_ids[0]]["a8"],
        model="pi05",
        expected_buffer_sha256=artifact_buffer_hash,
    )
    base_plan_path = Path(DEFAULT_CONFIGS[args.softfold_base]["plan"]).resolve()
    quant_selection_attestation = validate_quant_plan(
        json.loads(base_plan_path.read_text(encoding="utf-8")),
        model="pi05",
        source=str(base_plan_path),
    )
    records, buffer_provenance = load_model_records(
        buffer_path, args.n_obs, model="pi05"
    )
    record_details = record_metadata(records)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fp16_guided_quant_config_score",
        "cross_model_protocol": protocol_attestation(),
        "model_adapter": buffer_provenance["adapter"],
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_sha256": sha256_file(checkpoint_dir / "model.safetensors"),
        "buffer": str(buffer_path),
        "buffer_sha256": buffer_provenance["sha256"],
        "artifact_calibration_buffer_sha256": artifact_buffer_hash,
        "strict_artifacts": bool(args.strict_artifacts),
        "calibration_attestation": calibration_attestation,
        "quantization_selection": quant_selection_attestation,
        "n_obs": args.n_obs,
        "gamma": args.gamma,
        "selection_metric": args.selection_metric,
        "pac_overlap_weight": args.pac_overlap_weight,
        "softfold_grid": bool(args.softfold_grid),
        "softfold_grid_size": 81 if args.softfold_grid else 0,
        "softfold_grid_shard": (
            {
                "index": args.softfold_grid_shard_index,
                "count": args.softfold_grid_shard_count,
                "candidate_count": len(config_ids),
                "config_ids": config_ids,
            }
            if args.softfold_grid
            else None
        ),
        "quant_plan_sha256": (
            sha256_file(base_plan_path)
            if args.softfold_grid
            else None
        ),
        "a8_sha256": (
            sha256_file(Path(DEFAULT_CONFIGS[args.softfold_base]["a8"]).resolve())
            if args.softfold_grid
            else None
        ),
        "raw_correction_sha256": (
            sha256_file(Path(args.softfold_raw).expanduser().resolve())
            if args.softfold_grid and args.softfold_raw
            else None
        ),
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
            "pi05_atm_runtime": sha256_file(
                REPO_ROOT / "code/pi05/openpi/src/openpi/quant/atm_pi05.py"
            ),
        },
        "uses_task_labels_for_selection": False,
        "uses_rollout_success_for_selection": False,
        "records": record_details,
        "scores": {},
    }
    output = Path(args.out).expanduser().resolve()
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        invariant_keys = (
            "checkpoint_sha256",
            "buffer_sha256",
            "artifact_calibration_buffer_sha256",
            "n_obs",
            "gamma",
            "selection_metric",
            "pac_overlap_weight",
            "softfold_grid_shard",
            "quant_plan_sha256",
            "a8_sha256",
            "raw_correction_sha256",
            "source_sha256",
            "cross_model_protocol",
        )
        if any(previous.get(key) != payload.get(key) for key in invariant_keys):
            raise ValueError(f"existing score shard provenance drift: {output}")
        payload["scores"] = previous.get("scores") or {}

    configure_base()
    started = time.time()
    fp16 = load_policy(checkpoint_dir, args.device)
    reference, _actions, timings = run_records(fp16, records, args.device, noise_index=0)
    payload["fp16"] = {
        "latency_mean_s": float(np.mean(timings)),
        "elapsed_s": time.time() - started,
    }
    del fp16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for config_id in config_ids:
        if config_id in payload["scores"]:
            print(f"[fp16-guided] reuse {config_id}", flush=True)
            continue
        spec = registry[config_id]
        plan = Path(spec["plan"]).expanduser().resolve()
        a8 = Path(spec["a8"]).expanduser().resolve()
        atm = Path(spec["atm"]).expanduser().resolve() if spec.get("atm") else None
        started = time.time()
        configure_quant(
            spec={**spec, "plan": plan, "a8": a8, "atm": atm},
            pack_dir=pack_dir,
            artifact_buffer_hash=artifact_buffer_hash,
            strict_artifacts=args.strict_artifacts,
        )
        policy = load_policy(checkpoint_dir, args.device)
        runtime = enable_duquant_if_configured(policy._model)
        policy._model.to(args.device)
        atm_runtime = {"enabled": False}
        if atm is not None:
            enable_pi05_atm_if_configured(policy._model)
            atm_runtime = getattr(policy._model, "_openpi_atm_runtime", {"enabled": False})
        real_quant = finalize_real_quant(policy._model)
        runtime = getattr(policy._model, "_openpi_duquant_runtime", runtime)
        if not real_quant["packed_low_bit_residency"]:
            raise RuntimeError(f"{config_id}: packed W4 residency was not finalized")
        trajectory, _actions, timings = run_records(policy, records, args.device, noise_index=0)
        pair = summarize_pair(
            canonical_trajectory(reference, model="pi05"),
            canonical_trajectory(trajectory, model="pi05"),
            record_details,
        )
        metrics = pair["d_func_summary"]
        pac_metrics = pair["d_pac_summary"]
        payload["scores"][config_id] = {
            "plan": str(plan),
            "plan_sha256": digest_path(plan),
            "a8": str(a8),
            "a8_sha256": digest_path(a8),
            "atm": str(atm) if atm else None,
            "atm_sha256": digest_path(atm),
            "wrapped_layers": int(runtime["wrapped_layers"]),
            "atm_enabled": bool(atm_runtime.get("enabled")),
            "real_quant_residency": real_quant,
            "gate": spec.get("gate"),
            "d_func": float(metrics["d_func"]),
            "d_solver": float(metrics["d_solver"]),
            "d_final": float(metrics["d_final"]),
            "d_kin": float(metrics["d_kin"]),
            "d_grip": float(metrics["d_grip"]),
            "tail_cvar90": float(metrics["tail"]["cvar90"]),
            "per_dim": metrics["per_dim"],
            "per_obs": [float(value) for value in metrics["per_obs"]],
            "d_func_summary": metrics,
            "d_pac": pac_metrics["d_pac"],
            "d_pac_summary": pac_metrics,
            "cross_model_protocol_sha256": PROTOCOL_SHA256,
            "latency_mean_s": float(np.mean(timings)),
            "elapsed_s": time.time() - started,
        }
        atomic_json(output, payload)
        print(
            f"[fp16-guided] {config_id}: "
            f"D_func={payload['scores'][config_id]['d_func']:.6g} "
            f"D_PAC={payload['scores'][config_id]['d_pac']:.6g}",
            flush=True,
        )
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_key = args.selection_metric
    best = min(payload["scores"], key=lambda key: payload["scores"][key][metric_key])
    payload["best"] = {
        "config_id": best,
        "metric": metric_key,
        "value": payload["scores"][best][metric_key],
        "d_func": payload["scores"][best]["d_func"],
        "d_pac": payload["scores"][best]["d_pac"],
        "selection_rule": "diagnostic_argmin_only",
        "final_selection_rule": "one_standard_error_then_minimum_gate_amplitude",
        "status": "diagnostic_argmin_only_final_selection_uses_shared_one_standard_error_rule",
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "best": payload["best"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
