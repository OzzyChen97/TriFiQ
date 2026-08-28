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
    apply_errorfold,
    enable_duquant_if_configured,
    enable_pi05_atm_if_configured,
    finalize_real_quant,
    iter_duquant_layers,
    reset_errorfold_attention_folds,
)
from openpi.quant import sha256_file  # noqa: E402
from openpi.training import config  # noqa: E402
from pi05_full_context_adapter import run_records  # noqa: E402
from fit_softfold_compensation import materialize_grid  # noqa: E402
from quantvla_cross_model_protocol import (  # noqa: E402
    PROTOCOL,
    PROTOCOL_SHA256,
    protocol_artifact,
    protocol_attestation,
    validate_quant_plan,
)
from quantvla_dynamic_a8_protocol import (  # noqa: E402
    protocol_attestation as dynamic_a8_protocol_attestation,
)
from quantvla_metric_protocol import summarize_pair  # noqa: E402
from quantvla_model_adapters import (  # noqa: E402
    canonical_physical_chunk,
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
    "quantvla_w4a8": {
        "plan": ALIGNED_ROOT / "plans/pi05_quantvla_uniform_w4a8_d4.plan.json",
        "a8": ALIGNED_ROOT / "a8/pi05_quantvla_uniform_w4a8_d4_p999_b32x8.npz",
        "atm": None,
        "wrapped": 180,
    },
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
    parser.add_argument(
        "--activation-mode",
        choices=("static_a8", "dynamic_a8", "fp16"),
        default="static_a8",
        help="Shared activation policy; dynamic_a8 uses per-forward per-channel amax.",
    )
    parser.add_argument("--n-obs", type=int, default=16)
    parser.add_argument(
        "--flow-steps", type=int, default=int(PROTOCOL["closed_loop"]["flow_steps"])
    )
    parser.add_argument(
        "--noise-rule",
        choices=("A", "B"),
        default="A",
        help="A selects/fits; B is allowed only as a frozen single-config audit.",
    )
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
        "--candidate-plan",
        action="append",
        default=[],
        metavar="ID=PLAN.json",
        help=(
            "Register an arbitrary selector-free W4/FP16 plan for offline "
            "FP16-relative scoring. Repeat for multiple compression points."
        ),
    )
    parser.add_argument(
        "--candidate-hessian",
        action="append",
        default=[],
        metavar="ID=HESSIAN.npz",
        help=(
            "Bind a candidate-specific Hessian W4 inventory. This is required "
            "when candidates retain different FP16 layer subsets."
        ),
    )
    parser.add_argument(
        "--errorfold-candidate-manifest",
        default=None,
        help=(
            "Score a selector-free set of ErrorFold artifacts while reusing one "
            "packed model. The manifest freezes the common plan/A8/Hessian "
            "artifacts and is used by the shared blockwise SoftFold search."
        ),
    )
    parser.add_argument("--candidate-shard-index", type=int, default=0)
    parser.add_argument("--candidate-shard-count", type=int, default=1)
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
        "--hessian-w4",
        default=None,
        help="Frozen v3 group-64 Hessian W4 artifact used by every grid candidate.",
    )
    parser.add_argument(
        "--v3-plan",
        default=None,
        help="Complete adapter-bound W4/group64 plan held fixed by the v3 grid.",
    )
    parser.add_argument(
        "--v3-a8",
        default=None,
        help="One-prefix/four-flow-step A8 artifact held fixed by the v3 grid.",
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
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, dict[str, Any]]:
    return materialize_grid(
        raw_path=raw_path,
        output_dir=output_dir,
        base_spec=base_spec,
        shard_index=shard_index,
        shard_count=shard_count,
    )


def clear_quant_environment() -> None:
    for key in list(os.environ):
        if key.startswith(("OPENPI_DUQUANT_", "OPENPI_ATM_", "OPENPI_OHB_")) or key == "OPENPI_ERRORFOLD_PATH":
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
    activation_mode: str = "static_a8",
    flow_steps: int = 4,
) -> None:
    configure_base()
    plan = Path(spec["plan"]).expanduser().resolve()
    a8 = Path(spec["a8"]).expanduser().resolve()
    os.environ.update(
        {
            "OPENPI_DUQUANT_PLAN": str(plan),
            "OPENPI_DUQUANT_PLAN_STRICT": "1",
            "OPENPI_DUQUANT_WBITS_DEFAULT": "4",
            "OPENPI_DUQUANT_ABITS": "0" if activation_mode == "fp16" else "8",
            "OPENPI_DUQUANT_BLOCK": "64",
            "OPENPI_DUQUANT_BLOCK_OUT": "64",
            "OPENPI_DUQUANT_EXPECT_BLOCK": "64",
            "OPENPI_DUQUANT_EXPECT_WRAPPED": str(int(spec["wrapped"])),
            "OPENPI_DUQUANT_LS": "0.15",
            "OPENPI_DUQUANT_PERMUTE": "0",
            "OPENPI_DUQUANT_ROW_ROT": "0" if spec.get("hessian_w4") else "restore",
            "OPENPI_DUQUANT_ACT_PCT": "99.9",
            "OPENPI_DUQUANT_CALIB_STEPS": "32",
            "OPENPI_DUQUANT_DENOISING_STEPS": str(int(flow_steps)),
            "OPENPI_DUQUANT_PACKDIR": str(pack_dir),
            "OPENPI_DUQUANT_ACT_DYNAMIC": "1" if activation_mode == "dynamic_a8" else "0",
            "OPENPI_DUQUANT_REQUIRE_ACT_SCALE": "1" if activation_mode == "static_a8" else "0",
            "OPENPI_DUQUANT_CALIB_BUFFER_SHA256": artifact_buffer_hash,
            "OPENPI_DUQUANT_STRICT_ARTIFACTS": "1" if strict_artifacts else "0",
            "OPENPI_DUQUANT_PRECACHE_WEIGHTS": "1",
            "OPENPI_DUQUANT_TRITON": "1",
            "OPENPI_DUQUANT_QUIET": "1",
        }
    )
    if activation_mode == "static_a8":
        os.environ["OPENPI_DUQUANT_ACT_SCALE_PATH"] = str(a8)
    else:
        os.environ.pop("OPENPI_DUQUANT_ACT_SCALE_PATH", None)
    if spec.get("hessian_w4"):
        os.environ["OPENPI_DUQUANT_HESSIAN_W4_PATH"] = str(
            Path(spec["hessian_w4"]).expanduser().resolve()
        )
    if spec.get("errorfold"):
        os.environ["OPENPI_ERRORFOLD_PATH"] = str(
            Path(spec["errorfold"]).expanduser().resolve()
        )
    atm = spec.get("errorfold") or spec.get("atm")
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
                    spec.get(
                        "atm_application",
                        "fold_q_weight" if spec.get("errorfold") else "runtime_query",
                    )
                ),
                "OPENPI_OHB_APPLICATION": str(
                    spec.get(
                        "ohb_application",
                        "fold_o_weight_perhead"
                        if spec.get("errorfold")
                        else "runtime_output",
                    )
                ),
                "OPENPI_ATM_EXPECT_PLAN_SHA256": sha256_file(plan),
                "OPENPI_ATM_EXPECT_BUFFER_SHA256": artifact_buffer_hash,
            }
        )
        if not spec.get("errorfold"):
            os.environ["OPENPI_OHB_EXPECT_MODE"] = "per_head_pre_projection"


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
    if args.candidate_plan and (args.softfold_grid or args.errorfold_candidate_manifest):
        raise ValueError(
            "--candidate-plan cannot be combined with a reusable ErrorFold candidate set"
        )
    if args.softfold_grid and args.errorfold_candidate_manifest:
        raise ValueError(
            "--softfold-grid and --errorfold-candidate-manifest are mutually exclusive"
        )
    for raw_candidate in args.candidate_plan:
        if "=" not in raw_candidate:
            raise ValueError("--candidate-plan must use ID=PLAN.json")
        config_id, raw_path = raw_candidate.split("=", 1)
        if (
            not config_id
            or any(not (char.isalnum() or char in "_.-") for char in config_id)
            or config_id in registry
        ):
            raise ValueError(f"invalid or duplicate candidate id: {config_id!r}")
        plan = Path(raw_path).expanduser().resolve()
        payload = json.loads(plan.read_text(encoding="utf-8"))
        selection = validate_quant_plan(
            payload, model="pi05", source=f"candidate {config_id}"
        )
        registry[config_id] = {
            "plan": plan,
            # Dynamic A8/FP16 modes do not consume this table, but keeping a
            # valid placeholder makes the common scoring path explicit.
            "a8": DEFAULT_CONFIGS["quantvla_w4a8"]["a8"],
            "atm": None,
            "wrapped": selection["quantized_w4_layers"],
        }
    for raw_hessian in args.candidate_hessian:
        if "=" not in raw_hessian:
            raise ValueError("--candidate-hessian must use ID=HESSIAN.npz")
        config_id, raw_path = raw_hessian.split("=", 1)
        if config_id not in registry:
            raise ValueError(f"candidate Hessian references unknown id: {config_id!r}")
        hessian = Path(raw_path).expanduser().resolve()
        if not hessian.is_file() or not Path(str(hessian) + ".json").is_file():
            raise FileNotFoundError(f"candidate Hessian artifact is incomplete: {hessian}")
        registry[config_id] = {**registry[config_id], "hessian_w4": str(hessian)}
    base_spec: dict[str, Any] | None = None
    candidate_manifest: dict[str, Any] | None = None
    candidate_manifest_path: Path | None = None
    if args.softfold_grid:
        if not args.softfold_raw:
            raise ValueError("--softfold-grid requires --softfold-raw")
        if not args.hessian_w4:
            raise ValueError("v3 --softfold-grid requires --hessian-w4")
        if not args.v3_plan or not args.v3_a8:
            raise ValueError("v3 --softfold-grid requires --v3-plan and --v3-a8")
        grid_dir = (
            Path(args.softfold_grid_dir).expanduser().resolve()
            if args.softfold_grid_dir
            else Path(str(Path(args.out).expanduser().resolve()) + ".softfold_grid")
        )
        v3_plan = Path(args.v3_plan).expanduser().resolve()
        v3_a8 = Path(args.v3_a8).expanduser().resolve()
        v3_plan_payload = json.loads(v3_plan.read_text(encoding="utf-8"))
        v3_layers = v3_plan_payload.get("layers") or {}
        wrapped_layers = sum(
            not bool((entry if isinstance(entry, dict) else {}).get("skip", False))
            and int((entry if isinstance(entry, dict) else {}).get("bits", 0) or 0) > 0
            for entry in v3_layers.values()
        )
        if wrapped_layers <= 0:
            raise ValueError("v3 SoftFold plan does not contain a quantized W4 layer")
        base_spec = {
            **DEFAULT_CONFIGS[args.softfold_base],
            "plan": v3_plan,
            "a8": v3_a8,
            "wrapped": wrapped_layers,
            "hessian_w4": str(Path(args.hessian_w4).expanduser().resolve()),
        }
        registry.update(
            materialize_softfold_grid(
                raw_path=Path(args.softfold_raw).expanduser().resolve(),
                base_spec=base_spec,
                output_dir=grid_dir,
                shard_index=args.softfold_grid_shard_index,
                shard_count=args.softfold_grid_shard_count,
            )
        )
        all_grid_config_ids = sorted(key for key in registry if key.startswith("softfold_"))
        if args.softfold_grid_shard_count < 1:
            raise ValueError("--softfold-grid-shard-count must be positive")
        if not 0 <= args.softfold_grid_shard_index < args.softfold_grid_shard_count:
            raise ValueError("invalid --softfold-grid-shard-index")
        config_ids = all_grid_config_ids
        if not config_ids:
            raise ValueError("SoftFold grid shard is empty")
    elif args.errorfold_candidate_manifest:
        candidate_manifest_path = Path(
            args.errorfold_candidate_manifest
        ).expanduser().resolve()
        candidate_manifest = json.loads(
            candidate_manifest_path.read_text(encoding="utf-8")
        )
        if (
            candidate_manifest.get("schema_version") != 1
            or candidate_manifest.get("kind")
            != "blockwise_errorfold_candidate_manifest"
        ):
            raise ValueError("unsupported blockwise ErrorFold candidate manifest")
        common = candidate_manifest.get("common") or {}
        candidates = candidate_manifest.get("candidates") or {}
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError("blockwise ErrorFold manifest has no candidates")
        plan = Path(common["plan"]).expanduser().resolve()
        a8 = Path(common["a8"]).expanduser().resolve()
        hessian_w4 = Path(common["hessian_w4"]).expanduser().resolve()
        plan_payload = json.loads(plan.read_text(encoding="utf-8"))
        selection = validate_quant_plan(
            plan_payload, model="pi05", source=str(plan)
        )
        wrapped = int(selection["quantized_w4_layers"])
        base_spec = {
            **DEFAULT_CONFIGS["quantvla_w4a8_atmohb"],
            "plan": plan,
            "a8": a8,
            "wrapped": wrapped,
            "hessian_w4": str(hessian_w4),
        }
        if (
            args.candidate_shard_count < 1
            or not 0 <= args.candidate_shard_index < args.candidate_shard_count
        ):
            raise ValueError("invalid blockwise candidate shard")
        config_ids = []
        for candidate_index, (config_id, entry) in enumerate(sorted(candidates.items())):
            if (
                not config_id
                or any(
                    not (char.isalnum() or char in "_.-")
                    for char in config_id
                )
            ):
                raise ValueError(f"invalid blockwise candidate id: {config_id!r}")
            correction = Path(entry["errorfold"]).expanduser().resolve()
            if not correction.is_file():
                raise FileNotFoundError(correction)
            registry[config_id] = {
                **base_spec,
                "errorfold": correction,
                "atm": correction,
                "gate": entry.get("gate"),
                "gate_profile": entry.get("gate_profile"),
                "correction_norm": entry.get("correction_norm"),
            }
            if candidate_index % args.candidate_shard_count == args.candidate_shard_index:
                config_ids.append(config_id)
        if not config_ids:
            raise ValueError("blockwise candidate shard is empty")
    else:
        if args.candidate_shard_index != 0 or args.candidate_shard_count != 1:
            raise ValueError(
                "candidate sharding requires --errorfold-candidate-manifest"
            )
        if args.softfold_grid_shard_index != 0 or args.softfold_grid_shard_count != 1:
            raise ValueError("SoftFold grid sharding requires --softfold-grid")
        config_ids = [item.strip() for item in args.include.split(",") if item.strip()]
    unknown = sorted(set(config_ids) - set(registry))
    if unknown:
        raise ValueError(f"unknown config ids: {unknown}")
    if args.noise_rule == "B" and len(config_ids) != 1:
        raise ValueError("noise-B is a frozen audit and requires exactly one config")
    if args.hessian_w4 and not args.softfold_grid and not args.errorfold_candidate_manifest:
        hessian_override = str(Path(args.hessian_w4).expanduser().resolve())
        for config_id in config_ids:
            registry[config_id] = {
                **registry[config_id],
                "hessian_w4": registry[config_id].get("hessian_w4", hessian_override),
            }
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    buffer_path = Path(args.buffer).expanduser().resolve()
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    artifact_buffer_hash = sha256_file(Path(args.artifact_calibration_buffer).expanduser().resolve())
    calibration_attestation = None
    reusable_errorfold = bool(args.softfold_grid or args.errorfold_candidate_manifest)
    if args.activation_mode == "static_a8":
        assert base_spec is not None or config_ids
        calibration_attestation = validate_calibration_artifact(
            base_spec["a8"] if reusable_errorfold else registry[config_ids[0]]["a8"],
            model="pi05",
            expected_buffer_sha256=artifact_buffer_hash,
        )
    base_plan_path = Path(
        base_spec["plan"] if reusable_errorfold else registry[config_ids[0]]["plan"]
    ).resolve()
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
        "schema_version": 3 if reusable_errorfold else 1,
        "kind": (
            "errorfold_v3_pi05_softfold_grid_score"
            if args.softfold_grid
            else "errorfold_blockwise_profile_score"
            if args.errorfold_candidate_manifest
            else "fp16_guided_quant_config_score"
        ),
        "cross_model_protocol": protocol_attestation(),
        "model_adapter": buffer_provenance["adapter"],
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_sha256": sha256_file(checkpoint_dir / "model.safetensors"),
        "buffer": str(buffer_path),
        "buffer_sha256": buffer_provenance["sha256"],
        "artifact_calibration_buffer_sha256": artifact_buffer_hash,
        "strict_artifacts": bool(args.strict_artifacts),
        "activation_mode": args.activation_mode,
        "calibration_attestation": calibration_attestation,
        "quantization_selection": quant_selection_attestation,
        "n_obs": args.n_obs,
        "noise": args.noise_rule,
        "selection_role": (
            "noise_a_selection" if args.noise_rule == "A" else "frozen_noise_b_audit"
        ),
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
        "errorfold_candidate_manifest": (
            str(candidate_manifest_path) if candidate_manifest_path else None
        ),
        "errorfold_candidate_manifest_sha256": (
            sha256_file(candidate_manifest_path) if candidate_manifest_path else None
        ),
        "errorfold_candidate_shard": (
            {
                "index": args.candidate_shard_index,
                "count": args.candidate_shard_count,
                "candidate_count": len(config_ids),
                "config_ids": config_ids,
            }
            if candidate_manifest_path
            else None
        ),
        "quant_plan_sha256": (
            sha256_file(base_plan_path)
            if reusable_errorfold
            else None
        ),
        # Keep the shared provenance field identical to the GR00T scorer.
        # ``quant_plan_sha256`` remains as a compatibility alias for old
        # diagnostic readers.
        "plan_sha256": (
            sha256_file(base_plan_path)
            if reusable_errorfold
            else None
        ),
        "a8_sha256": (
            sha256_file(Path(base_spec["a8"]).resolve())
            if reusable_errorfold and args.activation_mode == "static_a8"
            else None
        ),
        "raw_correction_sha256": (
            sha256_file(Path(args.softfold_raw).expanduser().resolve())
            if args.softfold_grid and args.softfold_raw
            else None
        ),
        "hessian_w4_sha256": (
            sha256_file(Path(base_spec["hessian_w4"]).expanduser().resolve())
            if reusable_errorfold and base_spec and base_spec.get("hessian_w4")
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
            "pi05_w4_runtime": sha256_file(
                REPO_ROOT / "code/pi05/openpi/src/openpi/quant/duquant_triton.py"
            ),
        },
        "uses_task_labels_for_selection": False,
        "uses_rollout_success_for_selection": False,
        "records": record_details,
        "scores": {},
    }
    if args.activation_mode == "dynamic_a8":
        payload["dynamic_a8_protocol"] = dynamic_a8_protocol_attestation()
    output = Path(args.out).expanduser().resolve()
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        invariant_keys = (
            "checkpoint_sha256",
            "buffer_sha256",
            "artifact_calibration_buffer_sha256",
            "n_obs",
            "noise",
            "selection_role",
            "gamma",
            "selection_metric",
            "pac_overlap_weight",
            "softfold_grid_shard",
            "quant_plan_sha256",
            "plan_sha256",
            "a8_sha256",
            "raw_correction_sha256",
            "hessian_w4_sha256",
            "errorfold_candidate_manifest_sha256",
            "errorfold_candidate_shard",
            "source_sha256",
            "cross_model_protocol",
            "dynamic_a8_protocol",
            "activation_mode",
        )
        if any(previous.get(key) != payload.get(key) for key in invariant_keys):
            raise ValueError(f"existing score shard provenance drift: {output}")
        payload["scores"] = previous.get("scores") or {}

    configure_base()
    started = time.time()
    fp16 = load_policy(checkpoint_dir, args.device)
    reference, reference_actions, timings = run_records(
        fp16,
        records,
        args.device,
        noise_index=0 if args.noise_rule == "A" else 1,
        flow_steps=args.flow_steps,
    )
    payload["fp16"] = {
        "latency_mean_s": float(np.mean(timings)),
        "elapsed_s": time.time() - started,
    }
    del fp16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if reusable_errorfold:
        assert base_spec is not None
        shared_spec = {
            **base_spec,
            "atm": None,
            "errorfold": None,
            "atm_enable": False,
            "ohb_enable": False,
        }
        configure_quant(
            spec=shared_spec,
            pack_dir=pack_dir,
            artifact_buffer_hash=artifact_buffer_hash,
            strict_artifacts=args.strict_artifacts,
            activation_mode=args.activation_mode,
            flow_steps=args.flow_steps,
        )
        policy = load_policy(checkpoint_dir, args.device)
        runtime = enable_duquant_if_configured(policy._model)
        policy._model.to(args.device)
        quant_layers = iter_duquant_layers(policy._model)
        if (
            len(quant_layers) != int(shared_spec["wrapped"])
            or not all(
                module._hessian_w4_loaded and module._fused_ready
                for _, module in quant_layers
            )
        ):
            raise RuntimeError("shared pi0.5 grid model lacks complete packed Hessian W4")
    else:
        policy = None
        quant_layers = []

    for config_id in config_ids:
        if config_id in payload["scores"]:
            print(f"[fp16-guided] reuse {config_id}", flush=True)
            continue
        spec = registry[config_id]
        plan = Path(spec["plan"]).expanduser().resolve()
        a8 = Path(spec["a8"]).expanduser().resolve()
        correction = spec.get("errorfold") or spec.get("atm")
        atm = Path(correction).expanduser().resolve() if correction else None
        started = time.time()
        if reusable_errorfold:
            assert policy is not None
            reset_errorfold_attention_folds(policy._model)
            runtime = apply_errorfold(policy._model, Path(spec["errorfold"]))
            os.environ.update(
                {
                    "OPENPI_ERRORFOLD_PATH": str(Path(spec["errorfold"]).resolve()),
                    "OPENPI_ATM_ENABLE": "1",
                    "OPENPI_OHB_ENABLE": "1",
                    "OPENPI_ATM_ALPHA_PATH": str(Path(spec["errorfold"]).resolve()),
                    "OPENPI_ATM_SCOPE": "expert",
                    "OPENPI_OHB_SCOPE": "expert",
                    "OPENPI_ATM_STRICT": "1",
                    "OPENPI_ATM_EXPECT_LAYERS": "18",
                    "OPENPI_ATM_APPLICATION": "fold_q_weight",
                    "OPENPI_OHB_APPLICATION": "fold_o_weight_perhead",
                    "OPENPI_ERRORFOLD_OFFLINE_REUSE": "1",
                    "OPENPI_ATM_EXPECT_PLAN_SHA256": sha256_file(plan),
                    "OPENPI_ATM_EXPECT_BUFFER_SHA256": artifact_buffer_hash,
                }
            )
            os.environ.pop("OPENPI_OHB_EXPECT_MODE", None)
            enable_pi05_atm_if_configured(policy._model)
            atm_runtime = getattr(
                policy._model, "_openpi_atm_runtime", {"enabled": False}
            )
            real_quant = {
                "layers": len(quant_layers),
                "packed_weight_bytes": int(
                    sum(module._W_packed_u4.numel() for _, module in quant_layers)
                ),
                "fp_weight_sized_buffers": len(quant_layers),
                "packed_low_bit_residency": False,
                "true_nibble_kernel_active": True,
                "offline_model_reuse": True,
            }
        else:
            configure_quant(
                spec={**spec, "plan": plan, "a8": a8, "atm": atm},
                pack_dir=pack_dir,
                artifact_buffer_hash=artifact_buffer_hash,
                strict_artifacts=args.strict_artifacts,
                activation_mode=args.activation_mode,
                flow_steps=args.flow_steps,
            )
            policy = load_policy(checkpoint_dir, args.device)
            runtime = enable_duquant_if_configured(policy._model)
            policy._model.to(args.device)
            atm_runtime = {"enabled": False}
            if atm is not None:
                enable_pi05_atm_if_configured(policy._model)
                atm_runtime = getattr(
                    policy._model, "_openpi_atm_runtime", {"enabled": False}
                )
            real_quant = finalize_real_quant(policy._model)
            runtime = getattr(policy._model, "_openpi_duquant_runtime", runtime)
            if not real_quant["packed_low_bit_residency"]:
                raise RuntimeError(f"{config_id}: packed W4 residency was not finalized")
        trajectory, physical_actions, timings = run_records(
            policy,
            records,
            args.device,
            noise_index=0 if args.noise_rule == "A" else 1,
            flow_steps=args.flow_steps,
        )
        pair = summarize_pair(
            canonical_physical_chunk(reference_actions, model="pi05"),
            canonical_physical_chunk(physical_actions, model="pi05"),
            record_details,
        )
        metrics = pair["d_func_summary"]
        pac_metrics = pair["d_pac_summary"]
        payload["scores"][config_id] = {
            "plan": str(plan),
            "plan_sha256": digest_path(plan),
            "a8": str(a8) if args.activation_mode == "static_a8" else None,
            "a8_sha256": digest_path(a8) if args.activation_mode == "static_a8" else None,
            "hessian_w4": (
                str(Path(spec["hessian_w4"]).expanduser().resolve())
                if spec.get("hessian_w4") else None
            ),
            "hessian_w4_sha256": (
                digest_path(Path(spec["hessian_w4"]).expanduser().resolve())
                if spec.get("hessian_w4") else None
            ),
            "atm": str(atm) if atm else None,
            "atm_sha256": digest_path(atm),
            "wrapped_layers": int(runtime["wrapped_layers"]),
            "atm_enabled": bool(atm_runtime.get("enabled")),
            "real_quant_residency": real_quant,
            "gate": spec.get("gate"),
            "gate_profile": spec.get("gate_profile"),
            "correction_norm": spec.get("correction_norm"),
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
        if not reusable_errorfold:
            del policy
            policy = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if reusable_errorfold:
        assert policy is not None
        payload["deployment_residency_preflight"] = finalize_real_quant(policy._model)
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
        "selection_rule": (
            "diagnostic_argmin_only" if args.noise_rule == "A" else None
        ),
        "final_selection_rule": "paired_one_standard_error_then_minimum_correction_norm_gate_sum_interaction",
        "status": (
            "diagnostic_argmin_only_final_selection_uses_shared_one_standard_error_rule"
            if args.noise_rule == "A"
            else "frozen_noise_b_audit_no_selection"
        ),
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
