"""Serve an OpenPI pi0.5 policy with QuantVLA quantization (DuQuant W4A8 + ATM + OHB).

Configuration via environment variables (see openpi/src/openpi/quant/):
  OPENPI_DUQUANT_WBITS_DEFAULT=4 OPENPI_DUQUANT_ABITS=8 OPENPI_DUQUANT_BLOCK=64
  OPENPI_DUQUANT_LS=0.15 OPENPI_DUQUANT_PERMUTE=0 OPENPI_DUQUANT_ROW_ROT=restore
  OPENPI_DUQUANT_ACT_PCT=99.9 OPENPI_DUQUANT_CALIB_STEPS=32
  OPENPI_DUQUANT_PACKDIR=<pack cache dir>
  OPENPI_ATM_ENABLE=1 OPENPI_ATM_ALPHA_PATH=<alpha json> OPENPI_ATM_SCOPE=expert
  OPENPI_OHB_ENABLE=1 OPENPI_OHB_SCOPE=expert

Usage (same as scripts/serve_policy.py):
  python scripts/serve_pi05_quant_policy.py --env LIBERO \
      --policy.config pi05_libero --policy.dir <checkpoint_dir> --port 8000
"""

import dataclasses
import enum
import hashlib
import logging
import os
from pathlib import Path
import socket
import json
import sys

import tyro
import torch

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

from openpi.quant import enable_duquant_if_configured
from openpi.quant import enable_pi05_atm_if_configured
from openpi.quant import finalize_real_quant
from openpi.quant import configure_runtime_selector_from_env
from openpi_client.paired_noise import PROTOCOL as PAIRED_NOISE_PROTOCOL

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))
from quantvla_cross_model_protocol import (  # noqa: E402
    adapter_attestation,
    closed_loop_runtime_protocol,
    protocol_attestation,
    validate_quant_plan,
)


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    ROBOCASA = "robocasa"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    env: EnvMode = EnvMode.LIBERO
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    denoising_steps: int = 4
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(config="pi05_aloha", dir="gs://openpi-assets/checkpoints/pi05_base"),
    EnvMode.ALOHA_SIM: Checkpoint(config="pi0_aloha_sim", dir="gs://openpi-assets/checkpoints/pi0_aloha_sim"),
    EnvMode.DROID: Checkpoint(config="pi05_droid", dir="gs://openpi-assets/checkpoints/pi05_droid"),
    EnvMode.LIBERO: Checkpoint(config="pi05_libero", dir="gs://openpi-assets/checkpoints/pi05_libero"),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def apply_quantization(policy: _policy.Policy, *, denoising_steps: int = 4) -> _policy.Policy:
    """Apply DuQuant + ATM/OHB to the underlying pi0.5 model (in-place)."""
    if denoising_steps <= 0:
        raise ValueError("denoising_steps must be positive")
    # OpenPI passes ``sample_kwargs`` to ``model.sample_actions`` on every
    # request.  Set this explicitly so the deployed sampler, calibration
    # metadata and runtime attestation cannot silently disagree.
    policy._sample_kwargs["num_steps"] = int(denoising_steps)
    runtime_selector = configure_runtime_selector_from_env()
    model = policy._model
    adapter_only = os.environ.get("QUANTVLA_ADAPTER_ONLY", "0") not in (
        "0", "false", "False", ""
    )
    duquant_runtime = enable_duquant_if_configured(model)
    model.to("cuda")
    enable_pi05_atm_if_configured(model)
    if adapter_only and duquant_runtime.get("enabled"):
        residency = finalize_real_quant(model)
        duquant_runtime = getattr(model, "_openpi_duquant_runtime")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("real-quant finalized: %s", residency)
    atm_runtime = getattr(model, "_openpi_atm_runtime", {"enabled": False, "matched_layers": 0})
    runtime = policy._metadata.setdefault("openpi_runtime", {})
    runtime["config_id"] = os.environ.get("OPENPI_CONFIG_ID", "unspecified")
    expected_checkpoint_sha256 = os.environ.get("OPENPI_CHECKPOINT_SHA256")
    if expected_checkpoint_sha256:
        runtime["checkpoint_sha256"] = expected_checkpoint_sha256
    runtime["duquant"] = {
        key: value for key, value in duquant_runtime.items() if key != "wrapped_layer_names"
    }
    runtime["atm_ohb"] = atm_runtime
    runtime["errorfold"] = {
        "enabled": bool(duquant_runtime.get("errorfold_path")),
        "artifact_path": duquant_runtime.get("errorfold_path"),
        "errorfold_application": "fold_affine_into_weight_dequant_scale_and_bias",
        "selector_free": True,
    }
    runtime["runtime_selector"] = (
        runtime_selector.metadata()
        if runtime_selector is not None
        else {"enabled": False}
    )
    quantization_contract = {
        "logical_profile": (
            "quantvla_adapter_only_w4a8"
            if duquant_runtime.get("enabled") and adapter_only
            else ("gdsq_vla" if duquant_runtime.get("enabled") else "fp16")
        ),
        "quantization_method": (
            "real_quant_packed_w4_dequant_fp16_gemm"
            if duquant_runtime.get("packed_low_bit_residency")
            else ("duquant_fake_quant" if duquant_runtime.get("enabled") else "none")
        ),
        "layer_selection_policy": (
            "all_model_adapter_bound_target_linear_layers_uniform_w4"
            if adapter_only else "architecture_specific_gdsq_sensitivity_plan"
        ),
        "weight_quantizer": (
            "hessian_aware_gptq_feedback_signed_group64"
            if duquant_runtime.get("hessian_w4_loaded")
            else "signed_symmetric_per_output_channel"
        ),
        "activation_quantizer": "signed_symmetric_per_input_channel",
        "execution_backend": duquant_runtime.get("execution_backend"),
        "integer_gemm": bool(duquant_runtime.get("integer_gemm", False)),
        "packed_low_bit_residency": bool(
            duquant_runtime.get("packed_low_bit_residency", False)
        ),
        "packed_weight_bytes": int(duquant_runtime.get("packed_weight_bytes", 0)),
        "dequant_scale_bytes": int(duquant_runtime.get("dequant_scale_bytes", 0)),
        "input_gain_bytes": int(duquant_runtime.get("input_gain_bytes", 0)),
        "activation_scale_bytes": int(
            duquant_runtime.get("activation_scale_bytes", 0)
        ),
        "bias_bytes": int(duquant_runtime.get("bias_bytes", 0)),
        "auxiliary_static_bytes": int(
            duquant_runtime.get("auxiliary_static_bytes", 0)
        ),
        "fp_weight_sized_buffers": int(
            duquant_runtime.get("fp_weight_sized_buffers", 0)
        ),
        "weight_bits": int(duquant_runtime.get("weight_bits", 4)),
        "activation_bits": int(duquant_runtime.get("act_bits", 8)),
        "block_in": int(duquant_runtime.get("block_in", 64)),
        "block_out": int(duquant_runtime.get("block_out", 64)),
        "lambda_smooth": float(os.environ.get("OPENPI_DUQUANT_LS", "0.15")),
        "activation_percentile": float(duquant_runtime.get("act_percentile", 99.9)),
        "calibration_policy": (
            "offline_static_prefix_single_dit_per_flow_step"
            if duquant_runtime.get("hessian_w4_loaded")
            else "offline_static_per_channel_percentile"
        ),
        "calibration_batches": int(duquant_runtime.get("calib_batches", 32)),
        "calibration_batch_size": 8,
        "calibration_samples": int(duquant_runtime.get("calib_batches", 32)) * 8,
        "permutation": os.environ.get("OPENPI_DUQUANT_PERMUTE", "0") == "1",
        "row_rotation": os.environ.get("OPENPI_DUQUANT_ROW_ROT", "restore"),
        "static_activation_scales": True,
        "activation_scales_ready": bool(duquant_runtime.get("act_scales_ready", False)),
        "denoising_steps": int(denoising_steps),
        "n_action_steps": 16,
        "replan_steps": 16,
        "paired_noise": PAIRED_NOISE_PROTOCOL,
        "selector_loads_atm_and_ohb_superset": (
            runtime["runtime_selector"].get("enabled") is True
            and atm_runtime.get("atm_enabled") is True
            and atm_runtime.get("ohb_enabled") is True
        ),
        "correction_application": (
            "request_context_runtime"
            if runtime["runtime_selector"].get("enabled") is True
            else (
                "fold_affine_into_dequant_scale_and_bias"
                if duquant_runtime.get("errorfold_path")
                else "static_configuration"
            )
        ),
        "atm_application": atm_runtime.get("atm_application", "runtime_query"),
        "ohb_application": atm_runtime.get("ohb_application", "runtime_output"),
    }
    runtime["cross_model_quantization_contract"] = quantization_contract
    runtime["cross_model_quantization_contract_sha256"] = hashlib.sha256(
        json.dumps(
            quantization_contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    runtime["cross_model_protocol"] = protocol_attestation()
    runtime["model_adapter"] = adapter_attestation(
        "pi05", native_action_horizon=int(model.config.action_horizon)
    )
    runtime["protocol"] = closed_loop_runtime_protocol()
    if torch.cuda.is_available():
        device = next(model.parameters()).device
        runtime["gpu_memory_bytes"] = {
            "allocated": int(torch.cuda.memory_allocated(device)),
            "reserved": int(torch.cuda.memory_reserved(device)),
            "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
        }
    plan_path = Path(os.environ.get("OPENPI_DUQUANT_PLAN", "")).expanduser()
    if duquant_runtime.get("enabled") and adapter_only and plan_path.is_file():
        runtime["quantization_selection"] = validate_quant_plan(
            json.loads(plan_path.read_text(encoding="utf-8")),
            model="pi05",
            source=str(plan_path.resolve()),
        )
    native_rng_seed = os.environ.get("OPENPI_NATIVE_RNG_SEED")
    if native_rng_seed is not None:
        seed = int(native_rng_seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        runtime["native_rng_seed"] = seed
    _validate_formal_runtime(runtime)
    runtime_path = os.environ.get("OPENPI_RUNTIME_INFO_PATH")
    if runtime_path:
        path = Path(runtime_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(path) + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(policy._metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    return policy


def _validate_formal_runtime(runtime: dict) -> None:
    """Fail closed before a mislabeled formal server can accept requests."""
    if os.environ.get("OPENPI_FORMAL_MODE", "0") in ("0", "false", "False", ""):
        return

    config_id = runtime["config_id"]
    required_protocol = closed_loop_runtime_protocol()
    protocol = runtime.get("protocol") or {}
    mismatches = {
        key: (protocol.get(key), value)
        for key, value in required_protocol.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"formal cross-model protocol mismatch: {mismatches}")
    expected_checkpoint_sha256 = os.environ.get("OPENPI_CHECKPOINT_SHA256")
    if not expected_checkpoint_sha256:
        raise RuntimeError("formal runtime missing OPENPI_CHECKPOINT_SHA256")
    if runtime.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise RuntimeError(
            f"formal checkpoint hash mismatch: runtime={runtime.get('checkpoint_sha256')!r} "
            f"expected={expected_checkpoint_sha256!r}"
        )
    precision = runtime.get("model_dtype") or {}
    linear_dtypes = precision.get("linear_layers_by_weight_dtype") or {}
    if precision.get("resolved") != "float16":
        raise RuntimeError(f"formal runtime is not strict FP16: {precision}")
    if not linear_dtypes.get("float16") or any(
        count for dtype, count in linear_dtypes.items() if dtype != "float16"
    ):
        raise RuntimeError(f"formal runtime contains non-FP16 Linear weights: {linear_dtypes}")

    gdsq_wrapped_text = os.environ.get("OPENPI_FORMAL_EXPECT_WRAPPED")
    gdsq_wrapped = int(gdsq_wrapped_text) if gdsq_wrapped_text else None
    if os.environ.get("OPENPI_FORMAL_WEEK1_CONTROL", "0") not in (
        "0",
        "false",
        "False",
        "",
    ):
        if gdsq_wrapped is None:
            raise RuntimeError(
                f"{config_id}: OPENPI_FORMAL_EXPECT_WRAPPED is required for a week-1 control"
            )
        selector_runtime = runtime.get("runtime_selector") or {"enabled": False}
        if selector_runtime.get("enabled") is True:
            raise RuntimeError(f"{config_id}: selector must be disabled in a week-1 control")
        duquant = runtime.get("duquant") or {}
        required_duquant = {
            "enabled": True,
            "act_scales_ready": True,
            "block_in": 64,
            "block_out": 64,
            "act_bits": 8,
            "calib_batches": 32,
            "denoising_steps": 4,
            "wrapped_layers": gdsq_wrapped,
            "integer_gemm": False,
            "packed_low_bit_residency": False,
        }
        mismatches = {
            key: (duquant.get(key), value)
            for key, value in required_duquant.items()
            if duquant.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"{config_id}: week-1 DuQuant mismatch: {mismatches}")
        plan_path = Path(os.environ.get("OPENPI_DUQUANT_PLAN", "")).resolve()
        if not plan_path.is_file():
            raise RuntimeError(f"{config_id}: week-1 plan is missing: {plan_path}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        selected_bits = [
            int(row.get("bits", 0) or 0)
            for row in (plan.get("layers") or {}).values()
            if not bool(row.get("skip", not int(row.get("bits", 0) or 0)))
            and int(row.get("bits", 0) or 0) > 0
        ]
        if len(selected_bits) != gdsq_wrapped or not set(selected_bits) <= {4, 6}:
            raise RuntimeError(
                f"{config_id}: invalid week-1 plan bits/count: "
                f"wrapped={len(selected_bits)}, bits={sorted(set(selected_bits))}"
            )
        if Path(duquant.get("plan_path", "")).resolve() != plan_path:
            raise RuntimeError(f"{config_id}: runtime plan path mismatch")
        if not duquant.get("plan_sha256") or not duquant.get("act_scale_sha256"):
            raise RuntimeError(f"{config_id}: missing week-1 plan/A8 runtime hashes")
        atm = runtime.get("atm_ohb") or {}
        if atm.get("enabled") is True or atm.get("atm_enabled") is True or atm.get("ohb_enabled") is True:
            raise RuntimeError(f"{config_id}: correction must be disabled in a week-1 control")
        return

    expected = {
        "fp16": {"wrapped": 0, "enabled": False, "atm": False, "ohb": False},
        "quantvla_w4a8_atmohb": {"wrapped": 180, "enabled": True, "atm": True, "ohb": True},
        "quantvla_w4a8_paper": {"wrapped": 180, "enabled": True, "atm": True, "ohb": True},
        "errorfold_dfunc": {
            "wrapped": 180, "enabled": True, "atm": True, "ohb": True,
            "errorfold": True,
        },
        "errorfold_dpac_v2": {
            "wrapped": 180, "enabled": True, "atm": True, "ohb": True,
            "errorfold": True,
        },
        "quantvla_w4a8_softfold_dfunc": {"wrapped": 180, "enabled": True, "atm": True, "ohb": True},
        "quantvla_w4a8_softfold_dpac": {"wrapped": 180, "enabled": True, "atm": True, "ohb": True},
        "gdsq_vla_atmohb": {"wrapped": gdsq_wrapped, "enabled": True, "atm": True, "ohb": True},
        "gdsq_vla_atm_only": {"wrapped": gdsq_wrapped, "enabled": True, "atm": True, "ohb": False},
        "gdsq_vla_ohb_only": {"wrapped": gdsq_wrapped, "enabled": True, "atm": False, "ohb": True, "runtime_selector": False},
        "gdsq_vla_runtime_selector": {"wrapped": gdsq_wrapped, "enabled": True, "atm": True, "ohb": True, "runtime_selector": True},
        "gdsq_vla_softfold_dfunc": {"wrapped": gdsq_wrapped, "enabled": True, "atm": True, "ohb": True},
        "gdsq_vla_softfold_dpac": {"wrapped": gdsq_wrapped, "enabled": True, "atm": True, "ohb": True},
        "gdsq_vla": {"wrapped": gdsq_wrapped, "enabled": False, "atm": False, "ohb": False, "runtime_selector": False},
    }
    if config_id not in expected:
        raise RuntimeError(f"unknown formal config id: {config_id!r}")

    wanted = expected[config_id]
    selector_runtime = runtime.get("runtime_selector") or {"enabled": False}
    if bool(selector_runtime.get("enabled")) != bool(wanted.get("runtime_selector", False)):
        raise RuntimeError(f"{config_id}: runtime selector enablement mismatch: {selector_runtime}")
    if wanted.get("runtime_selector"):
        if selector_runtime.get("model_id") != "pi05":
            raise RuntimeError(f"{config_id}: runtime selector model mismatch: {selector_runtime}")
        if not selector_runtime.get("selector_sha256") or not selector_runtime.get("task_mapping_sha256"):
            raise RuntimeError(f"{config_id}: runtime selector missing hashes: {selector_runtime}")
    if wanted["wrapped"] is None:
        raise RuntimeError(
            f"{config_id}: OPENPI_FORMAL_EXPECT_WRAPPED must attest the frozen final plan"
        )
    duquant = runtime["duquant"]
    atm = runtime["atm_ohb"]
    if int(duquant.get("wrapped_layers", 0)) != wanted["wrapped"]:
        raise RuntimeError(
            f"{config_id}: wrapped_layers={duquant.get('wrapped_layers')} "
            f"!= {wanted['wrapped']}"
        )
    if wanted["wrapped"]:
        required = {
            "enabled": True,
            "act_scales_ready": True,
            "block_in": 64,
            "block_out": 64,
            "act_bits": 8,
            "weight_bits": 4,
            "calib_batches": 32,
            "denoising_steps": 4,
            "packed_low_bit_residency": True,
            "fp_weight_sized_buffers": 0,
        }
        for key, value in required.items():
            if duquant.get(key) != value:
                raise RuntimeError(
                    f"{config_id}: DuQuant runtime {key}={duquant.get(key)!r} != {value!r}"
                )
        if not duquant.get("plan_sha256") or not duquant.get("act_scale_sha256"):
            raise RuntimeError(f"{config_id}: missing plan/A8 runtime hashes")
        if int(duquant.get("packed_weight_bytes", 0)) <= 0:
            raise RuntimeError(f"{config_id}: packed W4 bytes were not materialized")
        if config_id.startswith("quantvla_") or config_id.startswith("errorfold_"):
            selection = runtime.get("quantization_selection") or {}
            if (
                selection.get("rule")
                != "all_model_adapter_bound_target_linear_layers_uniform_w4"
                or int(selection.get("quantized_w4_layers", -1)) != wanted["wrapped"]
                or int(selection.get("retained_fp16_target_layers", -1)) != 0
            ):
                raise RuntimeError(
                    f"{config_id}: adapter-only quantization selection mismatch: {selection}"
                )
    elif duquant.get("enabled"):
        raise RuntimeError("fp16: DuQuant was unexpectedly enabled")

    if bool(atm.get("enabled")) != wanted["enabled"]:
        raise RuntimeError(f"{config_id}: ATM/OHB enablement mismatch: {atm}")
    if wanted["enabled"]:
        required_atm = {
            "atm_enabled": wanted["atm"],
            "ohb_enabled": wanted["ohb"],
            "matched_layers": 18 if wanted["atm"] else 0,
            "ohb_layers": 18 if wanted["ohb"] else 0,
        }
        for key, value in required_atm.items():
            if atm.get(key) != value:
                raise RuntimeError(
                    f"{config_id}: ATM/OHB runtime {key}={atm.get(key)!r} != {value!r}"
                )
        metadata = atm.get("metadata") or {}
        if wanted.get("errorfold"):
            atm_required = {
                "flow_steps": 4,
                "folds": 8,
                "fit_pair": "original_fp16_vs_complete_quantized_network",
                "selector_free": True,
                "runtime_branch": False,
            }
            errorfold = runtime.get("errorfold") or {}
            if not errorfold.get("enabled") or not errorfold.get("selector_free"):
                raise RuntimeError(f"{config_id}: ErrorFold runtime is not frozen: {errorfold}")
            if int(duquant.get("hessian_w4_loaded", 0)) != wanted["wrapped"]:
                raise RuntimeError(f"{config_id}: Hessian W4 inventory is incomplete")
            contract = runtime.get("cross_model_quantization_contract") or {}
            if contract.get("row_rotation") != "0":
                raise RuntimeError(f"{config_id}: Hessian W4 requires identity row rotation")
            if selector_runtime.get("enabled") is True:
                raise RuntimeError(f"{config_id}: runtime selector is forbidden")
        else:
            atm_required = {
                "flow_steps": 4,
                "frames": 16,
                "batch_size": 8,
                "ohb_mode": "per_head_pre_projection",
            }
        atm_mismatches = {
            key: (metadata.get(key), value)
            for key, value in atm_required.items()
            if metadata.get(key) != value
        }
        if atm_mismatches:
            raise RuntimeError(
                f"{config_id}: ATM/OHB artifact is not GR00T-final aligned: {atm_mismatches}"
            )


def main(args: Args) -> None:
    policy = create_policy(args)
    apply_quantization(policy, denoising_steps=args.denoising_steps)

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
