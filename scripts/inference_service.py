# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GR00T Inference Service

This script provides both ZMQ and HTTP server/client implementations for deploying GR00T models.
The HTTP server exposes a REST API for easy integration with web applications and other services.

1. Default is zmq server.

Run server: python scripts/inference_service.py --server
Run client: python scripts/inference_service.py --client

2. Run as Http Server:

Dependencies for `http_server` mode:
    => Server (runs GR00T model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

HTTP Server Usage:
    python scripts/inference_service.py --server --http-server --port 8000

HTTP Client Usage (assuming a server running on 0.0.0.0:8000):
    python scripts/inference_service.py --client --http-server --host 0.0.0.0 --port 8000

You can use bore to forward the port to your client: `159.223.171.199` is bore.pub.
    bore local 8000 --to 159.223.171.199
"""

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Literal

from pathlib import Path

import numpy as np
import torch
import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))
from quantvla_cross_model_protocol import (  # noqa: E402
    adapter_attestation,
    closed_loop_runtime_protocol,
    protocol_attestation,
    validate_quant_plan,
)
from quantvla_dynamic_a8_protocol import (  # noqa: E402
    protocol_attestation as dynamic_a8_protocol_attestation,
    validate_runtime as validate_dynamic_a8_runtime,
)


@dataclass
class ArgsConfig:
    """Command line arguments for the inference service."""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """The embodiment tag for the model."""

    data_config: str = "fourier_gr1_arms_waist"
    """
    The name of the data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.

    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See gr00t/experiment/data_config.py for more details.
    """

    port: int = 5555
    """The port number for the server."""

    host: str = "localhost"
    """The host address for the server."""

    server: bool = False
    """Whether to run the server."""

    client: bool = False
    """Whether to run the client."""

    denoising_steps: int = 4
    """The number of denoising steps to use."""

    api_token: str = None
    """API token for authentication. If not provided, authentication is disabled."""

    http_server: bool = False
    """Whether to run it as HTTP server. Default is ZMQ server."""


#####################################################################################


def _example_zmq_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example ZMQ client call to the server.
    """
    # Original ZMQ client mode
    # Create a policy wrapper
    policy_client = RobotInferenceClient(host=host, port=port, api_token=api_token)

    print("Available modality config available:")
    modality_configs = policy_client.get_modality_config()
    print(modality_configs.keys())

    time_start = time.time()
    action = policy_client.get_action(obs)
    print(f"Total time taken to get action from server: {time.time() - time_start} seconds")
    return action


def _example_http_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example HTTP client call to the server.
    """
    import json_numpy

    json_numpy.patch()
    import requests

    # Send request to HTTP server
    print("Testing HTTP server...")

    time_start = time.time()
    response = requests.post(f"http://{host}:{port}/act", json={"observation": obs})
    print(f"Total time taken to get action from HTTP server: {time.time() - time_start} seconds")

    if response.status_code == 200:
        action = response.json()
        return action
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}


def _maybe_close_a8_calibration(
    policy, data_config: str = "", model_path: str = ""
) -> None:
    """Static-A8 calibration closure before serving (review round 2, item 3).

    No-op for FP16 models (no GR00T_DUQUANT_* env) and for dynamic-act models
    (no static calibrators). For static-A8 quantized models: load persisted
    frozen scales when GR00T_DUQUANT_ACT_SCALE_PATH exists, otherwise run the
    fixed synthetic buffer (calib_steps × batch_size obs, seed 0) and save.
    """
    if not any(
        k.startswith("GR00T_DUQUANT_") and k != "GR00T_DUQUANT_PACKDIR"
        for k in os.environ
    ):
        return
    from pathlib import Path as _Path

    _here = _Path(__file__).resolve().parents[1]
    if str(_here / "scripts" / "tools") not in sys.path:
        sys.path.insert(0, str(_here / "scripts" / "tools"))

    from gr00t.quantization.duquant_layers import static_calibrators_required
    from gr00t_v2_common import (
        count_wrapped_layers,
        ensure_a8_calibrated,
        fixed_calibration_buffer,
    )
    from quantvla_cross_model_protocol import protocol_artifact
    from quantvla_model_adapters import gr00t_rollout_inputs, load_model_records

    act_dynamic = os.environ.get("GR00T_DUQUANT_ACT_DYNAMIC", "0") not in ("0", "false", "False")
    if act_dynamic or not static_calibrators_required(policy.model):
        return
    scale_path = os.environ.get("GR00T_DUQUANT_ACT_SCALE_PATH") or None
    calib_steps = int(os.environ.get("GR00T_DUQUANT_CALIB_STEPS", "32"))
    batch_size = 8
    horizon = int(policy.model.action_head.config.action_horizon)
    action_dim = int(policy.model.action_head.config.action_dim)
    # review round 3: self-contained seed-based buffer — identical data every
    # start, so the sha256 + sidecar prove the scales match the experiment.
    # v1.4 Stage D: GR00T_OBS_FORMAT=robocasa365 for the RoboCasa365 checkpoints
    fmt = os.environ.get("GR00T_OBS_FORMAT", "libero")
    source_buffer_path = None
    if fmt == "robocasa365":
        source_buffer_path = protocol_artifact("calibration_buffer")
        records, provenance = load_model_records(
            source_buffer_path, calib_steps * batch_size, model="gr00t"
        )
        warm_obs, warm_noises = gr00t_rollout_inputs(records)
        sha = provenance["sha256"]
    else:
        warm_obs, warm_noises, sha = fixed_calibration_buffer(
            0, calib_steps * batch_size, horizon, action_dim, fmt=fmt
        )
    import hashlib as _hl

    plan_path = os.environ.get("GR00T_DUQUANT_PLAN")
    plan_sha = None
    if plan_path and os.path.exists(plan_path):
        with open(plan_path, "rb") as f:
            plan_sha = _hl.sha256(f.read()).hexdigest()
    checkpoint_path = str(_Path(model_path).resolve()) if model_path else None
    act_meta = {
        "buffer_sha256": sha,
        "calibration_seed": 0,
        "data_config": data_config,
        "obs_format": fmt,
        "act_percentile": float(os.environ.get("GR00T_DUQUANT_ACT_PCT", "99.9")),
        "calib_batches": calib_steps,
        "denoising_steps": int(
            os.environ.get("GR00T_DENOISING_STEPS", str(policy.denoising_steps))
        ),
        "plan_sha256": plan_sha,
        "checkpoint_path": checkpoint_path,
        "wrapped_layers": count_wrapped_layers(policy.model),
    }
    if source_buffer_path is not None:
        act_meta.update(
            {
                "source_buffer_sha256": sha,
                "source_buffer_path": str(source_buffer_path),
            }
        )
    if os.environ.get("GR00T_DUQUANT_HESSIAN_W4_PATH"):
        # The v3 A8 artifact has one prefix table and four deterministic DiT
        # tables and therefore uses the shared schema, not the legacy
        # seed/path sidecar.  Requiring legacy-only fields here would reject a
        # valid v3 artifact after the expensive server model load.
        from quantvla_cross_model_protocol import PROTOCOL_SHA256

        act_meta = {
            "schema_version": 3,
            "kind": "v3_per_flow_step_a8",
            "protocol_sha256": PROTOCOL_SHA256,
            "plan_sha256": plan_sha,
            "calibration_buffer_sha256": sha,
            "source_buffer_sha256": sha,
            "wrapped_layers": count_wrapped_layers(policy.model),
            "act_percentile": float(
                os.environ.get("GR00T_DUQUANT_ACT_PCT", "99.9")
            ),
            "calib_batches": calib_steps,
            "denoising_steps": int(
                os.environ.get(
                    "GR00T_DENOISING_STEPS", str(policy.denoising_steps)
                )
            ),
            "prefix_llm_tables": 1,
            "dit_flow_step_tables": 4,
        }
    print(f"[inference] static A8 calibration warmup: {calib_steps * batch_size} "
          f"shared obs (buffer sha256 {sha[:16]}...; plan {plan_sha})", flush=True)
    t0 = time.time()
    ensure_a8_calibrated(
        policy, warm_obs, warm_noises, batch_size,
        act_dynamic=False, act_scale_path=scale_path, act_scale_meta=act_meta,
    )
    print(f"[inference] static A8 calibration complete in {time.time() - t0:.1f}s", flush=True)


def _maybe_close_omega_qvla_calibration(policy) -> None:
    """Freeze Omega-QVLA LLM A4 scales before any evaluation request.

    Released DiT records carry offline per-step tables, while released LLM
    records initialize their activation scale on the first forward.  Drive
    that forward with the exact preregistered RoboCasa365 observation instead
    of allowing the first test episode to mutate model state.
    """
    if os.environ.get("GR00T_GPTQ", "0") in ("0", "false", "False"):
        return
    task_set = os.environ.get("OMEGA_QVLA_TASK_SET", "")
    expected_sha = os.environ.get("OMEGA_QVLA_CALIBRATION_SHA256", "")
    if not task_set or not expected_sha:
        raise RuntimeError("Omega-QVLA requires frozen task-set calibration attestation")

    from pathlib import Path as _Path

    here = _Path(__file__).resolve().parents[1]
    tools_path = str(here / "scripts/tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    from omega_qvla_robocasa365_build import frozen_samples
    from gr00t.quantization.gptq_layers import GptqLinear

    samples, actual_sha = frozen_samples(task_set)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Omega-QVLA calibration hash drift: {actual_sha} != {expected_sha}"
        )
    started = time.time()
    policy.get_action(samples[0]["obs"])
    layers = [module for module in policy.model.modules() if isinstance(module, GptqLinear)]
    incomplete = [
        layer.name for layer in layers
        if not layer._has_act_scale_table and not layer._act_scale_initialized
    ]
    if not layers or incomplete:
        raise RuntimeError(
            f"Omega-QVLA activation warmup incomplete: layers={len(layers)} "
            f"missing={incomplete[:3]}"
        )
    policy.model._omega_qvla_calibration = {
        "task_set": task_set,
        "buffer_sha256": actual_sha,
        "warmup_observations": 1,
        "wrapped_layers": len(layers),
    }
    print(
        f"[inference] Omega-QVLA frozen A4 warmup complete: task_set={task_set} "
        f"layers={len(layers)} sha256={actual_sha[:16]}... "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )


def _extract_eval_metadata(observations: dict) -> tuple[dict, dict[str, Any] | None]:
    """Remove evaluator-only metadata before policy transforms see observations."""
    if not isinstance(observations, dict):
        return observations, None
    clean = dict(observations)
    metadata = clean.pop("eval_metadata", None)
    if metadata is not None and not isinstance(metadata, dict):
        metadata = {"task_name": str(metadata)}
    return clean, metadata


def _attach_runtime_selector(action: dict, decision) -> dict:
    if decision is None:
        return action
    out = dict(action)
    out["runtime_selector"] = decision.response_payload()
    return out


def _selected_action_handler(policy, observations: dict) -> dict:
    from gr00t.atm import runtime_selector_context

    clean_observations, metadata = _extract_eval_metadata(observations)
    with runtime_selector_context(metadata) as decision:
        action = policy.get_action(clean_observations)
    return _attach_runtime_selector(action, decision)


def _seeded_action_handler(policy, payload: dict) -> dict:
    """Evaluate one observation with deterministic, request-local FM noise.

    The seed is derived by the RoboCasa client from
    (task, environment seed, replan index).  Creating the noise with a local
    CPU generator makes requests independent of server RNG state and client
    interleaving while preserving the standard-normal distribution.
    """
    if not isinstance(payload, dict) or "observations" not in payload:
        raise ValueError("get_action_seeded requires {observations, action_seed}")
    if "action_seed" not in payload:
        raise ValueError("get_action_seeded payload is missing action_seed")

    import torch
    from gr00t.atm import runtime_selector_context

    seed = int(payload["action_seed"])
    if seed < 0 or seed >= 2**63:
        raise ValueError(f"action_seed must be in [0, 2**63), got {seed}")
    horizon = int(policy.model.action_head.config.action_horizon)
    action_dim = int(policy.model.action_head.config.action_dim)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    action_noise = torch.randn(
        (horizon, action_dim), generator=generator, dtype=torch.float32
    )
    clean_observations, metadata = _extract_eval_metadata(payload["observations"])
    with runtime_selector_context(metadata) as decision:
        action = policy.get_action(clean_observations, action_noise=action_noise)
    return _attach_runtime_selector(action, decision)


def _runtime_info(policy) -> dict:
    from pathlib import Path

    from gr00t.atm import get_runtime_selector
    from gr00t.quantization.duquant_layers import DuQuantLinear, static_scales_ready
    from gr00t_v2_common import count_wrapped_layers

    def uniform_value(layers, field: str):
        values = {getattr(layer.cfg, field) for layer in layers}
        if len(values) != 1:
            raise RuntimeError(f"GR00T aligned quantization requires uniform {field}: {sorted(map(str, values))}")
        return next(iter(values))

    def sha256_path(value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    modules = list(policy.model.modules())
    quant_layers = [module for module in modules if isinstance(module, DuQuantLinear)]
    try:
        from gr00t.quantization.gptq_layers import GptqLinear

        gptq_layers = [module for module in modules if isinstance(module, GptqLinear)]
    except ImportError:
        gptq_layers = []
    if quant_layers and gptq_layers:
        raise RuntimeError("RoboCasa server cannot mix GDSQ DuQuant and Omega-QVLA GPTQ")
    runtime_selector = get_runtime_selector()
    selector_metadata = (
        runtime_selector.metadata()
        if runtime_selector is not None
        else {"enabled": False, "kind": "atmohb_runtime_selector"}
    )
    atm_runtime = getattr(policy.model, "_gr00t_atm_runtime", {"enabled": False})
    plan_path = os.environ.get("GR00T_DUQUANT_PLAN")
    act_scale_path = os.environ.get("GR00T_DUQUANT_ACT_SCALE_PATH")
    hessian_runtime = getattr(policy.model, "_gr00t_duquant_runtime", {})
    if gptq_layers:
        weight_bits = {int(layer.weight_bits) for layer in gptq_layers}
        activation_bits = {
            int(layer._record_a_bits or layer.cfg.act_bits or 0)
            for layer in gptq_layers
        }
        if weight_bits != {4} or activation_bits != {4}:
            raise RuntimeError(
                f"Omega-QVLA requires W4A4, got W{sorted(weight_bits)} "
                f"A{sorted(activation_bits)}"
            )
        unavailable = [layer.name for layer in gptq_layers if not layer._quant_available]
        if unavailable:
            raise RuntimeError(f"Omega-QVLA pack has fallback layers: {unavailable[:3]}")
        quantization_contract = {
            "logical_profile": "omega_qvla_w4a4",
            "quantization_method": "omega_qvla_offline_gptq",
            "upstream_commit": os.environ.get("OMEGA_QVLA_COMMIT"),
            "rotation": "svd_hadamard",
            "llm_quantizer": "gptq",
            "dit_quantizer": "rtn_residual_per_step",
            "weight_bits": 4,
            "activation_bits": 4,
            "llm_activation_scale_policy": "frozen_preregistered_first_observation",
            "calibration_attestation": getattr(
                policy.model, "_omega_qvla_calibration", None
            ),
            "denoising_steps": int(policy.denoising_steps),
            "n_action_steps": 16,
            "replan_steps": 16,
            "paired_noise": "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1",
            "integer_gemm": False,
            "claim_scope": "algorithmic W4A4; fake-quantized FP matmul runtime",
        }
    elif quant_layers:
        plan_target_layers = 0
        if plan_path:
            plan_target_layers = len(
                (json.loads(Path(plan_path).read_text(encoding="utf-8")).get("layers") or {})
            )
        mixed_static_profile = bool(
            plan_target_layers and len(quant_layers) < plan_target_layers
        )
        weight_bits = {int(layer.weight_bits) for layer in quant_layers}
        if len(weight_bits) != 1:
            raise RuntimeError(
                f"GR00T aligned quantization requires uniform weight bits: {sorted(weight_bits)}"
            )
        fused_values = {bool(layer.cfg.use_fused) for layer in quant_layers}
        if len(fused_values) != 1:
            raise RuntimeError("GR00T aligned quantization cannot mix eager and fused execution")
        fused = next(iter(fused_values))
        residency = getattr(policy.model, "_quantvla_real_quant_residency", {})
        packed_residency = bool(
            fused
            and residency.get("packed_low_bit_residency")
            and all(layer._inference_only_ready for layer in quant_layers)
        )
        quantization_contract = {
            "logical_profile": (
                "quantvla_adapter_only_w4a8"
                if os.environ.get("QUANTVLA_ADAPTER_ONLY", "0")
                not in ("0", "false", "False", "")
                else "gdsq_vla"
            ),
            "quantization_method": (
                "real_quant_packed_w4_dequant_fp16_gemm"
                if packed_residency else "duquant_fake_quant"
            ),
            "layer_selection_policy": (
                "shared_static_compression_profile_over_model_adapter_bound_layers"
                if mixed_static_profile
                else "all_model_adapter_bound_target_linear_layers_uniform_w4"
                if os.environ.get("QUANTVLA_ADAPTER_ONLY", "0")
                not in ("0", "false", "False", "")
                else "architecture_specific_gdsq_sensitivity_plan"
            ),
            "weight_quantizer": (
                "hessian_aware_gptq_feedback_signed_group64"
                if hessian_runtime.get("hessian_w4_loaded")
                else "signed_symmetric_per_output_channel"
            ),
            "activation_quantizer": "signed_symmetric_per_input_channel",
            "execution_backend": (
                "triton_w4_nibble_dequant_fp16_gemm"
                if fused else "fake_quant_fp16_gemm"
            ),
            "integer_gemm": False,
            "packed_low_bit_residency": packed_residency,
            "packed_weight_bytes": int(residency.get("packed_weight_bytes", 0)),
            "dequant_scale_bytes": int(residency.get("dequant_scale_bytes", 0)),
            "input_gain_bytes": int(residency.get("input_gain_bytes", 0)),
            "activation_scale_bytes": int(
                residency.get("activation_scale_bytes", 0)
            ),
            "bias_bytes": int(residency.get("bias_bytes", 0)),
            "auxiliary_static_bytes": int(
                residency.get("auxiliary_static_bytes", 0)
            ),
            "fp_weight_sized_buffers": int(
                residency.get("fp_weight_sized_buffers", len(quant_layers))
            ),
            "weight_bits": next(iter(weight_bits)),
            "activation_bits": int(uniform_value(quant_layers, "act_bits")),
            "block_in": int(uniform_value(quant_layers, "block_size")),
            "block_out": int(uniform_value(quant_layers, "block_out_size")),
            "lambda_smooth": float(uniform_value(quant_layers, "lambda_smooth")),
            "activation_percentile": float(uniform_value(quant_layers, "act_percentile")),
            "calibration_policy": (
                "online_dynamic_per_forward_per_channel_amax"
                if bool(uniform_value(quant_layers, "act_dynamic"))
                else (
                    "offline_static_prefix_single_dit_per_flow_step"
                    if hessian_runtime.get("hessian_w4_loaded")
                    else "offline_static_per_channel_percentile"
                )
            ),
            "calibration_batches": int(uniform_value(quant_layers, "calib_batches")),
            "calibration_batch_size": 8,
            "calibration_samples": int(uniform_value(quant_layers, "calib_batches")) * 8,
            "permutation": bool(uniform_value(quant_layers, "enable_permute")),
            "row_rotation": str(uniform_value(quant_layers, "row_rot_mode")),
            "static_activation_scales": not bool(uniform_value(quant_layers, "act_dynamic")),
            "activation_scales_ready": bool(static_scales_ready(policy.model)),
            "denoising_steps": int(policy.denoising_steps),
            "n_action_steps": 16,
            "replan_steps": 16,
            "paired_noise": "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1",
            "selector_loads_atm_and_ohb_superset": (
                selector_metadata.get("enabled") is True
                and atm_runtime.get("atm_enabled") is True
                and atm_runtime.get("ohb_enabled") is True
            ),
            "correction_application": (
                "request_context_runtime"
                if selector_metadata.get("enabled") is True
                else (
                    "fold_affine_into_dequant_scale_and_bias"
                    if hessian_runtime.get("errorfold_path")
                    else "static_configuration"
                )
            ),
            "atm_application": atm_runtime.get("atm_application", "runtime_query"),
            "ohb_application": atm_runtime.get("ohb_application", "runtime_output"),
        }
    else:
        quantization_contract = {
            "logical_profile": "fp16",
            "quantization_method": "none",
        }
    contract_sha256 = hashlib.sha256(
        json.dumps(quantization_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "config_id": os.environ.get("GR00T_CONFIG_ID", "unspecified"),
        "wrapped_layers": len(gptq_layers) if gptq_layers else count_wrapped_layers(policy.model),
        "model_path": str(policy.model_path.resolve()),
        "plan": plan_path,
        "plan_sha256": sha256_path(plan_path),
        "packdir": os.environ.get("GR00T_DUQUANT_PACKDIR"),
        "omega_pack_path": os.environ.get("GR00T_GPTQ_PATH"),
        "omega_pack_sha256": sha256_path(os.environ.get("GR00T_GPTQ_PATH")),
        "omega_calibration": getattr(policy.model, "_omega_qvla_calibration", None),
        "act_scale_path": act_scale_path,
        "act_scale_sha256": sha256_path(act_scale_path),
        "hessian_w4_path": hessian_runtime.get("hessian_w4_path"),
        "hessian_w4_sha256": hessian_runtime.get("hessian_w4_sha256"),
        "hessian_group_size": hessian_runtime.get("hessian_group_size"),
        "errorfold_path": hessian_runtime.get("errorfold_path"),
        "errorfold_sha256": sha256_path(hessian_runtime.get("errorfold_path")),
        "atm_enabled": atm_runtime.get("atm_enabled", False),
        "ohb_enabled": atm_runtime.get("ohb_enabled", False),
        "atm_path": atm_runtime.get("artifact_path") or os.environ.get("GR00T_ATM_ALPHA_PATH"),
        "atm_artifact_sha256": sha256_path(
            atm_runtime.get("artifact_path") or os.environ.get("GR00T_ATM_ALPHA_PATH")
        ),
        "atm_layers": int(atm_runtime.get("matched_layers", 0)),
        "ohb_layers": int(atm_runtime.get("ohb_layers", 0)),
        "denoising_steps": int(policy.denoising_steps),
        "quantization_contract": quantization_contract,
        "quantization_contract_sha256": contract_sha256,
        "runtime_selector": selector_metadata,
        "cross_model_protocol": protocol_attestation(),
        "model_adapter": adapter_attestation(
            "gr00t",
            native_action_horizon=int(policy.model.action_head.config.action_horizon),
        ),
        "protocol": closed_loop_runtime_protocol(),
    }
    if quant_layers and not quantization_contract.get("static_activation_scales", True):
        validate_dynamic_a8_runtime(
            quantization_contract, source="GR00T quantization contract"
        )
        payload["dynamic_a8_protocol"] = dynamic_a8_protocol_attestation()
    if torch.cuda.is_available():
        device = next(policy.model.parameters()).device
        payload["gpu_memory_bytes"] = {
            "allocated": int(torch.cuda.memory_allocated(device)),
            "reserved": int(torch.cuda.memory_reserved(device)),
            "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
        }
    if quant_layers and os.environ.get("QUANTVLA_ADAPTER_ONLY", "0") not in (
        "0", "false", "False", ""
    ):
        if not plan_path:
            raise RuntimeError("adapter-only quantized runtime requires a plan")
        payload["quantization_selection"] = validate_quant_plan(
            json.loads(Path(plan_path).read_text(encoding="utf-8")),
            model="gr00t",
            source=str(Path(plan_path).resolve()),
        )
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload["metadata_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def main(args: ArgsConfig):
    if args.server:
        # Create a policy
        # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
        # the model path, transform name, embodiment tag, and denoising steps for the robot
        # inference system. This policy object is then utilized in the server mode to start
        # the Robot Inference Server for making predictions based on the specified model and
        # configuration.

        # we will use an existing data config to create the modality config and transform
        # if a new data config is specified, this expect user to
        # construct your own modality config and transform
        # see gr00t/utils/data.py for more details
        data_config = load_data_config(args.data_config)
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )

        # Review round 2, item 3: close the static-A8 calibration loop BEFORE
        # the server accepts any request. Without this, the first LIBERO
        # get_action calls would online-calibrate the frozen scales on TEST
        # observations (model state changing during evaluation). A fixed
        # synthetic buffer (same sha256 across starts) completes the
        # calibration; GR00T_DUQUANT_ACT_SCALE_PATH persists it across runs.
        _maybe_close_a8_calibration(
            policy, data_config=args.data_config, model_path=args.model_path
        )
        if os.environ.get("QUANTVLA_ADAPTER_ONLY", "0") not in (
            "0", "false", "False", ""
        ) and os.environ.get("GR00T_DUQUANT_PLAN"):
            from gr00t.quantization import finalize_real_quant

            residency = finalize_real_quant(policy.model)
            setattr(policy.model, "_quantvla_real_quant_residency", residency)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[inference] real-quant finalized: {residency}", flush=True)
        _maybe_close_omega_qvla_calibration(policy)

        from gr00t.atm import configure_runtime_selector_from_env

        runtime_selector = configure_runtime_selector_from_env()
        if runtime_selector is not None:
            print(
                "[inference] GR00T runtime ATM/OHB selector enabled: "
                f"{runtime_selector.selector_path} model={runtime_selector.model_id}",
                flush=True,
            )

        # Start the server
        if args.http_server:
            from gr00t.eval.http_server import HTTPInferenceServer  # noqa: F401

            server = HTTPInferenceServer(
                policy, port=args.port, host=args.host, api_token=args.api_token
            )
            server.run()
        else:
            server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token)
            server.register_endpoint(
                "get_action",
                lambda observations: _selected_action_handler(policy, observations),
            )
            server.register_endpoint(
                "get_action_seeded",
                lambda payload: _seeded_action_handler(policy, payload),
            )
            server.register_endpoint(
                "get_runtime_info", lambda: _runtime_info(policy), requires_input=False
            )
            server.run()

    # Here is mainly a testing code
    elif args.client:
        # In this mode, we will send a random observation to the server and get an action back
        # This is useful for testing the server and client connection

        # Making prediction...
        # - obs: video.ego_view: (1, 256, 256, 3)
        # - obs: state.left_arm: (1, 7)
        # - obs: state.right_arm: (1, 7)
        # - obs: state.left_hand: (1, 6)
        # - obs: state.right_hand: (1, 6)
        # - obs: state.waist: (1, 3)

        # - action: action.left_arm: (16, 7)
        # - action: action.right_arm: (16, 7)
        # - action: action.left_hand: (16, 6)
        # - action: action.right_hand: (16, 6)
        # - action: action.waist: (16, 3)
        obs = {
            "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
            "state.left_arm": np.random.rand(1, 7),
            "state.right_arm": np.random.rand(1, 7),
            "state.left_hand": np.random.rand(1, 6),
            "state.right_hand": np.random.rand(1, 6),
            "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["do your thing!"],
        }

        if args.http_server:
            action = _example_http_client_call(obs, args.host, args.port, args.api_token)
        else:
            action = _example_zmq_client_call(obs, args.host, args.port, args.api_token)

        for key, value in action.items():
            print(f"Action: {key}: {value.shape}")
    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    # SIGUSR1 dumps all python thread stacks to the log — the runtime watchdog
    # uses it to diagnose hung servers (review round 5, item 3)
    import faulthandler
    import signal as _signal

    faulthandler.register(_signal.SIGUSR1)

    config = tyro.cli(ArgsConfig)
    main(config)
