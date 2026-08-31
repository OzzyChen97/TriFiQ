import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


_PYTORCH_PRECISIONS = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "float32": "float32",
    "fp32": "float32",
}


def _configure_pytorch_precision(model, requested: str) -> dict[str, Any]:
    """Configure and describe the actual PyTorch inference precision."""
    import torch

    normalized = _PYTORCH_PRECISIONS.get(requested.strip().lower())
    if normalized is None:
        raise ValueError(
            f"Unsupported OPENPI_MODEL_DTYPE={requested!r}; expected one of "
            f"{sorted(_PYTORCH_PRECISIONS)}"
        )

    # Preserve upstream's mixed BF16/FP32 behavior by default.  Formal FP16
    # explicitly converts all GEMM modules, after which the PaliGemma helper
    # restores known numerically sensitive embeddings/norms to FP32.
    qvla_uniform_bf16 = False
    if normalized == "bfloat16":
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        # QVLA's frozen protocol requires every excluded/projector/action
        # module to remain BF16.  Preserve OpenPI's historical FP32 islands
        # for all ordinary callers, but remove them in the explicitly scoped
        # QVLA calibration and formal-runtime environments.
        qvla_uniform_bf16 = (
            os.environ.get("QVLA_ACTQUANT_METHOD", "").strip().lower() == "qvla"
            or os.environ.get("QVLA_ACTQUANT_CALIBRATION_DTYPE", "").strip().lower()
            in {"bf16", "bfloat16"}
        )
        if qvla_uniform_bf16:
            model.to(dtype=torch.bfloat16)
    elif normalized == "float16":
        model.to(dtype=torch.float16)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("float16")
    else:
        model.to(dtype=torch.float32)

    parameter_dtypes: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype).removeprefix("torch.")
        parameter_dtypes[key] = parameter_dtypes.get(key, 0) + parameter.numel()

    linear_dtypes: dict[str, int] = {}
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            key = str(module.weight.dtype).removeprefix("torch.")
            linear_dtypes[key] = linear_dtypes.get(key, 0) + 1

    return {
        "requested": requested,
        "resolved": normalized,
        "parameter_elements_by_dtype": parameter_dtypes,
        "linear_layers_by_weight_dtype": linear_dtypes,
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "qvla_uniform_bf16": qvla_uniform_bf16,
    }


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        requested_dtype = os.environ.get("OPENPI_MODEL_DTYPE", "bfloat16")
        precision_metadata = _configure_pytorch_precision(model, requested_dtype)
        logging.info("PyTorch inference precision: %s", precision_metadata)
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    policy_metadata = dict(train_config.policy_metadata or {})
    if is_pytorch:
        policy_metadata["openpi_runtime"] = {
            "checkpoint_dir": str(pathlib.Path(checkpoint_dir).resolve()),
            "model_dtype": precision_metadata,
            "pytorch_device": pytorch_device,
            "pytorch_compile_mode": train_config.model.pytorch_compile_mode,
        }

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
