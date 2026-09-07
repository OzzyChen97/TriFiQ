"""Model-native calibration adapters for RoboCasa365 teacher archives."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GR00T_ACTION_SLICES = (
    ("action.end_effector_position", 0, 3),
    ("action.end_effector_rotation", 3, 6),
    ("action.gripper_close", 6, 7),
    ("action.base_motion", 7, 11),
    ("action.control_mode", 11, 12),
)


def _ensure_paths() -> None:
    paths = (
        REPO_ROOT / "code",
        REPO_ROOT / "scripts",
        REPO_ROOT / "scripts" / "tools",
        REPO_ROOT / "code" / "pi05" / "openpi" / "src",
        REPO_ROOT / "code" / "pi05" / "openpi" / "packages" / "openpi-client" / "src",
    )
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_gr00t_policy(checkpoint: str | Path, device: str):
    _ensure_paths()
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    data_config = load_data_config("examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig")
    return Gr00tPolicy(
        model_path=str(Path(checkpoint).resolve()),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=4,
        device=device,
    )


def load_pi05_policy(checkpoint: str | Path, device: str):
    _ensure_paths()
    from openpi.policies import policy_config
    from openpi.training import config

    checkpoint = Path(checkpoint).resolve()
    if checkpoint.is_file():
        checkpoint = checkpoint.parent
    return policy_config.create_trained_policy(
        config.get_config("pi05_pretrain_human300"),
        str(checkpoint),
        pytorch_device=device,
    )


def _resize_uint8(image: np.ndarray, size: int) -> np.ndarray:
    value = np.asarray(image)
    if value.shape == (size, size, 3):
        return value
    tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1)[None].float()
    value = torch.nn.functional.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False, antialias=True
    ).round().clamp_(0, 255).to(torch.uint8)[0].permute(1, 2, 0).numpy()
    return value


def gr00t_raw_record(frame: dict[str, Any], *, include_actions: bool) -> dict[str, Any]:
    state = np.asarray(frame["state"], dtype=np.float32)
    result: dict[str, Any] = {
        "video.robot0_agentview_left": _resize_uint8(frame["image"], 256)[None],
        "video.robot0_agentview_right": _resize_uint8(frame["right_image"], 256)[None],
        "video.robot0_eye_in_hand": _resize_uint8(frame["wrist_image"], 256)[None],
        "state.end_effector_position_relative": state[0:3][None],
        "state.end_effector_rotation_relative": state[3:7][None],
        "state.gripper_qpos": state[14:16][None],
        "state.base_position": state[7:10][None],
        "state.base_rotation": state[10:14][None],
        "annotation.human.task_description": [str(frame["prompt"])],
    }
    if include_actions:
        actions = np.asarray(frame["actions"], dtype=np.float32)
        if actions.shape != (16, 12):
            raise ValueError(f"GR00T teacher action shape drift: {actions.shape}")
        for key, start, stop in GR00T_ACTION_SLICES:
            result[key] = actions[:, start:stop]
    return result


def _stack_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if isinstance(values[0], np.ndarray):
            result[key] = np.stack(values)
        else:
            result[key] = np.asarray(
                [value[0] if isinstance(value, (list, tuple)) else value for value in values]
            )
    return result


def prepare_gr00t_batch(policy, frames: Sequence[dict[str, Any]], *, include_actions: bool):
    raw = _stack_records([gr00t_raw_record(frame, include_actions=include_actions) for frame in frames])
    # Gr00tPolicy keeps its modality transform in inference mode, where the
    # official transform intentionally drops action/action_mask.  Calibration
    # losses need those fields even though the policy model itself stays in
    # eval mode.  Switch only the data transform for this deterministic call
    # and always restore inference behavior before returning.
    if include_actions:
        policy.modality_transform.train()
    try:
        return policy.apply_transforms(raw)
    finally:
        if include_actions:
            policy.modality_transform.eval()


def pi05_raw_record(frame: dict[str, Any], *, include_actions: bool) -> dict[str, Any]:
    result = {
        "observation/image": np.asarray(frame["image"]),
        "observation/wrist_image": np.asarray(frame["wrist_image"]),
        "observation/right_image": np.asarray(frame["right_image"]),
        "observation/state": np.asarray(frame["state"], dtype=np.float32),
        "prompt": str(frame["prompt"]),
    }
    if include_actions:
        actions = np.asarray(frame["actions"], dtype=np.float32)
        if actions.shape != (50, 12):
            raise ValueError(f"pi0.5 teacher action shape drift: {actions.shape}")
        result["actions"] = actions
    return result


def _recursive_stack(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        if any(set(value) != set(first) for value in values):
            raise ValueError("transformed tree key mismatch")
        return {key: _recursive_stack([value[key] for value in values]) for key in first}
    return np.stack([np.asarray(value) for value in values])


def _recursive_torch(value: Any, device: str) -> Any:
    if isinstance(value, dict):
        return {key: _recursive_torch(child, device) for key, child in value.items()}
    array = np.asarray(value)
    # Normalization statistics may be stored as float64 and NumPy promotes
    # otherwise-float32 state/actions during arithmetic.  OpenPI's model
    # contract requires float32 observations and teacher actions; leaving the
    # promotion in place makes its FP32 loss backpropagate through an FP64
    # target and fails with "Found dtype Double but expected Float".
    if np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize > 4:
        array = array.astype(np.float32)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return torch.from_numpy(array).to(device)


def prepare_pi05_batch(policy, frames: Sequence[dict[str, Any]], *, include_actions: bool):
    _ensure_paths()
    from openpi.models import model as model_api

    transformed = [
        policy._input_transform(copy.deepcopy(pi05_raw_record(frame, include_actions=include_actions)))
        for frame in frames
    ]
    tree = _recursive_torch(_recursive_stack(transformed), policy._pytorch_device)
    actions = tree.pop("actions", None)
    if include_actions and actions is None:
        raise ValueError("pi0.5 input transform dropped teacher actions")
    observation = model_api.Observation.from_dict(tree)
    return observation, actions


def load_frozen(path: str | Path) -> dict[str, Any]:
    frozen_path = Path(path).resolve()
    result = json.loads(frozen_path.read_text(encoding="utf-8"))
    if result.get("kind") != "qvla_actquant_frozen_calibration":
        raise ValueError(f"not a frozen calibration manifest: {frozen_path}")
    if result.get("source") != "fp16_teacher_proxy" or result.get("source_protocol_equivalent") is not False:
        raise ValueError("calibration source semantics drift")
    return result


def iter_frame_batches(
    frozen: dict[str, Any],
    *,
    subset: str,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    if subset not in ("qvla", "actquant"):
        raise ValueError(subset)
    pending: list[dict[str, Any]] = []
    frame_index = 0
    for episode in frozen["episodes"]:
        if subset == "actquant" and not episode["actquant_subset"]:
            continue
        path = Path(episode["archive"])
        with np.load(path, allow_pickle=False) as archive:
            count = int(episode["frames"])
            # A compressed NPZ member is decompressed on every __getitem__.
            # Materialize each member once per episode instead of once per
            # frame; the yielded records and their order remain unchanged.
            images = archive["images"]
            wrist_images = archive["wrist_images"]
            right_images = archive["right_images"]
            states = archive["states"]
            prompts = archive["prompts"]
            teacher_actions = archive["teacher_actions"]
            lengths = {
                len(images), len(wrist_images), len(right_images),
                len(states), len(prompts), len(teacher_actions),
            }
            if lengths != {count}:
                raise ValueError(
                    f"archive/frame length drift for {path}: expected {count}, got {sorted(lengths)}"
                )
            for local_index in range(count):
                pending.append({
                    "image": images[local_index].copy(),
                    "wrist_image": wrist_images[local_index].copy(),
                    "right_image": right_images[local_index].copy(),
                    "state": states[local_index].copy(),
                    "prompt": str(prompts[local_index]),
                    "actions": teacher_actions[local_index].copy(),
                    "episode_key": episode["episode_key"],
                    "task": episode["task"],
                    "env_seed": int(episode["env_seed"]),
                    "replan_index": local_index,
                    "frame_index": frame_index,
                })
                frame_index += 1
                if len(pending) == batch_size:
                    yield pending
                    pending = []
    if pending:
        yield pending


def run_backbone_only(
    model_family: str,
    policy,
    prepared: Any,
    *,
    preserve_flow_rng: bool = False,
    actions: torch.Tensor | None = None,
) -> None:
    if model_family == "gr00t":
        backbone_inputs, action_inputs = policy.model.prepare_input(prepared)
        policy.model.backbone(backbone_inputs)
        if preserve_flow_rng:
            # The released HSIC path executes the complete training forward.
            # Its excluded action head draws CUDA Gaussian noise and then
            # samples a CPU-resident Beta distribution.  The latter advances
            # the same CPU RNG used by the next batch's train-mode image/state
            # transforms.  Reproduce those two draws without running the DiT,
            # otherwise a backbone-only optimization silently changes every
            # later calibration observation.
            action = action_inputs.action
            noise = torch.randn(
                action.shape, device=action.device, dtype=action.dtype
            )
            beta = policy.model.action_head.beta_dist
            if (
                beta.concentration1.dtype != torch.float32
                or beta.concentration0.dtype != torch.float32
            ):
                beta = torch.distributions.Beta(
                    beta.concentration1.float(), beta.concentration0.float()
                )
                policy.model.action_head.beta_dist = beta
            sampled_time = beta.sample([action.shape[0]]).to(
                action.device, dtype=action.dtype
            )
            del noise, sampled_time
        return
    if model_family == "pi05":
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        model = policy._model
        images, masks, tokens, token_masks, _ = model._preprocess_observation(
            prepared, train=preserve_flow_rng
        )
        if preserve_flow_rng:
            if actions is None:
                raise ValueError("pi0.5 flow-RNG preservation requires actions")
            # Match PI0Pytorch.forward exactly: preprocessing first, then
            # Gaussian noise and Beta time on the action device, then prefix
            # embedding.  Neither sampled value feeds the frozen target
            # backbone, so retaining them would only waste memory.
            noise = model.sample_noise(actions.shape, actions.device)
            sampled_time = model.sample_time(actions.shape[0], actions.device)
            del noise, sampled_time
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, masks, tokens, token_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        attention_mask = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Mirror the native sample_actions prefix-cache pass.  Passing no
        # suffix executes PaliGemma/SigLIP while deliberately bypassing the
        # action expert that the frozen target inventory excludes.
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        model.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        return
    raise ValueError(model_family)


def flow_loss(model_family: str, policy, prepared: Any, actions: torch.Tensor | None) -> torch.Tensor:
    if model_family == "gr00t":
        # GR00T stores its Beta distribution outside the nn.Module buffer
        # registry.  The local uniform-FP16 loader nevertheless converts its
        # concentration tensors to Half, for which PyTorch does not implement
        # Dirichlet/Beta sampling.  Restore only the stochastic sampler to
        # FP32 and let the released action head cast the sampled time back to
        # the action dtype.  Distribution parameters and RNG ordering stay
        # unchanged.
        beta = policy.model.action_head.beta_dist
        if (
            beta.concentration1.dtype != torch.float32
            or beta.concentration0.dtype != torch.float32
        ):
            policy.model.action_head.beta_dist = torch.distributions.Beta(
                beta.concentration1.float(), beta.concentration0.float()
            )
        output = policy.model(prepared)
        return output["loss"]
    if model_family == "pi05":
        if actions is None:
            raise ValueError("pi0.5 flow loss requires actions")
        return policy._model(prepared, actions).mean()
    raise ValueError(model_family)
