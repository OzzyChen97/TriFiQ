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
    normalized = policy.apply_transforms(raw)
    return normalized


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
            for local_index in range(count):
                pending.append({
                    "image": archive["images"][local_index].copy(),
                    "wrist_image": archive["wrist_images"][local_index].copy(),
                    "right_image": archive["right_images"][local_index].copy(),
                    "state": archive["states"][local_index].copy(),
                    "prompt": str(archive["prompts"][local_index]),
                    "actions": archive["teacher_actions"][local_index].copy(),
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


def run_backbone_only(model_family: str, policy, prepared: Any) -> None:
    if model_family == "gr00t":
        backbone_inputs, _ = policy.model.prepare_input(prepared)
        policy.model.backbone(backbone_inputs)
        return
    if model_family == "pi05":
        model = policy._model
        images, masks, tokens, token_masks, _ = model._preprocess_observation(prepared, train=False)
        model.embed_prefix(images, masks, tokens, token_masks)
        return
    raise ValueError(model_family)


def flow_loss(model_family: str, policy, prepared: Any, actions: torch.Tensor | None) -> torch.Tensor:
    if model_family == "gr00t":
        output = policy.model(prepared)
        return output["loss"]
    if model_family == "pi05":
        if actions is None:
            raise ValueError("pi0.5 flow loss requires actions")
        return policy._model(prepared, actions).mean()
    raise ValueError(model_family)
