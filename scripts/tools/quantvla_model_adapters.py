#!/usr/bin/env python3
"""The only model-varying boundary in the cross-model QuantVLA protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantvla_cross_model_protocol import PROTOCOL, adapter_attestation, sha256_file


NATIVE_HORIZONS = {"gr00t": 16, "pi05": 50}
CANONICAL_HORIZON = int(PROTOCOL["metrics"]["canonical_action"]["horizon"])
CANONICAL_ACTION_DIM = int(PROTOCOL["metrics"]["canonical_action"]["dimension"])
FLOW_STEPS = int(PROTOCOL["closed_loop"]["flow_steps"])
GR00T_PHYSICAL_ACTION_KEYS = (
    ("action.end_effector_position", 3),
    ("action.end_effector_rotation", 3),
    ("action.gripper_close", 1),
    ("action.base_motion", 4),
    ("action.control_mode", 1),
)


def _load_archive_rows(path: str | Path, n_obs: int) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as archive:
        required = {
            "images",
            "wrist_images",
            "right_images",
            "states",
            "prompts",
            "task_ids",
            "env_seeds",
            "action_noises",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"canonical buffer lacks fields: {missing}")
        if n_obs < 1 or n_obs > len(archive["states"]):
            raise ValueError(f"n_obs={n_obs} exceeds buffer size {len(archive['states'])}")
        rows = []
        for index in range(n_obs):
            state = np.asarray(archive["states"][index], dtype=np.float32)
            noise = np.asarray(archive["action_noises"][index], dtype=np.float32)
            if state.shape != (16,) or noise.shape != (50, 32):
                raise ValueError(
                    f"canonical row {index} shape drift: state={state.shape}, noise={noise.shape}"
                )
            rows.append(
                {
                    "index": index,
                    "image": np.asarray(archive["images"][index]),
                    "wrist_image": np.asarray(archive["wrist_images"][index]),
                    "right_image": np.asarray(archive["right_images"][index]),
                    "state": state,
                    "prompt": str(archive["prompts"][index]),
                    "task": str(archive["task_ids"][index]),
                    "seed": int(archive["env_seeds"][index]),
                    "env_step": (
                        int(archive["env_steps"][index])
                        if "env_steps" in archive.files else None
                    ),
                    "replan": (
                        int(archive["replan_indices"][index])
                        if "replan_indices" in archive.files else None
                    ),
                    "noise": noise,
                    "secondary_noise": (
                        np.asarray(archive["action_noises"][index + 32], dtype=np.float32)
                        if index + 32 < len(archive["action_noises"]) else None
                    ),
                }
            )
    return rows


def _gr00t_observation(row: dict[str, Any]) -> dict[str, Any]:
    # Canonical state order in the archive is eef-pos, eef-rot, base-pos,
    # base-rot, gripper. GR00T's data adapter expects gripper before base.
    state = row["state"]
    return {
        "video.robot0_agentview_left": row["image"][None, ...],
        "video.robot0_agentview_right": row["right_image"][None, ...],
        "video.robot0_eye_in_hand": row["wrist_image"][None, ...],
        "state.end_effector_position_relative": state[0:3][None, ...],
        "state.end_effector_rotation_relative": state[3:7][None, ...],
        "state.gripper_qpos": state[14:16][None, ...],
        "state.base_position": state[7:10][None, ...],
        "state.base_rotation": state[10:14][None, ...],
        "annotation.human.task_description": [row["prompt"]],
    }


def _pi05_observation(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation/image": row["image"],
        "observation/wrist_image": row["wrist_image"],
        "observation/right_image": row["right_image"],
        "observation/state": row["state"],
        "prompt": row["prompt"],
    }


def load_model_records(
    path: str | Path,
    n_obs: int,
    *,
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load identical archive rows and adapt only the model-facing boundary."""
    if model not in NATIVE_HORIZONS:
        raise ValueError(f"unknown model adapter: {model!r}")
    rows = _load_archive_rows(path, n_obs)
    native_horizon = NATIVE_HORIZONS[model]
    records = []
    for row in rows:
        observation = _gr00t_observation(row) if model == "gr00t" else _pi05_observation(row)
        record = {
            "observation": observation,
            "task": row["task"],
            "seed": row["seed"],
            "noises": [row["noise"][:native_horizon].copy()],
        }
        if row["secondary_noise"] is not None:
            secondary = row["secondary_noise"]
            if secondary.shape != (50, 32):
                raise ValueError(f"canonical secondary-noise shape drift: {secondary.shape}")
            record["noises"].append(secondary[:native_horizon].copy())
        if row["env_step"] is not None:
            record["env_step"] = row["env_step"]
        if row["replan"] is not None:
            record["replan"] = row["replan"]
        records.append(record)
    provenance = {
        "path": str(Path(path).expanduser().resolve()),
        "sha256": sha256_file(path),
        "rows": n_obs,
        "row_indices": list(range(n_obs)),
        "adapter": adapter_attestation(model, native_action_horizon=native_horizon),
    }
    return records, provenance


def canonical_trajectory(trajectory: Any, *, model: str) -> torch.Tensor:
    """Canonicalize a *diagnostic-only* native normalized solver trajectory.

    This API is intentionally not consumed by ``quantvla_metric_protocol``.
    It remains for CKA/CS and solver diagnostics so existing diagnostic tools
    can be compared without giving normalized coordinates selection authority.
    """
    if model not in NATIVE_HORIZONS:
        raise ValueError(f"unknown model adapter: {model!r}")
    value = torch.as_tensor(trajectory).detach().to(dtype=torch.float32, device="cpu")
    if value.ndim not in (3, 4):
        raise ValueError(f"trajectory must be (R,H,D) or (T+1,R,H,D), got {value.shape}")
    if value.ndim == 4 and value.shape[0] != FLOW_STEPS + 1:
        raise ValueError(f"expected {FLOW_STEPS + 1} solver states, got {value.shape[0]}")
    native_horizon = NATIVE_HORIZONS[model]
    if value.shape[-2] != native_horizon:
        raise ValueError(
            f"{model} adapter requires native horizon {native_horizon}, got {value.shape[-2]}"
        )
    if value.shape[-1] < CANONICAL_ACTION_DIM:
        raise ValueError(
            f"{model} trajectory has {value.shape[-1]} dims, needs {CANONICAL_ACTION_DIM}"
        )
    value = value[..., :CANONICAL_HORIZON, :CANONICAL_ACTION_DIM].contiguous()
    if not torch.isfinite(value).all():
        raise ValueError(f"{model} canonical trajectory contains non-finite values")
    return value


def canonical_physical_chunk(actions: Any, *, model: str) -> torch.Tensor:
    """Map a model-native physical final action chunk to shared ``(...,16,12)``."""
    if model not in NATIVE_HORIZONS:
        raise ValueError(f"unknown model adapter: {model!r}")
    value = torch.as_tensor(actions).detach().to(dtype=torch.float32, device="cpu")
    if value.ndim not in (2, 3):
        raise ValueError(f"physical actions must be (H,12) or (B,H,12), got {value.shape}")
    if value.shape[-2] != NATIVE_HORIZONS[model] or value.shape[-1] != CANONICAL_ACTION_DIM:
        raise ValueError(
            f"{model} physical action shape must end in "
            f"({NATIVE_HORIZONS[model]},{CANONICAL_ACTION_DIM}), got {value.shape}"
        )
    result = value[..., :CANONICAL_HORIZON, :].contiguous()
    if not torch.isfinite(result).all():
        raise ValueError(f"{model} physical action chunk contains non-finite values")
    return result


def gr00t_inverse_normalize_final(policy: Any, normalized_final: Any) -> torch.Tensor:
    """Apply GR00T's native inverse transform and concatenate deployed semantics."""
    value = torch.as_tensor(normalized_final).detach().to(dtype=torch.float32, device="cpu")
    if value.ndim not in (2, 3) or value.shape[-2] != NATIVE_HORIZONS["gr00t"]:
        raise ValueError(f"invalid GR00T normalized final chunk: {value.shape}")
    unnormalized = policy._get_unnormalized_action(value)
    components = []
    for key, width in GR00T_PHYSICAL_ACTION_KEYS:
        if key not in unnormalized:
            raise KeyError(f"GR00T inverse transform omitted {key}; got {sorted(unnormalized)}")
        component = torch.as_tensor(unnormalized[key]).detach().to(torch.float32)
        if component.ndim == value.ndim - 1 and width == 1:
            component = component.unsqueeze(-1)
        if component.shape[:-1] != value.shape[:-1] or component.shape[-1] != width:
            raise ValueError(
                f"GR00T physical {key} shape {component.shape}, expected "
                f"{tuple(value.shape[:-1]) + (width,)}"
            )
        components.append(component)
    return canonical_physical_chunk(torch.cat(components, dim=-1), model="gr00t")


def pi05_inverse_normalize_final(
    policy: Any,
    transformed_observation: dict[str, Any],
    normalized_final: Any,
) -> torch.Tensor:
    """Apply pi0.5's native output transform to one final 50-step chunk."""
    value = torch.as_tensor(normalized_final).detach().to(torch.float32).cpu().numpy()
    if value.shape != (NATIVE_HORIZONS["pi05"], CANONICAL_ACTION_DIM):
        raise ValueError(f"invalid pi0.5 normalized final chunk: {value.shape}")
    output = policy._output_transform(
        {"state": np.asarray(transformed_observation["state"]), "actions": value}
    )
    return canonical_physical_chunk(np.asarray(output["actions"], dtype=np.float32), model="pi05")


def gr00t_rollout_inputs(
    records: list[dict[str, Any]], *, noise_index: int = 0
) -> tuple[list[dict], list[torch.Tensor]]:
    observations = [record["observation"] for record in records]
    if any(len(record["noises"]) <= noise_index for record in records):
        raise ValueError(f"GR00T records do not contain noise index {noise_index}")
    noises = [
        torch.from_numpy(record["noises"][noise_index]).float()
        for record in records
    ]
    return observations, noises


def record_metadata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in records:
        item = {"task": row["task"], "seed": int(row["seed"])}
        for key in ("env_step", "replan"):
            if key in row:
                item[key] = int(row[key])
        result.append(item)
    return result


def validate_calibration_artifact(
    artifact: str | Path,
    *,
    model: str,
    expected_buffer_sha256: str,
) -> dict[str, Any]:
    """Require model-native A8 artifacts to name the same source buffer."""
    path = Path(artifact).expanduser().resolve()
    candidates = [Path(str(path) + ".meta.json"), Path(str(path) + ".json")]
    sidecar = next((candidate for candidate in candidates if candidate.is_file()), None)
    if sidecar is None:
        raise FileNotFoundError(f"{model} calibration sidecar missing for {path}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", payload)
    declared = metadata.get("calibration_buffer_sha256") or metadata.get(
        "source_buffer_sha256"
    )
    if declared != expected_buffer_sha256:
        raise ValueError(
            f"{model} A8 calibration data drift: {declared!r} != "
            f"{expected_buffer_sha256}; regenerate from the shared buffer"
        )
    return {
        "artifact": str(path),
        "artifact_sha256": sha256_file(path),
        "sidecar": str(sidecar),
        "sidecar_sha256": sha256_file(sidecar),
        "calibration_buffer_sha256": declared,
    }
