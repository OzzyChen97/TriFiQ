#!/usr/bin/env python3
"""Shared LIBERO adapters and physical-action metrics for DyPAC-VLA."""

from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from gr00t_func_metrics import _se3_exp, _se3_log


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "scripts/quantvla_libero_dypac_protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
HORIZON, ACTION_DIM = map(int, PROTOCOL["action"]["shape"])
FLOW_STEPS = int(PROTOCOL["action"]["pi05_flow_steps"])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp.{__import__('os').getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def load_records(
    path: str | Path,
    *,
    selection_only: bool = False,
    model: str | None = None,
) -> list[dict]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "images", "wrist_images", "states", "prompts", "task_ids",
            "suite_ids", "task_indices", "env_seeds", "replan_indices",
            "selection_rows", "action_noises", "action_noises_b", "teacher_actions",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"LIBERO DyPAC buffer is missing {sorted(missing)}")
        noise_shape = tuple(np.asarray(archive["action_noises"]).shape[-2:])
        inferred_model = "pi05" if noise_shape == (10, 32) else "gr00t" if noise_shape == (16, 32) else None
        source_model = model or inferred_model
        if source_model not in {"pi05", "gr00t"} or inferred_model != source_model:
            raise ValueError(
                f"cannot bind LIBERO DyPAC noise shape {noise_shape} to model {model!r}"
            )
        records = []
        for index in range(len(archive["states"])):
            if selection_only and not bool(archive["selection_rows"][index]):
                continue
            noise_a = np.asarray(archive["action_noises"][index], dtype=np.float32)
            noise_b = np.asarray(archive["action_noises_b"][index], dtype=np.float32)
            teacher = np.asarray(archive["teacher_actions"][index], dtype=np.float32)
            expected_noise = (10, 32) if source_model == "pi05" else (16, 32)
            if noise_a.shape != expected_noise or noise_b.shape != expected_noise:
                raise ValueError(f"row {index}: invalid {source_model} noise shape")
            if teacher.shape != (HORIZON, ACTION_DIM):
                raise ValueError(f"row {index}: invalid physical teacher shape {teacher.shape}")
            state = np.asarray(archive["states"][index], dtype=np.float32)
            if state.shape != (8,):
                raise ValueError(f"row {index}: invalid LIBERO state shape {state.shape}")
            if source_model == "pi05":
                observation = {
                    "observation/image": np.asarray(archive["images"][index]),
                    "observation/wrist_image": np.asarray(archive["wrist_images"][index]),
                    "observation/state": state,
                    "prompt": str(archive["prompts"][index]),
                }
            else:
                observation = {
                    "video.image": np.asarray(archive["images"][index])[None, ...],
                    "video.wrist_image": np.asarray(archive["wrist_images"][index])[None, ...],
                    "state.x": state[0:1].reshape(1, 1),
                    "state.y": state[1:2].reshape(1, 1),
                    "state.z": state[2:3].reshape(1, 1),
                    "state.roll": state[3:4].reshape(1, 1),
                    "state.pitch": state[4:5].reshape(1, 1),
                    "state.yaw": state[5:6].reshape(1, 1),
                    "state.gripper": state[6:8].reshape(1, 2),
                    "annotation.human.action.task_description": [
                        str(archive["prompts"][index])
                    ],
                }
            records.append(
                {
                    "model": source_model,
                    "observation": observation,
                    "suite": str(archive["suite_ids"][index]),
                    "task": str(archive["task_ids"][index]),
                    "task_index": int(archive["task_indices"][index]),
                    "state_index": int(archive["env_seeds"][index]),
                    "replan": int(archive["replan_indices"][index]),
                    "selection": bool(archive["selection_rows"][index]),
                    "noises": (noise_a, noise_b),
                    "teacher_actions": teacher,
                }
            )
    return records


GR00T_ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def _stack_gr00t_observations(observations: Sequence[Mapping[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for key in observations[0]:
        values = [row[key] for row in observations]
        if isinstance(values[0], np.ndarray):
            result[key] = np.stack(values, axis=0)
        else:
            result[key] = np.asarray(
                [value[0] if isinstance(value, (list, tuple)) else value for value in values]
            )
    return result


def _gr00t_physical_actions(output: Mapping[str, Any]) -> np.ndarray:
    components = []
    for key in GR00T_ACTION_KEYS:
        name = f"action.{key}"
        if name not in output:
            raise KeyError(f"GR00T output omitted {name}; got {sorted(output)}")
        value = np.asarray(output[name], dtype=np.float32)
        if value.ndim == 2:
            value = value[..., None]
        if value.ndim != 3 or value.shape[-1] != 1:
            raise ValueError(f"invalid GR00T physical component {name}: {value.shape}")
        components.append(value)
    actions = np.concatenate(components, axis=-1)
    if actions.shape[1] < HORIZON or not np.isfinite(actions).all():
        raise ValueError(f"invalid GR00T physical action chunk: {actions.shape}")
    return actions[:, :HORIZON].copy()


def run_gr00t_policy(
    policy,
    records: Sequence[dict],
    *,
    noise_index: int = 0,
    batch_size: int = 4,
) -> np.ndarray:
    physical: list[np.ndarray] = []
    for batch in iter_batches(records, batch_size):
        if any(record.get("model") != "gr00t" for record in batch):
            raise ValueError("run_gr00t_policy received a non-GR00T record")
        observations = _stack_gr00t_observations(
            [record["observation"] for record in batch]
        )
        noises = torch.from_numpy(
            np.stack([record["noises"][noise_index] for record in batch])
        ).float()
        output = policy.get_action(observations, action_noise=noises)
        physical.extend(_gr00t_physical_actions(output))
    return np.stack(physical)


def iter_batches(records: Sequence[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(records), int(batch_size)):
        yield list(records[start : start + int(batch_size)])


def prepare_batch(policy, records: Sequence[dict], device: str):
    import jax
    from openpi.models import model as model_api

    transformed = [
        policy._input_transform(copy.deepcopy(record["observation"])) for record in records
    ]
    stacked = jax.tree.map(
        lambda *values: np.stack([np.asarray(value) for value in values], axis=0),
        *transformed,
    )
    tensors = jax.tree.map(
        lambda value: torch.from_numpy(np.asarray(value)).to(device), stacked
    )
    return transformed, model_api.Observation.from_dict(tensors)


def run_policy(
    policy,
    records: Sequence[dict],
    device: str,
    *,
    noise_index: int = 0,
    batch_size: int = 4,
) -> np.ndarray:
    physical: list[np.ndarray] = []
    for batch in iter_batches(records, batch_size):
        transformed, observation = prepare_batch(policy, batch, device)
        noise = torch.from_numpy(
            np.stack([record["noises"][noise_index] for record in batch])
        ).to(device)
        with torch.inference_mode():
            final = policy._model.sample_actions(
                device, observation, noise=noise, num_steps=FLOW_STEPS
            )
        for index, action in enumerate(final):
            output = policy._output_transform(
                {
                    "state": np.asarray(transformed[index]["state"]),
                    "actions": action.detach().to(torch.float32).cpu().numpy(),
                }
            )
            value = np.asarray(output["actions"], dtype=np.float32)
            if value.shape[0] < HORIZON or value.shape[1] != ACTION_DIM:
                raise RuntimeError(f"invalid π0.5 LIBERO physical action shape: {value.shape}")
            physical.append(value[:HORIZON].copy())
    return np.stack(physical)


def canonical_actions(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if tensor.ndim != 3 or tuple(tensor.shape[-2:]) != (HORIZON, ACTION_DIM):
        raise ValueError(f"actions must be (N,{HORIZON},{ACTION_DIM}), got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError("actions contain non-finite values")
    return tensor


def physical_scale(teacher: Any) -> torch.Tensor:
    value = canonical_actions(teacher).reshape(-1, ACTION_DIM).to(torch.float64)
    median = value.median(dim=0).values
    mad = (value - median).abs().median(dim=0).values
    rms = value.square().mean(dim=0).sqrt()
    robust = torch.maximum(1.4826 * mad, 0.1 * rms)
    return torch.maximum(robust, 0.05 * robust.median()).clamp_min(1e-4).float()


def _rho(value: torch.Tensor) -> torch.Tensor:
    value = value.to(torch.float64)
    return (2.0 * torch.sqrt(1.0 + value.square()) - 2.0).float()


def _mean(value: torch.Tensor) -> float:
    return float(_rho(value).mean()) if value.numel() else 0.0


def sequence_d_pac(reference: torch.Tensor, candidate: torch.Tensor, scale: torch.Tensor) -> dict:
    ref = reference.reshape(-1, ACTION_DIM)
    quant = candidate.reshape(-1, ACTION_DIM)
    error = (quant - ref) / scale
    count = torch.arange(1, len(error) + 1, dtype=error.dtype).sqrt().unsqueeze(1)
    local = _mean(error)
    prefix = _mean(error.cumsum(0) / count)

    if torch.equal(ref[:, :6], quant[:, :6]):
        pose = 0.0
    else:
        ref_pose = torch.eye(4, dtype=torch.float32)
        quant_pose = torch.eye(4, dtype=torch.float32)
        pose_errors = []
        for index, (ref_action, quant_action) in enumerate(zip(ref, quant), start=1):
            ref_pose = ref_pose @ _se3_exp(ref_action[:6])
            quant_pose = quant_pose @ _se3_exp(quant_action[:6])
            pose_errors.append(
                _se3_log(torch.linalg.solve(ref_pose, quant_pose)) / scale[:6] / np.sqrt(index)
            )
        pose = _mean(torch.stack(pose_errors))

    ref_jump = reference[1:, 0] - reference[:-1, -1]
    quant_jump = candidate[1:, 0] - candidate[:-1, -1]
    stitch = _mean((quant_jump - ref_jump) / scale) if len(reference) > 1 else 0.0

    ref_grip = torch.sigmoid(4.0 * ref[:, 6:7] / scale[6:7])
    quant_grip = torch.sigmoid(4.0 * quant[:, 6:7] / scale[6:7])
    grip_state = _mean(quant_grip - ref_grip)
    grip_event = _mean(
        (quant_grip[1:] - quant_grip[:-1]) - (ref_grip[1:] - ref_grip[:-1])
    )
    grip = 0.5 * (grip_state + grip_event)
    components = {"local": local, "prefix": prefix, "pose": pose, "stitch": stitch, "grip": grip}
    return {"d_pac_sequence": float(sum(components.values()) / 5.0), "components": components}


def _d_func_sequence(reference: torch.Tensor, candidate: torch.Tensor, scale: torch.Tensor) -> dict:
    error = (candidate - reference) / scale
    per_chunk = _rho(error).mean(dim=(1, 2)).numpy()
    tail_count = max(1, int(np.ceil(0.1 * len(per_chunk))))
    cvar = float(np.sort(per_chunk)[-tail_count:].mean())
    final = _mean(error[-1])
    kin = 0.5 * (_mean(error[..., :3]) + _mean(error[..., 3:6]))
    grip = float((torch.sign(candidate[..., 6]) != torch.sign(reference[..., 6])).float().mean())
    return {
        "d_func_sequence": float((final + kin + grip + 2.0 * cvar) / 5.0),
        "final": final,
        "kin": kin,
        "grip": grip,
        "cvar90": cvar,
    }


def summarize_pair(reference: Any, candidate: Any, records: Sequence[Mapping[str, Any]], *, scale: Any | None = None) -> dict:
    ref = canonical_actions(reference)
    quant = canonical_actions(candidate)
    if ref.shape != quant.shape or len(ref) != len(records):
        raise ValueError("paired action/record shape mismatch")
    dimension_scale = physical_scale(ref) if scale is None else torch.as_tensor(scale).float()
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(str(record["suite"]), int(record["task_index"]), int(record["state_index"]))].append(index)
    sequences = []
    funcs = []
    for key, positions in sorted(groups.items()):
        positions.sort(key=lambda index: int(records[index]["replan"]))
        replans = [int(records[index]["replan"]) for index in positions]
        if replans != PROTOCOL["benchmark"]["retained_replan_indices"]:
            raise ValueError(f"incomplete ordered sequence {key}: {replans}")
        selected = torch.tensor(positions, dtype=torch.long)
        pac = sequence_d_pac(ref.index_select(0, selected), quant.index_select(0, selected), dimension_scale)
        func = _d_func_sequence(ref.index_select(0, selected), quant.index_select(0, selected), dimension_scale)
        sequences.append({"suite": key[0], "task_index": key[1], "state_index": key[2], **pac})
        funcs.append({"suite": key[0], "task_index": key[1], "state_index": key[2], **func})
    values = np.asarray([row["d_pac_sequence"] for row in sequences], dtype=np.float64)
    tail_count = max(1, int(np.ceil(0.1 * len(values))))
    mean = float(values.mean())
    cvar = float(np.sort(values)[-tail_count:].mean())
    suite_risk = {
        suite: float(np.mean([row["d_pac_sequence"] for row in sequences if row["suite"] == suite]))
        for suite in sorted({row["suite"] for row in sequences})
    }
    func_values = np.asarray([row["d_func_sequence"] for row in funcs], dtype=np.float64)
    return {
        "d_pac_summary": {
            "d_pac": mean + cvar,
            "mean": mean,
            "cvar90": cvar,
            "suite_mean": suite_risk,
            "suite_minimax": max(suite_risk.values()),
            "per_sequence": values.tolist(),
            "sequences": sequences,
            "dimension_scale": dimension_scale.tolist(),
        },
        "d_func_summary": {
            "d_func": float(func_values.mean()),
            "per_sequence": func_values.tolist(),
            "sequences": funcs,
        },
    }


def selftest() -> None:
    generator = torch.Generator().manual_seed(31)
    value = torch.randn(8, HORIZON, ACTION_DIM, generator=generator)
    records = [
        {"suite": "goal", "task_index": index // 4, "state_index": 0, "replan": [0, 2, 4, 6][index % 4]}
        for index in range(8)
    ]
    same = summarize_pair(value, value, records)
    assert same["d_pac_summary"]["d_pac"] == 0.0
    assert same["d_func_summary"]["d_func"] == 0.0
    changed = value.clone(); changed[..., 0] += 0.1
    assert summarize_pair(value, changed, records)["d_pac_summary"]["d_pac"] > 0
    print("[libero-dypac] metric selftest OK")


if __name__ == "__main__":
    selftest()
