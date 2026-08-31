"""Shared GR00T-aligned batching helpers for π0.5 calibration tools."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import jax
import numpy as np
import torch

from openpi.models import model as model_api


FLOW_STEPS = 4
CALIBRATION_BATCHES = 32
CALIBRATION_BATCH_SIZE = 8
ATM_OBSERVATIONS = 16
ATM_BATCH_SIZE = 8


def load_records(path: str | Path, n_observations: int) -> list[dict]:
    """Load observations plus the canonical seed-0 Torch noise stream."""
    with np.load(Path(path), allow_pickle=False) as archive:
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
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"GR00T-aligned calibration buffer is missing {sorted(missing)}")
        size = len(archive["states"])
        if not (0 < n_observations <= size):
            raise ValueError(f"n_observations must be in [1, {size}], got {n_observations}")
        records = []
        for index in range(n_observations):
            noise = np.asarray(archive["action_noises"][index])
            if noise.shape != (50, 32) or noise.dtype != np.float32:
                raise ValueError(f"invalid canonical action noise at row {index}: {noise.shape}/{noise.dtype}")
            records.append(
                {
                    "observation": {
                        "observation/image": np.asarray(archive["images"][index]),
                        "observation/wrist_image": np.asarray(archive["wrist_images"][index]),
                        "observation/right_image": np.asarray(archive["right_images"][index]),
                        "observation/state": np.asarray(archive["states"][index], dtype=np.float32),
                        "prompt": str(archive["prompts"][index]),
                    },
                    "task": str(archive["task_ids"][index]),
                    "seed": int(archive["env_seeds"][index]),
                    "noise": noise,
                }
            )
    return records


def iter_batches(records: list[dict], batch_size: int) -> Iterable[list[dict]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def prepare_observation_batch(policy, records: list[dict], device: str):
    transformed = [
        policy._input_transform(copy.deepcopy(record["observation"])) for record in records
    ]
    stacked = jax.tree.map(
        lambda *values: np.stack([np.asarray(value) for value in values], axis=0),
        *transformed,
    )
    tensors = jax.tree.map(
        lambda value: torch.from_numpy(np.asarray(value)).to(device),
        stacked,
    )
    return transformed, model_api.Observation.from_dict(tensors)


def sample_batch(
    policy,
    records: list[dict],
    device: str,
    *,
    num_steps: int = FLOW_STEPS,
    return_trajectory: bool = False,
):
    """Run one true model batch with the buffer's canonical action noises."""
    _, observation = prepare_observation_batch(policy, records, device)
    # Calibration buffers retain the canonical 50-step noise stream so the
    # same result-blind artifact can serve both RoboCasa (H=50) and LIBERO
    # pi0.5 (H=10).  Feed only the horizon declared by the loaded checkpoint;
    # otherwise the action embeddings have length 50 while the model builds a
    # length-10 suffix attention mask.
    horizon = int(policy._model.config.action_horizon)
    action_dim = int(policy._model.config.action_dim)
    canonical = np.stack([record["noise"] for record in records], axis=0)
    if canonical.ndim != 3 or canonical.shape[1] < horizon or canonical.shape[2] != action_dim:
        raise ValueError(
            "canonical action noise is incompatible with the loaded model: "
            f"noise={canonical.shape}, horizon={horizon}, action_dim={action_dim}"
        )
    noises = torch.from_numpy(canonical[:, :horizon, :]).to(device)
    with torch.inference_mode():
        return policy._model.sample_actions(
            device,
            observation,
            noise=noises,
            num_steps=int(num_steps),
            return_trajectory=return_trajectory,
        )
