#!/usr/bin/env python3
"""π0.5-only loading/execution adapter for shared full-context scoring."""

from __future__ import annotations

import copy
import time

import jax
import numpy as np
import torch

from openpi.models import model as model_api


def prepared_observation(policy, raw: dict, device: str):
    transformed = policy._input_transform(copy.deepcopy(raw))
    batched = jax.tree.map(lambda value: np.asarray(value)[None, ...], transformed)
    tensors = jax.tree.map(
        lambda value: torch.from_numpy(np.array(value)).to(device), batched
    )
    return transformed, model_api.Observation.from_dict(tensors)


def run_records(
    policy,
    records: list[dict],
    device: str,
    *,
    noise_index: int = 0,
    flow_steps: int = 10,
) -> tuple[torch.Tensor, np.ndarray, list[float]]:
    if flow_steps < 1:
        raise ValueError("flow_steps must be positive")
    trajectories = []
    physical_actions = []
    timings = []
    for record in records:
        transformed, observation = prepared_observation(
            policy, record["observation"], device
        )
        noise_tensor = torch.from_numpy(record["noises"][noise_index])[None, ...].to(
            device
        )
        started = time.perf_counter()
        with torch.inference_mode():
            final, trajectory = policy._model.sample_actions(
                device,
                observation,
                noise=noise_tensor,
                num_steps=flow_steps,
                return_trajectory=True,
            )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        final_numpy = final[0].detach().to(torch.float32).cpu().numpy()
        output = policy._output_transform(
            {"state": np.asarray(transformed["state"]), "actions": final_numpy}
        )
        physical = np.asarray(output["actions"], dtype=np.float64)
        if physical.shape != (50, 12) or not np.isfinite(physical).all():
            raise RuntimeError(f"invalid π0.5 physical action output: {physical.shape}")
        trajectories.append(trajectory.detach().to(torch.float32).cpu())
        physical_actions.append(physical)
    return torch.cat(trajectories, dim=1), np.stack(physical_actions), timings
