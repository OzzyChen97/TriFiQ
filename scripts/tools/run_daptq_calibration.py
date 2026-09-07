#!/usr/bin/env python3
"""Calibrate one frozen DA-PTQ model unit and export an auditable pack."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from daptq.core import (  # noqa: E402
    DAPTQ_GIT_COMMIT,
    DAPTQLinear,
    action_block_index,
    build_target_inventory,
    canonical_hash,
    make_block_rotations,
    mse_w4_quantize,
    pack_signed_nibbles,
    sha256_file,
)
from qvla_actquant.model_adapters import (  # noqa: E402
    load_frozen,
    load_gr00t_policy,
    load_pi05_policy,
    prepare_gr00t_batch,
    prepare_pi05_batch,
)
from quantvla_table1_bytes import TABLE1_FP16_BYTES  # noqa: E402


PROTOCOL_PATH = ROOT / "scripts" / "daptq_table1_protocol.json"


def _ordered_io_map(function, values: list[Any]) -> list[Any]:
    """Run independent NPZ reads concurrently without changing their order."""
    if not values:
        return []
    workers = min(len(values), max(1, int(os.environ.get("DAPTQ_IO_WORKERS", "8"))))
    if workers == 1:
        return [function(value) for value in values]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="daptq-npz") as pool:
        return list(pool.map(function, values))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_of(model_family: str, policy):
    return policy.model if model_family == "gr00t" else policy._model


def load_policy(model_family: str, checkpoint: Path, device: str):
    return (
        load_gr00t_policy(checkpoint, device)
        if model_family == "gr00t"
        else load_pi05_policy(checkpoint, device)
    )


def prepare(model_family: str, policy, frame: dict[str, Any], *, include_actions: bool):
    if model_family == "gr00t":
        return prepare_gr00t_batch(policy, [frame], include_actions=include_actions), None
    return prepare_pi05_batch(policy, [frame], include_actions=include_actions)


def _episode_first_action(episode: dict[str, Any]) -> np.ndarray:
    """Read only the small action member used to order all 512 trajectories."""
    path = Path(episode["archive"])
    with np.load(path, allow_pickle=False) as archive:
        if int(episode["frames"]) < 1:
            raise ValueError(f"calibration trajectory has no replan frame: {path}")
        return archive["teacher_actions"][0].copy()


def _episode_frame(
    episode: dict[str, Any], *, first_action: np.ndarray | None = None
) -> dict[str, Any]:
    path = Path(episode["archive"])
    with np.load(path, allow_pickle=False) as archive:
        if int(episode["frames"]) < 1:
            raise ValueError(f"calibration trajectory has no replan frame: {path}")
        return {
            "image": archive["images"][0].copy(),
            "wrist_image": archive["wrist_images"][0].copy(),
            "right_image": archive["right_images"][0].copy(),
            "state": archive["states"][0].copy(),
            "prompt": str(archive["prompts"][0]),
            "actions": (
                np.asarray(first_action).copy()
                if first_action is not None
                else archive["teacher_actions"][0].copy()
            ),
            "episode_key": episode["episode_key"],
            "task": episode["task"],
            "env_seed": int(episode["env_seed"]),
        }


def spatially_interleaved_frames(
    frozen: dict[str, Any], bins: int, *, materialize: int
) -> list[dict[str, Any]]:
    """Order all trajectories by motion while materializing only used images.

    NPZ image members contain an entire rollout and cannot be memory-mapped.
    Reading ``images[0]`` therefore decompresses the whole member.  DA-PTQ
    uses all 512 first-step actions for spatial ordering, but this adapter only
    forwards the first 64 ordered trajectories for its 16-step probe and
    64-batch A8/CSRC statistics.  Keep those semantics while avoiding image
    decompression for the remaining 448 trajectories.
    """
    episodes = list(frozen["episodes"])
    if len(episodes) != 512:
        raise ValueError(f"DA-PTQ requires exactly 512 trajectories, got {len(episodes)}")
    if materialize < 1 or materialize > len(episodes):
        raise ValueError(f"invalid DA-PTQ materialization count: {materialize}")
    actions = _ordered_io_map(_episode_first_action, episodes)
    xyz = np.stack([np.asarray(action)[0, :3] for action in actions])
    principal = xyz[:, int(np.argmax(np.var(xyz, axis=0)))]
    edges = np.quantile(principal, np.linspace(0.0, 1.0, bins + 1))
    labels = np.clip(np.searchsorted(edges[1:-1], principal, side="right"), 0, bins - 1)
    queues = [
        sorted(
            [
                (episode, action)
                for episode, action, label in zip(episodes, actions, labels, strict=True)
                if int(label) == index
            ],
            key=lambda row: row[0]["episode_key"],
        )
        for index in range(bins)
    ]
    ordered: list[tuple[dict[str, Any], np.ndarray]] = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))
    def materialize_frame(row: tuple[dict[str, Any], np.ndarray]) -> dict[str, Any]:
        episode, action = row
        return _episode_frame(episode, first_action=action)

    return _ordered_io_map(materialize_frame, ordered[:materialize])


def structural_weights(actions: torch.Tensor, protocol: dict[str, Any]) -> torch.Tensor:
    cfg = protocol["da_mpa"]
    q = actions.detach().float()[..., :7]
    if q.ndim == 3:
        q = q[:, 0, :]
    theta = torch.cumsum(q, dim=1)
    batch, dims = q.shape
    jx = torch.zeros((batch, dims), dtype=torch.float32, device=q.device)
    jy = torch.zeros_like(jx)
    for index in range(dims):
        jx[:, index] = -torch.sin(theta[:, index:]).sum(dim=1)
        jy[:, index] = torch.cos(theta[:, index:]).sum(dim=1)
    jacobian = torch.stack([jx, jy, torch.ones_like(jx)], dim=1)
    eye = torch.eye(3, device=q.device).expand(batch, -1, -1)
    inverse = torch.linalg.inv(
        jacobian @ jacobian.transpose(1, 2) + float(cfg["damping"]) * eye
    )
    pseudo = jacobian.transpose(1, 2) @ inverse
    axes = torch.tensor(
        [cfg["translation_weight"], cfg["translation_weight"], cfg["rotation_weight"]],
        device=q.device,
    ).view(1, 1, 3)
    scores = (pseudo.abs() * axes).sum(dim=2).mean(dim=0).clamp_min(1e-8)
    scores = scores / scores.mean()
    scores = 1.0 + float(cfg["scaling_gain"]) * (scores - 1.0)
    return scores.clamp_(0.05, 20.0)


def gr00t_weighted_loss(policy, prepared, protocol: dict[str, Any]) -> torch.Tensor:
    model = policy.model
    backbone_inputs, action_inputs = model.prepare_input(prepared)
    backbone_output = model.backbone(backbone_inputs)
    head = model.action_head
    backbone_output = head.process_backbone_output(backbone_output)
    vl_embs = backbone_output.backbone_features
    embodiment_id = action_inputs.embodiment_id
    state_features = head.state_encoder(action_inputs.state, embodiment_id)
    actions = action_inputs.action
    noise = torch.randn_like(actions)
    # GR00T's cached Beta distribution inherits the model default dtype and
    # torch does not implement Dirichlet/Beta sampling in BF16.  Sample the
    # same configured distribution explicitly in FP32, then cast the result
    # for the BF16 flow network.
    beta = torch.distributions.Beta(
        torch.tensor(
            head.config.noise_beta_alpha, device=actions.device, dtype=torch.float32
        ),
        torch.tensor(
            head.config.noise_beta_beta, device=actions.device, dtype=torch.float32
        ),
    )
    sample = beta.sample((actions.shape[0],))
    timestep = ((float(head.config.noise_s) - sample) / float(head.config.noise_s)).to(
        actions.dtype
    )
    noisy = (1 - timestep[:, None, None]) * noise + timestep[:, None, None] * actions
    velocity = actions - noise
    discrete = (timestep * head.num_timestep_buckets).long()
    action_features = head.action_encoder(noisy, discrete, embodiment_id)
    if head.config.add_pos_embed:
        ids = torch.arange(action_features.shape[1], device=actions.device)
        action_features = action_features + head.position_embedding(ids).unsqueeze(0)
    future = head.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
    joined = torch.cat((state_features, future, action_features), dim=1)
    hidden = head.model(
        hidden_states=joined,
        encoder_hidden_states=vl_embs,
        encoder_attention_mask=backbone_output.backbone_attention_mask,
        timestep=discrete,
    )
    prediction = head.action_decoder(hidden, embodiment_id)[:, -actions.shape[1] :]
    error = (prediction - velocity).square() * action_inputs.action_mask
    weights = torch.ones(error.shape[-1], device=error.device, dtype=error.dtype)
    weights[:7] = structural_weights(actions, protocol).to(error.dtype)
    weighted = error * weights.view(1, 1, -1)
    return weighted.sum() / (action_inputs.action_mask * weights.view(1, 1, -1)).sum()


def weighted_flow_loss(
    model_family: str,
    policy,
    prepared,
    actions: torch.Tensor | None,
    protocol: dict[str, Any],
) -> torch.Tensor:
    if model_family == "gr00t":
        return gr00t_weighted_loss(policy, prepared, protocol)
    if actions is None:
        raise ValueError("pi0.5 DA-MPA requires teacher actions")
    elementwise = policy._model(prepared, actions)
    weights = torch.ones(elementwise.shape[-1], device=elementwise.device, dtype=elementwise.dtype)
    weights[:7] = structural_weights(actions, protocol).to(elementwise.dtype)
    return (elementwise * weights.view(1, 1, -1)).mean()


def profile_sensitivity(
    model_family: str,
    policy,
    frames: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, float]:
    model = model_of(model_family, policy)
    modules = dict(model.named_modules())
    last_block = max(int(row["action_block"]) for row in inventory if row["action_block"] is not None)
    skip_from = last_block - int(protocol["da_mpa"]["skip_last_action_blocks"]) + 1
    eligible = [
        row["name"]
        for row in inventory
        if row["role"] == "action_mlp" and int(row["action_block"]) < skip_from
    ]
    states = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    for parameter, _ in states:
        parameter.requires_grad_(False)
    for name in eligible:
        modules[name].weight.requires_grad_(True)
    totals = {name: 0.0 for name in eligible}
    counts = {name: 0 for name in eligible}
    try:
        for index, frame in enumerate(frames[: int(protocol["da_mpa"]["probe_steps"])]):
            seed_all(42 + 200_000 + index)
            model.zero_grad(set_to_none=True)
            prepared, actions = prepare(model_family, policy, frame, include_actions=True)
            loss = weighted_flow_loss(model_family, policy, prepared, actions, protocol)
            loss.backward()
            for name in eligible:
                gradient = modules[name].weight.grad
                if gradient is not None:
                    totals[name] += float(gradient.detach().abs().mean().item())
                    counts[name] += 1
            print(
                f"[daptq][probe] {model_family} {index + 1}/{protocol['da_mpa']['probe_steps']} "
                f"loss={float(loss.detach()):.6e}",
                flush=True,
            )
    finally:
        model.zero_grad(set_to_none=True)
        for parameter, required in states:
            parameter.requires_grad_(required)
    result = {name: totals[name] / max(1, counts[name]) for name in eligible}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("non-finite DA-MPA sensitivity")
    return result


def allocate_precision(
    inventory: list[dict[str, Any]],
    scores: dict[str, float],
    protocol: dict[str, Any],
) -> dict[str, int]:
    last_block = max(int(row["action_block"]) for row in inventory if row["action_block"] is not None)
    skip_from = last_block - int(protocol["da_mpa"]["skip_last_action_blocks"]) + 1
    eligible = [row["name"] for row in inventory if row["role"] == "action_mlp" and int(row["action_block"]) < skip_from]
    retain = int(math.ceil(len(eligible) * float(protocol["da_mpa"]["bf16_retention_ratio"])))
    high = set(sorted(eligible, key=lambda name: (-scores[name], name))[:retain])
    allocation: dict[str, int] = {}
    for row in inventory:
        name = row["name"]
        if row["role"] == "action_mlp" and int(row["action_block"]) >= skip_from:
            continue
        allocation[name] = 16 if name in high else 4
    if sum(bit == 16 for bit in allocation.values()) != retain:
        raise AssertionError("DA-MPA BF16 layer count drift")
    return allocation


class Moments:
    def __init__(self, width: int):
        self.count = 0
        self.sum = torch.zeros(width, dtype=torch.float64)
        self.cross = torch.zeros((width, width), dtype=torch.float64)

    def add(self, value: torch.Tensor) -> None:
        flat = value.detach().float().reshape(-1, value.shape[-1])[:4096].cpu().double()
        self.count += int(flat.shape[0])
        self.sum += flat.sum(dim=0)
        self.cross += flat.T @ flat

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.sum / max(1, self.count)
        covariance = self.cross / max(1, self.count) - torch.outer(mean, mean)
        return mean.float(), covariance.float()


def first_interface(inventory: list[dict[str, Any]], model_family: str) -> str:
    candidates = [row for row in inventory if row["role"] == "action_mlp" and row["action_block"] == 0]
    suffix = ".ff.net.2" if model_family == "gr00t" else ".mlp.down_proj"
    matches = [row["name"] for row in candidates if row["name"].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"cannot identify DA-PTQ interface layer: {matches}")
    return matches[0]


def make_rotated_weights(
    model: nn.Module,
    allocation: dict[str, int],
    interface: str,
    protocol: dict[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    modules = dict(model.named_modules())
    result: dict[str, dict[str, torch.Tensor]] = {}
    block = int(protocol["csrc"]["svd_block_size"])
    smoothing = float(protocol["csrc"]["smoothing"])
    for index, (name, bit) in enumerate(sorted(allocation.items())):
        module = modules[name]
        if bit == 4:
            permutation, rotations, output_rotations, rotated = make_block_rotations(
                module.weight, block_size=block, smoothing=smoothing
            )
            # CSRC is folded in the original output basis at the interface.
            # An identity output rotation there avoids a non-commuting fold.
            if name == interface:
                computed_output_rotations = output_rotations
                output_rotations = torch.eye(block).repeat(output_rotations.shape[0], 1, 1).to(
                    output_rotations.device
                )
                # Undo the output rotation applied by the generic helper.
                original_basis = rotated.clone()
                for output_index in range(int(computed_output_rotations.shape[0])):
                    start = output_index * block
                    stop = min((output_index + 1) * block, rotated.shape[0])
                    width = stop - start
                    if width > 1:
                        original_basis[start:stop] = (
                            computed_output_rotations[output_index, :width, :width].T
                            @ rotated[start:stop]
                        )
                rotated = original_basis
            result[name] = {
                "permutation": permutation.cpu(),
                "input_rotations": rotations.cpu(),
                "output_rotations": output_rotations.cpu(),
                "rotated_weight": rotated.cpu(),
            }
        print(f"[daptq][rotate] {index + 1}/{len(allocation)} {name} W{bit}", flush=True)
    return result


def run_forward(
    model_family: str,
    policy,
    frame: dict[str, Any],
    seed: int,
    protocol: dict[str, Any],
) -> None:
    seed_all(seed)
    prepared, actions = prepare(model_family, policy, frame, include_actions=True)
    with torch.no_grad():
        if model_family == "gr00t":
            # Use the same BF16-safe configured Beta sampling path as the
            # sensitivity probe; GR00T's native training forward samples its
            # cached Beta distribution in BF16 and is unsupported by torch.
            gr00t_weighted_loss(policy, prepared, protocol)
        else:
            policy._model(prepared, actions).mean()


def collect_activation_clips_and_fp_moments(
    model_family: str,
    policy,
    frames: list[dict[str, Any]],
    allocation: dict[str, int],
    transforms: dict[str, dict[str, torch.Tensor]],
    interface: str,
    protocol: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], Moments]:
    model = model_of(model_family, policy)
    modules = dict(model.named_modules())
    clips = {name: torch.zeros(modules[name].in_features) for name in allocation}
    moments = Moments(modules[interface].out_features)
    handles = []
    block = int(protocol["csrc"]["svd_block_size"])
    from daptq.core import apply_input_rotation

    for name in allocation:
        def prehook(_module, inputs, *, target=name):
            value = inputs[0].detach()
            if target in transforms:
                record = transforms[target]
                value = apply_input_rotation(
                    value,
                    record["permutation"].to(value.device),
                    record["input_rotations"].to(value.device),
                    block,
                )
            flat = value.abs().float().reshape(-1, value.shape[-1])[:4096]
            quantile = torch.quantile(flat, 0.999, dim=0).cpu().clamp_min_(1e-8)
            clips[target] = torch.maximum(clips[target], quantile)

        handles.append(modules[name].register_forward_pre_hook(prehook))
    handles.append(modules[interface].register_forward_hook(lambda _m, _i, output: moments.add(output)))
    try:
        batches = min(64, len(frames))
        for index, frame in enumerate(frames[:batches]):
            run_forward(model_family, policy, frame, 42 + index, protocol)
            print(f"[daptq][a8-fp] {model_family} {index + 1}/{batches}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if any(not bool(torch.isfinite(value).all()) or bool((value <= 0).any()) for value in clips.values()):
        raise ValueError("invalid DA-PTQ activation clips")
    return clips, moments


def initial_quantized_weight(
    module: nn.Linear,
    name: str,
    bit: int,
    transforms: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    if bit == 16:
        return module.weight.detach().cpu()
    codes, scales = mse_w4_quantize(transforms[name]["rotated_weight"].to(module.weight.device))
    return (codes.float() * scales[:, None]).cpu()


def rotated_bias(module: nn.Linear, transform: dict[str, torch.Tensor] | None, block: int) -> torch.Tensor | None:
    if module.bias is None:
        return None
    bias = module.bias.detach().float().cpu().clone()
    if transform is None:
        return bias
    rotations = transform["output_rotations"]
    for index in range(int(rotations.shape[0])):
        start, stop = index * block, min((index + 1) * block, bias.numel())
        width = stop - start
        if width > 1:
            bias[start:stop] = rotations[index, :width, :width] @ bias[start:stop]
    return bias


def collect_quantized_moments(
    model_family: str,
    policy,
    frames: list[dict[str, Any]],
    allocation: dict[str, int],
    transforms: dict[str, dict[str, torch.Tensor]],
    clips: dict[str, torch.Tensor],
    interface: str,
    protocol: dict[str, Any],
) -> Moments:
    model = model_of(model_family, policy)
    originals: dict[str, nn.Linear] = {}
    modules = dict(model.named_modules())
    block = int(protocol["csrc"]["svd_block_size"])
    from daptq.core import _parent_module

    for name, bit in allocation.items():
        source = modules[name]
        originals[name] = source
        transform = transforms.get(name)
        replacement = DAPTQLinear(
            source,
            weight=initial_quantized_weight(source, name, bit, transforms),
            bias=rotated_bias(source, transform, block),
            activation_clip=clips[name],
            permutation=transform["permutation"] if transform else None,
            input_rotations=transform["input_rotations"] if transform else None,
            output_rotations=transform["output_rotations"] if transform else None,
            block_size=block,
            weight_bits=bit,
            layer_name=name,
        )
        parent, child = _parent_module(model, name)
        parent._modules[child] = replacement
    moments = Moments(originals[interface].out_features)
    interface_module = model.get_submodule(interface)
    handle = interface_module.register_forward_hook(lambda _m, _i, output: moments.add(output))
    try:
        batches = min(64, len(frames))
        for index, frame in enumerate(frames[:batches]):
            run_forward(model_family, policy, frame, 42 + index, protocol)
            print(f"[daptq][a8-q] {model_family} {index + 1}/{batches}", flush=True)
    finally:
        handle.remove()
        for name, source in originals.items():
            parent, child = _parent_module(model, name)
            parent._modules[child] = source
    return moments


def fit_csrc(
    fp: Moments,
    quantized: Moments,
    protocol: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    mu_fp, cov_fp = fp.finalize()
    mu_q, cov_q = quantized.finalize()
    eps = 1e-5
    std_fp = torch.diag(cov_fp).clamp_min(eps).sqrt()
    std_q = torch.diag(cov_q).clamp_min(eps).sqrt()
    raw = (std_fp / std_q).clamp(0.5, 2.0)
    group = int(protocol["csrc"]["group_size"])
    shrink = float(protocol["csrc"]["group_shrinkage"])
    gain = raw.clone()
    for start in range(0, gain.numel(), group):
        stop = min(start + group, gain.numel())
        group_gain = raw[start:stop].mean()
        gain[start:stop] = 1.0 + shrink * (group_gain - 1.0)
    diagonal = torch.diag(gain)
    scaled_q = diagonal @ cov_q @ diagonal

    def matrix_power_psd(value: torch.Tensor, power: float) -> torch.Tensor:
        values, vectors = torch.linalg.eigh(value.double())
        values = values.clamp_min(eps).pow(power)
        return (vectors * values.unsqueeze(0)) @ vectors.T

    dense = matrix_power_psd(cov_fp, 0.5) @ matrix_power_psd(scaled_q, -0.5)
    delta = dense.float() - torch.eye(dense.shape[0])
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    rank = min(int(protocol["csrc"]["low_rank"]), delta.shape[0])
    low_rank = (u[:, :rank] * s[:rank]) @ vh[:rank]
    matrix = (torch.eye(delta.shape[0]) + low_rank) @ diagonal
    bias = mu_fp - matrix @ mu_q
    meta = {
        "fp_samples": fp.count,
        "quantized_samples": quantized.count,
        "rank": rank,
        "diagonal_gain_min": float(gain.min()),
        "diagonal_gain_max": float(gain.max()),
        "low_rank_singular_values": [float(value) for value in s[:rank]],
    }
    return matrix.float(), bias.float(), meta


def export_pack(
    args,
    policy,
    inventory: list[dict[str, Any]],
    allocation: dict[str, int],
    scores: dict[str, float],
    transforms: dict[str, dict[str, torch.Tensor]],
    clips: dict[str, torch.Tensor],
    interface: str,
    csrc_matrix: torch.Tensor,
    csrc_bias: torch.Tensor,
    csrc_meta: dict[str, Any],
    protocol: dict[str, Any],
    frozen: dict[str, Any],
    started: float,
) -> None:
    model = model_of(args.model_family, policy)
    modules = dict(model.named_modules())
    arrays: dict[str, np.ndarray] = {}
    records = []
    low_native_weight_bytes = 0
    low_packed_weight_bytes = 0
    for index, (name, bit) in enumerate(sorted(allocation.items())):
        module = modules[name]
        prefix = f"layer_{index:04d}"
        arrays[f"{prefix}_activation_clip"] = clips[name].numpy().astype(np.float16)
        bias_override = None
        if bit == 4:
            weight = transforms[name]["rotated_weight"].to(module.weight.device)
            bias = rotated_bias(module, transforms[name], int(protocol["csrc"]["svd_block_size"]))
            bias_override = bias
            if name == interface:
                weight = (csrc_matrix.to(weight.device) @ weight.float()).to(weight.dtype)
                bias_value = (
                    torch.zeros(module.out_features)
                    if bias is None
                    else bias
                )
                bias_override = csrc_matrix @ bias_value + csrc_bias
            codes, scales = mse_w4_quantize(weight)
            packed = pack_signed_nibbles(codes)
            arrays[f"{prefix}_packed_w4"] = packed
            arrays[f"{prefix}_weight_scale"] = scales.cpu().numpy().astype(np.float16)
            arrays[f"{prefix}_permutation"] = transforms[name]["permutation"].numpy().astype(np.int32)
            arrays[f"{prefix}_input_rotations"] = transforms[name]["input_rotations"].numpy().astype(np.float16)
            arrays[f"{prefix}_output_rotations"] = transforms[name]["output_rotations"].numpy().astype(np.float16)
            low_native_weight_bytes += int(module.weight.numel()) * 2
            low_packed_weight_bytes += int(packed.nbytes + arrays[f"{prefix}_weight_scale"].nbytes)
        elif name == interface:
            weight = csrc_matrix.to(module.weight.device) @ module.weight.detach().float()
            arrays[f"{prefix}_fp_weight"] = weight.cpu().numpy().astype(np.float16)
            bias_value = (
                torch.zeros(module.out_features)
                if module.bias is None
                else module.bias.detach().float().cpu()
            )
            bias_override = csrc_matrix @ bias_value + csrc_bias
        if bias_override is not None:
            arrays[f"{prefix}_bias_override"] = bias_override.numpy().astype(np.float16)
        row = next(item for item in inventory if item["name"] == name)
        records.append(
            {
                **row,
                "array_prefix": prefix,
                "weight_bits": bit,
                "sensitivity": scores.get(name),
                "csrc_interface": name == interface,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output_dir / "pack_arrays.npz"
    temporary = args.output_dir / f".pack_arrays.{os.getpid()}.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, arrays_path)
    paper_static = TABLE1_FP16_BYTES[args.model_family] - low_native_weight_bytes + low_packed_weight_bytes
    manifest = {
        "schema_version": 1,
        "kind": "daptq_robocasa365_pack_v1",
        "method": "DA-PTQ",
        "model_family": args.model_family,
        "model_unit": frozen["model_unit"],
        "upstream_repository": protocol["source"]["repository"],
        "upstream_commit": DAPTQ_GIT_COMMIT,
        "upstream_license_declared": False,
        "implementation_policy": protocol["source"]["implementation_policy"],
        "protocol": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "frozen_manifest": str(args.frozen_manifest.resolve()),
        "frozen_manifest_sha256": sha256_file(args.frozen_manifest),
        "calibration_source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "test_results_used": False,
        "flow_steps": 4,
        "trajectory_count": len(frozen["episodes"]),
        "arrays_file": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
        "layers": records,
        "allocation": {
            "w4_layers": sum(value == 4 for value in allocation.values()),
            "bf16_layers": sum(value == 16 for value in allocation.values()),
            "untouched_final_action_blocks": int(protocol["da_mpa"]["skip_last_action_blocks"]),
            "sensitivity_sha256": canonical_hash(scores),
        },
        "csrc": {"interface_layer": interface, **csrc_meta},
        "hyperparameters": {
            "probe_steps": int(protocol["da_mpa"]["probe_steps"]),
            "bf16_retention_ratio": float(protocol["da_mpa"]["bf16_retention_ratio"]),
            "svd_block_size": int(protocol["csrc"]["svd_block_size"]),
            "smoothing": float(protocol["csrc"]["smoothing"]),
            "group_size": int(protocol["csrc"]["group_size"]),
            "group_shrinkage": float(protocol["csrc"]["group_shrinkage"]),
            "activation_bits": 8,
            "activation_percentile": 99.9,
            "activation_calibration_batches": 64,
        },
        "storage": {
            "scope": "table1_linear_weights_and_biases_theoretical_tight_pack",
            "fp16_baseline_bytes": TABLE1_FP16_BYTES[args.model_family],
            "replaced_fp16_weight_bytes": low_native_weight_bytes,
            "packed_w4_and_scale_bytes": low_packed_weight_bytes,
            "total_static_bytes": paper_static,
            "compression_ratio": TABLE1_FP16_BYTES[args.model_family] / paper_static,
            "npz_file_bytes_not_used_for_table": arrays_path.stat().st_size,
            "activation_scales_and_rotation_metadata_excluded_like_existing_table1_rows": True
        },
        "wall_seconds": time.time() - started,
    }
    manifest["semantic_sha256"] = canonical_hash({key: value for key, value in manifest.items() if key != "wall_seconds"})
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-family", choices=("gr00t", "pi05"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.time()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    frozen = load_frozen(args.frozen_manifest)
    if frozen["model"] != args.model_family:
        raise ValueError("DA-PTQ frozen manifest/model mismatch")
    seed_all(42)
    policy = load_policy(args.model_family, args.checkpoint, args.device)
    model = model_of(args.model_family, policy)
    model.eval()
    inventory = build_target_inventory(model, args.model_family)
    materialize = max(
        int(protocol["da_mpa"]["probe_steps"]),
        int(protocol["quantization"]["activation_calibration_batches"]),
    )
    frames = spatially_interleaved_frames(
        frozen,
        int(protocol["calibration"]["spatial_bins"]),
        materialize=materialize,
    )
    scores = profile_sensitivity(args.model_family, policy, frames, inventory, protocol)
    allocation = allocate_precision(inventory, scores, protocol)
    interface = first_interface(inventory, args.model_family)
    transforms = make_rotated_weights(model, allocation, interface, protocol)
    clips, fp_moments = collect_activation_clips_and_fp_moments(
        args.model_family, policy, frames, allocation, transforms, interface, protocol
    )
    q_moments = collect_quantized_moments(
        args.model_family, policy, frames, allocation, transforms, clips, interface, protocol
    )
    csrc_matrix, csrc_bias, csrc_meta = fit_csrc(fp_moments, q_moments, protocol)
    export_pack(
        args,
        policy,
        inventory,
        allocation,
        scores,
        transforms,
        clips,
        interface,
        csrc_matrix,
        csrc_bias,
        csrc_meta,
        protocol,
        frozen,
        started,
    )
    del policy
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
