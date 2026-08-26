#!/usr/bin/env python3
"""Shared Hessian-aware group-64 W4 and deterministic per-step A8 utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from quantvla_cross_model_protocol import PROTOCOL, PROTOCOL_SHA256


CONFIG = PROTOCOL["hessian_w4a8"]
GROUP_SIZE = int(PROTOCOL["deployment"]["block_in"])
CLIP_RATIOS = tuple(float(value) for value in CONFIG["clipping_ratios"])


@dataclass(frozen=True)
class HessianW4Result:
    codes: torch.Tensor
    scales: torch.Tensor
    clipping: torch.Tensor
    reconstruction_error: torch.Tensor
    group_size: int

    @property
    def dequantized(self) -> torch.Tensor:
        expanded = self.scales.repeat_interleave(self.group_size, dim=1)
        return self.codes.to(expanded.dtype) * expanded[:, : self.codes.shape[1]]


def _second_order(inputs: torch.Tensor, damping: float) -> torch.Tensor:
    value = inputs.detach().reshape(-1, inputs.shape[-1])
    hessian = value.T @ value / max(value.shape[0], 1)
    diagonal_mean = torch.diagonal(hessian).mean().clamp_min(1e-12)
    hessian.diagonal().add_(float(damping) * diagonal_mean)
    return hessian


def _gptq_group(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential GPTQ-style feedback for every output row in one group."""
    try:
        inverse = torch.linalg.inv(hessian)
        feedback = torch.linalg.cholesky(inverse, upper=True)
    except torch.linalg.LinAlgError:
        inverse = torch.linalg.pinv(hessian)
        feedback = torch.linalg.cholesky(
            inverse + torch.eye(inverse.shape[0], dtype=inverse.dtype) * 1e-8,
            upper=True,
        )
    working = weight.clone()
    codes = torch.empty_like(working, dtype=torch.int8)
    dequant = torch.empty_like(working)
    for column in range(working.shape[1]):
        quantized = torch.clamp(
            torch.round(working[:, column] / scale), -8, 7
        ).to(torch.int8)
        reconstructed = quantized.to(working.dtype) * scale
        codes[:, column] = quantized
        dequant[:, column] = reconstructed
        denominator = feedback[column, column].abs().clamp_min(1e-12)
        error = (working[:, column] - reconstructed) / denominator
        if column + 1 < working.shape[1]:
            working[:, column + 1 :] -= (
                error[:, None] * feedback[column, column + 1 :][None, :]
            )
    return codes, dequant


def hessian_aware_w4(
    weight: Any,
    fp16_inputs: Any,
    *,
    group_size: int = GROUP_SIZE,
    clipping_ratios: Sequence[float] = CLIP_RATIOS,
    damping: float = 0.01,
    device: str | torch.device | None = None,
) -> HessianW4Result:
    """Search clipping and apply error feedback independently per group/row."""
    compute_device = torch.device(device or "cpu")
    compute_dtype = torch.float32 if compute_device.type == "cuda" else torch.float64
    matrix = torch.as_tensor(weight).detach().to(
        device=compute_device, dtype=compute_dtype
    )
    inputs = torch.as_tensor(fp16_inputs).detach().to(
        device=compute_device, dtype=compute_dtype
    )
    if matrix.ndim != 2 or inputs.ndim < 2 or inputs.shape[-1] != matrix.shape[1]:
        raise ValueError(f"W/X shape mismatch: {matrix.shape}/{inputs.shape}")
    if group_size != GROUP_SIZE:
        raise ValueError(f"v3 freezes group_size={GROUP_SIZE}")
    ratios = sorted({float(value) for value in clipping_ratios}, reverse=True)
    if tuple(sorted(ratios)) != tuple(sorted(CLIP_RATIOS)):
        raise ValueError(f"v3 freezes clipping search {CLIP_RATIOS}")
    if any(not 0.0 < value <= 1.0 for value in ratios):
        raise ValueError("clipping ratios must be in (0,1]")
    flat_inputs = inputs.reshape(-1, inputs.shape[-1])
    out_features, in_features = matrix.shape
    groups = (in_features + group_size - 1) // group_size
    all_codes = torch.empty_like(matrix, dtype=torch.int8)
    all_scales = torch.empty(
        (out_features, groups), dtype=compute_dtype, device=compute_device
    )
    all_clipping = torch.empty_like(all_scales)
    all_error = torch.empty_like(all_scales)

    for group in range(groups):
        start = group * group_size
        stop = min(start + group_size, in_features)
        block = matrix[:, start:stop]
        hessian = _second_order(flat_inputs[:, start:stop], damping)
        best_error = torch.full(
            (out_features,),
            float("inf"),
            dtype=compute_dtype,
            device=compute_device,
        )
        best_codes = torch.empty_like(block, dtype=torch.int8)
        best_scales = torch.empty(
            out_features, dtype=compute_dtype, device=compute_device
        )
        best_ratios = torch.empty_like(best_scales)
        for ratio in ratios:
            scale = (block.abs().amax(dim=1) * ratio / 7.0).clamp_min(1e-12)
            codes, dequant = _gptq_group(block, hessian, scale)
            error = dequant - block
            reconstruction = torch.einsum("oi,ij,oj->o", error, hessian, error)
            # Ratios run identity-first, so strict improvement implements the
            # frozen tie-break: less clipping / closer to identity.
            improved = reconstruction < best_error - 1e-15
            best_error[improved] = reconstruction[improved]
            best_codes[improved] = codes[improved]
            best_scales[improved] = scale[improved]
            best_ratios[improved] = ratio
        all_codes[:, start:stop] = best_codes
        all_scales[:, group] = best_scales
        all_clipping[:, group] = best_ratios
        all_error[:, group] = best_error
    return HessianW4Result(
        codes=all_codes,
        scales=all_scales.to(torch.float32),
        clipping=all_clipping.to(torch.float32),
        reconstruction_error=all_error.to(torch.float32),
        group_size=group_size,
    )


def pack_signed_nibbles(codes: Any) -> torch.Tensor:
    value = torch.as_tensor(codes).to(torch.int8)
    if value.ndim != 2 or bool((value < -8).any()) or bool((value > 7).any()):
        raise ValueError("signed W4 codes must be a 2-D tensor in [-8,7]")
    if value.shape[1] & 1:
        value = torch.nn.functional.pad(value, (0, 1))
    unsigned = torch.bitwise_and(value, 0x0F).to(torch.uint8)
    return torch.bitwise_or(unsigned[:, 0::2], unsigned[:, 1::2] << 4).contiguous()


def unpack_signed_nibbles(packed: Any, in_features: int) -> torch.Tensor:
    value = torch.as_tensor(packed, dtype=torch.uint8)
    low = (value & 0x0F).to(torch.int16)
    high = ((value >> 4) & 0x0F).to(torch.int16)
    codes = torch.stack((low, high), dim=-1).reshape(value.shape[0], -1)[:, :in_features]
    return torch.where(codes >= 8, codes - 16, codes).to(torch.int8)


def a8_scale_table(
    activations: Any,
    *,
    flow_steps: int | None = None,
    percentile: float | None = None,
) -> torch.Tensor:
    """One prefix/LLM table or four deterministic DiT flow-step tables."""
    value = torch.as_tensor(activations).detach().to(torch.float32).abs()
    percentile = (
        float(PROTOCOL["deployment"]["activation_percentile"])
        if percentile is None
        else float(percentile)
    )
    quantile = percentile / 100.0
    if flow_steps is None:
        flat = value.reshape(-1, value.shape[-1])
        return torch.quantile(flat, quantile, dim=0).clamp_min(1e-6) / 127.0
    if int(flow_steps) != int(PROTOCOL["closed_loop"]["flow_steps"]):
        raise ValueError("v3 DiT A8 requires exactly four flow-step tables")
    if value.shape[0] != flow_steps:
        raise ValueError(f"step activation axis {value.shape[0]} != {flow_steps}")
    return torch.stack(
        [
            torch.quantile(value[step].reshape(-1, value.shape[-1]), quantile, dim=0)
            .clamp_min(1e-6)
            / 127.0
            for step in range(flow_steps)
        ]
    )


def artifact_arrays(result: HessianW4Result) -> dict[str, np.ndarray]:
    return {
        "packed_u4": pack_signed_nibbles(result.codes).cpu().numpy(),
        "scales": result.scales.detach().cpu().numpy(),
        "clipping": result.clipping.detach().cpu().numpy(),
        "reconstruction_error": result.reconstruction_error.detach().cpu().numpy(),
    }


def selftest() -> None:
    generator = torch.Generator().manual_seed(19)
    inputs = torch.randn(128, 64, generator=generator)
    weight = torch.randn(12, 64, generator=generator) * 0.1
    result = hessian_aware_w4(weight, inputs)
    packed = pack_signed_nibbles(result.codes)
    torch.testing.assert_close(unpack_signed_nibbles(packed, 64), result.codes)
    assert result.scales.shape == (12, 1)
    assert packed.numel() == weight.numel() // 2
    print(f"[quantvla-hessian-w4] selftest OK {PROTOCOL_SHA256}")


if __name__ == "__main__":
    selftest()
