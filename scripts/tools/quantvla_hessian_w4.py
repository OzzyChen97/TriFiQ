#!/usr/bin/env python3
"""Hessian-aware group-64 W4 quantization and signed-nibble packing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from dypac_vla_protocol import GROUP_SIZE, PROTOCOL_SHA256


CLIP_RATIOS = (0.85, 0.9, 0.95, 0.975, 1.0)


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


def _quantize_group(
    weight: torch.Tensor, hessian: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential Hessian-aware error feedback for one input-channel group."""
    try:
        inverse = torch.linalg.inv(hessian)
        feedback = torch.linalg.cholesky(inverse, upper=True)
    except torch.linalg.LinAlgError:
        inverse = torch.linalg.pinv(hessian)
        feedback = torch.linalg.cholesky(
            inverse + torch.eye(inverse.shape[0], dtype=inverse.dtype, device=inverse.device) * 1e-8,
            upper=True,
        )
    working = weight.clone()
    codes = torch.empty_like(working, dtype=torch.int8)
    dequantized = torch.empty_like(working)
    for column in range(working.shape[1]):
        quantized = torch.round(working[:, column] / scale).clamp(-8, 7).to(torch.int8)
        reconstructed = quantized.to(working.dtype) * scale
        codes[:, column] = quantized
        dequantized[:, column] = reconstructed
        denominator = feedback[column, column].abs().clamp_min(1e-12)
        error = (working[:, column] - reconstructed) / denominator
        if column + 1 < working.shape[1]:
            working[:, column + 1 :] -= error[:, None] * feedback[column, column + 1 :][None, :]
    return codes, dequantized


def hessian_aware_w4(
    weight: Any,
    reference_inputs: Any,
    *,
    group_size: int = GROUP_SIZE,
    clipping_ratios: Sequence[float] = CLIP_RATIOS,
    damping: float = 0.01,
    device: str | torch.device | None = None,
) -> HessianW4Result:
    """Search clipping and quantize every output row with group-64 feedback."""
    compute_device = torch.device(device or "cpu")
    compute_dtype = torch.float32 if compute_device.type == "cuda" else torch.float64
    matrix = torch.as_tensor(weight).detach().to(device=compute_device, dtype=compute_dtype)
    inputs = torch.as_tensor(reference_inputs).detach().to(device=compute_device, dtype=compute_dtype)
    if matrix.ndim != 2 or inputs.ndim < 2 or inputs.shape[-1] != matrix.shape[1]:
        raise ValueError(f"weight/input shape mismatch: {matrix.shape}/{inputs.shape}")
    if group_size != GROUP_SIZE:
        raise ValueError(f"DyPAC-VLA fixes W4 group size to {GROUP_SIZE}")
    ratios = tuple(sorted({float(value) for value in clipping_ratios}, reverse=True))
    if tuple(sorted(ratios)) != tuple(sorted(CLIP_RATIOS)):
        raise ValueError(f"DyPAC-VLA fixes the clipping search to {CLIP_RATIOS}")
    if any(not 0.0 < value <= 1.0 for value in ratios):
        raise ValueError("clipping ratios must lie in (0,1]")

    flat_inputs = inputs.reshape(-1, inputs.shape[-1])
    out_features, in_features = matrix.shape
    groups = (in_features + group_size - 1) // group_size
    all_codes = torch.empty_like(matrix, dtype=torch.int8)
    all_scales = torch.empty((out_features, groups), dtype=compute_dtype, device=compute_device)
    all_clipping = torch.empty_like(all_scales)
    all_error = torch.empty_like(all_scales)

    for group in range(groups):
        start = group * group_size
        stop = min(start + group_size, in_features)
        block = matrix[:, start:stop]
        hessian = _second_order(flat_inputs[:, start:stop], damping)
        best_error = torch.full((out_features,), float("inf"), dtype=compute_dtype, device=compute_device)
        best_codes = torch.empty_like(block, dtype=torch.int8)
        best_scales = torch.empty(out_features, dtype=compute_dtype, device=compute_device)
        best_ratios = torch.empty_like(best_scales)
        for ratio in ratios:
            scale = (block.abs().amax(dim=1) * ratio / 7.0).clamp_min(1e-12)
            codes, dequantized = _quantize_group(block, hessian, scale)
            error = dequantized - block
            reconstruction = torch.einsum("oi,ij,oj->o", error, hessian, error)
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
    assert result.group_size == 64
    assert result.scales.shape == (12, 1)
    assert result.scales.dtype == torch.float32
    assert packed.numel() == weight.numel() // 2
    print(f"[quantvla-hessian-w4] selftest OK {PROTOCOL_SHA256}")


if __name__ == "__main__":
    selftest()