#!/usr/bin/env python3
"""Selector-free, per-forward, input-channel DyRange-A8."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from dypac_vla_protocol import PROTOCOL, protocol_attestation


QMIN, QMAX = (int(value) for value in PROTOCOL["dyrange_a8"]["signed_code_range"])
SCALE_FLOOR = 1e-6


def dyrange_scale(current_input: Any) -> torch.Tensor:
    """Compute one scale per input channel from the current forward input only."""
    value = torch.as_tensor(current_input)
    if value.ndim < 1 or value.shape[-1] == 0 or not torch.isfinite(value).all():
        raise ValueError("DyRange-A8 input must be finite with a final channel axis")
    compute = value.detach().to(dtype=torch.float32)
    reduction_axes = tuple(range(compute.ndim - 1))
    maxima = compute.abs().amax(dim=reduction_axes) if reduction_axes else compute.abs()
    scale = (maxima / float(QMAX)).clamp_min(SCALE_FLOOR)
    output_dtype = value.dtype if value.is_floating_point() else torch.float32
    return scale.to(dtype=output_dtype, device=value.device)


def dyrange_a8(current_input: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize/dequantize one current activation tensor with fresh channel scales."""
    value = torch.as_tensor(current_input)
    if not value.is_floating_point():
        value = value.to(torch.float32)
    scale = dyrange_scale(value)
    codes = torch.round(value / scale).clamp(QMIN, QMAX).to(torch.int8)
    dequantized = codes.to(value.dtype) * scale
    return dequantized, scale, codes


def validate_runtime(contract: Mapping[str, Any], *, source: str) -> None:
    expected = {
        "activation_bits": 8,
        "range_source": "current_forward_input",
        "granularity": "input_channel",
        "runtime_selector": False,
        "history_or_ema": False,
        "calibration_table": False,
    }
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{source}: invalid DyRange-A8 runtime contract: {mismatches}")


def protocol_runtime_contract() -> dict[str, Any]:
    return {
        "activation_bits": 8,
        "range_source": "current_forward_input",
        "granularity": "input_channel",
        "runtime_selector": False,
        "history_or_ema": False,
        "calibration_table": False,
        "dypac_vla_protocol": protocol_attestation(),
    }


def selftest() -> None:
    first = torch.tensor(
        [[[1.0, 10.0], [-2.0, 5.0]], [[0.5, -4.0], [0.0, 8.0]]],
        dtype=torch.float32,
    )
    second = torch.tensor([[8.0, 1.0], [-4.0, -2.0]], dtype=torch.float32)
    _, first_scale, first_codes = dyrange_a8(first)
    _, second_scale, _ = dyrange_a8(second)
    torch.testing.assert_close(first_scale, torch.tensor([2.0 / 127.0, 10.0 / 127.0]))
    torch.testing.assert_close(second_scale, torch.tensor([8.0 / 127.0, 2.0 / 127.0]))
    assert first_codes.shape == first.shape
    assert not torch.equal(first_scale, second_scale)

    # A later call has no effect on recomputing the first input's scale.
    _, repeated_scale, _ = dyrange_a8(first)
    torch.testing.assert_close(repeated_scale, first_scale)
    contract = protocol_runtime_contract()
    validate_runtime(contract, source="selftest")
    print("[quantvla-dyrange-a8] selftest OK")


if __name__ == "__main__":
    selftest()