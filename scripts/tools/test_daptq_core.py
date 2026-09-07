#!/usr/bin/env python3
"""Numerical regression tests for the clean-room DA-PTQ primitives."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from daptq.core import (  # noqa: E402
    DAPTQLinear,
    apply_input_rotation,
    make_block_rotations,
    mse_w4_quantize,
    pack_signed_nibbles,
    unpack_signed_nibbles,
)


class DAPTQCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_signed_nibble_roundtrip_odd_width(self) -> None:
        codes = torch.randint(-8, 8, (9, 17), dtype=torch.int8)
        packed = pack_signed_nibbles(codes)
        restored = unpack_signed_nibbles(packed, codes.shape[1])
        np.testing.assert_array_equal(restored, codes.numpy())
        self.assertEqual(packed.shape, (9, 9))

    def test_input_and_output_rotations_preserve_linear_map(self) -> None:
        weight = torch.randn(19, 17)
        bias = torch.randn(19)
        value = torch.randn(5, 3, 17)
        permutation, input_rotations, output_rotations, rotated_weight = make_block_rotations(
            weight, block_size=8, smoothing=0.15
        )
        rotated_input = apply_input_rotation(
            value, permutation, input_rotations, block_size=8
        )
        rotated_bias = bias.clone()
        for index in range(output_rotations.shape[0]):
            start, stop = index * 8, min((index + 1) * 8, bias.numel())
            width = stop - start
            if width > 1:
                rotated_bias[start:stop] = (
                    output_rotations[index, :width, :width] @ bias[start:stop]
                )
        result = F.linear(rotated_input, rotated_weight, rotated_bias)
        for index in range(output_rotations.shape[0]):
            start, stop = index * 8, min((index + 1) * 8, result.shape[-1])
            width = stop - start
            if width > 1:
                result[..., start:stop] = (
                    result[..., start:stop]
                    @ output_rotations[index, :width, :width]
                )
        expected = F.linear(value, weight, bias)
        torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)

    def test_fake_quant_wrapper_matches_explicit_computation(self) -> None:
        source = torch.nn.Linear(13, 11, bias=True)
        codes, scales = mse_w4_quantize(source.weight)
        dequantized = codes.float() * scales[:, None]
        clip = torch.linspace(0.5, 1.7, source.in_features)
        wrapped = DAPTQLinear(
            source,
            weight=dequantized,
            bias=source.bias,
            activation_clip=clip,
            permutation=None,
            input_rotations=None,
            output_rotations=None,
            block_size=8,
            weight_bits=4,
            layer_name="toy",
        )
        value = torch.randn(4, 13)
        scale = clip / 127.0
        quantized = torch.round(value / scale).clamp(-128, 127) * scale
        expected = F.linear(quantized, dequantized, source.bias)
        torch.testing.assert_close(wrapped(value), expected)
        self.assertFalse(wrapped.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
