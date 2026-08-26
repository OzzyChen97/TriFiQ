#!/usr/bin/env python3
"""pi0.5 model adapter for the shared QuantVLA functional metrics.

No metric formula lives here. The adapter only converts pi0.5's native
``(50, 32)`` solver trajectory to the shared executed ``(16, 12)`` action
space; GR00T and pi0.5 then call the same metric functions.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from quantvla_cross_model_protocol import PROTOCOL_SHA256, adapter_attestation
from quantvla_metric_protocol import (
    ACTION_DIM,
    ACTION_HORIZON as EXECUTED_ACTIONS,
    FUNCTIONAL_FORMULA_ID,
    GAMMA,
    PAC_FORMULA_ID,
    d_func as canonical_d_func,
    d_pac_sequence as canonical_d_pac_sequence,
)
from quantvla_model_adapters import FLOW_STEPS, canonical_trajectory


ACTION_HORIZON = 50


def adapt_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
    return canonical_trajectory(trajectory, model="pi05")


def _adapter() -> dict[str, Any]:
    return {
        **adapter_attestation("pi05", native_action_horizon=ACTION_HORIZON),
        "source_action_dimension": 32,
        "trajectory_conversion": "prefix_16_actions_prefix_12_dimensions",
    }


def d_func(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    gamma: float = GAMMA,
) -> dict[str, Any]:
    result = canonical_d_func(
        adapt_trajectory(reference),
        adapt_trajectory(candidate),
        gamma=gamma,
    )
    result["adapter"] = _adapter()
    return result


def d_pac_sequence(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    replan_indices: Sequence[int],
    *,
    overlap_weight: float = 0.0,
    gamma: float = GAMMA,
) -> dict[str, Any]:
    # Retain the old keyword only to turn historical pi-only configurations
    # into an explicit error instead of silently changing their meaning.
    if float(overlap_weight) != 0.0:
        raise ValueError(
            "pi0.5-only forecast overlap is forbidden by the adapter-only protocol"
        )
    result = canonical_d_pac_sequence(
        adapt_trajectory(reference),
        adapt_trajectory(candidate),
        replan_indices,
        gamma=gamma,
    )
    result["adapter"] = _adapter()
    return result


def selftest() -> None:
    generator = torch.Generator().manual_seed(0)
    reference = torch.randn(
        FLOW_STEPS + 1, 8, ACTION_HORIZON, 32, generator=generator
    )
    same = d_func(reference, reference)
    assert same["d_func"] == 0.0
    assert same["formula_id"] == FUNCTIONAL_FORMULA_ID
    assert same["protocol_sha256"] == PROTOCOL_SHA256

    candidate = reference.clone()
    candidate[..., :EXECUTED_ACTIONS, :ACTION_DIM] += 0.1
    assert d_func(reference, candidate)["d_func"] > 0.0

    excluded = reference.clone()
    excluded[..., EXECUTED_ACTIONS:, :] += 100.0
    excluded[..., :, ACTION_DIM:] += 100.0
    assert d_func(reference, excluded)["d_func"] == 0.0

    pac = d_pac_sequence(reference[:, :4], candidate[:, :4], range(4))
    assert pac["d_pac_sequence"] > 0.0
    assert pac["weights"]["overlap"] == 0.0
    assert pac["formula_id"] == PAC_FORMULA_ID
    try:
        d_pac_sequence(reference[:, :4], candidate[:, :4], range(4), overlap_weight=0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("nonzero pi0.5 overlap must fail")
    print("[pi05_func_metrics] selftest OK (adapter-only, shared formulas)")


if __name__ == "__main__":
    selftest()
