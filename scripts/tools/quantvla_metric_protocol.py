#!/usr/bin/env python3
"""Canonical metric implementation used after either model adapter."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gr00t_func_metrics import (
    aggregate_d_pac_sequences as _aggregate_d_pac_sequences,
    d_func as _d_func,
    d_pac_sequence as _d_pac_sequence,
)
from quantvla_cross_model_protocol import PROTOCOL, PROTOCOL_SHA256


ACTION = PROTOCOL["metrics"]["canonical_action"]
DFUNC = PROTOCOL["metrics"]["d_func"]
DPAC = PROTOCOL["metrics"]["d_pac"]
ACTION_HORIZON = int(ACTION["horizon"])
ACTION_DIM = int(ACTION["dimension"])
ACTION_LAYOUT = {
    "trans": tuple(ACTION["layout"]["translation"]),
    "rot": tuple(ACTION["layout"]["rotation"]),
    "grip": tuple(ACTION["layout"]["gripper"]),
}
DFUNC_WEIGHTS = {key: float(value) for key, value in DFUNC["weights"].items()}
DPAC_WEIGHTS = {key: float(value) for key, value in DPAC["weights"].items()}
GAMMA = float(DFUNC["gamma"])
FUNCTIONAL_FORMULA_ID = DFUNC["formula_id"]
PAC_FORMULA_ID = DPAC["formula_id"]


def _canonical(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().to(dtype=torch.float32, device="cpu")
    if tensor.ndim not in (3, 4):
        raise ValueError(f"{name} must be (R,H,D) or (T+1,R,H,D), got {tensor.shape}")
    if tuple(tensor.shape[-2:]) != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(
            f"{name} is outside canonical action space: {tuple(tensor.shape[-2:])} "
            f"!= {(ACTION_HORIZON, ACTION_DIM)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def d_func(reference: Any, candidate: Any, gamma: float = GAMMA) -> dict[str, Any]:
    if float(gamma) != GAMMA:
        raise ValueError(f"canonical D_func freezes gamma={GAMMA}, got {gamma}")
    ref = _canonical(reference, "reference")
    quant = _canonical(candidate, "candidate")
    if ref.shape != quant.shape:
        raise ValueError(f"paired trajectory shape mismatch: {ref.shape} != {quant.shape}")
    result = _d_func(
        ref,
        quant,
        gamma=GAMMA,
        layout=ACTION_LAYOUT,
        weights=DFUNC_WEIGHTS,
    )
    result.update(
        {
            "formula_id": FUNCTIONAL_FORMULA_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "canonical_action_shape": [ACTION_HORIZON, ACTION_DIM],
        }
    )
    return result


def d_pac_sequence(
    reference: Any,
    candidate: Any,
    replan_indices: Sequence[int],
    *,
    gamma: float = GAMMA,
) -> dict[str, Any]:
    if float(gamma) != GAMMA:
        raise ValueError(f"canonical D_PAC freezes gamma={GAMMA}, got {gamma}")
    ref = _canonical(reference, "reference")
    quant = _canonical(candidate, "candidate")
    result = _d_pac_sequence(
        ref,
        quant,
        replan_indices,
        executed_actions=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        layout=ACTION_LAYOUT,
        weights=DPAC_WEIGHTS,
        gamma=GAMMA,
        prefix_exponent=float(DPAC["prefix_exponent"]),
        scale_epsilon=float(DPAC["scale_epsilon"]),
        gripper_kappa=float(DPAC["gripper_kappa"]),
        gripper_delta_weight=float(DPAC["gripper_delta_weight"]),
    )
    result.update(
        {
            "formula_id": PAC_FORMULA_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "canonical_action_shape": [ACTION_HORIZON, ACTION_DIM],
        }
    )
    return result


def aggregate_d_pac_sequences(sequences: Sequence[Mapping[str, Any] | float]) -> dict[str, Any]:
    result = _aggregate_d_pac_sequences(
        sequences,
        tail_weight=float(DPAC["outer_cvar_weight"]),
        cvar_alpha=float(DPAC["outer_cvar_alpha"]),
    )
    result.update({"formula_id": PAC_FORMULA_ID, "protocol_sha256": PROTOCOL_SHA256})
    return result


def summarize_pair(
    reference: Any,
    candidate: Any,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply identical formulas, grouping and statistics to either model."""
    ref = _canonical(reference, "reference")
    quant = _canonical(candidate, "candidate")
    if ref.shape != quant.shape:
        raise ValueError(f"paired trajectory shape mismatch: {ref.shape} != {quant.shape}")
    batch_axis = 1 if ref.ndim == 4 else 0
    if ref.shape[batch_axis] != len(records):
        raise ValueError(f"metric records={len(records)} but trajectory rows={ref.shape[batch_axis]}")
    functional = d_func(ref, quant)

    groups: dict[tuple[str, int, int | None], list[int]] = defaultdict(list)
    has_replans = bool(records) and all("replan" in record for record in records)
    for index, record in enumerate(records):
        singleton = None if has_replans else index
        groups[(str(record.get("task")), int(record.get("seed", -1)), singleton)].append(index)

    sequences = []
    for (task, seed, _singleton), positions in sorted(groups.items()):
        replans = [int(records[index].get("replan", 0)) for index in positions]
        selected = torch.tensor(positions, dtype=torch.long)
        axis = 1 if ref.ndim == 4 else 0
        row = d_pac_sequence(
            ref.index_select(axis, selected),
            quant.index_select(axis, selected),
            replans,
        )
        sequences.append(
            {"task": task, "seed": seed, "record_indices": positions, **row}
        )
    pac = aggregate_d_pac_sequences(sequences)
    pac.update(
        {
            "teacher": "original_native_fp16_checkpoint",
            "sequence_grouping": (
                DPAC["sequence_grouping"]
                if has_replans else "independent_action_prefix"
            ),
            "sequences": sequences,
        }
    )
    return {"d_func_summary": functional, "d_pac_summary": pac}


def selftest() -> None:
    generator = torch.Generator().manual_seed(4)
    value = torch.randn(5, 4, ACTION_HORIZON, ACTION_DIM, generator=generator)
    records = [
        {"task": "t", "seed": 0, "replan": index}
        for index in range(value.shape[1])
    ]
    same = summarize_pair(value, value, records)
    assert same["d_func_summary"]["d_func"] == 0.0
    assert same["d_pac_summary"]["d_pac"] == 0.0
    changed = value.clone()
    changed[..., 0] += 0.01
    result = summarize_pair(value, changed, records)
    assert result["d_func_summary"]["d_func"] > 0.0
    assert result["d_pac_summary"]["d_pac"] > 0.0
    assert result["d_pac_summary"]["sequences"][0]["weights"]["overlap"] == 0.0
    print("[quantvla-metric-protocol] selftest OK")


if __name__ == "__main__":
    selftest()
