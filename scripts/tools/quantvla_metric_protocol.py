#!/usr/bin/env python3
"""Shared physical-action metrics for the v3 GR00T/pi0.5 protocol.

This module deliberately accepts only final, inverse-normalized action chunks.
Native solver states remain adapter-local diagnostics and cannot enter either
selection metric.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gr00t_func_metrics import _se3_exp, _se3_log, d_func as _legacy_d_func
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


def canonical_physical_actions(value: Any, name: str = "actions") -> torch.Tensor:
    """Validate ``(..., sequence, 16, 12)`` physical final chunks."""
    tensor = torch.as_tensor(value).detach().to(dtype=torch.float32, device="cpu")
    if tensor.ndim < 3:
        raise ValueError(f"{name} must end in (sequence,16,12), got {tensor.shape}")
    if tuple(tensor.shape[-2:]) != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(
            f"{name} is outside canonical physical action space: "
            f"{tuple(tensor.shape[-2:])} != {(ACTION_HORIZON, ACTION_DIM)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor.contiguous()


def physical_action_scale(fp16_selection_actions: Any) -> torch.Tensor:
    """One global FP16-only robust scale, never re-estimated per sequence."""
    teacher = canonical_physical_actions(fp16_selection_actions, "fp16_selection_actions")
    flat = teacher.reshape(-1, ACTION_DIM).to(torch.float64)
    median = flat.median(dim=0).values
    mad = (flat - median).abs().median(dim=0).values
    rms = flat.square().mean(dim=0).sqrt()
    robust = torch.maximum(1.4826 * mad, 0.1 * rms)
    shared_floor = 0.05 * robust.median()
    scale = torch.maximum(robust, shared_floor).clamp_min(1e-4)
    if not torch.isfinite(scale).all() or bool((scale <= 0).any()):
        raise ValueError("invalid global physical action scale")
    return scale.to(torch.float32)


def charbonnier(value: torch.Tensor) -> torch.Tensor:
    """Robust penalty frozen by D_PAC-v2; exactly zero at the origin."""
    value64 = value.to(torch.float64)
    return (2.0 * torch.sqrt(1.0 + value64.square()) - 2.0).to(torch.float32)


def _mean_rho(value: torch.Tensor) -> float:
    return float(charbonnier(value).mean()) if value.numel() else 0.0


def _ordered_chunks(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    replans: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError(
            f"paired sequence chunks must share (R,16,12), got "
            f"{reference.shape} and {candidate.shape}"
        )
    indices = [int(value) for value in replans]
    if len(indices) != reference.shape[0] or len(indices) != len(set(indices)):
        raise ValueError("replan indices must be unique and match the sequence length")
    order = torch.tensor(sorted(range(len(indices)), key=indices.__getitem__), dtype=torch.long)
    return (
        reference.index_select(0, order),
        candidate.index_select(0, order),
        [indices[index] for index in order.tolist()],
    )


def _prefix_loss(error: torch.Tensor) -> float:
    n = torch.arange(1, error.shape[0] + 1, dtype=error.dtype).sqrt().unsqueeze(1)
    return _mean_rho(error.cumsum(dim=0) / n)


def _pose_loss(reference: torch.Tensor, candidate: torch.Tensor, scale: torch.Tensor) -> float:
    if torch.equal(reference[:, :6], candidate[:, :6]):
        return 0.0
    reference_pose = torch.eye(4, dtype=torch.float32)
    candidate_pose = torch.eye(4, dtype=torch.float32)
    relative_logs = []
    for index, (teacher_delta, quant_delta) in enumerate(zip(reference, candidate), start=1):
        reference_pose = reference_pose @ _se3_exp(teacher_delta[:6])
        candidate_pose = candidate_pose @ _se3_exp(quant_delta[:6])
        relative = torch.linalg.solve(reference_pose, candidate_pose)
        relative_logs.append(_se3_log(relative) / scale[:6] / np.sqrt(index))
    return _mean_rho(torch.stack(relative_logs))


def _stitch_loss(reference: torch.Tensor, candidate: torch.Tensor, scale: torch.Tensor) -> float:
    if reference.shape[0] < 2:
        return 0.0
    teacher_jump = reference[1:, 0] - reference[:-1, -1]
    quant_jump = candidate[1:, 0] - candidate[:-1, -1]
    return _mean_rho((quant_jump - teacher_jump) / scale)


def _gripper_loss(reference: torch.Tensor, candidate: torch.Tensor, scale: torch.Tensor) -> float:
    lo, hi = ACTION_LAYOUT["grip"]
    if lo >= reference.shape[-1] or hi <= lo:
        return 0.0
    teacher_state = torch.sigmoid(4.0 * reference[:, lo:hi] / scale[lo:hi])
    quant_state = torch.sigmoid(4.0 * candidate[:, lo:hi] / scale[lo:hi])
    state = _mean_rho(quant_state - teacher_state)
    if reference.shape[0] < 2:
        return state
    teacher_event = teacher_state[1:] - teacher_state[:-1]
    quant_event = quant_state[1:] - quant_state[:-1]
    event = _mean_rho(quant_event - teacher_event)
    return 0.5 * (state + event)


def _physical_event_grip(
    reference: torch.Tensor, candidate: torch.Tensor, scale: torch.Tensor
) -> float:
    """Grip safety from binary physical events instead of soft sigmoid drift.

    Used as the fallback when the soft gripper component is identically zero
    across the expanded buffer. Terms: binary state disagreement, relative
    close/open transition count error, and relative first-close index error.
    """
    lo, hi = ACTION_LAYOUT["grip"]
    if lo >= reference.shape[-1] or hi <= lo or reference.shape[0] < 2:
        return 0.0
    teacher = torch.sigmoid(4.0 * reference[:, lo:hi] / scale[lo:hi])
    quant = torch.sigmoid(4.0 * candidate[:, lo:hi] / scale[lo:hi])
    teacher_bin = (teacher > 0.5).float()
    quant_bin = (quant > 0.5).float()

    def transitions(binary: torch.Tensor) -> float:
        return float((binary[1:] - binary[:-1]).clamp(min=0.0).sum().item())

    def first_close(binary: torch.Tensor) -> int:
        closed = binary.max(dim=-1).values
        nonzero = (closed > 0).nonzero(as_tuple=True)[0]
        return int(nonzero[0].item()) if len(nonzero) else int(binary.shape[0])

    teacher_transitions = transitions(teacher_bin)
    quant_transitions = transitions(quant_bin)
    disagreement = float((teacher_bin != quant_bin).float().mean().item())
    transitions_error = abs(quant_transitions - teacher_transitions) / max(
        teacher_transitions, 1.0
    )
    index_error = abs(first_close(quant_bin) - first_close(teacher_bin)) / float(
        max(reference.shape[0], 1)
    )
    return (disagreement + transitions_error + index_error) / 3.0


def d_func(reference: Any, candidate: Any, gamma: float = GAMMA) -> dict[str, Any]:
    """The frozen old local selector, now fed the same physical final chunks."""
    if float(gamma) != GAMMA:
        raise ValueError(f"canonical D_func freezes gamma={GAMMA}, got {gamma}")
    ref = canonical_physical_actions(reference, "reference")
    quant = canonical_physical_actions(candidate, "candidate")
    if ref.shape != quant.shape:
        raise ValueError(f"paired physical action shape mismatch: {ref.shape} != {quant.shape}")
    flat_ref = ref.reshape(-1, ACTION_HORIZON, ACTION_DIM).unsqueeze(0)
    flat_quant = quant.reshape(-1, ACTION_HORIZON, ACTION_DIM).unsqueeze(0)
    result = _legacy_d_func(
        flat_ref,
        flat_quant,
        gamma=GAMMA,
        layout=ACTION_LAYOUT,
        weights=DFUNC_WEIGHTS,
    )
    result.update(
        {
            "formula_id": FUNCTIONAL_FORMULA_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "action_space": ACTION["space"],
            "canonical_action_shape": [ACTION_HORIZON, ACTION_DIM],
        }
    )
    return result


def d_pac_sequence(
    reference: Any,
    candidate: Any,
    replan_indices: Sequence[int],
    *,
    scale: Any,
    grip_mode: str = "soft",
) -> dict[str, Any]:
    """Compute the five equal physical components for one ordered sequence."""
    if grip_mode not in ("soft", "physical_events"):
        raise ValueError(f"unknown grip_mode: {grip_mode}")
    ref = canonical_physical_actions(reference, "reference")
    quant = canonical_physical_actions(candidate, "candidate")
    ref, quant, sorted_replans = _ordered_chunks(ref, quant, replan_indices)
    dimension_scale = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
    if dimension_scale.shape != (ACTION_DIM,) or bool((dimension_scale <= 0).any()):
        raise ValueError(f"scale must be a positive ({ACTION_DIM},) global vector")
    teacher_control = ref.reshape(-1, ACTION_DIM)
    quant_control = quant.reshape(-1, ACTION_DIM)
    normalized_error = (quant_control - teacher_control) / dimension_scale
    components = {
        "local": _mean_rho(normalized_error),
        "prefix": _prefix_loss(normalized_error),
        "pose": _pose_loss(teacher_control, quant_control, dimension_scale),
        "stitch": _stitch_loss(ref, quant, dimension_scale),
        "grip": (
            _physical_event_grip(teacher_control, quant_control, dimension_scale)
            if grip_mode == "physical_events"
            else _gripper_loss(teacher_control, quant_control, dimension_scale)
        ),
    }
    combined = sum(DPAC_WEIGHTS[name] * value for name, value in components.items())
    combined /= sum(DPAC_WEIGHTS.values())
    return {
        "d_pac_sequence": float(combined),
        **{f"d_{name}": float(value) for name, value in components.items()},
        "components": components,
        "weights": dict(DPAC_WEIGHTS),
        "dimension_scale": dimension_scale.tolist(),
        "replan_indices": sorted_replans,
        "n_replans": int(ref.shape[0]),
        "executed_actions": ACTION_HORIZON,
        "formula_id": PAC_FORMULA_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "action_space": ACTION["space"],
        "teacher": "original_native_fp16_checkpoint",
    }


def aggregate_d_pac_sequences(sequences: Sequence[Mapping[str, Any] | float]) -> dict[str, Any]:
    values = np.asarray(
        [
            float(item["d_pac_sequence"] if isinstance(item, Mapping) else item)
            for item in sequences
        ],
        dtype=np.float64,
    )
    alpha = float(DPAC["outer_cvar_alpha"])
    if values.size:
        tail_count = max(1, int(np.ceil((1.0 - alpha) * values.size)))
        cvar = float(np.sort(values)[-tail_count:].mean())
        mean = float(values.mean())
    else:
        cvar = mean = 0.0
    return {
        "d_pac": mean + float(DPAC["outer_cvar_weight"]) * cvar,
        "mean": mean,
        "cvar": cvar,
        "cvar_alpha": alpha,
        "tail_weight": float(DPAC["outer_cvar_weight"]),
        "n_sequences": int(values.size),
        "per_sequence": values.tolist(),
        "formula_id": PAC_FORMULA_ID,
        "protocol_sha256": PROTOCOL_SHA256,
    }


def summarize_pair(
    reference: Any,
    candidate: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    scale: Any | None = None,
    grip_mode: str = "soft",
) -> dict[str, Any]:
    """Apply the identical grouping, global scale and formulas to either model."""
    ref = canonical_physical_actions(reference, "reference")
    quant = canonical_physical_actions(candidate, "candidate")
    if ref.ndim != 3 or quant.shape != ref.shape:
        raise ValueError("summarize_pair expects paired (observation,16,12) chunks")
    if ref.shape[0] != len(records):
        raise ValueError(f"metric records={len(records)} but action rows={ref.shape[0]}")
    global_scale = physical_action_scale(ref) if scale is None else torch.as_tensor(scale)
    functional = d_func(ref, quant)

    groups: dict[tuple[str, int, int | None], list[int]] = defaultdict(list)
    has_replans = bool(records) and all("replan" in record for record in records)
    for index, record in enumerate(records):
        singleton = None if has_replans else index
        groups[(str(record.get("task")), int(record.get("seed", -1)), singleton)].append(index)

    # D_func remains the old local formula, but the paired one-SE procedure is
    # frozen at the same sequence unit as D_PAC-v2.  Re-evaluate the complete
    # frozen D_func formula inside every task/seed group; averaging only its
    # per-observation relative-MSE inputs would silently discard d_final,
    # d_kin, d_grip and the local CVaR term while still calling the selector
    # ``d_func_v1``.
    functional_sequences = []
    for (task, seed, _singleton), positions in sorted(groups.items()):
        selected = torch.tensor(positions, dtype=torch.long)
        sequence_summary = d_func(
            ref.index_select(0, selected),
            quant.index_select(0, selected),
        )
        functional_sequences.append(
            {
                "task": task,
                "seed": seed,
                "record_indices": positions,
                "d_func_sequence": float(sequence_summary["d_func"]),
                "components": {
                    "final": float(sequence_summary["d_final"]),
                    "kin": float(sequence_summary["d_kin"]),
                    "grip": float(sequence_summary["d_grip"]),
                    "tail_cvar90": float(sequence_summary["tail"]["cvar90"]),
                },
            }
        )
    functional["per_sequence"] = [
        row["d_func_sequence"] for row in functional_sequences
    ]
    functional["sequences"] = functional_sequences
    functional["selection_sample_unit"] = "paired_task_seed_sequence"

    sequences = []
    for (task, seed, _singleton), positions in sorted(groups.items()):
        selected = torch.tensor(positions, dtype=torch.long)
        replans = [int(records[index].get("replan", 0)) for index in positions]
        row = d_pac_sequence(
            ref.index_select(0, selected),
            quant.index_select(0, selected),
            replans,
            scale=global_scale,
            grip_mode=grip_mode,
        )
        sequences.append({"task": task, "seed": seed, "record_indices": positions, **row})
    pac = aggregate_d_pac_sequences(sequences)
    pac.update(
        {
            "teacher": "original_native_fp16_checkpoint",
            "sequence_grouping": DPAC["sequence_grouping"] if has_replans else "independent",
            "dimension_scale": torch.as_tensor(global_scale).tolist(),
            "sequences": sequences,
        }
    )
    return {"d_func_summary": functional, "d_pac_summary": pac}


def summarize_noise_a_b(
    teacher_a: Any,
    candidate_a: Any,
    teacher_b: Any,
    candidate_b: Any,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Selection noise A and held-out audit noise B using A's frozen FP16 scale."""
    scale = physical_action_scale(teacher_a)
    return {
        "selection_noise_A": summarize_pair(teacher_a, candidate_a, records, scale=scale),
        "heldout_noise_B": summarize_pair(teacher_b, candidate_b, records, scale=scale),
        "selection_frozen_before_noise_B": True,
        "dimension_scale_from": "FP16_noise_A_entire_selection_buffer",
    }


def selftest() -> None:
    generator = torch.Generator().manual_seed(4)
    value = torch.randn(8, ACTION_HORIZON, ACTION_DIM, generator=generator)
    records = [{"task": "t", "seed": index // 4, "replan": index % 4} for index in range(8)]
    same = summarize_pair(value, value, records)
    assert same["d_func_summary"]["d_func"] == 0.0
    assert same["d_pac_summary"]["d_pac"] == 0.0
    changed = value.clone()
    changed[..., 0] += 0.01
    result = summarize_pair(value, changed, records)
    assert result["d_func_summary"]["d_func"] > 0.0
    assert result["d_pac_summary"]["d_pac"] > 0.0
    print("[quantvla-metric-protocol] selftest OK")


if __name__ == "__main__":
    selftest()
