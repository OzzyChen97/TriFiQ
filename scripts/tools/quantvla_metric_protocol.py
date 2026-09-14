#!/usr/bin/env python3
"""D-PAC immediate and accumulated deviation on executed action prefixes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch


DEFAULT_EXECUTION_PREFIX = 16


def _actions(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().to(dtype=torch.float64, device="cpu")
    if tensor.ndim != 3 or tensor.shape[0] == 0 or tensor.shape[1] == 0 or tensor.shape[2] == 0:
        raise ValueError(f"{name} must have shape (replans, chunk_horizon, action_dim)")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor.contiguous()


def reference_action_scale(reference_selection_actions: Any) -> torch.Tensor:
    """Compute the one reference-derived coordinate scale shared by every mask."""
    tensor = torch.as_tensor(reference_selection_actions).detach().to(dtype=torch.float64, device="cpu")
    if tensor.ndim < 2 or tensor.shape[-1] == 0 or not torch.isfinite(tensor).all():
        raise ValueError("reference selection actions must be finite with a final coordinate axis")
    flat = tensor.reshape(-1, tensor.shape[-1])
    median = flat.median(dim=0).values
    mad = (flat - median).abs().median(dim=0).values
    rms = flat.square().mean(dim=0).sqrt()
    base = torch.maximum(1.4826 * mad, 0.1 * rms)
    shared_floor = 0.05 * base.median()
    scale = torch.maximum(base, shared_floor).clamp_min(1e-4)
    if bool((scale <= 0.0).any()) or not torch.isfinite(scale).all():
        raise ValueError("reference action scale is invalid")
    return scale


def charbonnier(value: torch.Tensor) -> torch.Tensor:
    value64 = value.to(torch.float64)
    return 2.0 * (torch.sqrt(1.0 + value64.square()) - 1.0)


def executed_prefixes(
    reference_chunks: Any,
    candidate_chunks: Any,
    replan_indices: Sequence[int],
    *,
    execution_prefix: int = DEFAULT_EXECUTION_PREFIX,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Sort replans and concatenate only the prefix executed before replanning."""
    reference = _actions(reference_chunks, name="reference chunks")
    candidate = _actions(candidate_chunks, name="candidate chunks")
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate chunks must have the same shape")
    indices = [int(value) for value in replan_indices]
    if len(indices) != reference.shape[0] or len(set(indices)) != len(indices):
        raise ValueError("replan indices must be unique and match the number of chunks")
    if not 1 <= int(execution_prefix) <= reference.shape[1]:
        raise ValueError("execution_prefix must lie within the action chunk horizon")
    order = torch.tensor(sorted(range(len(indices)), key=indices.__getitem__), dtype=torch.long)
    sorted_reference = reference.index_select(0, order)[:, :execution_prefix]
    sorted_candidate = candidate.index_select(0, order)[:, :execution_prefix]
    return (
        sorted_reference.reshape(-1, reference.shape[-1]),
        sorted_candidate.reshape(-1, candidate.shape[-1]),
        [indices[position] for position in order.tolist()],
    )


def dpac_sequence(
    reference_chunks: Any,
    candidate_chunks: Any,
    replan_indices: Sequence[int],
    *,
    scale: Any,
    execution_prefix: int = DEFAULT_EXECUTION_PREFIX,
) -> dict[str, Any]:
    """Evaluate one sequence using all cumulative targets over its executed prefix."""
    reference, candidate, sorted_replans = executed_prefixes(
        reference_chunks,
        candidate_chunks,
        replan_indices,
        execution_prefix=execution_prefix,
    )
    coordinate_scale = torch.as_tensor(scale).detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    if coordinate_scale.shape != (reference.shape[-1],):
        raise ValueError("scale must contain one positive value per action coordinate")
    if bool((coordinate_scale <= 0.0).any()) or not torch.isfinite(coordinate_scale).all():
        raise ValueError("scale must contain finite positive values")

    normalized = (candidate - reference) / coordinate_scale
    immediate_per_step = charbonnier(normalized).mean(dim=-1)
    steps = torch.arange(1, normalized.shape[0] + 1, dtype=torch.float64)
    cumulative_targets = normalized.cumsum(dim=0) / steps.sqrt().unsqueeze(1)
    accumulated_per_step = charbonnier(cumulative_targets).mean(dim=-1)
    return {
        "immediate_loss": float(immediate_per_step.mean()),
        "accumulated_loss": float(accumulated_per_step.mean()),
        "immediate_per_step": immediate_per_step.tolist(),
        "accumulated_per_step": accumulated_per_step.tolist(),
        "normalized_error": normalized.tolist(),
        "cumulative_targets": cumulative_targets.tolist(),
        "replan_indices": sorted_replans,
        "execution_prefix": int(execution_prefix),
        "executed_steps": int(normalized.shape[0]),
        "action_dim": int(normalized.shape[1]),
    }


def summarize_pair(
    reference_chunks: Any,
    candidate_chunks: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    scale: Any | None = None,
    execution_prefix: int = DEFAULT_EXECUTION_PREFIX,
) -> dict[str, Any]:
    """Pool steps and coordinates per task-state sequence, then weight sequences equally."""
    reference = _actions(reference_chunks, name="reference chunks")
    candidate = _actions(candidate_chunks, name="candidate chunks")
    if reference.shape != candidate.shape or reference.shape[0] != len(records):
        raise ValueError("paired chunks and records must have the same observation inventory")
    action_scale = reference_action_scale(reference) if scale is None else torch.as_tensor(scale)

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if "replan" not in record:
            raise ValueError("every D-PAC record must contain its replan index")
        sequence_id = str(record.get("sequence_id", record.get("seed", "-1")))
        groups[(str(record.get("task", "")), sequence_id)].append(index)

    sequences = []
    for (task, sequence_id), positions in sorted(groups.items()):
        selected = torch.tensor(positions, dtype=torch.long)
        row = dpac_sequence(
            reference.index_select(0, selected),
            candidate.index_select(0, selected),
            [int(records[position]["replan"]) for position in positions],
            scale=action_scale,
            execution_prefix=execution_prefix,
        )
        sequences.append(
            {
                "task": task,
                "sequence_id": sequence_id,
                "record_indices": positions,
                **row,
            }
        )
    if not sequences:
        raise ValueError("D-PAC requires at least one sequence")
    return {
        "immediate_loss": float(sum(row["immediate_loss"] for row in sequences) / len(sequences)),
        "accumulated_loss": float(sum(row["accumulated_loss"] for row in sequences) / len(sequences)),
        "sequence_weighting": "equal",
        "dimension_scale": torch.as_tensor(action_scale, dtype=torch.float64).tolist(),
        "sequences": sequences,
    }


def summarize_selection_and_audit(
    reference_selection_chunks: Any,
    candidate_selection_chunks: Any,
    reference_audit_chunks: Any,
    candidate_audit_chunks: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    execution_prefix: int = DEFAULT_EXECUTION_PREFIX,
) -> dict[str, Any]:
    """Score selection and audit noise streams with the same selection-reference scale."""
    scale = reference_action_scale(reference_selection_chunks)
    return {
        "selection": summarize_pair(
            reference_selection_chunks,
            candidate_selection_chunks,
            records,
            scale=scale,
            execution_prefix=execution_prefix,
        ),
        "audit": summarize_pair(
            reference_audit_chunks,
            candidate_audit_chunks,
            records,
            scale=scale,
            execution_prefix=execution_prefix,
        ),
        "dimension_scale_from": "reference_selection_actions",
        "audit_changes_selected_mask": False,
    }


def selftest() -> None:
    reference = torch.zeros(2, 4, 1)
    persistent = reference.clone()
    persistent[:, :2, 0] = 1.0
    cancelling = reference.clone()
    cancelling[0, 0, 0] = 1.0
    cancelling[0, 1, 0] = -1.0
    cancelling[1, 0, 0] = 1.0
    cancelling[1, 1, 0] = -1.0
    replans = [0, 1]
    scale = torch.ones(1)

    persistent_score = dpac_sequence(
        reference, persistent, replans, scale=scale, execution_prefix=2
    )
    cancelling_score = dpac_sequence(
        reference, cancelling, replans, scale=scale, execution_prefix=2
    )
    assert abs(persistent_score["immediate_loss"] - cancelling_score["immediate_loss"]) < 1e-12
    assert persistent_score["accumulated_loss"] > cancelling_score["accumulated_loss"]
    assert persistent_score["cumulative_targets"][-1][0] == 2.0

    # The unexecuted suffix cannot affect either loss.
    suffix_only = reference.clone()
    suffix_only[:, 2:, 0] = 1000.0
    suffix_score = dpac_sequence(reference, suffix_only, replans, scale=scale, execution_prefix=2)
    assert suffix_score["immediate_loss"] == 0.0
    assert suffix_score["accumulated_loss"] == 0.0

    records = [
        {"task": "demo", "sequence_id": "a", "replan": 0},
        {"task": "demo", "sequence_id": "a", "replan": 1},
    ]
    summary = summarize_pair(reference, persistent, records, scale=scale, execution_prefix=2)
    assert summary["accumulated_loss"] == persistent_score["accumulated_loss"]

    unequal_reference = torch.zeros(3, 2, 1)
    unequal_candidate = unequal_reference.clone()
    unequal_candidate[:2, :, 0] = 1.0
    unequal_records = [
        {"task": "demo", "sequence_id": "long", "replan": 0},
        {"task": "demo", "sequence_id": "long", "replan": 1},
        {"task": "demo", "sequence_id": "short", "replan": 0},
    ]
    unequal = summarize_pair(
        unequal_reference,
        unequal_candidate,
        unequal_records,
        scale=scale,
        execution_prefix=2,
    )
    long_only = dpac_sequence(
        unequal_reference[:2],
        unequal_candidate[:2],
        [0, 1],
        scale=scale,
        execution_prefix=2,
    )
    assert abs(unequal["immediate_loss"] - 0.5 * long_only["immediate_loss"]) < 1e-12

    streams = summarize_selection_and_audit(
        reference,
        persistent,
        reference + 100.0,
        reference + 100.0,
        records,
        execution_prefix=2,
    )
    assert streams["selection"]["dimension_scale"] == streams["audit"]["dimension_scale"]
    assert streams["audit_changes_selected_mask"] is False
    print("[quantvla-dpac] selftest OK")


if __name__ == "__main__":
    selftest()