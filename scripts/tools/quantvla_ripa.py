#!/usr/bin/env python3
"""RIPA representation probes and dual-site damage aggregation."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from dypac_vla_protocol import PROTOCOL, RIPA_ALPHA


RMS_LOG_RATIO_THRESHOLD = float(PROTOCOL["ripa"]["guards"]["rms_log_ratio_threshold"])
TAIL_EXCEEDANCE_THRESHOLD = float(
    PROTOCOL["ripa"]["guards"]["reference_tail_exceedance_threshold"]
)
TAIL_QUANTILE = float(PROTOCOL["ripa"]["guards"]["reference_tail_quantile"])
ROWS_PER_HOOK_CALL = int(PROTOCOL["ripa"]["probe_sampling"]["rows_per_hook_call"])
MAX_CONCATENATED_ROWS = int(
    PROTOCOL["ripa"]["probe_sampling"]["maximum_concatenated_rows"]
)


def sample_probe_rows(
    activation: Any,
    *,
    rows_per_call: int = ROWS_PER_HOOK_CALL,
    maximum_rows: int = MAX_CONCATENATED_ROWS,
) -> torch.Tensor:
    """Select the fixed number of evenly spaced token rows used by each RIPA hook."""
    value = torch.as_tensor(activation).detach().to(dtype=torch.float32, device="cpu")
    if value.ndim < 2 or value.shape[-1] == 0 or not torch.isfinite(value).all():
        raise ValueError("RIPA hook activation must be finite with a final feature axis")
    if rows_per_call <= 0 or maximum_rows <= 0:
        raise ValueError("RIPA row caps must be positive")
    rows = value.reshape(-1, value.shape[-1])
    take = min(int(rows_per_call), rows.shape[0], int(maximum_rows))
    indices = torch.linspace(0, rows.shape[0] - 1, take).round().to(torch.long)
    return rows.index_select(0, indices).contiguous()


def concatenate_probe_rows(
    hook_calls: Any, *, maximum_rows: int = MAX_CONCATENATED_ROWS
) -> torch.Tensor:
    """Concatenate sampled hook rows and apply the protocol's deterministic global cap."""
    sampled = [sample_probe_rows(value, maximum_rows=maximum_rows) for value in hook_calls]
    if not sampled:
        raise ValueError("RIPA probe requires at least one hook call")
    rows = torch.cat(sampled, dim=0)
    if rows.shape[0] <= maximum_rows:
        return rows.contiguous()
    indices = torch.linspace(0, rows.shape[0] - 1, maximum_rows).round().to(torch.long)
    return rows.index_select(0, indices).contiguous()


def _paired_matrices(reference: Any, intervention: Any, *, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(reference).detach().to(dtype=torch.float64, device="cpu")
    y = torch.as_tensor(intervention).detach().to(dtype=torch.float64, device="cpu")
    if x.ndim != 2 or x.shape != y.shape or x.shape[0] < 2:
        raise ValueError(f"{name} must be paired 2-D matrices with at least two rows")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError(f"{name} contains non-finite values")
    return x, y


def relational_damage(reference: Any, intervention: Any) -> float:
    """Damage to centered sample-relation geometry at the action site."""
    x, y = _paired_matrices(reference, intervention, name="relational probe")
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    a = x @ x.T
    b = y @ y.T
    denominator = torch.linalg.matrix_norm(a) * torch.linalg.matrix_norm(b)
    if float(denominator) <= 1e-18:
        return 0.0 if torch.allclose(x, y, atol=1e-12, rtol=0.0) else 1.0
    similarity = torch.sum(a * b) / denominator
    return float((1.0 - similarity.clamp(0.0, 1.0)).clamp_min(0.0))


def _log_mean_gaussian(left: torch.Tensor, right: torch.Tensor, sigma_squared: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(left, right).square()
    values = -distances / (2.0 * sigma_squared)
    return torch.logsumexp(values.reshape(-1), dim=0) - math.log(values.numel())


def distributional_damage(reference: Any, intervention: Any) -> float:
    """Reference-bandwidth Gaussian distributional damage at the layer site."""
    x, y = _paired_matrices(reference, intervention, name="distributional probe")
    squared = torch.cdist(x, x).square()
    off_diagonal = squared[~torch.eye(x.shape[0], dtype=torch.bool)]
    sigma_squared = off_diagonal.median().clamp_min(1e-6)
    value = (
        _log_mean_gaussian(x, x, sigma_squared)
        + _log_mean_gaussian(y, y, sigma_squared)
        - 2.0 * _log_mean_gaussian(x, y, sigma_squared)
    )
    return float(value.clamp_min(0.0))


def _minmax_bank(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        raise ValueError("damage bank is empty")
    ordered = {str(name): float(value) for name, value in values.items()}
    if not all(math.isfinite(value) for value in ordered.values()):
        raise ValueError("damage bank contains non-finite values")
    lo = min(ordered.values())
    hi = max(ordered.values())
    if hi - lo <= 1e-15:
        return {name: 0.0 for name in ordered}
    return {name: (value - lo) / (hi - lo) for name, value in ordered.items()}


def aggregate_probe_bank(
    relational: Mapping[str, float],
    distributional: Mapping[str, float],
    *,
    alpha: float = RIPA_ALPHA,
) -> dict[str, dict[str, float]]:
    """Normalize each site over the layer bank and combine them with alpha=16/17."""
    if set(relational) != set(distributional) or not relational:
        raise ValueError("the two RIPA probe banks must contain the same layers")
    if not math.isclose(float(alpha), RIPA_ALPHA, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"the public Method fixes alpha={RIPA_ALPHA}")
    rel_norm = _minmax_bank(relational)
    dist_norm = _minmax_bank(distributional)
    return {
        name: {
            "relational_damage": float(relational[name]),
            "distributional_damage": float(distributional[name]),
            "normalized_relational_damage": rel_norm[name],
            "normalized_distributional_damage": dist_norm[name],
            "importance": alpha * rel_norm[name] + (1.0 - alpha) * dist_norm[name],
        }
        for name in sorted(relational)
    }


def guard_statistics(reference_layer_rows: Any, intervention_layer_rows: Any) -> dict[str, float | bool]:
    """RMS log-ratio and reference-tail checks used to protect a layer."""
    x, y = _paired_matrices(reference_layer_rows, intervention_layer_rows, name="RIPA guard")
    rms_x = x.square().mean(dim=0).sqrt()
    rms_y = y.square().mean(dim=0).sqrt()
    rms_log_ratio = ((rms_y + 1e-6) / (rms_x + 1e-6)).log().abs().median()
    threshold = torch.quantile(x.abs().reshape(-1), TAIL_QUANTILE)
    tail_exceedance = (y.abs() > threshold).to(torch.float64).mean()
    protected = bool(
        float(rms_log_ratio) > RMS_LOG_RATIO_THRESHOLD
        or float(tail_exceedance) > TAIL_EXCEEDANCE_THRESHOLD
    )
    return {
        "rms_log_ratio": float(rms_log_ratio),
        "reference_tail_exceedance": float(tail_exceedance),
        "protected": protected,
    }


def probe_layer(
    *,
    reference_action_rows: Any,
    intervention_action_rows: Any,
    reference_layer_rows: Any,
    intervention_layer_rows: Any,
) -> dict[str, float | bool]:
    """Evaluate the paper's two RIPA sites for one isolated W4 intervention."""
    guards = guard_statistics(reference_layer_rows, intervention_layer_rows)
    return {
        "relational_damage": relational_damage(reference_action_rows, intervention_action_rows),
        "distributional_damage": distributional_damage(reference_layer_rows, intervention_layer_rows),
        **guards,
    }


def selftest() -> None:
    generator = torch.Generator().manual_seed(11)
    action = torch.randn(24, 8, generator=generator)
    layer = torch.randn(24, 12, generator=generator)
    scaled = layer * 4.0
    assert relational_damage(action, action * 3.0) < 1e-10
    assert distributional_damage(layer, layer) < 1e-10
    assert distributional_damage(layer, scaled) > 0.0
    bank = aggregate_probe_bank(
        {"a": 0.0, "b": 1.0},
        {"a": 1.0, "b": 0.0},
    )
    assert math.isclose(bank["a"]["importance"], 1.0 / 17.0)
    assert math.isclose(bank["b"]["importance"], 16.0 / 17.0)
    sampled = sample_probe_rows(torch.arange(60, dtype=torch.float32).reshape(2, 5, 6))
    assert sampled.shape == (4, 6)
    concatenated = concatenate_probe_rows(
        [torch.randn(3, 7, 6, generator=generator) for _ in range(80)]
    )
    assert concatenated.shape == (256, 6)
    guards = guard_statistics(layer, scaled)
    assert abs(float(guards["rms_log_ratio"]) - math.log(4.0)) < 1e-3
    assert guards["protected"] is True
    print("[quantvla-ripa] selftest OK")


if __name__ == "__main__":
    selftest()