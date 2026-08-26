#!/usr/bin/env python3
"""Closed-form, selector-free ErrorFold fitting and deployment folding.

All fit operations use paired FP16/full-quant activations from the shared
calibration buffer.  There is no optimizer, gradient, task mapping or success
label in this module.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from quantvla_cross_model_protocol import PROTOCOL, PROTOCOL_SHA256


ERRORFOLD = PROTOCOL["errorfold"]
DEFAULT_FOLDS = int(ERRORFOLD["folds"])
DEFAULT_RIDGE = float(ERRORFOLD["ridge"])
DEFAULT_EPSILON = float(ERRORFOLD["epsilon"])


def _paired_2d(fp16: Any, quant: Any) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_raw = torch.as_tensor(fp16).detach().to(torch.float64)
    candidate_raw = torch.as_tensor(quant).detach().to(torch.float64)
    if teacher_raw.ndim < 2 or candidate_raw.ndim < 2:
        raise ValueError("ErrorFold activations need at least sample and channel axes")
    teacher = teacher_raw.reshape(-1, teacher_raw.shape[-1])
    candidate = candidate_raw.reshape(-1, candidate_raw.shape[-1])
    if teacher.shape != candidate.shape or teacher.shape[0] < 2:
        raise ValueError(f"paired activations need matching (N,C), got {teacher.shape}/{candidate.shape}")
    if not torch.isfinite(teacher).all() or not torch.isfinite(candidate).all():
        raise ValueError("ErrorFold activations contain non-finite values")
    return teacher, candidate


def ridge_affine(fp16: Any, quant: Any, *, ridge: float = DEFAULT_RIDGE) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output closed-form ridge fit ``yF ~= gain*yQ + bias``."""
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    teacher, candidate = _paired_2d(fp16, quant)
    teacher_mean = teacher.mean(dim=0)
    candidate_mean = candidate.mean(dim=0)
    centered_teacher = teacher - teacher_mean
    centered_candidate = candidate - candidate_mean
    covariance = (centered_candidate * centered_teacher).mean(dim=0)
    variance = centered_candidate.square().mean(dim=0)
    # Ridge is relative to channel energy so units and model normalization do
    # not change the strength of identity regularization.
    ridge_term = float(ridge) * variance.mean().clamp_min(DEFAULT_EPSILON)
    gain = (covariance + ridge_term) / (variance + ridge_term)
    bias = teacher_mean - gain * candidate_mean
    return gain.to(torch.float32), bias.to(torch.float32)


def _fold_assignments(n_rows: int, folds: int) -> torch.Tensor:
    if folds < 2 or n_rows < folds:
        raise ValueError(f"need at least {folds} rows for {folds}-fold reliability")
    # Deterministic interleaving preserves calibration-buffer order while
    # preventing contiguous task/episode blocks from becoming a fold identity.
    return torch.arange(n_rows, dtype=torch.long) % folds


def _standard_error(estimates: torch.Tensor) -> torch.Tensor:
    if estimates.shape[0] < 2:
        return torch.zeros_like(estimates[0])
    # Every estimate is a delete-one-fold fit, so the rows are correlated.
    # Use the jackknife standard error instead of treating them as K
    # independent fits (which would understate uncertainty by K-1).
    folds = estimates.shape[0]
    centered = estimates - estimates.mean(dim=0, keepdim=True)
    return torch.sqrt((folds - 1.0) / folds * centered.square().sum(dim=0))


def reliability(theta: torch.Tensor, standard_error: torch.Tensor, *, epsilon: float = DEFAULT_EPSILON) -> torch.Tensor:
    square = theta.square()
    return square / (square + standard_error.square() + float(epsilon))


def fit_errorfold(
    fp16: Any,
    quant: Any,
    *,
    kind: str = "linear",
    folds: int = DEFAULT_FOLDS,
    ridge: float = DEFAULT_RIDGE,
) -> dict[str, Any]:
    """Fit raw affine and 8-fold reliability for Linear, ATM or head output."""
    if kind not in {"linear", "attention_logits", "attention_head_output"}:
        raise ValueError(f"unsupported ErrorFold kind: {kind}")
    teacher, candidate = _paired_2d(fp16, quant)
    gain, bias = ridge_affine(teacher, candidate, ridge=ridge)
    assignments = _fold_assignments(teacher.shape[0], int(folds))
    gain_folds = []
    bias_folds = []
    for fold in range(int(folds)):
        fit_mask = assignments != fold
        fold_gain, fold_bias = ridge_affine(
            teacher[fit_mask], candidate[fit_mask], ridge=ridge
        )
        gain_folds.append(fold_gain)
        bias_folds.append(fold_bias)
    gain_se = _standard_error(torch.stack(gain_folds))
    bias_se = _standard_error(torch.stack(bias_folds))
    gain_delta = gain - 1.0
    gain_rel = reliability(gain_delta, gain_se)
    bias_rel = reliability(bias, bias_se)
    identity_prediction = candidate.to(torch.float32)
    fitted_prediction = (
        candidate.to(torch.float32) * gain.unsqueeze(0) + bias.unsqueeze(0)
    )
    target = teacher.to(torch.float32)
    return {
        "kind": kind,
        "gain": gain.tolist(),
        "bias": bias.tolist(),
        "gain_delta": gain_delta.tolist(),
        "gain_standard_error": gain_se.tolist(),
        "bias_standard_error": bias_se.tolist(),
        "gain_reliability": gain_rel.tolist(),
        "bias_reliability": bias_rel.tolist(),
        "correction_norm": float(
            torch.sqrt(gain_delta.square().mean() + bias.square().mean())
        ),
        "identity_mse": float((identity_prediction - target).square().mean()),
        "fitted_mse": float((fitted_prediction - target).square().mean()),
        "n_samples": int(teacher.shape[0]),
        "channels": int(teacher.shape[1]),
        "folds": int(folds),
        "ridge": float(ridge),
        "protocol_sha256": PROTOCOL_SHA256,
    }


def gated_affine(entry: Mapping[str, Any], gate: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return identity-shrunk gain/bias for one global SoftFold gate."""
    if not 0.0 <= float(gate) <= 1.0:
        raise ValueError(f"ErrorFold gate must be in [0,1], got {gate}")
    raw_gain = torch.as_tensor(entry["gain"], dtype=torch.float32)
    raw_bias = torch.as_tensor(entry["bias"], dtype=torch.float32)
    gain_rel = torch.as_tensor(entry["gain_reliability"], dtype=torch.float32)
    bias_rel = torch.as_tensor(entry["bias_reliability"], dtype=torch.float32)
    gain = 1.0 + float(gate) * gain_rel * (raw_gain - 1.0)
    bias = float(gate) * bias_rel * raw_bias
    if not torch.isfinite(gain).all() or not torch.isfinite(bias).all():
        raise ValueError("folded ErrorFold affine must be finite")
    return gain, bias


def materialize_entry(entry: Mapping[str, Any], gate: float) -> dict[str, Any]:
    gain, bias = gated_affine(entry, gate)
    softmax_invariant_bias_dropped = entry.get("kind") == "attention_logits"
    if softmax_invariant_bias_dropped:
        # The fitted ATM intercept is one scalar per head and is broadcast to
        # every (query,key) logit in that head.  Softmax removes that constant
        # exactly, so deployment folds only the gain and reports the effective
        # correction norm of what is actually resident.
        bias = torch.zeros_like(bias)
    return {
        **dict(entry),
        "effective_gain": gain.tolist(),
        "effective_bias": bias.tolist(),
        "global_gate": float(gate),
        "softmax_invariant_bias_dropped": softmax_invariant_bias_dropped,
        "runtime_selector": False,
    }


def fold_linear_parameters(
    weight: Any,
    bias: Any | None,
    gain: Any,
    correction_bias: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold output affine into a Linear's rows and bias exactly."""
    matrix = torch.as_tensor(weight)
    output_gain = torch.as_tensor(gain, device=matrix.device, dtype=matrix.dtype).reshape(-1)
    additive = torch.as_tensor(
        correction_bias, device=matrix.device, dtype=matrix.dtype
    ).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != output_gain.numel() or additive.shape != output_gain.shape:
        raise ValueError("Linear fold shape mismatch")
    original_bias = (
        torch.zeros_like(output_gain)
        if bias is None
        else torch.as_tensor(bias, device=matrix.device, dtype=matrix.dtype).reshape(-1)
    )
    return matrix * output_gain[:, None], original_bias * output_gain + additive


def fold_dequant_scales(scales: Any, gain: Any) -> torch.Tensor:
    """Equivalent fold for signed W4 rows: scale'_o = gain_o*scale_o."""
    value = torch.as_tensor(scales)
    output_gain = torch.as_tensor(gain, device=value.device, dtype=value.dtype).reshape(-1)
    if value.shape[0] != output_gain.numel() or bool((output_gain <= 0).any()):
        raise ValueError("dequant-scale ErrorFold requires one positive gain per output")
    return value * output_gain.reshape((-1,) + (1,) * (value.ndim - 1))


def selftest() -> None:
    generator = torch.Generator().manual_seed(5)
    quant = torch.randn(64, 7, generator=generator)
    teacher = quant * 1.1 + 0.2
    fitted = fit_errorfold(teacher, quant)
    gain, bias = gated_affine(fitted, 1.0)
    assert fitted["fitted_mse"] < fitted["identity_mse"]
    torch.testing.assert_close(quant * gain + bias, teacher, atol=2e-5, rtol=2e-5)
    print("[quantvla-errorfold] selftest OK")


if __name__ == "__main__":
    selftest()
