#!/usr/bin/env python3
"""Tail-aware functional metrics for QuantVLA v1.4 and D_PAC v1.

The v1.3 D_solver is a late-step-weighted MEAN relative error; the v1.3
LIBERO full test showed its ranking does not transfer consistently to
closed-loop success (v2 4.7-8.5x better on D_solver, +/-0.8-1.2 sigma on SR).
v1.4 replaces it with D_func, a tail-aware functional metric computed from the
SAME paired denoising trajectories (T+1, B, H, D):

  d_final    final action-chunk deviation (last denoising step, relative)
  d_kin      per-dim kinematic errors: translation (dims 0:3), rotation
             (3:6), gripper (6:6+g) — layout configurable per embodiment
  d_grip     gripper sign-mismatch rate: fraction of steps where the
             ref and quantized gripper deltas disagree in sign (binarization
             proxy for grasp/contact transitions)
  tail       p90 / p95 / CVaR0.9 of the per-obs divergences
  grasp-w    grasp-window weighting: steps with large gripper-state change
             (contact/grasp transition proxy) upweighted in the mean

  D_func = w_final*d_final + w_kin*d_kin + w_grip*d_grip + w_tail*CVaR0.9
  default weights 1/1/1/2 (tail emphasized); frozen for the experiment.

All components come from trajectory pairs the probe/scorer already collect —
no new rollouts. The end-effector-Jacobian weighting from the plan is replaced
by the grasp-window proxy here because the synthetic measurement protocol has
no EE Jacobian; that substitution is documented in the report.

Usage:
    python -m gr00t_func_metrics            # selftest
    from gr00t_func_metrics import d_func   # (ref_traj, q_traj, gamma) -> dict

``d_pac_sequence`` adds the control-time axis without claiming closed-loop
equivalence.  It compares paired FP16/quantized outputs on the same frozen
observations and noise, and therefore measures action-prefix drift or (when
consecutive observations are available) replan-sequence drift.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

# LIBERO embodiment: [dx, dy, dz, droll, dpitch, dyaw, gripper(, gripper2)]
DEFAULT_LAYOUT = {"trans": (0, 3), "rot": (3, 6), "grip": (6, 8)}
DEFAULT_WEIGHTS = {"final": 1.0, "kin": 1.0, "grip": 1.0, "tail": 2.0}
DEFAULT_PAC_WEIGHTS = {
    "func": 1.0,
    "prefix_pose": 1.0,
    "stitch": 1.0,
    "grip_time": 1.0,
    "overlap": 0.0,
}


def per_obs_divergences(ref: torch.Tensor, q: torch.Tensor, gamma: float) -> List[float]:
    """Late-step-weighted per-obs relative divergence (v1.3 D_solver inputs)."""
    ref = ref.float()[:, : q.shape[1]]
    q = q.float()
    num = ((ref - q) ** 2).sum(dim=(-1, -2))
    den = (ref**2).sum(dim=(-1, -2)).clamp_min(1e-8)
    rel = num / den
    k_steps = rel.shape[0]
    weights = torch.tensor([gamma ** (k + 1) for k in range(k_steps)])
    weights = weights / weights.sum()
    per_obs = (rel * weights[:, None]).sum(dim=0)
    return [float(v) for v in per_obs]


def tail_stats(per_obs: List[float]) -> Dict[str, float]:
    """p90 / p95 / CVaR0.9 of the per-obs divergence list."""
    a = np.asarray(per_obs, dtype=np.float64)
    if a.size == 0:
        return {"p90": 0.0, "p95": 0.0, "cvar90": 0.0, "n": 0}
    p90 = float(np.quantile(a, 0.90))
    p95 = float(np.quantile(a, 0.95))
    tail = a[a >= p90]
    cvar = float(tail.mean()) if tail.size else p90
    return {"p90": p90, "p95": p95, "cvar90": cvar, "n": int(a.size)}


def final_action_deviation(ref: torch.Tensor, q: torch.Tensor) -> float:
    """Relative error of the FINAL denoising step's action chunk."""
    r = ref.float()[-1]
    s = q.float()[-1]
    num = float(((r - s) ** 2).sum())
    den = float((r**2).sum().clamp_min(1e-8))
    return num / den


def per_dim_errors(ref: torch.Tensor, q: torch.Tensor, layout: Dict[str, tuple]) -> Dict[str, float]:
    """Per-dimension-group relative errors on the final action chunk."""
    r = ref.float()[-1]  # (B, H, D)
    s = q.float()[-1]
    out: Dict[str, float] = {}
    for grp, (lo, hi) in layout.items():
        if lo >= r.shape[-1]:
            out[grp] = 0.0
            continue
        hi = min(hi, r.shape[-1])
        num = float(((r[..., lo:hi] - s[..., lo:hi]) ** 2).sum())
        den = float((r[..., lo:hi] ** 2).sum().clamp_min(1e-8))
        out[grp] = num / den
    return out


def gripper_sign_mismatch(ref: torch.Tensor, q: torch.Tensor, grip_idx: tuple) -> Dict[str, float]:
    """Fraction of (obs, step) pairs whose gripper delta sign disagrees.

    Uses per-obs deltas over the trajectory (steps 1..T+1); a step counts when
    BOTH deltas exceed eps (active gripper motion) and their signs differ.
    """
    r = ref.float()
    s = q.float()
    lo, hi = grip_idx
    if lo >= r.shape[-1]:
        return {"rate": 0.0, "n_active": 0}
    hi = min(hi, r.shape[-1])
    dr = (r[1:, ..., lo:hi] - r[:-1, ..., lo:hi]).sum(dim=-1)  # (T, B)
    ds = (s[1:, ..., lo:hi] - s[:-1, ..., lo:hi]).sum(dim=-1)
    eps = 1e-4
    active = (dr.abs() > eps) & (ds.abs() > eps)
    mismatch = active & (dr.sign() != ds.sign())
    n_active = int(active.sum())
    n_mis = int(mismatch.sum())
    return {"rate": (n_mis / n_active) if n_active else 0.0, "n_active": n_active}


def grasp_window_weight(ref: torch.Tensor, grip_idx: tuple) -> torch.Tensor:
    """Per-(obs) grasp-window weight: 1 + |gripper delta| normalized (contact
    transition proxy). Shape (B,)."""
    r = ref.float()
    lo, hi = grip_idx
    if lo >= r.shape[-1]:
        return torch.ones(r.shape[1])
    hi = min(hi, r.shape[-1])
    d = (r[1:, ..., lo:hi] - r[:-1, ..., lo:hi]).abs().sum(dim=-1)  # (T, B, H)
    d = d.sum(dim=(0, 2))  # (B,) total gripper motion per obs
    mx = d.max().clamp_min(1e-8)
    return 1.0 + (d / mx)


def d_func(
    ref: torch.Tensor,
    q: torch.Tensor,
    gamma: float = 1.2,
    layout: Optional[Dict[str, tuple]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Combined tail-aware functional metric + all components."""
    layout = layout or DEFAULT_LAYOUT
    weights = weights or DEFAULT_WEIGHTS
    per_obs = per_obs_divergences(ref, q, gamma)
    tail = tail_stats(per_obs)
    d_fin = final_action_deviation(ref, q)
    dims = per_dim_errors(ref, q, layout)
    d_kin = (dims["trans"] + dims["rot"]) / 2.0
    d_grip = gripper_sign_mismatch(ref, q, layout["grip"])["rate"]
    # grasp-window weighted mean over per-obs (proxy for contact weighting)
    gw = grasp_window_weight(ref, layout["grip"])
    gw = gw / gw.mean()  # normalize to mean 1
    d_mean = float((torch.tensor(per_obs, dtype=torch.float32) * gw).mean())
    combined = (
        weights["final"] * d_fin
        + weights["kin"] * d_kin
        + weights["grip"] * d_grip
        + weights["tail"] * tail["cvar90"]
    )
    return {
        "d_func": combined,
        "d_final": d_fin,
        "d_kin": d_kin,
        "d_grip": d_grip,
        "d_mean": d_mean,
        "d_solver": float(np.mean(per_obs)),
        "d_solver_std": float(np.std(per_obs)) if len(per_obs) > 1 else 0.0,
        "tail": tail,
        "per_dim": dims,
        "weights": dict(weights),
        "per_obs": per_obs,
    }


# --------------------------------------------------------------------------- #
# D_PAC v1: FP16-anchored control-time accumulation
# --------------------------------------------------------------------------- #
def _as_sequence_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().to(dtype=torch.float32, device="cpu")
    if tensor.ndim not in (3, 4):
        raise ValueError(
            f"{name} must be (R,H,D) final chunks or (T+1,R,H,D) trajectories, "
            f"got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _final_chunks(sequence: torch.Tensor) -> torch.Tensor:
    return sequence[-1] if sequence.ndim == 4 else sequence


def _mad_scale(reference: torch.Tensor, eps: float) -> torch.Tensor:
    """FP16-only per-action-dimension scale used by every control-time term."""
    flat = reference.reshape(-1, reference.shape[-1])
    median = flat.median(dim=0).values
    mad = (flat - median).abs().median(dim=0).values
    return mad.clamp_min(float(eps))


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind()
    return torch.stack(
        (
            torch.stack((x * 0.0, -z, y)),
            torch.stack((z, x * 0.0, -x)),
            torch.stack((-y, x, x * 0.0)),
        )
    )


def _se3_exp(twist: torch.Tensor) -> torch.Tensor:
    """Stable SE(3) exponential for one [translation, rotation-vector] action."""
    translation = twist[:3]
    rotation = twist[3:6]
    theta = float(torch.linalg.vector_norm(rotation))
    omega = _skew(rotation)
    omega2 = omega @ omega
    eye = torch.eye(3, dtype=twist.dtype)
    if theta < 1e-5:
        # Taylor expansions through the first non-trivial correction.
        a = 1.0 - theta * theta / 6.0
        b = 0.5 - theta * theta / 24.0
        c = 1.0 / 6.0 - theta * theta / 120.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta * theta)
        c = (theta - np.sin(theta)) / (theta * theta * theta)
    rotation_matrix = eye + float(a) * omega + float(b) * omega2
    v_matrix = eye + float(b) * omega + float(c) * omega2
    transform = torch.eye(4, dtype=twist.dtype)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = v_matrix @ translation
    return transform


def _se3_log(transform: torch.Tensor) -> torch.Tensor:
    """Stable SE(3) logarithm returning [translation, rotation-vector]."""
    rotation_matrix = transform[:3, :3]
    translation = transform[:3, 3]
    cosine = ((torch.trace(rotation_matrix) - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = float(torch.acos(cosine))
    antisymmetric = rotation_matrix - rotation_matrix.T
    if theta < 1e-5:
        omega = 0.5 * torch.stack(
            (antisymmetric[2, 1], antisymmetric[0, 2], antisymmetric[1, 0])
        )
    else:
        multiplier = theta / (2.0 * np.sin(theta))
        omega = float(multiplier) * torch.stack(
            (antisymmetric[2, 1], antisymmetric[0, 2], antisymmetric[1, 0])
        )
    omega_matrix = _skew(omega)
    omega2 = omega_matrix @ omega_matrix
    eye = torch.eye(3, dtype=transform.dtype)
    omega_norm = float(torch.linalg.vector_norm(omega))
    if omega_norm < 1e-5:
        coefficient = 1.0 / 12.0
    else:
        coefficient = 1.0 / (omega_norm * omega_norm) - (
            (1.0 + np.cos(omega_norm))
            / (2.0 * omega_norm * np.sin(omega_norm))
        )
    v_inverse = eye - 0.5 * omega_matrix + float(coefficient) * omega2
    return torch.cat((v_inverse @ translation, omega))


def _weighted_prefix_mean(values: torch.Tensor, exponent: float) -> float:
    if values.numel() == 0:
        return 0.0
    n = values.shape[0]
    weights = torch.arange(1, n + 1, dtype=values.dtype) / float(n)
    weights = weights.pow(float(exponent))
    per_prefix = values.square().mean(dim=-1)
    return float((per_prefix * weights).sum() / weights.sum().clamp_min(1e-12))


def prefix_accumulated_drift(
    reference_actions: torch.Tensor,
    candidate_actions: torch.Tensor,
    scale: torch.Tensor,
    *,
    exponent: float = 2.0,
) -> float:
    """All-prefix accumulated normalized action error (the core D_PAC term)."""
    error = (candidate_actions - reference_actions) / scale
    cumulative = error.cumsum(dim=0)
    return _weighted_prefix_mean(cumulative, exponent)


def pose_prefix_drift(
    reference_actions: torch.Tensor,
    candidate_actions: torch.Tensor,
    scale: torch.Tensor,
    *,
    exponent: float = 2.0,
) -> float:
    """SE(3)-composed prefix drift for the first six action dimensions."""
    if reference_actions.shape[-1] < 6:
        return 0.0
    if torch.equal(reference_actions[:, :6], candidate_actions[:, :6]):
        return 0.0
    reference_pose = torch.eye(4, dtype=torch.float32)
    candidate_pose = torch.eye(4, dtype=torch.float32)
    logarithms: List[torch.Tensor] = []
    for reference_delta, candidate_delta in zip(reference_actions, candidate_actions):
        reference_pose = reference_pose @ _se3_exp(reference_delta[:6])
        candidate_pose = candidate_pose @ _se3_exp(candidate_delta[:6])
        relative = torch.linalg.inv(reference_pose) @ candidate_pose
        logarithms.append(_se3_log(relative) / scale[:6])
    return _weighted_prefix_mean(torch.stack(logarithms), exponent)


def replan_stitch_drift(
    reference_chunks: torch.Tensor,
    candidate_chunks: torch.Tensor,
    scale: torch.Tensor,
    executed_actions: int,
) -> float:
    """Difference between quantized and FP16 chunk-boundary jump dynamics."""
    if reference_chunks.shape[0] < 2 or executed_actions <= 0:
        return 0.0
    last = min(executed_actions, reference_chunks.shape[1]) - 1
    reference_jump = reference_chunks[1:, 0] - reference_chunks[:-1, last]
    candidate_jump = candidate_chunks[1:, 0] - candidate_chunks[:-1, last]
    return float((((candidate_jump - reference_jump) / scale) ** 2).mean())


def gripper_timing_drift(
    reference_actions: torch.Tensor,
    candidate_actions: torch.Tensor,
    grip_slice: tuple[int, int],
    *,
    kappa: float = 8.0,
    delta_weight: float = 1.0,
) -> float:
    """Soft gripper-state and event-timing mismatch over executed control time."""
    lo, hi = grip_slice
    if lo >= reference_actions.shape[-1] or lo >= hi:
        return 0.0
    hi = min(hi, reference_actions.shape[-1])
    reference_state = torch.sigmoid(float(kappa) * reference_actions[:, lo:hi])
    candidate_state = torch.sigmoid(float(kappa) * candidate_actions[:, lo:hi])
    state_error = (candidate_state - reference_state).square().mean()
    if reference_state.shape[0] < 2:
        return float(state_error)
    reference_event = reference_state[1:] - reference_state[:-1]
    candidate_event = candidate_state[1:] - candidate_state[:-1]
    event_error = (candidate_event - reference_event).square().mean()
    return float(state_error + float(delta_weight) * event_error)


def forecast_overlap_drift(
    reference_chunks: torch.Tensor,
    candidate_chunks: torch.Tensor,
    scale: torch.Tensor,
    executed_actions: int,
) -> float:
    """Teacher-relative forecast consistency for the unexecuted chunk suffix."""
    horizon = reference_chunks.shape[1]
    overlap = min(horizon - executed_actions, horizon)
    if reference_chunks.shape[0] < 2 or overlap <= 0:
        return 0.0
    reference_consistency = (
        reference_chunks[:-1, executed_actions : executed_actions + overlap]
        - reference_chunks[1:, :overlap]
    )
    candidate_consistency = (
        candidate_chunks[:-1, executed_actions : executed_actions + overlap]
        - candidate_chunks[1:, :overlap]
    )
    return float((((candidate_consistency - reference_consistency) / scale) ** 2).mean())


def _mean_replan_d_func(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    gamma: float,
    layout: Dict[str, tuple],
    executed_actions: int,
) -> float:
    """Mean local D_func with observation-CVaR disabled (sequence tail is outer)."""
    if reference.ndim == 3:
        reference = reference.unsqueeze(0)
        candidate = candidate.unsqueeze(0)
    values: List[float] = []
    local_weights = dict(DEFAULT_WEIGHTS)
    local_weights["tail"] = 0.0
    horizon = min(executed_actions, reference.shape[-2])
    for replan in range(reference.shape[1]):
        result = d_func(
            reference[:, replan : replan + 1, :horizon],
            candidate[:, replan : replan + 1, :horizon],
            gamma=gamma,
            layout=layout,
            weights=local_weights,
        )
        values.append(float(result["d_func"]))
    return float(np.mean(values)) if values else 0.0


def d_pac_sequence(
    fp16_chunks: Any,
    quant_chunks: Any,
    replan_indices: Sequence[int],
    executed_actions: int = 16,
    *,
    action_dim: Optional[int] = None,
    layout: Optional[Dict[str, tuple]] = None,
    weights: Optional[Mapping[str, float]] = None,
    gamma: float = 1.2,
    prefix_exponent: float = 2.0,
    scale_epsilon: float = 1e-6,
    gripper_kappa: float = 8.0,
    gripper_delta_weight: float = 1.0,
) -> Dict[str, Any]:
    """Compute D_PAC for one ordered frozen-observation replan sequence.

    Inputs are either final chunks ``(R,H,D)`` or paired solver trajectories
    ``(T+1,R,H,D)``.  ``R`` is control-time/replan, not a random minibatch.
    The function sorts by the supplied unique replan indices and uses only
    FP16 statistics for normalization.  It is an open-loop surrogate: callers
    should label it ``replan_sequence_drift`` only when the observations are
    actually consecutive, otherwise ``action_prefix_accumulated_drift``.
    """
    reference = _as_sequence_tensor(fp16_chunks, "fp16_chunks")
    candidate = _as_sequence_tensor(quant_chunks, "quant_chunks")
    if reference.shape != candidate.shape:
        raise ValueError(
            f"paired sequence shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    n_replans = reference.shape[-3] if reference.ndim == 4 else reference.shape[0]
    indices = [int(value) for value in replan_indices]
    if len(indices) != n_replans:
        raise ValueError(f"expected {n_replans} replan indices, got {len(indices)}")
    if len(set(indices)) != len(indices):
        raise ValueError("replan_indices must be unique within one sequence")
    if executed_actions <= 0:
        raise ValueError("executed_actions must be positive")
    order = torch.tensor(sorted(range(n_replans), key=indices.__getitem__), dtype=torch.long)
    if reference.ndim == 4:
        reference = reference.index_select(1, order)
        candidate = candidate.index_select(1, order)
    else:
        reference = reference.index_select(0, order)
        candidate = candidate.index_select(0, order)
    sorted_indices = [indices[index] for index in order.tolist()]

    if action_dim is not None:
        if action_dim <= 0 or action_dim > reference.shape[-1]:
            raise ValueError(f"invalid action_dim={action_dim} for D={reference.shape[-1]}")
        reference = reference[..., :action_dim]
        candidate = candidate[..., :action_dim]
    layout = layout or DEFAULT_LAYOUT
    pac_weights = dict(DEFAULT_PAC_WEIGHTS)
    if weights is not None:
        unknown = set(weights) - set(pac_weights)
        if unknown:
            raise ValueError(f"unknown D_PAC weights: {sorted(unknown)}")
        pac_weights.update({key: float(value) for key, value in weights.items()})

    reference_final = _final_chunks(reference)
    candidate_final = _final_chunks(candidate)
    horizon = min(executed_actions, reference_final.shape[1])
    reference_executed = reference_final[:, :horizon]
    candidate_executed = candidate_final[:, :horizon]
    scale = _mad_scale(reference_executed, scale_epsilon)
    reference_control = reference_executed.reshape(-1, reference_executed.shape[-1])
    candidate_control = candidate_executed.reshape(-1, candidate_executed.shape[-1])

    d_prefix = prefix_accumulated_drift(
        reference_control,
        candidate_control,
        scale,
        exponent=prefix_exponent,
    )
    d_pose = pose_prefix_drift(
        reference_control,
        candidate_control,
        scale,
        exponent=prefix_exponent,
    )
    if reference_control.shape[-1] > 6:
        d_prefix_nonpose = prefix_accumulated_drift(
            reference_control[:, 6:],
            candidate_control[:, 6:],
            scale[6:],
            exponent=prefix_exponent,
        )
        d_prefix_pose = d_pose + d_prefix_nonpose
    else:
        d_prefix_nonpose = 0.0
        d_prefix_pose = d_pose if reference_control.shape[-1] >= 6 else d_prefix
    d_stitch = replan_stitch_drift(
        reference_final, candidate_final, scale, horizon
    )
    d_grip_time = gripper_timing_drift(
        reference_control,
        candidate_control,
        layout["grip"],
        kappa=gripper_kappa,
        delta_weight=gripper_delta_weight,
    )
    d_overlap = forecast_overlap_drift(
        reference_final, candidate_final, scale, horizon
    )
    d_local = _mean_replan_d_func(
        reference,
        candidate,
        gamma=gamma,
        layout=layout,
        executed_actions=horizon,
    )
    combined = (
        pac_weights["func"] * d_local
        + pac_weights["prefix_pose"] * d_prefix_pose
        + pac_weights["stitch"] * d_stitch
        + pac_weights["grip_time"] * d_grip_time
        + pac_weights["overlap"] * d_overlap
    )
    return {
        "d_pac_sequence": float(combined),
        "d_func_mean": d_local,
        "d_prefix": d_prefix,
        "d_pose": d_pose,
        "d_prefix_nonpose": d_prefix_nonpose,
        "d_prefix_pose": d_prefix_pose,
        "d_stitch": d_stitch,
        "d_grip_time": d_grip_time,
        "d_overlap": d_overlap,
        "dimension_scale": scale.tolist(),
        "weights": pac_weights,
        "replan_indices": sorted_indices,
        "n_replans": n_replans,
        "executed_actions": horizon,
        "metric": "d_pac_v1",
        "teacher": "original_fp16",
        "surrogate_scope": (
            "replan_sequence_drift" if n_replans > 1 else "action_prefix_accumulated_drift"
        ),
        "observation_cvar_disabled": True,
    }


def aggregate_d_pac_sequences(
    sequences: Sequence[Mapping[str, Any] | float],
    *,
    tail_weight: float = 1.0,
    cvar_alpha: float = 0.9,
) -> Dict[str, Any]:
    """Mean + sequence-level CVaR aggregation, with no duplicated obs-CVaR."""
    if not 0.0 <= cvar_alpha < 1.0:
        raise ValueError("cvar_alpha must be in [0, 1)")
    values = np.asarray(
        [
            float(value["d_pac_sequence"] if isinstance(value, Mapping) else value)
            for value in sequences
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        return {
            "d_pac": 0.0,
            "mean": 0.0,
            "cvar": 0.0,
            "cvar_alpha": cvar_alpha,
            "tail_weight": tail_weight,
            "n_sequences": 0,
            "per_sequence": [],
        }
    threshold = float(np.quantile(values, cvar_alpha))
    tail = values[values >= threshold]
    cvar = float(tail.mean()) if tail.size else threshold
    mean = float(values.mean())
    return {
        "d_pac": mean + float(tail_weight) * cvar,
        "mean": mean,
        "cvar": cvar,
        "cvar_alpha": cvar_alpha,
        "tail_weight": float(tail_weight),
        "n_sequences": int(values.size),
        "per_sequence": values.tolist(),
    }


def selftest() -> None:
    torch.manual_seed(0)
    t = torch.randn(9, 4, 16, 7)  # T+1=9, B=4, H=16, D=7 (libero layout)
    # identical -> everything 0
    d0 = d_func(t, t)
    assert d0["d_func"] == 0.0 and d0["d_final"] == 0.0 and d0["d_kin"] == 0.0
    assert d0["d_grip"] == 0.0 and d0["tail"]["cvar90"] == 0.0
    # uniform scaling -> final/deviation components exactly 1 (relative error)
    d2 = d_func(t, t * 2.0)
    assert abs(d2["d_final"] - 1.0) < 1e-5, d2["d_final"]
    assert abs(d2["d_solver"] - 1.0) < 1e-6, d2["d_solver"]  # P0-3 semantics
    # tail stats on a known list
    a = list(range(100))
    ts = tail_stats(a)
    assert abs(ts["p90"] - 89.1) < 1e-6 and abs(ts["p95"] - 94.05) < 1e-6 and abs(ts["cvar90"] - 94.5) < 1e-6, ts
    # gripper sign flip: q gripper = -ref gripper delta every step -> rate 1
    q = t.clone()
    q[1:, ..., 6] = -t[1:, ..., 6]  # deltas negate (ref delta d -> -d... set q step = -ref delta)
    gm = gripper_sign_mismatch(t, q, (6, 7))
    assert gm["n_active"] > 0 and gm["rate"] > 0.9, gm
    # per-dim: only translation corrupted -> trans error large, rot/grip 0
    q2 = t.clone()
    q2[-1, ..., 0:3] = t[-1, ..., 0:3] * 5.0
    pd = per_dim_errors(t, q2, DEFAULT_LAYOUT)
    assert pd["trans"] > 0.5 and pd["rot"] == 0.0 and pd["grip"] == 0.0, pd
    # combined D_func positive and >= mean component when tail dominates
    d3 = d_func(t, q2)
    assert d3["d_func"] > 0.0 and d3["per_dim"]["trans"] > 0.5
    # D_PAC identity and constant-bias-vs-alternating-bias invariant.
    chunks = torch.randn(4, 8, 7)
    pac0 = d_pac_sequence(chunks, chunks, [0, 1, 2, 3], executed_actions=8)
    assert pac0["d_pac_sequence"] == 0.0
    constant = chunks.clone()
    alternating = chunks.clone()
    constant[..., 0] += 0.01
    sign = torch.tensor([1.0, -1.0] * 16)[: chunks.shape[0] * chunks.shape[1]]
    alternating[..., 0] += (0.01 * sign).reshape(chunks.shape[0], chunks.shape[1])
    pac_constant = d_pac_sequence(chunks, constant, range(4), executed_actions=8)
    pac_alternating = d_pac_sequence(chunks, alternating, range(4), executed_actions=8)
    assert pac_constant["d_prefix"] > pac_alternating["d_prefix"] * 10.0
    print("[gr00t_func_metrics] selftest OK")
    print(f"  identical: d_func={d0['d_func']} d_solver={d0['d_solver']}")
    print(f"  2x scaled: d_final={d2['d_final']:.6f} d_solver={d2['d_solver']:.6f} (expect 1)")
    print(f"  trans-only corruption: d_func={d3['d_func']:.4f} trans={d3['per_dim']['trans']:.4f}")
    print(f"  gripper sign-flip rate: {gm['rate']:.4f} (n_active={gm['n_active']})")
    print(f"  tail(0..99): p90={ts['p90']} p95={ts['p95']} cvar90={ts['cvar90']}")
    print(
        f"  D_PAC constant/alternating prefix: {pac_constant['d_prefix']:.6g}/"
        f"{pac_alternating['d_prefix']:.6g}"
    )


if __name__ == "__main__":
    selftest()
