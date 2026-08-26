#!/usr/bin/env python3
"""GR00T-final functional metric with the minimal pi0.5 action adapter.

The metric itself is not reimplemented here.  The authoritative
``gr00t_func_metrics.d_func`` function is called directly after adapting the
pi0.5 trajectory to the action chunk that is actually deployed by the frozen
RoboCasa protocol:

* first 16 of pi0.5's 50 predicted actions (execute-16),
* first 12 action dimensions (the environment action, excluding 20 padding
  dimensions), and
* the embodiment-specific gripper slice ``6:7``.

The 12-D vector is deliberately retained for GR00T's final-action and tail
terms, so mobile-base/torso and control-mode deviations are not discarded.
They do not receive new hand-written kinematic or discrete penalties: the
only embodiment adaptation inside the GR00T formula is that pi0.5 has one
gripper dimension rather than GR00T's default two.  In particular, there is
no chunk-50 auxiliary loss and no ``16/50`` multiplier.
"""

from __future__ import annotations

from typing import Any

import torch

from gr00t_func_metrics import d_func as gr00t_final_d_func
from gr00t_func_metrics import d_pac_sequence as gr00t_d_pac_sequence


ACTION_DIM = 12
ACTION_HORIZON = 50
EXECUTED_ACTIONS = 16
FLOW_STEPS = 4
GR00T_LAYOUT = {"trans": (0, 3), "rot": (3, 6), "grip": (6, 7)}
GR00T_WEIGHTS = {"final": 1.0, "kin": 1.0, "grip": 1.0, "tail": 2.0}
FUNCTIONAL_FORMULA_ID = "gr00t_final_v1_4_execute16_deployed12_grip6to7"
PAC_FORMULA_ID = "d_pac_v1_pi05_execute16_deployed12_forecast50"


def adapt_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
    """Map a pi0.5 flow trajectory to GR00T-final action-chunk semantics."""
    if trajectory.ndim != 4:
        raise ValueError(
            f"trajectory must be (T+1,B,H,D), got {tuple(trajectory.shape)}"
        )
    if trajectory.shape[0] != FLOW_STEPS + 1:
        raise ValueError(
            f"formal pi0.5 metric requires {FLOW_STEPS} flow steps, "
            f"got trajectory length {trajectory.shape[0] - 1}"
        )
    if trajectory.shape[-2] != ACTION_HORIZON:
        raise ValueError(
            f"formal pi0.5 metric requires action horizon {ACTION_HORIZON}, "
            f"got {trajectory.shape[-2]}"
        )
    if trajectory.shape[-1] < ACTION_DIM:
        raise ValueError(
            f"trajectory action dimension {trajectory.shape[-1]} is smaller "
            f"than deployed dimension {ACTION_DIM}"
        )
    return trajectory[..., :EXECUTED_ACTIONS, :ACTION_DIM].float()


def d_func(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    gamma: float = 1.2,
) -> dict[str, Any]:
    """Apply the authoritative GR00T-final metric after layout adaptation."""
    adapted_reference = adapt_trajectory(reference)
    adapted_candidate = adapt_trajectory(candidate)
    if adapted_reference.shape != adapted_candidate.shape:
        raise ValueError(
            "paired trajectory shape mismatch after adaptation: "
            f"{tuple(adapted_reference.shape)} != {tuple(adapted_candidate.shape)}"
        )
    result = gr00t_final_d_func(
        adapted_reference,
        adapted_candidate,
        gamma=gamma,
        layout=GR00T_LAYOUT,
        weights=GR00T_WEIGHTS,
    )
    # Provenance only; no pi0.5-specific term is added to the scalar metric.
    result["adapter"] = {
        "formula_id": FUNCTIONAL_FORMULA_ID,
        "source_horizon": ACTION_HORIZON,
        "executed_actions": EXECUTED_ACTIONS,
        "deployed_action_dim": ACTION_DIM,
        "layout": dict(GR00T_LAYOUT),
        "excluded_horizon": [EXECUTED_ACTIONS, ACTION_HORIZON],
        "excluded_padding_dims": [ACTION_DIM, int(reference.shape[-1])],
    }
    return result


def d_pac_sequence(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    replan_indices,
    *,
    overlap_weight: float = 0.1,
    gamma: float = 1.2,
) -> dict[str, Any]:
    """π0.5 D_PAC adapter: execute-16 primary loss plus low-weight 16:50 overlap."""
    for name, trajectory in (("reference", reference), ("candidate", candidate)):
        if trajectory.ndim not in (3, 4):
            raise ValueError(f"{name} must be (R,H,D) or (T+1,R,H,D)")
        if trajectory.shape[-2] != ACTION_HORIZON:
            raise ValueError(
                f"formal pi0.5 D_PAC requires horizon {ACTION_HORIZON}, "
                f"got {trajectory.shape[-2]}"
            )
        if trajectory.shape[-1] < ACTION_DIM:
            raise ValueError(f"{name} has fewer than {ACTION_DIM} deployed dimensions")
    result = gr00t_d_pac_sequence(
        reference,
        candidate,
        replan_indices,
        executed_actions=EXECUTED_ACTIONS,
        action_dim=ACTION_DIM,
        layout=GR00T_LAYOUT,
        weights={"overlap": float(overlap_weight)},
        gamma=gamma,
    )
    result["adapter"] = {
        "formula_id": PAC_FORMULA_ID,
        "source_horizon": ACTION_HORIZON,
        "executed_actions": EXECUTED_ACTIONS,
        "forecast_overlap": [EXECUTED_ACTIONS, ACTION_HORIZON],
        "forecast_overlap_weight": float(overlap_weight),
        "deployed_action_dim": ACTION_DIM,
        "layout": dict(GR00T_LAYOUT),
    }
    return result


def selftest() -> None:
    generator = torch.Generator().manual_seed(0)
    reference = torch.randn(5, 8, 50, 32, generator=generator)

    same = d_func(reference, reference)
    assert same["d_func"] == 0.0 and same["d_solver"] == 0.0

    candidate = reference.clone()
    candidate[..., :EXECUTED_ACTIONS, :ACTION_DIM] *= 2.0
    direct = gr00t_final_d_func(
        adapt_trajectory(reference),
        adapt_trajectory(candidate),
        gamma=1.2,
        layout=GR00T_LAYOUT,
        weights=GR00T_WEIGHTS,
    )
    adapted = d_func(reference, candidate)
    for key in ("d_func", "d_final", "d_kin", "d_grip", "d_solver"):
        assert adapted[key] == direct[key], (key, adapted[key], direct[key])

    nonexecuted = reference.clone()
    nonexecuted[..., EXECUTED_ACTIONS:, :ACTION_DIM] *= 100.0
    assert d_func(reference, nonexecuted)["d_func"] == 0.0

    padding = reference.clone()
    padding[..., ACTION_DIM:] *= 100.0
    assert d_func(reference, padding)["d_func"] == 0.0

    base_only = reference.clone()
    base_only[-1, ..., :EXECUTED_ACTIONS, 7:11] *= 2.0
    base_result = d_func(reference, base_only)
    assert base_result["d_final"] > 0.0
    assert base_result["d_kin"] == 0.0
    assert base_result["d_grip"] == 0.0

    pac_same = d_pac_sequence(reference[:, :4], reference[:, :4], range(4))
    assert pac_same["d_pac_sequence"] == 0.0

    forecast_only = reference[:, :4].clone()
    forecast_only[..., EXECUTED_ACTIONS:ACTION_HORIZON, :ACTION_DIM] += 0.1
    pac_forecast = d_pac_sequence(reference[:, :4], forecast_only, range(4))
    assert pac_forecast["d_overlap"] > 0.0
    assert pac_forecast["d_func_mean"] == 0.0

    print("[pi05_func_metrics] selftest OK (D_func adapter + D_PAC execute16/forecast50)")


if __name__ == "__main__":
    selftest()
