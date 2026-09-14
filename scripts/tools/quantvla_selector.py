#!/usr/bin/env python3
"""Exact-byte RIPA allocation and constrained DyPAC-VLA mask selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

from dypac_vla_protocol import PROTOCOL


CONTROLLED_BOUNDS = PROTOCOL["d_pac"]["controlled_libero_bounds"]
DEFAULT_DELTA_R = float(CONTROLLED_BOUNDS["delta_R"])
DEFAULT_DELTA_A = float(CONTROLLED_BOUNDS["delta_A"])
DEFAULT_TIE_TOLERANCE = float(PROTOCOL["d_pac"]["tie_tolerance"])


@dataclass(frozen=True)
class LayerChoice:
    name: str
    native_extra_bytes: int
    importance: float
    protected: bool = False


@dataclass(frozen=True)
class AllocationResult:
    native_layers: tuple[str, ...]
    w4_layers: tuple[str, ...]
    all_w4_component_bytes: int
    total_component_bytes: int
    budget_bytes: int
    local_loss: float


def packed_w4_component_bytes(
    *,
    out_features: int,
    in_features: int,
    bias_elements: int = 0,
    group_size: int = 64,
    scale_bytes: int = 4,
    metadata_bytes: int = 0,
) -> int:
    """Packed W4 weights, FP32 group scales, native-16 bias, and explicit metadata."""
    if min(out_features, in_features) <= 0 or bias_elements < 0 or group_size <= 0:
        raise ValueError("invalid Linear component shape")
    packed_weights = int(out_features) * ((int(in_features) + 1) // 2)
    groups = (int(in_features) + int(group_size) - 1) // int(group_size)
    scales = int(out_features) * groups * int(scale_bytes)
    return packed_weights + scales + int(bias_elements) * 2 + int(metadata_bytes)


def native_16bit_component_bytes(
    *, out_features: int, in_features: int, bias_elements: int = 0, metadata_bytes: int = 0
) -> int:
    if min(out_features, in_features) <= 0 or bias_elements < 0:
        raise ValueError("invalid Linear component shape")
    return (
        int(out_features) * int(in_features) * 2
        + int(bias_elements) * 2
        + int(metadata_bytes)
    )


def native_extra_bytes_from_shape(
    *,
    out_features: int,
    in_features: int,
    bias_elements: int = 0,
    group_size: int = 64,
    w4_metadata_bytes: int = 0,
    native_metadata_bytes: int = 0,
) -> int:
    """Exact delta_C for changing one layer from packed W4 to native 16-bit."""
    packed = packed_w4_component_bytes(
        out_features=out_features,
        in_features=in_features,
        bias_elements=bias_elements,
        group_size=group_size,
        metadata_bytes=w4_metadata_bytes,
    )
    native = native_16bit_component_bytes(
        out_features=out_features,
        in_features=in_features,
        bias_elements=bias_elements,
        metadata_bytes=native_metadata_bytes,
    )
    increment = native - packed
    if increment <= 0:
        raise ValueError("native precision must have a positive byte increment over W4")
    return increment


def _better(
    candidate: tuple[float, tuple[str, ...]], current: tuple[float, tuple[str, ...]]
) -> bool:
    if candidate[0] > current[0] + 1e-15:
        return True
    if abs(candidate[0] - current[0]) <= 1e-15:
        return (len(candidate[1]), candidate[1]) < (len(current[1]), current[1])
    return False


def allocate_ripa_mask(
    layers: Iterable[LayerChoice], *, all_w4_component_bytes: int, budget_bytes: int
) -> AllocationResult:
    """Solve the paper's integer-byte allocation without layer-count or ratio proxies."""
    ordered = sorted(layers, key=lambda row: row.name)
    if not ordered or len({row.name for row in ordered}) != len(ordered):
        raise ValueError("RIPA layers must be non-empty and uniquely named")
    if not isinstance(all_w4_component_bytes, int) or not isinstance(budget_bytes, int):
        raise ValueError("component byte counts must be integers")
    if all_w4_component_bytes < 0 or budget_bytes < all_w4_component_bytes:
        raise ValueError("the byte budget cannot hold the all-W4 component inventory")
    for row in ordered:
        if not isinstance(row.native_extra_bytes, int) or row.native_extra_bytes <= 0:
            raise ValueError(f"{row.name}: native byte increment must be a positive integer")
        if not math.isfinite(row.importance) or row.importance < 0.0:
            raise ValueError(f"{row.name}: importance must be finite and non-negative")

    capacity = int(budget_bytes) - int(all_w4_component_bytes)
    mandatory = tuple(row.name for row in ordered if row.protected)
    mandatory_cost = sum(row.native_extra_bytes for row in ordered if row.protected)
    if mandatory_cost > capacity:
        raise ValueError("the byte budget cannot accommodate the protected RIPA set")

    optional = [row for row in ordered if not row.protected]
    states: dict[int, tuple[float, tuple[str, ...]]] = {mandatory_cost: (0.0, mandatory)}
    for row in optional:
        updated = dict(states)
        for cost, state in states.items():
            new_cost = cost + row.native_extra_bytes
            if new_cost > capacity:
                continue
            candidate = (state[0] + row.importance, tuple(sorted(state[1] + (row.name,))))
            previous = updated.get(new_cost)
            if previous is None or _better(candidate, previous):
                updated[new_cost] = candidate

        frontier: dict[int, tuple[float, tuple[str, ...]]] = {}
        best_benefit = -math.inf
        for cost in sorted(updated):
            state = updated[cost]
            if state[0] > best_benefit + 1e-15:
                frontier[cost] = state
                best_benefit = state[0]
        states = frontier

    best_cost, best = min(
        states.items(),
        key=lambda pair: (-pair[1][0], pair[0], len(pair[1][1]), pair[1][1]),
    )
    native = tuple(sorted(best[1]))
    native_set = set(native)
    w4 = tuple(row.name for row in ordered if row.name not in native_set)
    local_loss = sum(row.importance for row in ordered if row.name in w4)
    return AllocationResult(
        native_layers=native,
        w4_layers=w4,
        all_w4_component_bytes=int(all_w4_component_bytes),
        total_component_bytes=int(all_w4_component_bytes) + int(best_cost),
        budget_bytes=int(budget_bytes),
        local_loss=float(local_loss),
    )


def equal_byte_exchange(
    anchor: AllocationResult,
    layers: Sequence[LayerChoice],
    *,
    quantize_native_layers: Iterable[str],
    restore_w4_layers: Iterable[str],
) -> tuple[str, ...]:
    """Construct one candidate only when its exact component bytes equal the anchor."""
    by_name = {row.name: row for row in layers}
    if len(by_name) != len(layers):
        raise ValueError("layer names must be unique")
    quantize = set(quantize_native_layers)
    restore = set(restore_w4_layers)
    anchor_native = set(anchor.native_layers)
    anchor_w4 = set(anchor.w4_layers)
    if anchor_native & anchor_w4 or set(by_name) != anchor_native | anchor_w4:
        raise ValueError("anchor mask and layer inventory differ")
    if not quantize or not restore:
        raise ValueError("an equal-byte exchange must change both mask states")
    if not quantize.issubset(anchor_native) or not restore.issubset(anchor_w4):
        raise ValueError("exchange layers must use their anchor mask states")
    if any(by_name[name].protected for name in quantize):
        raise ValueError("a protected RIPA layer cannot be quantized")
    candidate_native = (anchor_native - quantize) | restore
    candidate_bytes = anchor.all_w4_component_bytes + sum(
        by_name[name].native_extra_bytes for name in candidate_native
    )
    if candidate_bytes != anchor.total_component_bytes:
        raise ValueError(
            "exchange is not byte matched: "
            f"candidate={candidate_bytes}, anchor={anchor.total_component_bytes}"
        )
    return tuple(sorted(candidate_native))


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    component_bytes: int
    native_layers: tuple[str, ...]
    local_loss: float
    immediate_loss: float
    accumulated_loss: float


def select_mask(
    candidates: Sequence[CandidateScore],
    *,
    anchor_id: str,
    protected_layers: Iterable[str] = (),
    delta_r: float = DEFAULT_DELTA_R,
    delta_a: float = DEFAULT_DELTA_A,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Minimize accumulated executed-prefix deviation under the paper's constraints."""
    by_id = {row.candidate_id: row for row in candidates}
    if not candidates or len(by_id) != len(candidates) or anchor_id not in by_id:
        raise ValueError("candidate ids must be unique and include the anchor")
    if delta_r < 0.0 or delta_a < 0.0 or tie_tolerance < 0.0:
        raise ValueError("selection tolerances must be non-negative")
    for row in candidates:
        values = (row.local_loss, row.immediate_loss, row.accumulated_loss)
        if (
            not isinstance(row.component_bytes, int)
            or row.component_bytes < 0
            or len(set(row.native_layers)) != len(row.native_layers)
            or not all(math.isfinite(value) and value >= 0.0 for value in values)
        ):
            raise ValueError(f"{row.candidate_id}: invalid score row")

    anchor = by_id[anchor_id]
    required = set(protected_layers)
    feasible: list[CandidateScore] = []
    rejected: dict[str, list[str]] = {}
    for row in candidates:
        reasons = []
        if row.component_bytes != anchor.component_bytes:
            reasons.append("component_bytes_differ_from_anchor")
        if not required.issubset(set(row.native_layers)):
            reasons.append("protected_layer_was_quantized")
        if row.local_loss > (1.0 + delta_r) * anchor.local_loss + 1e-15:
            reasons.append("ripa_local_bound")
        if row.immediate_loss > (1.0 + delta_a) * anchor.immediate_loss + 1e-15:
            reasons.append("immediate_action_bound")
        if reasons:
            rejected[row.candidate_id] = reasons
        else:
            feasible.append(row)
    if anchor not in feasible:
        raise ValueError("the anchor must remain a feasible fallback")

    minimum = min(row.accumulated_loss for row in feasible)
    tied = [row for row in feasible if row.accumulated_loss <= minimum + tie_tolerance]
    selected = anchor if anchor in tied else min(tied, key=lambda row: row.candidate_id)
    return {
        "selected_id": selected.candidate_id,
        "anchor_id": anchor_id,
        "component_bytes": anchor.component_bytes,
        "feasible_ids": [row.candidate_id for row in sorted(feasible, key=lambda row: row.candidate_id)],
        "rejected": rejected,
        "objective": "accumulated_executed_prefix_deviation",
        "selected_accumulated_loss": selected.accumulated_loss,
    }


def selftest() -> None:
    from quantvla_table1_bytes import validate_paper_values

    validate_paper_values()
    # The exact-byte optimum uses 6+5 bytes. A layer-count or rounded-ratio rule
    # cannot represent this packed-byte choice faithfully.
    result = allocate_ripa_mask(
        [
            LayerChoice("a", 6, 10.0),
            LayerChoice("b", 5, 8.0),
            LayerChoice("c", 10, 15.0),
        ],
        all_w4_component_bytes=100,
        budget_bytes=111,
    )
    assert result.native_layers == ("a", "b")
    assert result.total_component_bytes == 111
    assert result.local_loss == 15.0

    exchange_layers = [LayerChoice("keep", 5, 1.0), LayerChoice("restore", 5, 2.0)]
    exchange_anchor = AllocationResult(
        native_layers=("keep",),
        w4_layers=("restore",),
        all_w4_component_bytes=20,
        total_component_bytes=25,
        budget_bytes=25,
        local_loss=2.0,
    )
    assert equal_byte_exchange(
        exchange_anchor,
        exchange_layers,
        quantize_native_layers=("keep",),
        restore_w4_layers=("restore",),
    ) == ("restore",)

    packed = packed_w4_component_bytes(out_features=3, in_features=65, bias_elements=3)
    native = native_16bit_component_bytes(out_features=3, in_features=65, bias_elements=3)
    assert packed == 3 * 33 + 3 * 2 * 4 + 3 * 2
    assert native > packed
    assert native_extra_bytes_from_shape(
        out_features=3, in_features=65, bias_elements=3
    ) == native - packed

    decision = select_mask(
        [
            CandidateScore("anchor", 111, ("a", "b"), 1.0, 2.0, 3.0),
            CandidateScore("best", 111, ("a", "b"), 1.04, 2.05, 2.0),
            CandidateScore("wrong-bytes", 110, ("a", "b"), 0.5, 0.5, 0.5),
        ],
        anchor_id="anchor",
        protected_layers=("a",),
    )
    assert decision["selected_id"] == "best"
    assert decision["rejected"]["wrong-bytes"] == ["component_bytes_differ_from_anchor"]
    print("[quantvla-selector] selftest OK")


if __name__ == "__main__":
    selftest()