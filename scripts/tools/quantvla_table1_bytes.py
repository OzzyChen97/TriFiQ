#!/usr/bin/env python3
"""Audited Linear-component byte accounting used by the paper's result tables."""

from __future__ import annotations


FP16_COMPONENT_BYTES = {
    "gr00t": 2_139_537_408,
    "pi05": 4_416_602_112,
}

DYPAC_COMPONENT_BYTES = {
    "gr00t": {
        "selected": 962_314_240,
        "all_candidate_w4": 837_206_016,
    },
    "pi05": {
        "selected": 1_634_828_288,
        "all_candidate_w4": 1_242_169_344,
    },
}


def require_model(model: str) -> None:
    if model not in FP16_COMPONENT_BYTES:
        raise ValueError(f"unknown model: {model!r}")


def component_compression(model: str, component_bytes: int) -> float:
    """Compression only within the matched audited Linear-component inventory."""
    require_model(model)
    if int(component_bytes) <= 0:
        raise ValueError("component bytes must be positive")
    return FP16_COMPONENT_BYTES[model] / int(component_bytes)


def display_gib(component_bytes: int) -> str:
    if int(component_bytes) < 0:
        raise ValueError("component bytes must be non-negative")
    return f"{int(component_bytes) / 2**30:.3f}"


def result_row(model: str, operating_point: str) -> dict[str, float | int | str]:
    require_model(model)
    component_bytes = DYPAC_COMPONENT_BYTES[model][operating_point]
    return {
        "model": model,
        "operating_point": operating_point,
        "component_bytes": component_bytes,
        "component_gib": display_gib(component_bytes),
        "component_compression": component_compression(model, component_bytes),
        "storage_scope": "audited_linear_components",
    }


def validate_paper_values() -> None:
    assert display_gib(FP16_COMPONENT_BYTES["gr00t"]) == "1.993"
    assert display_gib(FP16_COMPONENT_BYTES["pi05"]) == "4.113"
    assert display_gib(DYPAC_COMPONENT_BYTES["gr00t"]["selected"]) == "0.896"
    assert display_gib(DYPAC_COMPONENT_BYTES["gr00t"]["all_candidate_w4"]) == "0.780"
    assert display_gib(DYPAC_COMPONENT_BYTES["pi05"]["selected"]) == "1.523"
    assert display_gib(DYPAC_COMPONENT_BYTES["pi05"]["all_candidate_w4"]) == "1.157"
    assert abs(component_compression("gr00t", 962_314_240) - 2.2233) < 1e-4
    assert abs(component_compression("pi05", 1_634_828_288) - 2.7016) < 1e-4


if __name__ == "__main__":
    validate_paper_values()
    print("[quantvla-component-bytes] values OK")