#!/usr/bin/env python3
"""Table-1 byte anchors and the full-context variable budget.

The Table-1 ``storage`` cells are theoretical tightly packed component
bytes over the paper scope (LLM+DiT / pi0.5 expert Linear weights and
biases). The frozen official-50 audited anchors are:

* GR00T: FP16 2,139,537,408 (1.993 GiB), QuantVLA W4A8 963,772,416 (0.898 GiB)
* pi0.5: FP16 4,416,602,112 (4.113 GiB), QuantVLA W4A8 1,490,466,816 (1.388 GiB)

The full-context flip inventory (candidate scope) covers
1,812,185,088 bytes for GR00T and the full 4,416,602,112 bytes for pi0.5.
Fixed bytes = Table-1 FP16 - candidate FP16 (non-candidate FP16 layers
plus biases). The v2 byte ceiling is ``floor(1.10 * QuantVLA anchor)``
over the total static scope, so the variable (candidate) budget is::

    variable_budget = floor(1.10 * anchor) - fixed_bytes

This deliberately replaces the v1 rule ``floor(1.10 * all-W4 candidate
bytes)``, which anchored the budget on the candidate matrix instead of
the Table-1 QuantVLA row.
"""

from __future__ import annotations

import math

TABLE1_FP16_BYTES = {"gr00t": 2_139_537_408, "pi05": 4_416_602_112}
TABLE1_QUANTVLA_BYTES = {"gr00t": 963_772_416, "pi05": 1_490_466_816}
CANDIDATE_FP16_BYTES = {"gr00t": 1_812_185_088, "pi05": 4_416_602_112}
MULTIPLIER = 1.1
DISPLAY_TOLERANCE_GIB = 0.0005  # Table cells are displayed with 3 decimals

MODELS = ("gr00t", "pi05")


def require_model(model: str) -> None:
    if model not in MODELS:
        raise ValueError(f"unknown model for Table-1 byte accounting: {model}")


def fixed_bytes(model: str) -> int:
    """Bytes outside the candidate scope: non-candidate FP16 layers + biases."""
    require_model(model)
    return TABLE1_FP16_BYTES[model] - CANDIDATE_FP16_BYTES[model]


def table1_total_static_budget(model: str, multiplier: float = MULTIPLIER) -> int:
    """The Table-1 total-static ceiling: floor(multiplier * QuantVLA anchor)."""
    require_model(model)
    return int(math.floor(multiplier * TABLE1_QUANTVLA_BYTES[model]))


def table1_variable_budget(model: str, multiplier: float = MULTIPLIER) -> int:
    """Budget over the candidate (variable) scope under the Table-1 ceiling."""
    require_model(model)
    return table1_total_static_budget(model, multiplier) - fixed_bytes(model)


def table1_total_static_bytes(model: str, candidate_plan_bytes: int) -> int:
    """Total static component bytes implied by a candidate-scope plan total."""
    require_model(model)
    return fixed_bytes(model) + int(candidate_plan_bytes)


def table1_total_static_compression(model: str, candidate_plan_bytes: int) -> float:
    """Compression of the Table-1 total-static scope vs the FP16 row."""
    require_model(model)
    return TABLE1_FP16_BYTES[model] / table1_total_static_bytes(model, candidate_plan_bytes)


def table1_display_gib(value: int) -> str:
    """Render bytes the way the paper table cells do (3 decimals, GiB)."""
    return f"{float(value) / 2**30:.3f}"


def selftest() -> None:
    assert fixed_bytes("gr00t") == 327_352_320
    assert fixed_bytes("pi05") == 0
    assert table1_variable_budget("gr00t") == 732_797_337
    assert table1_variable_budget("pi05") == 1_639_513_497
    assert table1_total_static_budget("gr00t") == 1_060_149_657
    assert table1_total_static_budget("pi05") == 1_639_513_497
    assert table1_display_gib(TABLE1_FP16_BYTES["gr00t"]) == "1.993"
    assert table1_display_gib(TABLE1_QUANTVLA_BYTES["gr00t"]) == "0.898"
    assert table1_display_gib(TABLE1_FP16_BYTES["pi05"]) == "4.113"
    assert table1_display_gib(TABLE1_QUANTVLA_BYTES["pi05"]) == "1.388"
    assert table1_total_static_bytes("gr00t", 0) == 327_352_320
    assert abs(table1_total_static_compression("gr00t", 732_797_337) - 2.017) < 0.01
    print("[table1-bytes] selftest OK")


if __name__ == "__main__":
    selftest()