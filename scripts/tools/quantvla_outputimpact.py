#!/usr/bin/env python3
"""Shared mechanics for FP16-guided single-layer OutputImpact probes."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import types
from typing import Any, Iterable

import torch


ARTIFACT_KIND = "outputimpact_single_layer_dpac_v2"


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def install_fp16_bypass(
    named_modules: Iterable[tuple[str, torch.nn.Module]],
    *,
    module_type: type[torch.nn.Module],
) -> dict[str, torch.nn.Module]:
    """Add a probe-only exact FP16 route without changing deployment runtime.

    Hessian artifacts use identity transform packs, so the unwrapped Linear is
    exactly ``F.linear(x, original_weight, original_bias)``.  The bypass also
    disables A8 for non-intervened layers, making the all-bypass reference an
    exact check against the independently loaded original FP16 teacher.
    """
    result: dict[str, torch.nn.Module] = {}
    for qualified_name, module in named_modules:
        if not isinstance(module, module_type):
            continue
        if getattr(module, "_outputimpact_original_forward", None) is not None:
            raise RuntimeError(f"{qualified_name}: OutputImpact bypass already installed")
        if getattr(module, "_weight", None) is None:
            raise RuntimeError(f"{qualified_name}: FP16 weight was already released")
        original_forward = module.forward
        module._outputimpact_original_forward = original_forward
        module._outputimpact_fp16 = True

        def forward_with_bypass(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            if bool(self._outputimpact_fp16):
                if self._weight is None:
                    raise RuntimeError(f"{self.name}: OutputImpact FP16 weight unavailable")
                return torch.nn.functional.linear(x, self._weight, self.bias)
            return self._outputimpact_original_forward(x)

        module.forward = types.MethodType(forward_with_bypass, module)
        result[qualified_name] = module
    if not result:
        raise RuntimeError("OutputImpact found no quantized Linear modules")
    return result


def set_single_intervention(
    layers: dict[str, torch.nn.Module], target: str | None
) -> None:
    if target is not None and target not in layers:
        raise KeyError(f"unknown OutputImpact target: {target}")
    for name, module in layers.items():
        module._outputimpact_fp16 = name != target


def identity_check(
    teacher_actions: Any,
    bypass_actions: Any,
    *,
    atol: float = 2e-3,
    rtol: float = 2e-3,
) -> dict[str, float | bool]:
    teacher = torch.as_tensor(teacher_actions, dtype=torch.float32, device="cpu")
    bypass = torch.as_tensor(bypass_actions, dtype=torch.float32, device="cpu")
    if teacher.shape != bypass.shape:
        raise ValueError(f"OutputImpact identity shape mismatch: {teacher.shape} != {bypass.shape}")
    delta = (teacher - bypass).abs()
    maximum = float(delta.max()) if delta.numel() else 0.0
    mean = float(delta.mean()) if delta.numel() else 0.0
    passed = bool(torch.allclose(teacher, bypass, atol=atol, rtol=rtol))
    if not passed:
        raise RuntimeError(
            "all-FP16 bypass does not reproduce the original teacher: "
            f"max_abs={maximum:.6g}, mean_abs={mean:.6g}"
        )
    return {
        "passed": True,
        "max_abs": maximum,
        "mean_abs": mean,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def impact_row(pair: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    pac = pair["d_pac_summary"]
    func = pair["d_func_summary"]
    value = float(pac["d_pac"])
    sequence = [float(item) for item in pac["per_sequence"]]
    if not math.isfinite(value) or value < 0.0 or not sequence:
        raise ValueError("invalid OutputImpact D_PAC result")
    return {
        "d_pac": value,
        "per_sequence": sequence,
        "d_pac_summary": pac,
        "d_func": float(func["d_func"]),
        "d_func_summary": func,
        "elapsed_s": float(elapsed_s),
    }


def shard_names(names: list[str], index: int, count: int) -> list[str]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid OutputImpact shard")
    return [name for position, name in enumerate(names) if position % count == index]
