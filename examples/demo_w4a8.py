#!/usr/bin/env python3
"""Self-contained DyPAC-VLA demo: Hessian-aware group-64 W4, DyRange-A8, and D-PAC.

Runs on CPU with no model, dataset, checkpoint, or simulator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from quantvla_hessian_w4 import (  # noqa: E402
    GROUP_SIZE,
    a8_scale_table,
    hessian_aware_w4,
    pack_signed_nibbles,
    unpack_signed_nibbles,
)
from quantvla_metric_protocol import ACTION_DIM, ACTION_HORIZON, summarize_pair  # noqa: E402
from quantvla_table1_bytes import (  # noqa: E402
    TABLE1_FP16_BYTES,
    TABLE1_QUANTVLA_BYTES,
    table1_display_gib,
    table1_total_static_compression,
    table1_variable_budget,
)


def demo_hessian_w4() -> tuple[torch.Tensor, torch.Tensor, object]:
    """Quantize a synthetic Linear weight with the frozen group-64 Hessian W4 routine."""
    torch.manual_seed(0)
    out_features, in_features, tokens = 96, 128, 512
    weight = torch.randn(out_features, in_features) * 0.02
    calibration = torch.randn(tokens, in_features)

    result = hessian_aware_w4(weight, calibration)
    packed = pack_signed_nibbles(result.codes)
    assert torch.equal(unpack_signed_nibbles(packed, in_features), result.codes)

    fp16_bytes = weight.numel() * 2
    code_bytes = packed.numel()
    scale_bytes = result.scales.numel() * 4
    relative = (result.dequantized - weight).norm() / weight.norm()

    print("== Hessian-aware group-64 W4 ==")
    print(f"weight shape              {tuple(weight.shape)}  groups={result.group_size}")
    print(f"FP16 weight bytes         {fp16_bytes:,}")
    print(f"packed signed-nibble W4   {code_bytes:,}  ({fp16_bytes / code_bytes:.2f}x smaller)")
    print(f"per-group scales (fp32)   {scale_bytes:,}")
    print(f"weight relative L2 error  {relative:.5f}")
    print(f"clipping ratios selected  {sorted(set(result.clipping.flatten().tolist()))}")
    print()
    return weight, result, calibration


def demo_dyrange_a8(weight: torch.Tensor, result: object, calibration: torch.Tensor) -> None:
    """Compare one static A8 table with deterministic per-flow-step DyRange-A8 tables."""
    torch.manual_seed(7)
    flow_steps, tokens = 4, 256
    # Flow steps do not share a dynamic range: amplitude drifts across the integration steps.
    # A single static table must cover the pooled range; DyRange-A8 adapts per forward.
    step_amplitude = torch.tensor([1.0, 1.5, 2.2, 3.0]).reshape(flow_steps, 1, 1)
    activations = torch.randn(flow_steps, tokens, weight.shape[1]) * step_amplitude

    static_scale = a8_scale_table(activations.reshape(-1, weight.shape[1]))
    step_scale = a8_scale_table(activations, flow_steps=flow_steps)

    def activation_error(x: torch.Tensor, scale: torch.Tensor) -> float:
        codes = torch.clamp(torch.round(x / scale), -127.0, 127.0)
        return float(((codes * scale) - x).norm() / x.norm())

    static_errors = [activation_error(activations[step], static_scale) for step in range(flow_steps)]
    step_errors = [activation_error(activations[step], step_scale[step]) for step in range(flow_steps)]
    static_mean = sum(static_errors) / flow_steps
    step_mean = sum(step_errors) / flow_steps

    print("== DyRange-A8 (deterministic per-forward activation scales) ==")
    print("activation-side relative L2 error per flow step (A8 input quantization):")
    print(f"  static A8 table    {[f'{value:.4f}' for value in static_errors]}  mean {static_mean:.4f}")
    print(f"  DyRange-A8 table   {[f'{value:.4f}' for value in step_errors]}  mean {step_mean:.4f}")
    print(f"  mean error ratio   DyRange-A8 / static = {step_mean / static_mean:.3f}")
    print("  (end-to-end W4A8 output error is dominated by the W4 weight error at this scale)")
    print()


def demo_d_pac() -> None:
    """Report D-PAC / D_func on a perturbed action chunk under the frozen metric protocol."""
    torch.manual_seed(1)
    replans = 8
    reference = torch.randn(replans, ACTION_HORIZON, ACTION_DIM) * 0.1
    candidate = reference.clone()
    candidate[..., 0] += 0.01
    records = [{"task": "demo", "seed": index // 4, "replan": index % 4} for index in range(replans)]

    summary = summarize_pair(reference, candidate, records)
    identical = summarize_pair(reference, reference, records)

    print("== D-PAC / D_func action-chunk metric ==")
    print(f"chunk shape               ({replans}, {ACTION_HORIZON}, {ACTION_DIM})  (replans, horizon, dim)")
    print(f"identical chunk           D_func={identical['d_func_summary']['d_func']:.6f}"
          f"  D_PAC={identical['d_pac_summary']['d_pac']:.6f}")
    print(f"perturbed chunk (+0.01 m) D_func={summary['d_func_summary']['d_func']:.6f}"
          f"  D_PAC={summary['d_pac_summary']['d_pac']:.6f}")
    print()


def demo_storage() -> None:
    """Print the paper's exact static byte accounting anchors."""
    print("== Static byte accounting (paper Table 1 scope) ==")
    for model in ("gr00t", "pi05"):
        fp16 = TABLE1_FP16_BYTES[model]
        anchor = TABLE1_QUANTVLA_BYTES[model]
        budget = table1_variable_budget(model)
        compression = table1_total_static_compression(model, budget)
        print(f"{model:6s} FP16 {table1_display_gib(fp16)} GiB"
              f" | QuantVLA anchor {table1_display_gib(anchor)} GiB"
              f" | variable budget {budget:,} B"
              f" | compression at budget {compression:.3f}x")
    print()


def main() -> None:
    weight, result, calibration = demo_hessian_w4()
    demo_dyrange_a8(weight, result, calibration)
    demo_d_pac()
    demo_storage()
    print(f"done (group size {GROUP_SIZE})")


if __name__ == "__main__":
    main()
