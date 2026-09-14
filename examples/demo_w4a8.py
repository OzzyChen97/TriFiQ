#!/usr/bin/env python3
"""CPU-only demonstration of the public DyPAC-VLA Method core."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from dypac_vla_protocol import RIPA_ALPHA  # noqa: E402
from quantvla_dynamic_a8_protocol import dyrange_a8  # noqa: E402
from quantvla_hessian_w4 import (  # noqa: E402
    GROUP_SIZE,
    hessian_aware_w4,
    pack_signed_nibbles,
    unpack_signed_nibbles,
)
from quantvla_metric_protocol import dpac_sequence  # noqa: E402
from quantvla_ripa import aggregate_probe_bank, probe_layer  # noqa: E402
from quantvla_selector import LayerChoice, allocate_ripa_mask  # noqa: E402
from quantvla_table1_bytes import result_row  # noqa: E402


def demo_w4() -> None:
    torch.manual_seed(0)
    weight = torch.randn(16, 64) * 0.02
    reference_inputs = torch.randn(128, 64)
    result = hessian_aware_w4(weight, reference_inputs)
    packed = pack_signed_nibbles(result.codes)
    assert torch.equal(unpack_signed_nibbles(packed, 64), result.codes)
    relative = float((result.dequantized - weight).norm() / weight.norm())
    print("== Hessian-aware group-64 W4 ==")
    print(f"group size: {result.group_size}")
    print(f"packed code bytes: {packed.numel():,}")
    print(f"FP32 scale bytes: {result.scales.numel() * 4:,}")
    print(f"weight relative L2 error: {relative:.6f}\n")


def demo_ripa_and_budget() -> None:
    generator = torch.Generator().manual_seed(1)
    action = torch.randn(24, 8, generator=generator)
    layer = torch.randn(24, 12, generator=generator)
    rows = {
        "layer_a": probe_layer(
            reference_action_rows=action,
            intervention_action_rows=action + 0.01 * torch.randn(action.shape, generator=generator),
            reference_layer_rows=layer,
            intervention_layer_rows=layer * 1.10,
        ),
        "layer_b": probe_layer(
            reference_action_rows=action,
            intervention_action_rows=action + 0.03 * torch.randn(action.shape, generator=generator),
            reference_layer_rows=layer,
            intervention_layer_rows=layer * 1.25,
        ),
        "layer_c": probe_layer(
            reference_action_rows=action,
            intervention_action_rows=action + 0.02 * torch.randn(action.shape, generator=generator),
            reference_layer_rows=layer,
            intervention_layer_rows=layer * 1.15,
        ),
    }
    bank = aggregate_probe_bank(
        {name: float(row["relational_damage"]) for name, row in rows.items()},
        {name: float(row["distributional_damage"]) for name, row in rows.items()},
    )
    allocation = allocate_ripa_mask(
        [
            LayerChoice(name, extra, bank[name]["importance"], bool(rows[name]["protected"]))
            for name, extra in (("layer_a", 6), ("layer_b", 5), ("layer_c", 10))
        ],
        all_w4_component_bytes=100,
        budget_bytes=111,
    )
    print("== RIPA dual-site probes and exact-byte allocation ==")
    print(f"alpha: {RIPA_ALPHA:.9f} (=16/17)")
    for name in sorted(bank):
        print(
            f"{name}: relational={bank[name]['relational_damage']:.6f} "
            f"distributional={bank[name]['distributional_damage']:.6f} "
            f"importance={bank[name]['importance']:.6f}"
        )
    print(f"native layers: {allocation.native_layers}")
    print(f"W4 layers: {allocation.w4_layers}")
    print(f"attained component bytes: {allocation.total_component_bytes}/{allocation.budget_bytes}\n")


def demo_dpac() -> None:
    reference = torch.zeros(3, 4, 2)
    candidate = reference.clone()
    candidate[:, :2, 0] = 0.2
    score = dpac_sequence(
        reference,
        candidate,
        [0, 1, 2],
        scale=torch.ones(2),
        execution_prefix=2,
    )
    print("== D-PAC over executed prefixes ==")
    print(f"executed steps: {score['executed_steps']}")
    print(f"immediate loss: {score['immediate_loss']:.6f}")
    print(f"accumulated loss: {score['accumulated_loss']:.6f}")
    print(f"final cumulative target: {score['cumulative_targets'][-1]}\n")


def demo_dyrange() -> None:
    first = torch.randn(2, 8, 4) * torch.tensor([1.0, 2.0, 4.0, 8.0])
    second = torch.randn(1, 3, 4) * torch.tensor([8.0, 4.0, 2.0, 1.0])
    first_q, first_scale, _ = dyrange_a8(first)
    _, second_scale, _ = dyrange_a8(second)
    error = float((first_q - first).norm() / first.norm())
    print("== DyRange-A8 per-forward input-channel ranges ==")
    print(f"first input scales: {[round(float(value), 6) for value in first_scale]}")
    print(f"second input scales: {[round(float(value), 6) for value in second_scale]}")
    print(f"first-input relative L2 error: {error:.6f}\n")


def demo_storage() -> None:
    print("== Audited Linear-component storage scope ==")
    for model in ("pi05", "gr00t"):
        for point in ("selected", "all_candidate_w4"):
            row = result_row(model, point)
            print(
                f"{model:5s} {point:16s} {row['component_bytes']:,} B "
                f"({row['component_gib']} GiB, {row['component_compression']:.2f}x)"
            )
    print()


def main() -> None:
    demo_w4()
    demo_ripa_and_budget()
    demo_dpac()
    demo_dyrange()
    demo_storage()
    print(f"done (W4 group size {GROUP_SIZE})")


if __name__ == "__main__":
    main()