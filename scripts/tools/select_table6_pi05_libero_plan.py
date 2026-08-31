#!/usr/bin/env python3
"""Select a pi0.5 W4/FP16 compression-anchor mask on LIBERO calibration data.

The selector is result blind.  It records input second moments on FP16 and
uses activation-weighted W4 reconstruction error to retain the most fragile
Linear layers in FP16 under the exact uniform-W6 candidate-weight budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / "code/pi05/openpi"
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "packages/openpi-client/src"))
sys.path.insert(0, str(ROOT / "scripts/tools"))

from openpi.policies import policy_config  # noqa: E402
from openpi.training import config  # noqa: E402
from pi05_batched_policy import iter_batches, load_records, sample_batch  # noqa: E402


CHECKPOINT = ROOT / "code/pi05/checkpoints/pi05_libero_pytorch"
CHECKPOINT_SHA256 = "0f8c489e37b01c72251c45f2e73595894f3933fc6297f4f1cf95fc8737db4c74"
TEMPLATE = (
    ROOT
    / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_quantvla_uniform_w4a8_d4.plan.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-obs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--max-rows-per-call", type=int, default=512)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SecondMoment:
    def __init__(self, max_rows: int) -> None:
        self.max_rows = max_rows
        self.sumsq: torch.Tensor | None = None
        self.rows = 0
        self.calls = 0

    def __call__(self, _module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        value = inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1])
        if value.shape[0] > self.max_rows:
            indices = torch.linspace(
                0, value.shape[0] - 1, self.max_rows, device=value.device
            ).round().long()
            value = value.index_select(0, indices)
        partial = value.square().sum(dim=0).cpu()
        self.sumsq = partial if self.sumsq is None else self.sumsq + partial
        self.rows += int(value.shape[0])
        self.calls += 1

    def mean(self) -> torch.Tensor:
        if self.sumsq is None or self.rows <= 0:
            raise RuntimeError("empty activation second moment")
        return (self.sumsq / self.rows).clamp_min_(1e-12)


def weighted_w4_error(weight: torch.Tensor, second_moment: torch.Tensor) -> tuple[float, float]:
    value = weight.detach().to(device="cpu", dtype=torch.float32)
    moment = second_moment.to(dtype=torch.float32)
    if value.shape[1] != moment.numel():
        raise ValueError(f"weight/activation shape mismatch: {value.shape}, {moment.shape}")
    total_error = 0.0
    total_reference = 0.0
    for start in range(0, value.shape[0], 256):
        block = value[start : start + 256]
        maximum = block.abs().amax(dim=1, keepdim=True)
        scale = (maximum / 7.0).clamp_min_(1e-8)
        quantized = torch.clamp(torch.round(block / scale), -8, 7) * scale
        total_error += float(((block - quantized).square() * moment).sum())
        total_reference += float((block.square() * moment).sum())
    return total_error, total_reference


def main() -> None:
    args = parse_args()
    if args.n_obs != 16 or args.batch_size != 4 or args.flow_steps != 8:
        raise ValueError("Table 6 selector is frozen to n_obs=16, batch=4, flow_steps=8")
    checkpoint_file = CHECKPOINT / "model.safetensors"
    if sha256_file(checkpoint_file) != CHECKPOINT_SHA256:
        raise ValueError("pi0.5 LIBERO checkpoint drift")
    buffer = Path(args.buffer).expanduser().resolve()
    buffer_meta = json.loads(Path(str(buffer) + ".json").read_text(encoding="utf-8"))
    if (
        buffer_meta.get("overlap_with_held_out") is not False
        or buffer_meta.get("test_rollout_feedback_used") is not False
    ):
        raise ValueError("selector buffer is not result blind")
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    candidate_names = sorted((template.get("layers") or {}).keys())
    if len(candidate_names) != 180:
        raise ValueError(f"candidate inventory drift: {len(candidate_names)}")

    records = load_records(buffer, args.n_obs)
    torch.manual_seed(0)
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_libero"), CHECKPOINT, pytorch_device=args.device
    )
    modules = dict(policy._model.named_modules())
    missing = sorted(set(candidate_names) - set(modules))
    if missing:
        raise RuntimeError(f"checkpoint misses {len(missing)} candidate layers")
    moments = {
        name: SecondMoment(args.max_rows_per_call) for name in candidate_names
    }
    handles = [
        modules[name].register_forward_pre_hook(moments[name]) for name in candidate_names
    ]
    action_horizon = int(policy._model.config.action_horizon)
    action_dim = int(policy._model.config.action_dim)
    try:
        for batch_index, batch in enumerate(iter_batches(records, args.batch_size), start=1):
            actions = sample_batch(
                policy, batch, args.device, num_steps=args.flow_steps
            )
            if tuple(actions.shape) != (args.batch_size, action_horizon, action_dim):
                raise RuntimeError(f"invalid calibration action shape: {tuple(actions.shape)}")
            print(f"[table6 pi05 selector] activation batch {batch_index}/4", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    rows = []
    total_parameters = 0
    for index, name in enumerate(candidate_names, start=1):
        module = modules[name]
        error, reference = weighted_w4_error(module.weight, moments[name].mean())
        parameters = int(module.weight.numel())
        relative = error / max(reference, 1e-20)
        savings = 1.5 * parameters
        rows.append(
            {
                "name": name,
                "parameters": parameters,
                "activation_weighted_error": error,
                "activation_weighted_reference": reference,
                "relative_w4_error": relative,
                "loss_per_saved_byte": relative / savings,
                "activation_rows": moments[name].rows,
                "activation_calls": moments[name].calls,
            }
        )
        total_parameters += parameters
        print(
            f"[table6 pi05 selector] score {index}/180 {name} rel={relative:.6g}",
            flush=True,
        )

    # FP16 costs two bytes/parameter and idealized uniform W6 costs 0.75.
    # Replacing a layer by W4 saves 1.5 bytes/parameter.  Greedily minimize
    # calibration loss per saved byte until the W6 anchor is met.
    fp16_bytes = 2.0 * total_parameters
    budget_bytes = 0.75 * total_parameters
    required_savings = fp16_bytes - budget_bytes
    selected: set[str] = set()
    accumulated_savings = 0.0
    for row in sorted(rows, key=lambda item: (item["loss_per_saved_byte"], item["name"])):
        selected.add(row["name"])
        accumulated_savings += 1.5 * row["parameters"]
        if accumulated_savings >= required_savings:
            break
    deployed_bytes = fp16_bytes - accumulated_savings
    layers = {
        name: {
            "bits": 4 if name in selected else 0,
            "group": 64,
            "skip": name not in selected,
            "reason": (
                "activation_weighted_w4_under_w6_anchor"
                if name in selected
                else "retained_fp16_by_libero_calibration"
            ),
        }
        for name in candidate_names
    }
    payload = {
        "schema_version": 1,
        "meta": {
            "kind": "table6_pi05_libero_compression_anchor_plan",
            "model": "pi05",
            "suite": args.suite,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "flow_steps": args.flow_steps,
            "selector": "activation_weighted_w4_reconstruction",
            "selector_role": "result_blind_compression_anchor",
            "uses_cka": False,
            "uses_cs": False,
            "uses_task_success": False,
            "uses_test_rollout_feedback": False,
            "calibration_buffer": str(buffer),
            "calibration_buffer_sha256": sha256_file(buffer),
            "calibration_initial_state_indices": buffer_meta["initial_state_indices"],
            "held_out_initial_state_indices": buffer_meta[
                "held_out_initial_state_indices"
            ],
            "n_obs": args.n_obs,
            "candidate_layers": len(candidate_names),
            "quantized_layers": len(selected),
            "retained_fp16_layers": len(candidate_names) - len(selected),
            "budget_reference": "uniform_w6_candidate_weight_bytes",
            "fp16_candidate_weight_bytes": fp16_bytes,
            "budget_bytes": budget_bytes,
            "deployed_candidate_weight_bytes": deployed_bytes,
            "candidate_weight_compression": fp16_bytes / deployed_bytes,
            "tie_rule": "minimum relative activation-weighted W4 error per saved byte, then layer name",
            "source_inventory": str(TEMPLATE),
            "source_inventory_sha256": sha256_file(TEMPLATE),
            "layer_scores": rows,
        },
        "layers": layers,
    }
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "out": str(output),
                "quantized_layers": len(selected),
                "retained_fp16_layers": len(candidate_names) - len(selected),
                "candidate_weight_compression": fp16_bytes / deployed_bytes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
