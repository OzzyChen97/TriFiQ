"""pi0.5 ATM/OHB calibration: collect teacher & quant stats, output alpha/beta JSON.

Usage:
  OPENPI_DUQUANT_PACKDIR=<packdir> OPENPI_DUQUANT_CALIB_STEPS=160 \
      python tools/pi05_calibrate_atm_ohb.py --n-frames 20 --out <json>
"""

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class StdCollector:
    """Aggregates per-layer per-head std for ATM alpha."""

    def __init__(self) -> None:
        self.sum: Dict[str, torch.Tensor] = {}
        self.count: Dict[str, int] = defaultdict(int)

    def __call__(self, layer_name: str, std_tensor: torch.Tensor) -> None:
        std_tensor = std_tensor.mean(dim=0)  # average over batch
        std_cpu = std_tensor.detach().to(torch.float32)
        if layer_name not in self.sum:
            self.sum[layer_name] = std_cpu.clone()
        else:
            self.sum[layer_name] += std_cpu
        self.count[layer_name] += 1

    def finalize(self) -> Dict[str, torch.Tensor]:
        return {name: t / max(self.count[name], 1) for name, t in self.sum.items()}


class RMSPerHeadCollector:
    """Aggregates per-head RMS for OHB beta."""

    def __init__(self) -> None:
        self.sum: Dict[str, torch.Tensor] = {}
        self.count: Dict[str, int] = defaultdict(int)

    def __call__(self, layer_name: str, rms_tensor: torch.Tensor) -> None:
        rms_cpu = rms_tensor.detach().to(torch.float32).cpu()
        if layer_name not in self.sum:
            self.sum[layer_name] = rms_cpu.clone()
        else:
            self.sum[layer_name] += rms_cpu
        self.count[layer_name] += 1

    def finalize(self) -> Dict[str, torch.Tensor]:
        return {name: t / max(self.count[name], 1) for name, t in self.sum.items()}


def compute_alpha_json(
    teacher_stats: Dict[str, torch.Tensor],
    quant_stats: Dict[str, torch.Tensor],
    min_alpha: float = 0.7,
    max_alpha: float = 1.4,
    neutral_threshold: float = 0.02,
) -> Dict[str, Dict[str, List[float]]]:
    alpha_data: Dict[str, Dict[str, List[float]]] = {}
    for name in sorted(teacher_stats.keys()):
        if name not in quant_stats:
            continue
        teacher_std = teacher_stats[name].to(torch.float32)
        quant_std = quant_stats[name].to(torch.float32)
        alpha = torch.where(
            quant_std > 0,
            teacher_std / (quant_std + 1e-6),
            torch.ones_like(teacher_std),
        )
        alpha = alpha.clamp(min_alpha, max_alpha)
        alpha = torch.where((alpha - 1.0).abs() < neutral_threshold, torch.ones_like(alpha), alpha)
        alpha_data[name] = {"all": alpha.tolist()}
    return alpha_data


def compute_beta_perhead_values(
    teacher_rms: Dict[str, torch.Tensor],
    quant_rms: Dict[str, torch.Tensor],
    *,
    log_clamp: float,
    neutral: float,
) -> Dict[str, List[float]]:
    beta_map: Dict[str, List[float]] = {}
    for name, teacher_tensor in teacher_rms.items():
        if name not in quant_rms:
            continue
        quant_tensor = quant_rms[name]
        beta_list = []
        for h in range(teacher_tensor.shape[0]):
            t = max(float(teacher_tensor[h]), 1e-8)
            q = max(float(quant_tensor[h]), 1e-8)
            rho = q / t
            log_beta = -math.log(max(rho, 1e-8))
            log_beta = max(-log_clamp, min(log_clamp, log_beta))
            if abs(log_beta) < neutral:
                beta = 1.0
            else:
                beta = math.exp(log_beta)
            beta_list.append(beta)
        beta_map[name] = beta_list
    return beta_map


def make_obs(n_frames: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    obs_list = []
    for i in range(n_frames):
        if os.environ.get("OPENPI_OBS_FORMAT") == "robocasa":
            # RoboCasa365: 12-dim state (padded to 32 by RobocasaInputs) + 3 cameras.
            obs_list.append({
                "observation/state": rng.random(12).astype(np.float32) * 0.2 - 0.1,
                "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
                "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
                "observation/right_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
                "prompt": "turn on the electric kettle",
            })
        else:
            obs_list.append({
                "observation/state": rng.random(8).astype(np.float32) * 0.2 - 0.1,
                "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
                "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
                "prompt": "pick up the black bowl and place it on the plate",
            })
    return obs_list


def collect_stats(policy, n_frames: int, scope: str) -> tuple[Dict, Dict]:
    """Run inferences collecting per-head logits std and output RMS."""
    model = policy._model
    std_collector = StdCollector()
    rms_collector = RMSPerHeadCollector()

    from openpi.quant.atm_pi05 import (
        ensure_pi05_attention_patch,
        register_atm_capture,
        register_ohb_perhead_capture,
    )
    # Install the patch (no alpha/beta -> behavior identical to original forward)
    ensure_pi05_attention_patch(model, scope=scope)
    register_atm_capture(model, std_collector, scope=scope)
    register_ohb_perhead_capture(model, rms_collector, scope=scope)

    obs_list = make_obs(n_frames)
    for i, obs in enumerate(obs_list):
        policy.infer(obs)
        if (i + 1) % 5 == 0:
            print(f"  collected {i+1}/{n_frames}", flush=True)

    return std_collector.finalize(), rms_collector.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/data1/wubohan/openpi/checkpoints/pi05_libero_pytorch")
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--out", default="/data1/wubohan/openpi/packs/atm_alpha_beta_pi05.json")
    parser.add_argument("--scope", default="expert")
    parser.add_argument("--alpha-log-clamp", type=float, default=0.30)
    parser.add_argument("--alpha-neutral", type=float, default=0.02)
    parser.add_argument("--ohb-log-clamp", type=float, default=0.30)
    parser.add_argument("--ohb-neutral", type=float, default=0.03)
    args = parser.parse_args()

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.quant import enable_duquant_if_configured

    print("=== Teacher (FP16) collection ===", flush=True)
    teacher = _policy_config.create_trained_policy(
        _config.get_config(args.config), args.checkpoint, pytorch_device="cuda"
    )
    teacher_std, teacher_rms = collect_stats(teacher, args.n_frames, args.scope)
    print(f"teacher: {len(teacher_std)} layers", flush=True)
    del teacher
    torch.cuda.empty_cache()

    print("=== Quant (W4A8) collection ===", flush=True)
    quant = _policy_config.create_trained_policy(
        _config.get_config(args.config), args.checkpoint, pytorch_device="cuda"
    )
    enable_duquant_if_configured(quant._model)
    quant._model.to("cuda")
    quant_std, quant_rms = collect_stats(quant, args.n_frames, args.scope)
    print(f"quant: {len(quant_std)} layers", flush=True)

    alpha_data = compute_alpha_json(teacher_std, quant_std)
    beta_perhead = compute_beta_perhead_values(
        teacher_rms, quant_rms, log_clamp=args.ohb_log_clamp, neutral=args.ohb_neutral
    )
    for name in alpha_data:
        alpha_data[name]["beta_perhead"] = beta_perhead.get(name, [1.0] * len(alpha_data[name]["all"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(alpha_data, f, indent=2)
    print(f"✅ ATM/OHB JSON saved to {args.out} ({len(alpha_data)} layers)", flush=True)


if __name__ == "__main__":
    main()
