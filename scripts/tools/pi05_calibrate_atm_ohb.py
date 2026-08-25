#!/usr/bin/env python3
"""Plan-specific static ATM/OHB calibration for π0.5 RoboCasa."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = REPO_ROOT / "code" / "pi05" / "openpi"
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

from openpi.policies import policy_config  # noqa: E402
from openpi.quant import enable_duquant_if_configured, load_quant_plan, sha256_file  # noqa: E402
from openpi.quant.atm_pi05 import (  # noqa: E402
    clear_atm_capture,
    ensure_pi05_attention_patch,
    register_atm_capture,
    register_ohb_output_capture,
    register_ohb_perhead_capture,
)
from openpi.training import config  # noqa: E402
from pi05_batched_policy import (  # noqa: E402
    ATM_BATCH_SIZE,
    ATM_OBSERVATIONS,
    FLOW_STEPS,
    iter_batches,
    load_records as load_canonical_records,
    sample_batch,
)


CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"


class StepVectorMean:
    def __init__(self):
        self.sums: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
        self.counts: dict[str, dict[int, int]] = defaultdict(dict)
        self.calls: dict[str, int] = defaultdict(int)

    def __call__(self, name: str, value: torch.Tensor) -> None:
        step = self.calls[name] % FLOW_STEPS
        self.calls[name] += 1
        vector = value.detach().to(torch.float32).mean(dim=0)
        previous = self.sums[name].get(step)
        self.sums[name][step] = vector.clone() if previous is None else previous + vector
        self.counts[name][step] = self.counts[name].get(step, 0) + 1

    def finalize(self) -> dict[str, dict[int, torch.Tensor]]:
        return {
            name: {
                step: (value / self.counts[name][step]).to(device="cpu")
                for step, value in by_step.items()
            }
            for name, by_step in self.sums.items()
        }


class StepPerHeadRMSMean(StepVectorMean):
    def __call__(self, name: str, value: torch.Tensor) -> None:
        step = self.calls[name] % FLOW_STEPS
        self.calls[name] += 1
        vector = value.detach().to(torch.float32)
        previous = self.sums[name].get(step)
        self.sums[name][step] = vector.clone() if previous is None else previous + vector
        self.counts[name][step] = self.counts[name].get(step, 0) + 1


class StepScalarMean(StepVectorMean):
    def __call__(self, name: str, value: torch.Tensor) -> None:
        step = self.calls[name] % FLOW_STEPS
        self.calls[name] += 1
        scalar = value.detach().to(torch.float32).reshape(()).cpu()
        previous = self.sums[name].get(step)
        self.sums[name][step] = scalar.clone() if previous is None else previous + scalar
        self.counts[name][step] = self.counts[name].get(step, 0) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch"),
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--a8-scale", required=True)
    parser.add_argument(
        "--pack-dir",
        default=str(REPO_ROOT / "runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"),
    )
    parser.add_argument(
        "--buffer",
        default=str(
            REPO_ROOT
            / "runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
        ),
    )
    parser.add_argument("--n-frames", type=int, default=ATM_OBSERVATIONS)
    parser.add_argument("--batch-size", type=int, default=ATM_BATCH_SIZE)
    parser.add_argument("--scope", default="expert")
    parser.add_argument(
        "--ohb-mode",
        choices=("per_head_pre_projection", "per_layer_post_projection"),
        default="per_head_pre_projection",
    )
    parser.add_argument("--permute", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--atm-application",
        choices=("runtime_query", "fold_q_weight"),
        default="runtime_query",
        help="Recorded runtime application mode; calibration itself always observes uncorrected models.",
    )
    parser.add_argument(
        "--ohb-application",
        choices=("runtime_output", "fold_o_weight"),
        default="runtime_output",
        help="Recorded deployment mode; calibration always captures post-projection output.",
    )
    parser.add_argument("--alpha-min", type=float, default=0.7)
    parser.add_argument("--alpha-max", type=float, default=1.4)
    parser.add_argument(
        "--log-clamp",
        "--beta-log-clamp",
        dest="beta_log_clamp",
        type=float,
        default=0.30,
    )
    parser.add_argument("--alpha-neutral", type=float, default=0.02)
    parser.add_argument("--beta-neutral", type=float, default=0.03)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def collect(
    policy,
    rows: list[dict],
    scope: str,
    ohb_mode: str,
    batch_size: int,
    device: str,
) -> tuple[dict, dict, dict]:
    std = StepVectorMean()
    rms = StepPerHeadRMSMean() if ohb_mode == "per_head_pre_projection" else StepScalarMean()
    ensure_pi05_attention_patch(policy._model, scope=scope)
    register_atm_capture(policy._model, std, scope=scope)
    if ohb_mode == "per_head_pre_projection":
        register_ohb_perhead_capture(policy._model, rms, scope=scope)
    else:
        register_ohb_output_capture(policy._model, rms, scope=scope)
    timings = []
    try:
        for index, batch in enumerate(iter_batches(rows, batch_size)):
            started = time.perf_counter()
            actions = sample_batch(
                policy, batch, device, num_steps=FLOW_STEPS
            ).detach().to(torch.float32).cpu().numpy()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            if actions.shape != (len(batch), 50, 32) or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid calibration output at batch {index}")
            print(
                f"[ATM/OHB] batch {index + 1}/{math.ceil(len(rows) / batch_size)} "
                f"B={len(batch)} elapsed={timings[-1]:.3f}s",
                flush=True,
            )
    finally:
        clear_atm_capture(policy._model)
    expected_calls = math.ceil(len(rows) / batch_size) * FLOW_STEPS
    if len(std.calls) != 18 or len(rms.calls) != 18:
        raise RuntimeError(
            f"expert capture inventory mismatch: ATM={len(std.counts)} OHB={len(rms.counts)}"
        )
    bad_counts = {name: count for name, count in std.calls.items() if count != expected_calls}
    bad_counts.update(
        {f"ohb:{name}": count for name, count in rms.calls.items() if count != expected_calls}
    )
    if bad_counts:
        raise RuntimeError(f"expert capture call-count mismatch: {bad_counts}")
    stats = {
        "frames": len(rows),
        "batch_size": batch_size,
        "batches": math.ceil(len(rows) / batch_size),
        "latency_mean_s": float(np.mean(timings)),
        "latency_p95_s": float(np.percentile(timings, 95)),
        "capture_counts": dict(std.calls),
        "per_step_capture_counts": {
            name: dict(counts) for name, counts in std.counts.items()
        },
    }
    return std.finalize(), rms.finalize(), stats


def correction_values(
    teacher_std: dict[str, dict[int, torch.Tensor]],
    teacher_rms: dict[str, dict[int, torch.Tensor]],
    quant_std: dict[str, dict[int, torch.Tensor]],
    quant_rms: dict[str, dict[int, torch.Tensor]],
    *,
    alpha_min: float,
    alpha_max: float,
    beta_log_clamp: float,
    alpha_neutral: float,
    beta_neutral: float,
    ohb_mode: str,
) -> tuple[dict, dict]:
    if set(teacher_std) != set(quant_std) or set(teacher_rms) != set(quant_rms):
        raise ValueError("teacher/quant ATM/OHB layer sets differ")
    if set(teacher_std) != set(teacher_rms):
        raise ValueError("ATM/OHB collectors saw different layer sets")
    layers = {}
    per_step_values = {}
    for name in sorted(teacher_std):
        steps = sorted(
            set(teacher_std[name])
            & set(quant_std[name])
            & set(teacher_rms[name])
            & set(quant_rms[name])
        )
        if steps != list(range(FLOW_STEPS)):
            raise ValueError(f"incomplete denoising-step statistics for {name}: {steps}")
        alpha_steps = []
        beta_steps = []
        serialized_steps = {}
        for step in steps:
            teacher_alpha = teacher_std[name][step].to(torch.float32)
            quant_alpha = quant_std[name][step].to(torch.float32)
            alpha = torch.where(
                quant_alpha > 0,
                teacher_alpha / (quant_alpha + 1e-6),
                torch.ones_like(teacher_alpha),
            )
            alpha = alpha.clamp(alpha_min, alpha_max)
            alpha = torch.where(
                (alpha - 1.0).abs() < alpha_neutral,
                torch.ones_like(alpha),
                alpha,
            )

            teacher_beta = teacher_rms[name][step].to(torch.float32).reshape(-1)
            quant_beta = quant_rms[name][step].to(torch.float32).reshape(-1)
            # This intentionally mirrors GR00T ``compute_per_step_beta``'s
            # scalar Python-math order, including its clamps and strict
            # neutral comparison.
            beta_values = []
            for head in range(teacher_beta.shape[0]):
                teacher_value = max(float(teacher_beta[head]), 1e-8)
                quant_value = max(float(quant_beta[head]), 1e-8)
                log_beta = -math.log(max(quant_value / teacher_value, 1e-8))
                log_beta = max(-beta_log_clamp, min(beta_log_clamp, log_beta))
                beta_values.append(
                    1.0 if abs(log_beta) < beta_neutral else math.exp(log_beta)
                )
            beta = torch.tensor(beta_values, dtype=torch.float32).reshape(
                teacher_rms[name][step].shape
            )
            if not torch.isfinite(alpha).all() or not torch.isfinite(beta).all():
                raise ValueError(f"non-finite correction for {name} at step {step}")
            alpha_steps.append(alpha)
            beta_steps.append(beta)
            serialized_steps[str(step)] = {
                "all": [float(value) for value in alpha.reshape(-1)],
                (
                    "beta_perhead"
                    if ohb_mode == "per_head_pre_projection"
                    else "beta"
                ): (
                    [float(value) for value in beta.reshape(-1)]
                    if ohb_mode == "per_head_pre_projection"
                    else float(beta)
                ),
            }
        pooled_alpha = torch.stack(alpha_steps).mean(dim=0)
        pooled_beta = torch.stack(beta_steps).mean(dim=0)
        layers[name] = {"all": [float(value) for value in pooled_alpha.reshape(-1)]}
        if ohb_mode == "per_head_pre_projection":
            layers[name]["beta_perhead"] = [
                float(value) for value in pooled_beta.reshape(-1)
            ]
        else:
            layers[name]["beta"] = float(pooled_beta)
        per_step_values[name] = serialized_steps
    if len(layers) != 18:
        raise ValueError(f"formal expert ATM/OHB requires 18 layers, got {len(layers)}")
    return layers, per_step_values


def apply_plan_aware_neutral(
    layers: dict,
    per_step_values: dict,
    plan_layers: dict,
    *,
    alpha_neutral: float,
    beta_neutral: float,
) -> dict:
    """Mirror GR00T final's all-FP16 attention-block neutralization rule."""
    marks = {}
    for name, entry in layers.items():
        projections = [key for key in plan_layers if key.startswith(name + ".")]
        fp16_block = not projections or all(
            bool(plan_layers[key].get("skip", False)) for key in projections
        )
        if not fp16_block:
            marks[name] = {"fp16_block": False, "forced_neutral": False}
            continue
        alpha = entry.get("all") or []
        beta = entry.get("beta_perhead") or [entry.get("beta", 1.0)]
        neutral = all(abs(float(value) - 1.0) < alpha_neutral for value in alpha)
        neutral = neutral and all(
            abs(math.log(max(float(value), 1e-9))) < beta_neutral for value in beta
        )
        if neutral:
            entry["all"] = [1.0] * len(alpha)
            if "beta_perhead" in entry:
                entry["beta_perhead"] = [1.0] * len(beta)
            else:
                entry["beta"] = 1.0
            for step in per_step_values[name].values():
                step["all"] = [1.0] * len(step["all"])
                if "beta_perhead" in step:
                    step["beta_perhead"] = [1.0] * len(step["beta_perhead"])
                else:
                    step["beta"] = 1.0
            marks[name] = {"fp16_block": True, "forced_neutral": True}
        else:
            marks[name] = {
                "fp16_block": True,
                "forced_neutral": False,
                "note": "correction retained because upstream quantization causes drift",
            }
    return marks


def compute_cv_stats(per_step_values: dict, threshold: float = 0.05) -> dict:
    alpha_cv = []
    beta_cv = []
    for steps in per_step_values.values():
        ordered = [steps[str(step)] for step in range(FLOW_STEPS)]
        alpha = np.asarray([row["all"] for row in ordered], dtype=np.float64)
        beta_key = "beta_perhead" if "beta_perhead" in ordered[0] else "beta"
        beta = np.asarray([row[beta_key] for row in ordered], dtype=np.float64)
        if beta.ndim == 1:
            beta = beta[:, None]
        alpha_cv.extend((alpha.std(axis=0) / np.maximum(np.abs(alpha.mean(axis=0)), 1e-9)).tolist())
        beta_cv.extend((beta.std(axis=0) / np.maximum(np.abs(beta.mean(axis=0)), 1e-9)).tolist())
    return {
        "cv_threshold": threshold,
        "head_fraction_required": 0.95,
        "n_heads_alpha": len(alpha_cv),
        "n_heads_alpha_below": sum(value < threshold for value in alpha_cv),
        "n_heads_beta": len(beta_cv),
        "n_heads_beta_below": sum(value < threshold for value in beta_cv),
        "static_sufficient": bool(
            (not alpha_cv or np.mean(np.asarray(alpha_cv) < threshold) >= 0.95)
            and (not beta_cv or np.mean(np.asarray(beta_cv) < threshold) >= 0.95)
        ),
    }


def configure_quant(args: argparse.Namespace, plan_hash: str, buffer_hash: str, wrapped: int) -> None:
    os.environ.update(
        {
            "TORCHDYNAMO_DISABLE": "1",
            "OPENPI_MODEL_DTYPE": "float16",
            "OPENPI_DUQUANT_PLAN": str(Path(args.plan).resolve()),
            "OPENPI_DUQUANT_PLAN_STRICT": "1",
            "OPENPI_DUQUANT_WBITS_DEFAULT": "4",
            "OPENPI_DUQUANT_ABITS": "8",
            "OPENPI_DUQUANT_BLOCK": "64",
            "OPENPI_DUQUANT_BLOCK_OUT": "64",
            "OPENPI_DUQUANT_EXPECT_BLOCK": "64",
            "OPENPI_DUQUANT_EXPECT_WRAPPED": str(wrapped),
            "OPENPI_DUQUANT_LS": "0.15",
            "OPENPI_DUQUANT_PERMUTE": "1" if args.permute else "0",
            "OPENPI_DUQUANT_ROW_ROT": "restore",
            "OPENPI_DUQUANT_ACT_PCT": "99.9",
            "OPENPI_DUQUANT_CALIB_STEPS": "32",
            "OPENPI_DUQUANT_DENOISING_STEPS": str(FLOW_STEPS),
            "OPENPI_DUQUANT_PACKDIR": str(Path(args.pack_dir).resolve()),
            "OPENPI_DUQUANT_ACT_SCALE_PATH": str(Path(args.a8_scale).resolve()),
            "OPENPI_DUQUANT_REQUIRE_ACT_SCALE": "1",
            "OPENPI_DUQUANT_CALIB_BUFFER_SHA256": buffer_hash,
            "OPENPI_CHECKPOINT_SHA256": CHECKPOINT_SHA256,
            "OPENPI_DUQUANT_STRICT_ARTIFACTS": "1",
            "OPENPI_DUQUANT_PRECACHE_WEIGHTS": "1",
            "OPENPI_DUQUANT_TRITON": "0",
            "OPENPI_DUQUANT_QUIET": "1",
            "OPENPI_ATM_EXPECT_PLAN_SHA256": plan_hash,
        }
    )


def main() -> None:
    args = parse_args()
    plan = load_quant_plan(args.plan)
    wrapped = len(plan.quantized_bits)
    if wrapped <= 0:
        raise ValueError("ATM/OHB quant plan selects no layers")
    buffer_path = Path(args.buffer).resolve()
    buffer_hash = sha256_file(buffer_path)
    if args.n_frames != ATM_OBSERVATIONS or args.batch_size != ATM_BATCH_SIZE:
        raise ValueError(
            f"GR00T-final ATM/OHB requires n_frames={ATM_OBSERVATIONS}, "
            f"batch_size={ATM_BATCH_SIZE}"
        )
    rows = load_canonical_records(buffer_path, args.n_frames)
    configure_quant(args, plan.sha256, buffer_hash, wrapped)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    started = time.time()

    print("[ATM/OHB] collecting FP16 teacher", flush=True)
    teacher = policy_config.create_trained_policy(
        config.get_config("pi05_pretrain_human300"), checkpoint_dir, pytorch_device=args.device
    )
    teacher_std, teacher_rms, teacher_stats = collect(
        teacher, rows, args.scope, args.ohb_mode, args.batch_size, args.device
    )
    del teacher
    torch.cuda.empty_cache()

    print(f"[ATM/OHB] collecting quant plan ({wrapped} wrappers)", flush=True)
    quant = policy_config.create_trained_policy(
        config.get_config("pi05_pretrain_human300"), checkpoint_dir, pytorch_device=args.device
    )
    runtime = enable_duquant_if_configured(quant._model)
    quant._model.to(args.device)
    if runtime["wrapped_layers"] != wrapped or not runtime.get("act_scales_ready"):
        raise RuntimeError(f"invalid quant runtime: {runtime}")
    quant_std, quant_rms, quant_stats = collect(
        quant, rows, args.scope, args.ohb_mode, args.batch_size, args.device
    )
    layers, per_step_values = correction_values(
        teacher_std,
        teacher_rms,
        quant_std,
        quant_rms,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        beta_log_clamp=args.beta_log_clamp,
        alpha_neutral=args.alpha_neutral,
        beta_neutral=args.beta_neutral,
        ohb_mode=args.ohb_mode,
    )
    plan_marks = apply_plan_aware_neutral(
        layers,
        per_step_values,
        plan.layers,
        alpha_neutral=args.alpha_neutral,
        beta_neutral=args.beta_neutral,
    )
    cv_stats = compute_cv_stats(per_step_values)
    payload = {
        "schema_version": 1,
        "meta": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "plan": str(Path(args.plan).resolve()),
            "plan_sha256": plan.sha256,
            "wrapped_layers": wrapped,
            "a8_scale": str(Path(args.a8_scale).resolve()),
            "a8_scale_sha256": sha256_file(args.a8_scale),
            "pack_manifest": str((Path(args.pack_dir) / "manifest.json").resolve()),
            "pack_manifest_sha256": sha256_file(Path(args.pack_dir) / "manifest.json"),
            "calibration_buffer": str(buffer_path),
            "calibration_buffer_sha256": buffer_hash,
            "frames": args.n_frames,
            "batch_size": args.batch_size,
            "flow_steps": FLOW_STEPS,
            "noise_protocol": "torch-cpu-sequential-seed0-f32-v1",
            "evaluation_noise_protocol": (
                "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"
            ),
            "calibration_seed": 0,
            "scope": args.scope,
            "ohb_mode": args.ohb_mode,
            "ohb_capture_point": (
                "pre_head_concat" if args.ohb_mode == "per_head_pre_projection"
                else "post_o_proj_pre_residual"
            ),
            "atm_application": args.atm_application,
            "ohb_application": args.ohb_application,
            "enable_permute": args.permute,
            "attention_layers": len(layers),
            "formula_authority": "scripts/tools/calibrate_atm_perstep_gr00t.py",
            "pooling": "mean of four per-denoising-step correction ratios",
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "beta_log_clamp": args.beta_log_clamp,
            "alpha_neutral": args.alpha_neutral,
            "beta_neutral": args.beta_neutral,
            "plan_marks": plan_marks,
            "cv_stats": cv_stats,
            "teacher_stats": teacher_stats,
            "quant_stats": quant_stats,
            "elapsed_s": time.time() - started,
        },
        "layers": layers,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(out) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(out)
    print(
        json.dumps(
            {
                "out": str(out), "sha256": sha256_file(out), "attention_layers": len(layers),
                "plan_sha256": plan.sha256, "elapsed_s": payload["meta"]["elapsed_s"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
