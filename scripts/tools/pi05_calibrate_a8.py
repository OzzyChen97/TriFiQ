#!/usr/bin/env python3
"""Calibrate and persist plan-specific static A8 scales for π0.5."""

from __future__ import annotations

import argparse
import json
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
from openpi.quant import enable_duquant_if_configured, save_act_scales, sha256_file, static_scales_ready  # noqa: E402
from openpi.training import config  # noqa: E402
from pi05_batched_policy import (  # noqa: E402
    CALIBRATION_BATCHES,
    CALIBRATION_BATCH_SIZE,
    FLOW_STEPS,
    iter_batches,
    load_records,
    sample_batch,
)


CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch"),
    )
    parser.add_argument(
        "--plan",
        default=str(
            REPO_ROOT
            / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_quantvla_uniform_w4a8_d4.plan.json"
        ),
    )
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
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-wrapped", type=int, default=None)
    parser.add_argument(
        "--n-frames", type=int, default=CALIBRATION_BATCHES * CALIBRATION_BATCH_SIZE
    )
    parser.add_argument("--batch-size", type=int, default=CALIBRATION_BATCH_SIZE)
    parser.add_argument("--flow-steps", type=int, default=FLOW_STEPS)
    parser.add_argument("--permute", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace, buffer_hash: str) -> None:
    values = {
        "TORCHDYNAMO_DISABLE": "1",
        "OPENPI_MODEL_DTYPE": "float16",
        "OPENPI_DUQUANT_PLAN": str(Path(args.plan).resolve()),
        "OPENPI_DUQUANT_PLAN_STRICT": "1",
        "OPENPI_DUQUANT_WBITS_DEFAULT": "4",
        "OPENPI_DUQUANT_ABITS": "8",
        "OPENPI_DUQUANT_BLOCK": "64",
        "OPENPI_DUQUANT_BLOCK_OUT": "64",
        "OPENPI_DUQUANT_EXPECT_BLOCK": "64",
        "OPENPI_DUQUANT_LS": "0.15",
        "OPENPI_DUQUANT_PERMUTE": "1" if args.permute else "0",
        "OPENPI_DUQUANT_ROW_ROT": "restore",
        "OPENPI_DUQUANT_ACT_PCT": "99.9",
        "OPENPI_DUQUANT_CALIB_STEPS": "32",
        "OPENPI_DUQUANT_DENOISING_STEPS": str(args.flow_steps),
        "OPENPI_DUQUANT_PACKDIR": str(Path(args.pack_dir).resolve()),
        "OPENPI_DUQUANT_ACT_SCALE_PATH": str(Path(args.out).resolve()),
        "OPENPI_DUQUANT_CALIB_BUFFER_SHA256": buffer_hash,
        "OPENPI_CHECKPOINT_SHA256": CHECKPOINT_SHA256,
        "OPENPI_DUQUANT_STRICT_ARTIFACTS": "1",
        "OPENPI_DUQUANT_PRECACHE_WEIGHTS": "1",
        "OPENPI_DUQUANT_TRITON": "0",
        "OPENPI_DUQUANT_QUIET": "1",
    }
    if args.expected_wrapped is not None:
        values["OPENPI_DUQUANT_EXPECT_WRAPPED"] = str(args.expected_wrapped)
    os.environ.update(values)


def main() -> None:
    args = parse_args()
    if args.flow_steps <= 0:
        raise ValueError("--flow-steps must be positive")
    buffer_path = Path(args.buffer).resolve()
    output_path = Path(args.out).resolve()
    buffer_hash = sha256_file(buffer_path)
    if args.force:
        # Removal is exact and recoverable by recalibration; never touch the
        # surrounding directory or unrelated artifacts.
        output_path.unlink(missing_ok=True)
        Path(str(output_path) + ".json").unlink(missing_ok=True)
    configure_environment(args, buffer_hash)
    if args.n_frames != CALIBRATION_BATCHES * args.batch_size:
        raise ValueError(
            f"A8 calibration requires {CALIBRATION_BATCHES} complete batches; "
            f"got n_frames={args.n_frames}, batch_size={args.batch_size}"
        )
    observations = load_records(buffer_path, args.n_frames)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    started = time.time()
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_pretrain_human300"), checkpoint_dir, pytorch_device=args.device
    )
    runtime = enable_duquant_if_configured(policy._model)
    policy._model.to(args.device)
    if output_path.is_file():
        if not static_scales_ready(policy._model):
            raise RuntimeError("existing A8 artifact loaded but scales are not ready")
        print(json.dumps({"status": "validated_existing", "runtime": runtime}, indent=2, sort_keys=True))
        return

    timings = []
    for index, batch in enumerate(iter_batches(observations, args.batch_size)):
        request_started = time.time()
        actions = sample_batch(
            policy, batch, args.device, num_steps=args.flow_steps
        ).detach().to(torch.float32).cpu().numpy()
        if actions.shape != (args.batch_size, 50, 32) or not np.isfinite(actions).all():
            raise RuntimeError(f"calibration batch {index} produced invalid actions {actions.shape}")
        timings.append(time.time() - request_started)
        print(
            f"[pi05 A8] batch {index + 1}/{CALIBRATION_BATCHES} "
            f"(B={args.batch_size}) elapsed={timings[-1]:.3f}s "
            f"ready={static_scales_ready(policy._model)}",
            flush=True,
        )
    if not static_scales_ready(policy._model):
        raise RuntimeError("A8 scales are not ready after 32 complete calibration requests")
    payload = save_act_scales(
        policy._model,
        output_path,
        {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "calibration_buffer_sha256": buffer_hash,
            "calibration_buffer_path": str(buffer_path),
            "plan_path": str(Path(args.plan).resolve()),
            "pack_dir": str(Path(args.pack_dir).resolve()),
            "calibration_reference": "plan_true_mixed_deployment",
            "enable_permute": args.permute,
            "calibration_seed": 0,
            "calibration_batch_size": args.batch_size,
            "calibration_observations": args.n_frames,
            "protocol": f"full-context-target16-flow{args.flow_steps}",
        },
    )
    summary = {
        "status": "calibrated",
        "output": str(output_path),
        "npz_sha256": payload["npz_sha256"],
        "runtime": runtime,
        "request_latency_s": {
            "mean": float(np.mean(timings)),
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
        },
        "elapsed_s": time.time() - started,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
