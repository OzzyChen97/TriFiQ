#!/usr/bin/env python3
"""Persist plan-specific static-A8 scales without calibrating ATM/OHB."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from gr00t_v2_common import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    ensure_a8_calibrated,
    ensure_flash_attn_rpath,
    load_policy,
    resolve_data_config,
    set_quant_env,
    strip_quant_env,
)
from quantvla_cross_model_protocol import (
    protocol_artifact,
    protocol_attestation,
    validate_quant_plan,
)
from quantvla_model_adapters import gr00t_rollout_inputs, load_model_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--packdir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--suite", default="robocasa365_atomic")
    p.add_argument("--data-config", default=None)
    p.add_argument("--obs-format", default="robocasa365")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--calib-steps", type=int, default=32)
    p.add_argument("--calibration-seed", type=int, default=0)
    p.add_argument(
        "--buffer", default=str(protocol_artifact("calibration_buffer", verify=False))
    )
    p.add_argument("--act-pct", type=float, default=99.9)
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--ls", type=float, default=0.15)
    p.add_argument("--row-rot", default="restore")
    p.add_argument("--include", default=DEFAULT_INCLUDE)
    p.add_argument("--exclude", default=DEFAULT_EXCLUDE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.data_config = resolve_data_config(args.suite, args.data_config)
    plan_path = Path(args.plan).resolve()
    model_path = Path(args.model_path).resolve()
    out_path = Path(args.out).resolve()
    plan = json.loads(plan_path.read_text())
    quant_selection = validate_quant_plan(
        plan, model="gr00t", source=str(plan_path)
    )
    expected_wrapped = sum(
        1 for value in (plan.get("layers") or {}).values() if not value.get("skip")
    )
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    strip_quant_env()
    set_quant_env(
        args.include, args.exclude, str(Path(args.packdir).resolve()),
        bits_default=4, group=args.group, ls=args.ls,
        act_pct=args.act_pct, calib_steps=args.calib_steps,
        row_rot=args.row_rot, act_dynamic=False,
    )
    os.environ["GR00T_DUQUANT_PLAN"] = str(plan_path)
    os.environ["GR00T_OBS_FORMAT"] = args.obs_format
    os.environ["GR00T_DENOISING_STEPS"] = str(args.denoising_steps)
    os.environ["GR00T_ATM_ENABLE"] = "0"
    os.environ["GR00T_OHB_ENABLE"] = "0"
    ensure_flash_attn_rpath()
    policy = load_policy(
        str(model_path), data_config=args.data_config,
        denoising_steps=args.denoising_steps, device=args.device,
    )
    horizon = int(policy.model.action_head.config.action_horizon)
    action_dim = int(policy.model.action_head.config.action_dim)
    n_obs = args.calib_steps * args.batch_size
    if args.obs_format != "robocasa365":
        raise ValueError("adapter-only A8 calibration is restricted to RoboCasa365")
    records, buffer_provenance = load_model_records(
        args.buffer, n_obs, model="gr00t"
    )
    obs, noises = gr00t_rollout_inputs(records)
    buffer_sha = buffer_provenance["sha256"]
    meta = {
        "buffer_sha256": buffer_sha,
        "source_buffer_sha256": buffer_sha,
        "source_buffer_path": str(Path(args.buffer).expanduser().resolve()),
        "calibration_seed": args.calibration_seed,
        "data_config": args.data_config,
        "obs_format": args.obs_format,
        "act_percentile": args.act_pct,
        "calib_batches": args.calib_steps,
        "denoising_steps": args.denoising_steps,
        "plan_sha256": plan_sha,
        "checkpoint_path": str(model_path),
        "wrapped_layers": expected_wrapped,
        "cross_model_protocol": protocol_attestation(),
        "quantization_selection": quant_selection,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_a8_calibrated(
        policy, obs, noises, args.batch_size,
        expected_wrapped=expected_wrapped,
        act_scale_path=str(out_path), act_scale_meta=meta,
    )
    if not out_path.exists() or not Path(str(out_path) + ".meta.json").exists():
        raise SystemExit(f"A8 persistence failed: {out_path}")
    print(f"[calibrate-a8] saved {expected_wrapped} plan-specific scales -> {out_path}")


if __name__ == "__main__":
    main()
