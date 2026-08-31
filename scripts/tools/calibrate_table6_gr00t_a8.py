#!/usr/bin/env python3
"""Calibrate plan-specific GR00T static A8 scales on LIBERO-only data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tools"))

from gr00t_v2_common import (  # noqa: E402
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    ensure_a8_calibrated,
    ensure_flash_attn_rpath,
    load_policy,
    set_quant_env,
    strip_quant_env,
)
from quantvla_model_adapters import _resize_uint8_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_config(suite: str) -> str:
    return (
        "examples.Libero.custom_data_config:LiberoDataConfigMeanStd"
        if suite == "goal"
        else "examples.Libero.custom_data_config:LiberoDataConfig"
    )


def load_buffer(path: Path) -> tuple[list[dict], list[torch.Tensor]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "images",
            "wrist_images",
            "states",
            "prompts",
            "action_noises",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"LIBERO calibration buffer lacks {missing}")
        if len(archive["states"]) != 256:
            raise ValueError("Table 6 A8 calibration requires exactly 256 observations")
        observations = []
        noises = []
        for index in range(256):
            state = np.asarray(archive["states"][index], dtype=np.float32)
            if state.shape != (8,):
                raise ValueError(f"row {index}: state shape {state.shape} != (8,)")
            image = _resize_uint8_image(archive["images"][index], 256)
            wrist_image = _resize_uint8_image(archive["wrist_images"][index], 256)
            observations.append(
                {
                    "video.image": image[None, ...],
                    "video.wrist_image": wrist_image[None, ...],
                    "state.x": np.asarray([[state[0]]], dtype=np.float32),
                    "state.y": np.asarray([[state[1]]], dtype=np.float32),
                    "state.z": np.asarray([[state[2]]], dtype=np.float32),
                    "state.roll": np.asarray([[state[3]]], dtype=np.float32),
                    "state.pitch": np.asarray([[state[4]]], dtype=np.float32),
                    "state.yaw": np.asarray([[state[5]]], dtype=np.float32),
                    "state.gripper": state[6:8][None, ...],
                    "annotation.human.action.task_description": [
                        str(archive["prompts"][index])
                    ],
                }
            )
            noises.append(
                torch.from_numpy(
                    np.asarray(archive["action_noises"][index, :16], dtype=np.float32)
                )
            )
    return observations, noises


def main() -> None:
    args = parse_args()
    if args.flow_steps != 8 or args.batch_size != 8:
        raise ValueError("Table 6 GR00T A8 is frozen to flow_steps=8 and batch_size=8")
    plan_path = Path(args.plan).expanduser().resolve()
    buffer_path = Path(args.buffer).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve()
    sidecar = json.loads(
        Path(str(buffer_path) + ".json").read_text(encoding="utf-8")
    )
    if (
        sidecar.get("overlap_with_held_out") is not False
        or sidecar.get("test_rollout_feedback_used") is not False
    ):
        raise ValueError("GR00T A8 buffer is not result blind")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected = sum(
        not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) > 0
        for row in (plan.get("layers") or {}).values()
    )
    if expected <= 0:
        raise ValueError("quant plan selects no layers")
    observations, noises = load_buffer(buffer_path)

    strip_quant_env()
    set_quant_env(
        DEFAULT_INCLUDE,
        DEFAULT_EXCLUDE,
        str(Path(args.pack_dir).expanduser().resolve()),
        bits_default=4,
        group=64,
        ls=0.15,
        act_pct=99.9,
        calib_steps=32,
        row_rot="restore",
        act_dynamic=False,
    )
    os.environ.update(
        {
            "GR00T_DUQUANT_PLAN": str(plan_path),
            "GR00T_DENOISING_STEPS": str(args.flow_steps),
            "GR00T_ATM_ENABLE": "0",
            "GR00T_OHB_ENABLE": "0",
        }
    )
    ensure_flash_attn_rpath()
    policy = load_policy(
        str(Path(args.model_path).expanduser().resolve()),
        data_config=data_config(args.suite),
        denoising_steps=args.flow_steps,
        device=args.device,
    )
    metadata = {
        "schema_version": 1,
        "kind": "table6_gr00t_libero_plan_specific_a8",
        "suite": args.suite,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "source_buffer_path": str(buffer_path),
        "source_buffer_sha256": sha256_file(buffer_path),
        "calibration_buffer_sha256": sha256_file(buffer_path),
        "calibration_initial_state_indices": sidecar["initial_state_indices"],
        "held_out_initial_state_indices": sidecar[
            "held_out_initial_state_indices"
        ],
        "test_rollout_feedback_used": False,
        "wrapped_layers": expected,
        "act_percentile": 99.9,
        "calib_batches": 32,
        "denoising_steps": args.flow_steps,
        "image_resize": {
            "source_hw": [224, 224],
            "target_hw": [256, 256],
            "mode": "bilinear_antialias",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_a8_calibrated(
        policy,
        observations,
        noises,
        args.batch_size,
        expected_wrapped=expected,
        act_scale_path=str(output_path),
        act_scale_meta=metadata,
    )
    if not output_path.is_file():
        raise RuntimeError(f"A8 artifact was not written: {output_path}")
    print(
        json.dumps(
            {
                "out": str(output_path),
                "sha256": sha256_file(output_path),
                "wrapped_layers": expected,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
