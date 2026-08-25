#!/usr/bin/env python3
"""Create the immutable data-free π0.5 RoboCasa calibration/probe buffer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
)
PROMPT = "add ice cubes to the blender"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("size must be positive")
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Mirror GR00T's seed-0 data-free RoboCasa365 buffer, adapted only to the
    # π0.5 checkpoint's native 224px input and concatenated 16-D state format.
    images = rng.integers(0, 256, size=(args.size, 224, 224, 3), dtype=np.uint8)
    wrists = rng.integers(0, 256, size=(args.size, 224, 224, 3), dtype=np.uint8)
    right_images = rng.integers(0, 256, size=(args.size, 224, 224, 3), dtype=np.uint8)
    states = []
    for _ in range(args.size):
        eef_position = rng.uniform(
            [-0.0224, -0.4913, 0.2313], [0.5440, 0.4281, 0.7073]
        ).astype(np.float32)
        eef_quaternion = rng.standard_normal(4).astype(np.float32)
        eef_quaternion /= max(float(np.linalg.norm(eef_quaternion)), 1e-8)
        if eef_quaternion[-1] < 0:
            eef_quaternion *= -1
        base_position = rng.uniform(
            [0.6646, -3.6397, 0.7000], [5.1259, -0.7034, 0.7044]
        ).astype(np.float32)
        yaw = rng.uniform(-2.45, 2.78)
        base_quaternion = np.asarray(
            [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=np.float32
        )
        gripper = rng.uniform([0.0031, -0.0406], [0.0405, -0.0053]).astype(np.float32)
        states.append(
            np.concatenate(
                [eef_position, eef_quaternion, base_position, base_quaternion, gripper]
            )
        )
    states = np.stack(states).astype(np.float32, copy=False)
    prompts = np.asarray([PROMPT] * args.size)
    task_ids = np.asarray(["AddIceCubes"] * args.size)
    env_seeds = np.arange(args.size, dtype=np.int64)
    torch_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    action_noises = torch.stack(
        [
            torch.randn((50, 32), generator=torch_generator, dtype=torch.float32)
            for _ in range(args.size)
        ]
    ).numpy()

    temporary = Path(str(out) + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            images=images,
            wrist_images=wrists,
            right_images=right_images,
            states=states,
            prompts=prompts,
            task_ids=task_ids,
            env_seeds=env_seeds,
            action_noises=action_noises,
        )
    temporary.replace(out)
    payload = {
        "schema_version": 1,
        "kind": "gr00t-aligned-data-free-l1-robocasa-pi05",
        "size": args.size,
        "seed": args.seed,
        "npz": str(out),
        "sha256": sha256_file(out),
        "image_shape": [224, 224, 3],
        "state_dim": 16,
        "state_order": [
            "end_effector_position_relative[3]",
            "end_effector_rotation_relative[4]",
            "base_position[3]",
            "base_rotation[4]",
            "gripper_qpos[2]",
        ],
        "prompt": PROMPT,
        "checkpoint_native_normalization": "z-score",
        "calibration_batches": 32,
        "calibration_batch_size": 8,
        "denoising_steps": 4,
        "atm_observations": 16,
        "atm_batch_size": 8,
        "noise_generation": "torch.Generator(cpu).manual_seed(0), sequential torch.randn float32",
    }
    sidecar = Path(str(out) + ".json")
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
