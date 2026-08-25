"""pi0.5 DuQuant calibration: pack weights + activation calibration.

Usage:
  OPENPI_DUQUANT_DRYRUN=1 python tools/pi05_calibrate_duquant.py   # dry-run layer scan
  OPENPI_DUQUANT_PACKDIR=<dir> OPENPI_DUQUANT_CALIB_STEPS=160 \
      python tools/pi05_calibrate_duquant.py --n-frames 200         # pack + calibrate
"""

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/data1/wubohan/openpi/checkpoints/pi05_libero_pytorch")
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.quant import enable_duquant_if_configured

    print(f"Loading policy ({args.config}) from {args.checkpoint} ...", flush=True)
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config), args.checkpoint, pytorch_device=args.device
    )
    model = policy._model
    print("Applying DuQuant...", flush=True)
    enable_duquant_if_configured(model)
    model.to(args.device)

    if os.environ.get("OPENPI_DUQUANT_DRYRUN", "0") not in ("0", "false", "False"):
        print("Dry-run complete. Exiting.", flush=True)
        return

    # Run calibration forwards
    obs_list = make_obs(args.n_frames)
    print(f"Running {args.n_frames} calibration forwards (10 denoise steps each)...", flush=True)
    for i, obs in enumerate(obs_list):
        policy.infer(obs)
        if (i + 1) % 5 == 0 or i == args.n_frames - 1:
            print(f"  calibrated {i+1}/{args.n_frames}", flush=True)
    print("✅ DuQuant calibration done.", flush=True)


if __name__ == "__main__":
    main()
