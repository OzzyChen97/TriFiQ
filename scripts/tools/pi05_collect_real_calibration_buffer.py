#!/usr/bin/env python3
"""Collect a real, on-policy RoboCasa buffer for paper-faithful π0.5 calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import robocasa  # noqa: F401 -- import before local repository modules
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_robocasa365_pi05_eval import (  # noqa: E402
    canonical_hash,
    construct_env,
    image_observation,
    language_observation,
    state_observation,
)
from openpi_client.paired_noise import PROTOCOL as NOISE_PROTOCOL  # noqa: E402
from openpi_client.paired_noise import paired_action_noise  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


TASKS = (
    "OpenCabinet",
    "OpenStandMixerHead",
    "PickPlaceDrawerToCounter",
    "CoffeeSetupMug",
)
DEFAULT_OUT = (
    REPO_ROOT
    / "runs/pi05_quantvla_paper/calibration/robocasa_real_observations_128.npz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--egl-device", type=int, required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--samples-per-task", type=int, default=32)
    parser.add_argument("--trial-seeds", default="0,1,2,3,4")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(token.strip()) for token in value.split(",") if token.strip()]
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(seed < 0 for seed in seeds):
        raise ValueError("paper calibration requires exactly five unique non-negative trial seeds")
    return seeds


def validate_fp16_server(metadata: dict) -> str:
    runtime = metadata.get("openpi_runtime") or {}
    duquant = runtime.get("duquant") or {}
    dtype = runtime.get("model_dtype") or {}
    if runtime.get("config_id") != "fp16":
        raise ValueError(f"calibration source server is not fp16: {runtime.get('config_id')!r}")
    if bool(duquant.get("enabled")) or int(duquant.get("wrapped_layers", 0)) != 0:
        raise ValueError("calibration source server unexpectedly has quantized layers")
    if dtype.get("resolved") != "float16":
        raise ValueError(f"calibration source dtype is not float16: {dtype!r}")
    return canonical_hash(metadata)


def observation_row(obs: dict, task: str, seed: int, env_step: int, replan: int, resize: int) -> dict:
    return {
        "image": image_observation(obs, "video.robot0_agentview_left", resize),
        "wrist_image": image_observation(obs, "video.robot0_eye_in_hand", resize),
        "right_image": image_observation(obs, "video.robot0_agentview_right", resize),
        "state": state_observation(obs),
        "prompt": language_observation(obs),
        "task": task,
        "seed": seed,
        "env_step": env_step,
        "replan": replan,
    }


def collect_trial(
    client: WebsocketClientPolicy,
    *,
    task: str,
    seed: int,
    target: int,
    split: str,
    resize: int,
    replan_steps: int,
    egl_device: int,
) -> list[dict]:
    env = None
    rows: list[dict] = []
    try:
        env, _ = construct_env(task, split, egl_device)
        obs, _ = env.reset(seed=seed)
        horizon = get_task_horizon(task)
        steps = 0
        replans = 0
        success = False
        while len(rows) < target and steps < horizon and not success:
            if not rows or rows[-1]["env_step"] != steps:
                rows.append(observation_row(obs, task, seed, steps, replans, resize))
            if len(rows) >= target:
                break
            request = {
                "observation/image": rows[-1]["image"],
                "observation/wrist_image": rows[-1]["wrist_image"],
                "observation/right_image": rows[-1]["right_image"],
                "observation/state": rows[-1]["state"],
                "prompt": rows[-1]["prompt"],
            }
            noise = paired_action_noise(task, seed, replans)
            response = client.infer(request, noise=noise)
            actions = np.asarray(response["actions"])
            if actions.shape != (50, 12) or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid FP16 calibration actions: {actions.shape}")
            replans += 1
            for index in range(min(replan_steps, horizon - steps)):
                obs, _reward, _terminated, _truncated, info = env.step(convert_action(actions[index]))
                steps += 1
                success = bool(info.get("success", False))
                if len(rows) < target:
                    rows.append(observation_row(obs, task, seed, steps, replans, resize))
                if len(rows) >= target or success:
                    break
        if len(rows) != target:
            raise RuntimeError(
                f"trial ended before calibration quota: {task}/{seed}={len(rows)}/{target}, "
                f"steps={steps}, success={success}"
            )
        return rows
    finally:
        if env is not None:
            env.close()


def interleave(trials: dict[tuple[str, int], list[dict]], seeds: list[int]) -> list[dict]:
    result = []
    maximum = max(len(rows) for rows in trials.values())
    for sample_index in range(maximum):
        for seed in seeds:
            for task in TASKS:
                rows = trials[(task, seed)]
                if sample_index < len(rows):
                    result.append(rows[sample_index])
    return result


def main() -> None:
    args = parse_args()
    if args.split != "pretrain" or args.replan_steps != 5 or args.resize != 224:
        raise ValueError("paper calibration requires split=pretrain, replan=5 and resize=224")
    if args.samples_per_task != 32:
        raise ValueError("paper calibration requires 32 observations per task")
    if any(task not in TASK_SET_REGISTRY["atomic_seen"] for task in TASKS):
        raise RuntimeError("paper calibration task inventory changed")
    seeds = parse_seeds(args.trial_seeds)
    output = Path(args.out).resolve()
    sidecar = Path(str(output) + ".json")
    if (output.exists() or sidecar.exists()) and not args.force:
        raise FileExistsError(f"refusing to overwrite calibration artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    client = WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    server_hash = validate_fp16_server(server_metadata)
    base, remainder = divmod(args.samples_per_task, len(seeds))
    quotas = {seed: base + (index < remainder) for index, seed in enumerate(seeds)}
    trials = {}
    for task in TASKS:
        for seed in seeds:
            target = int(quotas[seed])
            print(f"[real calibration] {task}/{seed}: target={target}", flush=True)
            trials[(task, seed)] = collect_trial(
                client,
                task=task,
                seed=seed,
                target=target,
                split=args.split,
                resize=args.resize,
                replan_steps=args.replan_steps,
                egl_device=args.egl_device,
            )
    rows = interleave(trials, seeds)
    if len(rows) != 128:
        raise RuntimeError(f"expected 128 calibration observations, got {len(rows)}")

    temporary = Path(str(output) + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            images=np.stack([row["image"] for row in rows]),
            wrist_images=np.stack([row["wrist_image"] for row in rows]),
            right_images=np.stack([row["right_image"] for row in rows]),
            states=np.stack([row["state"] for row in rows]).astype(np.float32),
            prompts=np.asarray([row["prompt"] for row in rows]),
            task_ids=np.asarray([row["task"] for row in rows]),
            env_seeds=np.asarray([row["seed"] for row in rows], dtype=np.int64),
            env_steps=np.asarray([row["env_step"] for row in rows], dtype=np.int64),
            replan_indices=np.asarray([row["replan"] for row in rows], dtype=np.int64),
        )
    temporary.replace(output)
    first32 = rows[:32]
    payload = {
        "schema_version": 1,
        "kind": "real-on-policy-robocasa-pi05",
        "npz": str(output),
        "sha256": sha256_file(output),
        "rows": len(rows),
        "a8_prefix_rows": 32,
        "tasks": list(TASKS),
        "trial_seeds": seeds,
        "trials_per_task": len(seeds),
        "samples_per_task": args.samples_per_task,
        "first32_task_counts": {
            task: sum(row["task"] == task for row in first32) for task in TASKS
        },
        "first32_seed_counts": {
            str(seed): sum(row["seed"] == seed for row in first32) for seed in seeds
        },
        "prompts": sorted({row["prompt"] for row in rows}),
        "prompt_sha256": hashlib.sha256(
            "\n".join(row["prompt"] for row in rows).encode("utf-8")
        ).hexdigest(),
        "state_dim": 16,
        "image_shape": [224, 224, 3],
        "split": args.split,
        "render_enabled": True,
        "replan_steps": args.replan_steps,
        "source_policy": "fp16",
        "source_server_metadata_sha256": server_hash,
        "source_server_metadata": server_metadata,
        "noise_protocol": NOISE_PROTOCOL,
        "max_trials_per_task": 5,
        "calibration_steps": 128,
    }
    sidecar_tmp = Path(str(sidecar) + f".tmp.{os.getpid()}")
    sidecar_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar_tmp.replace(sidecar)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
