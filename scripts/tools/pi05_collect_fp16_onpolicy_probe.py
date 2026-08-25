#!/usr/bin/env python3
"""Collect FP16-teacher on-policy RoboCasa states for quantization scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import robocasa  # noqa: F401 -- must import before RoboCasa environment helpers
import torch
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPI_CLIENT = REPO_ROOT / "code/pi05/openpi/packages/openpi-client/src"
sys.path.insert(0, str(OPENPI_CLIENT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from openpi_client.paired_noise import PROTOCOL as NOISE_PROTOCOL  # noqa: E402
from openpi_client.paired_noise import paired_action_noise  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from run_robocasa365_pi05_eval import (  # noqa: E402
    canonical_hash,
    construct_env,
    image_observation,
    language_observation,
    state_observation,
)


DEFAULT_TASKS = (
    "OpenStandMixerHead",
    "OpenCabinet",
    "PickPlaceDrawerToCounter",
    "CoffeeSetupMug",
)
DEFAULT_OUT = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/diagnostics/fp16_onpolicy_probe/"
    / "fp16_onpolicy_target_4tasks_s0-1_r4_n32.npz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            result.extend(range(int(left), int(right) + 1))
        else:
            result.append(int(token))
    if not result or len(result) != len(set(result)) or any(item < 0 for item in result):
        raise ValueError(f"invalid non-negative integer list: {value!r}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18602)
    parser.add_argument("--egl-device", type=int, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--trial-seeds", default="0,1")
    parser.add_argument("--split", default="target")
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--replans-per-trial", type=int, default=4)
    parser.add_argument(
        "--max-replans",
        type=int,
        default=None,
        help="Number of FP16 replans to execute before selecting rows. Defaults to --replans-per-trial.",
    )
    parser.add_argument(
        "--selection",
        choices=("early", "tail", "all"),
        default="early",
        help="Which on-policy replan states to persist from each trial.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def observation_row(obs: dict, *, task: str, seed: int, env_step: int, replan: int, resize: int) -> dict:
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


def validate_fp16_server(metadata: dict) -> str:
    runtime = metadata.get("openpi_runtime") or {}
    duquant = runtime.get("duquant") or {}
    dtype = runtime.get("model_dtype") or {}
    if runtime.get("config_id") != "fp16":
        raise ValueError(f"source server is not fp16: {runtime.get('config_id')!r}")
    if bool(duquant.get("enabled")) or int(duquant.get("wrapped_layers", 0)) != 0:
        raise ValueError("source server unexpectedly has quantized layers")
    if dtype.get("resolved") != "float16":
        raise ValueError(f"source server dtype is not float16: {dtype!r}")
    return canonical_hash(metadata)


def collect_trial(
    client: WebsocketClientPolicy,
    *,
    task: str,
    seed: int,
    split: str,
    resize: int,
    replan_steps: int,
    replans_per_trial: int,
    max_replans: int,
    selection: str,
    egl_device: int,
) -> tuple[list[dict], dict]:
    env = None
    rows: list[dict] = []
    steps = 0
    success = False
    terminated = False
    truncated = False
    try:
        env, construct_s = construct_env(task, split, egl_device)
        obs, _ = env.reset(seed=seed)
        horizon = get_task_horizon(task)
        candidates: list[dict] = []
        for replan in range(max_replans):
            candidates.append(
                observation_row(
                    obs,
                    task=task,
                    seed=seed,
                    env_step=steps,
                    replan=replan,
                    resize=resize,
                )
            )
            noise = paired_action_noise(task, seed, replan)
            request = {
                "observation/image": candidates[-1]["image"],
                "observation/wrist_image": candidates[-1]["wrist_image"],
                "observation/right_image": candidates[-1]["right_image"],
                "observation/state": candidates[-1]["state"],
                "prompt": candidates[-1]["prompt"],
            }
            response = client.infer(request, noise=noise)
            actions = np.asarray(response["actions"])
            if actions.shape != (50, 12) or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid FP16 actions for {task}/{seed}/{replan}: {actions.shape}")
            for index in range(min(replan_steps, horizon - steps)):
                obs, _reward, terminated, truncated, info = env.step(convert_action(actions[index]))
                steps += 1
                success = bool(info.get("success", False))
                if success or terminated or truncated or steps >= horizon:
                    break
            if success or terminated or truncated or steps >= horizon:
                break
        if selection == "all":
            rows = candidates
        elif selection == "tail":
            rows = candidates[-replans_per_trial:]
        else:
            rows = candidates[:replans_per_trial]
        return rows, {
            "task": task,
            "seed": seed,
            "rows": len(rows),
            "candidate_rows": len(candidates),
            "env_steps": steps,
            "success": success,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "construct_s": construct_s,
        }
    finally:
        if env is not None:
            env.close()


def make_noises(rows: list[dict]) -> np.ndarray:
    first = [
        np.asarray(paired_action_noise(row["task"], int(row["seed"]), int(row["replan"])), dtype=np.float32)
        for row in rows
    ]
    generator = torch.Generator(device="cpu").manual_seed(17)
    second = [
        torch.randn((50, 32), generator=generator, dtype=torch.float32).numpy()
        for _ in rows
    ]
    return np.stack(first + second).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    seeds = parse_int_list(args.trial_seeds)
    max_replans = args.max_replans if args.max_replans is not None else args.replans_per_trial
    if args.replans_per_trial <= 0 or args.replan_steps <= 0 or max_replans <= 0:
        raise ValueError("replan counts must be positive")
    if args.replans_per_trial > max_replans and args.selection != "all":
        raise ValueError("--replans-per-trial cannot exceed --max-replans")
    missing = [task for task in tasks if task not in TASK_SET_REGISTRY["atomic_seen"]]
    if missing:
        raise ValueError(f"tasks are not in atomic_seen registry: {missing}")
    output = Path(args.out).expanduser().resolve()
    sidecar = Path(str(output) + ".json")
    if (output.exists() or sidecar.exists()) and not args.force:
        raise FileExistsError(f"refusing to overwrite existing probe buffer: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    client = WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    server_hash = validate_fp16_server(server_metadata)
    rows: list[dict] = []
    trials: list[dict] = []
    for task in tasks:
        for seed in seeds:
            print(f"[fp16-onpolicy] {task}/{seed}", flush=True)
            trial_rows, summary = collect_trial(
                client,
                task=task,
                seed=seed,
                split=args.split,
                resize=args.resize,
                replan_steps=args.replan_steps,
                replans_per_trial=args.replans_per_trial,
                max_replans=max_replans,
                selection=args.selection,
                egl_device=args.egl_device,
            )
            rows.extend(trial_rows)
            trials.append(summary)
            print(
                f"[fp16-onpolicy] {task}/{seed}: rows={summary['rows']} "
                f"steps={summary['env_steps']} success={summary['success']}",
                flush=True,
            )
    if not rows:
        raise RuntimeError("collected zero FP16 on-policy observations")

    action_noises = make_noises(rows)
    stored_rows = rows + rows
    temporary = Path(str(output) + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            images=np.stack([row["image"] for row in stored_rows]),
            wrist_images=np.stack([row["wrist_image"] for row in stored_rows]),
            right_images=np.stack([row["right_image"] for row in stored_rows]),
            states=np.stack([row["state"] for row in stored_rows]).astype(np.float32),
            prompts=np.asarray([row["prompt"] for row in stored_rows]),
            task_ids=np.asarray([row["task"] for row in stored_rows]),
            env_seeds=np.asarray([row["seed"] for row in stored_rows], dtype=np.int64),
            env_steps=np.asarray([row["env_step"] for row in stored_rows], dtype=np.int64),
            replan_indices=np.asarray([row["replan"] for row in stored_rows], dtype=np.int64),
            action_noises=action_noises,
        )
    temporary.replace(output)
    payload = {
        "schema_version": 1,
        "kind": "fp16-teacher-onpolicy-target-probe",
        "npz": str(output),
        "sha256": sha256_file(output),
        "tasks": tasks,
        "trial_seeds": seeds,
        "split": args.split,
        "resize": args.resize,
        "replan_steps": args.replan_steps,
        "replans_per_trial": args.replans_per_trial,
        "max_replans": max_replans,
        "selection": args.selection,
        "n_obs": len(rows),
        "stored_rows": len(stored_rows),
        "noise_protocol": NOISE_PROTOCOL,
        "secondary_noise_seed": 17,
        "source_policy": "fp16",
        "source_port": args.port,
        "source_server_metadata_sha256": server_hash,
        "source_server_metadata": server_metadata,
        "trials": trials,
        "records": [
            {
                "task": row["task"],
                "seed": int(row["seed"]),
                "env_step": int(row["env_step"]),
                "replan": int(row["replan"]),
            }
            for row in rows
        ],
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("npz", "sha256", "n_obs", "stored_rows")}, indent=2))


if __name__ == "__main__":
    main()
