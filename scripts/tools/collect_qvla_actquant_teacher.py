#!/usr/bin/env python3
"""Collect deterministic FP16-teacher calibration trajectories from RoboCasa365.

Each episode is an independently checksummed compressed archive.  Workers own
disjoint hash shards and may be restarted safely; no success filtering is
performed.  The archive records every replan observation and the full teacher
action chunk returned for that observation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import robocasa  # noqa: F401 -- register environments before repository path edits
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))
sys.path.insert(0, str(REPO_ROOT / "code" / "pi05" / "openpi" / "packages" / "openpi-client" / "src"))

from openpi_client import image_tools  # noqa: E402
from openpi_client.paired_noise import paired_action_noise  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from run_robocasa365_gr00t_eval import (  # noqa: E402
    ACTION_DIMS,
    SEND_LANG_KEYS,
    SEND_STATE_KEYS,
    SEND_VIDEO_KEYS,
    _Gr00tZMQClient,
    _normalize_action_chunks,
    action_noise_seed,
    paired_action_noise_tensor,
)


ACTION_ORDER = tuple(ACTION_DIMS)


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def image(obs: dict, key: str, size: int) -> np.ndarray:
    value = np.asarray(obs[key])
    while value.ndim > 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"invalid image {key}: {value.shape}")
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(value, size, size))


def state(obs: dict) -> np.ndarray:
    values = []
    for key in (
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.base_position",
        "state.base_rotation",
        "state.gripper_qpos",
    ):
        value = np.asarray(obs[key], dtype=np.float32)
        while value.ndim > 1 and value.shape[0] == 1:
            value = value[0]
        values.append(value.reshape(-1))
    result = np.concatenate(values).astype(np.float32, copy=False)
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError(f"invalid canonical state: {result.shape}")
    return result


def prompt(obs: dict) -> str:
    value = obs["annotation.human.task_description"]
    if isinstance(value, (list, tuple, np.ndarray)):
        value = np.asarray(value).reshape(-1)[0]
    return str(value)


def construct_env(task: str, split: str, seed: int, egl_device: int):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)
    lock_path = REPO_ROOT / "runs" / f".robocasa365_construct_gpu{egl_device}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        env = RoboCasaGymEnv(
            env_name=task,
            enable_render=True,
            split=split,
            seed=seed,
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
    return env


def gr00t_request(obs: dict, task: str, seed: int, replan: int) -> dict:
    request = {}
    for key in SEND_VIDEO_KEYS + SEND_STATE_KEYS + SEND_LANG_KEYS:
        value = obs[key]
        if key.startswith("video."):
            value = np.asarray(value)
            if value.ndim == 3:
                value = value[np.newaxis]
        elif key.startswith("state."):
            value = np.asarray(value, dtype=np.float32)
            if value.ndim == 1:
                value = value[np.newaxis]
        elif isinstance(value, str):
            value = [value]
        request[key] = value
    request["eval_metadata"] = {
        "task_name": task,
        "task": task,
        "seed": seed,
        "replan_index": replan,
        "model_id": "gr00t",
        "calibration": True,
    }
    return request


def gr00t_actions(response: dict) -> tuple[np.ndarray, list[dict]]:
    chunks = _normalize_action_chunks(response)
    length = min(len(chunks[key]) for key in ACTION_ORDER)
    dense = np.concatenate([chunks[key][:length] for key in ACTION_ORDER], axis=1).astype(np.float32)
    actions = [
        {f"action.{key}": np.atleast_1d(chunks[key][index]) for key in ACTION_ORDER}
        for index in range(length)
    ]
    return dense, actions


def write_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_fp16_teacher(model: str, metadata: dict) -> dict:
    section = metadata if model == "gr00t" else metadata.get("openpi_runtime", {})
    precision = section.get("model_dtype") or {}
    linear_dtypes = (
        precision.get("linear_conv_layers_by_weight_dtype")
        if model == "gr00t"
        else precision.get("linear_layers_by_weight_dtype")
    ) or {}
    if precision.get("resolved") != "float16" or set(linear_dtypes) != {"float16"}:
        raise RuntimeError(
            f"{model} teacher is not attested FP16: {precision or 'missing model_dtype'}"
        )
    return {**precision, "strict_all_linear_conv_fp16": True}


def collect_episode(
    *,
    model: str,
    client,
    task: str,
    seed: int,
    split: str,
    egl_device: int,
    resize: int,
    replan_steps: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    env = None
    started = time.perf_counter()
    try:
        env = construct_env(task, split, seed, egl_device)
        seed_all(seed)
        obs, _ = env.reset(seed=seed)
        horizon = get_task_horizon(task)
        frames = {"images": [], "wrist_images": [], "right_images": [], "states": []}
        prompts, teacher_actions, action_noises, env_steps = [], [], [], []
        steps = replans = 0
        done = success = False
        termination = "official_horizon"
        while steps < horizon and not done:
            frames["images"].append(image(obs, "video.robot0_agentview_left", resize))
            frames["wrist_images"].append(image(obs, "video.robot0_eye_in_hand", resize))
            frames["right_images"].append(image(obs, "video.robot0_agentview_right", resize))
            frames["states"].append(state(obs))
            prompts.append(prompt(obs))
            env_steps.append(steps)
            if model == "gr00t":
                noise_seed = action_noise_seed(task, seed, replans)
                noise = paired_action_noise_tensor(noise_seed)
                response = client.get_action(
                    gr00t_request(obs, task, seed, replans), action_seed=noise_seed
                )
                dense, executable = gr00t_actions(response)
            else:
                request = {
                    "observation/image": frames["images"][-1],
                    "observation/wrist_image": frames["wrist_images"][-1],
                    "observation/right_image": frames["right_images"][-1],
                    "observation/state": frames["states"][-1],
                    "prompt": prompts[-1],
                    "__openpi_eval_metadata__": {
                        "task_name": task,
                        "task_set": split,
                        "seed": seed,
                        "replan_index": replans,
                        "config_id": "fp16_teacher_proxy",
                        "calibration": True,
                    },
                }
                noise = paired_action_noise(task, seed, replans)
                response = client.infer(request, noise=noise)
                dense = np.asarray(response["actions"], dtype=np.float32)
                if dense.ndim != 2 or dense.shape[1] != 12:
                    raise RuntimeError(f"invalid pi0.5 teacher actions: {dense.shape}")
                executable = [convert_action(dense[index]) for index in range(len(dense))]
            if not np.isfinite(dense).all():
                raise RuntimeError("teacher actions contain non-finite values")
            teacher_actions.append(dense)
            action_noises.append(np.asarray(noise, dtype=np.float32))
            replans += 1
            execute = min(replan_steps, horizon - steps, len(executable))
            for index in range(execute):
                obs, _, terminated, truncated, info = env.step(executable[index])
                steps += 1
                success = bool(info.get("success", False))
                if success or terminated or truncated:
                    done = True
                    termination = "success" if success else "terminated" if terminated else "truncated"
                    break
        if not teacher_actions:
            raise RuntimeError("episode produced no calibration frames")
        arrays = {
            **{key: np.stack(value) for key, value in frames.items()},
            "prompts": np.asarray(prompts),
            "teacher_actions": np.stack(teacher_actions).astype(np.float32),
            "action_noises": np.stack(action_noises).astype(np.float32),
            "env_steps": np.asarray(env_steps, dtype=np.int64),
            "task": np.asarray(task),
            "env_seed": np.asarray(seed, dtype=np.int64),
        }
        row = {
            "status": "complete",
            "model": model,
            "task": task,
            "env_seed": seed,
            "episode_key": f"{task}/{seed}",
            "split": split,
            "source": "fp16_teacher_proxy",
            "source_protocol_equivalent": False,
            "success": success,
            "termination": termination,
            "steps": steps,
            "replans": replans,
            "frame_count": replans,
            "teacher_action_shape": list(arrays["teacher_actions"].shape),
            "action_noise_shape": list(arrays["action_noises"].shape),
            "wall_seconds": time.perf_counter() - started,
            "test_results_used": False,
        }
        return row, arrays
    finally:
        if env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=("gr00t", "pi05"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--egl-device", type=int, required=True)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--start-gate", type=Path)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard")
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    if manifest["model"] != args.model:
        raise SystemExit("manifest/model mismatch")
    episodes = [
        row
        for row in manifest["qvla_episodes"]
        if int.from_bytes(hashlib.sha256(row["episode_key"].encode()).digest()[:8], "big")
        % args.shard_count
        == args.shard_index
    ]
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    output = args.out_dir.resolve()
    episode_dir = output / "episodes"
    journal = output / f"worker_{args.shard_index:03d}_of_{args.shard_count:03d}.jsonl"
    output.mkdir(parents=True, exist_ok=True)

    def read_existing() -> dict[str, dict]:
        result: dict[str, dict] = {}
        for path in sorted(output.glob("worker_*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row["episode_key"]
                if key in result and result[key] != row:
                    raise ValueError(f"conflicting calibration row: {key}")
                result[key] = row
        return result

    existing = read_existing()
    # ``worker`` alone is ambiguous after a deterministic shard-count change
    # (for example 8 -> 12 workers).  Preserve exactly the rows already in this
    # layout-specific journal instead of copying same-index rows from older
    # layouts.  Global episode-key resume semantics still come from
    # ``existing`` above.
    owned: dict[str, dict] = {}
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                owned[row["episode_key"]] = row

    def persist() -> None:
        temporary = journal.with_name(f".{journal.name}.tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            for key in sorted(owned):
                handle.write(json.dumps(owned[key], sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, journal)

    client = (
        _Gr00tZMQClient(host=args.host, port=args.port, timeout_ms=120_000)
        if args.model == "gr00t"
        else WebsocketClientPolicy(args.host, args.port)
    )
    server_metadata = client.runtime_info() if args.model == "gr00t" else client.get_server_metadata()
    # GR00T publishes a semantic identity excluding transient allocator
    # counters; pi0.5 metadata is captured once by its server and is stable.
    server_metadata_sha256 = (
        str(server_metadata["metadata_sha256"])
        if args.model == "gr00t"
        else canonical_hash(server_metadata)
    )
    teacher_precision = require_fp16_teacher(args.model, server_metadata)
    if args.start_gate is not None:
        if args.ready_file is None:
            raise ValueError("--start-gate requires --ready-file")
        ready = args.ready_file.resolve()
        ready.parent.mkdir(parents=True, exist_ok=True)
        temporary = ready.with_name(f".{ready.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "server_metadata_sha256": server_metadata_sha256,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, ready)
        gate = args.start_gate.resolve()
        print(f"[teacher] ready; waiting for start gate {gate}", flush=True)
        while not gate.is_file():
            time.sleep(0.5)
        # The previous layout remains live during warm-up.  Refresh only after
        # the controller closes the gate transition so rows committed while we
        # waited are skipped rather than duplicated.
        existing = read_existing()
        print(f"[teacher] start gate opened; refreshed {len(existing)} rows", flush=True)
    for index, spec in enumerate(episodes, start=1):
        key = spec["episode_key"]
        if key in existing:
            archive = Path(existing[key]["archive"])
            if archive.is_file() and sha256_file(archive) == existing[key]["archive_sha256"]:
                print(f"[teacher] reuse {key}", flush=True)
                continue
            raise RuntimeError(f"committed calibration archive is missing or corrupt: {key}")
        row, arrays = collect_episode(
            model=args.model,
            client=client,
            task=spec["task"],
            seed=int(spec["env_seed"]),
            split=spec["split"],
            egl_device=args.egl_device,
            resize=args.resize,
            replan_steps=args.replan_steps,
        )
        archive = episode_dir / spec["task"] / f"seed_{int(spec['env_seed']):05d}.npz"
        write_npz_atomic(archive, **arrays)
        row.update(
            {
                "archive": str(archive),
                "archive_sha256": sha256_file(archive),
                "manifest_sha256": sha256_file(args.manifest.resolve()),
                "server_metadata_sha256": server_metadata_sha256,
                "teacher_precision": teacher_precision,
                "worker": args.shard_index,
                "worker_count": args.shard_count,
            }
        )
        owned[key] = row
        existing[key] = row
        persist()
        print(
            f"[teacher] {index}/{len(episodes)} {key}: frames={row['frame_count']} "
            f"success={row['success']} wall={row['wall_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
