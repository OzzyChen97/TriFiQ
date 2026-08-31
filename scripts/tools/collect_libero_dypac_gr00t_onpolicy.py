#!/usr/bin/env python3
"""Collect result-blind FP16 on-policy LIBERO observations for GR00T DyPAC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from gr00t.eval.service import ExternalRobotInferenceClient  # noqa: E402


PROTOCOL_PATH = ROOT / "scripts/quantvla_libero_dypac_protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SUITES = PROTOCOL["benchmark"]["suites"]
SELECTION_TASKS = set(PROTOCOL["calibration"]["selection_task_indices"])
RETAINED_REPLANS = tuple(PROTOCOL["benchmark"]["retained_replan_indices"])
QUERIED_REPLANS = tuple(PROTOCOL["benchmark"]["queried_replan_indices"])
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--egl-device", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_seed(suite: str, task: int, state: int, replan: int, stream: str) -> int:
    material = (
        f"{PROTOCOL['protocol_id']}|{suite}|{task}|{state}|{replan}|{stream}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (2**63)


def action_noise(seed: int) -> np.ndarray:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn((16, 32), generator=generator, dtype=torch.float32).numpy()


def policy_observation(obs: dict, prompt: str) -> tuple[dict, dict[str, np.ndarray]]:
    from examples.Libero.eval.utils import get_libero_image, quat2axisangle

    image, wrist = get_libero_image(obs)
    xyz = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    rpy = np.asarray(quat2axisangle(obs["robot0_eef_quat"]), dtype=np.float32)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    state = np.concatenate((xyz, rpy, gripper)).astype(np.float32, copy=False)
    if state.shape != (8,):
        raise RuntimeError(f"unexpected LIBERO state shape: {state.shape}")
    request = {
        "video.image": np.ascontiguousarray(image)[None, ...],
        "video.wrist_image": np.ascontiguousarray(wrist)[None, ...],
        "state.x": state[0:1].reshape(1, 1),
        "state.y": state[1:2].reshape(1, 1),
        "state.z": state[2:3].reshape(1, 1),
        "state.roll": state[3:4].reshape(1, 1),
        "state.pitch": state[4:5].reshape(1, 1),
        "state.yaw": state[5:6].reshape(1, 1),
        "state.gripper": state[6:8].reshape(1, 2),
        "annotation.human.action.task_description": [str(prompt)],
    }
    return request, {
        "image": np.ascontiguousarray(image),
        "wrist": np.ascontiguousarray(wrist),
        "state": state,
    }


def physical_actions(response: dict) -> np.ndarray:
    components = []
    for key in ACTION_KEYS:
        name = f"action.{key}"
        if name not in response:
            raise KeyError(f"GR00T response omitted {name}; got {sorted(response)}")
        value = np.asarray(response[name], dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2 or value.shape[-1] != 1:
            raise ValueError(f"invalid GR00T action component {name}: {value.shape}")
        components.append(value)
    value = np.concatenate(components, axis=-1)
    if value.shape != (16, 7) or not np.isfinite(value).all():
        raise ValueError(f"invalid GR00T physical action chunk: {value.shape}")
    return value


def environment_episodes() -> list[tuple[int, int]]:
    rows = []
    for task in range(PROTOCOL["benchmark"]["tasks_per_suite"]):
        rows.extend(
            (task, state)
            for state in range(3 if task in SELECTION_TASKS else 1)
        )
    if len(rows) != 16:
        raise RuntimeError(f"suite episode design drifted: {len(rows)} != 16")
    return rows


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device)
    output = Path(args.out).expanduser().resolve()
    sidecar = Path(str(output) + ".json")
    if (output.exists() or sidecar.exists()) and not args.force:
        raise FileExistsError(f"refusing to overwrite GR00T DyPAC shard: {output}")

    from libero.libero import benchmark
    from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env

    client = ExternalRobotInferenceClient(host=args.host, port=args.port)
    runtime = client.call_endpoint("get_runtime_info")
    if runtime.get("config_id") != "fp16_libero_dypac_calibration":
        raise RuntimeError(
            f"collector requires attested FP16 server, got {runtime.get('config_id')!r}"
        )
    if int(runtime.get("wrapped_layers", -1)) != 0:
        raise RuntimeError("collector source policy unexpectedly enables quantization")
    if int(runtime.get("denoising_steps", -1)) != 10:
        raise RuntimeError("collector source policy does not use the frozen 10-step solver")

    suite_name = SUITES[args.suite]
    suite = benchmark.get_benchmark_dict()[suite_name]()
    rows: list[dict] = []
    episode_summaries = []
    for task_id, state_index in environment_episodes():
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env = None
        started = time.monotonic()
        try:
            env, prompt = get_libero_env(task, resolution=256, seed=0)
            env.reset()
            obs = env.set_init_state(initial_states[state_index])
            for _ in range(PROTOCOL["benchmark"]["stabilization_steps"]):
                obs, _, _, _ = env.step(get_libero_dummy_action())
            for replan in QUERIED_REPLANS:
                request, stored = policy_observation(obs, prompt)
                seed_a = action_seed(args.suite, task_id, state_index, replan, "A")
                seed_b = action_seed(args.suite, task_id, state_index, replan, "B")
                response = client.call_endpoint(
                    "get_action_seeded",
                    {"observations": request, "action_seed": seed_a},
                )
                actions = physical_actions(response)
                if replan in RETAINED_REPLANS:
                    rows.append(
                        {
                            **stored,
                            "prompt": str(prompt),
                            "task": f"{suite_name}:{task_id}",
                            "task_id": task_id,
                            "state_index": state_index,
                            "replan": replan,
                            "noise_a": action_noise(seed_a),
                            "noise_b": action_noise(seed_b),
                            "seed_a": seed_a,
                            "seed_b": seed_b,
                            "teacher_actions": actions[:5].copy(),
                        }
                    )
                execute = actions[: PROTOCOL["benchmark"]["replan_steps"]].copy()
                execute[:, -1] = np.sign(1.0 - 2.0 * execute[:, -1])
                for action in execute:
                    obs, _, _, _ = env.step(action.tolist())
        finally:
            if env is not None:
                env.close()
        episode_summaries.append(
            {
                "task_id": task_id,
                "initial_state_index": state_index,
                "retained_rows": len(RETAINED_REPLANS),
                "elapsed_s": time.monotonic() - started,
            }
        )
        print(
            f"[libero-dypac/gr00t] {args.suite} task={task_id} "
            f"state={state_index} rows={len(rows)}/64",
            flush=True,
        )

    if len(rows) != 64:
        raise RuntimeError(f"suite row count drifted: {len(rows)} != 64")
    selection_rows = sum(row["task_id"] in SELECTION_TASKS for row in rows)
    if selection_rows != 36:
        raise RuntimeError(f"suite selection count drifted: {selection_rows} != 36")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            images=np.stack([row["image"] for row in rows]),
            wrist_images=np.stack([row["wrist"] for row in rows]),
            states=np.stack([row["state"] for row in rows]),
            prompts=np.asarray([row["prompt"] for row in rows]),
            task_ids=np.asarray([row["task"] for row in rows]),
            suite_ids=np.asarray([args.suite] * len(rows)),
            task_indices=np.asarray([row["task_id"] for row in rows], dtype=np.int64),
            env_seeds=np.asarray([row["state_index"] for row in rows], dtype=np.int64),
            replan_indices=np.asarray([row["replan"] for row in rows], dtype=np.int64),
            selection_rows=np.asarray(
                [row["task_id"] in SELECTION_TASKS for row in rows], dtype=np.bool_
            ),
            action_noises=np.stack([row["noise_a"] for row in rows]),
            action_noises_b=np.stack([row["noise_b"] for row in rows]),
            action_seeds=np.asarray([row["seed_a"] for row in rows], dtype=np.uint64),
            action_seeds_b=np.asarray([row["seed_b"] for row in rows], dtype=np.uint64),
            teacher_actions=np.stack([row["teacher_actions"] for row in rows]),
        )
    temporary.replace(output)
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_gr00t_fp16_onpolicy_shard",
        "model": "gr00t",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "suite": args.suite,
        "suite_name": suite_name,
        "rows": 64,
        "selection_rows": 36,
        "tasks": 10,
        "episodes": 16,
        "formal_initial_state_indices": PROTOCOL["benchmark"]["formal_initial_state_indices"],
        "overlap_with_formal": False,
        "uses_success_labels": False,
        "uses_test_rollout_feedback": False,
        "source_server_metadata": runtime,
        "episodes_detail": episode_summaries,
        "npz": str(output),
        "sha256": sha256_file(output),
    }
    temporary_sidecar = Path(str(sidecar) + f".tmp.{os.getpid()}")
    temporary_sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_sidecar.replace(sidecar)
    print(json.dumps({key: payload[key] for key in ("npz", "sha256", "rows", "selection_rows")}, indent=2))


if __name__ == "__main__":
    main()
