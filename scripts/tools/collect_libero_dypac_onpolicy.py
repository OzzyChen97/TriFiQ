#!/usr/bin/env python3
"""Collect result-blind FP16 on-policy LIBERO observations for DyPAC-VLA.

Each suite contributes exactly 64 rows.  Tasks 0, 4, and 8 contribute three
initial states and every other task contributes state 0.  Four ordered replans
are retained per episode after executing the FP16 teacher throughout.  Rewards,
termination flags, and success labels are deliberately ignored and not stored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OPENPI_CLIENT = ROOT / "code/pi05/openpi/packages/openpi-client/src"
sys.path.insert(0, str(OPENPI_CLIENT))

from openpi_client import image_tools  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


PROTOCOL_PATH = ROOT / "scripts/quantvla_libero_dypac_protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SUITES = PROTOCOL["benchmark"]["suites"]
SELECTION_TASKS = set(PROTOCOL["calibration"]["selection_task_indices"])
RETAINED_REPLANS = tuple(PROTOCOL["benchmark"]["retained_replan_indices"])
QUERIED_REPLANS = tuple(PROTOCOL["benchmark"]["queried_replan_indices"])


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


def protocol_sha256() -> str:
    return sha256_file(PROTOCOL_PATH)


def action_noise(suite: str, task: int, state: int, replan: int, stream: str) -> np.ndarray:
    material = (
        f"{PROTOCOL['protocol_id']}|{suite}|{task}|{state}|{replan}|{stream}"
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)


def policy_observation(obs: dict, prompt: str) -> tuple[dict, dict[str, np.ndarray]]:
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))

    from examples.Libero.eval.utils import quat2axisangle

    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            np.asarray(quat2axisangle(obs["robot0_eef_quat"]), dtype=np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32, copy=False)
    if state.shape != (8,):
        raise RuntimeError(f"unexpected LIBERO state shape: {state.shape}")
    request = {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": str(prompt),
    }
    return request, {"image": image, "wrist": wrist, "state": state}


def environment_episodes() -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for task in range(PROTOCOL["benchmark"]["tasks_per_suite"]):
        count = 3 if task in SELECTION_TASKS else 1
        rows.extend((task, state) for state in range(count))
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
        raise FileExistsError(f"refusing to overwrite DyPAC calibration shard: {output}")

    from libero.libero import benchmark
    from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env

    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    runtime = metadata.get("openpi_runtime") or {}
    if runtime.get("config_id") != "fp16_libero_dypac_calibration":
        raise RuntimeError(f"collector requires attested FP16 server, got {runtime.get('config_id')!r}")
    if (runtime.get("duquant") or {}).get("enabled"):
        raise RuntimeError("collector source policy unexpectedly enables DuQuant")
    if int((runtime.get("protocol") or {}).get("flow_steps", -1)) != 10:
        raise RuntimeError("collector source policy does not use the frozen 10-step solver")

    suite_name = SUITES[args.suite]
    suite = benchmark.get_benchmark_dict()[suite_name]()
    rows: list[dict] = []
    episode_summaries: list[dict] = []
    for task_id, state_index in environment_episodes():
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env = None
        started = time.monotonic()
        try:
            env, prompt = get_libero_env(task, resolution=256)
            env.reset()
            obs = env.set_init_state(initial_states[state_index])
            for _ in range(PROTOCOL["benchmark"]["stabilization_steps"]):
                obs, _, _, _ = env.step(get_libero_dummy_action())
            for replan in QUERIED_REPLANS:
                request, stored = policy_observation(obs, prompt)
                noise_a = action_noise(args.suite, task_id, state_index, replan, "A")
                response = client.infer(request, noise=noise_a)
                actions = np.asarray(response["actions"], dtype=np.float32)
                if actions.shape != (10, 7) or not np.isfinite(actions).all():
                    raise RuntimeError(
                        f"invalid FP16 action chunk at {args.suite}/{task_id}/{state_index}/{replan}: "
                        f"{actions.shape}"
                    )
                if replan in RETAINED_REPLANS:
                    rows.append(
                        {
                            **stored,
                            "prompt": str(prompt),
                            "task": f"{suite_name}:{task_id}",
                            "task_id": task_id,
                            "state_index": state_index,
                            "replan": replan,
                            "noise_a": noise_a,
                            "noise_b": action_noise(
                                args.suite, task_id, state_index, replan, "B"
                            ),
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
            f"[libero-dypac] {args.suite} task={task_id} state={state_index} "
            f"rows={len(rows)}/64",
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
            states=np.stack([row["state"] for row in rows]).astype(np.float32),
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
            teacher_actions=np.stack([row["teacher_actions"] for row in rows]),
        )
    temporary.replace(output)
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_fp16_onpolicy_shard",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": protocol_sha256(),
        "suite": args.suite,
        "suite_name": suite_name,
        "rows": len(rows),
        "selection_rows": selection_rows,
        "tasks": 10,
        "episodes": len(episode_summaries),
        "formal_initial_state_indices": PROTOCOL["benchmark"]["formal_initial_state_indices"],
        "overlap_with_formal": False,
        "uses_success_labels": False,
        "uses_test_rollout_feedback": False,
        "source_server_metadata": metadata,
        "episodes_detail": episode_summaries,
        "npz": str(output),
        "sha256": sha256_file(output),
    }
    temporary_sidecar = Path(str(sidecar) + f".tmp.{os.getpid()}")
    temporary_sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_sidecar.replace(sidecar)
    print(json.dumps({k: payload[k] for k in ("npz", "sha256", "rows", "selection_rows")}, indent=2))


if __name__ == "__main__":
    main()
