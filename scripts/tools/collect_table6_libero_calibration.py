#!/usr/bin/env python3
"""Collect result-blind LIBERO calibration observations for Table 6.

The collector uses only initial states 0--4 and six no-op stabilization frames
per state.  Formal evaluation uses initial states 10--19, so no test rollout or
success label can enter calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUITE_NAMES = {
    "goal": "libero_goal",
    "spatial": "libero_spatial",
    "object": "libero_object",
    "long": "libero_10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITE_NAMES), required=True)
    parser.add_argument("--egl-device", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--states-per-task", type=int, default=5)
    parser.add_argument("--frames-per-state", type=int, default=6)
    parser.add_argument("--keep", type=int, default=256)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.states_per_task != 5 or args.frames_per_state != 6 or args.keep != 256:
        raise ValueError("Table 6 calibration is frozen to 5 states x 6 frames and 256 rows")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device)

    # Import after the EGL device is frozen.
    from libero.libero import benchmark
    from examples.Libero.eval.utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        quat2axisangle,
    )

    suite_name = SUITE_NAMES[args.suite]
    suite = benchmark.get_benchmark_dict()[suite_name]()
    per_task: dict[int, list[dict]] = {}
    for task_id in range(10):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if len(initial_states) < args.states_per_task:
            raise RuntimeError(f"{suite_name}/{task_id} has too few initial states")
        env = None
        rows: list[dict] = []
        try:
            env, description = get_libero_env(task, resolution=224)
            for initial_state_index in range(args.states_per_task):
                env.reset()
                obs = env.set_init_state(initial_states[initial_state_index])
                for frame_index in range(args.frames_per_state):
                    image, wrist = get_libero_image(obs)
                    state = np.concatenate(
                        (
                            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                            np.asarray(
                                quat2axisangle(obs["robot0_eef_quat"]),
                                dtype=np.float32,
                            ),
                            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                        )
                    ).astype(np.float32, copy=False)
                    if state.shape != (8,):
                        raise RuntimeError(f"unexpected LIBERO state shape: {state.shape}")
                    rows.append(
                        {
                            "image": np.ascontiguousarray(image),
                            "wrist": np.ascontiguousarray(wrist),
                            "right": np.zeros_like(image),
                            "state": state,
                            "prompt": str(description),
                            "task": f"{suite_name}:{task_id}",
                            "initial_state_index": initial_state_index,
                            "frame_index": frame_index,
                        }
                    )
                    if frame_index + 1 < args.frames_per_state:
                        obs, _reward, _done, _info = env.step(get_libero_dummy_action())
        finally:
            if env is not None:
                env.close()
        expected = args.states_per_task * args.frames_per_state
        if len(rows) != expected:
            raise RuntimeError(f"{suite_name}/{task_id}: {len(rows)} != {expected}")
        per_task[task_id] = rows
        print(f"[table6 calibration] {suite_name} task={task_id}: {len(rows)}", flush=True)

    # Round-robin tasks so every prefix, including the frozen first 256 rows,
    # remains task-balanced.
    ordered = [
        per_task[task_id][sample_index]
        for sample_index in range(args.states_per_task * args.frames_per_state)
        for task_id in range(10)
    ][: args.keep]
    if len(ordered) != args.keep:
        raise RuntimeError(f"calibration coverage drift: {len(ordered)} != {args.keep}")
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(1)
    noises_a = rng_a.standard_normal((args.keep, 50, 32)).astype(np.float32)
    noises_b = rng_b.standard_normal((args.keep, 50, 32)).astype(np.float32)

    output = Path(args.out).expanduser().resolve()
    sidecar = Path(str(output) + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite calibration artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            images=np.stack([row["image"] for row in ordered]),
            wrist_images=np.stack([row["wrist"] for row in ordered]),
            right_images=np.stack([row["right"] for row in ordered]),
            states=np.stack([row["state"] for row in ordered]).astype(np.float32),
            prompts=np.asarray([row["prompt"] for row in ordered]),
            task_ids=np.asarray([row["task"] for row in ordered]),
            env_seeds=np.asarray(
                [row["initial_state_index"] for row in ordered], dtype=np.int64
            ),
            env_steps=np.asarray([row["frame_index"] for row in ordered], dtype=np.int64),
            replan_indices=np.zeros(args.keep, dtype=np.int64),
            action_noises=noises_a,
            action_noises_b=noises_b,
        )
    temporary.replace(output)
    payload = {
        "schema_version": 1,
        "kind": "table6_libero_result_blind_calibration",
        "suite": suite_name,
        "rows": args.keep,
        "tasks": 10,
        "initial_state_indices": list(range(args.states_per_task)),
        "frames_per_initial_state": args.frames_per_state,
        "held_out_initial_state_indices": list(range(10, 20)),
        "overlap_with_held_out": False,
        "policy_queries": 0,
        "success_labels_observed": False,
        "test_rollout_feedback_used": False,
        "state_dim": 8,
        "image_shape": list(ordered[0]["image"].shape),
        "noise_a_seed": 0,
        "noise_b_seed": 1,
        "npz": str(output),
        "sha256": sha256_file(output),
    }
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
