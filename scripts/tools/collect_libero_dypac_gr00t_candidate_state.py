#!/usr/bin/env python3
"""Collect one GR00T FCP candidate's result-blind states in one LIBERO suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from gr00t.eval.service import ExternalRobotInferenceClient  # noqa: E402
from collect_libero_dypac_gr00t_onpolicy import (  # noqa: E402
    action_seed,
    physical_actions,
    policy_observation,
)
from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--suite", choices=tuple(PROTOCOL["benchmark"]["suites"]), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--egl-device", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device)
    from libero.libero import benchmark
    from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env

    client = ExternalRobotInferenceClient(host=args.host, port=args.port)
    metadata = client.call_endpoint("get_runtime_info")
    if (
        metadata.get("protocol_id") != PROTOCOL["protocol_id"]
        or metadata.get("candidate_id") != args.candidate_id
        or metadata.get("suite") != args.suite
    ):
        raise RuntimeError("GR00T candidate-state server metadata mismatch")
    suite_name = PROTOCOL["benchmark"]["suites"][args.suite]
    suite = benchmark.get_benchmark_dict()[suite_name]()
    teacher_rows = []
    candidate_rows = []
    records = []
    started = time.monotonic()
    for task_id in PROTOCOL["calibration"]["selection_task_indices"]:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env = None
        try:
            env, prompt = get_libero_env(task, resolution=256, seed=0)
            env.reset()
            obs = env.set_init_state(initial_states[0])
            for _ in range(PROTOCOL["benchmark"]["stabilization_steps"]):
                obs, _, _, _ = env.step(get_libero_dummy_action())
            for replan in PROTOCOL["benchmark"]["queried_replan_indices"]:
                request, _ = policy_observation(obs, prompt)
                seed = action_seed(args.suite, task_id, 0, replan, "A")
                response = client.call_endpoint(
                    "get_action_seeded_paired",
                    {"observations": request, "action_seed": seed},
                )
                candidate = physical_actions(response["candidate"])
                teacher = physical_actions(response["teacher"])
                if replan in PROTOCOL["benchmark"]["retained_replan_indices"]:
                    candidate_rows.append(candidate[:5].copy())
                    teacher_rows.append(teacher[:5].copy())
                    records.append(
                        {
                            "suite": args.suite,
                            "task_index": task_id,
                            "state_index": 0,
                            "replan": replan,
                        }
                    )
                execute = candidate[:5].copy()
                execute[:, -1] = np.sign(1.0 - 2.0 * execute[:, -1])
                for action in execute:
                    obs, _, _, _ = env.step(action.tolist())
        finally:
            if env is not None:
                env.close()
        print(
            f"[candidate-state/gr00t] {args.candidate_id} {args.suite}/{task_id}",
            flush=True,
        )
    if len(records) != 12:
        raise RuntimeError(f"GR00T candidate-state suite rows {len(records)} != 12")
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_gr00t_candidate_state_suite_actions",
        "model": "gr00t",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_id": args.candidate_id,
        "suite": args.suite,
        "server_metadata": metadata,
        "sequences": 3,
        "rows": 12,
        "initial_state_indices": [0],
        "formal_initial_state_overlap": False,
        "uses_success_labels": False,
        "uses_test_rollout_feedback": False,
        "records": records,
        "teacher_actions": np.stack(teacher_rows).tolist(),
        "candidate_actions": np.stack(candidate_rows).tolist(),
        "elapsed_s": time.monotonic() - started,
    }
    atomic_json(args.out, payload)
    print(json.dumps({"candidate_id": args.candidate_id, "suite": args.suite, "rows": 12}, indent=2))


if __name__ == "__main__":
    main()
