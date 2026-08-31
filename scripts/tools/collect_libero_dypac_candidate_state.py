#!/usr/bin/env python3
"""Audit one FCP candidate on its own result-blind LIBERO states."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/pi05/openpi/packages/openpi-client/src"))
sys.path.insert(0, str(ROOT / "scripts/tools"))

from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from collect_libero_dypac_onpolicy import (  # noqa: E402
    action_noise,
    policy_observation,
)
from quantvla_libero_dypac import (  # noqa: E402
    PROTOCOL,
    PROTOCOL_PATH,
    atomic_json,
    physical_scale,
    sha256_file,
    summarize_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
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

    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    if metadata.get("protocol_id") != PROTOCOL["protocol_id"] or metadata.get("candidate_id") != args.candidate_id:
        raise RuntimeError("candidate-state server metadata mismatch")
    teacher_rows = []; candidate_rows = []; records = []
    started = time.monotonic()
    for suite_short, suite_name in PROTOCOL["benchmark"]["suites"].items():
        suite = benchmark.get_benchmark_dict()[suite_name]()
        for task_id in PROTOCOL["calibration"]["selection_task_indices"]:
            task = suite.get_task(task_id); initial_states = suite.get_task_init_states(task_id)
            env = None
            try:
                env, prompt = get_libero_env(task, resolution=256)
                env.reset(); obs = env.set_init_state(initial_states[0])
                for _ in range(PROTOCOL["benchmark"]["stabilization_steps"]):
                    obs, _, _, _ = env.step(get_libero_dummy_action())
                for replan in PROTOCOL["benchmark"]["queried_replan_indices"]:
                    request, _ = policy_observation(obs, prompt)
                    noise = action_noise(suite_short, task_id, 0, replan, "A")
                    response = client.infer(request, noise=noise)
                    candidate = np.asarray(response["actions"], dtype=np.float32)
                    teacher = np.asarray(response["teacher_actions"], dtype=np.float32)
                    if candidate.shape != (10, 7) or teacher.shape != (10, 7):
                        raise RuntimeError("candidate-state server returned invalid action shapes")
                    if replan in PROTOCOL["benchmark"]["retained_replan_indices"]:
                        candidate_rows.append(candidate[:5].copy()); teacher_rows.append(teacher[:5].copy())
                        records.append({"suite": suite_short, "task_index": task_id, "state_index": 0, "replan": replan})
                    execute = candidate[:5].copy(); execute[:, -1] = np.sign(1.0 - 2.0 * execute[:, -1])
                    for action in execute:
                        obs, _, _, _ = env.step(action.tolist())
            finally:
                if env is not None:
                    env.close()
            print(f"[candidate-state] {args.candidate_id} {suite_short}/{task_id}", flush=True)
    teacher = np.stack(teacher_rows); candidate = np.stack(candidate_rows)
    scale = physical_scale(teacher)
    summary = summarize_pair(teacher, candidate, records, scale=scale)
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_candidate_state_score",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_id": args.candidate_id,
        "server_metadata": metadata,
        "sequences": 12,
        "rows": 48,
        "initial_state_indices": [0],
        "formal_initial_state_overlap": False,
        "uses_success_labels": False,
        "uses_test_rollout_feedback": False,
        "d_pac": summary["d_pac_summary"]["d_pac"],
        "d_func": summary["d_func_summary"]["d_func"],
        "d_pac_summary": summary["d_pac_summary"],
        "d_func_summary": summary["d_func_summary"],
        "elapsed_s": time.monotonic() - started,
    }
    atomic_json(args.out, payload)
    print(json.dumps({"candidate_id": args.candidate_id, "d_pac": payload["d_pac"], "rows": 48}, indent=2))


if __name__ == "__main__":
    main()
