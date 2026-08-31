#!/usr/bin/env python3
"""RoboCasa365 GR00T N1.5 eval client (QuantVLA v1.4, Stage D).

Runs in the robocasa365 conda env and talks to a GR00T inference service
(scripts/inference_service.py) loaded with a target_posttraining checkpoint
and --data-config examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig.

CRITICAL: import robocasa BEFORE touching sys.path — putting <repo>/code on
sys.path first shadows the real robocasa package with the repo directory and
kills task registration (396 -> 19 envs, D-023).

Metrics (v1): per-trial full-task success + episode length + failure step.
Subgoal-level metrics (avg completed subgoals, P(>=k), transition delay)
require the benchmark's per-stage annotations and land in a follow-up.

Usage:
    # server (groot_test env):
    #   python scripts/inference_service.py --model_path <robocasa365 ckpt> \
    #       --data-config examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig --port 5570
    # client (robocasa365 env):
    python scripts/run_robocasa365_gr00t_eval.py --port 5570 \
        --task-set atomic_seen --n-trials 3 --tasks AddIceCubes,PrepareSmoothie \
        --out runs/robocasa365_eval/atomic_seen_smoke.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import robocasa  # noqa: F401  — MUST precede any sys.path change (D-023)
from robocasa.utils.dataset_registry import TARGET_TASKS  # noqa: E402
from robocasa.utils.dataset_registry_utils import get_task_horizon  # noqa: E402
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

import numpy as np  # noqa: E402
from quantvla_cross_model_protocol import (  # noqa: E402
    closed_loop_row_protocol,
    closed_loop_runtime_protocol,
    require_protocol_attestation,
)

# obs keys the RoboCasa365DataConfig consumes — filter the wrapper's obs
# (which also emits legacy res256/res512 aliases and extra state keys) so the
# server-side transforms only see configured modalities
SEND_VIDEO_KEYS = [
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
]
SEND_STATE_KEYS = [
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
]
SEND_LANG_KEYS = ["annotation.human.task_description"]

ACTION_DIMS = {
    "end_effector_position": 3,
    "end_effector_rotation": 3,
    "gripper_close": 1,
    "base_motion": 4,
    "control_mode": 1,
}

ACTION_NOISE_SCHEME = "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"


def canonical_state(obs: dict) -> np.ndarray:
    """RoboCasa obs -> canonical 16-dim state (eef pos/rot, base pos/rot, gripper)."""
    parts = [
        np.asarray(obs["state.end_effector_position_relative"], dtype=np.float32).reshape(-1),
        np.asarray(obs["state.end_effector_rotation_relative"], dtype=np.float32).reshape(-1),
        np.asarray(obs["state.base_position"], dtype=np.float32).reshape(-1),
        np.asarray(obs["state.base_rotation"], dtype=np.float32).reshape(-1),
        np.asarray(obs["state.gripper_qpos"], dtype=np.float32).reshape(-1),
    ]
    state = np.concatenate(parts)
    if state.size != 16:
        raise ValueError(f"canonical state size drift: {state.size}")
    return state


def paired_action_noise_tensor(noise_seed: int) -> np.ndarray:
    """Bitwise replica of the server-side paired noise (torch-cpu-normal-v1)."""
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(noise_seed)
    return torch.randn((50, 32), generator=generator, dtype=torch.float32).numpy()
ENVIRONMENT_SEED_PROTOCOL = "python-random/numpy-global/robocasa-constructor-and-reset-v1"


def seed_environment_rng(seed: int) -> None:
    """Seed RNGs used before and during RoboCasa environment construction."""
    random.seed(int(seed))
    np.random.seed(int(seed))


def action_noise_seed(task: str, env_seed: int, replan_index: int) -> int:
    """Stable request seed, independent of Python hash randomization/sharding."""
    payload = f"quantvla-robocasa365-v1\0{task}\0{env_seed}\0{replan_index}".encode()
    # torch.Generator.manual_seed accepts signed 64-bit seeds portably.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _normalize_action_chunks(action_chunk: dict) -> dict[str, np.ndarray]:
    """Return server actions as per-key ``(horizon, action_dim)`` arrays.

    ``inference_service.py`` normally removes the batch dimension and returns
    ``(16, D)``. Some service implementations preserve it as ``(1, 16, D)``;
    accepting both avoids confusing the horizon with a batch dimension.
    """
    chunks: dict[str, np.ndarray] = {}
    for key, expected_dim in ACTION_DIMS.items():
        value = np.asarray(action_chunk[f"action.{key}"])
        while value.ndim > 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 1:
            if expected_dim == 1:
                value = value[:, np.newaxis]
            elif value.shape[0] == expected_dim:
                value = value[np.newaxis, :]
        if value.ndim != 2 or value.shape[1] != expected_dim:
            raise ValueError(
                f"Unexpected action.{key} shape {value.shape}; expected (H, {expected_dim})"
            )
        chunks[key] = value
    return chunks


class _Gr00tZMQClient:
    """Minimal ZMQ REQ client for the GR00T inference service.

    Self-contained (zmq + msgpack + numpy only) so the robocasa365 conda env
    does not need the heavy gr00t data-stack dependencies. Protocol mirrors
    gr00t.eval.service.BaseInferenceClient: msgpack {"endpoint", "data"},
    15s recv/send timeouts.
    """

    def __init__(self, host: str = "localhost", port: int = 5570, timeout_ms: int = 15000):
        import zmq
        import msgpack

        self.msgpack = msgpack
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            import io

            out = io.BytesIO()
            np.save(out, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": out.getvalue()}
        return obj

    @staticmethod
    def _decode(obj):
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            import io

            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    def call(self, endpoint: str, data: dict | None = None) -> dict:
        import zmq

        request = {"endpoint": endpoint, "data": data or {}}
        self.sock.send(self.msgpack.packb(request, default=self._encode))
        try:
            resp = self.msgpack.unpackb(self.sock.recv(), object_hook=self._decode)
        except zmq.Again:
            raise RuntimeError(f"inference server timeout on {endpoint}") from None
        if "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp

    def get_action(self, obs: dict, action_seed: int | None = None) -> dict:
        if action_seed is None:
            return self.call("get_action", obs)
        return self.call(
            "get_action_seeded",
            {"observations": obs, "action_seed": int(action_seed)},
        )

    def runtime_info(self) -> dict:
        return self.call("get_runtime_info")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RoboCasa365 GR00T eval client")
    p.add_argument("--port", type=int, default=5570)
    p.add_argument("--task-set", default="atomic_seen",
                   choices=["atomic_seen", "composite_seen", "composite_unseen", "custom"])
    p.add_argument("--tasks", default=None, help="comma list overriding the task set")
    p.add_argument("--split", default="target")
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--exact-seed",
        type=int,
        default=None,
        help=("Use this exact environment seed. This mode is reserved for the "
              "per-trial crash-tolerant driver and requires exactly one task "
              "and one trial."),
    )
    p.add_argument(
        "--exact-seeds",
        default=None,
        help=("Comma-separated exact environment seeds for one task. This is "
              "used by the crash-tolerant batched driver."),
    )
    p.add_argument(
        "--fresh-env-per-trial",
        action="store_true",
        help=("Reconstruct the RoboCasa environment for every trial. This avoids "
              "the known second-reset crash while amortizing Python imports over "
              "multiple trials."),
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Episode horizon override; 0 uses RoboCasa's official per-task horizon.",
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=16,
        help="Number of predicted action-chunk steps to execute before replanning.",
    )
    p.add_argument(
        "--paired-action-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Use deterministic common-random-number diffusion noise keyed by "
              "task/environment-seed/replan-index."),
    )
    p.add_argument(
        "--expect-runtime-selector",
        action="store_true",
        help="Require each server response to include runtime selector metadata.",
    )
    p.add_argument(
        "--candidate-state-archive",
        default=None,
        help=("Optional .npz path. When set, one observation row (images, "
              "states, prompt, task/seed/step/replan keys, paired action "
              "noise) is appended at every replan for the candidate-state "
              "teacher audit. Requires --paired-action-noise."),
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.split != "target" or args.n_action_steps != 16:
        raise SystemExit("cross-model formal evaluation requires target/execute-16")
    if not args.paired_action_noise or not args.fresh_env_per_trial:
        raise SystemExit("cross-model formal evaluation requires paired noise and a fresh env")
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = list(TARGET_TASKS[args.task_set])
    if args.exact_seed is not None and args.exact_seeds is not None:
        raise SystemExit("--exact-seed and --exact-seeds are mutually exclusive")
    exact_seeds = None
    if args.exact_seeds is not None:
        try:
            exact_seeds = [int(v.strip()) for v in args.exact_seeds.split(",") if v.strip()]
        except ValueError as exc:
            raise SystemExit("--exact-seeds must contain integers") from exc
        if (not exact_seeds or len(exact_seeds) != len(set(exact_seeds)) or
                len(tasks) != 1 or args.n_trials != len(exact_seeds)):
            raise SystemExit(
                "--exact-seeds requires one task, unique seeds, and matching --n-trials"
            )
    if args.exact_seed is not None and (len(tasks) != 1 or args.n_trials != 1):
        raise SystemExit("--exact-seed requires exactly one task and --n-trials 1")
    if args.candidate_state_archive and not args.paired_action_noise:
        raise SystemExit("--candidate-state-archive requires --paired-action-noise")
    client = _Gr00tZMQClient(host="localhost", port=args.port)
    try:
        server_metadata = client.runtime_info()
    except Exception:
        server_metadata = {}
    require_protocol_attestation(server_metadata, source="GR00T runtime")
    expected_server_protocol = closed_loop_runtime_protocol()
    actual_server_protocol = server_metadata.get("protocol") or {}
    protocol_mismatches = {
        key: (actual_server_protocol.get(key), value)
        for key, value in expected_server_protocol.items()
        if actual_server_protocol.get(key) != value
    }
    if protocol_mismatches:
        raise SystemExit(f"GR00T server cross-model protocol mismatch: {protocol_mismatches}")
    if (server_metadata.get("model_adapter") or {}).get("model") != "gr00t":
        raise SystemExit("GR00T server adapter attestation is missing")
    server_metadata_sha256 = server_metadata.get("metadata_sha256")
    if not server_metadata_sha256:
        server_metadata_sha256 = hashlib.sha256(
            json.dumps(server_metadata, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    runtime_selector_metadata = server_metadata.get("runtime_selector") or {}
    if args.expect_runtime_selector and not runtime_selector_metadata.get("enabled"):
        raise SystemExit(f"server runtime selector is not enabled: {runtime_selector_metadata}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    capture_rows: list[dict] = [] if args.candidate_state_archive else []

    def persist_results() -> None:
        """Atomically checkpoint completed trials for crash-safe batch recovery."""
        tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row) + "\n")
        os.replace(tmp_path, out_path)

    def construct_env(task: str, seed: int):
        # enable_render=True is REQUIRED: False zero-fills camera obs and the
        # policy runs blind (D-040).
        # Seed before construction because RoboCasa samples layouts, object
        # instances, placements, and some camera perturbations while building
        # the environment. reset(seed=...) alone does not pair those choices.
        seed_environment_rng(seed)
        # D-041: concurrent EGL context creation on one device deadlocks the
        # NVIDIA EGL driver (10 stuck constructions, 44s CPU / 9min wall, no
        # sockets). Serialize env construction across processes with a repo-
        # local flock (repo path, NOT /tmp — /tmp is per-launcher tmpfs).
        import fcntl
        # Serialize only within one EGL device.  The previous single global
        # lock forced all GPU1-6 clients through GPU3 one at a time and made
        # environment construction the dominant wall-time cost.  Independent
        # A40 devices can safely construct contexts concurrently.
        egl_device = str(os.environ.get("MUJOCO_EGL_DEVICE_ID", "3"))
        lock_path = REPO_ROOT / "runs" / f".robocasa365_construct_gpu{egl_device}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        construct_t0 = time.perf_counter()
        with open(lock_path, "w") as _lf:
            fcntl.flock(_lf, fcntl.LOCK_EX)
            # Use RoboCasa's own GR00T wrapper. It is the environment used by
            # the official Isaac-GR00T run_eval.py protocol and already emits
            # the exact PandaOmron keys expected by the checkpoint.
            built_env = RoboCasaGymEnv(
                env_name=task,
                enable_render=True,
                split=args.split,
                seed=int(seed),
            )
            fcntl.flock(_lf, fcntl.LOCK_UN)
        return built_env, time.perf_counter() - construct_t0

    def trial_seed(task_index: int, trial: int) -> int:
        return (
            args.exact_seed if args.exact_seed is not None
            else exact_seeds[trial] if exact_seeds is not None
            else args.seed * 1000 + task_index * 10 + trial
        )

    for ti, task in enumerate(tasks):
        task_max_steps = args.max_steps or get_task_horizon(task)
        env = None
        shared_construct_seconds = None
        if not args.fresh_env_per_trial:
            env, shared_construct_seconds = construct_env(task, trial_seed(ti, 0))
        for trial in range(args.n_trials):
            seed = trial_seed(ti, trial)
            if args.fresh_env_per_trial:
                env, env_construct_seconds = construct_env(task, seed)
            else:
                env_construct_seconds = shared_construct_seconds if trial == 0 else 0.0
            episode_t0 = time.perf_counter()
            try:
                # Re-seed global RNGs because RoboCasa camera randomization uses
                # np.random in addition to env.rng during reset.
                seed_environment_rng(seed)
                obs, _ = env.reset(seed=seed)
                done = False
                steps = 0
                replan_index = 0
                success = False
                inference_seconds = 0.0
                env_step_seconds = 0.0
                runtime_selector_records = []
                while not done and steps < task_max_steps:
                    send_obs = {}
                    for k in SEND_VIDEO_KEYS + SEND_STATE_KEYS + SEND_LANG_KEYS:
                        if k not in obs:
                            continue
                        v = obs[k]
                        if k.startswith("video."):
                            # VideoToTensor expects (T, H, W, C) sequences
                            v = np.asarray(v)
                            if v.ndim == 3:
                                v = v[np.newaxis, ...]
                        elif k.startswith("state."):
                            # state values carry (T, D) batch dims like the LIBERO
                            # obs format
                            v = np.asarray(v, dtype=np.float32)
                            if v.ndim == 1:
                                v = v[np.newaxis, :]
                        elif isinstance(v, str):
                            v = [v]  # language values travel as lists
                        send_obs[k] = v
                    send_obs["eval_metadata"] = {
                        "task_name": task,
                        "task": task,
                        "seed": seed,
                        "trial": trial,
                        "replan_index": replan_index,
                        "model_id": "gr00t",
                    }
                    noise_seed = (
                        action_noise_seed(task, seed, replan_index)
                        if args.paired_action_noise else None
                    )
                    if args.candidate_state_archive:
                        if noise_seed is None:
                            raise RuntimeError("candidate-state capture requires paired noise")
                        capture_rows.append(
                            {
                                "image": np.asarray(obs["video.robot0_agentview_left"]),
                                "wrist_image": np.asarray(obs["video.robot0_eye_in_hand"]),
                                "right_image": np.asarray(obs["video.robot0_agentview_right"]),
                                "state": canonical_state(obs),
                                "prompt": str(obs["annotation.human.task_description"]),
                                "task": task,
                                "seed": seed,
                                "env_step": steps,
                                "replan": replan_index,
                                "noise": paired_action_noise_tensor(noise_seed),
                            }
                        )
                    infer_t0 = time.perf_counter()
                    action_chunk = client.get_action(send_obs, action_seed=noise_seed)
                    runtime_selector = action_chunk.get("runtime_selector") if isinstance(action_chunk, dict) else None
                    if args.expect_runtime_selector:
                        if not isinstance(runtime_selector, dict) or not runtime_selector.get("enabled"):
                            raise RuntimeError(
                                f"runtime selector response missing/disabled: {runtime_selector}"
                            )
                        if runtime_selector.get("task_name") != task:
                            raise RuntimeError(f"runtime selector task mismatch: {runtime_selector}")
                    if runtime_selector:
                        runtime_selector_records.append(dict(runtime_selector))
                    inference_seconds += time.perf_counter() - infer_t0
                    replan_index += 1
                    chunks = _normalize_action_chunks(action_chunk)
                    chunk_len = min(
                        args.n_action_steps,
                        task_max_steps - steps,
                        *(len(value) for value in chunks.values()),
                    )
                    for action_step in range(chunk_len):
                        action = {
                            f"action.{key}": np.atleast_1d(value[action_step])
                            for key, value in chunks.items()
                        }
                        step_t0 = time.perf_counter()
                        obs, reward, done, truncated, info = env.step(action)
                        env_step_seconds += time.perf_counter() - step_t0
                        steps += 1
                        success = bool(info.get("success", False))
                        if success or done or truncated:
                            done = True
                            break
            finally:
                if args.fresh_env_per_trial and env is not None:
                    env.close()
                    env = None
            if args.expect_runtime_selector and not runtime_selector_records:
                raise RuntimeError("runtime selector produced no records")
            selector_variants = {str(row.get("selected_variant")) for row in runtime_selector_records}
            selector_config_ids = {str(row.get("selected_config_id")) for row in runtime_selector_records}
            selector_hashes = {str(row.get("selector_sha256")) for row in runtime_selector_records}
            selector_rules = {str(row.get("selector_rule_name")) for row in runtime_selector_records}
            if (
                len(selector_variants) > 1
                or len(selector_config_ids) > 1
                or len(selector_hashes) > 1
                or len(selector_rules) > 1
            ):
                raise RuntimeError(f"runtime selector changed within episode: {runtime_selector_records}")
            selector_row = runtime_selector_records[0] if runtime_selector_records else {}
            results.append({
                "status": "complete",
                "task": task, "trial": trial, "seed": seed,
                "success": success, "steps": steps,
                "max_steps": task_max_steps,
                "n_action_steps": args.n_action_steps,
                "replans": replan_index,
                "env_construct_seconds": env_construct_seconds,
                "episode_wall_seconds": time.perf_counter() - episode_t0,
                "inference_seconds": inference_seconds,
                "env_step_seconds": env_step_seconds,
                "server_metadata_sha256": server_metadata_sha256,
                "runtime_selector_enabled": bool(selector_row.get("enabled", False)),
                "selected_variant": selector_row.get("selected_variant"),
                "selected_config_id": selector_row.get("selected_config_id"),
                "selector_sha256": selector_row.get("selector_sha256"),
                "selector_rule_name": selector_row.get("selector_rule_name"),
                "selector_task_name": selector_row.get("task_name"),
                "selector_model_id": selector_row.get("model_id"),
                "paired_action_noise": args.paired_action_noise,
                "action_noise_scheme": (
                    ACTION_NOISE_SCHEME if args.paired_action_noise else None
                ),
                "environment_seed_protocol": ENVIRONMENT_SEED_PROTOCOL,
                "native_action_horizon": 16,
                **closed_loop_row_protocol(),
            })
            print(f"[robocasa365-eval] {task} trial {trial}: success={success} "
                  f"steps={steps} ({time.time() - t0:.0f}s)", flush=True)
            persist_results()
        if env is not None:
            env.close()
    if args.candidate_state_archive:
        archive_path = Path(args.candidate_state_archive)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if not capture_rows:
            raise RuntimeError("candidate-state capture produced no rows")
        payload = {
            "images": np.stack([row["image"] for row in capture_rows]),
            "wrist_images": np.stack([row["wrist_image"] for row in capture_rows]),
            "right_images": np.stack([row["right_image"] for row in capture_rows]),
            "states": np.stack([row["state"] for row in capture_rows]),
            "prompts": np.asarray([row["prompt"] for row in capture_rows]),
            "task_ids": np.asarray([row["task"] for row in capture_rows]),
            "env_seeds": np.asarray([row["seed"] for row in capture_rows], dtype=np.int64),
            "env_steps": np.asarray([row["env_step"] for row in capture_rows], dtype=np.int64),
            "replan_indices": np.asarray([row["replan"] for row in capture_rows], dtype=np.int64),
            "action_noises": np.stack([row["noise"] for row in capture_rows]),
        }
        tmp_path = archive_path.with_name(f".{archive_path.name}.tmp.{os.getpid()}")
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, archive_path)
        print(f"[robocasa365-eval] candidate-state archive: {archive_path} "
              f"({len(capture_rows)} rows)", flush=True)
    n_ok = sum(1 for r in results if r["success"])
    print(f"[robocasa365-eval] done: {n_ok}/{len(results)} episodes "
          f"({n_ok / len(results):.1%}) -> {out_path}")


if __name__ == "__main__":
    main()
