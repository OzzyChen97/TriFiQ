#!/usr/bin/env python3
"""Crash-safe official RoboCasa365 evaluator for the π0.5 four-config matrix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import robocasa  # noqa: F401 -- must precede repository path additions
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

import numpy as np
from openpi_client import image_tools
from openpi_client.paired_noise import PROTOCOL as ACTION_NOISE_PROTOCOL
from openpi_client.paired_noise import paired_action_noise
from quantvla_cross_model_protocol import (  # noqa: E402
    closed_loop_row_protocol,
    closed_loop_runtime_protocol,
    require_protocol_attestation,
)
from quantvla_dynamic_a8_protocol import (  # noqa: E402
    require_protocol_attestation as require_dynamic_a8_protocol_attestation,
    validate_runtime as validate_dynamic_a8_runtime,
)
from openpi_client.websocket_client_policy import WebsocketClientPolicy


FORMAL_SPLIT = "target"
FORMAL_N_ACTION_STEPS = 16
FORMAL_FLOW_STEPS = 4
ACTION_NOISE_REQUEST_KEY = "__openpi_action_noise__"
EVAL_METADATA_REQUEST_KEY = "__openpi_eval_metadata__"
ENVIRONMENT_SEED_PROTOCOL = "python-random/numpy-global/robocasa-constructor-and-reset-v1"


def seed_environment_rng(seed: int) -> None:
    """Seed RNGs used before and during RoboCasa environment construction."""
    random.seed(int(seed))
    np.random.seed(int(seed))


def parse_seed_spec(value: str) -> list[int]:
    seeds = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"descending seed range: {token}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("trial seeds must be unique non-negative integers")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument(
        "--task-set",
        choices=["atomic_seen", "composite_seen", "composite_unseen"],
        required=True,
    )
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument("--trial-seeds", default="0-49")
    parser.add_argument("--split", default=FORMAL_SPLIT)
    parser.add_argument("--replan-steps", type=int, default=FORMAL_N_ACTION_STEPS)
    parser.add_argument(
        "--action-noise-mode",
        choices=("paired", "native"),
        default="paired",
        help="paired is the formal deterministic protocol; native is a separate robustness arm",
    )
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--resume-dir",
        default=None,
        help=(
            "Optional config result directory. Completed rows from every JSONL for the "
            "same task set are used as a read-only global resume index."
        ),
    )
    parser.add_argument("--egl-device", type=int, required=True)
    parser.add_argument("--expected-server-metadata-sha256", default=None)
    parser.add_argument("--expect-runtime-selector", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args()


def canonical_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_committed(
    path: Path,
    config_id: str,
    task_set: str,
    expected_server_metadata_sha256: str | None = None,
) -> dict[tuple[str, int], dict]:
    if not path.is_file():
        return {}
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL row {line_number} in {path}") from error
        if row.get("status") != "complete":
            raise ValueError(f"non-complete row committed at {line_number}")
        if row.get("config") != config_id or row.get("task_set") != task_set:
            raise ValueError(f"foreign config/task-set row at {line_number}")
        if (
            expected_server_metadata_sha256 is not None
            and row.get("server_metadata_sha256") != expected_server_metadata_sha256
        ):
            raise ValueError(
                f"server metadata hash mismatch at {path}:{line_number}: "
                f"{row.get('server_metadata_sha256')!r} != "
                f"{expected_server_metadata_sha256!r}"
            )
        key = (str(row["task"]), int(row["seed"]))
        if key in rows:
            raise ValueError(f"duplicate committed key {key}")
        rows[key] = row
    return rows


def load_global_resume(
    *,
    output: Path,
    resume_dir: Path | None,
    config_id: str,
    task_set: str,
    expected_server_metadata_sha256: str,
) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], dict]]:
    """Load the writer-local journal and an immutable cross-file resume index.

    Formal schedules keep writers disjoint.  This index permits a later schedule
    amendment to repartition tasks without copying old rows or committing a key
    twice.  Duplicate historical keys are rejected instead of silently choosing
    one row.
    """
    output = output.resolve()
    local = load_committed(
        output, config_id, task_set, expected_server_metadata_sha256
    )
    global_rows = dict(local)
    if resume_dir is None:
        return local, global_rows
    root = resume_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"resume directory does not exist: {root}")
    for candidate in sorted(root.glob(f"{task_set}_*.jsonl")):
        if candidate.resolve() == output:
            continue
        rows = load_committed(
            candidate, config_id, task_set, expected_server_metadata_sha256
        )
        overlap = set(global_rows) & set(rows)
        if overlap:
            sample = sorted(overlap)[:3]
            raise ValueError(
                f"duplicate committed keys across resume files; "
                f"file={candidate}, count={len(overlap)}, sample={sample}"
            )
        global_rows.update(rows)
    return local, global_rows


def persist(path: Path, rows: dict[tuple[str, int], dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(rows):
            handle.write(json.dumps(rows[key], sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def construct_env(task: str, split: str, egl_device: int, seed: int):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)
    lock_path = REPO_ROOT / "runs" / f".robocasa365_construct_gpu{egl_device}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Pair layout/style, object instance, placement, and camera sampling
        # across configs. reset(seed=...) alone runs after construction and is
        # insufficient because RoboCasa consumes RNGs while building the env.
        seed_environment_rng(seed)
        env = RoboCasaGymEnv(
            env_name=task,
            enable_render=True,
            split=split,
            seed=int(seed),
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
    return env, time.perf_counter() - started


def image_observation(obs: dict, key: str, resize: int) -> np.ndarray:
    value = np.asarray(obs[key])
    while value.ndim > 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"unexpected {key} image shape: {value.shape}")
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(value, resize, resize))


def state_observation(obs: dict) -> np.ndarray:
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
    state = np.concatenate(values).astype(np.float32, copy=False)
    if state.shape != (16,):
        raise ValueError(f"unexpected RoboCasa π0.5 state shape: {state.shape}")
    return state


def language_observation(obs: dict) -> str:
    value = obs["annotation.human.task_description"]
    if isinstance(value, (list, tuple, np.ndarray)):
        value = np.asarray(value).reshape(-1)[0]
    return str(value)


def run_trial(
    *,
    client: WebsocketClientPolicy,
    config_id: str,
    task_set: str,
    task: str,
    seed: int,
    split: str,
    replan_steps: int,
    resize: int,
    egl_device: int,
    max_steps_override: int,
    server_metadata_sha256: str,
    action_noise_mode: str,
    flow_steps: int,
    expect_runtime_selector: bool,
) -> dict:
    env = None
    episode_started = time.perf_counter()
    try:
        env, construct_seconds = construct_env(task, split, egl_device, seed)
        # Camera randomization still uses NumPy's global RNG on reset.
        seed_environment_rng(seed)
        obs, _ = env.reset(seed=seed)
        horizon = max_steps_override or get_task_horizon(task)
        success = False
        done = False
        termination = "official_horizon"
        steps = 0
        replans = 0
        inference_seconds = 0.0
        server_infer_ms = []
        policy_infer_ms = []
        runtime_selector_records = []
        env_step_seconds = 0.0
        while steps < horizon and not done:
            request = {
                "observation/image": image_observation(
                    obs, "video.robot0_agentview_left", resize
                ),
                "observation/wrist_image": image_observation(
                    obs, "video.robot0_eye_in_hand", resize
                ),
                "observation/right_image": image_observation(
                    obs, "video.robot0_agentview_right", resize
                ),
                "observation/state": state_observation(obs),
                "prompt": language_observation(obs),
                EVAL_METADATA_REQUEST_KEY: {
                    "task_name": task,
                    "task_set": task_set,
                    "seed": int(seed),
                    "replan_index": int(replans),
                    "config_id": config_id,
                },
            }
            noise = (
                paired_action_noise(task, seed, replans)
                if action_noise_mode == "paired"
                else None
            )
            infer_started = time.perf_counter()
            response = client.infer(request, noise=noise)
            inference_seconds += time.perf_counter() - infer_started
            runtime_selector = response.get("runtime_selector")
            if expect_runtime_selector:
                if not isinstance(runtime_selector, dict) or not runtime_selector.get("enabled"):
                    raise RuntimeError(f"runtime selector response missing/disabled: {runtime_selector}")
                if runtime_selector.get("task_name") != task:
                    raise RuntimeError(f"runtime selector task mismatch: {runtime_selector}")
            if runtime_selector:
                runtime_selector_records.append(dict(runtime_selector))
            actions = np.asarray(response["actions"])
            if actions.shape != (50, 12) or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid policy actions: shape={actions.shape}")
            if response.get("server_timing", {}).get("infer_ms") is not None:
                server_infer_ms.append(float(response["server_timing"]["infer_ms"]))
            if response.get("policy_timing", {}).get("infer_ms") is not None:
                policy_infer_ms.append(float(response["policy_timing"]["infer_ms"]))
            replans += 1
            execute = min(replan_steps, horizon - steps)
            for index in range(execute):
                action = convert_action(actions[index])
                step_started = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                env_step_seconds += time.perf_counter() - step_started
                steps += 1
                success = bool(info.get("success", False))
                # Match the GR00T N1.5 evaluator exactly: success, terminated,
                # or truncated ends the episode; otherwise the official task
                # horizon is the outer bound.
                if success or terminated or truncated:
                    done = True
                    termination = (
                        "success" if success else "terminated" if terminated else "truncated"
                    )
                    break
        if expect_runtime_selector and not runtime_selector_records:
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
        return {
            "status": "complete",
            "config": config_id,
            "task_set": task_set,
            "task": task,
            "seed": seed,
            "split": split,
            "success": success,
            "steps": steps,
            "max_steps": horizon,
            "replans": replans,
            "replan_steps": replan_steps,
            "n_action_steps": replan_steps,
            "flow_steps": flow_steps,
            "native_action_horizon": 50,
            "paired_action_noise": action_noise_mode == "paired",
            "action_noise_protocol": (
                ACTION_NOISE_PROTOCOL if action_noise_mode == "paired" else "policy-native-rng"
            ),
            "environment_seed_protocol": ENVIRONMENT_SEED_PROTOCOL,
            "fresh_environment": True,
            **closed_loop_row_protocol(),
            "render_enabled": True,
            "termination": termination,
            "egl_device": egl_device,
            "server_metadata_sha256": server_metadata_sha256,
            "runtime_selector_enabled": bool(selector_row.get("enabled", False)),
            "selected_variant": selector_row.get("selected_variant"),
            "selected_config_id": selector_row.get("selected_config_id"),
            "selector_sha256": selector_row.get("selector_sha256"),
            "selector_rule_name": selector_row.get("selector_rule_name"),
            "selector_task_name": selector_row.get("task_name"),
            "selector_model_id": selector_row.get("model_id"),
            "env_construct_seconds": construct_seconds,
            "episode_wall_seconds": time.perf_counter() - episode_started,
            "inference_seconds": inference_seconds,
            "env_step_seconds": env_step_seconds,
            "server_infer_ms_mean": float(np.mean(server_infer_ms)) if server_infer_ms else None,
            "server_infer_ms_p95": float(np.percentile(server_infer_ms, 95)) if server_infer_ms else None,
            "policy_infer_ms_mean": float(np.mean(policy_infer_ms)) if policy_infer_ms else None,
        }
    finally:
        if env is not None:
            env.close()


def main() -> None:
    args = parse_args()
    if args.split != FORMAL_SPLIT:
        raise SystemExit(f"formal π0.5 evaluation requires --split {FORMAL_SPLIT}")
    if args.replan_steps != FORMAL_N_ACTION_STEPS:
        raise SystemExit(
            f"formal π0.5 evaluation requires --replan-steps {FORMAL_N_ACTION_STEPS}"
        )
    if args.action_noise_mode != "paired":
        raise SystemExit("cross-model formal evaluation requires paired action noise")
    if args.action_noise_mode != "paired":
        raise SystemExit("formal π0.5 evaluation requires GR00T-aligned paired action noise")
    seeds = parse_seed_spec(args.trial_seeds)
    registered = list(TASK_SET_REGISTRY[args.task_set])
    if args.tasks:
        requested = [value.strip() for value in args.tasks.split(",") if value.strip()]
        unknown = set(requested) - set(registered)
        if unknown:
            raise SystemExit(f"tasks are not in {args.task_set}: {sorted(unknown)}")
        tasks = requested
    else:
        tasks = registered
    if not (0 <= args.task_shard_index < args.task_shard_count):
        raise SystemExit("invalid task shard")
    tasks = [
        task for index, task in enumerate(tasks) if index % args.task_shard_count == args.task_shard_index
    ]
    if not tasks:
        raise SystemExit("task shard is empty")

    client = WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    metadata_hash = canonical_hash(server_metadata)
    server_runtime = server_metadata.get("openpi_runtime") or {}
    require_protocol_attestation(server_runtime, source="pi0.5 runtime")
    server_quant_contract = server_runtime.get("cross_model_quantization_contract") or {}
    if server_quant_contract.get("static_activation_scales") is False:
        require_dynamic_a8_protocol_attestation(server_runtime, source="pi0.5 runtime")
        validate_dynamic_a8_runtime(
            server_quant_contract, source="pi0.5 runtime contract"
        )
    if (server_runtime.get("model_adapter") or {}).get("model") != "pi05":
        raise SystemExit("pi0.5 server adapter attestation is missing")
    server_protocol = server_runtime.get("protocol") or {}
    expected_protocol = closed_loop_runtime_protocol()
    protocol_mismatches = {
        key: (server_protocol.get(key), value)
        for key, value in expected_protocol.items()
        if server_protocol.get(key) != value
    }
    if protocol_mismatches:
        raise SystemExit(f"server cross-model protocol mismatch: {protocol_mismatches}")
    runtime_selector_metadata = ((server_metadata.get("openpi_runtime") or {}).get("runtime_selector") or {})
    if args.expect_runtime_selector and not runtime_selector_metadata.get("enabled"):
        raise SystemExit(f"server runtime selector is not enabled: {runtime_selector_metadata}")
    if (
        args.expected_server_metadata_sha256
        and metadata_hash != args.expected_server_metadata_sha256
    ):
        raise SystemExit(
            f"server metadata hash mismatch: {metadata_hash} != "
            f"{args.expected_server_metadata_sha256}"
        )
    output = Path(args.out).resolve()
    committed, resume_index = load_global_resume(
        output=output,
        resume_dir=Path(args.resume_dir) if args.resume_dir else None,
        config_id=args.config_id,
        task_set=args.task_set,
        expected_server_metadata_sha256=metadata_hash,
    )
    for task in tasks:
        for seed in seeds:
            key = (task, seed)
            if key in resume_index:
                print(f"[pi05 eval] reuse {args.config_id}/{task}/{seed}", flush=True)
                continue
            row = run_trial(
                client=client,
                config_id=args.config_id,
                task_set=args.task_set,
                task=task,
                seed=seed,
                split=args.split,
                replan_steps=args.replan_steps,
                resize=args.resize,
                egl_device=args.egl_device,
                max_steps_override=args.max_steps,
                server_metadata_sha256=metadata_hash,
                action_noise_mode=args.action_noise_mode,
                flow_steps=FORMAL_FLOW_STEPS,
                expect_runtime_selector=args.expect_runtime_selector,
            )
            committed[key] = row
            resume_index[key] = row
            persist(output, committed)
            print(
                f"[pi05 eval] {args.config_id}/{task}/{seed}: "
                f"success={row['success']} steps={row['steps']} "
                f"wall={row['episode_wall_seconds']:.1f}s",
                flush=True,
            )
    expected_keys = {(task, seed) for task in tasks for seed in seeds}
    missing = expected_keys - set(resume_index)
    if missing:
        raise RuntimeError(f"evaluation shard incomplete: {len(missing)} missing rows")
    successes = sum(bool(resume_index[key]["success"]) for key in expected_keys)
    print(
        f"[pi05 eval] complete {args.config_id}/{args.task_set} shard "
        f"{args.task_shard_index}: {successes}/{len(expected_keys)} -> {output}"
    )


if __name__ == "__main__":
    main()
