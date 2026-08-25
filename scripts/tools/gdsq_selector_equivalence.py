#!/usr/bin/env python3
"""Bitwise static-vs-v8-selector action equivalence on 256 frozen observations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR = REPO_ROOT / "runs/atmohb_dynamic_selector_v8/selector.json"
SELECTOR_SHA256 = "0f3178726c2b784898f18dfde248d9a9bae152da6ffcfdc299e8bdc02f0bd871"
SELECTOR_RULE = "v8_no_oracle_absolute_mechanism_gate_aligned_runtime_rule"
PI05_BUFFER = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
)
N_OBSERVATIONS = 256
GR00T_BUFFER_SEED = 20260823
ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def paired_seed(task: str, env_seed: int, replan_index: int) -> int:
    payload = f"quantvla-robocasa365-v1\0{task}\0{env_seed}\0{replan_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


class Gr00tClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 120_000):
        import msgpack
        import zmq

        self.msgpack = msgpack
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")

    @staticmethod
    def _encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            output = io.BytesIO()
            np.save(output, value, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return value

    @staticmethod
    def _decode(value: Any) -> Any:
        if isinstance(value, dict) and "__ndarray_class__" in value:
            return np.load(io.BytesIO(value["as_npy"]), allow_pickle=False)
        return value

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        import zmq

        self.socket.send(
            self.msgpack.packb(
                {"endpoint": endpoint, "data": data or {}}, default=self._encode
            )
        )
        try:
            response = self.msgpack.unpackb(
                self.socket.recv(), object_hook=self._decode
            )
        except zmq.Again as error:
            raise RuntimeError(f"GR00T server timeout on {endpoint}") from error
        if "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def runtime_info(self) -> dict[str, Any]:
        return self.call("get_runtime_info")

    def infer(self, observation: dict[str, Any], noise_seed: int) -> dict[str, Any]:
        return self.call(
            "get_action_seeded",
            {"observations": observation, "action_seed": int(noise_seed)},
        )


def gr00t_actions(response: dict[str, Any]) -> np.ndarray:
    chunks = []
    expected_dims = (3, 3, 1, 4, 1)
    for key, expected_dim in zip(ACTION_KEYS, expected_dims):
        value = np.asarray(response[key])
        while value.ndim > 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 1 and expected_dim == 1:
            value = value[:, None]
        if value.ndim != 2 or value.shape[1] != expected_dim:
            raise ValueError(f"invalid {key} shape: {value.shape}")
        chunks.append(value)
    return np.concatenate(chunks, axis=1)


def verify_selector_response(
    response: dict[str, Any], *, model: str, variant: str, config_id: str
) -> None:
    row = response.get("runtime_selector")
    if not isinstance(row, dict) or row.get("enabled") is not True:
        raise ValueError(f"{model}: selector attestation missing: {row}")
    expected = {
        "model_id": model,
        "selected_variant": variant,
        "selected_config_id": config_id,
        "selector_sha256": SELECTOR_SHA256,
        "selector_rule_name": SELECTOR_RULE,
        "atm_enabled": variant == "atm",
        "ohb_enabled": variant == "ohb",
    }
    mismatches = {
        key: (row.get(key), value) for key, value in expected.items() if row.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{model}: selector attestation mismatch: {mismatches}")


def verify_gr00t_metadata(static: dict[str, Any], runtime: dict[str, Any]) -> None:
    if static.get("config_id") != "cscka_final":
        raise ValueError(f"unexpected GR00T static config: {static.get('config_id')}")
    if runtime.get("config_id") != "cscka_final_runtime_selector":
        raise ValueError(f"unexpected GR00T selector config: {runtime.get('config_id')}")
    for row in (static, runtime):
        if row.get("denoising_steps") != 4 or row.get("wrapped_layers") != 100:
            raise ValueError(f"unaligned GR00T runtime: {row}")
    if static.get("plan_sha256") != runtime.get("plan_sha256"):
        raise ValueError("GR00T servers use different quantization plans")
    if static.get("act_scale_sha256") != runtime.get("act_scale_sha256"):
        raise ValueError("GR00T servers use different A8 scales")
    selector_meta = runtime.get("runtime_selector") or {}
    required = {
        "selector_sha256": SELECTOR_SHA256,
        "rule_name": SELECTOR_RULE,
        "model_id": "gr00t",
        "selection_scope": "model_level_absolute_mechanism_gate",
        "uses_task_metadata_for_selection": False,
    }
    mismatches = {
        key: (selector_meta.get(key), value)
        for key, value in required.items()
        if selector_meta.get(key) != value
    }
    if mismatches:
        raise ValueError(f"GR00T selector metadata mismatch: {mismatches}")


def run_gr00t(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "scripts/tools"))
    from gr00t_v2_common import fixed_calibration_buffer

    static_client = Gr00tClient(args.host, args.static_port)
    selector_client = Gr00tClient(args.host, args.selector_port)
    static_meta, selector_meta = static_client.runtime_info(), selector_client.runtime_info()
    verify_gr00t_metadata(static_meta, selector_meta)
    if args.ready_only:
        return {"ready": True, "model": "gr00t"}
    observations, noises, buffer_sha = fixed_calibration_buffer(
        GR00T_BUFFER_SEED, args.n, 16, 12, fmt="robocasa365"
    )
    action_hashes: list[str] = []
    mismatch_count = 0
    max_abs = 0.0
    started = time.perf_counter()
    for index, (observation, frozen_noise) in enumerate(zip(observations, noises)):
        task = "AddIceCubes"
        metadata = {
            "task_name": task,
            "seed": index,
            "replan_index": 0,
            "model_id": "gr00t",
        }
        request = dict(observation)
        request["eval_metadata"] = metadata
        noise_seed = paired_seed(task, index, 0)
        static_response = static_client.infer(request, noise_seed)
        selector_response = selector_client.infer(request, noise_seed)
        verify_selector_response(
            selector_response,
            model="gr00t",
            variant="baseline",
            config_id="cscka_final",
        )
        static_actions = gr00t_actions(static_response)
        selector_actions = gr00t_actions(selector_response)
        if not np.isfinite(static_actions).all() or not np.isfinite(selector_actions).all():
            raise RuntimeError(f"GR00T non-finite action tensor at observation {index}")
        delta = np.abs(static_actions.astype(np.float64) - selector_actions.astype(np.float64))
        max_abs = max(max_abs, float(delta.max(initial=0.0)))
        if not np.array_equal(static_actions, selector_actions):
            mismatch_count += 1
        action_hashes.append(array_hash(static_actions))
        # The generated noise participates in the frozen-buffer hash.  The
        # server uses the formal task/seed keyed noise above for both paths.
        if tuple(frozen_noise.shape) != (16, 12):
            raise AssertionError(f"unexpected GR00T frozen noise shape {frozen_noise.shape}")
        if (index + 1) % 32 == 0:
            print(f"[selector-equivalence] GR00T {index + 1}/{args.n}", flush=True)
    return {
        "schema_version": 1,
        "model": "gr00t",
        "complete": mismatch_count == 0 and len(action_hashes) == args.n,
        "observations": args.n,
        "frozen_observation_generator_seed": GR00T_BUFFER_SEED,
        "frozen_buffer_sha256": buffer_sha,
        "paired_noise_protocol": "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1",
        "denoising_steps": 4,
        "comparison": {
            "bitwise_equal_observations": args.n - mismatch_count,
            "mismatched_observations": mismatch_count,
            "max_abs": max_abs,
            "static_action_hash_sequence_sha256": canonical_hash(action_hashes),
        },
        "static_server": {"metadata": static_meta, "metadata_sha256": canonical_hash(static_meta)},
        "selector_server": {
            "metadata": selector_meta,
            "metadata_sha256": canonical_hash(selector_meta),
            "selected_variant": "baseline",
            "selected_config_id": "cscka_final",
        },
        "manifest_sha256": args.manifest_sha256,
        "elapsed_seconds": time.perf_counter() - started,
    }


def verify_pi05_metadata(static: dict[str, Any], runtime: dict[str, Any]) -> None:
    static_row = static.get("openpi_runtime") or {}
    runtime_row = runtime.get("openpi_runtime") or {}
    if static_row.get("config_id") != "gdsq_vla_ohb_only":
        raise ValueError(f"unexpected π0.5 static config: {static_row.get('config_id')}")
    if runtime_row.get("config_id") != "gdsq_vla_runtime_selector":
        raise ValueError(f"unexpected π0.5 selector config: {runtime_row.get('config_id')}")
    for row in (static_row, runtime_row):
        protocol = row.get("protocol") or {}
        if protocol.get("flow_steps") != 4:
            raise ValueError(f"π0.5 server is not 4-step aligned: {row}")
        if (row.get("duquant") or {}).get("wrapped_layers") != 80:
            raise ValueError(f"π0.5 wrapper count mismatch: {row}")
    if (static_row.get("duquant") or {}).get("plan_sha256") != (
        runtime_row.get("duquant") or {}
    ).get("plan_sha256"):
        raise ValueError("π0.5 servers use different quantization plans")
    selector_meta = runtime_row.get("runtime_selector") or {}
    required = {
        "selector_sha256": SELECTOR_SHA256,
        "rule_name": SELECTOR_RULE,
        "model_id": "pi05",
        "selection_scope": "model_level_absolute_mechanism_gate",
        "uses_task_metadata_for_selection": False,
    }
    mismatches = {
        key: (selector_meta.get(key), value)
        for key, value in required.items()
        if selector_meta.get(key) != value
    }
    if mismatches:
        raise ValueError(f"π0.5 selector metadata mismatch: {mismatches}")


def run_pi05(args: argparse.Namespace) -> dict[str, Any]:
    client_root = REPO_ROOT / "code/pi05/openpi/packages/openpi-client/src"
    sys.path.insert(0, str(client_root))
    from openpi_client.paired_noise import paired_action_noise
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    static_client = WebsocketClientPolicy(args.host, args.static_port)
    selector_client = WebsocketClientPolicy(args.host, args.selector_port)
    static_meta = static_client.get_server_metadata()
    selector_meta = selector_client.get_server_metadata()
    verify_pi05_metadata(static_meta, selector_meta)
    if args.ready_only:
        return {"ready": True, "model": "pi05"}
    buffer_path = Path(args.buffer).resolve()
    action_hashes: list[str] = []
    mismatch_count = 0
    max_abs = 0.0
    started = time.perf_counter()
    with np.load(buffer_path, allow_pickle=False) as archive:
        if args.n > len(archive["states"]):
            raise ValueError(f"requested {args.n} observations from {len(archive['states'])}")
        for index in range(args.n):
            task, seed = str(archive["task_ids"][index]), int(archive["env_seeds"][index])
            request = {
                "observation/image": np.asarray(archive["images"][index]),
                "observation/wrist_image": np.asarray(archive["wrist_images"][index]),
                "observation/right_image": np.asarray(archive["right_images"][index]),
                "observation/state": np.asarray(archive["states"][index], dtype=np.float32),
                "prompt": str(archive["prompts"][index]),
                "__openpi_eval_metadata__": {
                    "task_name": task,
                    "seed": seed,
                    "replan_index": 0,
                    "model_id": "pi05",
                },
            }
            noise = paired_action_noise(task, seed, 0)
            static_response = static_client.infer(request, noise=noise)
            selector_response = selector_client.infer(request, noise=noise)
            verify_selector_response(
                selector_response,
                model="pi05",
                variant="ohb",
                config_id="gdsq_vla_ohb_only",
            )
            static_actions = np.asarray(static_response["actions"])
            selector_actions = np.asarray(selector_response["actions"])
            if static_actions.shape != (50, 12) or selector_actions.shape != (50, 12):
                raise RuntimeError(
                    f"π0.5 invalid action shapes {static_actions.shape}/{selector_actions.shape}"
                )
            if not np.isfinite(static_actions).all() or not np.isfinite(selector_actions).all():
                raise RuntimeError(f"π0.5 non-finite action tensor at observation {index}")
            delta = np.abs(static_actions.astype(np.float64) - selector_actions.astype(np.float64))
            max_abs = max(max_abs, float(delta.max(initial=0.0)))
            if not np.array_equal(static_actions, selector_actions):
                mismatch_count += 1
            action_hashes.append(array_hash(static_actions))
            if (index + 1) % 32 == 0:
                print(f"[selector-equivalence] π0.5 {index + 1}/{args.n}", flush=True)
    return {
        "schema_version": 1,
        "model": "pi05",
        "complete": mismatch_count == 0 and len(action_hashes) == args.n,
        "observations": args.n,
        "frozen_buffer": {"path": str(buffer_path), "sha256": sha256_file(buffer_path)},
        "paired_noise_protocol": "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1",
        "denoising_steps": 4,
        "comparison": {
            "bitwise_equal_observations": args.n - mismatch_count,
            "mismatched_observations": mismatch_count,
            "max_abs": max_abs,
            "static_action_hash_sequence_sha256": canonical_hash(action_hashes),
        },
        "static_server": {"metadata": static_meta, "metadata_sha256": canonical_hash(static_meta)},
        "selector_server": {
            "metadata": selector_meta,
            "metadata_sha256": canonical_hash(selector_meta),
            "selected_variant": "ohb",
            "selected_config_id": "gdsq_vla_ohb_only",
        },
        "manifest_sha256": args.manifest_sha256,
        "elapsed_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("gr00t", "pi05"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--static-port", required=True, type=int)
    parser.add_argument("--selector-port", required=True, type=int)
    parser.add_argument("--n", type=int, default=N_OBSERVATIONS)
    parser.add_argument("--buffer", default=str(PI05_BUFFER))
    parser.add_argument("--manifest-sha256", default=None)
    parser.add_argument("--ready-only", action="store_true")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256_file(SELECTOR) != SELECTOR_SHA256:
        raise ValueError("frozen v8 selector SHA mismatch")
    if not args.ready_only and args.n != N_OBSERVATIONS:
        raise ValueError(f"formal equivalence requires exactly {N_OBSERVATIONS} observations")
    report = run_gr00t(args) if args.model == "gr00t" else run_pi05(args)
    if args.ready_only:
        print(json.dumps(report, sort_keys=True))
        return
    if not args.out:
        raise ValueError("--out is required unless --ready-only")
    out = Path(args.out).resolve()
    atomic_json(out, report)
    print(
        json.dumps(
            {
                "model": args.model,
                "complete": report["complete"],
                "comparison": report["comparison"],
                "out": str(out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
