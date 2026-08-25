#!/usr/bin/env python3
"""Deterministic end-to-end websocket smoke for the four formal π0.5 servers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from openpi_client.paired_noise import PROTOCOL as NOISE_PROTOCOL
from openpi_client.paired_noise import paired_action_noise
from openpi_client.websocket_client_policy import WebsocketClientPolicy


REPO_ROOT = Path(__file__).resolve().parents[2]
ALIGNED_ROOT = REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned"
DEFAULT_BUFFER = ALIGNED_ROOT / "calibration/pi05_robocasa365_seed0_n256.npz"
DEFAULT_GDSQ_PLAN = ALIGNED_ROOT / "plans/pi05_gdsq_cscka_16to1_d4.final_plan.json"
EXPECTED_CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--servers",
        default=(
            "fp16=18101,quantvla_w4a8_atmohb=18102,"
            "gdsq_vla_atmohb=18103,gdsq_vla=18104"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--buffer", default=str(DEFAULT_BUFFER))
    parser.add_argument("--gdsq-plan", default=str(DEFAULT_GDSQ_PLAN))
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def parse_servers(value: str) -> dict[str, int]:
    servers = {}
    for item in value.split(","):
        config_id, port_text = item.split("=", 1)
        if config_id in servers:
            raise ValueError(f"duplicate config id: {config_id}")
        servers[config_id] = int(port_text)
    expected = {"fp16", "quantvla_w4a8_atmohb", "gdsq_vla_atmohb", "gdsq_vla"}
    if set(servers) != expected:
        raise ValueError(f"server config ids {set(servers)} != {expected}")
    return servers


def canonical_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_request(path: Path) -> tuple[dict, str, int]:
    with np.load(path, allow_pickle=False) as archive:
        request = {
            "observation/image": np.asarray(archive["images"][0]),
            "observation/wrist_image": np.asarray(archive["wrist_images"][0]),
            "observation/right_image": np.asarray(archive["right_images"][0]),
            "observation/state": np.asarray(archive["states"][0], dtype=np.float32),
            "prompt": str(archive["prompts"][0]),
        }
        return request, str(archive["task_ids"][0]), int(archive["env_seeds"][0])


def validate_metadata(config_id: str, metadata: dict, gdsq_wrapped: int) -> None:
    runtime = metadata.get("openpi_runtime") or {}
    if runtime.get("config_id") != config_id:
        raise ValueError(f"{config_id}: server config id mismatch")
    protocol = runtime.get("protocol") or {}
    required_protocol = {
        "action_horizon": 50,
        "n_action_steps": 16,
        "replan_steps": 16,
        "flow_steps": 4,
        "split": "target",
        "fresh_environment_per_episode": True,
        "official_task_horizon": True,
        "render": True,
        "paired_noise": NOISE_PROTOCOL,
    }
    mismatches = {
        key: (protocol.get(key), value)
        for key, value in required_protocol.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{config_id}: GR00T-aligned protocol mismatch: {mismatches}")
    if runtime.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"{config_id}: checkpoint hash mismatch "
            f"{runtime.get('checkpoint_sha256')!r} != {EXPECTED_CHECKPOINT_SHA256!r}"
        )
    precision = runtime.get("model_dtype") or {}
    if precision.get("resolved") != "float16":
        raise ValueError(f"{config_id}: server is not strict FP16")
    expected = {
        "fp16": (0, False),
        "quantvla_w4a8_atmohb": (180, True),
        "gdsq_vla_atmohb": (gdsq_wrapped, True),
        "gdsq_vla": (gdsq_wrapped, False),
    }
    wrapped, atm_enabled = expected[config_id]
    if int((runtime.get("duquant") or {}).get("wrapped_layers", 0)) != wrapped:
        raise ValueError(f"{config_id}: wrapper count mismatch")
    if bool((runtime.get("atm_ohb") or {}).get("enabled")) != atm_enabled:
        raise ValueError(f"{config_id}: ATM/OHB enablement mismatch")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parse_args()
    servers = parse_servers(args.servers)
    plan_path = Path(args.gdsq_plan).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("meta", {}).get("kind") != "gdsq_vla_pi05_faithful_final_frozen":
        raise ValueError("smoke requires the frozen faithful-final GDSQ plan")
    gdsq_wrapped = sum(
        not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
        for row in plan.get("layers", {}).values()
    )
    if len(plan.get("layers", {})) != 180 or gdsq_wrapped <= 0:
        raise ValueError("frozen GDSQ plan has an invalid layer inventory")
    request, task, seed = load_request(Path(args.buffer).resolve())
    noise = paired_action_noise(task, seed, 0)
    rows = {}
    for config_id, port in servers.items():
        client = WebsocketClientPolicy(args.host, port)
        metadata = client.get_server_metadata()
        validate_metadata(config_id, metadata, gdsq_wrapped)
        started = time.perf_counter()
        first = np.asarray(client.infer(request, noise=noise)["actions"])
        second = np.asarray(client.infer(request, noise=noise)["actions"])
        elapsed = time.perf_counter() - started
        if first.shape != (50, 12) or not np.isfinite(first).all():
            raise RuntimeError(f"{config_id}: invalid first action tensor {first.shape}")
        if not np.array_equal(first, second):
            raise RuntimeError(f"{config_id}: repeated paired inference is not bitwise deterministic")
        rows[config_id] = {
            "port": port,
            "metadata_sha256": canonical_hash(metadata),
            "actions_sha256": array_hash(first),
            "actions_dtype": str(first.dtype),
            "actions_shape": list(first.shape),
            "finite": True,
            "bitwise_repeat_equal": True,
            "two_request_wall_seconds": elapsed,
            "runtime": metadata["openpi_runtime"],
        }
        print(f"[pi05 smoke] {config_id}: OK ({elapsed:.3f}s)", flush=True)
    payload = {
        "schema_version": 1,
        # This flag is written only after all four servers have passed metadata,
        # finite-output, shape, and bitwise-repeat checks above.  The final
        # alignment audit deliberately fails closed when it is absent.
        "complete": True,
        "task": task,
        "seed": seed,
        "replan_index": 0,
        "noise_protocol": NOISE_PROTOCOL,
        "noise_sha256": array_hash(noise),
        "servers": rows,
    }
    atomic_json(Path(args.out).resolve(), payload)
    print(f"[pi05 smoke] report: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
