#!/usr/bin/env python3
"""Create an immutable manifest for the four-task π0.5 QuantVLA sanity matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from openpi_client.paired_noise import PROTOCOL as PAIRED_NOISE_PROTOCOL


CONFIGS = (
    "fp16",
    "quantvla_w4a8_atmohb",
    "gdsq_vla_atmohb",
    "gdsq_vla",
)
TASKS = (
    "OpenCabinet",
    "OpenStandMixerHead",
    "PickPlaceDrawerToCounter",
    "CoffeeSetupMug",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--noise-mode", choices=("paired", "native"), required=True)
    parser.add_argument(
        "--server", action="append", required=True, help="CONFIG,GPU,PORT,RUNTIME_JSON"
    )
    parser.add_argument("--paper-root", required=True)
    return parser.parse_args()


def atomic_write(path: Path, payload: dict) -> None:
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


def main() -> None:
    args = parse_args()
    servers = []
    for value in args.server:
        config, gpu_text, port_text, runtime_text = value.split(",", 3)
        runtime_path = Path(runtime_text).resolve()
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime = payload.get("openpi_runtime") or {}
        if runtime.get("config_id") != config:
            raise ValueError(f"server config mismatch: {config} vs {runtime.get('config_id')}")
        servers.append(
            {
                "config": config,
                "gpu": int(gpu_text),
                "port": int(port_text),
                "runtime_path": str(runtime_path),
                "runtime_sha256": sha256_file(runtime_path),
                "server_metadata_sha256": canonical_hash(payload),
            }
        )
    if sorted(row["config"] for row in servers) != sorted(CONFIGS):
        raise ValueError("sanity manifest requires exactly one server for each of four configs")
    paper_root = Path(args.paper_root).resolve()
    artifact_paths = {
        "buffer": paper_root / "calibration/robocasa_real_observations_128.npz",
        "pack_manifest": paper_root / "packs/pi05_robocasa_block64_permute_w4a8_ls015/manifest.json",
        "full_w4a8_plan": paper_root / "plans/pi05_quantvla_paper_w4a8.plan.json",
        "full_w4a8_a8": paper_root / "a8/pi05_quantvla_paper_real32_p999_b32.npz",
        "full_w4a8_atm_ohb": paper_root / "atm_ohb/pi05_quantvla_paper_real128_scalar.json",
        "gdsq_plan": paper_root / "plans/pi05_gdsq_vla_final_paper.plan.json",
        "gdsq_a8": paper_root / "a8/pi05_gdsq_vla_final_real32_p999_b32.npz",
        "gdsq_atm_ohb": paper_root / "atm_ohb/pi05_gdsq_vla_final_real128_scalar.json",
    }
    artifacts = {}
    for name, path in artifact_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}
    payload = {
        "schema_version": 1,
        "immutable": True,
        "kind": "pi05_corrected_table1_paper_sanity30",
        "noise_mode": args.noise_mode,
        "noise_protocol": (
            PAIRED_NOISE_PROTOCOL if args.noise_mode == "paired" else "policy-native-rng"
        ),
        "tasks": list(TASKS),
        "trial_seeds": list(range(30)),
        "episodes_per_config": 120,
        "split": "pretrain",
        "fresh_environment_per_trial": True,
        "render_enabled": True,
        "official_task_horizon": True,
        "replan_steps": 5,
        "action_horizon": 50,
        "flow_steps": 10,
        "configs": list(CONFIGS),
        "servers": servers,
        "artifacts": artifacts,
    }
    output = Path(args.out).resolve()
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"immutable sanity manifest changed: {output}")
        print(f"sanity manifest verified unchanged: {output}")
        return
    atomic_write(output, payload)
    print(f"sanity manifest created: {output}")


if __name__ == "__main__":
    main()
