#!/usr/bin/env python3
"""Validate a generic pi0.5 week-1 server against its plan and A8 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PAIRED_NOISE = "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--a8", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-wrapped", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runtime_path = Path(args.runtime).resolve()
    plan_path = Path(args.plan).resolve()
    a8_path = Path(args.a8).resolve()
    a8_sidecar = Path(str(a8_path) + ".json")
    for path in (runtime_path, plan_path, a8_path, a8_sidecar):
        require(path.is_file(), f"missing runtime artifact: {path}")
    metadata = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime = metadata.get("openpi_runtime") or {}
    duquant = runtime.get("duquant") or {}
    protocol = runtime.get("protocol") or {}
    selector = runtime.get("runtime_selector") or {}
    atm = runtime.get("atm_ohb") or {}
    precision = runtime.get("model_dtype") or {}
    linear_dtypes = precision.get("linear_layers_by_weight_dtype") or {}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selected = {
        name: int(row.get("bits", 0) or 0)
        for name, row in (plan.get("layers") or {}).items()
        if not bool(row.get("skip", not int(row.get("bits", 0) or 0)))
        and int(row.get("bits", 0) or 0) > 0
    }
    require(len(selected) == args.expected_wrapped, "plan wrapped-layer count mismatch")
    require(set(selected.values()) <= {4, 6}, "unsupported week-1 weight bits")

    a8_metadata = json.loads(a8_sidecar.read_text(encoding="utf-8"))
    a8_meta = a8_metadata.get("metadata") or {}
    checks = {
        "config": runtime.get("config_id") == args.config,
        "duquant_enabled": duquant.get("enabled") is True,
        "wrapped": int(duquant.get("wrapped_layers", -1)) == args.expected_wrapped,
        "plan_path": Path(duquant.get("plan_path", "")).resolve() == plan_path,
        "plan_sha": duquant.get("plan_sha256") == sha256_file(plan_path),
        "a8_path": Path(duquant.get("act_scale_path", "")).resolve() == a8_path,
        "a8_sha": duquant.get("act_scale_sha256") == sha256_file(a8_path),
        "a8_ready": duquant.get("act_scales_ready") is True,
        "activation_bits": int(duquant.get("act_bits", -1)) == 8,
        "block_in": int(duquant.get("block_in", -1)) == 64,
        "block_out": int(duquant.get("block_out", -1)) == 64,
        "calib_batches": int(duquant.get("calib_batches", -1)) == 32,
        "denoising_steps": int(duquant.get("denoising_steps", -1)) == 4,
        "fake_quant_backend": duquant.get("execution_backend") == "fake_quant_fp16_gemm",
        "no_integer_gemm": duquant.get("integer_gemm") is False,
        "no_packed_residency": duquant.get("packed_low_bit_residency") is False,
        "model_fp16": precision.get("resolved") == "float16",
        "linear_weights_fp16": bool(linear_dtypes.get("float16"))
        and not any(count for dtype, count in linear_dtypes.items() if dtype != "float16"),
        "a8_sidecar_npz": a8_metadata.get("npz_sha256") == sha256_file(a8_path),
        "a8_sidecar_plan": a8_meta.get("plan_sha256") == sha256_file(plan_path),
        "a8_sidecar_wrapped": int(a8_meta.get("wrapped_layers", -1)) == args.expected_wrapped,
        "selector_disabled": selector.get("enabled") is not True,
        "atm_disabled": atm.get("enabled") is not True
        and atm.get("atm_enabled") is not True,
        "ohb_disabled": atm.get("ohb_enabled") is not True,
    }
    required_protocol = {
        "action_horizon": 50,
        "n_action_steps": 16,
        "replan_steps": 16,
        "flow_steps": 4,
        "split": "target",
        "fresh_environment_per_episode": True,
        "official_task_horizon": True,
        "render": True,
        "paired_noise": PAIRED_NOISE,
    }
    checks.update(
        {
            f"protocol_{key}": protocol.get(key) == value
            for key, value in required_protocol.items()
        }
    )
    failed = [name for name, valid in checks.items() if not valid]
    require(not failed, f"runtime audit failed: {failed}")
    payload = {
        "schema_version": 1,
        "kind": "pi05_week1_runtime_attestation",
        "valid": True,
        "config_id": args.config,
        "runtime_path": str(runtime_path),
        "runtime_file_sha256": sha256_file(runtime_path),
        "server_metadata_sha256": canonical_hash(metadata),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "a8_path": str(a8_path),
        "a8_sha256": sha256_file(a8_path),
        "a8_sidecar_sha256": sha256_file(a8_sidecar),
        "wrapped_layers": args.expected_wrapped,
        "weight_bits": sorted(set(selected.values())),
        "activation_bits": 8,
        "selector_enabled": False,
        "atm_enabled": False,
        "ohb_enabled": False,
        "protocol": required_protocol,
    }
    output = Path(args.out).resolve()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.exists():
        require(output.read_text(encoding="utf-8") == rendered, "runtime audit drift")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
