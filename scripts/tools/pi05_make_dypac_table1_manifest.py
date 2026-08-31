#!/usr/bin/env python3
"""Freeze or verify the standalone DyPAC-VLA pi0.5 Table-1 execution manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import robocasa  # noqa: F401 -- register RoboCasa task sets before inspection
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from quantvla_cross_model_protocol import (  # noqa: E402
    closed_loop_runtime_protocol,
    require_protocol_attestation as require_cross_model_attestation,
)
from quantvla_dynamic_a8_protocol import (  # noqa: E402
    require_protocol_attestation as require_dynamic_a8_attestation,
    validate_runtime as validate_dynamic_a8_runtime,
)
from quantvla_full_context import (  # noqa: E402
    PROTOCOL,
    protocol_attestation,
    require_protocol_attestation as require_full_context_attestation,
)


CONFIG_ID = "full_context_w4a8_dynamic_profile"
FROZEN_PLAN_SHA256 = "e502f7cd7d126517c000f8b5b8e7c6e5537b31226910a2dd6834caaebf736c83"
CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def artifact(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = expected_sha256 or sha256_file(resolved)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest}


def directory_artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    records = []
    for child in sorted(value for value in resolved.rglob("*") if value.is_file()):
        records.append(
            {
                "path": str(child.relative_to(resolved)),
                "bytes": child.stat().st_size,
                "sha256": sha256_file(child),
            }
        )
    if not records:
        raise ValueError(f"empty artifact directory: {resolved}")
    return {
        "path": str(resolved),
        "files": len(records),
        "bytes": sum(row["bytes"] for row in records),
        "tree_sha256": canonical_hash(records),
    }


def git_state() -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
        ).strip()
    )
    return {"head": head, "dirty": dirty}


def parse_server(spec: str) -> dict[str, Any]:
    instance, gpu_text, port_text, runtime_text = spec.split(",", 3)
    runtime_path = Path(runtime_text).expanduser().resolve()
    metadata = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime = metadata.get("openpi_runtime") or {}
    if runtime.get("config_id") != CONFIG_ID:
        raise ValueError(f"{instance}: unexpected config {runtime.get('config_id')!r}")
    require_cross_model_attestation(runtime, source=f"{instance} runtime")
    require_dynamic_a8_attestation(runtime, source=f"{instance} runtime")
    require_full_context_attestation(runtime, source=f"{instance} runtime")

    expected_protocol = closed_loop_runtime_protocol()
    expected_protocol["flow_steps"] = 4
    actual_protocol = runtime.get("protocol") or {}
    mismatches = {
        key: (actual_protocol.get(key), value)
        for key, value in expected_protocol.items()
        if actual_protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{instance}: Table-1 protocol mismatch: {mismatches}")

    contract = runtime.get("cross_model_quantization_contract") or {}
    validate_dynamic_a8_runtime(contract, source=f"{instance} quantization contract")
    expected_contract = {
        "weight_bits": 4,
        "activation_bits": 8,
        "static_activation_scales": False,
        "packed_low_bit_residency": True,
        "fp_weight_sized_buffers": 0,
        "n_action_steps": 16,
        "replan_steps": 16,
        "denoising_steps": 4,
    }
    contract_mismatches = {
        key: (contract.get(key), value)
        for key, value in expected_contract.items()
        if contract.get(key) != value
    }
    if contract_mismatches:
        raise ValueError(f"{instance}: quantization contract mismatch: {contract_mismatches}")
    if (runtime.get("runtime_selector") or {}).get("enabled"):
        raise ValueError(f"{instance}: runtime selector must be disabled")
    if (runtime.get("errorfold") or {}).get("enabled"):
        raise ValueError(f"{instance}: runtime correction must be disabled")
    duquant = runtime.get("duquant") or {}
    if int(duquant.get("wrapped_layers", -1)) != 121:
        raise ValueError(f"{instance}: expected 121 W4 wrappers")
    if duquant.get("plan_sha256") != FROZEN_PLAN_SHA256:
        raise ValueError(f"{instance}: frozen plan hash mismatch")

    return {
        "instance": instance,
        "gpu": int(gpu_text),
        "port": int(port_text),
        "runtime_path": str(runtime_path),
        "runtime_file_sha256": sha256_file(runtime_path),
        "server_metadata_sha256": canonical_hash(metadata),
        "runtime": runtime,
    }


def validate_frozen_artifacts(plan_path: Path, hessian_path: Path, pack_dir: Path) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if sha256_file(plan_path) != FROZEN_PLAN_SHA256:
        raise ValueError("DyPAC-VLA pi0.5 frozen plan hash mismatch")
    require_full_context_attestation(plan.get("meta") or {}, source=str(plan_path))
    meta = plan.get("meta") or {}
    expected_meta = {
        "frozen": True,
        "model_adapter": "pi05",
        "activation_mode": "dynamic_a8",
        "runtime_selector": False,
        "runtime_correction": False,
        "flow_steps": 4,
    }
    mismatches = {
        key: (meta.get(key), value)
        for key, value in expected_meta.items()
        if meta.get(key) != value
    }
    selected = [
        name
        for name, row in (plan.get("layers") or {}).items()
        if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) > 0
    ]
    if mismatches or len(selected) != 121 or int(plan.get("total_bytes", -1)) != 1_634_828_288:
        raise ValueError(
            f"invalid frozen plan: meta={mismatches}, W4={len(selected)}, "
            f"bytes={plan.get('total_bytes')}"
        )

    sidecar_path = Path(str(hessian_path) + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    hessian_names = [str(value) for value in sidecar.get("layer_names") or []]
    expected_hessian = {
        "npz_sha256": sha256_file(hessian_path),
        "deployment_plan_sha256": FROZEN_PLAN_SHA256,
        "requantized": False,
        "group_size": 64,
    }
    hessian_mismatches = {
        key: (sidecar.get(key), value)
        for key, value in expected_hessian.items()
        if sidecar.get(key) != value
    }
    inventory_match = (
        len(hessian_names) == len(set(hessian_names))
        and len(selected) == len(set(selected))
        and set(hessian_names) == set(selected)
    )
    if hessian_mismatches or not inventory_match:
        raise ValueError(
            f"invalid Hessian subset: metadata={hessian_mismatches}, "
            f"inventory_match={inventory_match}"
        )

    pack_manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    pack_expected = {
        "kind": "hessian_w4_identity_pack",
        "model_adapter": "pi05",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "permutation": False,
        "row_rotation": "identity",
    }
    pack_mismatches = {
        key: (pack_manifest.get(key), value)
        for key, value in pack_expected.items()
        if pack_manifest.get(key) != value
    }
    if pack_mismatches:
        raise ValueError(f"invalid identity pack: {pack_mismatches}")


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve()
    hessian_path = Path(args.hessian).expanduser().resolve()
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    buffer_path = Path(args.calibration_buffer).expanduser().resolve()
    validate_frozen_artifacts(plan_path, hessian_path, pack_dir)
    servers = [parse_server(value) for value in args.server]
    if not servers or len({row["instance"] for row in servers}) != len(servers):
        raise ValueError("manifest requires unique runtime-attested servers")

    registered = {
        split: list(TASK_SET_REGISTRY[split])
        for split in ("atomic_seen", "composite_seen", "composite_unseen")
    }
    if registered != PROTOCOL["table1"]["tasks"]:
        raise ValueError("RoboCasa registry and frozen Table-1 task inventory differ")

    checkpoint_record = artifact(checkpoint, expected_sha256=CHECKPOINT_SHA256)
    checkpoint_record["hash_verification"] = "server launcher and runtime attestation"
    artifacts = {
        "checkpoint": checkpoint_record,
        "checkpoint_config": artifact(checkpoint.parent / "config.json"),
        "norm_stats": artifact(
            checkpoint.parent / "assets/pi05_pretrain_human300/norm_stats.json"
        ),
        "calibration_buffer": artifact(buffer_path),
        "frozen_plan": artifact(plan_path),
        "hessian_w4": artifact(hessian_path),
        "hessian_w4_sidecar": artifact(Path(str(hessian_path) + ".json")),
        "identity_pack": directory_artifact(pack_dir),
        "server_launcher": artifact(REPO_ROOT / "scripts/run_pi05_formal_server.sh"),
        "worker_launcher": artifact(REPO_ROOT / "scripts/run_pi05_formal_worker_seeded.sh"),
        "evaluator": artifact(REPO_ROOT / "scripts/run_robocasa365_pi05_eval.py"),
        "orchestrator": artifact(REPO_ROOT / "scripts/run_full_context_pi05_table1.sh"),
        "manifest_creator": artifact(Path(__file__).resolve()),
        "aggregator": artifact(REPO_ROOT / "scripts/tools/aggregate_pi05_dypac_table1.py"),
    }
    return {
        "schema_version": 1,
        "kind": "dypac_vla_pi05_robocasa365_table1_standalone_manifest",
        "immutable": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "config_id": CONFIG_ID,
        "model": "pi0.5",
        "method": "DyPAC-VLA",
        "git": git_state(),
        "full_context_protocol": protocol_attestation(),
        "table1_protocol": {
            **PROTOCOL["table1"],
            "canonical_action_horizon": 16,
            "native_action_horizon": 50,
            "n_action_steps": 16,
            "replan_steps": 16,
            "flow_steps": 4,
            "paired_action_noise": True,
            "action_noise_protocol": (
                "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"
            ),
            "fresh_environment_per_episode": True,
            "official_task_horizon": True,
            "render": True,
        },
        "frozen_plan_sha256": FROZEN_PLAN_SHA256,
        "artifacts": artifacts,
        "servers": servers,
        "worker_schedule": {
            "task_shards": len(servers),
            "seed_shards": ["0-24", "25-49"],
            "expected_workers": 2 * len(servers),
            "global_resume_key": ["config", "task_set", "task", "seed"],
        },
        "result_feedback_allowed_before_completion": False,
        "comparison_scope": (
            "standalone formal DyPAC-VLA row; no cross-config claim is licensed by this run"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--hessian", required=True)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-buffer", required=True)
    parser.add_argument("--server", action="append", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = Path(args.out).expanduser().resolve()
    candidate = build(args)
    if args.verify:
        existing = json.loads(output.read_text(encoding="utf-8"))
        candidate["created_utc"] = existing["created_utc"]
        if canonical_hash(candidate) != canonical_hash(existing):
            raise ValueError("immutable DyPAC-VLA pi0.5 Table-1 manifest drift")
        print(f"manifest verified: {output}")
        return
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable manifest: {output}")
    atomic_write(output, candidate)
    print(f"manifest created: {output}")
    print(f"manifest sha256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
