#!/usr/bin/env python3
"""Freeze RoboCasa365 Omega-QVLA evaluation specs from attested W4A4 packs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "runs/gdsq_extension_preregistered_v1/omega_qvla_robocasa365_v1"
CHECKPOINT_ROOT = REPO / (
    "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
    "target_posttraining"
)
LLM_RE = (
    r".*backbone\.eagle_model\.language_model\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*"
)
DIT_RE = (
    r".*action_head\.model\.transformer_blocks\.\d+\."
    r"(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2)).*"
)
PORTS = {
    "atomic_seen": 20410,
    "composite_seen": 20420,
    "composite_unseen": 20430,
}
GPU_LAYOUT = [1, 2, 4, 5, 6, 7]
SPEC_VERSION = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_pack_layer_count(path: Path) -> int:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(f"Omega-QVLA pack is not a mapping: {path}")
    names = [name for name in payload if name != "__meta__"]
    if not names:
        raise SystemExit(f"Omega-QVLA pack has zero layer records: {path}")
    return len(names)


def build_spec(task_set: str) -> dict[str, Any]:
    pack = ROOT / "packs" / task_set / "omega_qvla_w4a4.pt"
    attestation_path = pack.with_suffix(".attestation.json")
    calibration_path = ROOT / "calibration" / f"{task_set}.manifest.v3.json"
    for path in (pack, attestation_path, calibration_path):
        if not path.is_file():
            raise SystemExit(f"required artifact missing: {path}")
    if "libero" in str(pack).lower():
        raise SystemExit(f"LIBERO pack is forbidden for RoboCasa365: {pack}")
    attestation = json.loads(attestation_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    checks = [
        attestation.get("pack_sha256") == sha256_file(pack),
        attestation.get("task_set") == task_set,
        attestation.get("calibration_manifest_sha256") == sha256_file(calibration_path),
        calibration.get("task_set") == task_set,
        calibration.get("recipe", {}).get("weight_bits") == 4,
        calibration.get("recipe", {}).get("activation_bits") == 4,
        calibration.get("recipe", {}).get("denoising_steps") == 4,
        calibration.get("calibration", {}).get("test_results_used") is False,
    ]
    if not all(checks):
        raise SystemExit(f"Omega-QVLA artifact attestation failed for {task_set}")
    expected_wrapped = load_pack_layer_count(pack)
    base_port = PORTS[task_set]
    previous_spec = ROOT / "specs" / f"omega_qvla_{task_set}_v2.json"
    if not previous_spec.is_file():
        previous_spec = ROOT / "specs" / f"omega_qvla_{task_set}.json"
    return {
        "purpose": f"Omega-QVLA W4A4 on RoboCasa365 {task_set}",
        "comparisons": [],
        "decision": {
            "final_config": "omega_qvla_w4a4",
            "selection": "none; external baseline evaluated without result feedback",
        },
        "supersedes": ({
            "path": str(previous_spec.resolve()),
            "sha256": sha256_file(previous_spec),
            "reason": (
                "Resource-only reschedule: GPU 3 is occupied by a pre-existing "
                "training job, while GPU 1 is explicitly shared under the conservative "
                "free-memory gate. Method, pack, calibration, tasks, and seeds are unchanged."
            ),
        } if previous_spec.is_file() else None),
        "configs": [{
            "id": "omega_qvla_w4a4",
            "gpu": GPU_LAYOUT[0],
            "port": base_port,
            "replicas": [
                {"gpu": gpu, "port": base_port + index}
                for index, gpu in enumerate(GPU_LAYOUT[1:], 1)
            ],
            "expected_wrapped": expected_wrapped,
            "omega_pack": str(pack.resolve()),
            "omega_calibration_manifest": str(calibration_path.resolve()),
            "omega_include": f"(?:{LLM_RE}|{DIT_RE})",
            "meta": {
                "method": "Omega-QVLA",
                "weight_bits": 4,
                "activation_bits": 4,
                "upstream_commit": calibration["upstream"]["commit"],
                "rank0_fast_path_diff_sha256": calibration["upstream"][
                    "working_diff_sha256"
                ],
                "official_libero_packs_reused": False,
                "resource_schedule": "six-model-servers-twelve-simulator-shards-shared-memory-gate-v3",
                "allow_shared_gpu_processes_with_memory_gate": True,
            },
        }],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["materialize", "show"])
    parser.add_argument(
        "--task-set", required=True,
        choices=["atomic_seen", "composite_seen", "composite_unseen"],
    )
    args = parser.parse_args()
    spec = build_spec(args.task_set)
    path = ROOT / "specs" / f"omega_qvla_{args.task_set}_v{SPEC_VERSION}.json"
    if args.command == "materialize":
        if path.exists():
            saved = json.loads(path.read_text())
            if saved != spec:
                raise SystemExit(f"immutable Omega-QVLA spec mismatch: {path}")
        else:
            atomic_json(path, spec)
        print(path)
    else:
        print(json.dumps(spec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
