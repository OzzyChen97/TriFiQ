#!/usr/bin/env python3
"""Run one preregistered predictive-validity split on a secondary GPU.

This is an execution-only accelerator.  It reuses the frozen library, masks,
tasks, seeds, clients, and verification logic without reading outcomes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
from pathlib import Path

import run_gr00t_dpac_predictive_validity as runner
from quantvla_predictive_validity import artifact, atomic_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_identity(gpu: int) -> dict[str, str | int]:
    fields = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=name,uuid,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().split(",")
    if len(fields) != 4:
        raise RuntimeError(f"unexpected nvidia-smi response for GPU{gpu}: {fields}")
    return {
        "index": gpu,
        "name": fields[0].strip(),
        "uuid": fields[1].strip(),
        "memory_total_mib": int(fields[2].strip()),
        "memory_used_prelaunch_mib": int(fields[3].strip()),
    }


def verify_frozen_inputs_with_metadata_only_source_drift() -> dict:
    launch_path = runner.ROOT / "closed_loop_launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if launch["mask_library"] != artifact(runner.LIBRARY):
        raise RuntimeError("frozen mask library drift")
    offline = runner.OFFLINE_ROOT / "scores_all_splits.json"
    if launch["offline_scores"] != artifact(offline):
        raise RuntimeError("frozen offline scores drift")

    current_sources = runner.source_artifacts()
    service_key = "scripts/inference_service.py"
    for key, value in current_sources.items():
        if key != service_key and launch["source_files"].get(key) != value:
            raise RuntimeError(f"unapproved frozen source drift: {key}")

    # A concurrent user edit made the disabled predictive diagnostic disappear
    # from ordinary FP16 runtime-info receipts.  Reconstruct the launch text in
    # memory and require its exact frozen hash.  This proves the delta is only
    # that metadata presentation; predictive servers still emit the same field.
    source_path = runner.REPO / service_key
    current_text = source_path.read_text(encoding="utf-8")
    payload_old = '''        "qvla_actquant": reproduction_runtime,\n        "daptq": daptq_runtime,\n        "model_dtype": model_dtype,'''
    payload_frozen = '''        "qvla_actquant": reproduction_runtime,\n        "daptq": daptq_runtime,\n        "predictive_mask_runtime": predictive_metadata,\n        "model_dtype": model_dtype,'''
    conditional_block = '''    # Preserve the stable runtime identity of ordinary evaluation arms.  A\n    # disabled predictive-mask diagnostic is not part of their semantics and\n    # must not invalidate receipts produced before that diagnostic existed.\n    if predictive_runtime is not None:\n        payload["predictive_mask_runtime"] = predictive_metadata\n'''
    if current_text.count(payload_old) != 1 or current_text.count(conditional_block) != 1:
        raise RuntimeError("metadata-only source-drift shape changed")
    reconstructed = current_text.replace(payload_old, payload_frozen).replace(
        conditional_block, ""
    )
    reconstructed_bytes = reconstructed.encode("utf-8")
    reconstructed_hash = hashlib.sha256(reconstructed_bytes).hexdigest()
    frozen = launch["source_files"][service_key]
    if reconstructed_hash != frozen["sha256"] or len(reconstructed_bytes) != frozen["bytes"]:
        raise RuntimeError("source drift is not the approved metadata-only delta")
    return {
        "classification": "metadata_only_no_action_path_change",
        "frozen": frozen,
        "current": current_sources[service_key],
        "exact_frozen_hash_reconstructed_in_memory": True,
        "runtime_effect": (
            "ordinary FP16 runtime-info omits an enabled:false predictive diagnostic; "
            "predictive runtime-info and all action paths are unchanged"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="composite_unseen")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=27017)
    args = parser.parse_args()
    if args.split not in runner.SPLITS:
        raise ValueError(f"unknown split: {args.split}")
    if args.gpu == runner.GPU or args.port == runner.PORT:
        raise ValueError("secondary worker requires a distinct GPU and port")

    library = runner.load_library()
    source_drift_audit = verify_frozen_inputs_with_metadata_only_source_drift()
    identity = gpu_identity(args.gpu)
    if identity["name"] != "NVIDIA A40":
        raise RuntimeError(
            f"secondary GPU must match registered A40 hardware: {identity}"
        )

    # All existing server/client helpers reference these module-level execution
    # coordinates.  Changing them affects only this process, not the frozen
    # protocol or the already-running GPU7 scheduler.
    runner.GPU = args.gpu
    runner.PORT = args.port
    tasks = list(library["closed_loop"]["tasks"][args.split])
    mask_ids = [row["candidate_id"] for row in library["candidates"]]
    amendment_path = (
        runner.ROOT / "execution_amendments" / f"secondary_gpu_{args.split}.json"
    )
    payload = {
        "schema_version": 1,
        "kind": "gr00t_dpac_predictive_execution_amendment",
        "created_utc": utc_now(),
        "user_authorized": True,
        "reason": "Parallelize an untouched future split on the user-authorized secondary A40.",
        "scientific_protocol_changes": [],
        "execution_change": {
            "split": args.split,
            "gpu": identity,
            "port": args.port,
            "base_clients": 6,
            "companion_sidecar_clients": 9,
            "target_total_clients": 15,
            "mask_ids": mask_ids,
            "tasks": tasks,
        },
        "closed_loop_launch": artifact(runner.ROOT / "closed_loop_launch.json"),
        "mask_library": artifact(runner.LIBRARY),
        "worker_source": artifact(Path(__file__).resolve()),
        "source_drift_audit": source_drift_audit,
        "outcomes_inspected": False,
        "status": "running",
    }
    atomic_json(amendment_path, payload)

    runner.wait_for_gpu()
    server, handle = runner.start_server(
        library,
        args.split,
        predictive=True,
        label=f"full_{args.split}_predictive",
    )
    try:
        runner.run_client_wave(
            library,
            args.split,
            tasks,
            mask_ids,
            runner.CLOSED_ROOT / "masks",
        )
    finally:
        runner.stop_process(server)
        handle.close()

    runner.wait_for_gpu()
    server, handle = runner.start_server(
        library,
        args.split,
        predictive=False,
        label=f"full_{args.split}_fp16",
    )
    try:
        runner.run_fp16_client(
            library,
            args.split,
            tasks,
            runner.CLOSED_ROOT / "fp16",
        )
    finally:
        runner.stop_process(server)
        handle.close()

    payload["completed_utc"] = utc_now()
    payload["status"] = "complete"
    atomic_json(amendment_path, payload)
    runner.status()


if __name__ == "__main__":
    main()
