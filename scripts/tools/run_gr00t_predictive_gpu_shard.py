#!/usr/bin/env python3
"""Run an explicit, disjoint predictive-mask shard on one accelerator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import run_gr00t_dpac_predictive_validity as runner
from quantvla_predictive_validity import artifact, atomic_json
from recover_gr00t_predictive_split import finish_masks
from run_gr00t_predictive_secondary_gpu import (
    gpu_identity,
    verify_frozen_inputs_with_metadata_only_source_drift,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def active_predictive_outputs() -> set[Path]:
    output = subprocess.check_output(["ps", "-eo", "args="], text=True)
    paths: set[Path] = set()
    for line in output.splitlines():
        if "run_robocasa365_gr00t_eval.py" not in line:
            continue
        fields = line.split()
        try:
            paths.add(Path(fields[fields.index("--out") + 1]).resolve())
        except (ValueError, IndexError):
            continue
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--mask-ids", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--max-clients", type=int, default=14)
    args = parser.parse_args()

    splits = [value for value in args.splits.split(",") if value]
    if len(splits) != len(set(splits)) or not set(splits) <= set(runner.SPLITS):
        raise ValueError(f"invalid split inventory: {splits}")

    library = runner.load_library()
    registered = [row["candidate_id"] for row in library["candidates"]]
    mask_ids = [value for value in args.mask_ids.split(",") if value]
    if (
        not mask_ids
        or len(mask_ids) != len(set(mask_ids))
        or not set(mask_ids) <= set(registered)
    ):
        raise ValueError(f"invalid mask shard: {mask_ids}")

    identity = gpu_identity(args.gpu)
    if identity["name"] != "NVIDIA A40":
        raise RuntimeError(f"shard GPU must match registered A40 hardware: {identity}")
    source_drift_audit = verify_frozen_inputs_with_metadata_only_source_drift()
    runner.GPU = args.gpu
    runner.PORT = args.port

    root = runner.ROOT / "execution_amendments"
    root.mkdir(parents=True, exist_ok=True)
    amendment_path = root / f"gpu{args.gpu}_disjoint_shard.json"
    payload = {
        "schema_version": 1,
        "kind": "gr00t_dpac_predictive_disjoint_accelerator_shard",
        "created_utc": utc_now(),
        "user_authorized": True,
        "reason": "Use an additional user-authorized idle GPU without changing the frozen protocol.",
        "scientific_protocol_changes": [],
        "execution_change": {
            "gpu": identity,
            "port": args.port,
            "max_clients": args.max_clients,
            "splits_in_order": splits,
            "mask_ids": mask_ids,
            "ownership": "exclusive output paths; canonical recovery workers exclude these IDs",
        },
        "closed_loop_launch": artifact(runner.ROOT / "closed_loop_launch.json"),
        "mask_library": artifact(runner.LIBRARY),
        "worker_source": artifact(Path(__file__).resolve()),
        "source_drift_audit": source_drift_audit,
        "outcomes_inspected": False,
        "status": "preflight",
    }
    atomic_json(amendment_path, payload)

    claimed_paths = {
        (runner.CLOSED_ROOT / "masks" / split / f"{identifier}.jsonl").resolve()
        for split in splits
        for identifier in mask_ids
    }
    collisions = sorted(str(path) for path in claimed_paths & active_predictive_outputs())
    if collisions:
        raise RuntimeError(f"shard output paths already active: {collisions}")

    for split in splits:
        tasks = list(library["closed_loop"]["tasks"][split])
        payload["status"] = f"waiting_for_gpu:{split}"
        payload["last_observed_utc"] = utc_now()
        atomic_json(amendment_path, payload)
        runner.wait_for_gpu()
        payload["status"] = f"running:{split}"
        payload["last_observed_utc"] = utc_now()
        atomic_json(amendment_path, payload)
        server, handle = runner.start_server(
            library,
            split,
            predictive=True,
            label=f"gpu{args.gpu}_shard_{split}_predictive",
        )
        try:
            finish_masks(
                library,
                split,
                tasks,
                args.max_clients,
                identifiers=mask_ids,
            )
        finally:
            runner.stop_process(server)
            handle.close()
        payload.setdefault("completed_splits", []).append(split)
        payload["last_observed_utc"] = utc_now()
        atomic_json(amendment_path, payload)

    payload["status"] = "complete"
    payload["completed_utc"] = utc_now()
    atomic_json(amendment_path, payload)
    runner.status()


if __name__ == "__main__":
    main()
