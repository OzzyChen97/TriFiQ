#!/usr/bin/env python3
"""Adopt an orphaned predictive server and finish one registered split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import run_gr00t_dpac_predictive_validity as runner
from quantvla_predictive_validity import artifact, atomic_json, sha256_file
from run_gr00t_predictive_secondary_gpu import (
    gpu_identity,
    verify_frozen_inputs_with_metadata_only_source_drift,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_lines() -> list[str]:
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    return output.splitlines()


def active_client_ids(port: int) -> list[str]:
    result = []
    for line in process_lines():
        if "run_robocasa365_gr00t_eval.py" not in line:
            continue
        fields = line.split()
        try:
            if int(fields[fields.index("--port") + 1]) != port:
                continue
            result.append(fields[fields.index("--predictive-mask-id") + 1])
        except (TypeError, ValueError, IndexError):
            continue
    return sorted(result)


def server_pid(port: int) -> int:
    matches = []
    for line in process_lines():
        if "inference_service.py" not in line or "--server" not in line:
            continue
        fields = line.split()
        try:
            if int(fields[fields.index("--port") + 1]) == port:
                matches.append(int(fields[0]))
        except (TypeError, ValueError, IndexError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f"expected one server on port {port}, found {matches}")
    return matches[0]


def validate_server(library: dict, split: str, port: int) -> dict:
    info = runner.endpoint(port, "get_runtime_info", timeout_ms=120_000)
    runtime = info.get("predictive_mask_runtime") or {}
    if (
        info.get("config_id") != f"full_{split}_predictive"
        or runtime.get("manifest_sha256") != sha256_file(runner.LIBRARY)
        or runtime.get("candidate_count") != 60
        or info.get("wrapped_layers") != 116
    ):
        raise RuntimeError(f"cannot adopt unverified server: {info}")
    return {
        "config_id": info["config_id"],
        "manifest_sha256": runtime["manifest_sha256"],
        "candidate_count": runtime["candidate_count"],
        "wrapped_layers": info["wrapped_layers"],
        "model_path": info["model_path"],
    }


def stop_adopted_server(port: int) -> None:
    pid = server_pid(port)
    os.killpg(pid, signal.SIGTERM)
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
    raise RuntimeError(f"server {pid} on port {port} did not terminate")


def finish_masks(
    library: dict,
    split: str,
    tasks: list[str],
    max_clients: int,
    identifiers: list[str] | None = None,
) -> None:
    if identifiers is None:
        identifiers = [row["candidate_id"] for row in library["candidates"]]
    pending = list(identifiers)
    attempts = {identifier: 0 for identifier in identifiers}
    while pending:
        batch, pending = pending[:max_clients], pending[max_clients:]
        processes = []
        for identifier in batch:
            path = runner.CLOSED_ROOT / "masks" / split / f"{identifier}.jsonl"
            try:
                runner.verify_client_file(
                    path, tasks, mask_id=identifier, library=library
                )
                continue
            except (FileNotFoundError, RuntimeError, ValueError):
                pass
            attempts[identifier] += 1
            log_path = (
                runner.ROOT / "logs" / f"client_recovery_{split}_{identifier}.log"
            )
            handle = log_path.open("a", buffering=1)
            process = subprocess.Popen(
                runner.client_command(tasks, path, mask_id=identifier),
                cwd=runner.REPO,
                env=runner.clean_client_env(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((identifier, path, process, handle))
        retry = []
        for identifier, path, process, handle in processes:
            return_code = process.wait()
            handle.close()
            try:
                if return_code != 0:
                    raise RuntimeError(f"client exited {return_code}")
                runner.verify_client_file(
                    path, tasks, mask_id=identifier, library=library
                )
            except Exception as exc:
                if attempts[identifier] >= 3:
                    raise RuntimeError(
                        f"{split}/{identifier}: recovery failed three times: {exc}"
                    ) from exc
                retry.append(identifier)
        pending.extend(retry)
        runner.status()


def wait_for_full_split(
    library: dict,
    split: str,
    tasks: list[str],
    payload: dict,
    amendment_path: Path,
) -> None:
    """Keep the adopted server alive until every disjoint shard is committed."""

    identifiers = [row["candidate_id"] for row in library["candidates"]]
    while True:
        complete = []
        for identifier in identifiers:
            path = runner.CLOSED_ROOT / "masks" / split / f"{identifier}.jsonl"
            try:
                runner.verify_client_file(
                    path, tasks, mask_id=identifier, library=library
                )
                complete.append(identifier)
            except (FileNotFoundError, RuntimeError, ValueError):
                pass
        if len(complete) == len(identifiers):
            return
        payload["status"] = "waiting_for_disjoint_shards"
        payload["last_observed_utc"] = utc_now()
        payload["complete_mask_files"] = len(complete)
        payload["expected_mask_files"] = len(identifiers)
        atomic_json(amendment_path, payload)
        print(
            f"[predictive/coverage] recovery {split} waiting for disjoint "
            f"shards: {len(complete)}/{len(identifiers)} mask files",
            flush=True,
        )
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=runner.SPLITS)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--max-clients", type=int, default=14)
    parser.add_argument(
        "--exclude-mask-ids",
        default="",
        help="Comma-separated IDs owned by a disjoint accelerator shard.",
    )
    args = parser.parse_args()
    runner.GPU = args.gpu
    runner.PORT = args.port
    lock_path = runner.ROOT / "execution_amendments" / f"recovery_{args.split}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"recovery already active for {args.split}") from exc

    library = runner.load_library()
    all_identifiers = [row["candidate_id"] for row in library["candidates"]]
    excluded = [value for value in args.exclude_mask_ids.split(",") if value]
    if len(excluded) != len(set(excluded)) or not set(excluded) <= set(all_identifiers):
        raise ValueError(f"invalid excluded mask IDs: {excluded}")
    local_identifiers = [
        identifier for identifier in all_identifiers if identifier not in set(excluded)
    ]
    tasks = list(library["closed_loop"]["tasks"][args.split])
    amendment_path = (
        runner.ROOT / "execution_amendments" / f"recovery_{args.split}.json"
    )
    payload = {
        "schema_version": 1,
        "kind": "gr00t_dpac_predictive_execution_recovery",
        "created_utc": utc_now(),
        "user_authorized": True,
        "reason": "Adopt verified orphaned server/clients after supervisor session loss.",
        "scientific_protocol_changes": [],
        "execution_change": {
            "split": args.split,
            "gpu": gpu_identity(args.gpu),
            "port": args.port,
            "resume_max_clients": args.max_clients,
            "selection_rule": (
                "all incomplete preregistered IDs in manifest order excluding the "
                "explicit disjoint accelerator shard"
            ),
            "local_mask_ids": local_identifiers,
            "excluded_accelerator_mask_ids": excluded,
        },
        "mask_library": artifact(runner.LIBRARY),
        "source_drift_audit": verify_frozen_inputs_with_metadata_only_source_drift(),
        "recovery_source": artifact(Path(__file__).resolve()),
        "outcomes_inspected": False,
        "status": "waiting_for_orphan_clients",
    }
    payload["adopted_server"] = validate_server(library, args.split, args.port)
    atomic_json(amendment_path, payload)

    while True:
        active = active_client_ids(args.port)
        if not active:
            break
        payload["last_observed_utc"] = utc_now()
        payload["active_orphan_client_count"] = len(active)
        payload["active_orphan_mask_ids"] = active
        atomic_json(amendment_path, payload)
        print(
            f"[predictive/coverage] recovery {args.split} waiting for "
            f"{len(active)} adopted clients",
            flush=True,
        )
        time.sleep(30)

    validate_server(library, args.split, args.port)
    payload["status"] = "resuming_incomplete_masks"
    payload["orphan_clients_completed_utc"] = utc_now()
    atomic_json(amendment_path, payload)
    finish_masks(
        library,
        args.split,
        tasks,
        args.max_clients,
        identifiers=local_identifiers,
    )
    wait_for_full_split(library, args.split, tasks, payload, amendment_path)

    stop_adopted_server(args.port)
    runner.wait_for_gpu()
    server, handle = runner.start_server(
        library,
        args.split,
        predictive=False,
        label=f"recovery_{args.split}_fp16",
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

    payload["status"] = "complete"
    payload["completed_utc"] = utc_now()
    atomic_json(amendment_path, payload)
    runner.status()


if __name__ == "__main__":
    main()
