#!/usr/bin/env python3
"""Execution-only concurrency sidecar for an already frozen predictive run.

The sidecar never reads outcomes.  It waits for a named predictive server,
launches fixed preregistered mask IDs that are far behind the main runner's
current queue position, and leaves an explicit execution-amendment artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import subprocess
import time
from pathlib import Path

import run_gr00t_dpac_predictive_validity as runner
from quantvla_predictive_validity import artifact, atomic_json


DEFAULT_MASK_IDS = tuple(f"k16_m{index:02d}" for index in range(1, 10))
CLIENT_LAUNCH_STAGGER_SECONDS = 15
RUNTIME_MINIMUM_FREE_MIB = 2_048


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_config_id() -> str | None:
    try:
        value = runner.endpoint(runner.PORT, "get_runtime_info", timeout_ms=10_000)
    except Exception:
        return None
    return value.get("config_id") if isinstance(value, dict) else None


def wait_for_server(split: str) -> None:
    expected = f"full_{split}_predictive"
    while runtime_config_id() != expected:
        print(
            f"[predictive/coverage] sidecar waiting for {expected}", flush=True
        )
        time.sleep(30)


def wait_for_memory(minimum_free_mib: int) -> int:
    while True:
        free = runner.gpu_free_mib()
        if free >= minimum_free_mib:
            return free
        print(
            "[predictive/coverage] sidecar waiting for GPU7 headroom: "
            f"free={free} MiB required={minimum_free_mib} MiB",
            flush=True,
        )
        time.sleep(30)


def active_mask_ids() -> set[str]:
    output = subprocess.check_output(["ps", "-eo", "args="], text=True)
    result: set[str] = set()
    for line in output.splitlines():
        if "run_robocasa365_gr00t_eval.py" not in line:
            continue
        fields = line.split()
        try:
            if int(fields[fields.index("--port") + 1]) != runner.PORT:
                continue
            result.add(fields[fields.index("--predictive-mask-id") + 1])
        except (TypeError, ValueError, IndexError):
            continue
    return result


def run_split(
    split: str, mask_ids: tuple[str, ...], minimum_free_mib: int
) -> None:
    wait_for_server(split)
    prelaunch_free = wait_for_memory(minimum_free_mib)
    library = runner.load_library()
    tasks = list(library["closed_loop"]["tasks"][split])
    root = runner.CLOSED_ROOT / "masks"
    pending: list[str] = []
    skipped_existing_or_active: list[str] = []
    active = active_mask_ids()
    for identifier in mask_ids:
        path = root / split / f"{identifier}.jsonl"
        if identifier in active:
            skipped_existing_or_active.append(identifier)
            continue
        try:
            runner.verify_client_file(
                path, tasks, mask_id=identifier, library=library
            )
        except (FileNotFoundError, RuntimeError, ValueError):
            # Never share an output path with a main-runner process.  A partial
            # file is evidence that another/resumed client owns this ID.
            if path.exists():
                skipped_existing_or_active.append(identifier)
            else:
                pending.append(identifier)

    amendment_path = (
        runner.ROOT / "execution_amendments" / f"sidecar_{split}.json"
    )
    payload = {
        "schema_version": 1,
        "kind": "gr00t_dpac_predictive_execution_amendment",
        "created_utc": utc_now(),
        "user_authorized": True,
        "reason": "Use staggered 15-client GPU7 concurrency after the 16-client stress test proved unsafe.",
        "scientific_protocol_changes": [],
        "execution_change": {
            "registered_max_clients": 6,
            "gpu": runner.GPU,
            "port": runner.PORT,
            "base_runner_clients": 6,
            "additional_clients": len(pending),
            "target_total_clients": 6 + len(pending),
            "split": split,
            "selection_rule": (
                "fixed last nine preregistered IDs; no metric or outcome access"
            ),
            "mask_ids": pending,
            "skipped_existing_or_active_mask_ids": skipped_existing_or_active,
            "prelaunch_gpu7_free_mib": prelaunch_free,
            "minimum_free_mib": minimum_free_mib,
            "stable_14_client_peak_mib_from_atomic_seen": 35_378,
            "unsafe_16_client_peak_mib_from_atomic_seen": 45_410,
            "launch_stagger_seconds": CLIENT_LAUNCH_STAGGER_SECONDS,
            "runtime_minimum_free_mib": RUNTIME_MINIMUM_FREE_MIB,
        },
        "closed_loop_launch": artifact(runner.ROOT / "closed_loop_launch.json"),
        "mask_library": artifact(runner.LIBRARY),
        "sidecar_source": artifact(Path(__file__).resolve()),
        "outcomes_inspected": False,
        "status": "running",
    }
    atomic_json(amendment_path, payload)
    print(
        f"[predictive/coverage] sidecar launched {len(pending)} clients for {split}",
        flush=True,
    )

    processes = []
    minimum_observed_free_mib = prelaunch_free
    memory_rollback = None
    for identifier in pending:
        path = root / split / f"{identifier}.jsonl"
        log_path = (
            runner.ROOT / "logs" / f"client_sidecar_{split}_{identifier}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
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
        # Scene creation has a much larger transient EGL allocation than steady
        # stepping.  Staggering prevents all nine sidecar clients from hitting
        # that transient at the same time.
        if identifier != pending[-1]:
            time.sleep(CLIENT_LAUNCH_STAGGER_SECONDS)
        free = runner.gpu_free_mib()
        minimum_observed_free_mib = min(minimum_observed_free_mib, free)
        if free < RUNTIME_MINIMUM_FREE_MIB:
            runner.stop_process(process)
            memory_rollback = {
                "mask_id": identifier,
                "observed_free_mib": free,
                "action": "terminated only the most recently added sidecar client",
            }
            print(
                "[predictive/coverage] sidecar memory guard rolled back "
                f"{identifier}: free={free} MiB",
                flush=True,
            )
            break

    # Continue sampling through scene resets, whose transient allocation is much
    # larger than steady stepping.  At most the newest sidecar client is rolled
    # back; the main server and main-runner clients are never touched.
    while any(process.poll() is None for _, _, process, _ in processes):
        free = runner.gpu_free_mib()
        minimum_observed_free_mib = min(minimum_observed_free_mib, free)
        if free < RUNTIME_MINIMUM_FREE_MIB and memory_rollback is None:
            for identifier, _, process, _ in reversed(processes):
                if process.poll() is None:
                    runner.stop_process(process)
                    memory_rollback = {
                        "mask_id": identifier,
                        "observed_free_mib": free,
                        "action": (
                            "terminated only the most recently added live "
                            "sidecar client"
                        ),
                    }
                    print(
                        "[predictive/coverage] sidecar memory guard rolled back "
                        f"{identifier}: free={free} MiB",
                        flush=True,
                    )
                    break
        time.sleep(5)

    failures = []
    for identifier, path, process, handle in processes:
        return_code = process.wait()
        handle.close()
        try:
            if return_code != 0:
                raise RuntimeError(f"client exited {return_code}")
            runner.verify_client_file(
                path, tasks, mask_id=identifier, library=library
            )
            print(
                f"[predictive/coverage] sidecar complete: {split}/{identifier}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"mask_id": identifier, "error": str(exc)})

    payload["completed_utc"] = utc_now()
    payload["execution_change"]["minimum_observed_free_mib"] = (
        minimum_observed_free_mib
    )
    payload["execution_change"]["memory_rollback"] = memory_rollback
    payload["status"] = "complete" if not failures else "partial"
    payload["failures"] = failures
    atomic_json(amendment_path, payload)
    runner.status()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        default="composite_seen,composite_unseen",
        help="Comma-separated future predictive splits to accelerate.",
    )
    parser.add_argument(
        "--minimum-free-mib",
        type=int,
        default=18_000,
        help="Required device free memory before adding nine staggered EGL clients.",
    )
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--port", type=int, default=27007)
    args = parser.parse_args()
    runner.GPU = args.gpu
    runner.PORT = args.port
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    unknown = set(splits) - set(runner.SPLITS)
    if unknown:
        raise ValueError(f"unknown splits: {sorted(unknown)}")
    for split in splits:
        run_split(split, DEFAULT_MASK_IDS, args.minimum_free_mib)


if __name__ == "__main__":
    main()
