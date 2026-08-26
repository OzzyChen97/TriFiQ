#!/usr/bin/env python3
"""Run disjoint GR00T SoftFold shards early on the currently free GPU4.

The ordinary matrix runner later validates the same immutable manifest and
resumes only missing task/seed pairs.  Holding ``primary_wave.lock`` makes the
full runner wait while this wave owns the manifest-declared primary instance.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import run_robocasa_atomic_matrix as matrix


REPO = Path(__file__).resolve().parents[2]
TASKS = "CloseBlenderLid,CloseFridge,CloseToasterOvenDoor,NavigateKitchen,OpenDrawer"
EGL_POOL = [1, 2, 3, 4, 5, 6, 7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", default=str(REPO / "scripts/dpac_softfold_gr00t_atomic_spec.json")
    )
    parser.add_argument(
        "--run-dir",
        default=str(REPO / "runs/dpac_softfold_15x20_v1/closed_loop/gr00t/atomic_seen"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            REPO
            / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
            "target_posttraining/atomic_seen/checkpoint-60000"
        ),
    )
    parser.add_argument("--config-id", default="softfold_dpac")
    parser.add_argument("--shards", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--server-timeout", type=int, default=1200)
    return parser.parse_args()


def build(args: argparse.Namespace) -> tuple[dict, str, Path]:
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = matrix.build_manifest(
        Path(args.spec).resolve(),
        run_dir,
        "formal",
        list(range(20)),
        "OpenCabinet",
        Path(args.checkpoint).resolve(),
        "atomic_seen",
        TASKS,
        None,
        5,
        4,
        7200,
        5,
        "paired",
        10.0,
        EGL_POOL,
        True,
        False,
        True,
    )
    manifest, manifest_sha = matrix.write_or_validate_manifest(
        run_dir / "manifest.json", manifest
    )
    return manifest, manifest_sha, run_dir


def selected_config(manifest: dict, config_id: str) -> tuple[dict, dict]:
    matches = [config for config in manifest["configs"] if config["id"] == config_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one config {config_id!r}, got {len(matches)}")
    config = matches[0]
    instance = matrix.server_instances(config)[0]
    if int(instance["gpu"]) != 4:
        raise RuntimeError(f"overlap wave is restricted to the GPU4 primary: {instance}")
    return config, instance


def memory_preflight(
    manifest: dict, config: dict, instance: dict, shards: set[int], run_dir: Path
) -> None:
    client_counts = {gpu: 0 for gpu in EGL_POOL}
    for shard in shards:
        gpu = int(config["shard_egl_devices"][shard])
        client_counts[gpu] += 1
    # This wave has one server and at most two EGL clients per device.  Keep a
    # small real guard while avoiding the full-matrix all-device barrier.
    required = {
        gpu: 512.0 + clients * 2300.0 for gpu, clients in client_counts.items()
    }
    required[int(instance["gpu"])] += 16384.0
    free = matrix.gpu_free_memory_mib()
    insufficient = {
        gpu: {"free_mib": free.get(gpu), "required_mib": need}
        for gpu, need in required.items()
        if free.get(gpu, 0.0) < need
    }
    payload = {
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "config": config["id"],
        "server_instance": instance,
        "selected_shards": sorted(shards),
        "free_memory_mib": free,
        "required_free_memory_mib": required,
        "insufficient": insufficient,
    }
    (run_dir / "aggressive_overlap_preflight.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    if insufficient:
        raise SystemExit(f"overlap wave lacks memory: {insufficient}")


def run(args: argparse.Namespace) -> None:
    manifest, manifest_sha, run_dir = build(args)
    config, instance = selected_config(manifest, args.config_id)
    shards = {int(value) for value in args.shards.split(",") if value.strip()}
    if not shards or min(shards) < 0 or max(shards) >= len(manifest["shards"]):
        raise ValueError(f"invalid shard selection: {sorted(shards)}")
    memory_preflight(manifest, config, instance, shards, run_dir)

    servers = []
    clients = []
    monitor_stop = threading.Event()
    monitor = threading.Thread(
        target=matrix.monitor_gpus,
        args=(
            monitor_stop,
            run_dir / "gpu_efficiency.jsonl",
            [{"id": config["id"], "gpu": instance["gpu"]}],
            10.0,
        ),
        daemon=True,
    )
    monitor.start()
    try:
        proc, handle = matrix.start_server(config, instance, manifest, run_dir)
        servers.append((proc, handle, config, instance))
        runtime = matrix.wait_and_verify(
            config, instance, manifest, proc, args.server_timeout
        )
        (run_dir / "runtime_info_aggressive_overlap.json").write_text(
            json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[dpac-overlap] verified {config['id']} on GPU4", flush=True)
        clients = matrix.start_clients(
            manifest,
            manifest_sha,
            run_dir,
            config_ids={config["id"]},
            shard_indices=shards,
            instance_mode="primary_only",
        )
        failures = []
        for child, child_handle, label in clients:
            rc = child.wait()
            child_handle.close()
            print(f"[dpac-overlap] client {label} exit={rc}", flush=True)
            if rc != 0:
                failures.append((label, rc))
        if failures:
            raise SystemExit(f"overlap clients failed: {failures}; full runner will resume")
        print("[dpac-overlap] complete", flush=True)
    finally:
        for child, child_handle, _ in clients:
            matrix.stop_process(child)
            if not child_handle.closed:
                child_handle.close()
        for proc, handle, _, _ in servers:
            matrix.stop_process(proc)
            if not handle.closed:
                handle.close()
        monitor_stop.set()
        monitor.join(timeout=15)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "primary_wave.lock", "a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("full matrix or another overlap wave currently owns the lock") from exc
        run(args)


if __name__ == "__main__":
    main()
