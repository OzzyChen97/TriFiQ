#!/usr/bin/env python3
"""Create or verify an immutable pi0.5 week-1 control manifest.

The manifest freezes the exact plan, plan-specific A8 artifact, runtime
attestations, source/environment hashes, task/seed schedule, and any paired
reference manifests before an official result row is allowed to exist.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
PREREG = REPO_ROOT / "runs/gdsq_week1_preregistered_v1/preregistration.json"
PREREG_SHA256 = "216f1b6267b5bc9ff67cb19f9e3502c30836b0aa7e81e10ade522fa7d6104541"
BASE_MANIFEST = (
    REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/official_target_paired50/manifest.json"
)
DEV4 = [
    "CoffeeSetupMug",
    "OpenCabinet",
    "OpenStandMixerHead",
    "PickPlaceDrawerToCounter",
]
PAIRED_NOISE = "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"
CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scope", choices=("all50", "heldout46", "dev4"), required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--a8", required=True)
    parser.add_argument("--expected-wrapped", type=int, required=True)
    parser.add_argument("--orchestrator", required=True)
    parser.add_argument("--preflight-results", required=True)
    parser.add_argument("--preflight-audit", required=True)
    parser.add_argument(
        "--server",
        action="append",
        required=True,
        help="INSTANCE,CONFIG,GPU,PORT,RUNTIME_JSON,RUNTIME_AUDIT_JSON",
    )
    parser.add_argument(
        "--worker",
        action="append",
        required=True,
        help="WORKER_ID,SERVER_INSTANCE,SHARD_INDEX,SHARD_COUNT,SEED_START-SEED_END",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help="NAME,MANIFEST_JSON,CONFIG_ID for a preregistered paired contrast",
    )
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(resolved.is_file(), f"missing artifact: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def iter_sha_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("sha256"), str):
            yield value
        for child in value.values():
            yield from iter_sha_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_sha_records(child)


def task_sets_for_scope(scope: str) -> dict[str, list[str]]:
    base = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))["table_1_protocol"][
        "task_sets"
    ]
    if scope == "all50":
        return base
    if scope == "dev4":
        return {"atomic_seen": DEV4}
    return {
        name: [task for task in tasks if task not in DEV4]
        for name, tasks in base.items()
    }


def validate_plan_and_a8(
    plan_path: Path, a8_path: Path, expected_wrapped: int
) -> dict[str, Any]:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    plan_record = artifact(plan_path)
    require(
        any(record.get("sha256") == plan_record["sha256"] for record in iter_sha_records(prereg)),
        "plan SHA is not present in the frozen preregistration",
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selected = {
        name: int(row.get("bits", 0) or 0)
        for name, row in (plan.get("layers") or {}).items()
        if not bool(row.get("skip", not int(row.get("bits", 0) or 0)))
        and int(row.get("bits", 0) or 0) > 0
    }
    require(len(selected) == expected_wrapped, "plan wrapped-layer count mismatch")
    require(set(selected.values()) <= {4, 6}, "week-1 plan may only use W4/W6")
    a8_sidecar_path = Path(str(a8_path) + ".json")
    a8_record = artifact(a8_path)
    sidecar_record = artifact(a8_sidecar_path)
    sidecar = json.loads(a8_sidecar_path.read_text(encoding="utf-8"))
    metadata = sidecar.get("metadata") or {}
    checks = {
        "npz_sha": sidecar.get("npz_sha256") == a8_record["sha256"],
        "plan_sha": metadata.get("plan_sha256") == plan_record["sha256"],
        "wrapped": int(metadata.get("wrapped_layers", -1)) == expected_wrapped,
        "observations": int(metadata.get("calibration_observations", -1)) == 256,
        "batch_size": int(metadata.get("calibration_batch_size", -1)) == 8,
    }
    failed = [name for name, valid in checks.items() if not valid]
    require(not failed, f"A8 sidecar checks failed: {failed}")
    return {
        "plan": plan_record,
        "a8": a8_record,
        "a8_sidecar": sidecar_record,
        "wrapped_layers": expected_wrapped,
        "weight_bits": sorted(set(selected.values())),
        "activation_bits": 8,
        "theoretical_static_bytes": plan.get("total_bytes"),
        "budget_bytes": plan.get("budget_bytes"),
    }


def parse_server(
    spec: str,
    *,
    config: str,
    plan_sha: str,
    a8_sha: str,
    expected_wrapped: int,
) -> dict[str, Any]:
    instance, server_config, gpu_text, port_text, runtime_text, audit_text = spec.split(",", 5)
    require(IDENTIFIER.fullmatch(instance) is not None, f"invalid instance: {instance}")
    require(server_config == config, f"server config mismatch: {server_config}")
    gpu, port = int(gpu_text), int(port_text)
    require(0 <= gpu <= 7 and 1024 <= port <= 65535, f"invalid server resource: {spec}")
    runtime_path = Path(runtime_text).resolve()
    audit_path = Path(audit_text).resolve()
    runtime_record, audit_record = artifact(runtime_path), artifact(audit_path)
    runtime_metadata = json.loads(runtime_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checks = {
        "audit_valid": audit.get("valid") is True,
        "audit_config": audit.get("config_id") == config,
        "runtime_file_sha": audit.get("runtime_file_sha256") == runtime_record["sha256"],
        "server_metadata_sha": audit.get("server_metadata_sha256")
        == canonical_hash(runtime_metadata),
        "plan_sha": audit.get("plan_sha256") == plan_sha,
        "a8_sha": audit.get("a8_sha256") == a8_sha,
        "wrapped": int(audit.get("wrapped_layers", -1)) == expected_wrapped,
        "selector_disabled": audit.get("selector_enabled") is False,
        "atm_disabled": audit.get("atm_enabled") is False,
        "ohb_disabled": audit.get("ohb_enabled") is False,
        "noise": (audit.get("protocol") or {}).get("paired_noise") == PAIRED_NOISE,
    }
    failed = [name for name, valid in checks.items() if not valid]
    require(not failed, f"runtime attestation failed for {instance}: {failed}")
    return {
        "instance": instance,
        "config_id": config,
        "gpu": gpu,
        "port": port,
        "runtime": runtime_record,
        "runtime_audit": audit_record,
        "server_metadata_sha256": audit["server_metadata_sha256"],
    }


def parse_worker(spec: str, config: str) -> dict[str, Any]:
    worker_id, instance, shard_text, count_text, seed_text = spec.split(",", 4)
    require(IDENTIFIER.fullmatch(worker_id) is not None, f"invalid worker: {worker_id}")
    shard, count = int(shard_text), int(count_text)
    seed_start_text, seed_end_text = seed_text.split("-", 1)
    seed_start, seed_end = int(seed_start_text), int(seed_end_text)
    require(count > 0 and 0 <= shard < count, f"invalid task shard: {spec}")
    require(0 <= seed_start <= seed_end <= 49, f"invalid seed range: {spec}")
    return {
        "worker_id": worker_id,
        "server_instance": instance,
        "config_id": config,
        "task_shard_index": shard,
        "task_shard_count": count,
        "trial_seed_start": seed_start,
        "trial_seed_end": seed_end,
    }


def validate_schedule(
    servers: list[dict[str, Any]],
    workers: list[dict[str, Any]],
    task_sets: dict[str, list[str]],
) -> dict[str, Any]:
    require(servers and workers, "empty formal schedule")
    require(len({row["instance"] for row in servers}) == len(servers), "duplicate server")
    require(len({row["gpu"] for row in servers}) == len(servers), "duplicate server GPU")
    require(len({row["port"] for row in servers}) == len(servers), "duplicate server port")
    instances = {row["instance"] for row in servers}
    require(len({row["worker_id"] for row in workers}) == len(workers), "duplicate worker")
    require(
        all(row["server_instance"] in instances for row in workers),
        "worker references unknown server",
    )
    expected = {
        (task_set, task, seed)
        for task_set, tasks in task_sets.items()
        for task in tasks
        for seed in range(50)
    }
    covered: set[tuple[str, str, int]] = set()
    duplicates = 0
    for worker in workers:
        for task_set, tasks in task_sets.items():
            selected = [
                task
                for index, task in enumerate(tasks)
                if index % worker["task_shard_count"] == worker["task_shard_index"]
            ]
            for task in selected:
                for seed in range(worker["trial_seed_start"], worker["trial_seed_end"] + 1):
                    key = (task_set, task, seed)
                    duplicates += int(key in covered)
                    covered.add(key)
    require(duplicates == 0, f"schedule duplicates {duplicates} episode keys")
    require(
        covered == expected,
        f"schedule mismatch: missing={len(expected-covered)} extra={len(covered-expected)}",
    )
    return {
        "expected_episode_keys": len(expected),
        "covered_episode_keys": len(covered),
        "duplicate_episode_keys": duplicates,
        "keyset_sha256": canonical_hash(sorted(covered)),
    }


def parse_reference(spec: str) -> dict[str, Any]:
    name, manifest_text, config = spec.split(",", 2)
    require(IDENTIFIER.fullmatch(name) is not None, f"invalid reference name: {name}")
    require(IDENTIFIER.fullmatch(config) is not None, f"invalid reference config: {config}")
    manifest_path = Path(manifest_text).resolve()
    manifest_record = artifact(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    servers = manifest.get("servers") or []
    require(
        any(row.get("config_id") == config for row in servers),
        f"reference manifest has no server for {config}",
    )
    return {
        "name": name,
        "config_id": config,
        "manifest": manifest_record,
        "run_dir": str(manifest_path.parent),
    }


def environment_record() -> dict[str, Any]:
    frozen = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
    )
    normalized = "\n".join(
        sorted(line.strip() for line in frozen.splitlines() if line.strip())
    ) + "\n"
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pip_freeze_sha256": sha256_text(normalized),
        "pip_freeze_entries": len(normalized.splitlines()),
    }


def source_artifacts(orchestrator: Path) -> dict[str, Any]:
    paths = {
        "checkpoint": REPO_ROOT
        / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/model.safetensors",
        "checkpoint_config": REPO_ROOT
        / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/config.json",
        "norm_stats": REPO_ROOT
        / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/assets/pi05_pretrain_human300/norm_stats.json",
        "pack_manifest": REPO_ROOT
        / "runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015/manifest.json",
        "calibration_buffer": REPO_ROOT
        / "runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz",
        "preregistration": PREREG,
        "base_protocol_manifest": BASE_MANIFEST,
        "openpi_lock": REPO_ROOT / "code/pi05/openpi/uv.lock",
        "server_launcher": REPO_ROOT / "scripts/run_pi05_week1_server.sh",
        "worker_launcher": REPO_ROOT / "scripts/run_pi05_week1_worker.sh",
        "persistent_queue": REPO_ROOT / "scripts/run_gdsq_week1_queue.sh",
        "evaluator": REPO_ROOT / "scripts/run_robocasa365_pi05_eval.py",
        "runtime_auditor": REPO_ROOT / "scripts/tools/audit_pi05_week1_runtime.py",
        "preflight_auditor": REPO_ROOT / "scripts/tools/audit_pi05_week1_preflight.py",
        "aggregator": REPO_ROOT / "scripts/tools/aggregate_pi05_week1.py",
        "statistics": REPO_ROOT / "scripts/tools/parse_robocasa_atomic_matrix.py",
        "manifest_tool": Path(__file__).resolve(),
        "calibrator": REPO_ROOT / "scripts/tools/pi05_calibrate_a8.py",
        "duquant_layers": REPO_ROOT / "code/pi05/openpi/src/openpi/quant/duquant_layers.py",
        "server_runtime": REPO_ROOT / "code/pi05/openpi/scripts/serve_pi05_quant_policy.py",
        "orchestrator": orchestrator,
    }
    records = {name: artifact(path) for name, path in paths.items()}
    require(records["checkpoint"]["sha256"] == CHECKPOINT_SHA256, "checkpoint SHA drift")
    require(records["preregistration"]["sha256"] == PREREG_SHA256, "preregistration drift")
    return records


def validate_preflight(results: Path, audit_path: Path, config: str) -> dict[str, Any]:
    results_record, audit_record = artifact(results), artifact(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checks = {
        "valid": audit.get("valid") is True,
        "diagnostic": audit.get("diagnostic_only") is True,
        "no_claim": audit.get("paper_claim_enabled") is False,
        "config": audit.get("config_id") == config,
        "episodes": audit.get("expected_episodes") == audit.get("observed_episodes") == 4,
        "result_sha": audit.get("results_sha256") == results_record["sha256"],
    }
    failed = [name for name, valid in checks.items() if not valid]
    require(not failed, f"preflight checks failed: {failed}")
    return {"results": results_record, "audit": audit_record, "diagnostic_only": True}


def ensure_result_blind_freeze(run_dir: Path) -> None:
    offending = [
        path
        for path in (run_dir / "results").glob("**/*.jsonl")
        if path.is_file() and path.stat().st_size > 0
    ]
    require(not offending, f"official result rows predate manifest freeze: {offending[:3]}")


def build(args: argparse.Namespace, created_utc: str | None = None) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    out = Path(args.out).resolve()
    require(out == run_dir / "manifest.json", "manifest must be RUN_DIR/manifest.json")
    require(IDENTIFIER.fullmatch(args.config) is not None, "invalid config id")
    plan_path, a8_path = Path(args.plan).resolve(), Path(args.a8).resolve()
    quantization = validate_plan_and_a8(plan_path, a8_path, args.expected_wrapped)
    servers = [
        parse_server(
            spec,
            config=args.config,
            plan_sha=quantization["plan"]["sha256"],
            a8_sha=quantization["a8"]["sha256"],
            expected_wrapped=args.expected_wrapped,
        )
        for spec in args.server
    ]
    workers = [parse_worker(spec, args.config) for spec in args.worker]
    task_sets = task_sets_for_scope(args.scope)
    schedule = validate_schedule(servers, workers, task_sets)
    references = [parse_reference(spec) for spec in args.reference]
    require(len({row["name"] for row in references}) == len(references), "duplicate reference")
    protocol = {
        "benchmark": "RoboCasa365",
        "scope": args.scope,
        "split": "target",
        "task_sets": task_sets,
        "task_counts": {name: len(tasks) for name, tasks in task_sets.items()},
        "trial_seeds": list(range(50)),
        "fresh_environment_per_trial": True,
        "render_enabled": True,
        "official_task_horizon": True,
        "action_horizon": 50,
        "n_action_steps": 16,
        "replan_steps": 16,
        "flow_steps": 4,
        "paired_action_noise": True,
        "paired_action_noise_protocol": PAIRED_NOISE,
        "statistics": {
            "primary_metric": f"{sum(map(len, task_sets.values()))}-task macro success rate",
            "bootstrap_draws": 10_000,
            "bootstrap_unit": "task cluster",
            "paired_test": "task-level sign-flip",
            "multiplicity": "Holm over manifest references",
        },
    }
    return {
        "schema_version": 1,
        "kind": "pi05_gdsq_vla_week1_control",
        "immutable": True,
        "result_blind": True,
        "frozen_before_official_rollouts": True,
        "created_utc": created_utc or dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_id": args.config,
        "quantization": quantization,
        "protocol": protocol,
        "schedule": {
            "servers": len(servers),
            "workers": len(workers),
            "coverage": schedule,
            "resume_key": ["config", "task_set", "task", "seed"],
            "incomplete_trial_policy": "discard and reconstruct fresh environment",
        },
        "servers": servers,
        "workers": workers,
        "references": references,
        "preflight": validate_preflight(
            Path(args.preflight_results).resolve(), Path(args.preflight_audit).resolve(), args.config
        ),
        "artifacts": source_artifacts(Path(args.orchestrator).resolve()),
        "environment": environment_record(),
        "claim_policy": {
            "paper_result_requires_full_coverage": True,
            "paper_result_requires_manifest_hash_match": True,
            "superiority_requires_paired_ci_lower_above_zero": True,
            "incomplete_output_is_progress_only": True,
            "heldout_results_may_not_select_configuration": True,
        },
    }


def atomic_write(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()
    run_dir = Path(args.run_dir).resolve()
    if args.verify:
        require(out.is_file(), f"missing frozen manifest: {out}")
        existing = json.loads(out.read_text(encoding="utf-8"))
        rebuilt = build(args, existing.get("created_utc"))
        require(existing == rebuilt, "frozen week-1 manifest drift")
        status = "verified"
    else:
        require(not out.exists(), f"refusing to overwrite frozen manifest: {out}")
        ensure_result_blind_freeze(run_dir)
        payload = build(args)
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        status = "created"
    print(
        json.dumps(
            {
                "status": status,
                "manifest": str(out),
                "sha256": sha256_file(out),
                "scope": args.scope,
                "expected_episodes": json.loads(out.read_text(encoding="utf-8"))["schedule"][
                    "coverage"
                ]["expected_episode_keys"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
