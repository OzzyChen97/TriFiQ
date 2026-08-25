#!/usr/bin/env python3
"""Create or verify the immutable pi0.5 v8-selector official-run manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = "gdsq_vla_runtime_selector"
SELECTOR_SHA256 = "0f3178726c2b784898f18dfde248d9a9bae152da6ffcfdc299e8bdc02f0bd871"
RULE_NAME = "v8_no_oracle_absolute_mechanism_gate_aligned_runtime_rule"
EXPECTED_VARIANT = "ohb"
EXPECTED_SELECTED_CONFIG = "gdsq_vla_ohb_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--orchestrator", required=True)
    parser.add_argument(
        "--server",
        action="append",
        required=True,
        help="INSTANCE,CONFIG,GPU,PORT,RUNTIME_JSON",
    )
    parser.add_argument(
        "--worker",
        action="append",
        required=True,
        help="WORKER_ID,SERVER_INSTANCE,SHARD_INDEX,SHARD_COUNT,SEED_START-SEED_END",
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
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def configured_path(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable, str(REPO_ROOT / default))).expanduser().resolve()


def parse_server(spec: str) -> dict[str, Any]:
    instance, config, gpu_text, port_text, runtime_text = spec.split(",", 4)
    require(config == CONFIG, f"invalid selector server config: {config}")
    require(instance and instance.replace("_", "").isalnum(), f"invalid instance: {instance}")
    gpu, port = int(gpu_text), int(port_text)
    require(0 <= gpu <= 7 and 1024 <= port <= 65535, f"invalid server resource: {spec}")
    runtime_path = Path(runtime_text).expanduser().resolve()
    runtime_metadata = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime = runtime_metadata.get("openpi_runtime") or {}
    require(runtime.get("config_id") == CONFIG, f"runtime/config mismatch: {instance}")
    selector = runtime.get("runtime_selector") or {}
    selector_checks = {
        "enabled": selector.get("enabled") is True,
        "sha256": selector.get("selector_sha256") == SELECTOR_SHA256,
        "rule": selector.get("rule_name") == RULE_NAME,
        "model": selector.get("model_id") == "pi05",
        "scope": selector.get("selection_scope") == "model_level_absolute_mechanism_gate",
        "task_independent": selector.get("uses_task_metadata_for_selection") is False,
        "variant": (selector.get("model_decision") or {}).get("selected_variant")
        == EXPECTED_VARIANT,
        "selected_config": (selector.get("model_decision") or {}).get("selected_config_id")
        == EXPECTED_SELECTED_CONFIG,
    }
    failed = [name for name, valid in selector_checks.items() if not valid]
    require(not failed, f"runtime selector checks failed for {instance}: {failed}")
    protocol = runtime.get("protocol") or {}
    expected_protocol = {
        "action_horizon": 50,
        "n_action_steps": 16,
        "replan_steps": 16,
        "flow_steps": 4,
        "split": "target",
        "fresh_environment_per_episode": True,
        "official_task_horizon": True,
        "render": True,
        "paired_noise": "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1",
    }
    mismatches = {
        name: (protocol.get(name), expected)
        for name, expected in expected_protocol.items()
        if protocol.get(name) != expected
    }
    require(not mismatches, f"runtime protocol mismatch for {instance}: {mismatches}")
    duquant = runtime.get("duquant") or {}
    require(duquant.get("wrapped_layers") == 80, f"wrong wrapped-layer count: {instance}")
    require(duquant.get("denoising_steps") == 4, f"wrong denoising steps: {instance}")
    require(duquant.get("act_bits") == 8, f"wrong activation bits: {instance}")
    return {
        "instance": instance,
        "config_id": config,
        "gpu": gpu,
        "port": port,
        "runtime_path": str(runtime_path),
        "runtime_file_sha256": sha256_file(runtime_path),
        "server_metadata_sha256": canonical_hash(runtime_metadata),
        "runtime": runtime,
    }


def parse_worker(spec: str) -> dict[str, Any]:
    worker_id, instance, shard_text, count_text, seed_text = spec.split(",", 4)
    shard, count = int(shard_text), int(count_text)
    seed_start_text, seed_end_text = seed_text.split("-", 1)
    seed_start, seed_end = int(seed_start_text), int(seed_end_text)
    require(worker_id and worker_id.replace("_", "").isalnum(), f"invalid worker: {worker_id}")
    require(count > 0 and 0 <= shard < count, f"invalid task shard: {spec}")
    require(0 <= seed_start <= seed_end <= 49, f"invalid seed range: {spec}")
    return {
        "worker_id": worker_id,
        "config_id": CONFIG,
        "server_instance": instance,
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
    require(len(servers) == 5, "formal selector schedule requires five server replicas")
    require(len({row["instance"] for row in servers}) == len(servers), "duplicate server instance")
    require(len({row["gpu"] for row in servers}) == len(servers), "duplicate server GPU")
    require(len({row["port"] for row in servers}) == len(servers), "duplicate server port")
    instances = {row["instance"]: row for row in servers}
    require(len({row["worker_id"] for row in workers}) == len(workers), "duplicate worker id")
    for worker in workers:
        require(worker["server_instance"] in instances, f"unknown worker server: {worker}")

    expected = {
        (task_set, task, seed)
        for task_set, tasks in task_sets.items()
        for task in tasks
        for seed in range(50)
    }
    covered: set[tuple[str, str, int]] = set()
    duplicates = 0
    for worker in workers:
        shard = worker["task_shard_index"]
        count = worker["task_shard_count"]
        for task_set, tasks in task_sets.items():
            selected = [task for index, task in enumerate(tasks) if index % count == shard]
            for key in (
                (task_set, task, seed)
                for task in selected
                for seed in range(worker["trial_seed_start"], worker["trial_seed_end"] + 1)
            ):
                duplicates += int(key in covered)
                covered.add(key)
    require(duplicates == 0, f"worker schedule duplicates {duplicates} episode keys")
    require(covered == expected, f"worker coverage mismatch: missing={len(expected-covered)} extra={len(covered-expected)}")
    return {
        "expected_episode_keys": len(expected),
        "covered_episode_keys": len(covered),
        "duplicate_episode_keys": duplicates,
        "keyset_sha256": canonical_hash(sorted(covered)),
    }


def validate_selector(selector_path: Path) -> dict[str, Any]:
    require(sha256_file(selector_path) == SELECTOR_SHA256, "frozen selector SHA drift")
    payload = json.loads(selector_path.read_text(encoding="utf-8"))
    require(payload.get("rule_name") == RULE_NAME, "frozen selector rule drift")
    model = (payload.get("models") or {}).get("pi05") or {}
    fit = model.get("fit") or {}
    selected = set((model.get("selected") or {}).values())
    checks = {
        "selected_ohb": selected == {EXPECTED_VARIANT},
        "scope": fit.get("selection_scope") == "model_level_absolute_mechanism_gate",
        "no_task_metadata": fit.get("task_metadata_used_for_variant_selection") is False,
        "no_rollout_labels": fit.get("rollout_labels_used") is False,
        "no_runtime_feedback": fit.get("runtime_success_feedback_used") is False,
        "correction_admitted": fit.get("correction_admitted") is True,
        "candidate_ohb": fit.get("candidate_variant") == EXPECTED_VARIANT,
        "no_combined": (fit.get("mechanism_profile") or {}).get("combined_atmohb_allowed")
        is False,
    }
    failed = [name for name, valid in checks.items() if not valid]
    require(not failed, f"frozen selector semantic checks failed: {failed}")
    return {
        **artifact(selector_path),
        "rule_name": RULE_NAME,
        "selection_scope": "model_level_absolute_mechanism_gate",
        "model_id": "pi05",
        "selected_variant": EXPECTED_VARIANT,
        "selected_config_id": EXPECTED_SELECTED_CONFIG,
        "uses_task_metadata_for_selection": False,
        "rollout_labels_used": False,
        "runtime_success_feedback_used": False,
        "atmohb_output_allowed": False,
    }


def environment_record() -> dict[str, Any]:
    frozen = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
    )
    normalized = "\n".join(sorted(line.strip() for line in frozen.splitlines() if line.strip())) + "\n"
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pip_freeze_sha256": sha256_text(normalized),
        "pip_freeze_entries": len(normalized.splitlines()),
    }


def source_and_data_artifacts(orchestrator: Path) -> dict[str, Any]:
    paths = {
        "checkpoint": configured_path(
            "PI05_CHECKPOINT",
            "checkpoints/robocasa/pi05_pretrain_human300_pytorch/model.safetensors",
        ),
        "checkpoint_config": REPO_ROOT
        / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/config.json",
        "norm_stats": REPO_ROOT
        / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/assets/pi05_pretrain_human300/norm_stats.json",
        "pack_manifest": configured_path(
            "PI05_PACK_MANIFEST",
            "runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015/manifest.json",
        ),
        "calibration_buffer": configured_path(
            "PI05_CALIBRATION_BUFFER",
            "runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz",
        ),
        "gdsq_plan": configured_path(
            "PI05_GDSQ_PLAN",
            "runs/pi05_gdsq_gr00t_aligned/plans/pi05_gdsq_cscka_16to1_d4.final_plan.json",
        ),
        "gdsq_a8": configured_path(
            "PI05_GDSQ_A8",
            "runs/pi05_gdsq_gr00t_aligned/a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz",
        ),
        "gdsq_a8_sidecar": Path(
            str(
                configured_path(
                    "PI05_GDSQ_A8",
                    "runs/pi05_gdsq_gr00t_aligned/a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz",
                )
            )
            + ".json"
        ),
        "gdsq_atm_ohb": configured_path(
            "PI05_GDSQ_ATM",
            "runs/pi05_gdsq_gr00t_aligned/atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json",
        ),
        "preregistration": REPO_ROOT / "runs/gdsq_week1_preregistered_v1/preregistration.json",
        "equivalence_manifest": REPO_ROOT
        / "runs/gdsq_week1_preregistered_v1/selector_equivalence/manifest.json",
        "equivalence_summary": REPO_ROOT
        / "runs/gdsq_week1_preregistered_v1/selector_equivalence/summary.json",
        "server_launcher": REPO_ROOT / "scripts/run_pi05_formal_server.sh",
        "worker_launcher": REPO_ROOT / "scripts/run_pi05_formal_worker_seeded.sh",
        "evaluator": REPO_ROOT / "scripts/run_robocasa365_pi05_eval.py",
        "aggregator": REPO_ROOT / "scripts/tools/aggregate_pi05_selector_official.py",
        "gpu_monitor": REPO_ROOT / "scripts/tools/monitor_pi05_formal_gpu.py",
        "statistics": REPO_ROOT / "scripts/tools/parse_robocasa_atomic_matrix.py",
        "selector_runtime": REPO_ROOT
        / "code/pi05/openpi/src/openpi/quant/atm_runtime_selector.py",
        "manifest_tool": Path(__file__).resolve(),
        "preflight_auditor": REPO_ROOT / "scripts/tools/audit_pi05_selector_preflight.py",
        "orchestrator": orchestrator,
    }
    records = {name: artifact(path) for name, path in paths.items()}
    require(
        records["checkpoint"]["sha256"]
        == "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c",
        "checkpoint SHA drift",
    )
    require(
        records["equivalence_summary"]["sha256"]
        == "753665940cc4f9af40140d4604341deaedbcb6ab0f12347320abbcf34032115d",
        "selector equivalence gate drift",
    )
    return records


def static_reference() -> dict[str, Any]:
    records = {
        "manifest": artifact(
            REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/official_target_paired50/manifest.json"
        ),
        "summary": artifact(
            REPO_ROOT
            / "runs/pi05_gdsq_gr00t_aligned/official_target_paired50/aggregate/summary.json"
        ),
        "replay_audit": artifact(
            REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/audit/replay_audit_v2.json"
        ),
    }
    expected = {
        "manifest": "84f9e17ec6186242a52364ea3c565470d22180a27c68c65569ddd6361ffca93f",
        "summary": "c7b6441c6ca0a49aa5cfe8e673d1ee465a5d101076b6ad866b149fb545d10cbb",
        "replay_audit": "51eaad24453cbf5e7778f0f1304b2cd409f41614ba41edac3c2b57c71483f004",
    }
    for name, value in expected.items():
        require(records[name]["sha256"] == value, f"static {name} SHA drift")
    replay = json.loads(Path(records["replay_audit"]["path"]).read_text(encoding="utf-8"))
    require(replay.get("valid") is True, "static reference replay audit is invalid")
    return records


def build(args: argparse.Namespace, existing_created_utc: str | None = None) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    require(out == run_dir / "manifest.json", "manifest must be RUN_DIR/manifest.json")
    orchestrator = Path(args.orchestrator).expanduser().resolve()
    base_manifest_path = REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/official_target_paired50/manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    task_sets = base_manifest["table_1_protocol"]["task_sets"]
    servers = [parse_server(spec) for spec in args.server]
    workers = [parse_worker(spec) for spec in args.worker]
    schedule_coverage = validate_schedule(servers, workers, task_sets)
    selector_path = Path(
        os.environ.get(
            "PI05_RUNTIME_SELECTOR",
            str(REPO_ROOT / "runs/atmohb_dynamic_selector_v8/selector.json"),
        )
    ).resolve()
    payload = {
        "schema_version": 1,
        "kind": "pi05_gdsq_vla_v8_selector_official50",
        "immutable": True,
        "result_blind": True,
        "frozen_before_official_rollouts": True,
        "created_utc": existing_created_utc
        or dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_id": CONFIG,
        "selector": validate_selector(selector_path),
        "protocol": {
            "benchmark": "RoboCasa365",
            "split": "target",
            "task_sets": task_sets,
            "task_counts": {name: len(values) for name, values in task_sets.items()},
            "trial_seeds": list(range(50)),
            "fresh_environment_per_trial": True,
            "render_enabled": True,
            "official_task_horizon": True,
            "action_horizon": 50,
            "n_action_steps": 16,
            "replan_steps": 16,
            "flow_steps": 4,
            "paired_action_noise": True,
            "paired_action_noise_protocol": base_manifest["table_1_protocol"][
                "paired_action_noise_protocol"
            ],
            "statistics": {
                "primary_metric": "50-task macro success rate",
                "bootstrap_draws": 10_000,
                "bootstrap_unit": "task cluster",
                "paired_test": "task-level sign-flip",
                "multiplicity": "Holm over preregistered contrasts",
            },
            "contrasts": [
                [CONFIG, value]
                for value in (
                    "fp16",
                    "gdsq_vla",
                    "gdsq_vla_atmohb",
                    "quantvla_w4a8_atmohb",
                )
            ],
        },
        "schedule": {
            "servers": len(servers),
            "workers": len(workers),
            "coverage": schedule_coverage,
            "resume_key": ["config", "task_set", "task", "seed"],
            "incomplete_trial_policy": "discard and reconstruct fresh environment",
        },
        "servers": servers,
        "workers": workers,
        "artifacts": source_and_data_artifacts(orchestrator),
        "preflight": {
            "results": artifact(run_dir / "preflight/pi05_selector_2task2seed.jsonl"),
            "audit": artifact(run_dir / "preflight/summary.json"),
            "diagnostic_only": True,
            "paper_claim_enabled": False,
        },
        "static_reference": static_reference(),
        "environment": environment_record(),
        "claim_policy": {
            "paper_result_requires_full_coverage": True,
            "paper_result_requires_selector_attestation_on_every_row": True,
            "superiority_requires_ci_excludes_zero_against_strongest_same_budget_baseline": True,
            "same_budget_p0_currently_pending": True,
        },
    }
    return payload


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main() -> None:
    args = parse_args()
    output = Path(args.out).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.verify:
        existing = json.loads(output.read_text(encoding="utf-8"))
        candidate = build(args, str(existing["created_utc"]))
        require(canonical_hash(candidate) == canonical_hash(existing), "immutable manifest drift")
        print(f"selector manifest verified unchanged: {output}")
        print(f"selector manifest sha256: {sha256_file(output)}")
        return
    require(not output.exists(), f"refusing to replace immutable manifest: {output}")
    official_rows = list((run_dir / "results" / CONFIG).glob("*.jsonl"))
    require(
        not any(path.stat().st_size for path in official_rows),
        "official result rows exist before manifest freeze",
    )
    payload = build(args)
    atomic_write(output, payload)
    print(f"selector manifest created: {output}")
    print(f"selector manifest sha256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
