#!/usr/bin/env python3
"""Run audited DA-PTQ smoke and formal RoboCasa365 Table-1 evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

import qvla_actquant_formal_pipeline as base  # noqa: E402


RUN = ROOT / "runs" / "daptq_table1"
FORMAL = RUN / "formal"
PROTOCOL_PATH = ROOT / "scripts" / "daptq_table1_protocol.json"
CONFIG_ID = "daptq_w4a8"
DEFAULT_ELIGIBLE_GPUS = (3, 4, 5, 6)
UNITS = {
    "gr00t_atomic_seen": {
        "model": "gr00t",
        "split": "atomic_seen",
        "frozen_model_unit": "atomic_seen",
        "checkpoint": ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000",
    },
    "gr00t_composite_seen": {
        "model": "gr00t",
        "split": "composite_seen",
        "frozen_model_unit": "composite_seen",
        "checkpoint": ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000",
    },
    "gr00t_composite_unseen": {
        "model": "gr00t",
        "split": "composite_unseen",
        "frozen_model_unit": "composite_unseen",
        "checkpoint": ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_unseen/checkpoint-60000",
    },
    "pi05_all_target": {
        "model": "pi05",
        "split": "all_target",
        "frozen_model_unit": "all_target",
        "checkpoint": ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch",
    },
}


def pack_record(unit: str) -> dict[str, Any]:
    config = UNITS[unit]
    manifest_path = RUN / "artifacts" / unit / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "kind": "daptq_robocasa365_pack_v1",
        "method": "DA-PTQ",
        "model_family": config["model"],
        "model_unit": config["frozen_model_unit"],
        "flow_steps": 4,
        "trajectory_count": 512,
        "test_results_used": False,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"DA-PTQ artifact identity mismatch {manifest_path}: {mismatches}")
    arrays_path = manifest_path.parent / manifest["arrays_file"]
    if not arrays_path.is_file() or base.sha256_file(arrays_path) != manifest["arrays_sha256"]:
        raise ValueError(f"DA-PTQ array bundle changed: {arrays_path}")
    static_bytes = int(manifest["storage"]["total_static_bytes"])
    baseline_bytes = int(manifest["storage"]["fp16_baseline_bytes"])
    return {
        "model_unit": unit,
        "runtime_path": str(manifest_path.resolve()),
        "runtime_sha256": base.sha256_file(manifest_path),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": manifest["arrays_sha256"],
        "pack_manifest": str(manifest_path.resolve()),
        "pack_manifest_sha256": base.sha256_file(manifest_path),
        "static_bytes": static_bytes,
        "fp16_baseline_bytes": baseline_bytes,
        "compression_ratio": baseline_bytes / static_bytes,
        "w4_layers": int(manifest["allocation"]["w4_layers"]),
        "bf16_layers": int(manifest["allocation"]["bf16_layers"]),
    }


def unit_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": f"daptq_{unit}",
            "method": "daptq",
            "config_id": CONFIG_ID,
            "unit": unit,
            "model": config["model"],
            "split": config["split"],
            "checkpoint": str(config["checkpoint"].resolve()),
            "artifact": pack_record(unit),
        }
        for unit, config in UNITS.items()
    ]


def clean_quant_env() -> dict[str, str]:
    env = dict(os.environ)
    prefixes = (
        "GR00T_DUQUANT_",
        "OPENPI_DUQUANT_",
        "GR00T_GPTQ",
        "OPENPI_OMEGA_",
        "GR00T_ATM_",
        "OPENPI_ATM_",
        "GR00T_OHB_",
        "OPENPI_OHB_",
        "OPENPI_RUNTIME_SELECTOR_",
        "QVLA_ACTQUANT_",
        "DAPTQ_",
    )
    for key in list(env):
        if key.startswith(prefixes) or key in {
            "OPENPI_ERRORFOLD_PATH",
            "QUANTVLA_ADAPTER_ONLY",
        }:
            env.pop(key, None)
    return env


def pi_checkpoint_sha256() -> str:
    provenance = json.loads((RUN / "provenance.json").read_text(encoding="utf-8"))
    files = provenance["checkpoints"]["pi05_all_target"]["files"]
    model_files = [row for row in files if row["relative_path"] == "model.safetensors"]
    if len(model_files) != 1:
        raise ValueError("cannot resolve the frozen pi0.5 model.safetensors identity")
    return str(model_files[0]["sha256"])


def server_command(
    spec: dict[str, Any], placement: dict[str, Any]
) -> tuple[list[str], dict[str, str]]:
    env = clean_quant_env()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(placement["gpu"]),
            "PYTHONUNBUFFERED": "1",
            "TORCHDYNAMO_DISABLE": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
            "DAPTQ_ENABLE": "1",
            "DAPTQ_MANIFEST": spec["artifact"]["runtime_path"],
            "DAPTQ_MANIFEST_SHA256": spec["artifact"]["runtime_sha256"],
        }
    )
    code_path = f"{ROOT / 'code'}:{ROOT / 'scripts' / 'tools'}"
    env["PYTHONPATH"] = f"{code_path}:{env.get('PYTHONPATH', '')}"
    if spec["model"] == "gr00t":
        env.update(
            {
                "GR00T_CONFIG_ID": CONFIG_ID,
                "GR00T_MODEL_DTYPE": "bfloat16",
                "GR00T_ATM_ENABLE": "0",
                "GR00T_OHB_ENABLE": "0",
            }
        )
        command = [
            base.GROOT_PY,
            str(ROOT / "scripts" / "inference_service.py"),
            "--server",
            "--model-path",
            spec["checkpoint"],
            "--data-config",
            "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig",
            "--embodiment-tag",
            "new_embodiment",
            "--port",
            str(placement["port"]),
            "--denoising-steps",
            "4",
        ]
    else:
        runtime_path = FORMAL / "control" / f"{placement['instance']}.runtime.json"
        env.update(
            {
                "OPENPI_MODEL_DTYPE": "bfloat16",
                "OPENPI_CHECKPOINT_SHA256": pi_checkpoint_sha256(),
                "OPENPI_FORMAL_MODE": "1",
                "OPENPI_FORMAL_FLOW_STEPS": "4",
                "OPENPI_FULL_CONTEXT_PROTOCOL": "1",
                "OPENPI_CONFIG_ID": CONFIG_ID,
                "OPENPI_RUNTIME_INFO_PATH": str(runtime_path),
            }
        )
        command = [
            base.OPENPI_PY,
            "scripts/serve_pi05_quant_policy.py",
            "--env",
            "ROBOCASA",
            "--port",
            str(placement["port"]),
            "--denoising-steps",
            "4",
            "policy:checkpoint",
            "--policy.config",
            "pi05_pretrain_human300",
            "--policy.dir",
            spec["checkpoint"],
        ]
    return command, env


def validate_runtime(spec: dict[str, Any], runtime: dict[str, Any]) -> None:
    section = runtime.get("openpi_runtime") if spec["model"] == "pi05" else runtime
    if not isinstance(section, dict):
        raise ValueError(f"missing runtime attestation for {spec['id']}")
    daptq = section.get("daptq") or {}
    required = {
        "enabled": True,
        "method": "DA-PTQ",
        "manifest_sha256": spec["artifact"]["runtime_sha256"],
        "arrays_sha256": spec["artifact"]["arrays_sha256"],
        "activation_bits": 8,
        "flow_steps": 4,
        "runtime_memory_claim_allowed": False,
        "latency_claim_allowed": False,
    }
    mismatches = {
        key: (daptq.get(key), value)
        for key, value in required.items()
        if daptq.get(key) != value
    }
    if mismatches:
        raise ValueError(f"DA-PTQ runtime mismatch for {spec['id']}: {mismatches}")
    contract = section.get("cross_model_quantization_contract") or section.get(
        "quantization_contract"
    ) or {}
    contract_required = {
        "logical_profile": CONFIG_ID,
        "activation_bits": 8,
        "denoising_steps": 4,
        "n_action_steps": 16,
        "replan_steps": 16,
    }
    contract_mismatches = {
        key: (contract.get(key), value)
        for key, value in contract_required.items()
        if contract.get(key) != value
    }
    if contract_mismatches:
        raise ValueError(
            f"DA-PTQ contract mismatch for {spec['id']}: {contract_mismatches}"
        )
    forbidden = {
        "qvla_actquant": (section.get("qvla_actquant") or {}).get("enabled"),
        "duquant": (section.get("duquant") or {}).get("enabled"),
        "omega": (section.get("omega_qvla") or {}).get("enabled"),
        "runtime_selector": (section.get("runtime_selector") or {}).get("enabled"),
        "atm_ohb": (section.get("atm_ohb") or {}).get("enabled"),
        "errorfold": (section.get("errorfold") or {}).get("enabled"),
        "gr00t_wrappers": int(section.get("wrapped_layers", 0)) != 0,
        "gr00t_atm": section.get("atm_enabled"),
        "gr00t_ohb": section.get("ohb_enabled"),
    }
    enabled = [key for key, value in forbidden.items() if value]
    if enabled:
        raise ValueError(f"mixed quantizers enabled for {spec['id']}: {enabled}")
    protocol = section.get("protocol") or {}
    protocol_required = {
        "flow_steps": 4,
        "n_action_steps": 16,
        "replan_steps": 16,
        "split": "target",
        "render": True,
    }
    protocol_mismatches = {
        key: (protocol.get(key), value)
        for key, value in protocol_required.items()
        if protocol.get(key) != value
    }
    if protocol_mismatches:
        raise ValueError(
            f"runtime protocol mismatch for {spec['id']}: {protocol_mismatches}"
        )
    if spec["model"] == "pi05":
        stable = json.loads(json.dumps(runtime, sort_keys=True, default=str))
        stable_section = stable.get("openpi_runtime") or {}
        claimed = stable_section.pop("semantic_metadata_sha256", None)
        stable_section.pop("gpu_memory_bytes", None)
        if claimed != base.canonical_hash(stable):
            raise ValueError(f"pi0.5 semantic metadata SHA mismatch for {spec['id']}")


def choose_smoke_tasks(spec: dict[str, Any]) -> list[str]:
    splits, _ = base.load_tasks()
    if spec["model"] == "gr00t":
        return splits[spec["split"]][:5]
    return [
        splits["atomic_seen"][0],
        splits["atomic_seen"][1],
        splits["composite_seen"][0],
        splits["composite_seen"][1],
        splits["composite_unseen"][0],
    ]


def make_jobs(specs: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    _splits, task_to_split = base.load_tasks()
    jobs: list[dict[str, Any]] = []
    if stage == "smoke1":
        for spec in specs:
            task = choose_smoke_tasks(spec)[0]
            jobs.append(
                {
                    "id": f"smoke1__{spec['id']}",
                    "unit_id": spec["id"],
                    "model": spec["model"],
                    "method": "daptq",
                    "split": task_to_split[task],
                    "task": task,
                    "seeds": [100],
                }
            )
    elif stage == "smoke50":
        for spec in specs:
            for index, task in enumerate(choose_smoke_tasks(spec)):
                jobs.append(
                    {
                        "id": f"smoke50__{spec['id']}__{index:02d}",
                        "unit_id": spec["id"],
                        "model": spec["model"],
                        "method": "daptq",
                        "split": task_to_split[task],
                        "task": task,
                        "seeds": list(range(100, 110)),
                    }
                )
    elif stage == "formal":
        split_unit = {
            "atomic_seen": "gr00t_atomic_seen",
            "composite_seen": "gr00t_composite_seen",
            "composite_unseen": "gr00t_composite_unseen",
        }
        for task, split in sorted(task_to_split.items(), key=lambda item: (item[1], item[0])):
            for model in ("gr00t", "pi05"):
                unit = split_unit[split] if model == "gr00t" else "pi05_all_target"
                for seed_start in range(0, 50, 10):
                    jobs.append(
                        {
                            "id": f"formal__daptq__{model}__{split}__{task}__s{seed_start:02d}",
                            "unit_id": f"daptq_{unit}",
                            "model": model,
                            "method": "daptq",
                            "split": split,
                            "task": task,
                            "seeds": list(range(seed_start, seed_start + 10)),
                        }
                    )
    else:
        raise ValueError(stage)
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError(f"duplicate {stage} job IDs")
    if stage == "formal":
        coverage: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
        for job in jobs:
            for seed in job["seeds"]:
                key = (job["split"], job["task"], seed)
                if key in coverage[job["model"]]:
                    raise ValueError(f"duplicate formal key: {job['model']} {key}")
                coverage[job["model"]].add(key)
        if set(coverage) != {"gr00t", "pi05"} or any(
            len(keys) != 2500 for keys in coverage.values()
        ):
            raise ValueError("DA-PTQ formal schedule is not two complete 2,500-episode rows")
    return jobs


def artifact_for_arm(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "model_unit",
            "runtime_path",
            "runtime_sha256",
            "arrays_path",
            "arrays_sha256",
            "pack_manifest",
            "pack_manifest_sha256",
            "static_bytes",
            "fp16_baseline_bytes",
            "compression_ratio",
            "w4_layers",
            "bf16_layers",
        )
    }


def make_arm_manifests(
    specs: list[dict[str, Any]], servers: dict[str, list[dict[str, Any]]]
) -> list[Path]:
    outputs = []
    for model in ("gr00t", "pi05"):
        selected = [spec for spec in specs if spec["model"] == model]
        artifacts = [artifact_for_arm(spec["artifact"]) for spec in selected]
        static_bytes = sum(row["static_bytes"] for row in artifacts)
        baseline_bytes = sum(row["fp16_baseline_bytes"] for row in artifacts)
        value = {
            "schema_version": 1,
            "kind": "daptq_formal_arm",
            "immutable": True,
            "method": "daptq",
            "model": model,
            "display_name": "DA-PTQ W4A8",
            "protocol": str(PROTOCOL_PATH.resolve()),
            "protocol_sha256": base.sha256_file(PROTOCOL_PATH),
            "source_provenance": str((RUN / "provenance.json").resolve()),
            "source_provenance_sha256": base.sha256_file(RUN / "provenance.json"),
            "flow_steps": 4,
            "formal_seeds": list(range(50)),
            "episodes": 2500,
            "test_feedback_allowed": False,
            "artifacts": artifacts,
            "allowed_server_metadata_sha256": sorted(
                {
                    row["server_metadata_sha256"]
                    for spec in selected
                    for row in servers[spec["id"]]
                }
            ),
            "result_globs": [
                str((FORMAL / "raw" / "formal" / "daptq" / model / "*.jsonl").resolve())
            ],
            "storage": {
                "static_bytes_total_deployed_checkpoint_set": static_bytes,
                "static_gib_total_deployed_checkpoint_set": static_bytes / 1024**3,
                "fp16_baseline_bytes_total_deployed_checkpoint_set": baseline_bytes,
                "compression_ratio_total_deployed_checkpoint_set": baseline_bytes / static_bytes,
                "checkpoint_count": len(artifacts),
                "scope": (
                    "sum across three split-specific checkpoints"
                    if model == "gr00t"
                    else "single global checkpoint"
                ),
            },
        }
        path = FORMAL / "arm_manifests" / f"daptq_{model}.json"
        base.immutable_json(path, value)
        outputs.append(path)
    return outputs


def make_server_schedule(
    specs: list[dict[str, Any]], replicas: int, eligible_gpus: tuple[int, ...]
) -> dict[str, Any]:
    gpu_label = "-".join(map(str, eligible_gpus))
    path = FORMAL / f"server_schedule.gpus-{gpu_label}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing audited DA-PTQ GPU migration schedule: {path}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_artifacts = {
        spec["id"]: spec["artifact"]["runtime_sha256"] for spec in specs
    }
    if (
        value.get("artifact_sha256") != expected_artifacts
        or value.get("replicas") != replicas
        or value.get("eligible_gpus") != list(eligible_gpus)
    ):
        raise ValueError("migrated DA-PTQ server schedule does not match frozen packs")
    disallowed = sorted(
        {
            int(placement["gpu"])
            for placement in value.get("placements", [])
            if int(placement["gpu"]) not in eligible_gpus
        }
    )
    if disallowed:
        raise ValueError(f"DA-PTQ server schedule uses disallowed GPUs: {disallowed}")
    return value


def wait_for_schedulable(
    specs: list[dict[str, Any]],
    replicas: int,
    poll_seconds: int,
    eligible_gpus: tuple[int, ...],
) -> dict:
    while True:
        try:
            return make_server_schedule(specs, replicas, eligible_gpus)
        except RuntimeError as error:
            base.atomic_json(
                FORMAL / "control" / "resource_wait.json",
                {
                    "updated_at": base.now(),
                    "stage": "waiting_for_server_vram",
                    "error": str(error),
                    "free_mib": base.gpu_free_mib(eligible_gpus),
                },
            )
            time.sleep(max(15, poll_seconds))


def aggregate(arm_manifests: list[Path]) -> None:
    command = [
        base.GROOT_PY,
        str(ROOT / "scripts" / "tools" / "daptq_formal_results.py"),
        "--gr00t-arm",
        str(next(path for path in arm_manifests if path.name == "daptq_gr00t.json")),
        "--pi05-arm",
        str(next(path for path in arm_manifests if path.name == "daptq_pi05.json")),
        "--out",
        str(FORMAL / "aggregate.json"),
    ]
    base.run_checked(command, FORMAL / "control" / "aggregate.log")


def configure_base() -> None:
    base.RUN = RUN
    base.FORMAL = FORMAL
    base.PROTOCOL_PATH = PROTOCOL_PATH
    base.METHOD_CONFIG = {"daptq": CONFIG_ID}
    base.UNITS = UNITS
    base.server_command = server_command
    base.validate_runtime = validate_runtime
    base.choose_smoke_tasks = choose_smoke_tasks
    base.make_jobs = make_jobs


def record_smoke50_skip(specs: list[dict[str, Any]]) -> None:
    """Record an explicit, non-completion decision for the diagnostic smoke stage."""
    jobs = make_jobs(specs, "smoke50")
    receipts_dir = FORMAL / "control" / "receipts" / "smoke50"
    output_root = FORMAL / "smoke" / "smoke50"
    receipt_paths = sorted(receipts_dir.glob("*.json"))
    completed_rows = 0
    for path in receipt_paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete":
            raise ValueError(f"non-complete smoke50 receipt: {path}")
        completed_rows += int(receipt["rows"])

    diagnostic_rows = 0
    invalid_lines: list[str] = []
    output_paths = sorted(output_root.rglob("*.jsonl")) if output_root.is_dir() else []
    for path in output_paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines.append(f"{path}:{line_number}")
                continue
            if row.get("status") == "complete" and int(row.get("flow_steps", -1)) == 4:
                diagnostic_rows += 1

    failed_markers = sorted(
        str(path)
        for path in (FORMAL / "control").rglob("*.failed.json")
        if path.is_file()
    )
    if failed_markers:
        raise RuntimeError(f"cannot skip a failing smoke stage: {failed_markers}")

    record = {
        "schema_version": 1,
        "kind": "daptq_smoke_stage_skip",
        "recorded_at": base.now(),
        "stage": "smoke50",
        "decision": "skipped_by_user_after_smoke1_passed",
        "is_stage_complete": False,
        "excluded_from_formal_aggregate": True,
        "planned_jobs": len(jobs),
        "completed_job_receipts": len(receipt_paths),
        "validated_rows_in_completed_jobs": completed_rows,
        "retained_diagnostic_output_files": len(output_paths),
        "retained_parseable_4_flow_step_rows": diagnostic_rows,
        "invalid_or_interrupted_lines": invalid_lines,
        "failed_markers": failed_markers,
        "smoke1_status": json.loads(
            (FORMAL / "control" / "smoke1_status.json").read_text(encoding="utf-8")
        ),
        "orchestrator": str(Path(__file__).resolve()),
        "orchestrator_sha256": base.sha256_file(Path(__file__).resolve()),
    }
    base.atomic_json(FORMAL / "control" / "smoke50_skip.json", record)
    base.atomic_json(
        FORMAL / "control" / "smoke50_status.json",
        {
            "updated_at": base.now(),
            "stage": "smoke50",
            "state": "skipped",
            "pending": 0,
            "running": 0,
            "completed": len(receipt_paths),
            "planned": len(jobs),
            "is_stage_complete": False,
            "skip_record": str(FORMAL / "control" / "smoke50_skip.json"),
        },
    )
    base.append_event(
        {
            "event": "rollout_stage_skipped",
            "stage": "smoke50",
            "completed_job_receipts": len(receipt_paths),
            "retained_diagnostic_rows": diagnostic_rows,
            "reason": record["decision"],
        }
    )


def execute(
    max_workers: int,
    workers_per_gpu: int,
    replicas: int,
    poll_seconds: int,
    eligible_gpus: tuple[int, ...],
    *,
    skip_smoke50: bool = False,
) -> None:
    configure_base()
    lock_path = FORMAL / "control" / "pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("DA-PTQ formal pipeline is already running") from error
        specs = unit_specs()
        schedule = wait_for_schedulable(
            specs, replicas, poll_seconds, eligible_gpus
        )
        servers: dict[str, list[dict[str, Any]]] = {}
        try:
            servers = base.start_servers(specs, schedule)
            base.run_jobs(
                specs,
                servers,
                "smoke1",
                max_workers=min(max_workers, 8),
                workers_per_gpu=workers_per_gpu,
                eligible_gpus=eligible_gpus,
            )
            if skip_smoke50:
                record_smoke50_skip(specs)
            else:
                base.run_jobs(
                    specs,
                    servers,
                    "smoke50",
                    max_workers=max_workers,
                    workers_per_gpu=workers_per_gpu,
                    eligible_gpus=eligible_gpus,
                )
            arms = make_arm_manifests(specs, servers)
            base.run_jobs(
                specs,
                servers,
                "formal",
                max_workers=max_workers,
                workers_per_gpu=workers_per_gpu,
                eligible_gpus=eligible_gpus,
            )
            aggregate(arms)
            base.atomic_json(
                FORMAL / "complete.json",
                {
                    "completed_at": base.now(),
                    "status": "complete",
                    "new_formal_episodes": 5000,
                    "aggregate": str(FORMAL / "aggregate.json"),
                    "aggregate_sha256": base.sha256_file(FORMAL / "aggregate.json"),
                },
            )
        finally:
            if servers:
                base.stop_servers(schedule)


def status() -> dict[str, Any]:
    result = {}
    for stage in ("smoke1", "smoke50", "formal"):
        path = FORMAL / "control" / f"{stage}_status.json"
        result[stage] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    complete = FORMAL / "complete.json"
    result["complete"] = (
        json.loads(complete.read_text(encoding="utf-8")) if complete.is_file() else None
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "stop-servers"))
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--eligible-gpus",
        default=",".join(map(str, DEFAULT_ELIGIBLE_GPUS)),
        help="comma-separated physical GPU allowlist; hard-limited to 3,4,5,6",
    )
    parser.add_argument(
        "--skip-smoke50",
        action="store_true",
        help="reuse the completed smoke1 audit, retain partial smoke50 diagnostics, and run formal",
    )
    args = parser.parse_args()
    configure_base()
    if args.command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
        return
    if args.command == "stop-servers":
        for path in sorted(FORMAL.glob("server_schedule*.json")):
            base.stop_servers(json.loads(path.read_text(encoding="utf-8")))
        return
    if args.replicas != 2:
        raise ValueError("the frozen DA-PTQ formal protocol requires two replicas")
    if args.max_workers < 1 or args.workers_per_gpu < 1:
        raise ValueError("worker concurrency must be positive")
    eligible_gpus = base.parse_gpu_list(args.eligible_gpus)
    execute(
        args.max_workers,
        args.workers_per_gpu,
        args.replicas,
        args.poll_seconds,
        eligible_gpus,
        skip_smoke50=args.skip_smoke50,
    )


if __name__ == "__main__":
    main()
