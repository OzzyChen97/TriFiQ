#!/usr/bin/env python3
"""Freeze, audit, and aggregate the pi0.5 Omega-QVLA RoboCasa365 run."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

with contextlib.redirect_stdout(sys.stderr):
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon


REPO = Path(__file__).resolve().parents[2]
CONFIG = "omega_qvla_w4a4"
TASK_SETS = ("atomic_seen", "composite_seen", "composite_unseen")
NOISE_PROTOCOL = "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"missing artifact: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def validate_runtime(path: Path, task_set: str) -> dict[str, Any]:
    metadata = json.loads(path.read_text())
    runtime = metadata.get("openpi_runtime") or {}
    omega = runtime.get("omega_qvla") or {}
    checks = {
        "config": runtime.get("config_id") == CONFIG,
        "adapter": (runtime.get("model_adapter") or {}).get("model") == "pi05",
        "omega": omega.get("enabled") is True,
        "task_set": omega.get("task_set") == task_set,
        "wrapped": omega.get("wrapped_layers") == 252,
        "records": omega.get("pack_records") == 252,
        "w4": omega.get("weight_bits") == 4,
        "a4": omega.get("activation_bits") == 4,
        "flow": omega.get("denoising_steps") == 4,
        "execute": omega.get("execute_steps") == 16,
        "no_test_feedback": omega.get("test_results_used_for_calibration") is False,
        "selector_off": not bool((runtime.get("runtime_selector") or {}).get("enabled")),
        "duquant_off": not bool((runtime.get("duquant") or {}).get("enabled")),
    }
    failed = [key for key, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"runtime attestation failed {path}: {failed}")
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "server_metadata_sha256": canonical_hash(metadata),
        "omega_qvla": omega,
        "protocol": runtime.get("protocol"),
        "model_adapter": runtime.get("model_adapter"),
    }


def freeze(args: argparse.Namespace) -> None:
    task_set = args.task_set
    run_dir = Path(args.run_dir).resolve()
    pack = Path(args.pack).resolve()
    calibration = Path(args.calibration).resolve()
    preflight = Path(args.preflight).resolve()
    control = Path(args.control_dir).resolve()
    runtime_paths = sorted(control.glob("*.runtime.json"))
    if len(runtime_paths) != 6:
        raise ValueError(f"expected six runtime attestations, found {len(runtime_paths)}")
    servers = [validate_runtime(path, task_set) for path in runtime_paths]
    manifest = {
        "schema_version": 1,
        "kind": "omega_qvla_pi05_robocasa365_formal_manifest",
        "config_id": CONFIG,
        "task_set": task_set,
        "tasks": list(TASK_SET_REGISTRY[task_set]),
        "seeds": list(range(50)),
        "protocol": {
            "split": "target",
            "paired_action_noise": True,
            "paired_action_noise_protocol": NOISE_PROTOCOL,
            "denoising_steps": 4,
            "execute_steps": 16,
            "fresh_environment_per_episode": True,
            "official_task_horizon": True,
        },
        "pack": artifact(pack),
        "pack_attestation": artifact(pack.with_suffix(".attestation.json")),
        "calibration_manifest": artifact(calibration),
        "checkpoint": artifact(
            REPO / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/model.safetensors"
        ),
        "preflight": artifact(preflight),
        "servers": servers,
        "workers": [
            {
                "id": f"{task_set}_g{gpu}_{half}",
                "gpu": gpu,
                "task_shard_index": shard,
                "task_shard_count": 6,
                "seeds": seed_range,
            }
            for shard, gpu in enumerate((1, 2, 4, 5, 6, 7))
            for half, seed_range in (("lo", [0, 24]), ("hi", [25, 49]))
        ],
        "gr00t_alignment": {
            "reference_manifest": artifact(
                REPO
                / "runs/gdsq_extension_preregistered_v1/omega_qvla_robocasa365_v1"
                / "calibration"
                / f"{task_set}.manifest.v3.json"
            ),
            "matched_protocol_fields": [
                "task_set", "tasks", "seeds", "target_split", "paired_action_noise",
                "denoising_steps", "execute_steps", "calibration_samples",
                "calibration_seed", "W4A4", "llm_gptq", "action_rtn_perstep",
            ],
            "only_model_adapter_differs": True,
        },
        "source": {
            str(path.resolve()): sha256_file(path)
            for path in (
                Path(__file__),
                REPO / "scripts/run_omega_qvla_pi05_robocasa365.sh",
                REPO / "scripts/run_pi05_formal_server.sh",
                REPO / "scripts/run_pi05_formal_worker_seeded.sh",
                REPO / "scripts/run_robocasa365_pi05_eval.py",
                REPO / "code/pi05/openpi/scripts/serve_pi05_quant_policy.py",
            )
        },
    }
    manifest["manifest_payload_sha256"] = canonical_hash(manifest)
    path = run_dir / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != manifest:
            raise ValueError(f"immutable formal manifest mismatch: {path}")
    else:
        atomic_json(path, manifest)
    print(path)


def load_rows(run_dir: Path, *, require_complete: bool) -> tuple[dict, dict, list[str]]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_payload_sha256") != canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    ):
        raise ValueError(f"manifest payload hash drift: {manifest_path}")
    tasks = manifest["tasks"]
    expected = {(task, seed) for task in tasks for seed in manifest["seeds"]}
    server_hashes = {row["server_metadata_sha256"] for row in manifest["servers"]}
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    result_dir = run_dir / "results" / CONFIG
    for path in sorted(result_dir.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            source = f"{path}:{line_number}"
            try:
                row = json.loads(line)
                key = (str(row.get("task")), int(row.get("seed", -1)))
            except Exception as error:
                errors.append(f"{source}: malformed row: {error}")
                continue
            checks = {
                "expected_key": key in expected,
                "status": row.get("status") == "complete",
                "config": row.get("config") == CONFIG,
                "task_set": row.get("task_set") == manifest["task_set"],
                "split": row.get("split") == "target",
                "flow": row.get("flow_steps") == 4,
                "execute": row.get("replan_steps") == 16,
                "paired": row.get("paired_action_noise") is True,
                "noise": row.get("action_noise_protocol") == NOISE_PROTOCOL,
                "fresh": row.get("fresh_environment") is True,
                "render": row.get("render_enabled") is True,
                "horizon": key[0] in tasks
                and int(row.get("max_steps", -1)) == int(get_task_horizon(key[0])),
                "server": row.get("server_metadata_sha256") in server_hashes,
            }
            failed = [name for name, valid in checks.items() if not valid]
            if failed:
                errors.append(f"{source}: {failed}")
                continue
            if key in rows and rows[key] != row:
                errors.append(f"{source}: conflicting duplicate {key}")
                continue
            rows[key] = row
    missing = sorted(expected - set(rows))
    if require_complete and (missing or errors):
        raise ValueError(f"formal result incomplete: missing={len(missing)} errors={errors[:5]}")
    return manifest, rows, errors


def progress(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    manifest, rows, errors = load_rows(run_dir, require_complete=args.strict)
    expected = len(manifest["tasks"]) * len(manifest["seeds"])
    successes = sum(bool(row.get("success")) for row in rows.values())
    value = {
        "task_set": manifest["task_set"],
        "episodes": len(rows),
        "expected_episodes": expected,
        "successes": successes,
        "episode_sr": successes / len(rows) if rows else None,
        "validation_errors": errors,
        "complete": len(rows) == expected and not errors,
    }
    print(json.dumps(value, sort_keys=True))


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    frac = index - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def aggregate(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    all_task_sr: dict[str, float] = {}
    task_set_rows = {}
    artifacts = {}
    total_successes = 0
    total_episodes = 0
    for task_set in TASK_SETS:
        run_dir = root / "results" / task_set
        manifest, rows, errors = load_rows(run_dir, require_complete=True)
        if errors:
            raise ValueError(errors[:5])
        per_task = {
            task: sum(bool(rows[(task, seed)]["success"]) for seed in manifest["seeds"])
            / len(manifest["seeds"])
            for task in manifest["tasks"]
        }
        successes = sum(bool(row["success"]) for row in rows.values())
        total_successes += successes
        total_episodes += len(rows)
        all_task_sr.update(per_task)
        task_set_rows[task_set] = {
            "tasks": len(per_task),
            "episodes": len(rows),
            "successes": successes,
            "task_macro_sr": sum(per_task.values()) / len(per_task),
            "episode_sr": successes / len(rows),
            "per_task_sr": per_task,
        }
        artifacts[task_set] = artifact(run_dir / "manifest.json")
    rng = random.Random(20260826)
    task_values = list(all_task_sr.values())
    bootstrap = [
        sum(rng.choice(task_values) for _ in task_values) / len(task_values)
        for _ in range(args.bootstrap)
    ]
    summary = {
        "schema_version": 1,
        "config_id": CONFIG,
        "complete": total_episodes == 2500,
        "episodes": total_episodes,
        "successes": total_successes,
        "episode_sr": total_successes / total_episodes,
        "task_macro_sr": sum(task_values) / len(task_values),
        "task_cluster_bootstrap_95ci": [
            percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)
        ],
        "bootstrap_samples": args.bootstrap,
        "task_sets": task_set_rows,
        "formal_failures": 0,
        "manifests": artifacts,
    }
    out = root / "aggregate" / "summary.json"
    atomic_json(out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--task-set", required=True, choices=TASK_SETS)
    freeze_parser.add_argument("--run-dir", required=True)
    freeze_parser.add_argument("--control-dir", required=True)
    freeze_parser.add_argument("--pack", required=True)
    freeze_parser.add_argument("--calibration", required=True)
    freeze_parser.add_argument("--preflight", required=True)
    progress_parser = sub.add_parser("progress")
    progress_parser.add_argument("--run-dir", required=True)
    progress_parser.add_argument("--strict", action="store_true")
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--root", required=True)
    aggregate_parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        freeze(args)
    elif args.command == "progress":
        progress(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
