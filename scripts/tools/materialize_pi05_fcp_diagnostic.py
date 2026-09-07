#!/usr/bin/env python3
"""Freeze/verify the prospective pi0.5 FCP diagnostic and runtime manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tools"))

from quantvla_cross_model_protocol import (  # noqa: E402
    closed_loop_runtime_protocol,
    sha256_file,
    validate_quant_plan,
)
from quantvla_outputimpact import atomic_json  # noqa: E402


PROTOCOL = REPO / "scripts/quantvla_pi05_fcp_diagnostic_protocol.json"
TASK_PROTOCOL = REPO / "scripts/quantvla_full_context_protocol.json"
ROOT = REPO / "runs/full_context_v2/pi05_fcp_diagnostic"
PREREGISTRATION = ROOT / "preregistration.json"
EXECUTION = ROOT / "execution_manifest.json"
ANCHOR_RESULTS = (
    REPO
    / "runs/full_context_v2/pi05_table1/results/full_context_w4a8_dynamic_profile"
)
ANCHOR_MANIFEST = REPO / "runs/full_context_v2/pi05_table1/manifest.json"
CONFIG_ID = "full_context_w4a8_dynamic_profile"
NEW_ARMS = ("transferred_initializer", "single_best", "two_best")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def stable_write(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(f"immutable pi0.5 FCP artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def resolve_protocol_path(raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def verify_frozen_file(row: dict[str, Any], label: str) -> Path:
    path = resolve_protocol_path(str(row["path"]))
    actual = sha256_file(path)
    if actual != row["sha256"]:
        raise ValueError(f"{label} hash drift: {actual} != {row['sha256']}")
    return path


def active_w4(plan: dict[str, Any]) -> list[str]:
    return sorted(
        name
        for name, row in plan["layers"].items()
        if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
    )


def verify_hessian(path: Path, plan_path: Path, expected_layers: list[str]) -> dict[str, Any]:
    sidecar_path = Path(str(path) + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    checks = {
        "npz_sha256": sidecar.get("npz_sha256") == sha256_file(path),
        "plan_sha256": sidecar.get("deployment_plan_sha256") == sha256_file(plan_path),
        "inventory": sorted(sidecar.get("layer_names") or []) == expected_layers,
        "inventory_unique": len(sidecar.get("layer_names") or [])
        == len(set(sidecar.get("layer_names") or [])),
        "requantized": sidecar.get("requantized") is False,
        "group_size": int(sidecar.get("group_size", -1)) == 64,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{path}: Hessian subset drift: {failed}")
    return {"npz": artifact(path), "sidecar": artifact(sidecar_path)}


def load_anchor(
    expected: set[tuple[str, str, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seen: dict[tuple[str, str, int], tuple[Path, int]] = {}
    successes = 0
    sources: list[dict[str, Any]] = []
    for path in sorted(ANCHOR_RESULTS.glob("*.jsonl")):
        used = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config") != CONFIG_ID:
                continue
            key = (str(row.get("task_set")), str(row.get("task")), int(row.get("seed", -1)))
            if key not in expected:
                continue
            if key in seen:
                raise ValueError(f"duplicate projected-anchor key {key}: {path}:{line_number}")
            if row.get("status") != "complete":
                raise ValueError(f"incomplete projected-anchor row: {path}:{line_number}")
            if not bool(row.get("paired_action_noise", False)):
                raise ValueError(f"unpaired projected-anchor row: {path}:{line_number}")
            for field, wanted in (("flow_steps", 4), ("action_horizon", 16), ("replan_steps", 16)):
                if int(row.get(field, -1)) != wanted:
                    raise ValueError(f"projected-anchor {field} drift: {path}:{line_number}")
            seen[key] = (path, line_number)
            successes += int(bool(row["success"]))
            used += 1
        if used:
            sources.append({**artifact(path), "episodes": used})
    if set(seen) != expected:
        raise ValueError(f"projected-anchor coverage drift: {len(seen)}/{len(expected)}")
    if successes != 141:
        raise ValueError(f"known projected-anchor success count drift: {successes} != 141")
    return {
        "configuration": "projected_anchor",
        "results_root": str(ANCHOR_RESULTS.resolve()),
        "episodes": len(seen),
        "successes_known_before_registration": successes,
        "success_rate_known_before_registration": successes / len(seen),
        "formal_manifest": artifact(ANCHOR_MANIFEST),
    }, sources


def code_artifacts() -> dict[str, dict[str, Any]]:
    paths = {
        "server": REPO / "scripts/run_pi05_formal_server.sh",
        "worker": REPO / "scripts/run_pi05_formal_worker_seeded.sh",
        "evaluator": REPO / "scripts/run_robocasa365_pi05_eval.py",
        "hessian_subsetter": REPO / "scripts/tools/materialize_full_context_hessian_subset.py",
        "materializer": Path(__file__).resolve(),
        "aggregator": REPO / "scripts/tools/aggregate_pi05_fcp_diagnostic.py",
        "rollout_runner": REPO / "scripts/run_pi05_fcp_diagnostic.sh",
        "hardware_benchmark": REPO / "scripts/tools/benchmark_pi05_hardware.py",
        "hardware_aggregator": REPO / "scripts/tools/aggregate_pi05_hardware.py",
        "hardware_runner": REPO / "scripts/run_pi05_fcp_hardware_benchmark.sh",
    }
    return {key: artifact(path) for key, path in paths.items()}


def make_preregistration() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("kind") != "pi05_fcp_same_protocol_diagnostic_protocol":
        raise ValueError("unexpected pi0.5 FCP protocol kind")
    for label, row in protocol["frozen_inputs"].items():
        if "path" in row:
            verify_frozen_file(row, label)
    task_document = json.loads(TASK_PROTOCOL.read_text(encoding="utf-8"))
    tasks = {split: list(values) for split, values in task_document["table1"]["tasks"].items()}
    expected_counts = protocol["closed_loop"]["task_sets"]
    if {key: len(value) for key, value in tasks.items()} != expected_counts:
        raise ValueError("RoboCasa365 task inventory drift")
    seeds = [int(seed) for seed in protocol["closed_loop"]["seeds"]]
    expected = {
        (split, task, seed)
        for split, split_tasks in tasks.items()
        for task in split_tasks
        for seed in seeds
    }
    if len(expected) != 500:
        raise ValueError("closed-loop coverage contract is not 500 rows per configuration")

    configurations: dict[str, Any] = {}
    for arm, spec in protocol["configurations"].items():
        plan_path = resolve_protocol_path(spec["plan"])
        if sha256_file(plan_path) != spec["plan_sha256"]:
            raise ValueError(f"{arm}: plan hash drift")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        attestation = validate_quant_plan(plan, model="pi05", source=str(plan_path))
        selected = active_w4(plan)
        observed = {
            "w4_layers": len(selected),
            "fp16_layers": 180 - len(selected),
            "table1_total_static_bytes": int(
                plan.get("table1_total_static_bytes", plan.get("total_bytes", -1))
            ),
        }
        for field, value in observed.items():
            if value != int(spec[field]):
                raise ValueError(f"{arm}: {field} drift: {value} != {spec[field]}")
        if int(attestation["quantized_w4_layers"]) != observed["w4_layers"]:
            raise ValueError(f"{arm}: quant-plan attestation drift")
        if arm == "projected_anchor":
            hessian_path = REPO / "runs/full_context_v2/pi05_quick/artifacts/hessian_w4.npz"
        else:
            hessian_path = ROOT / "artifacts" / arm / "hessian_w4.npz"
        configurations[arm] = {
            **spec,
            "plan": artifact(plan_path),
            "protected_layers": sorted(set(plan["layers"]) - set(selected)),
            "hessian_w4": verify_hessian(hessian_path, plan_path, selected),
        }

    anchor, anchor_sources = load_anchor(expected)
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    payload = {
        "schema_version": 1,
        "kind": "pi05_fcp_same_protocol_preregistration",
        "immutable": True,
        "result_feedback_allowed": False,
        "protocol": artifact(PROTOCOL),
        "repository": {"head": head, "dirty_at_registration": True},
        "code": code_artifacts(),
        "runtime_contract": {
            "config_id": CONFIG_ID,
            "weight_quantization": "signed packed Hessian group-64 W4",
            "activation_quantization": "dynamic per-forward per-channel A8",
            "row_rotation": "identity",
            "runtime_selector": False,
            "runtime_correction": False,
            "flow_steps": 4,
            "action_horizon": 16,
            "replan_steps": 16,
            "paired_action_noise": True,
        },
        "evaluation": {
            "task_sets": tasks,
            "seeds": seeds,
            "coverage_keys_per_configuration": len(expected),
            "new_configurations": list(NEW_ARMS),
            "new_episodes": len(expected) * len(NEW_ARMS),
            "projected_anchor": anchor,
            "projected_anchor_source_files": anchor_sources,
        },
        "configurations": configurations,
        "statistics": protocol["statistics"],
        "hardware": protocol["hardware"],
        "acceptance": protocol["acceptance"],
        "interpretation": (
            "Prospective mechanism diagnostic after model selection. The known anchor and all "
            "new results are forbidden from changing the frozen pi0.5 selection."
        ),
    }
    return payload


def parse_server(raw: str, prereg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    arm, instance, gpu_text, port_text, runtime_text = raw.split(",", 4)
    if arm not in NEW_ARMS:
        raise ValueError(f"unknown runtime arm: {arm}")
    runtime_path = Path(runtime_text).expanduser().resolve()
    metadata = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime = metadata.get("openpi_runtime") or {}
    duquant = runtime.get("duquant") or {}
    expected = prereg["configurations"][arm]
    protocol = runtime.get("protocol") or {}
    wanted_protocol = closed_loop_runtime_protocol()
    wanted_protocol["flow_steps"] = 4
    mismatches = {
        key: (protocol.get(key), wanted)
        for key, wanted in wanted_protocol.items()
        if protocol.get(key) != wanted
    }
    checks = {
        "config_id": runtime.get("config_id") == CONFIG_ID,
        "model": (runtime.get("model_adapter") or {}).get("model") == "pi05",
        "plan": duquant.get("plan_sha256") == expected["plan"]["sha256"],
        "hessian": duquant.get("hessian_w4_sha256")
        == expected["hessian_w4"]["npz"]["sha256"],
        "wrapped": int(duquant.get("wrapped_layers", -1)) == int(expected["w4_layers"]),
        "loaded": int(duquant.get("hessian_w4_loaded", -1)) == int(expected["w4_layers"]),
        "packed": duquant.get("packed_low_bit_residency") is True,
        "selector": (runtime.get("runtime_selector") or {}).get("enabled") is False,
        "correction": (runtime.get("errorfold") or {}).get("enabled") is False,
        "protocol": not mismatches,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{instance}: runtime attestation drift: {failed}; {mismatches}")
    return arm, {
        "arm": arm,
        "instance": instance,
        "gpu": int(gpu_text),
        "port": int(port_text),
        "runtime": artifact(runtime_path),
        "server_metadata_sha256": canonical_hash(metadata),
        "plan_sha256": duquant["plan_sha256"],
        "hessian_w4_sha256": duquant["hessian_w4_sha256"],
        "wrapped_layers": int(duquant["wrapped_layers"]),
    }


def make_execution(server_specs: list[str]) -> dict[str, Any]:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg != make_preregistration():
        raise ValueError("preregistration verification failed before runtime freeze")
    servers: dict[str, list[dict[str, Any]]] = {arm: [] for arm in NEW_ARMS}
    for raw in server_specs:
        arm, row = parse_server(raw, prereg)
        servers[arm].append(row)
    if any(len(rows) < 1 for rows in servers.values()):
        raise ValueError("every new arm requires at least one runtime-attested server")
    identifiers = [row["instance"] for rows in servers.values() for row in rows]
    ports = [row["port"] for rows in servers.values() for row in rows]
    if len(identifiers) != len(set(identifiers)) or len(ports) != len(set(ports)):
        raise ValueError("duplicate runtime instance or port")
    return {
        "schema_version": 1,
        "kind": "pi05_fcp_same_protocol_execution_manifest",
        "immutable": True,
        "result_feedback_allowed": False,
        "preregistration": artifact(PREREGISTRATION),
        "servers": {key: sorted(value, key=lambda row: row["instance"]) for key, value in servers.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-runtime", action="store_true")
    parser.add_argument("--server", action="append", default=[], metavar="ARM,INSTANCE,GPU,PORT,RUNTIME")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.freeze_runtime:
        if not args.server:
            raise ValueError("--freeze-runtime requires --server entries")
        value = make_execution(args.server)
        path = EXECUTION
    else:
        if args.server:
            raise ValueError("--server is valid only with --freeze-runtime")
        value = make_preregistration()
        path = PREREGISTRATION
    if args.verify:
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"frozen artifact verification failed: {path}")
    else:
        stable_write(path, value)
    print(json.dumps({"path": str(path), "sha256": sha256_file(path), "verified": args.verify}, indent=2))


if __name__ == "__main__":
    main()
