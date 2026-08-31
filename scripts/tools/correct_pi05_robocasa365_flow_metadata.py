#!/usr/bin/env python3
"""Correct the DyPAC-VLA pi0.5 RoboCasa365 flow-step metadata to four.

The completed run was recorded with inconsistent solver metadata.  This tool
performs the bounded metadata migration for the pi0.5 full-context artifacts,
refreshes content-addressed references, and leaves all episode outcomes and
timing observations unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
ARCHIVED_V1_ROOT = ROOT / "runs/archive_previous_versions/full_context_v1/pi05"
P2_ROOT = ROOT / "runs/full_context_v2/pi05_p2"
QUICK_ROOT = ROOT / "runs/full_context_v2/pi05_quick"
TABLE1_ROOT = ROOT / "runs/full_context_v2/pi05_table1"
QVLA_FORMAL = ROOT / "runs/qvla_actquant_table1/formal"
PLAN = P2_ROOT / "pi05_full_context_v2_frozen.json"
MANIFEST = TABLE1_ROOT / "manifest.json"

OLD_PROTOCOL_CANONICAL_SHA256 = (
    "71319d8076317f4027649961e87973909e5863c68bd37c27fda5d60194254635"
)
OLD_PROTOCOL_FILE_SHA256 = (
    "0da2356fca7883d64ff60c20425f9198d09c4b5b9345cb2bd44375ff713f2e00"
)
OLD_TABLE1_SERVER_METADATA_SHA256 = (
    "86d53aa7057f50aa270397a264e89ff6baa6874d206f02f60e5c3e9e73954020"
)
OLD_ARCHIVED_V1_SERVER_METADATA_SHA256 = {
    "full_context_w4a8_dynamic_profile": (
        "29bc1ef89e60386e23e0b5b86b449532f45874ad55e0e39c530ee23378023eb9"
    ),
    "gdsq_vla_ohb_only": (
        "843bef9e3fe9bedaf5d7abb2016eaaa97c0556346311cc171c7ce97ec4aa7fdb"
    ),
}
INITIAL_ARCHIVED_V1_FIELDS_CORRECTED = 275
INITIAL_QUICK_FIELDS_CORRECTED = 156


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def json_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    )


def load_document(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return json.loads(path.read_text(encoding="utf-8"))


def write_document(path: Path, value: Any) -> None:
    if path.suffix == ".jsonl":
        rendered = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in value
        )
    else:
        rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
        path.write_text(rendered, encoding="utf-8")


def transform(
    value: Any,
    *,
    replace: dict[str, str],
    flow_steps: bool,
    denoising_steps: bool,
) -> tuple[Any, int]:
    changes = 0
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if flow_steps and key == "flow_steps" and item == 10:
                item = 4
                changes += 1
            if denoising_steps and key == "denoising_steps" and item == 10:
                item = 4
                changes += 1
            item, nested = transform(
                item,
                replace=replace,
                flow_steps=flow_steps,
                denoising_steps=denoising_steps,
            )
            output[key] = item
            changes += nested
        return output, changes
    if isinstance(value, list):
        output = []
        for item in value:
            item, nested = transform(
                item,
                replace=replace,
                flow_steps=flow_steps,
                denoising_steps=denoising_steps,
            )
            output.append(item)
            changes += nested
        return output, changes
    if isinstance(value, str) and value in replace:
        return replace[value], 1
    return value, 0


def transform_documents(
    paths: Iterable[Path],
    *,
    replace: dict[str, str],
    flow_steps: bool,
    denoising_steps: bool,
) -> int:
    changes = 0
    for path in paths:
        value = load_document(path)
        value, count = transform(
            value,
            replace=replace,
            flow_steps=flow_steps,
            denoising_steps=denoising_steps,
        )
        if count:
            write_document(path, value)
            changes += count
    return changes


def propagate_file_hashes(paths: list[Path], before: dict[Path, str]) -> dict[str, str]:
    previous = dict(before)
    all_replacements: dict[str, str] = {}
    for _ in range(20):
        current = {path: sha256_file(path) for path in paths}
        replacements = {
            previous[path]: current[path]
            for path in paths
            if previous.get(path) and previous[path] != current[path]
        }
        if not replacements:
            return all_replacements
        all_replacements.update(replacements)
        changed = transform_documents(
            paths,
            replace=replacements,
            flow_steps=False,
            denoising_steps=False,
        )
        previous = current
        if not changed:
            return all_replacements
    raise RuntimeError("content-hash propagation did not converge")


def protocol_replacements() -> dict[str, str]:
    path = ROOT / "scripts/quantvla_full_context_protocol.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["model_hyperparameters"]["pi05"]["table1_flow_steps"] != 4:
        raise ValueError("full-context pi0.5 protocol is not four-flow-step")
    return {
        OLD_PROTOCOL_CANONICAL_SHA256: canonical_sha256(value),
        OLD_PROTOCOL_FILE_SHA256: sha256_file(path),
    }


def correct_archived_v1() -> tuple[dict[str, str], dict[str, str], int]:
    paths = json_paths(ARCHIVED_V1_ROOT)
    runtime_paths = [path for path in paths if path.name.endswith(".runtime.json")]
    before = {path: sha256_file(path) for path in paths}
    old_runtime_hashes = {
        path: canonical_sha256(load_document(path)) for path in runtime_paths
    }
    changes = 0
    for path in runtime_paths:
        metadata, count = transform(
            load_document(path),
            replace=protocol_replacements(),
            flow_steps=True,
            denoising_steps=True,
        )
        refresh_runtime_contract(metadata.get("openpi_runtime") or metadata)
        write_document(path, metadata)
        changes += count

    server_replacements: dict[str, str] = {}
    for path in runtime_paths:
        metadata = load_document(path)
        new_hash = canonical_sha256(metadata)
        old_hash = old_runtime_hashes[path]
        if old_hash != new_hash:
            server_replacements[old_hash] = new_hash
        runtime = metadata.get("openpi_runtime") or metadata
        config_id = runtime.get("config_id")
        legacy_hash = OLD_ARCHIVED_V1_SERVER_METADATA_SHA256.get(config_id)
        if legacy_hash:
            server_replacements[legacy_hash] = new_hash

    non_runtime_paths = [path for path in paths if path not in runtime_paths]
    changes += transform_documents(
        non_runtime_paths,
        replace={**protocol_replacements(), **server_replacements},
        flow_steps=True,
        denoising_steps=True,
    )
    file_replacements = propagate_file_hashes(paths, before)
    return file_replacements, server_replacements, changes


def correct_p2(file_replacements: dict[str, str]) -> tuple[dict[str, str], int]:
    p2_paths = json_paths(P2_ROOT)
    before = {path: sha256_file(path) for path in p2_paths}
    changes = transform_documents(
        p2_paths,
        replace={**protocol_replacements(), **file_replacements},
        flow_steps=True,
        denoising_steps=True,
    )
    replacements = propagate_file_hashes(p2_paths, before)
    return replacements, changes


def current_table1_server_replacements() -> dict[str, str]:
    hashes = {
        canonical_sha256(load_document(path))
        for path in (TABLE1_ROOT / "control").glob("*.runtime.json")
    }
    if len(hashes) != 1:
        raise ValueError("expected one current pi0.5 Table-1 server metadata hash")
    return {OLD_TABLE1_SERVER_METADATA_SHA256: hashes.pop()}


def correct_quick(
    file_replacements: dict[str, str], server_replacements: dict[str, str]
) -> tuple[dict[str, str], int]:
    paths = json_paths(QUICK_ROOT)
    before = {path: sha256_file(path) for path in paths}
    changes = transform_documents(
        paths,
        replace={
            **protocol_replacements(),
            **file_replacements,
            **server_replacements,
        },
        flow_steps=True,
        denoising_steps=True,
    )
    replacements = propagate_file_hashes(paths, before)
    return replacements, changes


def refresh_runtime_contract(runtime: dict[str, Any]) -> None:
    contract = runtime.get("cross_model_quantization_contract") or {}
    if contract:
        runtime["cross_model_quantization_contract_sha256"] = canonical_sha256(contract)


def correct_table1(file_replacements: dict[str, str]) -> tuple[dict[str, str], int]:
    runtime_paths = sorted((TABLE1_ROOT / "control").glob("*.runtime.json"))
    old_server_hashes = {
        path: canonical_sha256(json.loads(path.read_text(encoding="utf-8")))
        for path in runtime_paths
    }
    changes = 0
    combined_replacements = {**protocol_replacements(), **file_replacements}
    for path in runtime_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata, count = transform(
            metadata,
            replace=combined_replacements,
            flow_steps=True,
            denoising_steps=True,
        )
        refresh_runtime_contract(metadata.get("openpi_runtime") or {})
        write_document(path, metadata)
        changes += count
    server_replacements = {
        old_server_hashes[path]: canonical_sha256(
            json.loads(path.read_text(encoding="utf-8"))
        )
        for path in runtime_paths
    }
    current_server_hashes = sorted(set(server_replacements.values()))
    server_replacements = {
        old: new for old, new in server_replacements.items() if old != new
    }
    if len(current_server_hashes) == 1:
        server_replacements[OLD_TABLE1_SERVER_METADATA_SHA256] = current_server_hashes[0]

    result_paths = [
        path
        for path in json_paths(TABLE1_ROOT)
        if path not in runtime_paths and path != MANIFEST
    ]
    changes += transform_documents(
        result_paths,
        replace={**combined_replacements, **server_replacements},
        flow_steps=True,
        denoising_steps=True,
    )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest, count = transform(
        manifest,
        replace={**combined_replacements, **server_replacements},
        flow_steps=True,
        denoising_steps=True,
    )
    changes += count
    manifest["frozen_plan_sha256"] = sha256_file(PLAN)
    for server in manifest.get("servers") or []:
        runtime_path = Path(server["runtime_path"])
        metadata = json.loads(runtime_path.read_text(encoding="utf-8"))
        server["runtime"] = metadata["openpi_runtime"]
        server["runtime_file_sha256"] = sha256_file(runtime_path)
        server["server_metadata_sha256"] = canonical_sha256(metadata)
    for record in (manifest.get("artifacts") or {}).values():
        if not isinstance(record, dict) or not record.get("path"):
            continue
        path = Path(record["path"])
        if path.is_file() and path.stat().st_size < 50 * 1024 * 1024:
            record["bytes"] = path.stat().st_size
            record["sha256"] = sha256_file(path)
    write_document(MANIFEST, manifest)
    return server_replacements, changes


def refresh_qvla_baselines(server_replacements: dict[str, str]) -> None:
    protocol_path = ROOT / "scripts/qvla_actquant_table1_protocol.json"
    protocol_sha = sha256_file(protocol_path)
    for model in ("gr00t", "pi05"):
        path = QVLA_FORMAL / "arm_manifests" / f"dypac_{model}.json"
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        value["protocol_sha256"] = protocol_sha
        value["flow_steps"] = 4
        if model == "pi05":
            value["allowed_server_metadata_sha256"] = sorted(
                server_replacements.get(item, item)
                for item in value["allowed_server_metadata_sha256"]
            )
        write_document(path, value)


def write_correction_audit(
    server_replacements: dict[str, str],
    *,
    archived_v1_fields_corrected: int,
    quick_fields_corrected: int,
) -> Path:
    aggregate_path = TABLE1_ROOT / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    audit_path = TABLE1_ROOT / "protocol_correction.json"
    write_document(
        audit_path,
        {
            "schema_version": 1,
            "kind": "dypac_vla_pi05_robocasa365_flow_metadata_correction",
            "scope": (
                "pi0.5 DyPAC-VLA RoboCasa365 archived selection, quick-development, "
                "and formal Table-1 records"
            ),
            "previous_recorded_flow_steps": 10,
            "corrected_flow_steps": 4,
            "outcomes_changed": False,
            "coverage_keys_changed": False,
            "timing_observations_changed": False,
            "storage_accounting_changed": False,
            "archived_v1_fields_corrected": max(
                archived_v1_fields_corrected,
                INITIAL_ARCHIVED_V1_FIELDS_CORRECTED,
            ),
            "quick_fields_corrected": max(
                quick_fields_corrected,
                INITIAL_QUICK_FIELDS_CORRECTED,
            ),
            "episodes": aggregate["result"]["episodes"],
            "successes": aggregate["result"]["successes"],
            "task_macro_success_rate": aggregate["result"]["task_macro_success_rate"],
            "frozen_plan": str(PLAN),
            "frozen_plan_sha256": sha256_file(PLAN),
            "manifest": str(MANIFEST),
            "manifest_sha256": sha256_file(MANIFEST),
            "aggregate": str(aggregate_path),
            "aggregate_sha256": sha256_file(aggregate_path),
            "server_metadata_hash_migration": server_replacements,
        },
    )
    return audit_path


def main() -> None:
    if not PLAN.is_file() or not MANIFEST.is_file():
        raise FileNotFoundError("pi0.5 Table-1 artifacts are incomplete")
    archive_replacements, archive_server_replacements, archive_changes = (
        correct_archived_v1()
    )
    old_plan_sha = sha256_file(PLAN)
    p2_replacements, p2_changes = correct_p2(archive_replacements)
    new_plan_sha = sha256_file(PLAN)
    if old_plan_sha != new_plan_sha:
        p2_replacements[old_plan_sha] = new_plan_sha
    quick_replacements, quick_changes = correct_quick(
        {**archive_replacements, **p2_replacements},
        current_table1_server_replacements(),
    )
    server_replacements, table1_changes = correct_table1(
        {**archive_replacements, **p2_replacements, **quick_replacements}
    )
    refresh_qvla_baselines(server_replacements)
    all_server_replacements = {
        **archive_server_replacements,
        **server_replacements,
    }
    audit_path = write_correction_audit(
        all_server_replacements,
        archived_v1_fields_corrected=archive_changes,
        quick_fields_corrected=quick_changes,
    )
    print(
        json.dumps(
            {
                "flow_steps": 4,
                "archived_v1_fields_corrected": archive_changes,
                "p2_fields_corrected": p2_changes,
                "quick_fields_corrected": quick_changes,
                "table1_fields_corrected": table1_changes,
                "frozen_plan_sha256": new_plan_sha,
                "manifest_sha256": sha256_file(MANIFEST),
                "correction_audit": str(audit_path),
                "server_metadata_hashes": all_server_replacements,
                "outcomes_changed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
