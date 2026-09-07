#!/usr/bin/env python3
"""Freeze the pre-rollout semantic-metadata-hash operational correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tools"))

from materialize_pi05_fcp_diagnostic import parse_server  # noqa: E402
from quantvla_cross_model_protocol import sha256_file  # noqa: E402
from quantvla_outputimpact import atomic_json  # noqa: E402


ROOT = REPO / "runs/full_context_v2/pi05_fcp_diagnostic"
PREREGISTRATION = ROOT / "preregistration.json"
EXECUTION = ROOT / "execution_manifest.json"
EXPECTED_PREREGISTRATION_SHA256 = "b2418230449262e402229d68dd1376e6113de0c7876daf0a245cebe8d6315b98"
ARMS = ("transferred_initializer", "single_best", "two_best")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def semantic_metadata_hash(value: dict[str, Any]) -> str:
    runtime = value.get("openpi_runtime") or {}
    claimed = runtime.get("semantic_metadata_sha256")
    if not claimed:
        raise ValueError("formal pi0.5 runtime omitted semantic_metadata_sha256")
    stable = json.loads(json.dumps(value, sort_keys=True))
    stable_runtime = stable.get("openpi_runtime") or {}
    stable_runtime.pop("gpu_memory_bytes", None)
    stable_runtime.pop("semantic_metadata_sha256", None)
    computed = canonical_hash(stable)
    if claimed != computed:
        raise ValueError(f"server semantic metadata SHA mismatch: {claimed} != {computed}")
    return str(claimed)


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="append", required=True, metavar="ARM,INSTANCE,GPU,PORT,RUNTIME")
    args = parser.parse_args()
    if sha256_file(PREREGISTRATION) != EXPECTED_PREREGISTRATION_SHA256:
        raise ValueError("original prospective preregistration hash drift")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    result_files = sorted((ROOT / "rollouts").glob("**/*.jsonl"))
    if result_files:
        raise ValueError(f"operational correction is legal only before rollout; found {result_files[:3]}")
    worker_pids = sorted((ROOT / "control/workers").glob("*.pid"))
    if worker_pids:
        raise ValueError("operational correction is legal only before worker launch")

    failed_smokes = []
    for arm in ARMS:
        log = ROOT / "preflight" / arm / "smoke.log"
        text = log.read_text(encoding="utf-8")
        if "server metadata hash mismatch" not in text:
            raise ValueError(f"{arm}: expected fail-closed semantic-hash smoke record is absent")
        failed_smokes.append(artifact(log))
        polluted = ROOT / "preflight" / arm / "semantic_smoke_stdout_pollution.failed.log"
        polluted_text = polluted.read_text(encoding="utf-8")
        if "server metadata hash mismatch" not in polluted_text:
            raise ValueError(f"{arm}: expected fail-closed stdout-pollution smoke record is absent")
        failed_smokes.append(artifact(polluted))
        missing_resume = ROOT / "preflight" / arm / "semantic_smoke_missing_resume.failed.log"
        missing_resume_text = missing_resume.read_text(encoding="utf-8")
        if "resume directory does not exist" not in missing_resume_text:
            raise ValueError(f"{arm}: expected fail-closed missing-resume smoke record is absent")
        failed_smokes.append(artifact(missing_resume))

    servers: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for raw in args.server:
        arm, row = parse_server(raw, prereg)
        runtime_path = Path(row["runtime"]["path"])
        metadata = json.loads(runtime_path.read_text(encoding="utf-8"))
        full_file_hash = canonical_hash(metadata)
        semantic_hash = semantic_metadata_hash(metadata)
        if full_file_hash == semantic_hash:
            raise ValueError(f"{row['instance']}: correction does not change the hash domain")
        row["superseded_full_json_canonical_sha256"] = full_file_hash
        row["server_metadata_sha256"] = semantic_hash
        servers[arm].append(row)
    if {arm: len(rows) for arm, rows in servers.items()} != {arm: 9 for arm in ARMS}:
        raise ValueError("corrected runtime manifest requires nine servers per arm")

    correction_tool = Path(__file__).resolve()
    launcher = REPO / "scripts/resume_pi05_fcp_after_semantic_hash_fix.sh"
    payload = {
        "schema_version": 1,
        "kind": "pi05_fcp_same_protocol_execution_manifest",
        "immutable": True,
        "result_feedback_allowed": False,
        "preregistration": artifact(PREREGISTRATION),
        "servers": {arm: sorted(rows, key=lambda value: value["instance"]) for arm, rows in servers.items()},
        "operational_erratum": {
            "kind": "pre_rollout_semantic_metadata_hash_correction",
            "scientific_protocol_changed": False,
            "model_or_artifact_changed": False,
            "statistics_changed": False,
            "new_rollout_rows_before_correction": 0,
            "cause": (
                "The outer runner supplied the canonical hash of the complete runtime JSON, "
                "while the already frozen server/evaluator contract uses the embedded semantic "
                "hash that excludes allocator counters."
            ),
            "prelaunch_plumbing_note": (
                "A second smoke also failed before inference because an environment warning was "
                "captured on stdout alongside the otherwise correct semantic hash. The final "
                "launcher reads the 64-hex claim directly; the freezer and evaluator each "
                "independently recompute and verify it. A third smoke confirmed the evaluator "
                "also fails closed on a missing resume directory; the launcher creates that empty "
                "directory explicitly before its final smoke."
            ),
            "resolution": (
                "Verify each embedded semantic hash with the evaluator's frozen algorithm and "
                "bind that value in the execution manifest before any rollout worker starts."
            ),
            "failed_smoke_logs": failed_smokes,
            "correction_code": {
                "manifest_tool": artifact(correction_tool),
                "worker_launcher": artifact(launcher),
            },
        },
    }
    if EXECUTION.is_file():
        existing = json.loads(EXECUTION.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("immutable corrected execution-manifest drift")
    else:
        atomic_json(EXECUTION, payload)
    print(json.dumps({"path": str(EXECUTION), "sha256": sha256_file(EXECUTION), "servers": 27}, indent=2))


if __name__ == "__main__":
    main()
