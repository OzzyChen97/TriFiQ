#!/usr/bin/env python3
"""Validate GDSQ-VLA paper claims against frozen experiment evidence.

The registry is intentionally conservative: a main-paper claim is admissible
only when every referenced experiment is complete, has full coverage, and all
declared artifacts still match their frozen SHA-256 values.  Planned and
diagnostic experiments may be described as such but cannot populate main-table
numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "docs/gdsq_vla_iclr2027/experiment_registry.json"
MAIN_PAPER_GLOBS = (
    "docs/gdsq_vla_iclr2027/sections/*.tex",
    "docs/gdsq_vla_iclr2027/tables/*.tex",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def coverage_errors(experiment_id: str, experiment: dict[str, Any]) -> list[str]:
    coverage = experiment.get("coverage") or {}
    errors: list[str] = []
    if int(coverage.get("missing_episodes", 0) or 0) != 0:
        errors.append(f"{experiment_id}: missing episodes = {coverage['missing_episodes']}")
    if int(coverage.get("duplicate_episodes", 0) or 0) != 0:
        errors.append(
            f"{experiment_id}: duplicate episodes = {coverage['duplicate_episodes']}"
        )
    expected = coverage.get("expected_episodes")
    observed = coverage.get("observed_episodes")
    if expected is not None and observed is not None and int(expected) != int(observed):
        errors.append(f"{experiment_id}: observed {observed} != expected {expected}")
    expected_per = coverage.get("expected_episodes_per_config")
    observed_per = coverage.get("observed_episodes_per_config")
    if (
        expected_per is not None
        and observed_per is not None
        and int(expected_per) != int(observed_per)
    ):
        errors.append(
            f"{experiment_id}: observed/config {observed_per} != expected/config {expected_per}"
        )
    return errors


def artifact_records(value: Any, prefix: str = ""):
    """Yield every nested path/SHA record so new manifests cannot bypass gates."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield prefix or "artifact", value
            return
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from artifact_records(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            yield from artifact_records(child, child_prefix)


def artifact_errors(experiment_id: str, experiment: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for label, artifact in artifact_records(experiment):
        path = resolve(str(artifact["path"]))
        if not path.is_file():
            errors.append(f"{experiment_id}: missing {label}: {path}")
            continue
        actual = sha256_file(path)
        expected = str(artifact["sha256"])
        if actual != expected:
            errors.append(
                f"{experiment_id}: {label} SHA mismatch: expected {expected}, got {actual}"
            )
    return errors


def claim_experiment_ids(claim: dict[str, Any]) -> list[str]:
    if claim.get("experiment"):
        return [str(claim["experiment"])]
    return [str(value) for value in claim.get("experiments", [])]


def gr00t_selector_reuse_errors(registry: dict[str, Any]) -> list[str]:
    """Require row-level selector metadata for the conditional static reuse."""
    experiment = (registry.get("experiments") or {}).get(
        "gr00t_runtime_selector_official50"
    ) or {}
    if experiment.get("status") != "complete":
        return []
    errors: list[str] = []
    summary_record = experiment.get("reuse_attestation") or {}
    rows_record = experiment.get("derived_selector_rows") or {}
    if not summary_record or not rows_record:
        return ["gr00t_runtime_selector_official50: missing row-level reuse attestation"]
    summary_path = resolve(str(summary_record.get("path", "")))
    rows_path = resolve(str(rows_record.get("path", "")))
    if not summary_path.is_file() or not rows_path.is_file():
        return errors  # artifact_errors emits the path-specific message.
    summary = read_json(summary_path)
    selector = registry.get("runtime_selector") or {}
    coverage = summary.get("coverage") or {}
    contract = summary.get("selector_contract") or {}
    checks = {
        "attestation_valid": summary.get("valid") is True,
        "raw_results_unchanged": summary.get("raw_results_modified") is False,
        "reuse_enabled": summary.get("reuse_static_result") is True,
        "coverage": coverage.get("expected_episodes")
        == coverage.get("observed_episodes")
        == 2500,
        "no_missing": coverage.get("missing_episodes") == 0,
        "no_duplicates": coverage.get("duplicate_episodes") == 0,
        "selector_sha": contract.get("selector_sha256") == selector.get("sha256"),
        "selector_rule": contract.get("selector_rule_name") == selector.get("rule_name"),
        "variant": contract.get("selected_variant") == "baseline",
        "config": contract.get("selected_config_id") == "cscka_final",
    }
    errors.extend(
        f"gr00t_runtime_selector_official50: reuse check failed: {name}"
        for name, valid in checks.items()
        if not valid
    )
    keys: set[tuple[str, str, int]] = set()
    rows = 0
    for line_number, line in enumerate(rows_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append(
                f"gr00t_runtime_selector_official50: malformed derived row {line_number}"
            )
            continue
        rows += 1
        key = (str(row.get("task_set")), str(row.get("task")), int(row.get("seed", -1)))
        if key in keys:
            errors.append(
                f"gr00t_runtime_selector_official50: duplicate derived row {key}"
            )
        keys.add(key)
        row_checks = {
            "selector_sha256": selector.get("sha256"),
            "selector_rule_name": selector.get("rule_name"),
            "selected_variant": "baseline",
            "selected_config_id": "cscka_final",
        }
        for field, expected in row_checks.items():
            if row.get(field) != expected:
                errors.append(
                    "gr00t_runtime_selector_official50: "
                    f"derived row {line_number} has invalid {field}"
                )
        if row.get("runtime_selector_enabled") is not True:
            errors.append(
                "gr00t_runtime_selector_official50: "
                f"derived row {line_number} does not enable selector"
            )
    if rows != 2500 or len(keys) != 2500:
        errors.append(
            "gr00t_runtime_selector_official50: "
            f"derived coverage rows={rows}, unique={len(keys)}, expected=2500"
        )
    return errors


def gr00t_uniform_w6_replay_errors(registry: dict[str, Any]) -> list[str]:
    """Require a valid immutable replay audit when the GR00T W6 row is enabled."""
    experiment = (registry.get("experiments") or {}).get(
        "gr00t_uniform_w6_official50"
    ) or {}
    if experiment.get("status") != "complete":
        return []
    errors: list[str] = []
    replay_record = experiment.get("replay_audit") or {}
    summary_record = experiment.get("summary") or {}
    if not replay_record or not summary_record:
        return ["gr00t_uniform_w6_official50: missing replay audit or summary"]
    replay_path = resolve(str(replay_record.get("path", "")))
    if not replay_path.is_file():
        return errors  # artifact_errors emits the path-specific failure.
    replay = read_json(replay_path)
    coverage = replay.get("coverage") or {}
    summary = replay.get("summary") or {}
    checks = {
        "replay_valid": replay.get("valid") is True,
        "summary_byte_exact": summary.get("byte_exact") is True,
        "summary_hash": summary.get("sha256") == summary_record.get("sha256"),
        "coverage": coverage.get("expected_episodes")
        == coverage.get("observed_episodes")
        == 2500,
        "no_missing": coverage.get("missing_episodes") == 0,
        "no_duplicates": coverage.get("duplicate_episodes") == 0,
        "bootstrap": summary.get("bootstrap_draws") == 10_000,
        "launch_manifests_unchanged": replay.get("policy", {}).get(
            "launch_manifests_mutated"
        )
        is False,
    }
    errors.extend(
        f"gr00t_uniform_w6_official50: replay check failed: {name}"
        for name, valid in checks.items()
        if not valid
    )
    registered_manifests = {
        str(record.get("sha256")) for record in experiment.get("manifests", [])
    }
    replay_manifests = {
        str(record.get("sha256"))
        for record in (replay.get("manifests") or {}).values()
    }
    if len(registered_manifests) != 3 or registered_manifests != replay_manifests:
        errors.append(
            "gr00t_uniform_w6_official50: replay/registry manifest inventory differs"
        )
    return errors


def validate_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    experiments = registry.get("experiments") or {}
    selector = registry.get("runtime_selector") or {}
    selector_path = resolve(str(selector.get("path", "")))
    if not selector_path.is_file():
        errors.append(f"runtime selector missing: {selector_path}")
    elif sha256_file(selector_path) != selector.get("sha256"):
        errors.append("runtime selector SHA no longer matches the frozen v8 artifact")

    for preregistration_id, preregistration in (
        registry.get("preregistrations") or {}
    ).items():
        path = resolve(str(preregistration.get("path", "")))
        if not path.is_file():
            errors.append(f"{preregistration_id}: missing preregistration {path}")
            continue
        if sha256_file(path) != preregistration.get("sha256"):
            errors.append(f"{preregistration_id}: preregistration SHA mismatch")
        payload = read_json(path)
        if payload.get("result_blind") is not True:
            errors.append(f"{preregistration_id}: preregistration is not result-blind")
        nested = {
            key: value
            for key, value in preregistration.items()
            if key not in {"path", "sha256"}
        }
        for label, artifact in artifact_records(nested):
            artifact_path = resolve(str(artifact["path"]))
            if not artifact_path.is_file():
                errors.append(
                    f"{preregistration_id}: missing {label}: {artifact_path}"
                )
            elif sha256_file(artifact_path) != artifact["sha256"]:
                errors.append(f"{preregistration_id}: {label} SHA mismatch")

    for experiment_id, experiment in experiments.items():
        if experiment.get("status") in {"running", "complete"}:
            errors.extend(artifact_errors(experiment_id, experiment))
        if experiment.get("status") == "complete":
            errors.extend(coverage_errors(experiment_id, experiment))
        if experiment.get("main_claim_enabled") is True and experiment.get("status") != "complete":
            errors.append(
                f"{experiment_id}: main_claim_enabled requires status=complete, "
                f"got {experiment.get('status')!r}"
            )
        preregistration_id = experiment.get("preregistration")
        if preregistration_id and preregistration_id not in (
            registry.get("preregistrations") or {}
        ):
            errors.append(
                f"{experiment_id}: unknown preregistration {preregistration_id}"
            )

    for claim_id, claim in (registry.get("paper_claims") or {}).items():
        if claim.get("enabled") is not True:
            continue
        for experiment_id in claim_experiment_ids(claim):
            experiment = experiments.get(experiment_id)
            if experiment is None:
                errors.append(f"{claim_id}: unknown experiment {experiment_id}")
                continue
            if experiment.get("status") != "complete":
                errors.append(
                    f"{claim_id}: enabled claim depends on non-complete {experiment_id}"
                )
            errors.extend(artifact_errors(experiment_id, experiment))
            errors.extend(coverage_errors(experiment_id, experiment))
    errors.extend(gr00t_selector_reuse_errors(registry))
    errors.extend(gr00t_uniform_w6_replay_errors(registry))
    return sorted(set(errors))


def paper_text() -> str:
    chunks: list[str] = []
    for pattern in MAIN_PAPER_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            chunks.append(f"% {path.relative_to(REPO_ROOT)}\n")
            chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def validate_forbidden_claims(registry: dict[str, Any]) -> list[str]:
    text = paper_text()
    errors: list[str] = []
    for pattern in registry.get("claim_policy", {}).get("forbidden_until_supported", []):
        if re.search(str(pattern), text, flags=re.IGNORECASE):
            errors.append(f"paper contains unsupported claim pattern: {pattern}")
    return errors


def report(registry: dict[str, Any]) -> dict[str, Any]:
    experiments = registry.get("experiments") or {}
    by_status: dict[str, list[str]] = {}
    for experiment_id, experiment in experiments.items():
        by_status.setdefault(str(experiment.get("status", "unknown")), []).append(experiment_id)
    errors = validate_registry(registry) + validate_forbidden_claims(registry)
    return {
        "schema_version": 1,
        "valid": not errors,
        "selector_sha256": registry.get("runtime_selector", {}).get("sha256"),
        "experiments_by_status": {
            status: sorted(values) for status, values in sorted(by_status.items())
        },
        "errors": sorted(set(errors)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--allow-unsupported-paper-draft",
        action="store_true",
        help="Validate evidence artifacts but do not fail on forbidden paper text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = read_json(resolve(args.registry))
    result = report(registry)
    if args.allow_unsupported_paper_draft:
        result["errors"] = validate_registry(registry)
        result["valid"] = not result["errors"]
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.report:
        path = resolve(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
