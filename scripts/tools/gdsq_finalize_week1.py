#!/usr/bin/env python3
"""Promote strictly complete week-1 artifacts into the claim-evidence registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "docs/gdsq_vla_cvpr2026/experiment_registry.json"
ROOT = REPO_ROOT / "runs/gdsq_week1_preregistered_v1"
EXECUTION = ROOT / "execution"
_TREE_CACHE: dict[tuple[str, tuple[str, ...]], tuple[str, int, int]] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def tree_digest(record: dict[str, Any]) -> tuple[str, int, int]:
    root = resolve(str(record["path"]))
    require(root.is_dir(), f"missing frozen tree: {root}")
    suffixes = tuple(sorted(str(value) for value in record.get("included_suffixes", [])))
    cache_key = (str(root), suffixes)
    if cache_key in _TREE_CACHE:
        return _TREE_CACHE[cache_key]
    allowed = set(suffixes)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (not allowed or path.suffix in allowed)
        and (not record.get("excludes_bytecode_caches") or "__pycache__" not in path.parts)
    )
    digest = hashlib.sha256()
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        file_sha = sha256_file(path)
        size = path.stat().st_size
        digest.update(
            relative + b"\0" + file_sha.encode() + b"\0" + str(size).encode() + b"\n"
        )
        total += size
    result = (digest.hexdigest(), len(files), total)
    _TREE_CACHE[cache_key] = result
    return result


def require_file_record(record: dict[str, Any], label: str) -> None:
    path = resolve(str(record.get("path", "")))
    require(path.is_file(), f"missing {label}: {path}")
    require(sha256_file(path) == record.get("sha256"), f"{label} SHA drift")


def require_tree_record(record: dict[str, Any], label: str) -> None:
    actual_sha, actual_files, actual_bytes = tree_digest(record)
    require(actual_sha == record.get("sha256_tree"), f"{label} tree SHA drift")
    require(actual_files == record.get("files"), f"{label} file-count drift")
    require(actual_bytes == record.get("bytes"), f"{label} byte-count drift")


def require_gr00t_manifest_v2(
    path: Path,
    *,
    expected_tasks: int,
    expected_configs: set[str],
    expected_seeds: list[int] | None = None,
    diagnostic_only: bool = False,
) -> tuple[dict[str, Any], str]:
    expected_seeds = list(range(50)) if expected_seeds is None else expected_seeds
    manifest = load(path)
    require(manifest.get("schema_version") == 2, f"{path}: provenance-v2 required")
    require(manifest.get("phase") == "formal", f"{path}: formal phase required")
    require(
        manifest.get("diagnostic_only") is diagnostic_only,
        f"{path}: diagnostic-only flag drift",
    )
    require(len(manifest.get("tasks", [])) == expected_tasks, f"{path}: task-count drift")
    require(manifest.get("seeds") == expected_seeds, f"{path}: seed schedule drift")
    require(
        {str(row.get("id")) for row in manifest.get("configs", [])} == expected_configs,
        f"{path}: configuration set drift",
    )
    protocol = manifest.get("protocol") or {}
    protocol_checks = {
        "paired_noise": protocol.get("paired_action_noise") is True,
        "noise_scheme": protocol.get("action_noise_scheme")
        == "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1",
        "denoising_steps": protocol.get("denoising_steps") == 4,
        "n_action_steps": protocol.get("n_action_steps") == 16,
        "render": protocol.get("render") is True,
        "scenarios": protocol.get("scenarios_per_task") == len(expected_seeds),
    }
    require(all(protocol_checks.values()), f"{path}: protocol drift {protocol_checks}")
    provenance = manifest.get("formal_provenance") or {}
    require(
        provenance.get("kind") == "robocasa365_formal_launch_provenance"
        and provenance.get("schema_version") == 2,
        f"{path}: missing formal provenance v2",
    )
    require_tree_record(provenance["checkpoint_tree"], f"{path}: checkpoint")
    for name, record in provenance.get("sources", {}).items():
        require_file_record(record, f"{path}: source {name}")
    for name, record in provenance.get("source_trees", {}).items():
        require_tree_record(record, f"{path}: source tree {name}")
    environment = provenance.get("environment") or {}
    require_file_record(environment["environment_spec"], f"{path}: environment spec")
    require_file_record(environment["requirements"], f"{path}: requirements")
    require(len(str(environment.get("pip_freeze_sha256", ""))) == 64, f"{path}: pip hash missing")
    for config in manifest["configs"]:
        for label in ("plan", "act_scale", "act_scale_meta"):
            record = config.get(label)
            if record:
                require_file_record(record, f"{path}: {config['id']} {label}")
        if config.get("packdir"):
            require_tree_record(config["packdir"], f"{path}: {config['id']} pack")
    return manifest, sha256_file(path)


def require_matrix_summary(
    summary: dict[str, Any],
    *,
    manifest_sha: str,
    expected_tasks: int,
    expected_configs: set[str],
    seeds_per_task: int = 50,
) -> None:
    require(summary.get("manifest_sha256") == manifest_sha, "matrix summary manifest drift")
    require(summary.get("bootstrap_draws") == 10_000, "matrix bootstrap drift")
    require(not summary.get("validation_errors"), "matrix contains validation errors")
    require(
        summary.get("primary_scope", {}).get("n_tasks") == expected_tasks,
        "matrix primary task-count drift",
    )
    require(
        summary.get("secondary_scope", {}).get("n_tasks") == expected_tasks,
        "matrix secondary task-count drift",
    )
    require(set(summary.get("configs", {})) == expected_configs, "matrix config-set drift")
    expected_episodes = expected_tasks * seeds_per_task
    for config in expected_configs:
        row = summary["configs"][config]
        require(row.get("episodes") == expected_episodes, f"{config}: episode coverage drift")
        require(len(row.get("per_task", {})) == expected_tasks, f"{config}: per-task drift")


def require_gr00t_preflight(run_dir: Path, expected_configs: set[str]) -> list[dict[str, Any]]:
    manifest_path, summary_path = run_dir / "manifest.json", run_dir / "summary.json"
    _manifest, manifest_sha = require_gr00t_manifest_v2(
        manifest_path,
        expected_tasks=2,
        expected_configs=expected_configs,
        expected_seeds=[0, 1],
        diagnostic_only=True,
    )
    summary = load(summary_path)
    require_matrix_summary(
        summary,
        manifest_sha=manifest_sha,
        expected_tasks=2,
        expected_configs=expected_configs,
        seeds_per_task=2,
    )
    return [artifact(manifest_path), artifact(summary_path)]


def coverage(tasks: int, episodes: int, configurations: int | None = None) -> dict[str, Any]:
    result = {
        "tasks": tasks,
        "seeds_per_task": 50,
        "expected_episodes": episodes,
        "observed_episodes": episodes,
        "missing_episodes": 0,
        "duplicate_episodes": 0,
    }
    if configurations is not None:
        result["configurations"] = configurations
        require(episodes % configurations == 0, "total episodes not divisible by configurations")
        result["expected_episodes_per_config"] = episodes // configurations
        result["observed_episodes_per_config"] = episodes // configurations
    return result


def promote_selector(registry: dict[str, Any]) -> None:
    manifest_path = ROOT / "pi05_selector_official50/manifest.json"
    summary_path = ROOT / "pi05_selector_official50/aggregate/summary.json"
    summary = load(summary_path)
    checks = {
        "complete": summary.get("complete") is True,
        "formal": summary.get("formal_result") is True,
        "expected": summary.get("expected_episodes") == 2500,
        "observed": summary.get("completed_episodes") == 2500,
        "missing": summary.get("missing_episodes") == 0,
        "duplicates": summary.get("duplicate_episodes") == 0,
        "attested": summary.get("selector_attestation", {}).get("validated_rows") == 2500,
        "manifest": summary.get("manifest_sha256") == sha256_file(manifest_path),
    }
    failed = [name for name, valid in checks.items() if not valid]
    require(not failed, f"selector promotion failed: {failed}")
    record = registry["experiments"]["pi05_runtime_selector_official50"]
    record.update(
        {
            "status": "complete",
            "manifest": artifact(manifest_path),
            "summary": artifact(summary_path),
            "coverage": coverage(50, 2500),
            "main_claim_enabled": True,
            "notes": "Strict all-50 selector aggregation complete; every row carries the frozen selector attestation.",
        }
    )


def promote_gr00t_w6(registry: dict[str, Any]) -> None:
    w6_path = EXECUTION / "aggregate/gr00t_uniform_w6/summary.json"
    replay_path = EXECUTION / "audit/gr00t_uniform_w6_replay_audit_v1.json"
    w6 = load(w6_path)
    replay = load(replay_path)
    require(w6.get("n_tasks") == 50 and w6.get("episodes_per_config") == 2500, "GR00T W6 coverage drift")
    require("uniform_w6" in w6.get("configs", {}), "GR00T W6 config missing")
    require(w6.get("bootstrap_draws") == 10_000, "GR00T W6 bootstrap drift")
    require(replay.get("valid") is True, "GR00T W6 replay audit failed")
    require(replay.get("summary", {}).get("byte_exact") is True, "GR00T W6 replay summary differs")
    require(
        replay.get("summary", {}).get("sha256") == sha256_file(w6_path),
        "GR00T W6 replay summary hash drift",
    )
    replay_coverage = replay.get("coverage") or {}
    require(
        replay_coverage.get("expected_episodes")
        == replay_coverage.get("observed_episodes")
        == 2500,
        "GR00T W6 replay coverage drift",
    )
    require(
        replay_coverage.get("missing_episodes") == 0
        and replay_coverage.get("duplicate_episodes") == 0,
        "GR00T W6 replay contains missing/duplicate rows",
    )
    source_by_task_set = {str(row["task_set"]): row for row in w6.get("sources", [])}
    require(set(source_by_task_set) == {"atomic_seen", "composite_seen", "composite_unseen"}, "GR00T W6 source drift")
    w6_preflights = []
    w6_manifests = []
    for task_set, task_count in (("atomic_seen", 18), ("composite_seen", 16), ("composite_unseen", 16)):
        run_dir = resolve(str(source_by_task_set[task_set]["run_dir"]))
        manifest_path = run_dir / "manifest.json"
        manifest = load(manifest_path)
        manifest_sha = sha256_file(manifest_path)
        require(manifest.get("schema_version") == 2, f"{task_set}: manifest schema drift")
        require(manifest.get("phase") == "formal", f"{task_set}: non-formal manifest")
        require(manifest.get("diagnostic_only") is False, f"{task_set}: diagnostic manifest")
        require(len(manifest.get("tasks", [])) == task_count, f"{task_set}: task-count drift")
        require(manifest.get("seeds") == list(range(50)), f"{task_set}: seed schedule drift")
        require(
            {str(row.get("id")) for row in manifest.get("configs", [])} == {"uniform_w6"},
            f"{task_set}: configuration drift",
        )
        require(
            (replay.get("manifests") or {}).get(task_set, {}).get("sha256") == manifest_sha,
            f"{task_set}: replay manifest hash drift",
        )
        require(source_by_task_set[task_set].get("manifest_sha256") == manifest_sha, f"{task_set}: aggregate source drift")
        w6_manifests.append(artifact(manifest_path))
        preflight_dir = EXECUTION / "preflight" / run_dir.name
        preflight_manifest = preflight_dir / "manifest.json"
        preflight_summary = preflight_dir / "summary.json"
        preflight_manifest_sha = sha256_file(preflight_manifest)
        summary = load(preflight_summary)
        require_matrix_summary(
            summary,
            manifest_sha=preflight_manifest_sha,
            expected_tasks=2,
            expected_configs={"uniform_w6"},
            seeds_per_task=2,
        )
        w6_preflights.extend([artifact(preflight_manifest), artifact(preflight_summary)])
    require(w6["configs"]["uniform_w6"].get("episodes") == 2500, "GR00T W6 episode drift")
    w6_record = registry["experiments"]["gr00t_uniform_w6_official50"]
    w6_record.update(
        {
            "status": "complete",
            "summary": artifact(w6_path),
            "replay_audit": artifact(replay_path),
            "manifests": w6_manifests,
            "preflight_artifacts": w6_preflights,
            "coverage": coverage(50, 2500),
            "main_claim_enabled": True,
            "notes": (
                "Strict all-50 Uniform-W6 aggregation complete. A three-manifest replay "
                "audit validates all 2,500 selector-disabled rows and reproduces the frozen "
                "summary byte-for-byte; later launcher/runtime source drift is disclosed "
                "without rewriting the launch manifests."
            ),
        }
    )


def promote_gr00t_p0(registry: dict[str, Any]) -> None:
    promote_gr00t_w6(registry)

    summary_path = EXECUTION / "runs/gr00t_same_budget_controls_primary14/summary.json"
    selection_path = EXECUTION / "selection/gr00t_controls_dev4.json"
    manifest_path = EXECUTION / "runs/gr00t_same_budget_controls_primary14/manifest.json"
    summary, selection = load(summary_path), load(selection_path)
    require(selection.get("heldout_results_read") is False, "GR00T control selection tainted")
    _manifest, control_manifest_sha = require_gr00t_manifest_v2(
        manifest_path,
        expected_tasks=14,
        expected_configs={"search_matched_random", "action_only"},
    )
    require_matrix_summary(
        summary,
        manifest_sha=control_manifest_sha,
        expected_tasks=14,
        expected_configs={"search_matched_random", "action_only"},
    )
    control_preflights = require_gr00t_preflight(
        EXECUTION / "preflight/gr00t_same_budget_controls_primary14",
        {"search_matched_random", "action_only"},
    )
    for experiment_id, config in (
        ("gr00t_search_matched_random", "search_matched_random"),
        ("gr00t_action_only_allocator", "action_only"),
    ):
        require(config in summary.get("configs", {}), f"GR00T control missing: {config}")
        registry["experiments"][experiment_id].update(
            {
                "status": "complete",
                "summary": artifact(summary_path),
                "manifest": artifact(manifest_path),
                "selection": artifact(selection_path),
                "preflight_artifacts": control_preflights,
                "coverage": coverage(14, 700),
                "main_claim_enabled": True,
            }
        )


def promote_pi05_w6(registry: dict[str, Any]) -> None:
    run_dir = EXECUTION / "runs/pi05_uniform_w6_official50"
    manifest_path, summary_path = run_dir / "manifest.json", run_dir / "aggregate/summary.json"
    summary = load(summary_path)
    require(summary.get("complete") is True and summary.get("formal_result") is True, "pi0.5 W6 incomplete")
    require(summary.get("completed_episodes") == 2500, "pi0.5 W6 coverage drift")
    require(summary.get("manifest_sha256") == sha256_file(manifest_path), "pi0.5 W6 manifest drift")
    registry["experiments"]["pi05_uniform_w6_official50"].update(
        {
            "status": "complete",
            "manifest": artifact(manifest_path),
            "summary": artifact(summary_path),
            "coverage": coverage(50, 2500),
            "main_claim_enabled": True,
        }
    )


def promote_gr00t_ablations(registry: dict[str, Any]) -> None:
    run_dir = EXECUTION / "runs/gr00t_core_ablations_primary14"
    summary_path, manifest_path = run_dir / "summary.json", run_dir / "manifest.json"
    summary = load(summary_path)
    statistics_path = EXECUTION / "aggregate/gr00t_core_ablation_statistics.json"
    statistics = load(statistics_path)
    configs = {
        "ablation_cka_only",
        "ablation_cs_only",
        "ablation_weights_uniform",
        "ablation_no_guards",
        "ablation_no_functional_adjudication",
    }
    _manifest, manifest_sha = require_gr00t_manifest_v2(
        manifest_path, expected_tasks=14, expected_configs=configs
    )
    require_matrix_summary(
        summary,
        manifest_sha=manifest_sha,
        expected_tasks=14,
        expected_configs=configs,
    )
    preflight_artifacts = require_gr00t_preflight(
        EXECUTION / "preflight/gr00t_core_ablations_primary14", configs
    )
    require(statistics.get("complete") is True, "ablation statistics incomplete")
    require(len(statistics.get("contrasts", {})) == 5, "five ablation contrasts required")
    require(
        statistics.get("sources", {}).get("ablation_matrix", {}).get("sha256")
        == sha256_file(summary_path),
        "ablation statistics source drift",
    )
    record = registry["experiments"]["gr00t_component_ablation50"]
    record.update(
        {
            "status": "complete",
            "summary": artifact(summary_path),
            "statistics": artifact(statistics_path),
            "manifest": artifact(manifest_path),
            "preflight_artifacts": preflight_artifacts,
            "coverage": coverage(14, 3500, configurations=5),
            "main_claim_enabled": False,
        }
    )


def promote_pi05_controls(registry: dict[str, Any]) -> None:
    selection_path = EXECUTION / "selection/pi05_controls_dev4.json"
    selection = load(selection_path)
    require(selection.get("heldout_results_read") is False, "pi0.5 control selection tainted")
    for experiment_id, config in (
        ("pi05_search_matched_random", "search_matched_random"),
        ("pi05_action_only_allocator", "action_only"),
    ):
        run_dir = EXECUTION / "runs/pi05_controls_heldout46" / config
        manifest_path, summary_path = run_dir / "manifest.json", run_dir / "aggregate/summary.json"
        summary = load(summary_path)
        require(summary.get("complete") is True and summary.get("formal_result") is True, f"{config} incomplete")
        require(summary.get("completed_episodes") == 2300, f"{config} coverage drift")
        require(summary.get("manifest_sha256") == sha256_file(manifest_path), f"{config} manifest drift")
        registry["experiments"][experiment_id].update(
            {
                "status": "complete",
                "manifest": artifact(manifest_path),
                "summary": artifact(summary_path),
                "selection": artifact(selection_path),
                "coverage": coverage(46, 2300),
                "main_claim_enabled": True,
            }
        )


def promote_joint_statistics(registry: dict[str, Any]) -> None:
    path = EXECUTION / "aggregate/week1_joint_statistics.json"
    summary = load(path)
    require(summary.get("complete") is True, "joint statistics incomplete")
    require(len(summary.get("contrasts", {})) == 6, "six preregistered contrasts required")
    record = registry["paper_claims"]["same_budget_superiority"]
    record["joint_statistics"] = artifact(path)
    record["enabled"] = bool(
        summary.get("same_budget_superiority_claim_enabled_across_architectures")
    )
    record["reporting_mode"] = {
        model: gate["reporting_mode"]
        for model, gate in summary["strongest_baseline_gates"].items()
    }


def update_claim_dependencies(registry: dict[str, Any]) -> None:
    experiments = registry["experiments"]
    selector_ids = ["gr00t_runtime_selector_official50", "pi05_runtime_selector_official50"]
    registry["paper_claims"]["final_runtime_selector_results"]["enabled"] = all(
        experiments[name].get("status") == "complete" for name in selector_ids
    )
    # Superiority is intentionally left closed here.  Only the joint-statistics
    # artifact may open it after the strongest-baseline CI/Holm rule passes.
    if not registry["paper_claims"]["same_budget_superiority"].get("joint_statistics"):
        registry["paper_claims"]["same_budget_superiority"]["enabled"] = False


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("selector", "gr00t-w6", "gr00t-p0", "pi05-w6", "gr00t-ablations", "pi05-controls", "joint", "all"),
    )
    parser.add_argument("--check", action="store_true", help="Validate without writing the registry")
    args = parser.parse_args()
    registry = load(REGISTRY)
    phases = {
        "selector": promote_selector,
        "gr00t-w6": promote_gr00t_w6,
        "gr00t-p0": promote_gr00t_p0,
        "pi05-w6": promote_pi05_w6,
        "gr00t-ablations": promote_gr00t_ablations,
        "pi05-controls": promote_pi05_controls,
        "joint": promote_joint_statistics,
    }
    selected = list(phases) if args.phase == "all" else [args.phase]
    for phase in selected:
        phases[phase](registry)
    update_claim_dependencies(registry)
    if not args.check:
        atomic_write(REGISTRY, registry)
    print(json.dumps({"valid": True, "phases": selected, "written": not args.check}, indent=2))


if __name__ == "__main__":
    main()
