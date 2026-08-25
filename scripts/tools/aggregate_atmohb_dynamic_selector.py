#!/usr/bin/env python3
"""Retrospectively replay a dynamic ATM/OHB selector on completed ablation rows.

This is a sanity check only.  The selector must already be frozen without oracle
thresholds; this script simply scores the chosen mapping against rows that are
available in the existing 50-seed ablation/base summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_PROVENANCE = REPO_ROOT / "runs/atmohb_seeded_atomic_10x10/final/provenance"
DEFAULT_GR00T_SUMMARY = FINAL_PROVENANCE / "gr00t_development_ablation_summary.json"
DEFAULT_PI05_SUMMARY = FINAL_PROVENANCE / "pi05_development_ablation_summary.json"

MODEL_SUMMARY_ARGS = {
    "gr00t": "gr00t_summary",
    "pi05": "pi05_summary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--gr00t-summary", default=str(DEFAULT_GR00T_SUMMARY))
    parser.add_argument("--pi05-summary", default=str(DEFAULT_PI05_SUMMARY))
    parser.add_argument(
        "--runtime-validation-root",
        default=None,
        help="Optional held-out validation root containing pi05 runtime-selector JSONL rows.",
    )
    parser.add_argument(
        "--runtime-validation-model-id",
        default="pi05",
        choices=sorted(MODEL_SUMMARY_ARGS),
        help="Model id to use when summarizing --runtime-validation-root.",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def per_task_sr(summary: dict[str, Any], config_id: str, task: str) -> float | None:
    config = (summary.get("configs") or {}).get(config_id) or {}
    per_task = config.get("per_task") or {}
    if task in per_task and isinstance(per_task[task], dict):
        return float(per_task[task]["sr"])
    per_task_sr_map = config.get("per_task_sr") or {}
    if task in per_task_sr_map:
        return float(per_task_sr_map[task])
    per_task_details = config.get("per_task_details") or {}
    if task in per_task_details:
        return float(per_task_details[task]["sr"])
    return None


def paired_delta(values_a: dict[str, float], values_b: dict[str, float]) -> dict[str, Any]:
    tasks = sorted(set(values_a) & set(values_b))
    deltas = {task: values_a[task] - values_b[task] for task in tasks}
    macro_delta = sum(deltas.values()) / len(deltas) if deltas else None
    return {"task_macro_delta": macro_delta, "per_task_delta": deltas, "n_tasks": len(tasks)}


def enforce_no_oracle(selector: dict[str, Any]) -> None:
    contract = selector.get("no_oracle_contract") or {}
    if contract.get("analysis_only_derived_threshold_flag") is True:
        raise ValueError("selector declares analysis_only-derived thresholds")
    if contract.get("uses_analysis_only_for_thresholds") is True:
        raise ValueError("selector declares analysis_only threshold use")
    if contract.get("uses_heldout_labels_for_thresholds") is True:
        raise ValueError("selector declares held-out label threshold use")
    for model_id, model in (selector.get("models") or {}).items():
        selected = model.get("selected") or {}
        tasks = model.get("tasks") or {}
        for task, decision in tasks.items():
            selected_variant = decision.get("selected_variant")
            if isinstance(selected_variant, list):
                raise ValueError(f"{model_id}/{task}: multiple selected variants")
            if task in selected and selected[task] != selected_variant:
                raise ValueError(f"{model_id}/{task}: selected mapping disagrees with task decision")
        if len(selected) != len(set(selected)):
            raise ValueError(f"{model_id}: duplicate task keys in selected mapping")


def replay_model(model_id: str, model: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    variant_config_ids = model.get("variant_config_ids") or {}
    baseline_config_id = model.get("baseline_config_id") or variant_config_ids.get("baseline")
    if not baseline_config_id:
        raise ValueError(f"{model_id}: missing baseline config id")

    selected_srs: dict[str, float] = {}
    baseline_srs: dict[str, float] = {}
    variant_counts = {variant: 0 for variant in variant_config_ids}
    missing: list[dict[str, str]] = []
    task_rows: dict[str, Any] = {}
    for task, decision in sorted((model.get("tasks") or {}).items()):
        variant = decision.get("selected_variant")
        if variant not in variant_config_ids:
            raise ValueError(f"{model_id}/{task}: unknown variant {variant!r}")
        variant_counts[variant] = variant_counts.get(variant, 0) + 1
        config_id = variant_config_ids[variant]
        selected_sr = per_task_sr(summary, config_id, task)
        baseline_sr = per_task_sr(summary, baseline_config_id, task)
        if selected_sr is None or baseline_sr is None:
            missing.append({"task": task, "variant": variant, "config_id": config_id})
            continue
        selected_srs[task] = selected_sr
        baseline_srs[task] = baseline_sr
        task_rows[task] = {
            "selected_variant": variant,
            "selected_config_id": config_id,
            "selected_sr": selected_sr,
            "baseline_config_id": baseline_config_id,
            "baseline_sr": baseline_sr,
            "delta_vs_baseline": selected_sr - baseline_sr,
        }

    paired = paired_delta(selected_srs, baseline_srs)
    selected_macro = sum(selected_srs.values()) / len(selected_srs) if selected_srs else None
    baseline_macro = sum(baseline_srs.values()) / len(baseline_srs) if baseline_srs else None
    return {
        "label": "RETROSPECTIVE_REPLAY_NOT_FINAL_VALIDATION",
        "baseline_config_id": baseline_config_id,
        "selected_variant_counts": variant_counts,
        "available_completed_task_count": len(selected_srs),
        "missing_task_count": len(missing),
        "missing": missing,
        "selected_task_macro_sr": selected_macro,
        "baseline_task_macro_sr": baseline_macro,
        "paired_delta_vs_baseline": paired,
        "tasks": task_rows,
    }


def build_replay(selector: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    enforce_no_oracle(selector)
    models = {}
    for model_id, model in sorted((selector.get("models") or {}).items()):
        if model_id not in summaries:
            raise ValueError(f"no summary supplied for model {model_id}")
        models[model_id] = replay_model(model_id, model, summaries[model_id])
    return {
        "schema_version": 1,
        "kind": "atmohb_dynamic_selector_retrospective_replay",
        "label": "RETROSPECTIVE_REPLAY_NOT_FINAL_VALIDATION",
        "sanity_check_only": True,
        "selector_rule": selector.get("rule_name"),
        "models": models,
    }


def build_runtime_validation(selector: dict[str, Any], root: Path, model_id: str = "pi05") -> dict[str, Any]:
    enforce_no_oracle(selector)
    model = (selector.get("models") or {}).get(model_id) or {}
    task_decisions = model.get("tasks") or {}
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    source_files: list[str] = []
    for path in sorted((root / model_id).glob("**/*.jsonl")):
        source_files.append(str(path))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "complete":
                raise ValueError(f"{path}:{line_number}: non-complete row")
            if row.get("runtime_selector_enabled") is not True:
                raise ValueError(f"{path}:{line_number}: runtime selector was not enabled")
            task = str(row.get("task"))
            seed = int(row.get("seed"))
            expected = task_decisions.get(task)
            if not expected:
                raise ValueError(f"{path}:{line_number}: task absent from selector: {task}")
            if row.get("selected_variant") != expected.get("selected_variant"):
                raise ValueError(
                    f"{path}:{line_number}: selected_variant {row.get('selected_variant')!r} "
                    f"!= frozen {expected.get('selected_variant')!r}"
                )
            if row.get("selected_config_id") != expected.get("selected_config_id"):
                raise ValueError(
                    f"{path}:{line_number}: selected_config_id {row.get('selected_config_id')!r} "
                    f"!= frozen {expected.get('selected_config_id')!r}"
                )
            key = (task, seed)
            if key in rows:
                raise ValueError(f"duplicate runtime selector row for {key}")
            rows[key] = row
    per_task: dict[str, dict[str, Any]] = {}
    for (task, _seed), row in rows.items():
        bucket = per_task.setdefault(
            task,
            {
                "selected_variant": row.get("selected_variant"),
                "selected_config_id": row.get("selected_config_id"),
                "successes": 0,
                "episodes": 0,
            },
        )
        bucket["successes"] += int(bool(row.get("success")))
        bucket["episodes"] += 1
    for bucket in per_task.values():
        bucket["sr"] = bucket["successes"] / bucket["episodes"] if bucket["episodes"] else None
    task_macro = (
        sum(float(bucket["sr"]) for bucket in per_task.values()) / len(per_task)
        if per_task
        else None
    )
    return {
        "schema_version": 1,
        "kind": "atmohb_runtime_selector_validation",
        "model_id": model_id,
        "source_root": str(root),
        "source_files": source_files,
        "completed_episodes": len(rows),
        "completed_tasks": len(per_task),
        "task_macro_sr": task_macro,
        "per_task": dict(sorted(per_task.items())),
    }


def main() -> None:
    args = parse_args()
    selector_path = Path(args.selector).resolve()
    selector = read_json(selector_path)
    summaries = {
        "gr00t": read_json(Path(args.gr00t_summary).resolve()),
        "pi05": read_json(Path(args.pi05_summary).resolve()),
    }
    payload = build_replay(selector, summaries)
    if args.runtime_validation_root:
        payload["runtime_validation"] = build_runtime_validation(
            selector,
            Path(args.runtime_validation_root).resolve(),
            model_id=args.runtime_validation_model_id,
        )
    payload["source_selector"] = str(selector_path)
    payload["source_summaries"] = {
        "gr00t": str(Path(args.gr00t_summary).resolve()),
        "pi05": str(Path(args.pi05_summary).resolve()),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "label": payload["label"], "models": sorted(payload["models"])}))


if __name__ == "__main__":
    main()