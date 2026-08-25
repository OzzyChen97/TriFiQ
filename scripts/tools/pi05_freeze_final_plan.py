#!/usr/bin/env python3
"""Freeze the development-selected π0.5 GDSQ plan into one immutable artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    selection_path = Path(args.selection).expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    metric_path = Path(__file__).with_name("pi05_func_metrics.py")
    metric_hash = sha256_file(metric_path)
    if selection.get("functional_metric_sha256") != metric_hash:
        raise ValueError("ratio selection was produced by a stale functional metric")
    selected = selection.get("selected")
    selection_basis = selection.get("selection_basis", "pi05_development_rollout")
    transferred = selection_basis == "transferred_gr00t_n15_final_ratio"
    if not isinstance(selected, dict):
        raise ValueError("ratio selection has no completed decision")
    if transferred:
        if selected.get("ratio") != 16 or selected.get("source_gr00t_dev_task_macro_sr") is None:
            raise ValueError("GR00T-transferred selection lacks its frozen 16:1 source evidence")
        source_selection = Path(selection.get("source_selection_path", "")).expanduser().resolve()
        if selection.get("source_selection_sha256") != sha256_file(source_selection):
            raise ValueError("GR00T source ratio-selection hash mismatch")
    elif selected.get("dev_task_macro_sr") is None:
        raise ValueError("ratio selection has no completed development-set decision")
    if "selection_rule" not in selection:
        raise ValueError("ratio selection is missing its preregistered rule")
    dev_summaries = selection.get("dev_summaries") or []
    if not dev_summaries:
        raise ValueError("ratio selection is missing development-summary provenance")
    for row in dev_summaries:
        path = Path(row["path"]).expanduser().resolve()
        if row.get("sha256") != sha256_file(path):
            raise ValueError(f"development summary hash mismatch: {path}")
    plan_path = Path(selected["plan"]).expanduser().resolve()
    report_path = Path(selected["report"]).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("complete") is not True:
        raise ValueError("selected TopK report is incomplete")
    if Path(report["final_plan_path"]).resolve() != plan_path:
        raise ValueError("selection/report final-plan paths differ")
    if selected.get("report_sha256") != sha256_file(report_path):
        raise ValueError("ratio selection/report hash mismatch")
    if selected.get("plan_sha256") != sha256_file(plan_path):
        raise ValueError("ratio selection/plan hash mismatch")
    if report.get("meta", {}).get("functional_metric_sha256") != metric_hash:
        raise ValueError("selected TopK report uses a stale functional metric")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("meta", {}).get("adjudicated") is not True:
        raise ValueError("selected plan has not passed TopK adjudication")
    quantized = sum(
        not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
        for row in plan["layers"].values()
    )
    if len(plan["layers"]) != 180 or quantized <= 0:
        raise ValueError("selected plan has an invalid π0.5 layer inventory")
    plan["meta"].update(
        {
            "kind": "gdsq_vla_pi05_faithful_final_frozen",
            "ratio_selection_path": str(selection_path),
            "ratio_selection_sha256": sha256_file(selection_path),
            "selected_ratio": selected.get("ratio"),
            "selected_config": selected["config"],
            "selection_basis": selection_basis,
            "development_task_macro_sr": selected.get("dev_task_macro_sr"),
            "source_gr00t_dev_task_macro_sr": selected.get(
                "source_gr00t_dev_task_macro_sr"
            ),
            "pi05_ratio_tuning_rollouts_performed": not transferred,
            "ratio_selection_rule": selection["selection_rule"],
            "development_summaries": dev_summaries,
            "functional_metric_path": str(metric_path.resolve()),
            "functional_metric_sha256": metric_hash,
            "selected_topk_report_path": str(report_path),
            "selected_topk_report_sha256": sha256_file(report_path),
            "selected_adjudicated_plan_path": str(plan_path),
            "selected_adjudicated_plan_sha256": sha256_file(plan_path),
            "quantized_layers": quantized,
            "retained_fp16_layers": 180 - quantized,
            "frozen_before_table1_test": True,
            "requires_fresh_plan_specific_a8": True,
            "requires_fresh_plan_specific_atm_ohb": True,
        }
    )
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": sha256_file(output),
                "selected_ratio": selected.get("ratio"),
                "selection_basis": selection_basis,
                "dev_task_macro_sr": selected.get("dev_task_macro_sr"),
                "source_gr00t_dev_task_macro_sr": selected.get(
                    "source_gr00t_dev_task_macro_sr"
                ),
                "quantized_layers": quantized,
                "retained_fp16_layers": 180 - quantized,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
