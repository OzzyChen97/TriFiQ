#!/usr/bin/env python3
"""Freeze the GR00T N1.5 final CKA:CS ratio for the aligned π0.5 port.

The ratio is not re-tuned on π0.5 RoboCasa rollouts.  This keeps the port a
true architecture transfer: GR00T selects 16:1 on its preregistered four-task
development matrix, while π0.5 only reruns architecture-specific sensitivity
and true-mixed TopK adjudication under the same target/execute-16/d4 protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ALIGNED_ROOT = REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned"
DEFAULT_GR00T_SELECTION = REPO_ROOT / "runs/robocasa365_cs_loss/final_selection_official50.json"
DEFAULT_PI05_REPORT = ALIGNED_ROOT / "adjudication/pi05_cscka_16to1_d4.report.json"
FINAL_METRIC = REPO_ROOT / "scripts/tools/pi05_func_metrics.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def executable_matrix(plan: dict[str, Any]) -> dict[str, Any]:
    layers = plan.get("layers") or {}
    if len(layers) != 180:
        raise ValueError(f"π0.5 adjudicated plan must contain 180 layers, got {len(layers)}")
    return {
        "layers": {
            name: {
                "bits": int(row.get("bits", 0) or 0),
                "group": int(row.get("group", 0) or 0),
                "skip": bool(row.get("skip", False)),
            }
            for name, row in sorted(layers.items())
        },
        "packdirs": plan.get("packdirs") or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gr00t-selection", default=str(DEFAULT_GR00T_SELECTION))
    parser.add_argument("--pi05-report", default=str(DEFAULT_PI05_REPORT))
    parser.add_argument(
        "--selection-out",
        default=str(ALIGNED_ROOT / "selection/final_ratio_selection.json"),
    )
    parser.add_argument(
        "--dev-summary-out",
        default=str(ALIGNED_ROOT / "selection/dev_summary.json"),
    )
    parser.add_argument(
        "--equivalence-out",
        default=str(ALIGNED_ROOT / "selection/executable_equivalence.json"),
    )
    args = parser.parse_args()

    metric_hash = sha256_file(FINAL_METRIC)
    source_path = Path(args.gr00t_selection).expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_selected = source.get("selected") or {}
    if source_selected.get("ratio") != 16 or source_selected.get("config") != "cscka_16to1":
        raise ValueError("authoritative GR00T final selection is not CKA:CS=16:1")
    if source_selected.get("dev_task_macro_sr") is None:
        raise ValueError("authoritative GR00T selection lacks development-set evidence")
    expected_rule = (
        "max dev task-macro SR; tie -> min D_func; tie -> larger CKA:CS ratio; "
        "CKA-only is the infinite-ratio (lambda_cs=0) boundary"
    )
    if source.get("selection_rule") != expected_rule:
        raise ValueError("authoritative GR00T ratio-selection rule changed")
    source_plan = Path(source_selected["plan"]).expanduser().resolve()
    source_report = Path(source_selected["report"]).expanduser().resolve()

    pi05_report_path = Path(args.pi05_report).expanduser().resolve()
    report = json.loads(pi05_report_path.read_text(encoding="utf-8"))
    if report.get("complete") is not True:
        raise ValueError("π0.5 TopK report is incomplete")
    if report.get("meta", {}).get("functional_metric_sha256") != metric_hash:
        raise ValueError("π0.5 TopK report uses a stale functional metric")
    if int(report.get("meta", {}).get("n_obs", 0)) != 16:
        raise ValueError("π0.5 TopK report is not the aligned n_obs=16 adjudication")
    pi05_plan_path = Path(report["final_plan_path"]).expanduser().resolve()
    if report.get("final_plan_sha256") != sha256_file(pi05_plan_path):
        raise ValueError("π0.5 TopK report/final-plan hash mismatch")
    plan = json.loads(pi05_plan_path.read_text(encoding="utf-8"))
    meta = plan.get("meta") or {}
    if meta.get("adjudicated") is not True:
        raise ValueError("π0.5 plan has not passed true-mixed TopK adjudication")
    if float((meta.get("lambda") or {}).get("cka", 0.0)) != 16.0:
        raise ValueError("π0.5 adjudicated plan does not use lambda_cka=16")
    if float((meta.get("lambda") or {}).get("cs", 0.0)) != 1.0:
        raise ValueError("π0.5 adjudicated plan does not use lambda_cs=1")
    matrix = executable_matrix(plan)
    wrapped_layers = sum(
        not row["skip"] and row["bits"] == 4 for row in matrix["layers"].values()
    )
    if wrapped_layers <= 0:
        raise ValueError("π0.5 transferred plan quantizes no layers")

    equivalence_path = Path(args.equivalence_out).expanduser().resolve()
    equivalence = {
        "schema_version": 2,
        "complete": True,
        "kind": "pi05_gr00t_n15_fixed_ratio_transfer",
        "selection_basis": "transferred_gr00t_n15_final_ratio",
        "source_model": "GR00T N1.5",
        "target_model": "pi0.5",
        "transferred_ratio": 16,
        "pi05_ratio_tuning_rollouts_performed": False,
        "source_selection_path": str(source_path),
        "source_selection_sha256": sha256_file(source_path),
        "source_plan_path": str(source_plan),
        "source_plan_sha256": sha256_file(source_plan),
        "source_report_path": str(source_report),
        "source_report_sha256": sha256_file(source_report),
        "pi05_topk_report_path": str(pi05_report_path),
        "pi05_topk_report_sha256": sha256_file(pi05_report_path),
        "pi05_plan_path": str(pi05_plan_path),
        "pi05_plan_sha256": sha256_file(pi05_plan_path),
        "pi05_executable_layer_matrix_sha256": canonical_hash(matrix),
        "pi05_wrapped_layers": wrapped_layers,
        "functional_metric_sha256": metric_hash,
        "protocol": {
            "split": "target",
            "n_action_steps": 16,
            "flow_steps": 4,
            "paired_noise": "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1",
        },
    }
    atomic_json(equivalence_path, equivalence)

    dev_summary_path = Path(args.dev_summary_out).expanduser().resolve()
    dev_summary = {
        "schema_version": 2,
        "complete": True,
        "kind": "transferred_gr00t_n15_ratio_selection_evidence",
        "selection_basis": "transferred_gr00t_n15_final_ratio",
        "pi05_ratio_tuning_rollouts_performed": False,
        "source_selection_path": str(source_path),
        "source_selection_sha256": sha256_file(source_path),
        "source_selection_rule": source["selection_rule"],
        "source_selected_ratio": 16,
        "source_selected_config": "cscka_16to1",
        "source_gr00t_dev_task_macro_sr": float(source_selected["dev_task_macro_sr"]),
        "equivalence_manifest_path": str(equivalence_path),
        "equivalence_manifest_sha256": sha256_file(equivalence_path),
        "functional_metric_sha256": metric_hash,
    }
    atomic_json(dev_summary_path, dev_summary)

    selection_path = Path(args.selection_out).expanduser().resolve()
    selection = {
        "schema_version": 2,
        "complete": True,
        "selection_basis": "transferred_gr00t_n15_final_ratio",
        "selection_rule": (
            "transfer the frozen GR00T N1.5 final CKA:CS ratio exactly; "
            "run architecture-specific pi0.5 TopK adjudication without ratio retuning"
        ),
        "functional_metric_sha256": metric_hash,
        "source_selection_path": str(source_path),
        "source_selection_sha256": sha256_file(source_path),
        "dev_summaries": [
            {"path": str(dev_summary_path), "sha256": sha256_file(dev_summary_path)}
        ],
        "selected": {
            "ratio": 16,
            "config": "cscka_16to1",
            "plan": str(pi05_plan_path),
            "plan_sha256": sha256_file(pi05_plan_path),
            "report": str(pi05_report_path),
            "report_sha256": sha256_file(pi05_report_path),
            "d_func": float(report["final"]["d_func"]),
            "d_solver": float(report["final"]["d_solver"]),
            "source": report["meta"]["final_source"],
            "source_gr00t_dev_task_macro_sr": float(source_selected["dev_task_macro_sr"]),
        },
    }
    atomic_json(selection_path, selection)
    print(
        json.dumps(
            {
                "selection": str(selection_path),
                "selection_sha256": sha256_file(selection_path),
                "transferred_ratio": 16,
                "pi05_plan": str(pi05_plan_path),
                "pi05_wrapped_layers": wrapped_layers,
                "pi05_ratio_tuning_rollouts_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
