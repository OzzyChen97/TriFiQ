#!/usr/bin/env python3
"""Freeze the CS+CKA ratio from proxy reports and the preregistered dev set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_RATIOS = [8, 16, 32, 64, 128, 256]
REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_METRIC = REPO_ROOT / "scripts/tools/pi05_func_metrics.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_row(path: Path, *, ratio: int | None, config: str) -> dict:
    path = path.expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("complete") is not True:
        raise SystemExit(f"incomplete adjudication report: {path}")
    metric_hash = sha256_file(FINAL_METRIC)
    if report.get("meta", {}).get("functional_metric_sha256") != metric_hash:
        raise SystemExit(f"stale functional metric in adjudication report: {path}")
    plan_path = Path(report["final_plan_path"]).expanduser().resolve()
    if not plan_path.is_file() or report.get("final_plan_sha256") != sha256_file(plan_path):
        raise SystemExit(f"adjudication report/plan artifact mismatch: {path}")
    return {
        "ratio": ratio,
        "d_func": float(report["final"]["d_func"]),
        "d_solver": float(report["final"]["d_solver"]),
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "report": str(path),
        "report_sha256": sha256_file(path),
        "functional_metric_sha256": metric_hash,
        "source": report["meta"]["final_source"],
        "config": config,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", required=True,
                   help="Directory containing gr00t_quant_plan_..._<ratio>to1_adjudicated.report.json")
    p.add_argument(
        "--ratios", default=",".join(str(value) for value in DEFAULT_RATIOS),
        help="Comma-separated CKA:CS ratios to rank.",
    )
    p.add_argument(
        "--dev-summary", action="append", default=[],
        help="One or more matrix summaries; config ids must be unique across files.",
    )
    p.add_argument(
        "--cka-only-report", default=None,
        help="Optional adjudication report for the lambda_cs=0 boundary candidate.",
    )
    p.add_argument(
        "--proxy-dev-topk", type=int, default=0,
        help=(
            "If positive, development candidates are the lowest-D_func Top-K plus "
            "the transferred ratio and CKA-only boundary. Zero preserves legacy all-ratio dev."
        ),
    )
    p.add_argument("--transferred-ratio", type=int, default=16)
    p.add_argument("--out", required=True)
    return p.parse_args()


def find_report(root: Path, ratio: int) -> Path:
    matches = sorted(root.glob(f"*cscka_{ratio}to1_adjudicated.report.json"))
    if len(matches) != 1:
        raise SystemExit(f"ratio {ratio}: expected one report under {root}, got {matches}")
    return matches[0]


def main() -> None:
    args = parse_args()
    root = Path(args.reports_dir)
    try:
        ratios = [int(value.strip()) for value in args.ratios.split(",") if value.strip()]
    except ValueError as exc:
        raise SystemExit("--ratios must contain integers") from exc
    if not ratios or len(ratios) != len(set(ratios)):
        raise SystemExit("--ratios must contain unique values")
    proxy = []
    for ratio in ratios:
        path = find_report(root, ratio)
        proxy.append(report_row(path, ratio=ratio, config=f"cscka_{ratio}to1"))
    if args.cka_only_report:
        path = Path(args.cka_only_report)
        proxy.append(report_row(path, ratio=None, config="ckaonly"))
    proxy.sort(key=lambda row: (
        row["d_func"], -(row["ratio"] if row["ratio"] is not None else float("inf"))
    ))
    if args.proxy_dev_topk < 0:
        raise SystemExit("--proxy-dev-topk must be non-negative")
    if args.proxy_dev_topk:
        wanted = {
            row["config"] for row in proxy[: args.proxy_dev_topk]
        }
        wanted.add(f"cscka_{args.transferred_ratio}to1")
        if args.cka_only_report:
            wanted.add("ckaonly")
        dev_candidates = [row for row in proxy if row["config"] in wanted]
    else:
        dev_candidates = list(proxy)
    decision = {
        "functional_metric_sha256": sha256_file(FINAL_METRIC),
        "proxy_ranking": proxy,
        "dev_candidates": dev_candidates,
    }

    if args.dev_summary:
        config_rows = {}
        dev_summary_provenance = []
        for summary_path in args.dev_summary:
            resolved_summary = Path(summary_path).expanduser().resolve()
            summary = json.loads(resolved_summary.read_text(encoding="utf-8"))
            if summary.get("complete") is not True:
                raise SystemExit(f"incomplete development summary: {resolved_summary}")
            if set(summary.get("development_tasks", [])) != {
                "OpenCabinet",
                "OpenStandMixerHead",
                "PickPlaceDrawerToCounter",
                "CoffeeSetupMug",
            }:
                raise SystemExit(f"wrong development task partition: {resolved_summary}")
            if summary.get("trial_seeds") != list(range(50)):
                raise SystemExit(f"wrong development seeds: {resolved_summary}")
            overlap = set(config_rows) & set(summary["configs"])
            if overlap:
                raise SystemExit(f"duplicate configs across dev summaries: {sorted(overlap)}")
            config_rows.update(summary["configs"])
            dev_summary_provenance.append(
                {
                    "path": str(resolved_summary),
                    "sha256": sha256_file(resolved_summary),
                    "equivalence_manifest_sha256": summary.get(
                        "equivalence_manifest_sha256"
                    ),
                }
            )
        dev_rows = []
        for row in dev_candidates:
            cid = row["config"]
            if cid not in config_rows:
                raise SystemExit(f"dev summary lacks {cid}")
            macro_sr = config_rows[cid].get("task_macro_sr")
            if macro_sr is None:
                macro_sr = config_rows[cid]["all18_task_macro_sr"]
            dev_rows.append({
                **row,
                "dev_task_macro_sr": float(macro_sr),
            })
        # Preregistered tie break: dev macro SR, then D_func, then more CKA.
        dev_rows.sort(key=lambda row: (
            -row["dev_task_macro_sr"], row["d_func"],
            -(row["ratio"] if row["ratio"] is not None else float("inf")),
        ))
        decision["dev_ranking"] = dev_rows
        decision["dev_summaries"] = dev_summary_provenance
        decision["selected"] = dev_rows[0]
        decision["selection_rule"] = (
            "max dev task-macro SR; tie -> min D_func; tie -> larger CKA:CS ratio; "
            "CKA-only is the infinite-ratio (lambda_cs=0) boundary"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
