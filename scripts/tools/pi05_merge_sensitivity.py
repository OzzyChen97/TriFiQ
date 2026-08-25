#!/usr/bin/env python3
"""Strictly merge the six π0.5 sensitivity shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from pi05_func_metrics import FUNCTIONAL_FORMULA_ID


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument(
        "--inventory",
        default=str(
            REPO_ROOT
            / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_candidate_inventory_d4.json"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT
            / "runs/pi05_gdsq_gr00t_aligned/sensitivity/pi05_sensitivity_action_n16_d4_merged.json"
        ),
    )
    return parser.parse_args()


def finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def main() -> None:
    args = parse_args()
    inputs = [Path(value).resolve() for value in args.input]
    if not inputs:
        inputs = sorted(
            (REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/sensitivity").glob(
                "pi05_sensitivity_action_n16_d4_s*.json"
            )
        )
    if len(inputs) != 6:
        raise SystemExit(f"expected exactly 6 sensitivity shards, got {len(inputs)}")
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    expected_names = [row["name"] for row in inventory["layers"]]
    merged_layers = {}
    shard_rows = []
    reference_meta = None
    ignored_meta_keys = {
        "shard_index", "expected_layers", "elapsed_s", "completed_layers", "block_banks",
        "reference_latency_s", "score_banks",
    }
    seen_shards = set()
    metric_path = Path(__file__).with_name("pi05_func_metrics.py")
    metric_hash = sha256_file(metric_path)
    for path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("complete") is not True:
            raise SystemExit(f"incomplete shard: {path}")
        meta = payload["meta"]
        if meta.get("functional_metric_sha256") != metric_hash:
            raise SystemExit(f"stale functional metric provenance in {path}")
        formula = meta.get("functional_formula") or {}
        if formula.get("formula_id") != FUNCTIONAL_FORMULA_ID:
            raise SystemExit(f"non-final functional formula in {path}")
        if meta.get("flow_steps") != 4 or meta.get("execute_actions") != 16:
            raise SystemExit(f"non-GR00T-aligned sensitivity protocol in {path}")
        shard_index = int(meta["shard_index"])
        if shard_index in seen_shards:
            raise SystemExit(f"duplicate shard index {shard_index}")
        seen_shards.add(shard_index)
        comparable = {key: value for key, value in meta.items() if key not in ignored_meta_keys}
        if reference_meta is None:
            reference_meta = comparable
        elif comparable != reference_meta:
            raise SystemExit(f"protocol metadata mismatch in {path}")
        expected_shard_names = set(meta["expected_layers"])
        if set(payload["layers"]) != expected_shard_names:
            raise SystemExit(f"shard layer matrix mismatch in {path}")
        overlap = set(merged_layers) & set(payload["layers"])
        if overlap:
            raise SystemExit(f"duplicate layer rows: {sorted(overlap)[:5]}")
        for name, row in payload["layers"].items():
            normalized = dict(row)
            wrapper = row["functional_vs_wrapper_ref"]
            pure = row["functional_vs_fp16"]
            normalized["d_func_b4"] = wrapper["d_func"]
            normalized["d_func_b4_std"] = wrapper.get("d_func_std", 0.0)
            normalized["d_solver_b4"] = wrapper["d_solver"]
            normalized["d_solver_b4_std"] = wrapper.get("d_solver_std", 0.0)
            normalized["d_func_fp16_b4"] = pure["d_func"]
            normalized["d_func_fp16_b4_std"] = pure.get("d_func_std", 0.0)
            normalized["d_solver_fp16_b4"] = pure["d_solver"]
            normalized["d_solver_fp16_b4_std"] = pure.get("d_solver_std", 0.0)
            merged_layers[name] = normalized
        shard_rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "shard_index": shard_index,
                "layers": len(payload["layers"]),
                "elapsed_s": meta.get("elapsed_s"),
            }
        )
    if seen_shards != set(range(6)):
        raise SystemExit(f"shard indices are incomplete: {sorted(seen_shards)}")
    missing = set(expected_names) - set(merged_layers)
    extra = set(merged_layers) - set(expected_names)
    if missing or extra or len(merged_layers) != 180:
        raise SystemExit(f"invalid merged inventory: missing={len(missing)} extra={len(extra)}")

    rms_values = [float(merged_layers[name]["b4"]["rms_ratio"]) for name in expected_names]
    sat_values = [float(merged_layers[name]["b4"]["sat_rate"]) for name in expected_names]
    tau_rms = float(np.quantile(rms_values, 0.99, method="higher") * 1.5)
    tau_sat = float(np.quantile(sat_values, 0.99, method="higher") * 1.5)
    metric_rows = []
    for name in expected_names:
        row = merged_layers[name]
        values = {
            "one_minus_cka": 1.0 - float(row["b4"].get("cka_dit", row["b4"]["cka"])),
            "one_minus_local_cka": 1.0 - float(row["b4"]["cka"]),
            "cs": float(row["b4"]["cs"]),
            "rms_ratio": float(row["b4"]["rms_ratio"]),
            "sat_rate": float(row["b4"]["sat_rate"]),
            "d_func": float(row["d_func_b4"]),
        }
        if all(finite(value) for value in values.values()):
            metric_rows.append(values)
    correlations = {}
    target = [row["d_func"] for row in metric_rows]
    for key in ("one_minus_cka", "one_minus_local_cka", "cs", "rms_ratio", "sat_rate"):
        values = [row[key] for row in metric_rows]
        correlations[key] = {
            "spearman": float(stats.spearmanr(values, target).statistic),
            "pearson": float(stats.pearsonr(values, target).statistic),
        }
    payload = {
        "schema_version": 1,
        "complete": True,
        "meta": {
            **reference_meta,
            "probe_script_sha256": sha256_file(Path(__file__).with_name("pi05_sensitivity_probe.py")),
            "functional_metric_sha256": metric_hash,
            "kernel_scores_sha256": sha256_file(
                REPO_ROOT / "code/pi05/openpi/src/openpi/quant/kernel_scores.py"
            ),
            "candidate_count": 180,
            "candidate_inventory_sha256": inventory["candidate_inventory_sha256"],
            "guard_thresholds": {
                "tau_rms": tau_rms,
                "tau_sat": tau_sat,
                "rule": "P99(method=higher) * 1.5 over all W4 candidates",
            },
            "correlations_vs_d_func_wrapper": correlations,
            "shards": sorted(shard_rows, key=lambda row: row["shard_index"]),
        },
        "layers": {name: merged_layers[name] for name in expected_names},
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(out) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(out)
    print(
        json.dumps(
            {
                "out": str(out),
                "sha256": sha256_file(out),
                "layers": len(merged_layers),
                "guard_thresholds": payload["meta"]["guard_thresholds"],
                "correlations": correlations,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
