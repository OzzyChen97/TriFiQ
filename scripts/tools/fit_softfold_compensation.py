#!/usr/bin/env python3
"""Fit selector-free SoftFold gates from FP16-relative D_PAC validation scores.

This tool deliberately does not train a gate network and never consumes task
labels or rollout success.  The caller evaluates the fixed 9x9 gate grid on a
frozen validation buffer with paired FP16 observations/noise; this script then
applies the one-standard-error rule and log-space folds the selected global
ATM/OHB amplitudes into a deployment artifact.

Expected score rows (under ``candidates`` or ``scores``) contain::

    {
      "gate": {"atm": 0.125, "ohb": 0.625},
      "per_sequence": [0.1, 0.2, ...]
    }

``d_pac_summary.per_sequence`` and scalar ``d_pac`` are also accepted.  The
model-specific scorer owns inference; this fitter only performs deterministic
statistical selection and coefficient interpolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


GRID = tuple(index / 8.0 for index in range(9))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_split(sample_hash: str) -> str:
    """Deterministic task-label-free 50/50 fit/validation assignment."""
    normalized = str(sample_hash).strip().lower()
    if not normalized:
        raise ValueError("sample hash must not be empty")
    value = int(hashlib.sha256(normalized.encode("utf-8")).hexdigest(), 16)
    return "fit" if value % 2 == 0 else "validation"


def split_sample_hashes(sample_hashes: Sequence[str]) -> dict[str, list[str]]:
    result = {"fit": [], "validation": []}
    for value in sample_hashes:
        result[sample_split(value)].append(str(value))
    return result


def _gate(row: Mapping[str, Any]) -> tuple[float, float]:
    gate = row.get("gate") or {}
    atm = gate.get("atm", row.get("g_atm", row.get("atm_gate")))
    ohb = gate.get("ohb", row.get("g_ohb", row.get("ohb_gate")))
    if atm is None or ohb is None:
        raise ValueError("every score row requires ATM and OHB gates")
    pair = (float(atm), float(ohb))
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in pair):
        raise ValueError(f"invalid SoftFold gate {pair}")
    return pair


def _sequence_values(
    row: Mapping[str, Any], metric: str = "d_pac_v1"
) -> list[float]:
    if metric == "d_func_v1":
        candidates: Any = row.get("per_obs")
        if candidates is None:
            summary = row.get("d_func_summary") or {}
            candidates = summary.get("per_obs")
        if candidates is None and row.get("d_func") is not None:
            candidates = [row["d_func"]]
        if candidates is None:
            raise ValueError("D_func score row requires per_obs, d_func_summary, or d_func")
        values = [float(value) for value in candidates]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("D_func validation values must be finite and non-empty")
        return values
    candidates = row.get("per_sequence")
    if candidates is None:
        summary = row.get("d_pac_summary") or {}
        candidates = summary.get("per_sequence")
        if candidates is None and summary.get("sequences") is not None:
            candidates = [item["d_pac_sequence"] for item in summary["sequences"]]
    if candidates is None and row.get("d_pac") is not None:
        candidates = [row["d_pac"]]
    if candidates is None:
        raise ValueError("score row requires per_sequence, d_pac_summary, or d_pac")
    values = [float(value) for value in candidates]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("D_PAC validation values must be finite and non-empty")
    return values


def score_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: Any = payload.get("candidates", payload.get("scores"))
    if isinstance(rows, Mapping):
        rows = [{"config_id": key, **value} for key, value in rows.items()]
    if not isinstance(rows, list) or not rows:
        raise ValueError("validation score artifact requires non-empty candidates/scores")
    return [dict(row) for row in rows]


def summarize_candidates(
    rows: Iterable[Mapping[str, Any]], metric: str = "d_pac_v1"
) -> list[dict[str, Any]]:
    summaries = []
    seen = set()
    for row in rows:
        gate_atm, gate_ohb = _gate(row)
        pair = (gate_atm, gate_ohb)
        if pair in seen:
            raise ValueError(f"duplicate SoftFold gate {pair}")
        seen.add(pair)
        values = _sequence_values(row, metric)
        mean = statistics.fmean(values)
        standard_error = (
            statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        )
        summaries.append(
            {
                "gate": {"atm": gate_atm, "ohb": gate_ohb},
                "mean": mean,
                "standard_error": standard_error,
                "n_sequences": len(values),
                "per_sequence": values,
                "config_id": row.get("config_id"),
            }
        )
    return summaries


def require_grid(summaries: Sequence[Mapping[str, Any]]) -> None:
    actual = {
        (round(float(row["gate"]["atm"]), 6), round(float(row["gate"]["ohb"]), 6))
        for row in summaries
    }
    expected = {(round(atm, 6), round(ohb, 6)) for atm in GRID for ohb in GRID}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"SoftFold v1 requires the complete 9x9 grid; missing={missing[:5]} "
            f"extra={extra[:5]}"
        )


def select_one_standard_error(
    summaries: Sequence[Mapping[str, Any]],
    *,
    lambda_identity: float = 0.0,
    lambda_interaction: float = 0.0,
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("no SoftFold candidates")
    if lambda_identity < 0.0 or lambda_interaction < 0.0:
        raise ValueError("regularization coefficients must be non-negative")
    ranked = []
    for value in summaries:
        row = dict(value)
        atm = float(row["gate"]["atm"])
        ohb = float(row["gate"]["ohb"])
        row["regularized_objective"] = (
            float(row["mean"])
            + lambda_identity * (atm + ohb)
            + lambda_interaction * atm * ohb
        )
        ranked.append(row)
    best = min(ranked, key=lambda row: (row["regularized_objective"], row["mean"]))
    threshold = float(best["regularized_objective"] + best["standard_error"])
    eligible = [row for row in ranked if row["regularized_objective"] <= threshold + 1e-15]
    selected = min(
        eligible,
        key=lambda row: (
            float(row["gate"]["atm"]) + float(row["gate"]["ohb"]),
            float(row["gate"]["atm"]) * float(row["gate"]["ohb"]),
            row["regularized_objective"],
            float(row["gate"]["ohb"]),
            float(row["gate"]["atm"]),
        ),
    )
    return {
        "selected": selected,
        "best": best,
        "one_standard_error_threshold": threshold,
        "eligible_count": len(eligible),
        "lambda_identity": float(lambda_identity),
        "lambda_interaction": float(lambda_interaction),
        "candidates": ranked,
    }


def softfold_value(value: float, gate: float) -> float:
    value = float(value)
    gate = float(gate)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"multiplicative correction must be positive and finite, got {value}")
    if not math.isfinite(gate) or not 0.0 <= gate <= 1.0:
        raise ValueError(f"gate must be in [0,1], got {gate}")
    return math.exp(gate * math.log(value))


def _softfold_values(values: Sequence[float], gate: float) -> list[float]:
    return [softfold_value(value, gate) for value in values]


def fold_layers(
    raw_layers: Mapping[str, Mapping[str, Any]],
    *,
    gate_atm: float,
    gate_ohb: float,
) -> tuple[dict[str, Any], str | None]:
    layers: dict[str, Any] = {}
    ohb_modes = set()
    for name, raw_entry in sorted(raw_layers.items()):
        entry: dict[str, Any] = {}
        alpha = raw_entry.get("all", raw_entry.get("alpha"))
        if alpha is not None:
            entry["all"] = _softfold_values(alpha, gate_atm)
            entry["alpha_raw"] = [float(value) for value in alpha]
        if raw_entry.get("beta_perhead") is not None:
            beta = raw_entry["beta_perhead"]
            entry["beta_perhead"] = _softfold_values(beta, gate_ohb)
            entry["beta_perhead_raw"] = [float(value) for value in beta]
            ohb_modes.add("per_head_pre_projection")
        elif raw_entry.get("beta") is not None:
            beta = float(raw_entry["beta"])
            entry["beta"] = softfold_value(beta, gate_ohb)
            entry["beta_raw"] = beta
            ohb_modes.add("scalar_post_projection")
        if not entry:
            raise ValueError(f"{name}: no static alpha/beta correction values")
        layers[str(name)] = entry
    if len(ohb_modes) > 1:
        raise ValueError("mixed scalar/per-head OHB artifacts cannot use one static fold mode")
    return layers, next(iter(ohb_modes), None)


def _metadata_hash(
    explicit: str | None,
    metadata: Mapping[str, Any],
    key: str,
    path: str | None,
) -> str | None:
    if explicit:
        return explicit
    if metadata.get(key):
        return str(metadata[key])
    return sha256_file(path) if path else None


def fit(args: argparse.Namespace) -> dict[str, Any]:
    raw_path = Path(args.raw_correction).expanduser().resolve()
    validation_path = Path(args.validation_scores).expanduser().resolve()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    raw_layers = raw.get("layers", raw)
    if not isinstance(raw_layers, Mapping) or not raw_layers:
        raise ValueError("raw correction artifact has no layers")
    summaries = summarize_candidates(score_rows(validation), args.metric)
    if not args.allow_partial_grid:
        require_grid(summaries)
    selection = select_one_standard_error(
        summaries,
        lambda_identity=args.lambda_identity,
        lambda_interaction=args.lambda_interaction,
    )
    gate = selection["selected"]["gate"]
    layers, ohb_mode = fold_layers(
        raw_layers,
        gate_atm=float(gate["atm"]),
        gate_ohb=float(gate["ohb"]),
    )
    raw_meta = raw.get("meta", {}) if isinstance(raw, Mapping) else {}
    checkpoint_hash = _metadata_hash(
        args.teacher_checkpoint_sha256,
        raw_meta,
        "checkpoint_sha256",
        args.teacher_checkpoint,
    )
    plan_hash = _metadata_hash(args.quant_plan_sha256, raw_meta, "plan_sha256", args.quant_plan)
    buffer_hash = _metadata_hash(
        args.buffer_sha256,
        raw_meta,
        "calibration_buffer_sha256",
        args.buffer,
    )
    if not checkpoint_hash or not plan_hash or not buffer_hash:
        raise ValueError("teacher checkpoint, quant plan, and buffer SHA256 provenance are required")
    ohb_application = {
        "per_head_pre_projection": "fold_o_weight_perhead",
        "scalar_post_projection": "fold_o_weight",
        None: None,
    }[ohb_mode]
    return {
        "schema_version": 1,
        "kind": "softfold_compensation",
        "teacher_checkpoint_sha256": checkpoint_hash,
        "quant_plan_sha256": plan_hash,
        "buffer_sha256": buffer_hash,
        "metric": args.metric,
        "gate": {"atm": float(gate["atm"]), "ohb": float(gate["ohb"])},
        "layers": layers,
        "selection": {
            "rule": "one_standard_error",
            "uses_task_labels": False,
            "uses_rollout_success": False,
            "fit_split": "sample_sha256_parity",
            "validation_only_for_gate": True,
            **{key: value for key, value in selection.items() if key != "candidates"},
        },
        "meta": {
            "checkpoint_sha256": checkpoint_hash,
            "plan_sha256": plan_hash,
            "calibration_buffer_sha256": buffer_hash,
            "metric": args.metric,
            "atm_application": "fold_q_weight",
            "ohb_mode": ohb_mode,
            "ohb_application": ohb_application,
            "selector_free": True,
            "runtime_branch": False,
            "raw_correction_sha256": sha256_file(raw_path),
            "validation_scores_sha256": sha256_file(validation_path),
            "gate_grid": list(GRID),
            "gate_candidates": len(summaries),
            **{
                key: raw_meta[key]
                for key in ("flow_steps", "frames", "batch_size")
                if raw_meta.get(key) is not None
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-correction", required=True)
    parser.add_argument("--validation-scores", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--metric",
        default="d_pac_v1",
        choices=("d_pac_v1", "d_func_v1"),
    )
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--teacher-checkpoint-sha256")
    parser.add_argument("--quant-plan")
    parser.add_argument("--quant-plan-sha256")
    parser.add_argument("--buffer")
    parser.add_argument("--buffer-sha256")
    parser.add_argument("--lambda-identity", type=float, default=0.0)
    parser.add_argument("--lambda-interaction", type=float, default=0.0)
    parser.add_argument("--allow-partial-grid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = fit(args)
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": sha256_file(output),
                "metric": payload["metric"],
                "gate": payload["gate"],
                "layers": len(payload["layers"]),
                "selector_free": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
