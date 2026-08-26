#!/usr/bin/env python3
"""Materialize and select the shared v3 ATM × ErrorFold 9x9 grid.

Selection uses paired per-sequence differences to the empirically best
candidate.  No task label, rollout success, learned gate or runtime branch is
accepted.
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

from quantvla_cross_model_protocol import (
    PROTOCOL,
    PROTOCOL_SHA256,
    protocol_attestation,
    require_protocol_attestation,
)
from quantvla_errorfold import materialize_entry


GRID = tuple(float(value) for value in PROTOCOL["softfold"]["grid"]["gate_atm"])
ERRORFOLD_GRID = tuple(
    float(value) for value in PROTOCOL["softfold"]["grid"]["gate_errorfold"]
)
if GRID != ERRORFOLD_GRID:
    raise ValueError("v3 SoftFold requires identical ATM/ErrorFold grids")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_split(sample_hash: str) -> str:
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
    errorfold = gate.get(
        "errorfold", gate.get("ef", row.get("g_errorfold", row.get("errorfold_gate")))
    )
    if atm is None or errorfold is None:
        raise ValueError("every v3 score row requires ATM and ErrorFold gates")
    pair = float(atm), float(errorfold)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in pair):
        raise ValueError(f"invalid SoftFold gate {pair}")
    return pair


def _sequence_values(row: Mapping[str, Any], metric: str = "d_pac_v2") -> list[float]:
    if metric == "d_func_v1":
        values: Any = row.get("per_sequence")
        if values is None:
            summary = row.get("d_func_summary") or {}
            values = summary.get("per_sequence")
        if values is None:
            # Compatibility fallback for synthetic/unit-test rows only.  All
            # formal v3 scorers emit sequence-aligned D_func samples.
            values = row.get("per_obs")
        if values is None:
            values = (row.get("d_func_summary") or {}).get("per_obs")
        if values is None and row.get("d_func") is not None:
            values = [row["d_func"]]
    else:
        values = row.get("per_sequence")
        summary = row.get("d_pac_summary") or {}
        if values is None:
            values = summary.get("per_sequence")
        if values is None and summary.get("sequences") is not None:
            values = [item["d_pac_sequence"] for item in summary["sequences"]]
        if values is None and row.get("d_pac") is not None:
            values = [row["d_pac"]]
    if values is None:
        raise ValueError(f"{metric} score row has no paired sample values")
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError("selection values must be finite and non-empty")
    return result


def score_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: Any = payload.get("candidates", payload.get("scores"))
    if isinstance(rows, Mapping):
        rows = [{"config_id": key, **value} for key, value in rows.items()]
    if not isinstance(rows, list) or not rows:
        raise ValueError("validation score artifact requires candidates/scores")
    return [dict(row) for row in rows]


def summarize_candidates(
    rows: Iterable[Mapping[str, Any]], metric: str = "d_pac_v2"
) -> list[dict[str, Any]]:
    summaries = []
    seen = set()
    n_samples = None
    for row in rows:
        gate_atm, gate_errorfold = _gate(row)
        pair = gate_atm, gate_errorfold
        if pair in seen:
            raise ValueError(f"duplicate SoftFold gate {pair}")
        seen.add(pair)
        values = _sequence_values(row, metric)
        if n_samples is None:
            n_samples = len(values)
        if len(values) != n_samples:
            raise ValueError("paired one-SE requires equal, aligned sample counts")
        summaries.append(
            {
                "gate": {"atm": gate_atm, "errorfold": gate_errorfold},
                "mean": statistics.fmean(values),
                "n_sequences": len(values),
                "per_sequence": values,
                "config_id": row.get("config_id"),
                "correction_norm": float(
                    row.get("correction_norm", gate_atm * gate_atm + gate_errorfold * gate_errorfold)
                ),
            }
        )
    return summaries


def require_grid(summaries: Sequence[Mapping[str, Any]]) -> None:
    actual = {
        (
            round(float(row["gate"]["atm"]), 6),
            round(float(row["gate"]["errorfold"]), 6),
        )
        for row in summaries
    }
    expected = {(round(atm, 6), round(ef, 6)) for atm in GRID for ef in GRID}
    if actual != expected:
        raise ValueError(
            f"SoftFold v3 requires complete 9x9; missing={sorted(expected-actual)[:5]} "
            f"extra={sorted(actual-expected)[:5]}"
        )


def select_one_standard_error(
    summaries: Sequence[Mapping[str, Any]],
    *,
    lambda_identity: float = 0.0,
    lambda_interaction: float = 0.0,
) -> dict[str, Any]:
    """Paired one-SE followed by correction norm, gate sum and interaction."""
    if not summaries:
        raise ValueError("no SoftFold candidates")
    if lambda_identity < 0.0 or lambda_interaction < 0.0:
        raise ValueError("regularization coefficients must be non-negative")
    ranked = []
    for value in summaries:
        row = dict(value)
        atm = float(row["gate"]["atm"])
        errorfold = float(row["gate"]["errorfold"])
        row["regularized_objective"] = (
            float(row["mean"])
            + lambda_identity * (atm + errorfold)
            + lambda_interaction * atm * errorfold
        )
        ranked.append(row)
    best = min(ranked, key=lambda row: (row["regularized_objective"], row["mean"]))
    best_values = best["per_sequence"]
    eligible = []
    for row in ranked:
        differences = [
            float(value) - float(best_value)
            for value, best_value in zip(row["per_sequence"], best_values)
        ]
        difference_mean = statistics.fmean(differences)
        difference_se = (
            statistics.stdev(differences) / math.sqrt(len(differences))
            if len(differences) > 1
            else 0.0
        )
        row["paired_difference_mean"] = difference_mean
        row["paired_difference_standard_error"] = difference_se
        row["paired_differences"] = differences
        if difference_mean <= difference_se + 1e-15:
            eligible.append(row)
    selected = min(
        eligible,
        key=lambda row: (
            float(row["correction_norm"]),
            float(row["gate"]["atm"]) + float(row["gate"]["errorfold"]),
            float(row["gate"]["atm"]) * float(row["gate"]["errorfold"]),
            row["regularized_objective"],
            float(row["gate"]["errorfold"]),
            float(row["gate"]["atm"]),
        ),
    )
    return {
        "selected": selected,
        "best": best,
        "eligible_count": len(eligible),
        "paired_one_standard_error": True,
        "lambda_identity": float(lambda_identity),
        "lambda_interaction": float(lambda_interaction),
        "candidates": ranked,
    }


def softfold_value(value: float, gate: float, reliability: float = 1.0) -> float:
    """Identity-centered scalar interpolation retained for simple ATM gains."""
    if not all(math.isfinite(item) for item in (value, gate, reliability)):
        raise ValueError("SoftFold inputs must be finite")
    if not 0.0 <= gate <= 1.0 or not 0.0 <= reliability <= 1.0:
        raise ValueError("gate and reliability must be in [0,1]")
    return 1.0 + gate * reliability * (value - 1.0)


def fold_layers(
    raw_layers: Mapping[str, Mapping[str, Any]],
    *,
    gate_atm: float,
    gate_errorfold: float,
) -> tuple[dict[str, Any], float]:
    layers: dict[str, Any] = {}
    squared_norm = 0.0
    for name, raw_entry in sorted(raw_layers.items()):
        kind = str(raw_entry.get("kind", "linear"))
        gate = gate_atm if kind == "attention_logits" else gate_errorfold
        if "gain" not in raw_entry or "bias" not in raw_entry:
            raise ValueError(f"{name}: v3 raw correction lacks ridge gain/bias")
        entry = materialize_entry(raw_entry, gate)
        gain_delta = [float(value) - 1.0 for value in entry["effective_gain"]]
        bias = [float(value) for value in entry["effective_bias"]]
        squared_norm += sum(value * value for value in gain_delta)
        squared_norm += sum(value * value for value in bias)
        layers[str(name)] = entry
    return layers, math.sqrt(squared_norm)


def softfold_config_id(gate_atm: float, gate_errorfold: float) -> str:
    return f"softfold_a{int(round(gate_atm * 8)):02d}_e{int(round(gate_errorfold * 8)):02d}"


def validate_raw_correction_protocol(raw: Mapping[str, Any], *, source: str) -> Mapping[str, Any]:
    raw_meta = raw.get("meta", {})
    expected = {
        "calibration_buffer_sha256": PROTOCOL["data"]["calibration_buffer"]["sha256"],
        "flow_steps": PROTOCOL["closed_loop"]["flow_steps"],
        "folds": PROTOCOL["errorfold"]["folds"],
        "fit_pair": PROTOCOL["errorfold"]["fit_pair"],
    }
    mismatches = {
        key: (raw_meta.get(key), value)
        for key, value in expected.items()
        if raw_meta.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{source}: raw ErrorFold protocol drift: {mismatches}")
    return raw_meta


def materialize_grid(
    *,
    raw_path: str | Path,
    output_dir: str | Path,
    base_spec: Mapping[str, Any] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, dict[str, Any]]:
    raw_path = Path(raw_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_layers = raw.get("layers", raw)
    if not isinstance(raw_layers, Mapping) or not raw_layers:
        raise ValueError("raw ErrorFold artifact has no layers")
    raw_meta = validate_raw_correction_protocol(raw, source=str(raw_path))
    registry: dict[str, dict[str, Any]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid SoftFold materialization shard")
    candidate_index = 0
    for gate_atm in GRID:
        for gate_errorfold in GRID:
            selected = candidate_index % shard_count == shard_index
            candidate_index += 1
            if not selected:
                continue
            layers, correction_norm = fold_layers(
                raw_layers, gate_atm=gate_atm, gate_errorfold=gate_errorfold
            )
            identifier = softfold_config_id(gate_atm, gate_errorfold)
            path = output_dir / f"{identifier}.json"
            payload = {
                "schema_version": 3,
                "kind": "errorfold_grid_candidate",
                "cross_model_protocol": protocol_attestation(),
                "meta": {
                    **raw_meta,
                    "metric": PROTOCOL["metrics"]["d_pac"]["formula_id"],
                    "atm_application": PROTOCOL["deployment"]["atm_application"],
                    "errorfold_application": PROTOCOL["deployment"]["errorfold_application"],
                    "selector_free": True,
                    "runtime_branch": False,
                },
                "gate": {"atm": gate_atm, "errorfold": gate_errorfold},
                "correction_norm": correction_norm,
                "layers": layers,
                "selection": {
                    "uses_task_labels": False,
                    "uses_rollout_success": False,
                    "status": "grid_candidate_not_selected",
                },
            }
            rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            if path.exists() and path.read_text(encoding="utf-8") != rendered:
                raise ValueError(f"frozen v3 grid candidate drift: {path}")
            if not path.exists():
                path.write_text(rendered, encoding="utf-8")
            registry[identifier] = {
                **dict(base_spec or {}),
                "path": path,
                "errorfold": path,
                "atm": path,
                "gate": payload["gate"],
                "correction_norm": correction_norm,
            }
    return registry


def _metadata_hash(explicit: str | None, metadata: Mapping[str, Any], key: str, path: str | None) -> str | None:
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
    require_protocol_attestation(validation, source=str(validation_path))
    raw_meta = validate_raw_correction_protocol(raw, source=str(raw_path))
    if validation.get("raw_correction_sha256") != sha256_file(raw_path):
        raise ValueError("validation scores do not descend from this raw ErrorFold artifact")
    if args.allow_partial_grid:
        raise ValueError("v3 requires the complete shared 9x9 grid")
    if float(args.lambda_identity) != 0.0 or float(args.lambda_interaction) != 0.0:
        raise ValueError("v3 freezes explicit lambdas at zero; reliability provides shrinkage")
    summaries = summarize_candidates(score_rows(validation), args.metric)
    require_grid(summaries)
    selection = select_one_standard_error(summaries)
    gate = selection["selected"]["gate"]
    layers, correction_norm = fold_layers(
        raw.get("layers", raw),
        gate_atm=float(gate["atm"]),
        gate_errorfold=float(gate["errorfold"]),
    )
    checkpoint_hash = _metadata_hash(
        args.teacher_checkpoint_sha256, raw_meta, "checkpoint_sha256", args.teacher_checkpoint
    )
    plan_hash = _metadata_hash(args.quant_plan_sha256, raw_meta, "plan_sha256", args.quant_plan)
    buffer_hash = _metadata_hash(
        args.buffer_sha256, raw_meta, "calibration_buffer_sha256", args.buffer
    )
    if not checkpoint_hash or not plan_hash or not buffer_hash:
        raise ValueError("teacher checkpoint, quant plan, and calibration buffer hashes are required")
    score_expected = {
        "checkpoint_sha256": checkpoint_hash,
        "plan_sha256": plan_hash,
        "artifact_calibration_buffer_sha256": buffer_hash,
    }
    mismatch = {
        key: (validation.get(key), expected)
        for key, expected in score_expected.items()
        if validation.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"ErrorFold score/fold provenance mismatch: {mismatch}")
    return {
        "schema_version": 3,
        "kind": "errorfold_compensation",
        "cross_model_protocol": protocol_attestation(),
        "teacher_checkpoint_sha256": checkpoint_hash,
        "quant_plan_sha256": plan_hash,
        "buffer_sha256": buffer_hash,
        "metric": args.metric,
        "gate": gate,
        "correction_norm": correction_norm,
        "layers": layers,
        "selection": {
            "rule": PROTOCOL["softfold"]["selection_rule"],
            "uses_task_labels": False,
            "uses_rollout_success": False,
            "sample_unit": "paired_shared_buffer_sequence",
            **{key: value for key, value in selection.items() if key != "candidates"},
        },
        "meta": {
            **raw_meta,
            "metric": args.metric,
            "atm_application": PROTOCOL["deployment"]["atm_application"],
            "errorfold_application": PROTOCOL["deployment"]["errorfold_application"],
            "selector_free": True,
            "runtime_branch": False,
            "protocol_sha256": PROTOCOL_SHA256,
            "raw_correction_sha256": sha256_file(raw_path),
            "validation_scores_sha256": sha256_file(validation_path),
            "gate_grid": list(GRID),
            "gate_candidates": len(summaries),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-correction", required=True)
    parser.add_argument("--validation-scores", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", default="d_pac_v2", choices=("d_pac_v2", "d_func_v1"))
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
    print(json.dumps({"out": str(output), "sha256": sha256_file(output), "gate": payload["gate"]}, indent=2))


if __name__ == "__main__":
    main()
