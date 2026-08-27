#!/usr/bin/env python3
"""Materialize and select the shared v3 ATM × ErrorFold 9x9 grid.

The preregistered selection uses paired per-sequence differences to the
empirically best candidate.  An explicit minimum-objective rule is also
available for the gate-light v5.1 ablation.  Neither rule accepts task labels,
rollout success, learned gates or runtime branches.
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
from quantvla_dynamic_a8_protocol import (
    protocol_attestation as dynamic_a8_protocol_attestation,
    require_protocol_attestation as require_dynamic_a8_protocol_attestation,
)
from quantvla_errorfold import materialize_entry


GRID = tuple(float(value) for value in PROTOCOL["softfold"]["grid"]["gate_atm"])
ERRORFOLD_GRID = tuple(
    float(value) for value in PROTOCOL["softfold"]["grid"]["gate_errorfold"]
)
if GRID != ERRORFOLD_GRID:
    raise ValueError("v3 SoftFold requires identical ATM/ErrorFold grids")

DPAC_PROTOCOL = PROTOCOL["metrics"]["d_pac"]


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


def _d_pac_selection_contributions(values: Sequence[float]) -> dict[str, Any]:
    """Represent mean + CVaR as aligned per-sequence contributions.

    The mean of the returned contribution vector is exactly
    ``mean(values) + tail_weight * CVaR_alpha(values)``.  This lets paired
    one-SE operate on sequence-aligned differences without silently dropping
    the preregistered outer CVaR term.
    """
    if not values:
        raise ValueError("D_PAC selection needs at least one sequence")
    alpha = float(DPAC_PROTOCOL["outer_cvar_alpha"])
    tail_weight = float(DPAC_PROTOCOL["outer_cvar_weight"])
    tail_count = max(1, int(math.ceil((1.0 - alpha) * len(values))))
    tail_indices = {
        index
        for index, _value in sorted(
            enumerate(values), key=lambda item: (float(item[1]), item[0]), reverse=True
        )[:tail_count]
    }
    multiplier = len(values) / tail_count
    contributions = [
        float(value)
        + (tail_weight * multiplier * float(value) if index in tail_indices else 0.0)
        for index, value in enumerate(values)
    ]
    sequence_mean = statistics.fmean(float(value) for value in values)
    cvar = statistics.fmean(float(values[index]) for index in sorted(tail_indices))
    objective = sequence_mean + tail_weight * cvar
    if not math.isclose(statistics.fmean(contributions), objective, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("D_PAC contribution decomposition drift")
    return {
        "sequence_mean": sequence_mean,
        "cvar": cvar,
        "cvar_alpha": alpha,
        "tail_weight": tail_weight,
        "selection_objective": objective,
        "paired_selection_values": contributions,
        "tail_sequence_indices": sorted(tail_indices),
    }


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
        if metric == "d_func_v1":
            objective = statistics.fmean(values)
            objective_fields = {
                "sequence_mean": objective,
                "cvar": None,
                "selection_objective": objective,
                "paired_selection_values": list(values),
            }
        else:
            objective_fields = _d_pac_selection_contributions(values)
            objective = float(objective_fields["selection_objective"])
        summaries.append(
            {
                "gate": {"atm": gate_atm, "errorfold": gate_errorfold},
                # Retain the historical field name for artifact consumers;
                # for D_PAC it now means the complete mean+CVaR objective.
                "mean": objective,
                "n_sequences": len(values),
                "per_sequence": values,
                "config_id": row.get("config_id"),
                "correction_norm": float(
                    row.get("correction_norm", gate_atm * gate_atm + gate_errorfold * gate_errorfold)
                ),
                **objective_fields,
            }
        )
    return summaries


def summarize_dual_candidates(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Align local D_func and long-horizon D_PAC objectives by gate."""
    materialized = [dict(row) for row in rows]
    local = summarize_candidates(materialized, "d_func_v1")
    long_horizon = summarize_candidates(materialized, "d_pac_v2")
    local_by_gate = {
        _gate(row): row
        for row in local
    }
    result = []
    for pac_row in long_horizon:
        pair = _gate(pac_row)
        func_row = local_by_gate[pair]
        result.append(
            {
                "gate": dict(pac_row["gate"]),
                "config_id": pac_row.get("config_id"),
                "correction_norm": float(pac_row["correction_norm"]),
                "d_func_objective": float(func_row["selection_objective"]),
                "d_pac_objective": float(pac_row["selection_objective"]),
                "d_func_per_sequence": list(func_row["per_sequence"]),
                "d_pac_per_sequence": list(pac_row["per_sequence"]),
            }
        )
    return result


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
    """Paired one-SE followed by correction norm, gate sum and interaction.

    A candidate is never allowed to manufacture eligibility by having an
    enormous paired standard error while its point estimate is worse than the
    identity correction.  Identity is part of every complete SoftFold grid,
    so this is a reference candidate invariant rather than an extra tuned
    threshold.
    """
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
    identity_rows = [
        row
        for row in ranked
        if float(row["gate"]["atm"]) == 0.0
        and float(row["gate"]["errorfold"]) == 0.0
    ]
    if len(identity_rows) != 1:
        raise ValueError("paired one-SE requires exactly one identity gate candidate")
    identity = identity_rows[0]
    identity_tolerance = max(1e-15, 1e-12 * abs(float(identity["mean"])))
    best_values = best.get("paired_selection_values", best["per_sequence"])
    eligible = []
    paired_eligible_count = 0
    for row in ranked:
        differences = [
            float(value) - float(best_value)
            for value, best_value in zip(
                row.get("paired_selection_values", row["per_sequence"]), best_values
            )
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
        row["paired_one_se_eligible"] = difference_mean <= difference_se + 1e-15
        row["identity_non_degrading"] = (
            float(row["mean"]) <= float(identity["mean"]) + identity_tolerance
        )
        if row["paired_one_se_eligible"]:
            paired_eligible_count += 1
        if row["paired_one_se_eligible"] and row["identity_non_degrading"]:
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
        "identity": identity,
        "eligible_count": len(eligible),
        "paired_eligible_before_identity_bound": paired_eligible_count,
        "paired_one_standard_error": True,
        "identity_non_degradation_bound": True,
        "lambda_identity": float(lambda_identity),
        "lambda_interaction": float(lambda_interaction),
        "candidates": ranked,
    }


def select_minimum_objective(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the global offline metric minimum with deterministic tie breaks.

    Reliability shrinkage has already been applied while materializing each
    ErrorFold candidate.  Consequently this rule needs no additional gate,
    success label, task mapping, or fitted hyperparameter.  The tiny tolerance
    is solely for numerically identical JSON round trips; it is not a one-SE
    acceptance band.
    """
    if not summaries:
        raise ValueError("no SoftFold candidates")
    ranked = [dict(value) for value in summaries]
    best_objective = min(float(row["selection_objective"]) for row in ranked)
    tie_tolerance = max(1e-15, 1e-12 * max(1.0, abs(best_objective)))
    tied = [
        row
        for row in ranked
        if abs(float(row["selection_objective"]) - best_objective) <= tie_tolerance
    ]
    selected = min(
        tied,
        key=lambda row: (
            float(row["correction_norm"]),
            float(row["gate"]["atm"]) + float(row["gate"]["errorfold"]),
            float(row["gate"]["atm"]) * float(row["gate"]["errorfold"]),
            float(row["gate"]["errorfold"]),
            float(row["gate"]["atm"]),
        ),
    )
    identity_rows = [
        row
        for row in ranked
        if float(row["gate"]["atm"]) == 0.0
        and float(row["gate"]["errorfold"]) == 0.0
    ]
    if len(identity_rows) != 1:
        raise ValueError("minimum-objective selection requires exactly one identity candidate")
    return {
        "selected": selected,
        "best": selected,
        "identity": identity_rows[0],
        "eligible_count": len(tied),
        "paired_one_standard_error": False,
        "identity_non_degradation_bound": False,
        "tie_tolerance": tie_tolerance,
        "candidates": ranked,
    }


def select_atm_only_minimum(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the offline minimum after disabling all Linear/head affines.

    The complete 9x9 grid is still required by ``fit`` before this component
    ablation is applied.  Restricting ErrorFold to zero isolates the smaller
    attention-logit correction and avoids choosing the family from rollout
    success.
    """
    restricted = [
        row
        for row in summaries
        if float(row["gate"]["errorfold"]) == 0.0
    ]
    if len(restricted) != len(GRID):
        raise ValueError("ATM-only selection requires all nine ErrorFold=0 candidates")
    result = select_minimum_objective(restricted)
    result["correction_family"] = "attention_logits_only"
    result["errorfold_gate_fixed"] = 0.0
    return result


def select_minimax_dual_regret(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Complement D_func and D_PAC without a fitted or hand-tuned weight.

    For each metric, its grid minimum has regret zero and identity has regret
    one.  Minimizing the worse of the two regrets finds the scale-independent
    knee of the Pareto frontier.  Because identity is always available, the
    selected candidate cannot degrade either metric when both improvement
    spans are non-zero.
    """
    if not summaries:
        raise ValueError("no SoftFold candidates")
    ranked = [dict(value) for value in summaries]
    identity_rows = [
        row
        for row in ranked
        if float(row["gate"]["atm"]) == 0.0
        and float(row["gate"]["errorfold"]) == 0.0
    ]
    if len(identity_rows) != 1:
        raise ValueError("dual-regret selection requires exactly one identity candidate")
    identity = identity_rows[0]
    best_values = {
        "d_func": min(float(row["d_func_objective"]) for row in ranked),
        "d_pac": min(float(row["d_pac_objective"]) for row in ranked),
    }
    identity_values = {
        "d_func": float(identity["d_func_objective"]),
        "d_pac": float(identity["d_pac_objective"]),
    }
    spans = {
        name: identity_values[name] - best_values[name]
        for name in best_values
    }
    for row in ranked:
        regrets = {}
        for name, field in (
            ("d_func", "d_func_objective"),
            ("d_pac", "d_pac_objective"),
        ):
            scale = spans[name]
            tolerance = max(1e-15, 1e-12 * max(1.0, abs(identity_values[name])))
            delta = float(row[field]) - best_values[name]
            if scale > tolerance:
                regrets[name] = delta / scale
            else:
                regrets[name] = 0.0 if delta <= tolerance else delta / tolerance
        row["normalized_regret"] = regrets
        row["worst_normalized_regret"] = max(regrets.values())
        row["mean_normalized_regret"] = statistics.fmean(regrets.values())
    selected = min(
        ranked,
        key=lambda row: (
            float(row["worst_normalized_regret"]),
            float(row["mean_normalized_regret"]),
            float(row["correction_norm"]),
            float(row["gate"]["atm"]) + float(row["gate"]["errorfold"]),
            float(row["gate"]["atm"]) * float(row["gate"]["errorfold"]),
            float(row["gate"]["errorfold"]),
            float(row["gate"]["atm"]),
        ),
    )
    return {
        "selected": selected,
        "best": {
            "d_func": min(ranked, key=lambda row: float(row["d_func_objective"])),
            "d_pac": min(ranked, key=lambda row: float(row["d_pac_objective"])),
        },
        "identity": identity,
        "individual_best_objectives": best_values,
        "identity_objectives": identity_values,
        "normalization_spans": spans,
        "eligible_count": len(ranked),
        "paired_one_standard_error": False,
        "identity_non_degradation_bound": True,
        "candidates": ranked,
    }


def select_atm_only_dual_pareto_knee(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Find the local/long-horizon Pareto knee in the ATM-only family.

    Each objective is normalized by the loss range between the D_func-optimal
    and D_PAC-optimal ATM candidates.  This gives both objectives equal regret
    at their opposite endpoint without a fitted weight or rollout feedback.
    """
    ranked = [
        dict(row)
        for row in summaries
        if float(row["gate"]["errorfold"]) == 0.0
    ]
    if len(ranked) != len(GRID):
        raise ValueError("ATM-only Pareto selection requires all nine ErrorFold=0 candidates")
    func_best = min(ranked, key=lambda row: float(row["d_func_objective"]))
    pac_best = min(ranked, key=lambda row: float(row["d_pac_objective"]))
    best_values = {
        "d_func": float(func_best["d_func_objective"]),
        "d_pac": float(pac_best["d_pac_objective"]),
    }
    endpoint_spans = {
        "d_func": float(pac_best["d_func_objective"]) - best_values["d_func"],
        "d_pac": float(func_best["d_pac_objective"]) - best_values["d_pac"],
    }
    for row in ranked:
        regrets = {}
        for name, field in (
            ("d_func", "d_func_objective"),
            ("d_pac", "d_pac_objective"),
        ):
            span = endpoint_spans[name]
            tolerance = max(1e-15, 1e-12 * max(1.0, abs(best_values[name])))
            delta = float(row[field]) - best_values[name]
            regrets[name] = (
                delta / span
                if span > tolerance
                else (0.0 if delta <= tolerance else delta / tolerance)
            )
        row["normalized_endpoint_regret"] = regrets
        row["worst_normalized_endpoint_regret"] = max(regrets.values())
        row["mean_normalized_endpoint_regret"] = statistics.fmean(regrets.values())
    selected = min(
        ranked,
        key=lambda row: (
            float(row["worst_normalized_endpoint_regret"]),
            float(row["mean_normalized_endpoint_regret"]),
            float(row["correction_norm"]),
            float(row["gate"]["atm"]),
        ),
    )
    return {
        "selected": selected,
        "best": {"d_func": func_best, "d_pac": pac_best},
        "individual_best_objectives": best_values,
        "endpoint_normalization_spans": endpoint_spans,
        "eligible_count": len(ranked),
        "paired_one_standard_error": False,
        "identity_non_degradation_bound": False,
        "correction_family": "attention_logits_only",
        "errorfold_gate_fixed": 0.0,
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
    if raw_meta.get("activation_mode") == "dynamic_a8":
        require_dynamic_a8_protocol_attestation(raw, source=source)
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
                "dynamic_a8_protocol": (
                    dynamic_a8_protocol_attestation()
                    if raw_meta.get("activation_mode") == "dynamic_a8"
                    else None
                ),
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
    dynamic_a8 = raw_meta.get("activation_mode") == "dynamic_a8"
    if dynamic_a8:
        require_dynamic_a8_protocol_attestation(
            validation, source=str(validation_path)
        )
        if validation.get("activation_mode") != "dynamic_a8":
            raise ValueError(
                "dynamic ErrorFold raw correction requires DyRange-A8 validation scores"
            )
    if validation.get("raw_correction_sha256") != sha256_file(raw_path):
        raise ValueError("validation scores do not descend from this raw ErrorFold artifact")
    if args.allow_partial_grid:
        raise ValueError("v3 requires the complete shared 9x9 grid")
    if float(args.lambda_identity) != 0.0 or float(args.lambda_interaction) != 0.0:
        raise ValueError("v3 freezes explicit lambdas at zero; reliability provides shrinkage")
    rows = score_rows(validation)
    if args.selection_rule in (
        "minimax_dual_regret",
        "atm_only_dual_pareto_knee",
    ):
        summaries = summarize_dual_candidates(rows)
        require_grid(summaries)
        if args.selection_rule == "minimax_dual_regret":
            selection = select_minimax_dual_regret(summaries)
            selection_rule = (
                "minimum_worst_identity_normalized_regret_over_d_func_and_d_pac"
            )
            metric_id = "d_func_v1_plus_d_pac_v2_minimax_normalized_regret"
        else:
            selection = select_atm_only_dual_pareto_knee(summaries)
            selection_rule = (
                "minimum_worst_endpoint_normalized_regret_over_d_func_and_d_pac_"
                "with_errorfold_gate_fixed_zero"
            )
            metric_id = "d_func_v1_plus_d_pac_v2_atm_only_pareto_knee"
    else:
        summaries = summarize_candidates(rows, args.metric)
        require_grid(summaries)
        metric_id = args.metric
    if args.selection_rule == "paired_one_se":
        selection = select_one_standard_error(summaries)
        selection_rule = PROTOCOL["softfold"]["selection_rule"]
    elif args.selection_rule == "minimum_objective":
        selection = select_minimum_objective(summaries)
        selection_rule = "minimum_offline_objective_then_correction_norm_gate_sum_interaction"
    elif args.selection_rule == "atm_only_minimum":
        selection = select_atm_only_minimum(summaries)
        selection_rule = (
            "minimum_offline_objective_with_errorfold_gate_fixed_zero_then_"
            "correction_norm_gate_sum_interaction"
        )
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
        "dynamic_a8_protocol": (
            dynamic_a8_protocol_attestation() if dynamic_a8 else None
        ),
        "teacher_checkpoint_sha256": checkpoint_hash,
        "quant_plan_sha256": plan_hash,
        "buffer_sha256": buffer_hash,
        "metric": metric_id,
        "gate": gate,
        "correction_norm": correction_norm,
        "layers": layers,
        "selection": {
            "rule": selection_rule,
            "rule_id": args.selection_rule,
            "uses_task_labels": False,
            "uses_rollout_success": False,
            "sample_unit": "paired_shared_buffer_sequence",
            **{key: value for key, value in selection.items() if key != "candidates"},
        },
        "meta": {
            **raw_meta,
            "metric": metric_id,
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
    parser.add_argument(
        "--selection-rule",
        default="paired_one_se",
        choices=(
            "paired_one_se",
            "minimum_objective",
            "minimax_dual_regret",
            "atm_only_minimum",
            "atm_only_dual_pareto_knee",
        ),
        help=(
            "paired_one_se is the frozen v3 rule; minimum_objective is the "
            "shared gate-light v5.1 ablation; minimax_dual_regret complements "
            "local D_func and long-horizon D_PAC without a tuned weight; "
            "atm_only_minimum is the shared attention-logit-only ablation; "
            "atm_only_dual_pareto_knee balances both metrics inside that family"
        ),
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
    print(json.dumps({"out": str(output), "sha256": sha256_file(output), "gate": payload["gate"]}, indent=2))


if __name__ == "__main__":
    main()
