#!/usr/bin/env python3
"""Shared statistics and exact budget solver for full-context FP16 protection.

This file is intentionally model agnostic.  Model loaders, native action
normalization, layer-name binding, and real-quant kernels remain in adapters;
all counterfactual statistics and mask decisions live here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "scripts/quantvla_full_context_protocol.json"
METRIC_PATH = REPO_ROOT / "scripts/tools/quantvla_metric_protocol.py"
QUICK_STATS_PATH = REPO_ROOT / "scripts/tools/aggregate_full_context_quick.py"
FORMAL_STATS_PATH = REPO_ROOT / "scripts/tools/aggregate_full_context_table1.py"


def canonical_hash(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("method_id") not in (
        "full_context_fp16_protection_v1",
        "full_context_fp16_protection_v2",
    ):
        raise ValueError(f"unexpected full-context protocol: {value.get('method_id')}")
    return value


PROTOCOL = load_protocol(PROTOCOL_PATH)
PROTOCOL_SHA256 = canonical_hash(PROTOCOL)
PROTOCOL_PATH_V2 = REPO_ROOT / "scripts/quantvla_full_context_protocol_v2.json"
PROTOCOL_V2 = load_protocol(PROTOCOL_PATH_V2)
PROTOCOL_V2_SHA256 = canonical_hash(PROTOCOL_V2)


def protocol_attestation() -> dict[str, Any]:
    return {
        "method_id": PROTOCOL["method_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_file": str(PROTOCOL_PATH),
        "protocol_file_sha256": sha256_file(PROTOCOL_PATH),
        "selection_core_sha256": sha256_file(Path(__file__)),
        "metric_core_sha256": sha256_file(METRIC_PATH),
        "quick_statistics_sha256": sha256_file(QUICK_STATS_PATH),
        "formal_statistics_sha256": sha256_file(FORMAL_STATS_PATH),
    }


def require_protocol_attestation(value: Mapping[str, Any], *, source: str) -> None:
    actual = value.get("full_context_protocol") or {}
    expected = protocol_attestation()
    drift = {
        key: (actual.get(key), expected_value)
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    if drift:
        raise ValueError(f"{source}: full-context protocol drift: {drift}")


def candidate_plan_mapping(
    *,
    named: Sequence[tuple[str, Path]] | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    """Load candidate plans from CLI pairs and/or a frozen manifest.

    Paths stored in manifests are resolved relative to the manifest directory
    when they are not absolute.  Duplicate identifiers fail closed, including
    duplicates that point to the same path.
    """
    pairs = list(named or ())
    manifest = None
    if manifest_path is not None:
        resolved = Path(manifest_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
        require_protocol_attestation(manifest, source=str(resolved))
        rows = manifest.get("candidates") or []
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{resolved}: candidate manifest is empty")
        for row in rows:
            identifier = str(row.get("candidate_id", ""))
            raw_path = Path(str(row.get("path", ""))).expanduser()
            if not identifier or not str(raw_path):
                raise ValueError(f"{resolved}: malformed candidate row")
            path = raw_path if raw_path.is_absolute() else resolved.parent / raw_path
            path = path.resolve()
            expected_sha = row.get("sha256")
            if expected_sha and sha256_file(path) != expected_sha:
                raise ValueError(f"{resolved}: candidate artifact drift for {identifier}")
            pairs.append((identifier, path))
    if not pairs:
        raise ValueError("at least one --candidate-plan or --candidate-manifest is required")
    candidates: dict[str, Path] = {}
    for identifier, path in pairs:
        if identifier in candidates:
            raise ValueError(f"duplicate candidate id: {identifier}")
        if not path.is_file():
            raise FileNotFoundError(path)
        candidates[identifier] = path
    return dict(sorted(candidates.items())), manifest


def shard_candidate_mapping(
    candidates: Mapping[str, Path], *, shard_index: int, shard_count: int
) -> dict[str, Path]:
    """Deterministically distribute sorted candidates over independent GPUs."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("candidate shard must satisfy 0 <= index < count")
    selected = {
        identifier: path
        for position, (identifier, path) in enumerate(sorted(candidates.items()))
        if position % shard_count == shard_index
    }
    if not selected:
        raise ValueError(
            f"candidate shard {shard_index}/{shard_count} is empty for {len(candidates)} candidates"
        )
    return selected


def standard_error(values: Sequence[float] | np.ndarray) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("paired values must be a finite non-empty vector")
    if vector.size == 1:
        return 0.0
    return float(vector.std(ddof=1) / math.sqrt(vector.size))


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    return vector


def score_vectors(score: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Extract sequence-paired metric and protected-component vectors."""
    functional = score.get("d_func_summary") or {}
    pac = score.get("d_pac_summary") or {}
    result = {
        "d_func": _finite_vector(functional.get("per_sequence"), name="d_func"),
        "d_pac": _finite_vector(pac.get("per_sequence"), name="d_pac"),
    }
    sequences = pac.get("sequences") or []
    if len(sequences) != result["d_pac"].size:
        raise ValueError("D_PAC sequence/component inventory mismatch")
    for component in PROTOCOL["selection"]["component_constraints"]:
        result[component] = _finite_vector(
            [(row.get("components") or {}).get(component) for row in sequences],
            name=component,
        )
    lengths = {vector.size for vector in result.values()}
    if len(lengths) != 1:
        raise ValueError(f"paired metric vectors differ in length: {sorted(lengths)}")
    return result


def task_group_map() -> dict[str, str]:
    """Map every registered Table-1 task to its target split."""
    groups: dict[str, str] = {}
    for group in ("atomic_seen", "composite_seen", "composite_unseen"):
        for task in PROTOCOL["table1"]["tasks"][group]:
            groups[str(task)] = group
    return groups


def task_scalars(score: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    """Per-(metric, task) scalars from a summarize_pair score document.

    Seeds are aggregated inside each task.  D_func is the macro mean of the
    task's per-sequence values; D_PAC recomputes the full mean + CVaR form
    per seed (tail over that seed's replans) and averages over seeds.  This
    is the single task-macro estimand used by both the numerator and the
    denominator of the paired objective.
    """
    from quantvla_cross_model_protocol import PROTOCOL as CROSS_MODEL_PROTOCOL

    functional = score.get("d_func_summary") or {}
    pac = score.get("d_pac_summary") or {}
    d_func_seq = _finite_vector(functional.get("per_sequence"), name="d_func")
    d_pac_seq = _finite_vector(pac.get("per_sequence"), name="d_pac")
    sequences = pac.get("sequences") or []
    if len(sequences) != d_pac_seq.size:
        raise ValueError("D_PAC sequence/component inventory mismatch")
    tasks = [str(row["task"]) for row in sequences]
    seeds = [int(row["seed"]) for row in sequences]
    alpha = float(CROSS_MODEL_PROTOCOL["metrics"]["d_pac"]["outer_cvar_alpha"])
    weight = float(CROSS_MODEL_PROTOCOL["metrics"]["d_pac"]["outer_cvar_weight"])
    result: dict[tuple[str, str], float] = {}
    for task in sorted(set(tasks)):
        indices = [index for index, value in enumerate(tasks) if value == task]
        result[("d_func", task)] = float(d_func_seq[indices].mean())
        by_seed: dict[int, list[float]] = {}
        for index in indices:
            by_seed.setdefault(seeds[index], []).append(float(d_pac_seq[index]))
        seed_values = []
        for values in by_seed.values():
            vector = np.asarray(values, dtype=np.float64)
            tail_count = max(1, int(math.ceil((1.0 - alpha) * vector.size)))
            tail = np.sort(vector)[-tail_count:]
            seed_values.append(float(vector.mean() + weight * tail.mean()))
        result[("d_pac", task)] = float(np.mean(seed_values))
    return result


def jackknife_task_se(values: np.ndarray) -> float:
    """Leave-one-task-out jackknife SE of the mean over tasks."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("task values must be a finite non-empty vector")
    if vector.size < 2:
        return 0.0
    mean = float(vector.mean())
    squared = float(np.sum((vector - mean) ** 2))
    # For the sample mean, the delete-one estimates are
    #   theta_(i) = (n * mean - x_i) / (n - 1).
    # Applying the jackknife variance formula to those estimates reduces to
    # sum_i (x_i - mean)^2 / (n * (n - 1)), which is also the usual standard
    # error of the mean.  Applying the jackknife prefactor directly to the
    # original observations would overestimate the SE by a factor of n - 1.
    return float(math.sqrt(squared / (vector.size * (vector.size - 1))))


def paired_candidate_summary(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    """Task-cluster paired minimax summary (v2 statistics).

    Numerator and denominator share one estimand: the per-task scalar of
    ``task_scalars``.  For every metric and every task cluster (all plus the
    registered splits), the paired upper bound is the task-level mean delta
    plus its leave-one-task-out jackknife SE, normalized by the baseline
    task-macro scalar of the same cluster.  The objective is the worst
    normalized upper bound across metrics x clusters.
    """
    candidate_tasks = task_scalars(candidate)
    baseline_tasks = task_scalars(baseline)
    if set(candidate_tasks) != set(baseline_tasks):
        raise ValueError("candidate and baseline task inventories differ")
    tasks = sorted({task for _, task in candidate_tasks})
    groups = task_group_map()
    clusters: dict[str, list[str]] = {"all": tasks}
    for task in tasks:
        group = groups.get(task)
        if group:
            clusters.setdefault(group, []).append(task)
    metrics: dict[str, Any] = {}
    for key in ("d_func", "d_pac"):
        entries = []
        for cluster, cluster_tasks in clusters.items():
            baseline_scalar = float(
                np.mean([baseline_tasks[(key, task)] for task in cluster_tasks])
            )
            deltas = np.asarray(
                [
                    candidate_tasks[(key, task)] - baseline_tasks[(key, task)]
                    for task in cluster_tasks
                ],
                dtype=np.float64,
            )
            mean = float(deltas.mean())
            se = jackknife_task_se(deltas)
            scale = max(baseline_scalar, 1e-12)
            entries.append(
                {
                    "cluster": cluster,
                    "n_tasks": int(deltas.size),
                    "baseline": baseline_scalar,
                    "delta_mean": mean,
                    "delta_se": se,
                    "upper_bound": mean + se,
                    "normalized_upper_bound": (mean + se) / scale,
                }
            )
        metrics[key] = {
            "clusters": entries,
            "upper_bound": max(entry["upper_bound"] for entry in entries),
            "normalized_upper_bound": max(
                entry["normalized_upper_bound"] for entry in entries
            ),
        }
    candidate_vectors = score_vectors(candidate)
    baseline_vectors = score_vectors(baseline)
    sequences = (candidate.get("d_pac_summary") or {}).get("sequences") or []
    if len(sequences) != candidate_vectors["d_pac"].size:
        raise ValueError("D_PAC sequence/component inventory mismatch")
    task_labels = [str(row["task"]) for row in sequences]
    components: dict[str, Any] = {}
    for key in PROTOCOL["selection"]["component_constraints"]:
        delta = candidate_vectors[key] - baseline_vectors[key]
        per_task: dict[str, list[float]] = {}
        for index, task in enumerate(task_labels):
            per_task.setdefault(task, []).append(float(delta[index]))
        task_deltas = np.asarray(
            [float(np.mean(values)) for values in per_task.values()],
            dtype=np.float64,
        )
        mean = float(task_deltas.mean())
        se = jackknife_task_se(task_deltas)
        components[key] = {
            "delta": delta.tolist(),
            "mean": mean,
            "se": se,
            "passes": bool(mean + se <= 0.0),
        }
    objective = max(
        metrics["d_func"]["normalized_upper_bound"],
        metrics["d_pac"]["normalized_upper_bound"],
    )
    return {
        "n_sequences": int(candidate_vectors["d_func"].size),
        "n_tasks": len(tasks),
        "metrics": metrics,
        "components": components,
        "objective": float(objective),
        "component_constraints_pass": all(row["passes"] for row in components.values()),
        "eligible": bool(
            objective < 0.0 and all(row["passes"] for row in components.values())
        ),
    }


def conservative_fp16_benefit(
    *, current_is_fp16: bool, flip: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    """Estimate the conservative value of choosing FP16 for one layer.

    ``flip`` is always measured against the current complete mask.  When the
    current state is W4, the flip is W4->FP16 and its upper loss bound is
    negated.  When current is FP16, the flip is FP16->W4 and its lower loss
    bound is the conservative value of retaining FP16.  Bounds use the same
    task-cluster scalar estimand as ``paired_candidate_summary``; the most
    conservative cluster wins.
    """
    flip_tasks = task_scalars(flip)
    base_tasks = task_scalars(baseline)
    if set(flip_tasks) != set(base_tasks):
        raise ValueError("flip and baseline task inventories differ")
    tasks = sorted({task for _, task in flip_tasks})
    groups = task_group_map()
    clusters: dict[str, list[str]] = {"all": tasks}
    for task in tasks:
        group = groups.get(task)
        if group:
            clusters.setdefault(group, []).append(task)
    result: dict[str, float] = {}
    for key in ("d_func", "d_pac"):
        bounds = []
        for cluster_tasks in clusters.values():
            scale = max(
                float(np.mean([base_tasks[(key, task)] for task in cluster_tasks])),
                1e-12,
            )
            deltas = np.asarray(
                [
                    flip_tasks[(key, task)] - base_tasks[(key, task)]
                    for task in cluster_tasks
                ],
                dtype=np.float64,
            )
            mean = float(deltas.mean())
            se = jackknife_task_se(deltas)
            if current_is_fp16:
                bounds.append((mean - se) / scale)
            else:
                bounds.append(-(mean + se) / scale)
        result[key] = min(bounds)
    return result


@dataclass(frozen=True)
class BudgetItem:
    name: str
    extra_bytes: int
    benefit_d_func: float
    benefit_d_pac: float


@dataclass(frozen=True)
class KnapsackResult:
    protected: tuple[str, ...]
    extra_bytes: int
    utility: float


def _state_better(
    candidate: tuple[float, tuple[str, ...]], current: tuple[float, tuple[str, ...]]
) -> bool:
    if candidate[0] > current[0] + 1e-15:
        return True
    if abs(candidate[0] - current[0]) <= 1e-15:
        return (len(candidate[1]), candidate[1]) < (len(current[1]), current[1])
    return False


def exact_weighted_knapsack(
    items: Iterable[BudgetItem], *, budget_bytes: int, lambda_d_func: float
) -> KnapsackResult:
    """Exact sparse-frontier 0/1 knapsack with deterministic tie-breaking."""
    if not isinstance(budget_bytes, int) or budget_bytes < 0:
        raise ValueError("budget_bytes must be a non-negative integer")
    if not math.isfinite(lambda_d_func) or not 0.0 <= lambda_d_func <= 1.0:
        raise ValueError("lambda_d_func must lie in [0,1]")
    ordered = sorted(items, key=lambda item: item.name)
    if len({item.name for item in ordered}) != len(ordered):
        raise ValueError("budget item names must be unique")
    for item in ordered:
        if item.extra_bytes <= 0:
            raise ValueError(f"{item.name}: FP16 extra bytes must be positive")
        if not all(math.isfinite(value) for value in (item.benefit_d_func, item.benefit_d_pac)):
            raise ValueError(f"{item.name}: non-finite benefit")

    states: dict[int, tuple[float, tuple[str, ...]]] = {0: (0.0, ())}
    for item in ordered:
        utility = (
            lambda_d_func * item.benefit_d_func
            + (1.0 - lambda_d_func) * item.benefit_d_pac
        )
        if utility <= 0.0 or item.extra_bytes > budget_bytes:
            continue
        updated = dict(states)
        for cost, state in states.items():
            new_cost = cost + item.extra_bytes
            if new_cost > budget_bytes:
                continue
            candidate = (state[0] + utility, state[1] + (item.name,))
            previous = updated.get(new_cost)
            if previous is None or _state_better(candidate, previous):
                updated[new_cost] = candidate

        # Exact dominance pruning: a state is removable only when a cheaper
        # state has at least as much utility.  Costs and utilities are not binned.
        frontier: dict[int, tuple[float, tuple[str, ...]]] = {}
        best_utility = -math.inf
        for cost in sorted(updated):
            state = updated[cost]
            if state[0] > best_utility + 1e-15:
                frontier[cost] = state
                best_utility = state[0]
        states = frontier

    best_cost, best = min(
        states.items(),
        key=lambda pair: (-pair[1][0], pair[0], len(pair[1][1]), pair[1][1]),
    )
    return KnapsackResult(best[1], int(best_cost), float(best[0]))


_ATTENTION_RE = re.compile(
    r"^backbone\.eagle_model\.language_model\.model\.layers\.(\d+)\.self_attn\.(?:q|k|v|o)_proj$"
)
_MLP_RE = re.compile(
    r"^backbone\.eagle_model\.language_model\.model\.layers\.(\d+)\.mlp\.(?:gate|up|down)_proj$"
)
_FF_PAIR_RE = re.compile(
    r"^action_head\.model\.transformer_blocks\.(\d+)\.ff\.net\.(?:0\.proj|2)$"
)


def v2_coordinate_candidates(
    items: Sequence[BudgetItem],
    *,
    capacity: int,
    historical_protected: Sequence[str] | None = None,
) -> list[tuple[str, tuple[str, ...], float]]:
    """DP-as-generator candidate list (<= 8 dedup, byte-feasible candidates).

    Emits the best single-layer restore, best two-layer combo, best attention
    group, best MLP group, best FF pair, half-budget DP, full-budget DP and
    the historical main-mask control. The DP solver only *generates* masks;
    the full-network scorer remains the sole decider.
    """
    by_name = {item.name: item for item in items}
    candidates: list[tuple[str, tuple[str, ...], float]] = []
    seen: set[tuple[str, ...]] = set()

    def add(identifier: str, protected: Iterable[str], utility: float) -> None:
        mask = tuple(sorted(set(protected)))
        if not mask or any(name not in by_name for name in mask):
            return
        extra = sum(int(by_name[name].extra_bytes) for name in mask)
        if extra > capacity or mask in seen:
            return
        seen.add(mask)
        candidates.append((identifier, mask, utility))

    def benefit(item: BudgetItem) -> float:
        return item.benefit_d_pac + item.benefit_d_func

    feasible = sorted(
        (item for item in items if item.extra_bytes <= capacity),
        key=benefit,
        reverse=True,
    )
    if feasible:
        add("single_best", [feasible[0].name], benefit(feasible[0]))
    top = feasible[:12]
    best_pair: tuple[str, ...] | None = None
    best_pair_value = -math.inf
    for left in range(len(top)):
        for right in range(left + 1, len(top)):
            if top[left].extra_bytes + top[right].extra_bytes > capacity:
                continue
            value = benefit(top[left]) + benefit(top[right])
            if value > best_pair_value:
                best_pair_value = value
                best_pair = (top[left].name, top[right].name)
    if best_pair:
        add("two_best", best_pair, best_pair_value)

    def best_group(regex: Any, family: str) -> None:
        groups: dict[str, list[BudgetItem]] = {}
        for item in items:
            match = regex.match(item.name)
            if match:
                groups.setdefault(match.group(1), []).append(item)
        ranked = sorted(
            (
                (sum(benefit(member) for member in members), group, members)
                for group, members in groups.items()
            ),
            key=lambda row: (-row[0], row[1]),
        )
        for value, group, members in ranked:
            extra = sum(member.extra_bytes for member in members)
            if extra <= capacity:
                add(f"{family}_{group}", [member.name for member in members], value)
                break

    best_group(_ATTENTION_RE, "attention")
    best_group(_MLP_RE, "mlp")
    best_group(_FF_PAIR_RE, "ff_pair")

    half = exact_weighted_knapsack(
        items, budget_bytes=max(1, capacity // 2), lambda_d_func=0.5
    )
    if half.protected:
        add("dp_half_budget", half.protected, half.utility)
    full_best: tuple[float, tuple[str, ...], float] | None = None
    for lam in PROTOCOL["counterfactual_search"]["scalarization_lambdas"]:
        result = exact_weighted_knapsack(
            items, budget_bytes=capacity, lambda_d_func=float(lam)
        )
        if not result.protected:
            continue
        if full_best is None or result.utility > full_best[2]:
            full_best = (float(lam), result.protected, result.utility)
    if full_best:
        add(
            f"dp_full_lambda_{str(full_best[0]).replace('.', 'p')}",
            full_best[1],
            full_best[2],
        )
    if historical_protected:
        add("historical_main_control", historical_protected, 0.0)
    return candidates


def paired_one_se(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_vectors = score_vectors(candidate)
    reference_vectors = score_vectors(reference)
    metrics = {}
    for key in ("d_func", "d_pac"):
        delta = candidate_vectors[key] - reference_vectors[key]
        mean = float(delta.mean())
        se = standard_error(delta)
        metrics[key] = {"mean": mean, "se": se, "passes": bool(mean <= se + 1e-15)}
    return {"metrics": metrics, "passes": all(row["passes"] for row in metrics.values())}


def required_grip_mode(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> str:
    """Switch to physical-event grip when the soft component is identically zero."""
    candidate_vectors = score_vectors(candidate)
    baseline_vectors = score_vectors(baseline)
    if "grip" not in candidate_vectors:
        return "soft"
    delta = candidate_vectors["grip"] - baseline_vectors["grip"]
    if np.allclose(delta, 0.0):
        return "physical_events"
    return "soft"


def select_frozen_candidate(
    *,
    scores: Mapping[str, Mapping[str, Any]],
    baseline_id: str,
    plan_rows: Mapping[str, Mapping[str, Any]],
    candidate_state: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if baseline_id not in scores or baseline_id not in plan_rows:
        raise ValueError("baseline is absent from scores or plan rows")
    if set(scores) != set(plan_rows):
        raise ValueError("score and candidate-plan inventories differ")
    baseline = scores[baseline_id]
    summaries = {
        identifier: paired_candidate_summary(score, baseline)
        for identifier, score in scores.items()
        if identifier != baseline_id
    }
    eligible = [identifier for identifier, row in summaries.items() if row["eligible"]]
    if not eligible:
        return {
            "selected_id": baseline_id,
            "fallback_to_baseline": True,
            "reason": "no_candidate_has_negative_paired_minimax_and_component_safety",
            "summaries": summaries,
        }
    best_id = min(eligible, key=lambda identifier: (summaries[identifier]["objective"], identifier))
    one_se = {
        identifier: paired_one_se(scores[identifier], scores[best_id])
        for identifier in eligible
    }
    qualified = [identifier for identifier in eligible if one_se[identifier]["passes"]]
    adjudication: dict[str, float] = {}
    if candidate_state is not None and qualified:
        # Top-3 by teacher-state objective enter candidate-state adjudication:
        # J_final = max(J_teacher_state, J_candidate_state) and must be < 0.
        top_three = sorted(
            qualified, key=lambda identifier: (summaries[identifier]["objective"], identifier)
        )[:3]
        for identifier in top_three:
            audit = candidate_state.get(identifier)
            if audit is None:
                raise ValueError(f"missing candidate-state audit for {identifier}")
            teacher_j = float(summaries[identifier]["objective"])
            candidate_j = float(audit["j_candidate_state"]["j"])
            adjudication[identifier] = max(teacher_j, candidate_j)
        qualified = [identifier for identifier in qualified if adjudication.get(identifier, 0.0) < 0.0]
    if not qualified:
        return {
            "selected_id": baseline_id,
            "fallback_to_baseline": True,
            "reason": (
                "candidate_state_adjudication_rejected_all_top_candidates"
                if adjudication
                else "no_candidate_passes_the_one_se_screen"
            ),
            "summaries": summaries,
            "candidate_state_adjudication": adjudication,
        }

    def tie_key(identifier: str) -> tuple[Any, ...]:
        row = plan_rows[identifier]
        protected = tuple(sorted(str(value) for value in row.get("protected_layers", ())))
        static_bytes = row.get("table1_total_static_bytes")
        if static_bytes is None:
            static_bytes = row["total_bytes"]
        return (
            int(static_bytes),
            int(row["retained_fp16_layers"]),
            protected,
            identifier,
        )

    selected = min(qualified, key=tie_key)
    return {
        "selected_id": selected,
        "fallback_to_baseline": False,
        "best_objective_id": best_id,
        "selection": summaries[selected],
        "summaries": summaries,
        "paired_one_se": one_se,
        "candidate_state_adjudication": adjudication or None,
        "tie_break": list(PROTOCOL["selection"]["tie_break"]),
    }


def select_activation_mode(
    *, static: Mapping[str, Any], dynamic: Mapping[str, Any], a16: Mapping[str, Any]
) -> dict[str, Any]:
    control = paired_candidate_summary(a16, static)
    bottleneck = all(
        control["metrics"][key]["upper_bound"] < 0.0 for key in ("d_func", "d_pac")
    )
    dynamic_summary = paired_candidate_summary(dynamic, static)
    selected = "dynamic_a8" if bottleneck and dynamic_summary["eligible"] else "static_a8"
    return {
        "a8_bottleneck": bottleneck,
        "a16_control": control,
        "dynamic_candidate": dynamic_summary,
        "selected_activation_mode": selected,
        "a16_deployable": False,
    }


def quick_advancement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate the frozen 5x10 development gate from paired episode rows."""
    if not rows:
        raise ValueError("quick gate requires paired rows")
    tasks: dict[str, list[tuple[bool, bool]]] = {}
    wins = losses = 0
    for row in rows:
        task = str(row["task"])
        main = bool(row["main_success"])
        candidate = bool(row["candidate_success"])
        tasks.setdefault(task, []).append((main, candidate))
        wins += int(candidate and not main)
        losses += int(main and not candidate)
    expected_tasks = set(PROTOCOL["quick_development"]["tasks"])
    expected_seeds = set(PROTOCOL["quick_development"]["seeds"])
    observed_tasks = set(tasks)
    observed_seeds = {int(row["seed"]) for row in rows}
    if observed_tasks != expected_tasks or observed_seeds != expected_seeds:
        raise ValueError("quick development task/seed coverage drift")
    if len(rows) != int(PROTOCOL["quick_development"]["episodes_per_config"]):
        raise ValueError("quick development episode count drift")
    observed_keys = {(str(row["task"]), int(row["seed"])) for row in rows}
    expected_keys = {
        (task, seed) for task in expected_tasks for seed in expected_seeds
    }
    if observed_keys != expected_keys or len(observed_keys) != len(rows):
        raise ValueError("quick development paired-key coverage drift")
    main_total = sum(int(bool(row["main_success"])) for row in rows)
    candidate_total = sum(int(bool(row["candidate_success"])) for row in rows)
    main_macro = float(np.mean([np.mean([m for m, _ in values]) for values in tasks.values()]))
    candidate_macro = float(np.mean([np.mean([c for _, c in values]) for values in tasks.values()]))
    return {
        "main_successes": main_total,
        "candidate_successes": candidate_total,
        "main_task_macro": main_macro,
        "candidate_task_macro": candidate_macro,
        "paired_wins": wins,
        "paired_losses": losses,
        "passes_success_gate": bool(
            candidate_total > main_total and candidate_macro >= main_macro and wins > losses
        ),
    }


def selftest() -> None:
    from quantvla_table1_bytes import selftest as table1_selftest
    from quantvla_table1_bytes import table1_variable_budget

    table1_selftest()
    assert PROTOCOL["byte_budget"]["maximum_quantvla_byte_multiplier"] == 1.1
    assert PROTOCOL["quick_development"]["seeds"] == list(range(50, 60))
    assert table1_variable_budget("gr00t") == 732_797_337
    assert table1_variable_budget("pi05") == 1_639_513_497
    assert PROTOCOL_V2["method_id"] == "full_context_fp16_protection_v2"
    assert PROTOCOL_V2["selection"]["component_rule"] == (
        "mean(delta_component)+SE_task(delta_component)<=0"
    )
    assert PROTOCOL_V2["selection"]["uses_task_ids_for_stratified_statistics"] is True
    assert (
        PROTOCOL_V2["selection"]["uses_task_labels_for_routing_or_task_specific_mask"]
        is False
    )
    assert PROTOCOL_V2["byte_budget"]["table1_quantvla_component_bytes"]["gr00t"] == (
        963_772_416
    )
    print(f"[full-context] selftest OK {PROTOCOL_SHA256}")


if __name__ == "__main__":
    selftest()
