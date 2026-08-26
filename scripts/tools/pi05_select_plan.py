#!/usr/bin/env python3
"""Faithful QuantVLA-final W4/FP16 selector for π0.5.

This is an architecture adapter around the authoritative GR00T final search
implementation in :mod:`gr00t_select_plan`.  It intentionally reuses that
implementation's joint min-max score fusion, action-importance weighting,
hard guards, greedy/MILP/perturbation/flip/lambda candidates, diverse TopK,
and byte model.  π0.5-specific code is limited to reading its immutable layer
inventory and serializing OpenPI-compatible plans.

The emitted primary plan is a proxy-search result only.  ``adjudicated`` stays
false until true mixed deployment has been scored against original FP16 with
D_PAC in ``pi05_topk_scorer.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "scripts" / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

import gr00t_select_plan as final_selector  # noqa: E402
from pi05_func_metrics import FUNCTIONAL_FORMULA_ID  # noqa: E402


DEFAULT_SENSITIVITY = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/sensitivity/pi05_sensitivity_action_n16_d4_merged.json"
)
DEFAULT_INVENTORY = (
    REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_candidate_inventory_d4.json"
)
DEFAULT_PACK = REPO_ROOT / "runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
DEFAULT_BUFFER = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
)
FINAL_METRIC = TOOLS_ROOT / "pi05_func_metrics.py"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_lambda_pairs(value: str) -> tuple[tuple[float, float], ...]:
    pairs: list[tuple[float, float]] = []
    for item in value.split(";"):
        left, right = item.split(",", 1)
        pair = (float(left), float(right))
        if pair[0] < 0 or pair[1] < 0 or pair == (0.0, 0.0):
            raise ValueError(f"invalid lambda pair: {item}")
        pairs.append(pair)
    if not pairs:
        raise ValueError("lambda-pairs must not be empty")
    return tuple(pairs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", default=str(DEFAULT_SENSITIVITY))
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--pack-dir", default=str(DEFAULT_PACK))
    parser.add_argument("--calibration-buffer", default=str(DEFAULT_BUFFER))
    parser.add_argument("--ratio", type=float, default=16.0)
    parser.add_argument("--cka-only", action="store_true")
    parser.add_argument("--cs-only", action="store_true")
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--row-rot", default="restore", choices=("restore",))
    parser.add_argument(
        "--budget-reference",
        choices=("quantvla-w4", "uniform-w6"),
        default="quantvla-w4",
        help="Exact QuantVLA all-W4 bytes (default) or the legacy uniform-W6 budget.",
    )
    parser.add_argument("--n-perturb", type=int, default=10)
    parser.add_argument("--perturb-sigma", type=float, default=0.25)
    parser.add_argument("--n-topk", type=int, default=10)
    parser.add_argument("--min-hamming", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--guard-margin", type=float, default=1.5)
    parser.add_argument(
        "--lambda-pairs",
        default="0,1;0,2;0,5;1,1;0.5,1;2,1",
        help="Same default lambda-sweep candidate set as GR00T final.",
    )
    parser.add_argument("--no-milp", action="store_true")
    parser.add_argument("--out", required=False)
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def _identity_and_lambdas(args: argparse.Namespace) -> tuple[str, float, float]:
    if args.cka_only and args.cs_only:
        raise ValueError("--cka-only and --cs-only are mutually exclusive")
    if args.cka_only:
        return "cka_only", 1.0, 0.0
    if args.cs_only:
        return "cs_only", 0.0, 1.0
    if not math.isfinite(args.ratio) or args.ratio <= 0:
        raise ValueError("ratio must be finite and positive")
    return f"cka_cs_{args.ratio:g}_to_1", float(args.ratio), 1.0


def _shapes(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    layers = inventory.get("layers") or []
    if len(layers) != 180:
        raise ValueError(f"π0.5 final inventory must contain 180 layers, got {len(layers)}")
    names = [str(row["name"]) for row in layers]
    if len(names) != len(set(names)):
        raise ValueError("duplicate names in π0.5 candidate inventory")
    return {
        str(row["name"]): {
            "out": int(row["out_features"]),
            "in": int(row["in_features"]),
            # The attested π0.5 candidate inventory contains Gemma projections;
            # the released checkpoint has no bias tensors for these 180 layers.
            "has_bias": False,
        }
        for row in layers
    }


def _openpi_entry(
    entry: dict[str, Any],
    *,
    group: int,
    score: float,
    weight: float,
    sensitivity: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    skip = bool(entry["skip"])
    measurement = sensitivity["layers"][name]
    b4 = measurement.get("b4") or {}
    return {
        "bits": 0 if skip else 4,
        "group": group,
        "skip": skip,
        "score": 0.0 if skip else float(score),
        "weight": float(weight),
        "cka_action": b4.get("cka_dit", b4.get("cka_action")),
        "cka_local": b4.get("cka"),
        "cs": b4.get("cs"),
        "rms_ratio": b4.get("rms_ratio"),
        "sat_rate": b4.get("sat_rate"),
        "d_func_ref": measurement.get("d_func_b4"),
        "d_func_ref_std": measurement.get("d_func_b4_std"),
        "d_pac_ref": measurement.get("d_pac_b4"),
        "d_pac_ref_std": measurement.get("d_pac_b4_std"),
    }


def select(args: argparse.Namespace) -> dict[str, Any]:
    identity, lambda_cka, lambda_cs = _identity_and_lambdas(args)
    sensitivity_path = Path(args.sensitivity).expanduser().resolve()
    inventory_path = Path(args.inventory).expanduser().resolve()
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    buffer_path = Path(args.calibration_buffer).expanduser().resolve()
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if sensitivity.get("complete") is not True:
        raise ValueError("sensitivity matrix is not complete")
    if sensitivity.get("meta", {}).get("cka_location") != (
        "action_expert_final_hidden_pre_action_out_proj"
    ):
        raise ValueError("final selector requires GR00T-equivalent final-hidden CKA")
    if sensitivity.get("meta", {}).get("cs_location") != (
        "intervened_target_linear_output"
    ):
        raise ValueError("final selector requires GR00T-equivalent local Linear CS")
    if int(sensitivity.get("meta", {}).get("n_obs", 0)) != 16:
        raise ValueError("final selector requires n_obs=16 sensitivity")
    if int(sensitivity.get("meta", {}).get("n_rollout_obs", 0)) != 8:
        raise ValueError("final selector requires GR00T-final n_rollout_obs=8 importance")
    if int(sensitivity.get("meta", {}).get("n_noises_per_obs", 0)) != 2:
        raise ValueError("final selector requires two paired noise sets per observation")
    if sensitivity.get("meta", {}).get("functional_metric_sha256") != sha256_file(
        FINAL_METRIC
    ):
        raise ValueError(
            "sensitivity was produced by a different π0.5 functional metric; "
            "rerun the complete 180-layer probe"
        )
    formula = sensitivity.get("meta", {}).get("functional_formula") or {}
    if formula.get("formula_id") != FUNCTIONAL_FORMULA_ID:
        raise ValueError("sensitivity does not use the authoritative GR00T-final D_func")
    if int(sensitivity.get("meta", {}).get("flow_steps", 0)) != 4:
        raise ValueError("final selector requires GR00T-aligned 4-step sensitivity")
    if int(sensitivity.get("meta", {}).get("execute_actions", 0)) != 16:
        raise ValueError("final selector requires GR00T-aligned execute-16 sensitivity")
    if lambda_cs > 0 and not bool(
        (sensitivity.get("meta", {}).get("cs_in_situ_check") or {}).get("cross_monotonic")
    ):
        raise ValueError("CS in-situ scale-response gate failed; use --cka-only (lambda_cs=0)")
    if sensitivity["meta"].get("buffer_sha256") != sha256_file(buffer_path):
        raise ValueError("sensitivity/calibration-buffer hash mismatch")
    manifest_path = pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("wrapped_layer_count") != 180:
        raise ValueError("block64 pack manifest is not complete")

    shapes = _shapes(inventory)
    names = list(shapes)
    if set(names) != set(sensitivity.get("layers", {})):
        raise ValueError("sensitivity and candidate inventories differ")

    # Force the exact final binary search space in the reused authority module.
    final_selector.BITS_ORDER[:] = [4]
    scores = final_selector.build_scores(
        sensitivity, names, lambda_cka, lambda_cs, cka_field="cka_dit"
    )
    unguarded_scores = scores
    weights, weight_log = final_selector.build_weights_with_log(
        sensitivity, names, metric="d_pac"
    )

    thresholds = sensitivity.get("meta", {}).get("guard_thresholds") or {}
    tau_rms = thresholds.get("tau_rms")
    tau_sat = thresholds.get("tau_sat")
    if tau_rms is None or tau_sat is None:
        estimated_rms, estimated_sat = final_selector.estimate_guard_thresholds(
            sensitivity, names, bit=4, margin=args.guard_margin
        )
        tau_rms = estimated_rms if tau_rms is None else tau_rms
        tau_sat = estimated_sat if tau_sat is None else tau_sat
    filtered_scores, removed = final_selector.filter_guarded(
        scores, sensitivity, names, tau_rms, tau_sat, bit=4
    )
    guard_diagnostics = list(removed)
    exact_quantvla_budget = args.budget_reference == "quantvla-w4"
    if exact_quantvla_budget:
        filtered_scores = unguarded_scores
        removed = []
    guarded_names = {row["layer"] for row in removed}

    fp16_bytes = sum(
        final_selector.layer_bytes_fp16(shape["out"], shape["in"], shape["has_bias"])
        for shape in shapes.values()
    )
    if exact_quantvla_budget:
        budget = final_selector.quantvla_w4_budget(shapes, args.group, args.row_rot)
        greedy = final_selector.quantvla_w4_plan(shapes, args.group, args.row_rot)
        greedy_objective = final_selector.plan_objective(greedy, filtered_scores, weights)
        candidates: list[dict[str, Any]] = [
            {
                "plan": greedy,
                "objective": greedy_objective,
                "source": "quantvla_full_w4",
            }
        ]
    else:
        uniform_w6 = {
            name: {"bits": 6, "group": args.group, "skip": False} for name in names
        }
        budget = final_selector.plan_total_bytes(uniform_w6, shapes, args.row_rot)
        greedy, greedy_objective = final_selector.greedy_plan(
            shapes, filtered_scores, weights, budget, args.row_rot
        )
        candidates = [
            {"plan": greedy, "objective": greedy_objective, "source": "greedy"}
        ]
    if not exact_quantvla_budget and not args.no_milp:
        milp, milp_objective = final_selector.milp_binary_plan(
            shapes,
            filtered_scores,
            weights,
            budget,
            args.row_rot,
            group=args.group,
        )
        if milp is not None:
            candidates.append(
                {"plan": milp, "objective": milp_objective, "source": "milp"}
            )
    if not exact_quantvla_budget:
        candidates.extend(
            final_selector.perturbed_plans(
                shapes,
                filtered_scores,
                weights,
                budget,
                args.row_rot,
                n=args.n_perturb,
                sigma=args.perturb_sigma,
                seed=1,
            )
        )
        candidates.extend(
            final_selector.flip_neighbors_plans(
                shapes,
                filtered_scores,
                weights,
                budget,
                args.row_rot,
                base_plan=greedy,
                seed=3,
            )
        )
        candidates.extend(
            final_selector.lambda_sweep_plans(
                shapes,
                sensitivity,
                names,
                weights,
                budget,
                args.row_rot,
                pairs=parse_lambda_pairs(args.lambda_pairs),
                guarded_names=guarded_names,
                cka_field="cka_dit",
            )
        )

    for candidate in candidates:
        final_selector.assert_plan_guards(candidate["plan"], guarded_names)
        candidate["objective_native"] = candidate["objective"]
        candidate["objective"] = final_selector.plan_objective(
            candidate["plan"], filtered_scores, weights
        )
        candidate["bytes"] = final_selector.plan_total_bytes(
            candidate["plan"], shapes, args.row_rot
        )
        if candidate["bytes"] > budget + 1e-3:
            raise RuntimeError(f"candidate {candidate['source']} exceeds budget")

    min_hamming = (
        args.min_hamming
        if args.min_hamming is not None
        else max(3, math.ceil(0.1 * len(shapes)))
    )
    topk = final_selector.select_diverse(
        candidates, k=args.n_topk, min_hamming=min_hamming
    )
    if not exact_quantvla_budget and len(topk) < 5:
        raise RuntimeError(f"diverse TopK collapsed to {len(topk)} candidates")
    primary = min(candidates, key=lambda candidate: candidate["objective"])
    primary_bytes = final_selector.plan_total_bytes(primary["plan"], shapes, args.row_rot)

    if exact_quantvla_budget:
        bootstrap = {
            "n_draws": 0,
            "deterministic_full_w4": True,
            "jaccard": {"mean": 1.0, "p05": 1.0, "p95": 1.0},
            "spearman_mean": 1.0,
        }
    else:
        bootstrap = final_selector.bootstrap_stability(
            sensitivity,
            shapes,
            filtered_scores,
            budget,
            args.row_rot,
            n=args.n_bootstrap,
            seed=2,
        )
    sensitivity_hash = sha256_file(sensitivity_path)
    inventory_hash = sha256_file(inventory_path)
    pack_manifest_hash = sha256_file(manifest_path)
    buffer_hash = sha256_file(buffer_path)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "meta": {
            "kind": "gdsq_vla_pi05_faithful_final_selector",
            "identity": identity,
            "algorithm_authority": str((TOOLS_ROOT / "gr00t_select_plan.py").resolve()),
            "algorithm_authority_sha256": sha256_file(TOOLS_ROOT / "gr00t_select_plan.py"),
            "architecture_adapter": str(Path(__file__).resolve()),
            "architecture_adapter_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoint_sha256": inventory["checkpoint_sha256"],
            "config_sha256": inventory["config_sha256"],
            "norm_stats_sha256": inventory["norm_stats_sha256"],
            "candidate_inventory_sha256": inventory["candidate_inventory_sha256"],
            "inventory_path": str(inventory_path),
            "inventory_file_sha256": inventory_hash,
            "sensitivity_path": str(sensitivity_path),
            "sensitivity_sha256": sensitivity_hash,
            "functional_metric_path": str(FINAL_METRIC.resolve()),
            "functional_metric_sha256": sha256_file(FINAL_METRIC),
            "functional_formula": formula,
            "calibration_buffer_path": str(buffer_path),
            "calibration_buffer_sha256": buffer_hash,
            "pack_dir": str(pack_dir),
            "pack_manifest_sha256": pack_manifest_hash,
            "cka_location": "action expert final hidden state before action_out_proj",
            "cs_location": "intervened target Linear output",
            "lambda": {"cka": lambda_cka, "cs": lambda_cs},
            "cka_to_cs_ratio": None if lambda_cs == 0 else lambda_cka / lambda_cs,
            "score_normalization": "joint min-max over all measured (layer,bit) pairs per term",
            "weight_metric": "d_pac",
            "weight_log": weight_log,
            "guard_thresholds": {
                "tau_rms": tau_rms,
                "tau_sat": tau_sat,
                "rule": (
                    "diagnostic only at exact QuantVLA bytes; full-W4 is adjudicated by FP16-relative D_PAC"
                    if exact_quantvla_budget
                    else "hard removal from W4 search; retained native FP16"
                ),
            },
            "guard_filtered_layers": sorted(guarded_names),
            "guard_diagnostic_layers": sorted(row["layer"] for row in guard_diagnostics),
            "search_space": "binary native-FP16/W4",
            "candidate_sources": (
                ["quantvla_full_w4"]
                if exact_quantvla_budget
                else ["greedy", "milp", "perturbed", "flip8/12/16/20", "lambda-sweep"]
            ),
            "lambda_pairs": [list(pair) for pair in parse_lambda_pairs(args.lambda_pairs)],
            "topk_size": len(topk),
            "topk_min_hamming": min_hamming,
            "budget_reference": (
                "quantvla_all_candidate_w4_static_bytes"
                if exact_quantvla_budget
                else "uniform_w6_static_bytes"
            ),
            "search_start": "QuantVLA full W4" if exact_quantvla_budget else "native FP16",
            "budget_semantics": "theoretical packed static weight bytes only",
            "skip_semantics": "native torch.nn.Linear FP16; no wrapper, rotation, or A8",
            "adjudication_metric": "d_pac_v1",
            "adjudication_rule": "minimum original-FP16-relative D_PAC after configuration freeze",
            "adjudicated": False,
            "group": args.group,
            "row_rot": args.row_rot,
            "act_bits": 8,
            "act_percentile": 99.9,
            "calibration_batches": 32,
            "denoising_steps": 4,
            "evaluation_split": "target",
            "n_action_steps": 16,
            "calibration_noise_protocol": sensitivity["meta"]["noise_protocol"],
            "paired_noise_protocol": sensitivity["meta"]["evaluation_noise_protocol"],
        },
        "budget_bytes": budget,
        "fp16_total_bytes": fp16_bytes,
        "budget_fraction_of_fp16": budget / fp16_bytes,
        "total_bytes": primary_bytes,
        "objective": float(primary["objective"]),
        "primary_source": primary["source"],
        "packdirs": {str(args.group): str(pack_dir)},
        "layers": {},
        "topk": [],
        "bootstrap": bootstrap,
    }
    for name in names:
        entry = primary["plan"][name]
        payload["layers"][name] = _openpi_entry(
            entry,
            group=args.group,
            score=filtered_scores[name].get(4, 0.0) or 0.0,
            weight=weights[name],
            sensitivity=sensitivity,
            name=name,
        )
    for index, candidate in enumerate(topk):
        payload["topk"].append(
            {
                "index": index,
                "source": candidate["source"],
                "objective": float(candidate["objective"]),
                "objective_native": float(candidate["objective_native"]),
                "bytes": float(candidate["bytes"]),
                "n_quantized": sum(
                    not bool(entry["skip"]) for entry in candidate["plan"].values()
                ),
                "n_skip": sum(
                    bool(entry["skip"]) for entry in candidate["plan"].values()
                ),
                "skip_layers": list(final_selector.plan_mask(candidate["plan"])),
                "d_func": None,
                "d_solver": None,
                "d_pac": None,
            }
        )
    return payload


def selftest() -> None:
    final_selector.BITS_ORDER[:] = [4]
    sensitivity = {
        "layers": {
            f"L{index}": {
                "b4": {
                    "cka_dit": 0.9 + index * 1e-4,
                    "cs": index * 1e-3,
                    "rms_ratio": 0.01,
                    "sat_rate": 0.0,
                },
                "d_func_b4": 0.01 + index * 1e-4,
            }
            for index in range(100)
        }
    }
    names = list(sensitivity["layers"])
    scores = final_selector.build_scores(sensitivity, names, 16.0, 1.0, "cka_dit")
    weights, log = final_selector.build_weights_with_log(sensitivity, names, "d_func")
    assert all(scores[name][4] is not None for name in names)
    assert abs(log["final"]["mean"] - 1.0) < 1e-8
    assert len(weights) == 100
    chosen = final_selector.select_final(
        [
            {"d_func": 1.0, "proxy": 2.0, "source": "a"},
            {"d_func": 1.04, "proxy": 1.0, "source": "b"},
            {"d_func": 1.06, "proxy": 0.0, "source": "c"},
        ],
        tol=0.05,
        key="d_func",
    )
    assert chosen["source"] == "b"
    print("[pi05_select_plan] faithful-final selftest OK")


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if not args.out:
        raise SystemExit("--out is required")
    payload = select(args)
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": sha256_file(output),
                "identity": payload["meta"]["identity"],
                "topk": len(payload["topk"]),
                "quantized_layers": sum(not row["skip"] for row in payload["layers"].values()),
                "retained_fp16_layers": sum(row["skip"] for row in payload["layers"].values()),
                "budget_bytes": payload["budget_bytes"],
                "total_bytes": payload["total_bytes"],
                "adjudicated": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
