#!/usr/bin/env python3
"""Generate a π0.5 GDSQ diagnostic plan that protects action-expert MLPs.

The current frozen π0.5 GDSQ plan was selected by the GR00T-final proxy plus
TopK D_func adjudication.  This diagnostic variant tests whether the RoboCasa
gap is caused by quantizing the π0.5 action expert MLP: every
``action_expert_mlp`` candidate is restored to native FP16, and the byte budget
is repaired by quantizing the lowest-cost non-expert FP16 layers according to
the same selector proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "scripts" / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

import gr00t_select_plan as final_selector  # noqa: E402


DEFAULT_SELECTOR = (
    REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_gdsq_cscka_16to1_d4.selector.json"
)
DEFAULT_FINAL = (
    REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_gdsq_cscka_16to1_d4.final_plan.json"
)
DEFAULT_INVENTORY = (
    REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_candidate_inventory_d4.json"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/diagnostics/expert_protect/"
    / "pi05_gdsq_expert_mlp_protected.plan.json"
)
DEFAULT_BUFFER = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", default=str(DEFAULT_SELECTOR))
    parser.add_argument("--final-plan", default=str(DEFAULT_FINAL))
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--sensitivity", default=None)
    parser.add_argument("--buffer", default=str(DEFAULT_BUFFER))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--row-rot", default="restore", choices=("restore",))
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def inventory_maps(inventory: dict[str, Any]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    families: dict[str, str] = {}
    shapes: dict[str, dict[str, Any]] = {}
    for row in inventory.get("layers") or []:
        name = str(row["name"])
        families[name] = str(row.get("family") or "")
        shapes[name] = {
            "out": int(row["out_features"]),
            "in": int(row["in_features"]),
            "has_bias": False,
        }
    return families, shapes


def is_w4(row: dict[str, Any]) -> bool:
    return not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4


def set_w4(row: dict[str, Any], group: int) -> None:
    row["bits"] = 4
    row["group"] = group
    row["skip"] = False


def set_fp16(row: dict[str, Any], group: int) -> None:
    row["bits"] = 0
    row["group"] = group
    row["skip"] = True


def layer_costs(
    name: str,
    shapes: dict[str, dict[str, Any]],
    group: int,
    row_rot: str,
) -> tuple[float, float, float]:
    shape = shapes[name]
    fp16 = final_selector.layer_bytes_fp16(shape["out"], shape["in"], shape["has_bias"])
    w4 = final_selector.layer_bytes_quant(
        shape["out"], shape["in"], shape["has_bias"], 4, group, row_rot
    )
    return fp16, w4, fp16 - w4


def plan_bytes(
    layers: dict[str, dict[str, Any]],
    shapes: dict[str, dict[str, Any]],
    row_rot: str,
) -> float:
    return final_selector.plan_total_bytes(layers, shapes, row_rot)


def mask_hash(layers: dict[str, dict[str, Any]]) -> str:
    quantized = sorted(name for name, row in layers.items() if is_w4(row))
    payload = json.dumps(quantized, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def build(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    selector_path = Path(args.selector).expanduser().resolve()
    final_path = Path(args.final_plan).expanduser().resolve()
    inventory_path = Path(args.inventory).expanduser().resolve()
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    base = json.loads(final_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    sensitivity_path = Path(
        args.sensitivity or selector["meta"]["sensitivity_path"]
    ).expanduser().resolve()
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    buffer_path = Path(args.buffer).expanduser().resolve()
    if sensitivity.get("meta", {}).get("buffer_sha256") != sha256_file(buffer_path):
        raise ValueError("sensitivity/calibration-buffer hash mismatch")
    families, shapes = inventory_maps(inventory)
    base_layers = base.get("layers") or {}
    if set(base_layers) != set(sensitivity.get("layers", {})) or set(base_layers) != set(shapes):
        raise ValueError("final plan, selector, and inventory layer sets differ")

    group = int(args.group)
    budget = float(base.get("meta", {}).get("budget_bytes") or selector["budget_bytes"])
    final_selector.BITS_ORDER[:] = [4]
    lambdas = selector["meta"]["lambda"]
    scores = final_selector.build_scores(
        sensitivity,
        list(base_layers),
        float(lambdas["cka"]),
        float(lambdas["cs"]),
        cka_field="cka_dit",
    )
    weights, weight_log = final_selector.build_weights_with_log(
        sensitivity,
        list(base_layers),
        metric="d_func",
    )
    thresholds = selector["meta"].get("guard_thresholds") or {}
    filtered_scores, removed = final_selector.filter_guarded(
        scores,
        sensitivity,
        list(base_layers),
        thresholds.get("tau_rms"),
        thresholds.get("tau_sat"),
        bit=4,
    )
    guarded_names = {row["layer"] for row in removed}
    layers = {
        name: {"bits": int(row.get("bits", 0) or 0), "group": group, "skip": bool(row.get("skip", False))}
        for name, row in base_layers.items()
    }
    original_bytes = plan_bytes(layers, shapes, args.row_rot)
    if original_bytes > budget + 1e-3:
        raise ValueError("base final plan already exceeds budget")

    protected: list[str] = []
    for name, family in families.items():
        if family == "action_expert_mlp" and is_w4(layers[name]):
            set_fp16(layers[name], group)
            protected.append(name)

    after_protect_bytes = plan_bytes(layers, shapes, args.row_rot)
    candidates: list[dict[str, Any]] = []
    for name, row in layers.items():
        if is_w4(row) or families[name] == "action_expert_mlp":
            continue
        fp16, w4, saved = layer_costs(name, shapes, group, args.row_rot)
        if saved <= 0:
            continue
        score = (filtered_scores.get(name) or {}).get(4)
        weight = weights.get(name)
        if score is None or weight is None:
            continue
        penalty = float(score) * float(weight)
        candidates.append(
            {
                "name": name,
                "saved_bytes": saved,
                "penalty": penalty,
                "penalty_per_gb_saved": penalty / max(saved / 1e9, 1e-12),
                "fp16_bytes": fp16,
                "w4_bytes": w4,
            }
        )
    candidates.sort(key=lambda row: (row["penalty_per_gb_saved"], row["penalty"], -row["saved_bytes"], row["name"]))

    repaired: list[dict[str, Any]] = []
    current = after_protect_bytes
    for candidate in candidates:
        if current <= budget + 1e-3:
            break
        name = candidate["name"]
        set_w4(layers[name], group)
        current -= candidate["saved_bytes"]
        repaired.append(candidate)
    if current > budget + 1e-3:
        raise RuntimeError("could not repair expert-protected plan under the existing budget")

    payload = {
        "schema_version": 2,
        "meta": {
            **(base.get("meta") or {}),
            "kind": "gdsq_vla_pi05_diagnostic_expert_mlp_protected",
            "diagnostic_hypothesis": "RoboCasa π0.5 action-expert MLP quantization is responsible for the observed GDSQ-VLA gap",
            "parent_final_plan_path": str(final_path),
            "parent_final_plan_sha256": sha256_file(final_path),
            "parent_selector_path": str(selector_path),
            "parent_selector_sha256": sha256_file(selector_path),
            "sensitivity_path": str(sensitivity_path),
            "sensitivity_sha256": sha256_file(sensitivity_path),
            "calibration_buffer_path": str(buffer_path),
            "calibration_buffer_sha256": sha256_file(buffer_path),
            "inventory_path": str(inventory_path),
            "inventory_file_sha256": sha256_file(inventory_path),
            "budget_bytes": budget,
            "total_bytes": current,
            "adjudicated": False,
            "requires_fresh_plan_specific_a8": True,
            "requires_fresh_plan_specific_atm_ohb": True,
            "expert_mlp_protected": True,
            "repair_rule": "quantize non-expert FP16 layers by lowest selector penalty per byte saved until the parent byte budget is met",
            "score_source": "recomputed from parent sensitivity; serialized skip-layer scores are intentionally ignored",
            "weight_log": weight_log,
            "guard_filtered_layers": sorted(guarded_names),
            "mask_sha256": mask_hash(layers),
            "parent_mask_sha256": mask_hash(base_layers),
        },
        "packdirs": base.get("packdirs", selector.get("packdirs", {})),
        "layers": layers,
    }
    report = {
        "schema_version": 1,
        "plan_path": str(Path(args.out).expanduser().resolve()),
        "parent_final_plan_path": str(final_path),
        "parent_final_plan_sha256": sha256_file(final_path),
        "parent_selector_path": str(selector_path),
        "parent_selector_sha256": sha256_file(selector_path),
        "sensitivity_path": str(sensitivity_path),
        "sensitivity_sha256": sha256_file(sensitivity_path),
        "budget_bytes": budget,
        "parent_total_bytes": original_bytes,
        "after_protect_bytes": after_protect_bytes,
        "final_total_bytes": current,
        "budget_slack_bytes": budget - current,
        "protected_expert_w4_to_fp16": protected,
        "repaired_nonexpert_fp16_to_w4": repaired,
        "counts": summarize(layers, families),
        "parent_counts": summarize(base_layers, families),
        "mask_sha256": payload["meta"]["mask_sha256"],
        "parent_mask_sha256": payload["meta"]["parent_mask_sha256"],
    }
    return payload, report


def summarize(layers: dict[str, dict[str, Any]], families: dict[str, str]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    total_w4 = 0
    total_fp16 = 0
    for name, row in layers.items():
        family = families.get(name, "unknown")
        counts.setdefault(family, {"w4": 0, "fp16": 0})
        if is_w4(row):
            counts[family]["w4"] += 1
            total_w4 += 1
        else:
            counts[family]["fp16"] += 1
            total_fp16 += 1
    return {"total_w4": total_w4, "total_fp16": total_fp16, "by_family": counts}


def selftest() -> None:
    assert math.isclose(1.0, 1.0)
    print("[pi05_make_expert_protected_plan] selftest OK")


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    plan, report = build(args)
    output = Path(args.out).expanduser().resolve()
    report_path = Path(str(output) + ".report.json")
    atomic_json(output, plan)
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": sha256_file(output),
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "counts": report["counts"],
                "budget_slack_bytes": report["budget_slack_bytes"],
                "protected": len(report["protected_expert_w4_to_fp16"]),
                "repaired": len(report["repaired_nonexpert_fp16_to_w4"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
