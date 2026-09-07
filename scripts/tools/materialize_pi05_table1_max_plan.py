#!/usr/bin/env python3
"""Materialize the pi0.5 Table-1 all-W4 (max-compression) deployment plan.

Takes the frozen pi0.5 DyPAC plan inventory (180 adapter-bound layers) and
sets every layer to W4/group-64.  Byte accounting follows the deployed Table-1
scope: fp16 = weight.size x 2, w4 = packed codes + per-output-channel scales
(fixed bytes = 0 for pi0.5).  The authoritative totals come from the frozen
full-W4 Hessian parent sidecar (packed_weight_bytes + scale bytes).
Re-running must reproduce the identical file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from quantvla_cross_model_protocol import sha256_file, validate_quant_plan

FP16_TOTAL_BYTES = 4_416_602_112
FIXED_BYTES = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-plan",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v2/pi05_p2/"
        "pi05_full_context_v2_frozen.json",
    )
    parser.add_argument(
        "--hessian-sidecar",
        default="/home1/gyy/vla/QuantVLA/runs/errorfold_v4_iter/calibration/"
        "pi05_full_w4/hessian_w4.npz.json",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = json.loads(Path(args.base_plan).expanduser().resolve().read_text(encoding="utf-8"))
    sidecar = json.loads(Path(args.hessian_sidecar).expanduser().resolve().read_text(encoding="utf-8"))
    if len(sidecar.get("layer_names") or []) != 180:
        raise SystemExit("pi05 full-W4 Hessian sidecar does not cover 180 layers")
    packed = int(sidecar["packed_weight_bytes"])
    scale_rows = sidecar.get("layers") or []
    scales_total = 0
    for row in scale_rows:
        name = row["name"]
        if name not in base["layers"]:
            raise SystemExit(f"Hessian layer missing from the base plan: {name}")
    # per-output-channel scales: out x 2 bytes per layer
    import numpy as np
    hessian_npz = str(Path(args.hessian_sidecar).with_suffix(""))
    with np.load(hessian_npz, allow_pickle=False) as data:
        for index, name in enumerate(sidecar["layer_names"]):
            scales_total += int(np.asarray(data[f"scales_{index:04d}"]).nbytes)
    all_w4 = packed + scales_total

    plan = copy.deepcopy(base)
    for name, entry in plan["layers"].items():
        entry.update({"bits": 4, "group": 64, "skip": False, "reason": "table1_max_all_w4"})
    plan["schema_version"] = max(int(plan.get("schema_version", 1)), 5)
    plan["meta"] = {
        "kind": "table1_max_all_w4_profile",
        "model_adapter": "pi05",
        "activation_mode": "dynamic_a8",
        "flow_steps": 4,
        "budget_anchor": "deployed_table1_scope_all_w4",
        "fixed_bytes": FIXED_BYTES,
        "target_compression": float(FP16_TOTAL_BYTES / all_w4),
        "target_compression_scope": "candidate",
        "runtime_selector": False,
        "runtime_correction": False,
        "uses_cka": False,
        "uses_cs": False,
        "uses_task_ids_for_stratified_statistics": True,
        "uses_task_labels_for_routing_or_task_specific_mask": False,
        "uses_success_labels": False,
        "uses_gradients": False,
        "uses_extra_training_data": False,
        "selection_rule": "all 180 adapter-bound layers at W4/group-64 (0 FP16 protections)",
        "base_plan_sha256": sha256_file(args.base_plan),
        "hessian_sidecar_sha256": sha256_file(args.hessian_sidecar),
    }
    plan.update(
        {
            "fp16_total_bytes": float(FP16_TOTAL_BYTES),
            "all_w4_total_bytes": int(all_w4),
            "total_bytes": int(all_w4),
            "fixed_bytes": FIXED_BYTES,
            "table1_total_static_bytes": int(all_w4),
            "table1_total_static_compression": float(FP16_TOTAL_BYTES / all_w4),
            "achieved_candidate_compression": float(FP16_TOTAL_BYTES / all_w4),
            "quantized_w4_layers": len(plan["layers"]),
            "retained_fp16_layers": 0,
            "protected_layers": [],
        }
    )
    validation = validate_quant_plan(plan, model="pi05", source="pi05 table1 max plan")
    if validation["quantized_w4_layers"] != 180:
        raise SystemExit(f"pi05 max plan validation mismatch: {validation}")

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if out.exists():
        if out.read_text(encoding="utf-8") != payload:
            raise SystemExit(f"pi05 max plan changed on re-materialization: {out}")
    else:
        out.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "sha256": sha256_file(out),
        "all_w4_total_bytes": all_w4,
        "static_compression": round(FP16_TOTAL_BYTES / all_w4, 4),
        "w4_layers": 180,
        "fp16_layers": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
