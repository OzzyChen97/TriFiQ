#!/usr/bin/env python3
"""Refresh the pi0.5 round-1 frozen candidate to the current protocol attestation.

The v1 pi0.5 frozen plan was sealed before the full-context protocol core
moved to the v2 statistics; its attestation block (selection/metric/quick
core shas) is stale relative to the pinned files today.  The mask content,
byte accounting and selection provenance are unchanged — only the three
attestation blocks are re-stamped with the current protocol files, and the
deployment Hessian sidecar's plan lineage is refreshed in a copied
artifact pair.  The original v1 artifacts are never modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from quantvla_cross_model_protocol import (
    protocol_attestation as cross_model_attestation,
    sha256_file,
)
from quantvla_dynamic_a8_protocol import (
    protocol_attestation as dynamic_a8_attestation,
)
from quantvla_full_context import protocol_attestation as full_context_attestation
from quantvla_outputimpact import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--hessian", required=True)
    parser.add_argument("--out-plan", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    plan_path = Path(args.plan).expanduser().resolve()
    hessian_path = Path(args.hessian).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    meta = dict(plan.get("meta") or {})
    checks = {
        "frozen": meta.get("frozen") is True,
        "model": meta.get("model_adapter") == "pi05",
        "noise_a": meta.get("selection_noise") == "A",
        "dynamic_a8": meta.get("activation_mode") == "dynamic_a8",
        "no_selector": meta.get("runtime_selector") is False,
        "no_correction": meta.get("runtime_correction") is False,
        "w4_layers": int(plan.get("quantized_w4_layers", 0)) == 172,
        "fp16_layers": int(plan.get("retained_fp16_layers", 0)) == 8,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"candidate plan invariants drifted: {failed}")
    meta["cross_model_protocol"] = cross_model_attestation()
    meta["full_context_protocol"] = full_context_attestation()
    meta["dynamic_a8_protocol"] = dynamic_a8_attestation()
    meta["attestation_refresh"] = {
        "source_plan": str(plan_path),
        "source_plan_sha256": sha256_file(plan_path),
        "mask_content_unchanged": True,
        "only_attestation_blocks_replaced": True,
    }
    plan["meta"] = meta
    out_plan_path = Path(args.out_plan).expanduser().resolve()
    atomic_json(out_plan_path, plan)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_hessian = out_dir / hessian_path.name
    shutil.copyfile(hessian_path, out_hessian)
    sidecar = json.loads(Path(str(hessian_path) + ".json").read_text(encoding="utf-8"))
    sidecar["deployment_plan_path"] = str(out_plan_path)
    sidecar["deployment_plan_sha256"] = sha256_file(out_plan_path)
    sidecar["full_context_protocol"] = full_context_attestation()
    sidecar["attestation_refresh"] = {
        "source_hessian": str(hessian_path),
        "source_hessian_sha256": sha256_file(hessian_path),
        "hessian_bytes_unchanged": True,
    }
    (Path(str(out_hessian) + ".json")).write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "out_plan": str(out_plan_path),
                "out_plan_sha256": sha256_file(out_plan_path),
                "out_hessian": str(out_hessian),
                "out_hessian_sha256": sha256_file(out_hessian),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
