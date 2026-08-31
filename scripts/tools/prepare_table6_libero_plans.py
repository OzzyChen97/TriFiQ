#!/usr/bin/env python3
"""Materialize checkpoint-pinned uniform plans for the Table 6 LIBERO run.

This tool deliberately does not create the DyPAC/FCP plan.  That plan is
allowed only after its model-specific LIBERO calibration artifact exists.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/table6_libero_v1/artifacts"
PI_TEMPLATE = (
    ROOT
    / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_quantvla_uniform_w4a8_d4.plan.json"
)
PI_CHECKPOINT = ROOT / "code/pi05/checkpoints/pi05_libero_pytorch/model.safetensors"
SUITES = ("goal", "spatial", "object", "long")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def pi05_plans() -> None:
    template = json.loads(PI_TEMPLATE.read_text(encoding="utf-8"))
    names = sorted((template.get("layers") or {}).keys())
    if len(names) != 180:
        raise ValueError(f"pi0.5 candidate inventory drift: {len(names)} != 180")
    checkpoint_sha = sha256_file(PI_CHECKPOINT)
    for suite in SUITES:
        for identifier, bits in (("quantvla_w4a8", 4), ("uniform_w6", 6)):
            payload = {
                "schema_version": 1,
                "meta": {
                    "kind": "table6_libero_uniform_control",
                    "model": "pi05",
                    "suite": suite,
                    "configuration": identifier,
                    "checkpoint_sha256": checkpoint_sha,
                    "flow_steps": 8,
                    "act_bits": 8,
                    "block_in": 64,
                    "block_out": 64,
                    "quantized_layers": len(names),
                    "calibration_split": "libero_initial_states_0_to_4",
                    "held_out_initial_states": list(range(10, 20)),
                    "test_rollout_feedback_used": False,
                    "source_inventory": str(PI_TEMPLATE.relative_to(ROOT)),
                    "source_inventory_sha256": sha256_file(PI_TEMPLATE),
                },
                "layers": {
                    name: {
                        "bits": bits,
                        "group": 64,
                        "skip": False,
                        "reason": f"uniform_w{bits}_control",
                    }
                    for name in names
                },
            }
            atomic_json(OUT / "pi05" / suite / f"{identifier}.plan.json", payload)


def gr00t_plans() -> None:
    for suite in SUITES:
        selector = (
            ROOT / "checkpoints/packs/gr00t/gr00t_quant_plan_long_transfer_v14.json"
            if suite == "long"
            else ROOT
            / f"checkpoints/packs/gr00t/gr00t_quant_plan_libero_{suite}_v14_adjudicated.final_plan.json"
        )
        template = json.loads(selector.read_text(encoding="utf-8"))
        names = sorted((template.get("layers") or {}).keys())
        if len(names) != 116:
            raise ValueError(f"GR00T {suite} candidate inventory drift: {len(names)} != 116")
        payload = {
            "schema_version": 1,
            "meta": {
                "kind": "table6_libero_uniform_control",
                "model": "gr00t",
                "suite": suite,
                "configuration": "quantvla_w4a8",
                "flow_steps": 8,
                "act_bits": 8,
                "block_in": 64,
                "block_out": 64,
                "quantized_layers": len(names),
                "calibration_split": "libero_initial_states_0_to_4",
                "held_out_initial_states": list(range(10, 20)),
                "test_rollout_feedback_used": False,
                "source_inventory": str(selector.relative_to(ROOT)),
                "source_inventory_sha256": sha256_file(selector),
            },
            "layers": {
                name: {
                    "bits": 4,
                    "group": 64,
                    "skip": False,
                    "reason": "uniform_w4_control",
                }
                for name in names
            },
        }
        atomic_json(OUT / "gr00t" / suite / "quantvla_w4a8.plan.json", payload)


def main() -> None:
    pi05_plans()
    gr00t_plans()
    print(f"wrote Table 6 uniform plans under {OUT}")


if __name__ == "__main__":
    main()
