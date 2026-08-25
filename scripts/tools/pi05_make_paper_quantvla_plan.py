#!/usr/bin/env python3
"""Create the isolated 180-layer paper-faithful π0.5 QuantVLA plan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "runs/pi05_gdsq_port/plans/pi05_quantvla_full_w4a8.plan.json"
DEFAULT_PACK = (
    REPO_ROOT
    / "runs/pi05_quantvla_paper/packs/pi05_robocasa_block64_permute_w4a8_ls015/manifest.json"
)
DEFAULT_OUT = REPO_ROOT / "runs/pi05_quantvla_paper/plans/pi05_quantvla_paper_w4a8.plan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", default=str(DEFAULT_BASE))
    parser.add_argument("--pack-manifest", default=str(DEFAULT_PACK))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser.parse_args()


def atomic_write(path: Path, payload: dict) -> None:
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


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_plan).resolve()
    pack_path = Path(args.pack_manifest).resolve()
    output = Path(args.out).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    layers = base.get("layers") or {}
    if len(layers) != 180 or any(
        bool(row.get("skip")) or int(row.get("bits", 0)) != 4 for row in layers.values()
    ):
        raise ValueError("base plan is not the 180-layer W4 reference")
    expected_pack = {
        "complete": True,
        "wrapped_layer_count": 180,
        "block_in": 64,
        "block_out": 64,
        "lambda_smooth": 0.15,
        "enable_permute": True,
    }
    mismatches = {
        key: (pack.get(key), value)
        for key, value in expected_pack.items()
        if pack.get(key) != value
    }
    if mismatches:
        raise ValueError(f"paper pack manifest mismatch: {mismatches}")
    payload = {
        "schema_version": 1,
        "meta": {
            **(base.get("meta") or {}),
            "kind": "quantvla_paper_w4a8",
            "enable_permute": True,
            "paper_faithful": True,
            "paper_scope": "all LLM Linear plus action-expert DiT MLP Linear",
            "atm_mode": "per_head_fold_q_weight",
            "ohb_mode": "per_layer_post_projection",
            "calibration_batches": 32,
            "calibration_steps": 128,
            "max_calibration_trials_per_task": 5,
        },
        "layers": {
            name: {**row, "reason": "quantvla_paper_w4a8"}
            for name, row in layers.items()
        },
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"refusing to replace a different paper plan: {output}")
        print(f"paper plan verified unchanged: {output}")
        return
    atomic_write(output, payload)
    print(f"paper plan created: {output}")


if __name__ == "__main__":
    main()
