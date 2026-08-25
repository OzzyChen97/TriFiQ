#!/usr/bin/env python3
"""Re-attest the frozen π0.5 GDSQ mask under paper-faithful DuQuant calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "runs/pi05_gdsq_final/final/pi05_gdsq_vla_final.plan.json"
DEFAULT_ROOT = REPO_ROOT / "runs/pi05_quantvla_paper"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", default=str(DEFAULT_BASE))
    parser.add_argument(
        "--pack-manifest",
        default=str(DEFAULT_ROOT / "packs/pi05_robocasa_block64_permute_w4a8_ls015/manifest.json"),
    )
    parser.add_argument(
        "--buffer",
        default=str(DEFAULT_ROOT / "calibration/robocasa_real_observations_128.npz"),
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_ROOT / "plans/pi05_gdsq_vla_final_paper.plan.json")
    )
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
    buffer_path = Path(args.buffer).resolve()
    output = Path(args.out).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    layers = base.get("layers") or {}
    selected = {
        name for name, row in layers.items()
        if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
    }
    if len(layers) != 180 or len(selected) != 69:
        raise ValueError("base GDSQ plan is not the frozen 69-W4/111-FP16 mask")
    if not buffer_path.is_file():
        raise FileNotFoundError(buffer_path)
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
            "kind": "gdsq_vla_pi05_faithful_final_paper_calibration_frozen",
            "parent_frozen_plan_path": str(base_path),
            "parent_frozen_plan_sha256": sha256_file(base_path),
            "paper_faithful_duquant": True,
            "enable_permute": True,
            "pack_manifest_path": str(pack_path),
            "pack_manifest_sha256": sha256_file(pack_path),
            "calibration_buffer_path": str(buffer_path),
            "calibration_buffer_sha256": sha256_file(buffer_path),
            "calibration_buffer_kind": "real-on-policy-robocasa-pi05",
            "calibration_batches": 32,
            "calibration_steps": 128,
            "max_calibration_trials_per_task": 5,
            "atm_mode": "per_head_fold_q_weight",
            "ohb_mode": "per_layer_post_projection",
            "quantized_layers": 69,
            "retained_fp16_layers": 111,
            "frozen_before_corrected_table1_test": True,
        },
        "layers": layers,
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"refusing to replace a different paper GDSQ plan: {output}")
        print(f"paper GDSQ plan verified unchanged: {output}")
        return
    atomic_write(output, payload)
    print(f"paper GDSQ plan created: {output}")


if __name__ == "__main__":
    main()
