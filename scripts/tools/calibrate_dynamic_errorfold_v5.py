#!/usr/bin/env python3
"""Refit ErrorFold against a complete DyRange-A8 + Hessian-W4 network.

The expensive FP16 input/weight capture and Hessian-W4 codes are immutable and
reused from a validated v3/v4 calibration directory.  Quantized outputs are
recollected with the shared dynamic activation formula, so a static-A8
ErrorFold artifact can never be silently reused for DyRange-A8 deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tools"))

from build_errorfold_artifact import build as build_errorfold  # noqa: E402
from calibrate_errorfold_v3 import (  # noqa: E402
    DEFAULTS,
    _capture_gr00t,
    _capture_pi05,
    _checkpoint_sha256,
    _load_attention,
    _quantized_plan_names,
    _save_attention,
)
from quantvla_cross_model_protocol import (  # noqa: E402
    PROTOCOL,
    PROTOCOL_SHA256,
    protocol_artifact,
    protocol_attestation,
    require_protocol_attestation,
    sha256_file,
    validate_quant_plan,
)
from quantvla_dynamic_a8_protocol import (  # noqa: E402
    PROTOCOL_SHA256 as DYNAMIC_PROTOCOL_SHA256,
    protocol_attestation as dynamic_protocol_attestation,
)
from quantvla_model_adapters import load_model_records  # noqa: E402
from quantvla_v3_capture import merge_errorfold_with_attention, save_npz  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=PROTOCOL["models"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--plan", default=None)
    parser.add_argument(
        "--buffer", default=str(protocol_artifact("calibration_buffer", verify=False))
    )
    parser.add_argument("--base-calibration-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pack-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _validate_reused_stage(
    *,
    base_dir: Path,
    checkpoint_sha256: str,
    plan_sha256: str,
    buffer_sha256: str,
    expected_names: list[str],
) -> dict[str, Path]:
    paths = {
        "fp16": base_dir / "fp16_capture.npz",
        "fp16_attention": base_dir / "fp16_attention.npz",
        "hessian": base_dir / "hessian_w4.npz",
        "hessian_meta": base_dir / "hessian_w4.npz.json",
        "manifest": base_dir / "calibration_manifest.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    base_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    require_protocol_attestation(base_manifest, source=str(paths["manifest"]))
    expected_manifest = {
        "checkpoint_sha256": checkpoint_sha256,
        "plan_sha256": plan_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
    }
    mismatches = {
        key: (base_manifest.get(key), value)
        for key, value in expected_manifest.items()
        if base_manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"reused calibration manifest drift: {mismatches}")
    with np.load(paths["fp16"], allow_pickle=False) as archive:
        checks = {
            "checkpoint_sha256": checkpoint_sha256,
            "plan_sha256": plan_sha256,
            "calibration_buffer_sha256": buffer_sha256,
        }
        for key, value in checks.items():
            if str(np.asarray(archive[key]).item()) != value:
                raise ValueError(f"reused FP16 capture {key} drift")
        names = [str(value) for value in archive["layer_names"].tolist()]
        if set(names) != set(expected_names) or len(names) != len(expected_names):
            raise ValueError("reused FP16 capture layer inventory drift")
    hessian_meta = json.loads(paths["hessian_meta"].read_text(encoding="utf-8"))
    if hessian_meta.get("capture_sha256") != sha256_file(paths["fp16"]):
        raise ValueError("reused Hessian/FP16 lineage drift")
    if hessian_meta.get("calibration_buffer_sha256") != buffer_sha256:
        raise ValueError("reused Hessian calibration-buffer drift")
    if hessian_meta.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("reused Hessian base-protocol drift")
    if [str(value) for value in hessian_meta.get("layer_names") or []] != names:
        raise ValueError("reused Hessian layer inventory/order drift")
    return paths


def _identity_pack(
    path: Path,
    *,
    model: str,
    checkpoint_sha256: str,
    plan_sha256: str,
    quantized_layers: int,
    retained_fp16_target_layers: int,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 5,
        "kind": "hessian_w4_identity_pack",
        "model_adapter": model,
        "checkpoint_sha256": checkpoint_sha256,
        "plan_sha256": plan_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "dynamic_a8_protocol_sha256": DYNAMIC_PROTOCOL_SHA256,
        "quantized_w4_layers": quantized_layers,
        "retained_fp16_target_layers": retained_fp16_target_layers,
        "packed_weights_source": "reused_hessian_w4.npz",
        "permutation": False,
        "row_rotation": "identity",
        "activation_mode": "dynamic_a8",
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    target = path / "manifest.json"
    if target.exists() and target.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"dynamic identity-pack provenance drift: {target}")
    target.write_text(rendered, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    model = args.model
    checkpoint = Path(args.checkpoint or DEFAULTS[model]["checkpoint"]).resolve()
    plan = Path(args.plan or DEFAULTS[model]["plan"]).resolve()
    buffer = Path(args.buffer).resolve()
    base_dir = Path(args.base_calibration_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = Path(args.pack_dir).resolve() if args.pack_dir else out_dir / "identity_pack"

    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    selection = validate_quant_plan(plan_payload, model=model, source=str(plan))
    names = _quantized_plan_names(plan_payload)
    if len(names) != selection["quantized_w4_layers"]:
        raise ValueError("dynamic ErrorFold plan/inventory mismatch")
    checkpoint_hash = _checkpoint_sha256(checkpoint, model)
    plan_hash = sha256_file(plan)
    buffer_hash = sha256_file(buffer)
    if buffer_hash != PROTOCOL["data"]["calibration_buffer"]["sha256"]:
        raise ValueError("dynamic ErrorFold requires the frozen 256-observation buffer")
    reused = _validate_reused_stage(
        base_dir=base_dir,
        checkpoint_sha256=checkpoint_hash,
        plan_sha256=plan_hash,
        buffer_sha256=buffer_hash,
        expected_names=names,
    )
    records, buffer_provenance = load_model_records(
        buffer, int(PROTOCOL["hessian_w4a8"]["calibration_observations"]), model=model
    )
    _identity_pack(
        pack_dir,
        model=model,
        checkpoint_sha256=checkpoint_hash,
        plan_sha256=plan_hash,
        quantized_layers=len(names),
        retained_fp16_target_layers=selection["retained_fp16_target_layers"],
    )

    paths = {
        "quant": out_dir / "quant_capture.npz",
        "quant_attention": out_dir / "quant_attention.npz",
        "paired": out_dir / "paired_errorfold_capture.npz",
        "raw": out_dir / "raw_errorfold.json",
        "manifest": out_dir / "calibration_manifest.json",
    }
    if args.force:
        for path in paths.values():
            path.unlink(missing_ok=True)
        for path in pack_dir.glob("*.npz"):
            path.unlink()
    outputs = [paths[key] for key in ("quant", "quant_attention", "paired", "raw")]
    if any(path.exists() for path in outputs) and not all(path.exists() for path in outputs):
        raise RuntimeError("incomplete dynamic ErrorFold stage; use --force")

    runtime: dict[str, Any] = {"crash_resume_reused": True}
    if not all(path.exists() for path in outputs):
        capture = _capture_gr00t if model == "gr00t" else _capture_pi05
        quant_arrays, quant_attention, runtime = capture(
            checkpoint=checkpoint,
            plan=plan,
            pack_dir=pack_dir,
            a8=base_dir / "a8_scales.npz",
            hessian=reused["hessian"],
            records=records,
            checkpoint_hash=checkpoint_hash,
            buffer_hash=buffer_hash,
            plan_hash=plan_hash,
            device=args.device,
            batch_size=args.batch_size,
            quantized=True,
            activation_mode="dynamic_a8",
        )
        save_npz(paths["quant"], quant_arrays)
        _save_attention(paths["quant_attention"], quant_attention)
        merge_errorfold_with_attention(
            reused["fp16"],
            paths["quant"],
            _load_attention(reused["fp16_attention"]),
            _load_attention(paths["quant_attention"]),
            paths["paired"],
        )
        build_errorfold(paths["paired"], paths["raw"], model)

    # The shared v3 builder deliberately knows only the base ErrorFold schema.
    # Bind its output to DyRange-A8 here so downstream grid materialization and
    # deployment can fail closed on an activation-policy mismatch.
    raw_payload = json.loads(paths["raw"].read_text(encoding="utf-8"))
    raw_payload["dynamic_a8_protocol"] = dynamic_protocol_attestation()
    raw_payload.setdefault("meta", {})["activation_mode"] = "dynamic_a8"
    paths["raw"].write_text(
        json.dumps(raw_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 5,
        "kind": "dynamic_errorfold_v5_calibration",
        "cross_model_protocol": protocol_attestation(),
        "dynamic_a8_protocol": dynamic_protocol_attestation(),
        "model_adapter": model,
        "activation_mode": "dynamic_a8",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "plan": str(plan),
        "plan_sha256": plan_hash,
        "calibration_buffer": buffer_provenance,
        "reused_fp16_capture": {
            "path": str(reused["fp16"]),
            "sha256": sha256_file(reused["fp16"]),
        },
        "reused_hessian_w4": {
            "path": str(reused["hessian"]),
            "sha256": sha256_file(reused["hessian"]),
        },
        "fit_pair": "original_fp16_vs_complete_hessian_w4_dyrange_a8_network",
        "gradient_updates": False,
        "fp16_weight_updates": False,
        "success_labels_used": False,
        "runtime": runtime,
        "artifacts": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
            if key != "manifest" and path.is_file()
        },
        "source_sha256": {
            "calibrator": sha256_file(Path(__file__)),
            "shared_calibrator": sha256_file(
                REPO / "scripts/tools/calibrate_errorfold_v3.py"
            ),
            "capture": sha256_file(REPO / "scripts/tools/quantvla_v3_capture.py"),
            "errorfold_builder": sha256_file(
                REPO / "scripts/tools/build_errorfold_artifact.py"
            ),
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
