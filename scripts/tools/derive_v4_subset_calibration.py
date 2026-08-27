#!/usr/bin/env python3
"""Derive a mixed-plan FP16/A8/Hessian stage from a verified full-W4 stage.

FP16 activations and per-layer Hessian W4 solutions are plan-independent.
This tool performs only inventory selection and provenance re-attestation; the
complete quantized-network capture and ErrorFold fit must still be rerun by
``calibrate_errorfold_v3.py`` for the derived plan.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from build_v3_a8_artifact import build as build_a8
from quantvla_cross_model_protocol import (
    PROTOCOL,
    PROTOCOL_SHA256,
    protocol_attestation,
    sha256_file,
    validate_quant_plan,
)
from quantvla_v3_capture import save_npz


def _quantized_names(plan: dict[str, Any]) -> list[str]:
    return [
        str(name)
        for name, row in (plan.get("layers") or {}).items()
        if not bool((row if isinstance(row, dict) else {}).get("skip", False))
    ]


def build(
    *,
    model: str,
    full_dir: str | Path,
    plan_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_dir = Path(full_dir).expanduser().resolve()
    plan_file = Path(plan_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    selection = validate_quant_plan(plan, model=model, source=str(plan_file))
    requested = set(_quantized_names(plan))
    if len(requested) != selection["quantized_w4_layers"]:
        raise ValueError("derived W4 layer count drift")
    source_capture = source_dir / "fp16_capture.npz"
    source_hessian = source_dir / "hessian_w4.npz"
    source_attention = source_dir / "fp16_attention.npz"
    for path in (
        source_capture,
        source_hessian,
        Path(str(source_hessian) + ".json"),
        source_attention,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    capture_out = output / "fp16_capture.npz"
    attention_out = output / "fp16_attention.npz"
    hessian_out = output / "hessian_w4.npz"
    a8_out = output / "a8_scales.npz"
    expected_outputs = (
        capture_out,
        attention_out,
        hessian_out,
        Path(str(hessian_out) + ".json"),
        a8_out,
        Path(str(a8_out) + (".meta.json" if model == "gr00t" else ".json")),
        output / "calibration_manifest.json",
    )
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"derived stage output is not empty: {output}")

    with np.load(source_capture, allow_pickle=False) as source:
        source_names = [str(value) for value in source["layer_names"].tolist()]
        selected = [(index, name) for index, name in enumerate(source_names) if name in requested]
        if {name for _, name in selected} != requested:
            raise ValueError("derived plan is not a subset of the full FP16 capture")
        arrays: dict[str, np.ndarray] = {
            "layer_names": np.asarray([name for _, name in selected]),
            "calibration_buffer_sha256": np.asarray(source["calibration_buffer_sha256"]),
            "checkpoint_sha256": np.asarray(source["checkpoint_sha256"]),
            "plan_sha256": np.asarray(sha256_file(plan_file)),
            "capture_sampling_strategy": np.asarray(source["capture_sampling_strategy"]),
            "capture_rows_per_observation": np.asarray(source["capture_rows_per_observation"]),
            "capture_rank2_rows_per_call": np.asarray(source["capture_rank2_rows_per_call"]),
            "capture_calls": np.asarray([source["capture_calls"][index] for index, _ in selected]),
            "capture_input_rows": np.asarray(
                [source["capture_input_rows"][index] for index, _ in selected]
            ),
            "capture_output_rows": np.asarray(
                [source["capture_output_rows"][index] for index, _ in selected]
            ),
            "derived_from_full_capture_sha256": np.asarray(sha256_file(source_capture)),
        }
        for output_index, (source_index, _name) in enumerate(selected):
            for prefix in ("inputs", "outputs", "weight"):
                arrays[f"{prefix}_{output_index:04d}"] = np.asarray(
                    source[f"{prefix}_{source_index:04d}"]
                )
            step_key = f"step_inputs_{source_index:04d}"
            if step_key in source:
                arrays[f"step_inputs_{output_index:04d}"] = np.asarray(source[step_key])
    save_npz(capture_out, arrays)
    shutil.copy2(source_attention, attention_out)
    build_a8(capture_out, a8_out, model)
    if model == "pi05":
        # OpenPI attests two distinct inventories: ``layer_names`` is the
        # wrapped mixed-plan subset, while ``candidate_inventory_sha256`` is
        # the complete adapter target inventory before applying that plan.
        # The generic A8 builder only sees the subset capture, so inherit the
        # latter from the verified full-W4 artifact.
        source_a8_sidecar = Path(str(source_dir / "a8_scales.npz") + ".json")
        derived_a8_sidecar = Path(str(a8_out) + ".json")
        source_a8 = json.loads(source_a8_sidecar.read_text(encoding="utf-8"))
        derived_a8 = json.loads(derived_a8_sidecar.read_text(encoding="utf-8"))
        candidate_hash = (source_a8.get("metadata") or {}).get(
            "candidate_inventory_sha256"
        )
        if not candidate_hash:
            raise ValueError("full-W4 OpenPI A8 artifact lacks candidate inventory hash")
        derived_a8.setdefault("metadata", {})[
            "candidate_inventory_sha256"
        ] = candidate_hash
        derived_a8["metadata"][
            "candidate_inventory_derived_from_full_a8_sha256"
        ] = sha256_file(source_a8_sidecar)
        temporary_sidecar = Path(str(derived_a8_sidecar) + f".tmp.{os.getpid()}")
        temporary_sidecar.write_text(
            json.dumps(derived_a8, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_sidecar.replace(derived_a8_sidecar)

    with np.load(source_hessian, allow_pickle=False) as source:
        hessian_names = [str(value) for value in source["layer_names"].tolist()]
        source_indices = {name: index for index, name in enumerate(hessian_names)}
        selected_names = [name for _, name in selected]
        hessian_arrays: dict[str, np.ndarray] = {
            "layer_names": np.asarray(selected_names)
        }
        for output_index, name in enumerate(selected_names):
            source_index = source_indices[name]
            for prefix in ("packed", "scales", "clipping", "error"):
                hessian_arrays[f"{prefix}_{output_index:04d}"] = np.asarray(
                    source[f"{prefix}_{source_index:04d}"]
                )
    temporary = Path(str(hessian_out) + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **hessian_arrays)
    temporary.replace(hessian_out)
    source_sidecar = json.loads(
        Path(str(source_hessian) + ".json").read_text(encoding="utf-8")
    )
    source_summaries = {row["name"]: row for row in source_sidecar["layers"]}
    hessian_sidecar = {
        **source_sidecar,
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "capture_path": str(capture_out),
        "capture_sha256": sha256_file(capture_out),
        "npz_sha256": sha256_file(hessian_out),
        "layer_names": selected_names,
        "layers": [source_summaries[name] for name in selected_names],
        "packed_weight_bytes": int(
            sum(np.asarray(hessian_arrays[f"packed_{index:04d}"]).nbytes for index in range(len(selected_names)))
        ),
        "derived_from_full_hessian_sha256": sha256_file(source_hessian),
    }
    Path(str(hessian_out) + ".json").write_text(
        json.dumps(hessian_sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with np.load(capture_out, allow_pickle=False) as capture:
        checkpoint_sha256 = str(np.asarray(capture["checkpoint_sha256"]).item())
        calibration_buffer_sha256 = str(
            np.asarray(capture["calibration_buffer_sha256"]).item()
        )
    manifest = {
        "schema_version": 4,
        "kind": "derived_mixed_plan_errorfold_calibration_stage",
        "cross_model_protocol": protocol_attestation(),
        "protocol_sha256": PROTOCOL_SHA256,
        "model_adapter": model,
        "checkpoint_sha256": checkpoint_sha256,
        "plan": str(plan_file),
        "plan_sha256": sha256_file(plan_file),
        "calibration_buffer": {
            "sha256": calibration_buffer_sha256,
            "rows": int(PROTOCOL["hessian_w4a8"]["calibration_observations"]),
        },
        "quantized_w4_layers": len(selected_names),
        "retained_fp16_target_layers": selection["retained_fp16_target_layers"],
        "gradient_updates": False,
        "fp16_weight_updates": False,
        "success_labels_used": False,
        "derived_from_full_capture_sha256": sha256_file(source_capture),
        "derived_from_full_hessian_sha256": sha256_file(source_hessian),
        "artifacts": {
            "fp16": {"path": str(capture_out), "sha256": sha256_file(capture_out)},
            "fp16_attention": {
                "path": str(attention_out), "sha256": sha256_file(attention_out)
            },
            "hessian": {"path": str(hessian_out), "sha256": sha256_file(hessian_out)},
            "a8": {"path": str(a8_out), "sha256": sha256_file(a8_out)},
        },
    }
    (output / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "model": model,
        "plan": str(plan_file),
        "quantized_w4_layers": len(selected),
        "retained_fp16_target_layers": selection["retained_fp16_target_layers"],
        "source_full_capture_sha256": sha256_file(source_capture),
        "derived_capture_sha256": sha256_file(capture_out),
        "derived_hessian_sha256": sha256_file(hessian_out),
        "protocol_sha256": PROTOCOL_SHA256,
    }


def build_deployment_subset(
    *,
    model: str,
    full_dir: str | Path,
    plan_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Subset only packed Hessian W4 for calibration-free dynamic A8 deployment.

    Dynamic A8 does not consume a frozen activation table.  Copying multi-GB
    FP16 captures merely to deploy a mixed W4/FP16 profile is unnecessary;
    this path preserves the original layer order and re-attests the exact
    packed-code subset selected by the static compression plan.
    """
    source_dir = Path(full_dir).expanduser().resolve()
    plan_file = Path(plan_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    selection = validate_quant_plan(plan, model=model, source=str(plan_file))
    requested = set(_quantized_names(plan))
    source_hessian = source_dir / "hessian_w4.npz"
    source_sidecar_path = Path(str(source_hessian) + ".json")
    for path in (source_hessian, source_sidecar_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    hessian_out = output / "hessian_w4.npz"
    hessian_sidecar_out = Path(str(hessian_out) + ".json")
    if hessian_out.exists() or hessian_sidecar_out.exists():
        raise FileExistsError(f"derived deployment output is not empty: {output}")

    with np.load(source_hessian, allow_pickle=False) as source:
        source_names = [str(value) for value in source["layer_names"].tolist()]
        selected_names = [name for name in source_names if name in requested]
        if set(selected_names) != requested:
            raise ValueError("deployment plan is not a subset of full Hessian W4")
        arrays: dict[str, np.ndarray] = {"layer_names": np.asarray(selected_names)}
        source_indices = {name: index for index, name in enumerate(source_names)}
        for output_index, name in enumerate(selected_names):
            source_index = source_indices[name]
            for prefix in ("packed", "scales", "clipping", "error"):
                arrays[f"{prefix}_{output_index:04d}"] = np.asarray(
                    source[f"{prefix}_{source_index:04d}"]
                )
    temporary = Path(str(hessian_out) + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(hessian_out)

    source_sidecar = json.loads(source_sidecar_path.read_text(encoding="utf-8"))
    summaries = {row["name"]: row for row in source_sidecar["layers"]}
    sidecar = {
        **source_sidecar,
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "npz_sha256": sha256_file(hessian_out),
        "layer_names": selected_names,
        "layers": [summaries[name] for name in selected_names],
        "packed_weight_bytes": int(
            sum(
                arrays[f"packed_{index:04d}"].nbytes
                for index in range(len(selected_names))
            )
        ),
        "plan_path": str(plan_file),
        "plan_sha256": sha256_file(plan_file),
        "derived_from_full_hessian_sha256": sha256_file(source_hessian),
        "deployment_only": True,
        "activation_policy": "online_dynamic_per_forward_per_channel_amax",
    }
    hessian_sidecar_out.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "model": model,
        "plan": str(plan_file),
        "quantized_w4_layers": len(selected_names),
        "retained_fp16_target_layers": selection["retained_fp16_target_layers"],
        "derived_hessian_sha256": sha256_file(hessian_out),
        "source_full_hessian_sha256": sha256_file(source_hessian),
        "deployment_only": True,
        "protocol_sha256": PROTOCOL_SHA256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=PROTOCOL["models"])
    parser.add_argument("--full-dir", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--deployment-only",
        action="store_true",
        help="Write only the packed W4 subset for dynamic-A8 deployment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            (build_deployment_subset if args.deployment_only else build)(
                model=args.model,
                full_dir=args.full_dir,
                plan_path=args.plan,
                output_dir=args.out_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
