#!/usr/bin/env python3
"""Materialize an inventory-exact Hessian-W4 deployment subset.

The offline full-context scorer keeps one all-W4 model alive and uses an exact
FP16 bypass to measure masks.  Deployment instead leaves protected layers as
native FP16 modules, so its Hessian artifact must contain only the W4 layers.
This tool copies the frozen parent NPZ entries without requantizing them and
records the parent/plan lineage in a new immutable sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from quantvla_cross_model_protocol import sha256_file, validate_quant_plan
from quantvla_full_context import (
    protocol_attestation as full_context_protocol_attestation,
)


ARRAY_PREFIXES = ("packed", "scales", "clipping", "error")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--model", choices=("gr00t", "pi05"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    parent = Path(args.parent).expanduser().resolve()
    parent_sidecar = Path(str(parent) + ".json")
    plan = Path(args.plan).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    output_sidecar = Path(str(output) + ".json")
    for path in (parent, parent_sidecar, plan):
        if not path.is_file():
            raise FileNotFoundError(path)

    parent_meta = json.loads(parent_sidecar.read_text(encoding="utf-8"))
    parent_hash = sha256_file(parent)
    if parent_meta.get("npz_sha256") != parent_hash:
        raise ValueError("parent Hessian hash drift")
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    plan_meta = plan_payload.get("meta") or {}
    if plan_meta.get("protocol_id") == "dypac-vla-libero-v1":
        protocol_path = Path(__file__).resolve().parents[1] / "quantvla_libero_dypac_protocol.json"
        protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        if (
            plan_meta.get("protocol_sha256") != protocol_hash
            or plan_meta.get("model") != args.model
            or int(plan_meta.get("flow_steps", -1)) != 10
            or plan_meta.get("activation_mode") != "dynamic_a8"
        ):
            raise ValueError("LIBERO DyPAC deployment-plan attestation mismatch")
        selection = {
            "quantized_w4_layers": int(plan_payload.get("quantized_w4_layers", -1))
        }
        if int(plan_payload.get("total_bytes", 1 << 62)) > int(
            plan_payload.get("budget_bytes", -1)
        ):
            raise ValueError("LIBERO DyPAC deployment plan exceeds its byte budget")
    else:
        selection = validate_quant_plan(plan_payload, model=args.model, source=str(plan))
    active = {
        name
        for name, row in plan_payload["layers"].items()
        if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
    }
    if len(active) != int(selection["quantized_w4_layers"]):
        raise ValueError("plan W4 inventory attestation mismatch")
    parent_names = list(parent_meta.get("layer_names") or [])
    if not active.issubset(parent_names):
        raise ValueError("plan contains a W4 layer absent from the parent Hessian")
    selected_names = [name for name in parent_names if name in active]
    selected_indices = [parent_names.index(name) for name in selected_names]

    plan_hash = sha256_file(plan)
    if output.exists() or output_sidecar.exists():
        if not (output.is_file() and output_sidecar.is_file()):
            raise FileExistsError("incomplete existing subset artifact")
        existing = json.loads(output_sidecar.read_text(encoding="utf-8"))
        checks = {
            "hash": existing.get("npz_sha256") == sha256_file(output),
            "parent": existing.get("parent_hessian_sha256") == parent_hash,
            "plan": existing.get("deployment_plan_sha256") == plan_hash,
            "inventory": existing.get("layer_names") == selected_names,
        }
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            raise FileExistsError(f"existing subset provenance drift: {failed}")
        print(json.dumps({"reused": str(output), "layers": len(selected_names)}, indent=2))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".npz.tmp", dir=output.parent
    )
    os.close(fd)
    temporary = Path(raw)
    try:
        with zipfile.ZipFile(parent, "r", allowZip64=True) as source, zipfile.ZipFile(
            temporary, "w", allowZip64=True
        ) as target:
            target.writestr(
                "layer_names.npy",
                npy_bytes(np.asarray(selected_names)),
                compress_type=zipfile.ZIP_STORED,
            )
            for new_index, old_index in enumerate(selected_indices):
                for prefix in ARRAY_PREFIXES:
                    source_name = f"{prefix}_{old_index:04d}.npy"
                    target_name = f"{prefix}_{new_index:04d}.npy"
                    source_info = source.getinfo(source_name)
                    with source.open(source_info, "r") as reader, target.open(
                        zipfile.ZipInfo(target_name), "w", force_zip64=True
                    ) as writer:
                        shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    layer_rows = {
        row["name"]: row for row in (parent_meta.get("layers") or [])
    }
    selected_rows = [layer_rows[name] for name in selected_names]
    metadata = {
        **parent_meta,
        "kind": "hessian_w4_group64_static_fp16_protection_subset",
        "layer_names": selected_names,
        "layers": selected_rows,
        "packed_weight_bytes": int(sum(int(row["packed_bytes"]) for row in selected_rows)),
        "npz_sha256": sha256_file(output),
        "parent_hessian_path": str(parent),
        "parent_hessian_sha256": parent_hash,
        "deployment_plan_path": str(plan),
        "deployment_plan_sha256": plan_hash,
        "full_context_protocol": full_context_protocol_attestation(),
        "runtime_selector": False,
        "runtime_correction": False,
        "requantized": False,
    }
    atomic_json(output_sidecar, metadata)
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": metadata["npz_sha256"],
                "layers": len(selected_names),
                "packed_weight_bytes": metadata["packed_weight_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
