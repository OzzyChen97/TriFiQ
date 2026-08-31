#!/usr/bin/env python3
"""Materialize an inventory-exact GR00T v3 Static-A8 table subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from quantvla_cross_model_protocol import (
    PROTOCOL_SHA256,
    sha256_file,
    validate_quant_plan,
)
from quantvla_outputimpact import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    parent = Path(args.parent).expanduser().resolve()
    parent_meta_path = Path(str(parent) + ".meta.json")
    plan_path = Path(args.plan).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    output_meta_path = Path(str(output) + ".meta.json")
    for path in (parent, parent_meta_path, plan_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    parent_meta = json.loads(parent_meta_path.read_text(encoding="utf-8"))
    supported_parent_protocols = {
        "f43aa056b5633b555acfecc48af2f2b5eec81ca7e2cf83c8b86f12fde91498e2",
        PROTOCOL_SHA256,
    }
    if (
        parent_meta.get("schema_version") != 3
        or parent_meta.get("kind") != "v3_per_flow_step_a8"
        or parent_meta.get("protocol_sha256") not in supported_parent_protocols
    ):
        raise ValueError("unsupported parent Static-A8 artifact")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selection = validate_quant_plan(plan, model="gr00t", source=str(plan_path))
    active = {
        name
        for name, row in plan["layers"].items()
        if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
    }
    if len(active) != int(selection["quantized_w4_layers"]):
        raise ValueError("plan W4 inventory attestation mismatch")
    parent_rows = parent_meta.get("table_rows") or {}
    if not active.issubset(parent_rows):
        raise ValueError("plan contains a W4 layer absent from parent A8 tables")
    selected_names = sorted(active)
    parent_hash = sha256_file(parent)
    plan_hash = sha256_file(plan_path)

    if output.exists() or output_meta_path.exists():
        if not (output.is_file() and output_meta_path.is_file()):
            raise FileExistsError("incomplete existing A8 subset")
        existing = json.loads(output_meta_path.read_text(encoding="utf-8"))
        checks = {
            "hash": existing.get("npz_sha256") == sha256_file(output),
            "parent": existing.get("parent_a8_sha256") == parent_hash,
            "plan": existing.get("plan_sha256") == plan_hash,
            "inventory": sorted((existing.get("table_rows") or {}).keys())
            == selected_names,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise FileExistsError(f"existing A8 subset provenance drift: {failed}")
        print(json.dumps({"reused": str(output), "layers": len(active)}, indent=2))
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
            for name in selected_names:
                source_name = f"{name}.npy"
                info = source.getinfo(source_name)
                with source.open(info, "r") as reader, target.open(
                    zipfile.ZipInfo(source_name), "w", force_zip64=True
                ) as writer:
                    shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    table_rows = {name: int(parent_rows[name]) for name in selected_names}
    metadata = {
        **parent_meta,
        "schema_version": 3,
        "kind": "v3_per_flow_step_a8",
        "protocol_sha256": PROTOCOL_SHA256,
        "plan_sha256": plan_hash,
        "wrapped_layers": len(active),
        "candidate_inventory_sha256": hashlib.sha256(
            ("\n".join(selected_names) + "\n").encode("utf-8")
        ).hexdigest(),
        "table_rows": table_rows,
        "npz_sha256": sha256_file(output),
        "parent_a8_path": str(parent),
        "parent_a8_sha256": parent_hash,
        "parent_a8_meta_sha256": sha256_file(parent_meta_path),
        "parent_protocol_sha256": parent_meta.get("protocol_sha256"),
        "deployment_plan_path": str(plan_path),
        "inventory_subset": True,
        "recalibrated": False,
    }
    atomic_json(output_meta_path, metadata)
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": metadata["npz_sha256"],
                "layers": len(active),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
