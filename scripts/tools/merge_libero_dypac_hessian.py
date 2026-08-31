#!/usr/bin/env python3
"""Merge and attest per-layer LIBERO DyPAC Hessian-W4 records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-dir", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=("pi05", "gr00t"), default="pi05")
    parser.add_argument("--suite")
    args = parser.parse_args()
    output = Path(args.out).expanduser().resolve()
    if output.exists() or Path(str(output) + ".json").exists():
        raise FileExistsError(f"refusing to overwrite Hessian artifact: {output}")
    inventory_path = Path(args.inventory).expanduser().resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    layer_dir = Path(args.layer_dir).expanduser().resolve()
    arrays: dict[str, np.ndarray] = {"layer_names": np.asarray([row["name"] for row in inventory["layers"]])}
    summaries = []
    for index, row in enumerate(inventory["layers"]):
        path = layer_dir / f"layer_{index:04d}.npz"
        with np.load(path, allow_pickle=False) as source:
            if str(source["layer_name"].item()) != row["name"]:
                raise ValueError(f"layer record mismatch at {index}")
            arrays[f"packed_{index:04d}"] = np.asarray(source["packed_u4"])
            arrays[f"scales_{index:04d}"] = np.asarray(source["scales"])
            arrays[f"clipping_{index:04d}"] = np.asarray(source["clipping"])
            arrays[f"error_{index:04d}"] = np.asarray(source["reconstruction_error"])
        summaries.append({
            "name": row["name"],
            "packed_bytes": int(arrays[f"packed_{index:04d}"].nbytes),
            "scale_bytes": int(arrays[f"scales_{index:04d}"].nbytes),
            "mean_reconstruction_error": float(arrays[f"error_{index:04d}"].mean()),
            "mean_clipping": float(arrays[f"clipping_{index:04d}"].mean()),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(output)
    capture = Path(args.capture).expanduser().resolve()
    metadata = {
        "schema_version": 3,
        "kind": "hessian_w4_group64_dypac_libero",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "model": args.model,
        "suite": args.suite,
        "group_size": 64,
        "clipping_ratios": PROTOCOL["quantization"]["clipping_ratios"],
        "calibration_buffer_sha256": str(np.load(capture, allow_pickle=False)["calibration_buffer_sha256"].item()),
        "capture_path": str(capture),
        "capture_sha256": sha256_file(capture),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "npz_sha256": sha256_file(output),
        "layer_names": arrays["layer_names"].tolist(),
        "layers": summaries,
        "packed_weight_bytes": sum(row["packed_bytes"] for row in summaries),
        "scale_bytes": sum(row["scale_bytes"] for row in summaries),
        "gradient_updates": False,
        "fp16_weight_updates": False,
    }
    atomic_json(Path(str(output) + ".json"), metadata)
    print(json.dumps({"out": str(output), "layers": len(summaries), "sha256": metadata["npz_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
