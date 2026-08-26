#!/usr/bin/env python3
"""Build a frozen v3 group-64 Hessian W4 artifact from shared FP16 captures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from quantvla_cross_model_protocol import PROTOCOL, PROTOCOL_SHA256, sha256_file
from quantvla_hessian_w4 import artifact_arrays, hessian_aware_w4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True, choices=PROTOCOL["models"])
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for 64x64 Hessian solves and row-parallel feedback.",
    )
    return parser.parse_args()


def build(
    capture_path: str | Path,
    output_path: str | Path,
    model: str,
    *,
    device: str | torch.device | None = None,
) -> dict:
    capture = Path(capture_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    arrays: dict[str, np.ndarray] = {}
    summaries = []
    with np.load(capture, allow_pickle=False) as source:
        names = [str(value) for value in source["layer_names"].tolist()]
        if str(source["calibration_buffer_sha256"].item()) != PROTOCOL["data"]["calibration_buffer"]["sha256"]:
            raise ValueError("Hessian capture does not descend from the shared 256-row buffer")
        arrays["layer_names"] = np.asarray(names)
        for index, name in enumerate(names):
            result = hessian_aware_w4(
                torch.from_numpy(np.asarray(source[f"weight_{index:04d}"])),
                torch.from_numpy(np.asarray(source[f"inputs_{index:04d}"])),
                device=device,
            )
            artifact = artifact_arrays(result)
            arrays[f"packed_{index:04d}"] = artifact["packed_u4"]
            arrays[f"scales_{index:04d}"] = artifact["scales"]
            arrays[f"clipping_{index:04d}"] = artifact["clipping"]
            arrays[f"error_{index:04d}"] = artifact["reconstruction_error"]
            summaries.append(
                {
                    "name": name,
                    "mean_reconstruction_error": float(result.reconstruction_error.mean()),
                    "mean_clipping": float(result.clipping.mean()),
                    "packed_bytes": int(artifact["packed_u4"].nbytes),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(output)
    payload = {
        "schema_version": 3,
        "kind": "hessian_w4_group64",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "model": model,
        "group_size": 64,
        "clipping_ratios": PROTOCOL["hessian_w4a8"]["clipping_ratios"],
        "calibration_buffer_sha256": PROTOCOL["data"]["calibration_buffer"]["sha256"],
        "capture_path": str(capture),
        "capture_sha256": sha256_file(capture),
        "npz_sha256": sha256_file(output),
        "layer_names": arrays["layer_names"].tolist(),
        "layers": summaries,
        "packed_weight_bytes": int(sum(row["packed_bytes"] for row in summaries)),
        "gradient_updates": False,
        "fp16_weight_updates": False,
        "compute_device": str(device or "cpu"),
    }
    sidecar = Path(str(output) + ".json")
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    payload = build(args.capture, args.out, args.model, device=args.device)
    print(json.dumps({"out": str(Path(args.out).resolve()), "layers": len(payload["layer_names"]), "sha256": payload["npz_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
