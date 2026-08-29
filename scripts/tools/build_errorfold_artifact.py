#!/usr/bin/env python3
"""Fit raw v3 ErrorFold statistics from paired FP16/full-quant captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantvla_cross_model_protocol import PROTOCOL, PROTOCOL_SHA256, protocol_attestation, sha256_file
from quantvla_errorfold import fit_errorfold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True, choices=PROTOCOL["models"])
    return parser.parse_args()


def build(capture_path: str | Path, output_path: str | Path, model: str) -> dict:
    capture = Path(capture_path).expanduser().resolve()
    with np.load(capture, allow_pickle=False) as source:
        names = [str(value) for value in source["layer_names"].tolist()]
        kinds = [str(value) for value in source["layer_kinds"].tolist()]
        buffer_hash = str(source["calibration_buffer_sha256"].item())
        if buffer_hash != PROTOCOL["data"]["calibration_buffer"]["sha256"]:
            raise ValueError("ErrorFold capture does not use the shared calibration buffer")
        layers = {
            name: fit_errorfold(
                np.asarray(source[f"fp16_{index:04d}"]),
                np.asarray(source[f"quant_{index:04d}"]),
                kind=kind,
            )
            for index, (name, kind) in enumerate(zip(names, kinds))
        }
        checkpoint_hash = str(source["checkpoint_sha256"].item())
        plan_hash = str(source["plan_sha256"].item())
    payload = {
        "schema_version": 3,
        "kind": "raw_errorfold",
        "cross_model_protocol": protocol_attestation(),
        "model": model,
        "meta": {
            "checkpoint_sha256": checkpoint_hash,
            "plan_sha256": plan_hash,
            "calibration_buffer_sha256": buffer_hash,
            "flow_steps": PROTOCOL["closed_loop"]["flow_steps"],
            "folds": PROTOCOL["errorfold"]["folds"],
            "fit_pair": PROTOCOL["errorfold"]["fit_pair"],
            "protocol_sha256": PROTOCOL_SHA256,
            "capture_path": str(capture),
            "capture_sha256": sha256_file(capture),
        },
        "layers": layers,
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    payload = build(args.capture, args.out, args.model)
    print(json.dumps({"out": str(Path(args.out).resolve()), "layers": len(payload["layers"])}, indent=2))


if __name__ == "__main__":
    main()
