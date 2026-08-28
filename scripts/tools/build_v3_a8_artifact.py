#!/usr/bin/env python3
"""Build one-prefix/native-flow-step static A8 tables from an FP16 capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from quantvla_cross_model_protocol import PROTOCOL, PROTOCOL_SHA256, sha256_file
from quantvla_hessian_w4 import a8_scale_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True, choices=PROTOCOL["models"])
    return parser.parse_args()


def build(
    capture_path: str | Path,
    output_path: str | Path,
    model: str,
    *,
    flow_steps: int | None = None,
) -> dict:
    capture = Path(capture_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    arrays: dict[str, np.ndarray] = {}
    table_rows: dict[str, int] = {}
    with np.load(capture, allow_pickle=False) as source:
        captured_flow_steps = int(
            np.asarray(source["capture_flow_steps"]).item()
            if "capture_flow_steps" in source
            else PROTOCOL["closed_loop"]["flow_steps"]
        )
        flow_steps = captured_flow_steps if flow_steps is None else int(flow_steps)
        if flow_steps != captured_flow_steps:
            raise ValueError(
                f"A8 flow-step request {flow_steps} != capture {captured_flow_steps}"
            )
        names = [str(value) for value in source["layer_names"].tolist()]
        if model == "pi05":
            arrays["layer_names"] = np.asarray(names)
        for index, name in enumerate(names):
            step_key = f"step_inputs_{index:04d}"
            if step_key in source:
                scale = a8_scale_table(
                    torch.from_numpy(np.asarray(source[step_key])), flow_steps=flow_steps
                )
            else:
                scale = a8_scale_table(
                    torch.from_numpy(np.asarray(source[f"inputs_{index:04d}"]))
                )
            key = name if model == "gr00t" else f"scale_{index:04d}"
            arrays[key] = scale.cpu().numpy()
            table_rows[name] = int(scale.shape[0]) if scale.ndim == 2 else 1
        checkpoint_hash = str(source["checkpoint_sha256"].item())
        plan_hash = str(source["plan_sha256"].item())
        buffer_hash = str(source["calibration_buffer_sha256"].item())
    if buffer_hash != PROTOCOL["data"]["calibration_buffer"]["sha256"]:
        raise ValueError("A8 capture does not descend from shared calibration buffer")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(output)
    metadata = {
        "schema_version": 3,
        "kind": "v3_per_flow_step_a8",
        "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_sha256": checkpoint_hash,
        "plan_sha256": plan_hash,
        "calibration_buffer_sha256": buffer_hash,
        "source_buffer_sha256": buffer_hash,
        "wrapped_layers": len(names),
        "candidate_inventory_sha256": hashlib.sha256(
            ("\n".join(sorted(names)) + "\n").encode("utf-8")
        ).hexdigest(),
        "act_percentile": PROTOCOL["deployment"]["activation_percentile"],
        "calib_batches": PROTOCOL["deployment"]["calibration_batches"],
        "denoising_steps": flow_steps,
        "table_rows": table_rows,
        "prefix_llm_tables": 1,
        "dit_flow_step_tables": flow_steps,
        "capture_path": str(capture),
        "capture_sha256": sha256_file(capture),
    }
    if model == "gr00t":
        sidecar = Path(str(output) + ".meta.json")
        payload = metadata
    else:
        sidecar = Path(str(output) + ".json")
        payload = {
            "schema_version": 1,
            "layer_names": names,
            "npz_sha256": sha256_file(output),
            "metadata": metadata,
        }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    payload = build(args.capture, args.out, args.model)
    print(json.dumps({"out": str(Path(args.out).resolve()), "model": args.model}, indent=2))


if __name__ == "__main__":
    main()
