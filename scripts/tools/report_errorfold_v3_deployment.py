#!/usr/bin/env python3
"""Normalize v3 residency, memory, byte and latency measurements for audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--runtime-key", default=None)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--quantvla-static-bytes", type=int, required=True)
    parser.add_argument("--auxiliary-static-bytes", type=int, default=0)
    parser.add_argument("--latency-ms", type=float, action="append", default=[])
    parser.add_argument(
        "--results",
        action="append",
        default=[],
        help="JSONL episode rows; per-request latency is inference_seconds/replans.",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _runtime_payload(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("openpi_runtime"):
        value = value["openpi_runtime"]
    duquant = value.get("duquant") or value.get("quantization_contract") or {}
    memory = value.get("gpu_memory_bytes") or {}
    return {
        "duquant": duquant,
        "memory": memory,
        "runtime": value,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    runtime_path = Path(args.runtime).expanduser().resolve()
    runtime_document = json.loads(runtime_path.read_text(encoding="utf-8"))
    if args.runtime_key:
        runtime_document = runtime_document[args.runtime_key]
    parsed = _runtime_payload(runtime_document)
    duquant = parsed["duquant"]
    packed = int(duquant.get("packed_weight_bytes", 0))
    auxiliary = (
        int(args.auxiliary_static_bytes)
        if int(args.auxiliary_static_bytes) > 0
        else int(duquant.get("auxiliary_static_bytes", 0))
    )
    theoretical = packed + auxiliary
    ratio = theoretical / int(args.quantvla_static_bytes) if args.quantvla_static_bytes else float("inf")
    latencies = [float(value) for value in args.latency_ms]
    for raw_path in args.results:
        path = Path(raw_path).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            replans = int(row.get("replans", 0) or 0)
            inference = float(row.get("inference_seconds", 0.0) or 0.0)
            if replans > 0 and inference > 0.0:
                latencies.append(1000.0 * inference / replans)
    if not latencies:
        raise ValueError("deployment report requires measured per-request latency")
    return {
        "schema_version": 3,
        "kind": "errorfold_v3_deployment_report",
        "config_id": args.config_id,
        "runtime_path": str(runtime_path),
        "packed_low_bit_residency": bool(duquant.get("packed_low_bit_residency", False)),
        "packed_weight_bytes": packed,
        "fp_weight_sized_buffers": int(duquant.get("fp_weight_sized_buffers", -1)),
        "theoretical_static_bytes": theoretical,
        "auxiliary_static_bytes": auxiliary,
        "quantvla_reference_static_bytes": int(args.quantvla_static_bytes),
        "compression_ratio_to_quantvla": ratio,
        "compression_rate_claim_allowed": ratio <= 1.05,
        "gpu_memory_bytes": parsed["memory"],
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "samples": len(latencies),
        },
    }


def main() -> None:
    args = parse_args()
    payload = build(args)
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
