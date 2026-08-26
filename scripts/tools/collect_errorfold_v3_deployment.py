#!/usr/bin/env python3
"""Collect runtime memory/bytes and measured closed-loop request latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from quantvla_cross_model_protocol import PROTOCOL


CONFIGS = tuple(row["id"] for row in PROTOCOL["evaluation_matrix"]["configs"])


def _runtime(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("openpi_runtime"):
        value = value["openpi_runtime"]
    contract: dict[str, Any] = {}
    contract.update(value.get("quantization_contract") or {})
    contract.update(value.get("cross_model_quantization_contract") or {})
    contract.update(value.get("duquant") or {})
    return {"runtime": value, "contract": contract}


def _latencies(paths: Iterable[Path], config: str) -> list[float]:
    values = []
    for path in sorted(set(paths)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("config") or row.get("config_id")) != config:
                continue
            replans = int(row.get("replans", 0) or 0)
            seconds = float(row.get("inference_seconds", 0.0) or 0.0)
            if replans > 0 and seconds > 0.0:
                values.append(1000.0 * seconds / replans)
    if not values:
        raise ValueError(f"{config}: no measured closed-loop request latency")
    return values


def _static_bytes(contract: dict[str, Any]) -> int:
    return int(contract.get("packed_weight_bytes", 0)) + int(
        contract.get("auxiliary_static_bytes", 0)
    )


def _report(
    *,
    model: str,
    task_set: str | None,
    config: str,
    runtime_path: Path,
    runtime: dict[str, Any],
    latencies: list[float],
    quantvla_bytes: int,
) -> dict[str, Any]:
    parsed = _runtime(runtime)
    contract = parsed["contract"]
    runtime_value = parsed["runtime"]
    static = _static_bytes(contract)
    ratio = static / quantvla_bytes if quantvla_bytes and static else (
        0.0 if config == "fp16" else float("inf")
    )
    return {
        "schema_version": 3,
        "kind": "errorfold_v3_deployment_report",
        "model": model,
        "task_set": task_set,
        "config_id": config,
        "runtime_path": str(runtime_path),
        "packed_low_bit_residency": bool(
            contract.get("packed_low_bit_residency", False)
        ),
        "packed_weight_bytes": int(contract.get("packed_weight_bytes", 0)),
        "fp_weight_sized_buffers": int(
            contract.get("fp_weight_sized_buffers", 0 if config == "fp16" else -1)
        ),
        "dequant_scale_bytes": int(contract.get("dequant_scale_bytes", 0)),
        "input_gain_bytes": int(contract.get("input_gain_bytes", 0)),
        "activation_scale_bytes": int(
            contract.get("activation_scale_bytes", 0)
        ),
        "bias_bytes": int(contract.get("bias_bytes", 0)),
        "auxiliary_static_bytes": int(contract.get("auxiliary_static_bytes", 0)),
        "theoretical_static_bytes": static,
        "quantvla_reference_static_bytes": quantvla_bytes,
        "compression_ratio_to_quantvla": ratio,
        "compression_rate_claim_allowed": config == "fp16" or ratio <= 1.05,
        "wrapped_layers": int(
            contract.get("wrapped_layers", runtime_value.get("wrapped_layers", 0))
        ),
        "weight_bits": int(contract.get("weight_bits", 0)),
        "activation_bits": int(contract.get("act_bits", contract.get("activation_bits", 0))),
        "block_in": int(contract.get("block_in", 0)),
        "block_out": int(contract.get("block_out", 0)),
        "row_rotation": str(contract.get("row_rotation", "")),
        "hessian_w4_loaded": int(
            contract.get("hessian_w4_loaded", runtime_value.get("hessian_w4_loaded", 0))
        ),
        "hessian_group_size": int(
            contract.get("hessian_group_size", runtime_value.get("hessian_group_size", 0)) or 0
        ),
        "runtime_selector_enabled": bool(
            (runtime_value.get("runtime_selector") or {}).get("enabled", False)
        ),
        "gpu_memory_bytes": runtime_value.get("gpu_memory_bytes") or {},
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "samples": len(latencies),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).expanduser().resolve()
    output_root = root / "deployment"
    for task_set in PROTOCOL["closed_loop"]["tasks"]:
        run_dir = root / "closed_loop/gr00t" / task_set
        runtime_path = run_dir / "runtime_info.json"
        document = json.loads(runtime_path.read_text(encoding="utf-8"))
        primary = {config: document[config] for config in CONFIGS}
        reference = _static_bytes(_runtime(primary["quantvla_w4a8_paper"])["contract"])
        for config in CONFIGS:
            latency = _latencies(run_dir.glob(f"{config}_s*.jsonl"), config)
            report = _report(
                model="gr00t",
                task_set=task_set,
                config=config,
                runtime_path=runtime_path,
                runtime=primary[config],
                latencies=latency,
                quantvla_bytes=reference,
            )
            path = output_root / "gr00t" / f"{task_set}_{config}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    runtime_path = root / "closed_loop/pi05/runtime_info.json"
    document = json.loads(runtime_path.read_text(encoding="utf-8"))
    primary = {config: document[config] for config in CONFIGS}
    reference = _static_bytes(_runtime(primary["quantvla_w4a8_paper"])["contract"])
    for config in CONFIGS:
        result_dir = root / "closed_loop/pi05/results" / config
        latency = _latencies(result_dir.glob("*.jsonl"), config)
        report = _report(
            model="pi05",
            task_set=None,
            config=config,
            runtime_path=runtime_path,
            runtime=primary[config],
            latencies=latency,
            quantvla_bytes=reference,
        )
        path = output_root / "pi05" / f"{config}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"deployment_root": str(output_root), "complete": True}, indent=2))


if __name__ == "__main__":
    main()
