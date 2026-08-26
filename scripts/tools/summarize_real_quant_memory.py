#!/usr/bin/env python3
"""Aggregate the per-GPU real/fake residency probe artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(directory: Path, prefix: str, gpu: int) -> dict:
    return json.loads((directory / f"{prefix}_gpu{gpu}.json").read_text(encoding="utf-8"))


def _gib(value: int | float | None) -> float | None:
    return None if value is None else float(value) / 2**30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for gpu in range(8):
        real = _load(args.directory, "real", gpu)
        fake = _load(args.directory, "fake", gpu)
        row = {"gpu": gpu, "models": {}}
        for model in ("gr00t", "pi05"):
            real_mode = real["models"][model]["modes"]["real_w4"]
            fake_mode = fake["models"][model]["modes"]["current_fake"]
            row["models"][model] = {
                "real_status": real_mode["status"],
                "real_allocated_gib": _gib(real_mode.get("torch_allocated_bytes")),
                "real_reserved_gib": _gib(real_mode.get("torch_reserved_bytes")),
                "real_peak_with_kernel_gib": _gib(real_mode.get("peak_with_kernel_bytes")),
                "free_before_real_gib": real_mode["free_before_gib"],
                "kernel_warm_ms": real_mode.get("packed_kernel", {}).get("warm_call_ms"),
                "fake_status": fake_mode["status"],
                "fake_allocated_gib": _gib(fake_mode.get("torch_allocated_bytes")),
                "fake_reserved_gib": _gib(fake_mode.get("torch_reserved_bytes")),
                "free_before_fake_gib": fake_mode["free_before_gib"],
            }
        rows.append(row)

    models = {}
    for model in ("gr00t", "pi05"):
        real_ok = [row["models"][model] for row in rows if row["models"][model]["real_status"] == "ok"]
        fake_ok = [row["models"][model] for row in rows if row["models"][model]["fake_status"] == "ok"]
        real_allocated = statistics.median(row["real_allocated_gib"] for row in real_ok)
        fake_allocated = statistics.median(row["fake_allocated_gib"] for row in fake_ok)
        models[model] = {
            "real_success_gpus": len(real_ok),
            "fake_success_gpus": len(fake_ok),
            "real_allocated_gib_median": real_allocated,
            "real_reserved_gib_median": statistics.median(row["real_reserved_gib"] for row in real_ok),
            "real_peak_with_kernel_gib_median": statistics.median(
                row["real_peak_with_kernel_gib"] for row in real_ok
            ),
            "fake_allocated_gib_median": fake_allocated,
            "fake_reserved_gib_median": statistics.median(row["fake_reserved_gib"] for row in fake_ok),
            "allocated_reduction_gib": fake_allocated - real_allocated,
            "allocated_reduction_fraction": 1.0 - real_allocated / fake_allocated,
            "kernel_warm_ms_median": statistics.median(row["kernel_warm_ms"] for row in real_ok),
        }

    output = {
        "schema_version": 1,
        "kind": "real_quant_memory_all_gpu_summary",
        "scope": (
            "checkpoint-backed policy tensor residency plus packed GEMM workspace; "
            "no simulator or end-to-end activation peak"
        ),
        "models": models,
        "gpus": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
