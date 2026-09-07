#!/usr/bin/env python3
"""Strictly aggregate the preregistered three-trial GR00T hardware benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from quantvla_cross_model_protocol import sha256_file  # noqa: E402
from quantvla_outputimpact import atomic_json  # noqa: E402


CONFIGS = (
    "native_fp16",
    "quantvla_w4a8",
    "context_base",
    "single_best",
    "two_best",
    "attention_6",
    "mlp_2",
    "ff_pair_0",
    "dp_full_lambda_1p0",
)


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sd": float(statistics.stdev(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected_configs = protocol["hardware_evaluation"]["configurations"]
    if expected_configs != list(CONFIGS):
        raise ValueError("hardware configuration order drift")
    rows: dict[str, list[tuple[Path, dict[str, Any]]]] = {key: [] for key in CONFIGS}
    for path in sorted(args.trials.glob("trial*_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        identifier = value.get("config_id")
        if identifier in rows:
            rows[identifier].append((path.resolve(), value))
    if any({value["trial"] for _path, value in values} != {0, 1, 2} for values in rows.values()):
        raise ValueError("every hardware configuration requires exactly trials 0, 1, and 2")
    device_uuids = {
        value["hardware"]["uuid"]
        for values in rows.values()
        for _path, value in values
    }
    if len(device_uuids) != 1:
        raise ValueError(f"hardware device drift: {device_uuids}")
    summaries = {}
    sources = {}
    for identifier in CONFIGS:
        values = [value for _path, value in sorted(rows[identifier], key=lambda item: item[1]["trial"])]
        pooled_latency = np.asarray(
            [latency for value in values for latency in value["latency"]["per_request_ms"]],
            dtype=np.float64,
        )
        trial_p50 = [float(value["latency"]["p50_ms"]) for value in values]
        trial_means = [float(value["latency"]["mean_ms"]) for value in values]
        gross_energy = [float(value["energy"]["gross_board_joules_per_request"]) for value in values]
        adjusted_energy = [float(value["energy"]["idle_adjusted_board_joules_per_request"]) for value in values]
        peak_allocated = [int(value["memory"]["peak_allocated_bytes"]) for value in values]
        peak_reserved = [int(value["memory"]["peak_reserved_bytes"]) for value in values]
        peak_nvml = [float(value["memory"]["nvml_peak_device_used_mib"]) for value in values]
        summaries[identifier] = {
            "trials": 3,
            "requests": int(pooled_latency.size),
            "activation_mode": values[0]["activation_mode"],
            "quantized_w4_layers": (
                0
                if values[0]["quantization_attestation"] is None
                else int(values[0]["quantization_attestation"]["quantized_w4_layers"])
            ),
            "latency_ms": {
                "median_of_trial_p50": float(statistics.median(trial_p50)),
                "pooled_p50": float(np.percentile(pooled_latency, 50)),
                "pooled_p95": float(np.percentile(pooled_latency, 95)),
                "trial_mean": mean_sd(trial_means),
            },
            "peak_memory": {
                "allocated_bytes_median": int(statistics.median(peak_allocated)),
                "reserved_bytes_median": int(statistics.median(peak_reserved)),
                "nvml_device_used_mib_median": float(statistics.median(peak_nvml)),
                "allocated_gib_median": float(statistics.median(peak_allocated) / 2**30),
            },
            "energy_joules_per_request": {
                "gross_board": mean_sd(gross_energy),
                "idle_adjusted_board": mean_sd(adjusted_energy),
            },
        }
        sources[identifier] = [
            {"path": str(path), "sha256": sha256_file(path), "trial": value["trial"]}
            for path, value in sorted(rows[identifier], key=lambda item: item[1]["trial"])
        ]

    reference = summaries["native_fp16"]
    for identifier, row in summaries.items():
        row["relative_to_native_fp16"] = {
            "latency_ratio": row["latency_ms"]["median_of_trial_p50"]
            / reference["latency_ms"]["median_of_trial_p50"],
            "peak_allocated_ratio": row["peak_memory"]["allocated_bytes_median"]
            / reference["peak_memory"]["allocated_bytes_median"],
            "gross_energy_ratio": row["energy_joules_per_request"]["gross_board"]["mean"]
            / reference["energy_joules_per_request"]["gross_board"]["mean"],
        }
    payload = {
        "schema_version": 1,
        "kind": "gr00t_real_hardware_summary",
        "complete": True,
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "device_uuid": next(iter(device_uuids)),
        "scope": "A40 model-only batch-1 policy inference; GPU-board energy only",
        "summaries": summaries,
        "sources": sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, payload)
    print(json.dumps({"out": str(args.out.resolve()), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
