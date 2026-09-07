#!/usr/bin/env python3
"""Strictly aggregate the preregistered pi0.5 pre/post-projection hardware trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tools"))

from quantvla_cross_model_protocol import sha256_file  # noqa: E402
from quantvla_outputimpact import atomic_json  # noqa: E402


CONFIGS = ("transferred_initializer", "projected_anchor")


def mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": float(statistics.mean(values)), "sd": float(statistics.stdev(values))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    hardware_protocol = protocol["hardware"]
    if hardware_protocol["configurations"] != list(CONFIGS):
        raise ValueError("hardware configuration order drift")
    expected_sequence = [tuple(value) for value in hardware_protocol["order"]]

    rows: dict[str, list[tuple[Path, dict[str, Any]]]] = {key: [] for key in CONFIGS}
    for path in sorted(args.trials.glob("trial*_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("kind") != "pi05_fcp_real_hardware_trial":
            raise ValueError(f"unexpected hardware row kind: {path}")
        identifier = value.get("config_id")
        if identifier not in rows:
            raise ValueError(f"unexpected hardware configuration: {path}")
        rows[identifier].append((path.resolve(), value))
    if any(len(values) != 3 for values in rows.values()):
        raise ValueError("each hardware configuration requires exactly three trials")
    observed_sequence = sorted(
        (int(value["sequence_index"]), int(value["trial"]), str(value["config_id"]))
        for values in rows.values()
        for _path, value in values
    )
    wanted_sequence = sorted((index, int(trial), str(config_id)) for index, (trial, config_id) in enumerate(expected_sequence))
    if observed_sequence != wanted_sequence:
        raise ValueError(f"interleaved trial order drift: {observed_sequence} != {wanted_sequence}")
    if any({int(value["trial"]) for _path, value in values} != {0, 1, 2} for values in rows.values()):
        raise ValueError("trial IDs must be exactly 0, 1, and 2")
    device_uuids = {value["hardware"]["uuid"] for values in rows.values() for _path, value in values}
    device_names = {value["hardware"]["name"] for values in rows.values() for _path, value in values}
    if len(device_uuids) != 1 or device_names != {"NVIDIA A40"}:
        raise ValueError(f"hardware device drift: uuid={device_uuids}, name={device_names}")

    summaries: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for identifier in CONFIGS:
        values = [value for _path, value in sorted(rows[identifier], key=lambda item: item[1]["trial"])]
        frozen = protocol["configurations"][identifier]
        for value in values:
            checks = {
                "plan": value["plan"]["sha256"] == frozen["plan_sha256"],
                "w4": int(value["quantization_attestation"]["quantized_w4_layers"]) == int(frozen["w4_layers"]),
                "requests": int(value["protocol"]["measurement_requests"]) == 60
                and len(value["latency"]["per_request_ms"]) == 60,
                "warmup": int(value["protocol"]["warmup_requests"]) == 10,
                "flow": int(value["protocol"]["flow_steps"]) == 4,
                "batch": int(value["protocol"]["batch_size"]) == 1,
                "exclusive": value.get("exclusive_gpu") is True,
                "packed": value["real_quant_residency"]["packed_low_bit_residency"] is True,
            }
            failed = [key for key, passed in checks.items() if not passed]
            if failed:
                raise ValueError(f"{identifier} trial {value['trial']} drift: {failed}")
        pooled_latency = np.asarray(
            [latency for value in values for latency in value["latency"]["per_request_ms"]],
            dtype=np.float64,
        )
        trial_p50 = [float(value["latency"]["p50_ms"]) for value in values]
        trial_means = [float(value["latency"]["mean_ms"]) for value in values]
        trial_throughput = [float(value["latency"]["throughput_requests_per_second"]) for value in values]
        gross_energy = [float(value["energy"]["gross_board_joules_per_request"]) for value in values]
        adjusted_energy = [float(value["energy"]["idle_adjusted_board_joules_per_request"]) for value in values]
        peak_allocated = [int(value["memory"]["peak_allocated_bytes"]) for value in values]
        peak_reserved = [int(value["memory"]["peak_reserved_bytes"]) for value in values]
        peak_nvml = [float(value["memory"]["nvml_peak_device_used_mib"]) for value in values]
        summaries[identifier] = {
            "trials": 3,
            "requests": int(pooled_latency.size),
            "w4_layers": int(frozen["w4_layers"]),
            "fp16_layers": int(frozen["fp16_layers"]),
            "table1_total_static_bytes": int(frozen["table1_total_static_bytes"]),
            "deployment_budget_compliant": bool(frozen["deployment_budget_compliant"]),
            "latency_ms": {
                "median_of_trial_p50": float(statistics.median(trial_p50)),
                "pooled_p50": float(np.percentile(pooled_latency, 50)),
                "pooled_p95": float(np.percentile(pooled_latency, 95)),
                "trial_mean": mean_sd(trial_means),
                "throughput_requests_per_second": mean_sd(trial_throughput),
            },
            "peak_memory": {
                "allocated_bytes_median": int(statistics.median(peak_allocated)),
                "reserved_bytes_median": int(statistics.median(peak_reserved)),
                "nvml_device_used_mib_median": float(statistics.median(peak_nvml)),
                "allocated_gib_median": float(statistics.median(peak_allocated) / 2**30),
                "reserved_gib_median": float(statistics.median(peak_reserved) / 2**30),
            },
            "energy_joules_per_request": {
                "gross_board": mean_sd(gross_energy),
                "idle_adjusted_board": mean_sd(adjusted_energy),
            },
        }
        sources[identifier] = [
            {"path": str(path), "sha256": sha256_file(path), "trial": int(value["trial"]), "requests": 60}
            for path, value in sorted(rows[identifier], key=lambda item: item[1]["trial"])
        ]

    before = summaries["transferred_initializer"]
    after = summaries["projected_anchor"]
    comparison = {
        "orientation": "projected_anchor relative to transferred_initializer",
        "latency_ratio": after["latency_ms"]["median_of_trial_p50"] / before["latency_ms"]["median_of_trial_p50"],
        "peak_allocated_ratio": after["peak_memory"]["allocated_bytes_median"] / before["peak_memory"]["allocated_bytes_median"],
        "peak_reserved_ratio": after["peak_memory"]["reserved_bytes_median"] / before["peak_memory"]["reserved_bytes_median"],
        "nvml_peak_ratio": after["peak_memory"]["nvml_device_used_mib_median"] / before["peak_memory"]["nvml_device_used_mib_median"],
        "gross_energy_ratio": after["energy_joules_per_request"]["gross_board"]["mean"] / before["energy_joules_per_request"]["gross_board"]["mean"],
        "idle_adjusted_energy_ratio": after["energy_joules_per_request"]["idle_adjusted_board"]["mean"] / before["energy_joules_per_request"]["idle_adjusted_board"]["mean"],
    }
    payload = {
        "schema_version": 1,
        "kind": "pi05_fcp_real_hardware_summary",
        "complete": True,
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "device_uuid": next(iter(device_uuids)),
        "device_name": next(iter(device_names)),
        "coverage": {"trials": 6, "measured_requests": 360},
        "scope": "A40 model-only batch-1 policy inference; GPU-board energy only",
        "summaries": summaries,
        "projected_vs_transferred": comparison,
        "sources": sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, payload)
    print(json.dumps({"out": str(args.out.resolve()), "summaries": summaries, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
