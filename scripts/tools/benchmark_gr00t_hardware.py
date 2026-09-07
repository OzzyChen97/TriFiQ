#!/usr/bin/env python3
"""Measure isolated GR00T policy latency, CUDA peak memory, and GPU-board energy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from gr00t.quantization import finalize_real_quant  # noqa: E402
from gr00t_v2_common import (  # noqa: E402
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    ensure_a8_calibrated,
    ensure_flash_attn_rpath,
    load_policy,
    set_quant_env,
    strip_quant_env,
)
from quantvla_cross_model_protocol import sha256_file, validate_quant_plan  # noqa: E402
from quantvla_model_adapters import gr00t_rollout_inputs, load_model_records  # noqa: E402
from quantvla_outputimpact import atomic_json  # noqa: E402


def query_gpu(gpu: int) -> dict[str, Any]:
    fields = (
        "index,uuid,name,driver_version,memory.total,power.limit,temperature.gpu,"
        "clocks.current.graphics,clocks.current.memory,pstate"
    )
    line = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    values = [value.strip() for value in line.split(",")]
    return dict(zip(fields.split(","), values, strict=True))


def gpu_compute_processes(gpu: int) -> list[dict[str, Any]]:
    uuid = query_gpu(gpu)["uuid"]
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",", 3)]
        if len(fields) == 4 and fields[0] == uuid:
            rows.append(
                {
                    "pid": int(fields[1]),
                    "used_memory_mib": float(fields[2]),
                    "process": fields[3],
                }
            )
    return rows


class NvidiaSmiSampler:
    def __init__(self, gpu: int, interval_ms: int = 100):
        self.gpu = gpu
        self.interval_ms = interval_ms
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def sample() -> None:
            while not self._stop.is_set():
                sampled = time.perf_counter()
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        str(self.gpu),
                        "--query-gpu=power.draw,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    capture_output=True,
                )
                if result.returncode == 0:
                    fields = [value.strip() for value in result.stdout.strip().split(",")]
                    if len(fields) == 3:
                        try:
                            self.samples.append(
                                {
                                    "time": sampled,
                                    "power_w": float(fields[0]),
                                    "memory_used_mib": float(fields[1]),
                                    "utilization_percent": float(fields[2]),
                                }
                            )
                        except ValueError:
                            pass
                self._stop.wait(self.interval_ms / 1000.0)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def window(samples: list[dict[str, float]], start: float, end: float) -> list[dict[str, float]]:
    return [row for row in samples if start <= row["time"] <= end]


def integrate_energy(samples: list[dict[str, float]], start: float, end: float) -> float:
    selected = window(samples, start, end)
    if len(selected) < 2:
        raise RuntimeError("fewer than two power samples in measurement window")
    times = np.asarray([row["time"] for row in selected], dtype=np.float64)
    power = np.asarray([row["power_w"] for row in selected], dtype=np.float64)
    # Extend the nearest readings to the exact measurement boundaries.
    times = np.concatenate(([start], times, [end]))
    power = np.concatenate(([power[0]], power, [power[-1]]))
    return float(np.trapz(power, times))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--pack-dir")
    parser.add_argument("--hessian-w4")
    parser.add_argument("--act-scale")
    parser.add_argument("--activation-mode", choices=("fp16", "static_a8", "dynamic_a8"), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup != 10 or args.requests != 60 or args.idle_seconds != 5.0:
        raise ValueError("measurement-count drift from frozen hardware protocol")
    foreign = [row for row in gpu_compute_processes(args.physical_gpu) if row["pid"] != os.getpid()]
    if foreign:
        raise RuntimeError(f"selected GPU is not exclusive before model load: {foreign}")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    buffer_path = Path(args.buffer).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve() if args.plan else None
    pack_dir = Path(args.pack_dir).expanduser().resolve() if args.pack_dir else None
    hessian_path = Path(args.hessian_w4).expanduser().resolve() if args.hessian_w4 else None
    act_scale_path = Path(args.act_scale).expanduser().resolve() if args.act_scale else None
    if (plan_path is None) != (args.activation_mode == "fp16"):
        raise ValueError("native fp16 must have no plan and quantized modes must have a plan")
    if plan_path is not None:
        if pack_dir is None:
            raise ValueError("quantized mode requires --pack-dir")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        quant_attestation = validate_quant_plan(plan, model="gr00t", source=str(plan_path))
    else:
        quant_attestation = None

    records, buffer_provenance = load_model_records(buffer_path, 48, model="gr00t")
    observations, noises = gr00t_rollout_inputs(records, noise_index=0)
    ensure_flash_attn_rpath()
    strip_quant_env()
    os.environ["GR00T_ATM_ENABLE"] = "0"
    os.environ["GR00T_OHB_ENABLE"] = "0"
    if plan_path is not None:
        row_rot = "0" if hessian_path is not None else "restore"
        set_quant_env(
            DEFAULT_INCLUDE,
            DEFAULT_EXCLUDE,
            str(pack_dir),
            row_rot=row_rot,
            act_dynamic=args.activation_mode == "dynamic_a8",
        )
        os.environ["GR00T_DUQUANT_PLAN"] = str(plan_path)
        os.environ["GR00T_DUQUANT_FUSED"] = "1"
        os.environ["GR00T_DUQUANT_ABITS"] = "8"
        if hessian_path is not None:
            os.environ["GR00T_DUQUANT_HESSIAN_W4_PATH"] = str(hessian_path)
        if act_scale_path is not None:
            os.environ["GR00T_DUQUANT_ACT_SCALE_PATH"] = str(act_scale_path)
    torch.manual_seed(0)
    policy = load_policy(
        str(checkpoint),
        data_config="examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig",
        denoising_steps=4,
        device="cuda",
    )
    residency = None
    if plan_path is not None:
        if args.activation_mode == "static_a8":
            if act_scale_path is None:
                raise ValueError("static A8 requires --act-scale")
            scale_meta = json.loads(
                Path(str(act_scale_path) + ".meta.json").read_text(encoding="utf-8")
            )
            if scale_meta.get("plan_sha256") != sha256_file(plan_path):
                raise ValueError("static A8 scale/plan lineage drift")
            ensure_a8_calibrated(
                policy,
                [],
                [],
                1,
                expected_wrapped=int(quant_attestation["quantized_w4_layers"]),
                act_scale_path=str(act_scale_path),
            )
        residency = finalize_real_quant(policy.model)
        torch.cuda.empty_cache()
    torch.cuda.synchronize()

    def infer(index: int) -> None:
        policy.get_action(
            observations[index % len(observations)],
            action_noise=noises[index % len(noises)],
        )

    for index in range(args.warmup):
        infer(index)
    torch.cuda.synchronize()
    sampler = NvidiaSmiSampler(args.physical_gpu, interval_ms=100)
    sampler.start()
    idle_start = time.perf_counter()
    time.sleep(args.idle_seconds)
    idle_end = time.perf_counter()
    idle_samples = window(sampler.samples, idle_start, idle_end)
    if len(idle_samples) < 10:
        sampler.stop()
        raise RuntimeError("insufficient idle power samples")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident_allocated = int(torch.cuda.memory_allocated())
    resident_reserved = int(torch.cuda.memory_reserved())
    latencies_ms = []
    measure_start = time.perf_counter()
    for index in range(args.requests):
        torch.cuda.synchronize()
        started = time.perf_counter()
        infer(index + args.warmup)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    measure_end = time.perf_counter()
    sampler.stop()
    measurement_samples = window(sampler.samples, measure_start, measure_end)
    if len(measurement_samples) < 10:
        raise RuntimeError("insufficient measurement power samples")
    foreign_after = [row for row in gpu_compute_processes(args.physical_gpu) if row["pid"] != os.getpid()]
    if foreign_after:
        raise RuntimeError(f"selected GPU lost exclusivity during measurement: {foreign_after}")

    idle_power = float(statistics.median(row["power_w"] for row in idle_samples))
    gross_energy = integrate_energy(sampler.samples, measure_start, measure_end)
    duration = measure_end - measure_start
    dynamic_energy = max(0.0, gross_energy - idle_power * duration)
    latencies = np.asarray(latencies_ms, dtype=np.float64)
    device = query_gpu(args.physical_gpu)
    payload = {
        "schema_version": 1,
        "kind": "gr00t_real_hardware_trial",
        "config_id": args.config_id,
        "trial": args.trial,
        "hardware": device,
        "physical_gpu": args.physical_gpu,
        "process_pid": os.getpid(),
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint / "model-00001-of-00002.safetensors")},
        "selection_buffer": {"path": str(buffer_path), "sha256": buffer_provenance["sha256"]},
        "plan": None if plan_path is None else {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "hessian_w4": None if hessian_path is None else {"path": str(hessian_path), "sha256": sha256_file(hessian_path)},
        "act_scale": None if act_scale_path is None else {"path": str(act_scale_path), "sha256": sha256_file(act_scale_path)},
        "activation_mode": args.activation_mode,
        "quantization_attestation": quant_attestation,
        "real_quant_residency": residency,
        "protocol": {
            "flow_steps": 4,
            "batch_size": 1,
            "warmup_requests": args.warmup,
            "measurement_requests": args.requests,
            "idle_seconds": args.idle_seconds,
            "power_sample_interval_ms": 100,
            "scope": "policy get_action including transforms and inverse normalization; excludes model load, simulator, and transport",
        },
        "latency": {
            "per_request_ms": latencies.tolist(),
            "mean_ms": float(latencies.mean()),
            "std_ms": float(latencies.std(ddof=1)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "min_ms": float(latencies.min()),
            "max_ms": float(latencies.max()),
            "throughput_requests_per_second": float(args.requests / duration),
            "measurement_wall_seconds": duration,
        },
        "memory": {
            "resident_allocated_bytes": resident_allocated,
            "resident_reserved_bytes": resident_reserved,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "nvml_peak_device_used_mib": max(row["memory_used_mib"] for row in measurement_samples),
        },
        "energy": {
            "sensor": "nvidia-smi/NVML power.draw",
            "scope": "GPU board only",
            "idle_power_w_median": idle_power,
            "gross_board_joules": gross_energy,
            "gross_board_joules_per_request": gross_energy / args.requests,
            "idle_adjusted_board_joules": dynamic_energy,
            "idle_adjusted_board_joules_per_request": dynamic_energy / args.requests,
            "mean_power_w": float(np.mean([row["power_w"] for row in measurement_samples])),
            "peak_power_w": float(max(row["power_w"] for row in measurement_samples)),
            "idle_samples": len(idle_samples),
            "measurement_samples": len(measurement_samples),
        },
        "exclusive_gpu": True,
    }
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    print(json.dumps({"out": str(output), "config": args.config_id, "trial": args.trial, "p50_ms": payload["latency"]["p50_ms"], "peak_gib": payload["memory"]["peak_allocated_bytes"] / 2**30, "gross_j_per_request": payload["energy"]["gross_board_joules_per_request"]}, indent=2))


if __name__ == "__main__":
    main()
