#!/usr/bin/env python3
"""Measure real pi0.5 FCP latency, CUDA/NVML memory, and A40 board energy."""

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


REPO = Path(__file__).resolve().parents[2]
OPENPI = REPO / "code" / "pi05" / "openpi"
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(REPO / "scripts" / "tools"))

from openpi.policies import policy_config  # noqa: E402
from openpi.quant import enable_duquant_if_configured, finalize_real_quant  # noqa: E402
from openpi.training import config  # noqa: E402
from quantvla_cross_model_protocol import sha256_file, validate_quant_plan  # noqa: E402
from quantvla_model_adapters import load_model_records  # noqa: E402
from quantvla_outputimpact import atomic_json  # noqa: E402


CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"


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


def sample_window(samples: list[dict[str, float]], start: float, end: float) -> list[dict[str, float]]:
    return [row for row in samples if start <= row["time"] <= end]


def integrate_energy(samples: list[dict[str, float]], start: float, end: float) -> float:
    selected = sample_window(samples, start, end)
    if len(selected) < 2:
        raise RuntimeError("fewer than two power samples in measurement window")
    times = np.asarray([row["time"] for row in selected], dtype=np.float64)
    power = np.asarray([row["power_w"] for row in selected], dtype=np.float64)
    times = np.concatenate(([start], times, [end]))
    power = np.concatenate(([power[0]], power, [power[-1]]))
    return float(np.trapz(power, times))


def clear_quant_environment() -> None:
    for key in list(os.environ):
        if key.startswith(("OPENPI_DUQUANT_", "OPENPI_ATM_", "OPENPI_OHB_", "OPENPI_RUNTIME_SELECTOR_")):
            os.environ.pop(key, None)
        elif key in ("OPENPI_ERRORFOLD_PATH", "GR00T_GPTQ", "QUANTVLA_ADAPTER_ONLY"):
            os.environ.pop(key, None)


def configure_quant(plan: Path, hessian: Path, pack_dir: Path, calibration_hash: str, wrapped: int) -> None:
    clear_quant_environment()
    os.environ.update(
        {
            "TORCHDYNAMO_DISABLE": "1",
            "OPENPI_MODEL_DTYPE": "float16",
            "OPENPI_CHECKPOINT_SHA256": CHECKPOINT_SHA256,
            "OPENPI_DUQUANT_PLAN": str(plan),
            "OPENPI_DUQUANT_PLAN_STRICT": "1",
            "OPENPI_DUQUANT_WBITS_DEFAULT": "4",
            "OPENPI_DUQUANT_ABITS": "8",
            "OPENPI_DUQUANT_BLOCK": "64",
            "OPENPI_DUQUANT_BLOCK_OUT": "64",
            "OPENPI_DUQUANT_EXPECT_BLOCK": "64",
            "OPENPI_DUQUANT_EXPECT_WRAPPED": str(wrapped),
            "OPENPI_DUQUANT_LS": "0.15",
            "OPENPI_DUQUANT_PERMUTE": "0",
            "OPENPI_DUQUANT_ROW_ROT": "0",
            "OPENPI_DUQUANT_ACT_PCT": "99.9",
            "OPENPI_DUQUANT_CALIB_STEPS": "32",
            "OPENPI_DUQUANT_DENOISING_STEPS": "4",
            "OPENPI_DUQUANT_PACKDIR": str(pack_dir),
            "OPENPI_DUQUANT_ACT_DYNAMIC": "1",
            "OPENPI_DUQUANT_REQUIRE_ACT_SCALE": "0",
            "OPENPI_DUQUANT_CALIB_BUFFER_SHA256": calibration_hash,
            "OPENPI_DUQUANT_STRICT_ARTIFACTS": "1",
            "OPENPI_DUQUANT_PRECACHE_WEIGHTS": "1",
            "OPENPI_DUQUANT_TRITON": "1",
            "OPENPI_DUQUANT_QUIET": "1",
            "OPENPI_DUQUANT_HESSIAN_W4_PATH": str(hessian),
            "OPENPI_ATM_ENABLE": "0",
            "OPENPI_OHB_ENABLE": "0",
            "QUANTVLA_ADAPTER_ONLY": "1",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", required=True, choices=("transferred_initializer", "projected_anchor"))
    parser.add_argument("--trial", required=True, type=int)
    parser.add_argument("--sequence-index", required=True, type=int)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-buffer", required=True, type=Path)
    parser.add_argument("--calibration-buffer", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--hessian-w4", required=True, type=Path)
    parser.add_argument("--pack-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trial not in (0, 1, 2):
        raise ValueError("trial must be 0, 1, or 2")
    if args.warmup != 10 or args.requests != 60 or args.idle_seconds != 5.0:
        raise ValueError("hardware measurement-count drift")
    foreign = [row for row in gpu_compute_processes(args.physical_gpu) if row["pid"] != os.getpid()]
    if foreign:
        raise RuntimeError(f"selected GPU is not exclusive before model load: {foreign}")

    checkpoint = args.checkpoint.expanduser().resolve()
    input_buffer = args.input_buffer.expanduser().resolve()
    calibration_buffer = args.calibration_buffer.expanduser().resolve()
    plan_path = args.plan.expanduser().resolve()
    hessian = args.hessian_w4.expanduser().resolve()
    pack_dir = args.pack_dir.expanduser().resolve()
    if sha256_file(checkpoint / "model.safetensors") != CHECKPOINT_SHA256:
        raise ValueError("checkpoint hash drift")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_attestation = validate_quant_plan(plan, model="pi05", source=str(plan_path))
    wrapped = int(plan_attestation["quantized_w4_layers"])
    hessian_meta = json.loads(Path(str(hessian) + ".json").read_text(encoding="utf-8"))
    if (
        hessian_meta.get("npz_sha256") != sha256_file(hessian)
        or hessian_meta.get("deployment_plan_sha256") != sha256_file(plan_path)
        or len(hessian_meta.get("layer_names") or []) != wrapped
        or hessian_meta.get("requantized") is not False
    ):
        raise ValueError("Hessian subset lineage drift")

    records, buffer_provenance = load_model_records(input_buffer, 32, model="pi05")
    calibration_hash = sha256_file(calibration_buffer)
    configure_quant(plan_path, hessian, pack_dir, calibration_hash, wrapped)
    torch.manual_seed(0)
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_pretrain_human300"),
        checkpoint,
        sample_kwargs={"num_steps": 4},
        pytorch_device="cuda",
    )
    runtime = enable_duquant_if_configured(policy._model)
    policy._model.to("cuda")
    residency = finalize_real_quant(policy._model)
    runtime = getattr(policy._model, "_openpi_duquant_runtime", runtime)
    if (
        int(runtime.get("wrapped_layers", -1)) != wrapped
        or int(runtime.get("hessian_w4_loaded", -1)) != wrapped
        or residency.get("packed_low_bit_residency") is not True
    ):
        raise RuntimeError("real packed-W4 runtime attestation failed")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    @torch.inference_mode()
    def infer(index: int) -> None:
        record = records[index % len(records)]
        policy.infer(record["observation"], noise=record["noises"][0])

    for index in range(args.warmup):
        infer(index)
    torch.cuda.synchronize()

    sampler = NvidiaSmiSampler(args.physical_gpu, interval_ms=100)
    sampler.start()
    idle_start = time.perf_counter()
    time.sleep(args.idle_seconds)
    idle_end = time.perf_counter()
    idle_samples = sample_window(sampler.samples, idle_start, idle_end)
    if len(idle_samples) < 10:
        sampler.stop()
        raise RuntimeError("insufficient idle power samples")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident_allocated = int(torch.cuda.memory_allocated())
    resident_reserved = int(torch.cuda.memory_reserved())
    latencies_ms: list[float] = []
    measure_start = time.perf_counter()
    for index in range(args.requests):
        torch.cuda.synchronize()
        started = time.perf_counter()
        infer(index + args.warmup)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    measure_end = time.perf_counter()
    sampler.stop()
    measurement_samples = sample_window(sampler.samples, measure_start, measure_end)
    if len(measurement_samples) < 10:
        raise RuntimeError("insufficient measurement power samples")
    foreign_after = [row for row in gpu_compute_processes(args.physical_gpu) if row["pid"] != os.getpid()]
    if foreign_after:
        raise RuntimeError(f"selected GPU lost exclusivity during measurement: {foreign_after}")

    idle_power = float(statistics.median(row["power_w"] for row in idle_samples))
    gross_energy = integrate_energy(sampler.samples, measure_start, measure_end)
    duration = measure_end - measure_start
    adjusted_energy = max(0.0, gross_energy - idle_power * duration)
    latencies = np.asarray(latencies_ms, dtype=np.float64)
    payload = {
        "schema_version": 1,
        "kind": "pi05_fcp_real_hardware_trial",
        "config_id": args.config_id,
        "trial": args.trial,
        "sequence_index": args.sequence_index,
        "hardware": query_gpu(args.physical_gpu),
        "physical_gpu": args.physical_gpu,
        "process_pid": os.getpid(),
        "checkpoint": {"path": str(checkpoint), "sha256": CHECKPOINT_SHA256},
        "input_buffer": {"path": str(input_buffer), "sha256": buffer_provenance["sha256"], "rows": 32},
        "calibration_buffer": {"path": str(calibration_buffer), "sha256": calibration_hash},
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "hessian_w4": {"path": str(hessian), "sha256": sha256_file(hessian)},
        "identity_pack": {"path": str(pack_dir), "manifest_sha256": sha256_file(pack_dir / "manifest.json")},
        "quantization_attestation": plan_attestation,
        "real_quant_runtime": runtime,
        "real_quant_residency": residency,
        "protocol": {
            "flow_steps": 4,
            "batch_size": 1,
            "warmup_requests": args.warmup,
            "measurement_requests": args.requests,
            "idle_seconds": args.idle_seconds,
            "power_sample_interval_ms": 100,
            "scope": "policy infer including native transforms and inverse normalization; excludes model load, simulator, and transport",
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
            "idle_adjusted_board_joules": adjusted_energy,
            "idle_adjusted_board_joules_per_request": adjusted_energy / args.requests,
            "mean_power_w": float(np.mean([row["power_w"] for row in measurement_samples])),
            "peak_power_w": float(max(row["power_w"] for row in measurement_samples)),
            "idle_samples": len(idle_samples),
            "measurement_samples": len(measurement_samples),
        },
        "exclusive_gpu": True,
    }
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "out": str(output),
                "config": args.config_id,
                "trial": args.trial,
                "p50_ms": payload["latency"]["p50_ms"],
                "peak_gib": payload["memory"]["peak_allocated_bytes"] / 2**30,
                "gross_j_per_request": payload["energy"]["gross_board_joules_per_request"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
