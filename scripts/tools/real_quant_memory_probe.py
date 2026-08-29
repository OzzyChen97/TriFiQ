#!/usr/bin/env python3
"""Empirically allocate inference-only QuantVLA tensor residency on one GPU.

The probe reads checkpoint metadata without materializing checkpoint values.
Every checkpoint-backed tensor needed by the policy is allocated at its formal
runtime dtype.  A plan-selected Linear weight is replaced by exactly the
inference tensors a real DuQuant W4A8 runtime needs:

* two signed W4 values per uint8 byte;
* one FP16/BF16 weight scale per output row;
* one FP16/BF16 A8 scale per input channel;
* block input/output rotations in the runtime dtype;
* no source, transformed, or fake-quantized FP weight copy.

For comparison, ``current_fake`` allocates the buffers held by today's
DuQuantLinear implementations.  The real-W4 mode also executes the packed
nibble Triton GEMM on one resident layer, proving the format is executable and
capturing its small kernel workspace peak.  This is a policy-tensor residency
probe, not an end-to-end simulator or activation-peak measurement.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "gr00t": {
        "checkpoint": REPO_ROOT
        / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000",
        "plan": REPO_ROOT
        / "checkpoints/packs/robocasa365/gr00t_quant_plan_robocasa365_cscka_16to1_adjudicated.final_plan.json",
        "runtime_dtype": "bfloat16",
    },
    "pi05": {
        "checkpoint": REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/model.safetensors",
        "plan": REPO_ROOT
        / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_gdsq_cscka_16to1_d4.final_plan.json",
        "runtime_dtype": "float16",
    },
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    category: str

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * torch.empty((), dtype=self.dtype).element_size()


_SAFETENSOR_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _checkpoint_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    single = path / "model.safetensors"
    files = [single] if single.is_file() else sorted(path.glob("model-*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors checkpoint found under {path}")
    return files


def _runtime_dtype(model: str, name: str, source_dtype: torch.dtype) -> torch.dtype:
    if not source_dtype.is_floating_point:
        return source_dtype
    if model == "gr00t":
        return torch.bfloat16
    # scripts/run_pi05_formal_server.sh sets OPENPI_MODEL_DTYPE=float16.  The
    # PaliGemma helper retains only scalar normalization islands in FP32.
    fp32_norm = any(
        marker in name
        for marker in ("input_layernorm", "post_attention_layernorm", "model.norm")
    )
    is_adarms_dense = ".dense.weight" in name or ".dense.bias" in name
    return torch.float32 if fp32_norm and not is_adarms_dense else torch.float16


def read_checkpoint_specs(model: str, checkpoint: Path) -> dict[str, TensorSpec]:
    specs: dict[str, TensorSpec] = {}
    for file_path in _checkpoint_files(checkpoint):
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in specs:
                    raise ValueError(f"duplicate checkpoint tensor: {name}")
                view = handle.get_slice(name)
                source_dtype = _SAFETENSOR_DTYPES[str(view.get_dtype())]
                specs[name] = TensorSpec(
                    name=name,
                    shape=tuple(int(value) for value in view.get_shape()),
                    dtype=_runtime_dtype(model, name, source_dtype),
                    category="checkpoint_retained",
                )
    return specs


def read_quantized_layers(plan_path: Path) -> dict[str, dict]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    layers = payload.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"plan has no layer map: {plan_path}")
    return {
        str(name): dict(config)
        for name, config in layers.items()
        if not bool(config.get("skip", False)) and int(config.get("bits", 0) or 0) > 0
    }


def _quant_metadata_specs(
    layer: str,
    out_features: int,
    in_features: int,
    group: int,
    dtype: torch.dtype,
) -> list[TensorSpec]:
    return [
        TensorSpec(f"{layer}.w_scale", (out_features,), dtype, "weight_scale"),
        TensorSpec(f"{layer}.a8_scale", (in_features,), dtype, "activation_scale"),
        TensorSpec(
            f"{layer}.r_in",
            (math.ceil(in_features / group), group, group),
            dtype,
            "input_rotation",
        ),
        TensorSpec(
            f"{layer}.r_out",
            (math.ceil(out_features / group), group, group),
            dtype,
            "output_rotation",
        ),
    ]


def build_specs(
    model: str,
    checkpoint_specs: dict[str, TensorSpec],
    quantized: dict[str, dict],
    mode: str,
) -> tuple[list[TensorSpec], dict]:
    selected_weights = {f"{layer}.weight": (layer, config) for layer, config in quantized.items()}
    missing = sorted(set(selected_weights) - set(checkpoint_specs))
    if missing:
        raise ValueError(f"{len(missing)} plan weights are absent from checkpoint; first={missing[0]}")

    specs: list[TensorSpec] = []
    quant_parameters = 0
    for name, source in checkpoint_specs.items():
        selected = selected_weights.get(name)
        if selected is None:
            specs.append(source)
            continue

        layer, config = selected
        if len(source.shape) != 2:
            raise ValueError(f"selected weight is not 2-D: {name} {source.shape}")
        out_features, in_features = source.shape
        group = int(config.get("group", 64))
        bits = int(config.get("bits", 4))
        if bits != 4:
            raise ValueError(f"real-W4 probe received W{bits} layer: {layer}")
        quant_parameters += out_features * in_features
        if mode == "real_w4":
            specs.append(
                TensorSpec(
                    f"{layer}.w4_packed",
                    (out_features, (in_features + 1) // 2),
                    torch.uint8,
                    "packed_weight",
                )
            )
        elif mode == "real_int8_container":
            specs.append(
                TensorSpec(
                    f"{layer}.w4_int8_container",
                    (out_features, in_features),
                    torch.int8,
                    "packed_weight",
                )
            )
        elif mode == "current_fake":
            for suffix in ("source", "transformed", "fake_quantized"):
                specs.append(
                    TensorSpec(
                        f"{layer}.{suffix}_weight",
                        source.shape,
                        source.dtype,
                        "fake_fp_weight",
                    )
                )
            if model == "gr00t":
                # GR00T allocates this buffer even when GR00T_DUQUANT_FUSED=0.
                specs.append(
                    TensorSpec(
                        f"{layer}.w4_int8_container",
                        source.shape,
                        torch.int8,
                        "packed_weight",
                    )
                )
        else:
            raise ValueError(f"unsupported mode: {mode}")
        specs.extend(
            _quant_metadata_specs(
                layer, out_features, in_features, group, source.dtype
            )
        )

    category_bytes: dict[str, int] = {}
    for spec in specs:
        category_bytes[spec.category] = category_bytes.get(spec.category, 0) + spec.nbytes
    return specs, {
        "quantized_layers": len(quantized),
        "quantized_parameters": quant_parameters,
        "tensor_count": len(specs),
        "planned_bytes": sum(spec.nbytes for spec in specs),
        "category_bytes": category_bytes,
    }


def _gib(value: int | float) -> float:
    return float(value) / 2**30


def _run_packed_kernel(
    allocated: list[tuple[TensorSpec, torch.Tensor]], model: str
) -> dict:
    sys.path.insert(0, str(REPO_ROOT / "code"))
    from gr00t.quantization.duquant_fused import fused_linear_w4_nibbles

    tensors = {spec.name: tensor for spec, tensor in allocated}
    choices: list[tuple[int, str, TensorSpec, torch.Tensor]] = []
    for spec, tensor in allocated:
        if spec.category == "packed_weight" and spec.name.endswith(".w4_packed"):
            layer = spec.name.removesuffix(".w4_packed")
            choices.append((spec.numel, layer, spec, tensor))
    if not choices:
        return {"status": "not_applicable"}
    _, layer, qspec, qweight = max(choices)
    scale = tensors[f"{layer}.w_scale"]
    in_features = qspec.shape[1] * 2
    qweight.zero_()
    scale.fill_(1.0)
    dtype = torch.bfloat16 if model == "gr00t" else torch.float16
    x = torch.zeros((8, in_features), device=qweight.device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    y = fused_linear_w4_nibbles(x, qweight, scale)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    y = fused_linear_w4_nibbles(x, qweight, scale)
    torch.cuda.synchronize()
    warm_ms = (time.perf_counter() - started) * 1000.0
    finite = bool(torch.isfinite(y).all().item())
    return {
        "status": "ok" if finite else "nonfinite",
        "layer": layer,
        "input_shape": list(x.shape),
        "packed_weight_shape": list(qweight.shape),
        "output_shape": list(y.shape),
        "first_call_ms_including_compile": first_ms,
        "warm_call_ms": warm_ms,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def allocate_mode(
    specs: list[TensorSpec],
    mode: str,
    model: str,
    guard_bytes: int,
    run_kernel: bool,
) -> dict:
    planned = sum(spec.nbytes for spec in specs)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    free_before, total = torch.cuda.mem_get_info()
    base_allocated = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    result = {
        "mode": mode,
        "planned_bytes": planned,
        "planned_gib": _gib(planned),
        "free_before_bytes": int(free_before),
        "free_before_gib": _gib(free_before),
        "guard_bytes": guard_bytes,
        "status": "pending",
    }
    if planned + guard_bytes > free_before:
        result.update(
            status="insufficient_free_memory",
            shortfall_bytes=int(planned + guard_bytes - free_before),
        )
        return result

    allocated: list[tuple[TensorSpec, torch.Tensor]] = []
    torch.cuda.reset_peak_memory_stats()
    try:
        started = time.perf_counter()
        for spec in specs:
            allocated.append((spec, torch.empty(spec.shape, dtype=spec.dtype, device="cuda")))
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        free_resident, _ = torch.cuda.mem_get_info()
        result.update(
            status="ok",
            allocation_seconds=load_seconds,
            torch_allocated_bytes=int(torch.cuda.memory_allocated() - base_allocated),
            torch_reserved_bytes=int(torch.cuda.memory_reserved() - base_reserved),
            device_free_delta_bytes=int(free_before - free_resident),
            free_resident_bytes=int(free_resident),
            tensor_count=len(allocated),
        )
        if run_kernel and mode == "real_w4":
            result["packed_kernel"] = _run_packed_kernel(allocated, model)
            result["peak_with_kernel_bytes"] = int(
                torch.cuda.max_memory_allocated() - base_allocated
            )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        result.update(status="oom", error=str(exc).splitlines()[0])
    finally:
        allocated.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info()
        result["free_after_bytes"] = int(free_after)
        # Other users may allocate concurrently, so this is an observed device
        # delta after our tensors are released rather than a process-leak claim.
        result["device_free_change_after_release_bytes"] = int(free_before - free_after)
    return result


def probe_model(
    model: str,
    checkpoint: Path,
    plan: Path,
    modes: list[str],
    guard_bytes: int,
    run_kernel: bool,
) -> dict:
    checkpoint_specs = read_checkpoint_specs(model, checkpoint)
    quantized = read_quantized_layers(plan)
    mode_specs: dict[str, tuple[list[TensorSpec], dict]] = {}
    for mode in modes:
        mode_specs[mode] = build_specs(model, checkpoint_specs, quantized, mode)
    summary = {
        "model": model,
        "checkpoint": str(checkpoint.resolve()),
        "plan": str(plan.resolve()),
        "runtime_dtype": DEFAULTS[model]["runtime_dtype"],
        "checkpoint_tensor_count": len(checkpoint_specs),
        "checkpoint_runtime_bytes_without_quant_wrapping": sum(
            spec.nbytes for spec in checkpoint_specs.values()
        ),
        "modes": {},
    }
    for mode in modes:
        specs, accounting = mode_specs[mode]
        measured = allocate_mode(
            specs,
            mode,
            model,
            guard_bytes=guard_bytes,
            run_kernel=run_kernel,
        )
        measured["accounting"] = accounting
        summary["modes"][mode] = measured
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="physical GPU id for reporting")
    parser.add_argument("--models", default="gr00t,pi05")
    parser.add_argument("--modes", default="real_w4,current_fake")
    parser.add_argument("--guard-mib", type=int, default=256)
    parser.add_argument("--no-kernel", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.cuda.set_device(0)
    # Initialize the context before measuring free memory so context overhead
    # is not incorrectly attributed to packed policy tensors.
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    report = {
        "schema_version": 1,
        "kind": "real_quant_inference_only_gpu_residency_probe",
        "timestamp_unix": time.time(),
        "host": platform.node(),
        "physical_gpu": args.gpu,
        "cuda_visible_devices": visible,
        "device_name": torch.cuda.get_device_name(0),
        "device_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "scope": (
            "all checkpoint-backed policy tensors at formal runtime dtype; selected weights "
            "replaced by inference-only DuQuant buffers; excludes activations and simulator"
        ),
        "models": {},
    }
    for model in models:
        if model not in DEFAULTS:
            raise ValueError(f"unknown model: {model}")
        report["models"][model] = probe_model(
            model,
            Path(DEFAULTS[model]["checkpoint"]),
            Path(DEFAULTS[model]["plan"]),
            modes,
            guard_bytes=args.guard_mib * 2**20,
            run_kernel=not args.no_kernel,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
