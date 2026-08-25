#!/usr/bin/env python3
"""Audit Omega-QVLA theoretical storage in the paper's Linear-layer scope.

The released/evaluation packs store dequantized FP16 weight and dense rotation
tensors, so their on-disk size is not a packed-W4 deployment size.  This audit
derives the equivalent tightly packed representation from an attested pack:

* W4 weights and FP16 GPTQ scales for every input block;
* FP16 64x64 input/output rotation blocks;
* int32 DiT permutation indices;
* the frozen activation-scale tables (FP32 for DiT, FP16 for the LLM); and
* FP16 biases and any nonzero low-rank residual factors.

The scope is deliberately identical to ``robocasa_paper_memory.py``.  Vision
modules, activations, runtime workspaces, and the simulator remain excluded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from robocasa_paper_memory import (  # noqa: E402
    _base_bytes,
    paper_scope,
    read_tensor_shapes,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def verified_path(record: dict[str, Any], label: str) -> Path:
    path = resolve(str(record["path"]))
    require(path.is_file(), f"missing {label}: {path}")
    require(sha256_file(path) == record["sha256"], f"{label} SHA drift: {path}")
    return path


def rotation_block_bytes(features: int, block_size: int, element_bytes: int) -> int:
    require(features > 0 and block_size > 0 and element_bytes > 0, "invalid rotation shape")
    return sum(
        min(block_size, features - start) ** 2 * element_bytes
        for start in range(0, features, block_size)
    )


def load_pack(path: Path) -> dict[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # older torch without mmap/weights_only
        value = torch.load(path, map_location="cpu")
    require(isinstance(value, dict), "Omega-QVLA pack must be a dictionary")
    return value


def calculate(manifest_path: Path, summary_path: Path) -> dict[str, Any]:
    import torch

    manifest_path = resolve(manifest_path)
    summary_path = resolve(summary_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary.get("validation_errors") == [], "result summary has validation errors")
    require(summary.get("task_set") == "atomic_seen", "storage audit expects Atomic-Seen pack")

    configs = manifest.get("configs") or []
    require(len(configs) == 1, "Omega-QVLA manifest must contain exactly one config")
    config = configs[0]
    require(config.get("id") == "omega_qvla_w4a4", "unexpected Omega-QVLA config id")
    require(config.get("meta", {}).get("weight_bits") == 4, "manifest is not W4")
    require(config.get("meta", {}).get("activation_bits") == 4, "manifest is not A4")
    require(config.get("expected_wrapped") == 180, "expected wrapped-layer count drift")

    pack_path = verified_path(config["omega_pack"], "Omega-QVLA pack")
    attestation_path = verified_path(
        config["omega_pack_attestation"], "Omega-QVLA pack attestation"
    )
    calibration_path = verified_path(
        config["omega_calibration_manifest"], "Omega-QVLA calibration manifest"
    )
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    require(attestation.get("pack_sha256") == sha256_file(pack_path), "pack attestation drift")
    require(
        attestation.get("calibration_manifest_sha256") == sha256_file(calibration_path),
        "calibration attestation drift",
    )
    recipe = calibration["recipe"]
    require(recipe.get("weight_bits") == 4, "calibration recipe is not W4")
    require(recipe.get("activation_bits") == 4, "calibration recipe is not A4")
    input_block = int(recipe["duquant_block_size"])
    output_block = int(recipe["duquant_block_out"])
    default_gptq_block = int(recipe["gptq_block_size"])

    checkpoint = resolve(manifest["checkpoint_path"])
    shapes = read_tensor_shapes(checkpoint)
    scope_layers = paper_scope(shapes)
    fp16_bytes, fp16_weight_bytes, fp16_bias_bytes = _base_bytes(shapes, scope_layers)
    pack = load_pack(pack_path)
    records = {name: record for name, record in pack.items() if name != "__meta__"}
    expected_records = {name[:-7] for name in scope_layers}
    require(set(records) == expected_records, "pack records do not exactly cover paper scope")

    breakdown = {
        "packed_weights": 0,
        "weight_scales": 0,
        "input_rotations": 0,
        "output_rotations": 0,
        "permutation_indices": 0,
        "activation_scales": 0,
        "low_rank_factors": 0,
        "fp16_biases": fp16_bias_bytes,
    }
    record_classes = {"llm_gptq": 0, "dit_per_step": 0}
    artifact_tensor_bytes = 0
    quantized_parameters = 0

    for name, record in sorted(records.items()):
        require(isinstance(record, dict), f"non-dict pack record: {name}")
        weight_name = f"{name}.weight"
        out_features, in_features = shapes[weight_name]
        weight = record.get("baseline_q", record.get("weight_res_q"))
        require(torch.is_tensor(weight), f"missing quantized weight: {name}")
        require(tuple(weight.shape) == (out_features, in_features), f"weight shape drift: {name}")
        require(int(record.get("weight_bits", 0)) == 4, f"non-W4 record: {name}")
        quantized_parameters += out_features * in_features
        breakdown["packed_weights"] += math.ceil(out_features * in_features * 4 / 8)
        gptq_block = int(record.get("gptq_block_size", default_gptq_block))
        breakdown["weight_scales"] += out_features * math.ceil(in_features / gptq_block) * 2

        rotation = record.get("duquant_rotation")
        require(torch.is_tensor(rotation), f"missing input rotation: {name}")
        require(tuple(rotation.shape) == (in_features, in_features), f"input rotation drift: {name}")
        require(rotation.dtype == torch.float16, f"input rotation precision drift: {name}")
        breakdown["input_rotations"] += rotation_block_bytes(in_features, input_block, 2)

        is_dit = record.get("format") == "dit_svdquant_v1"
        if is_dit:
            record_classes["dit_per_step"] += 1
            require(int(record.get("a_bits", 0)) == 4, f"non-A4 DiT record: {name}")
            table = record.get("act_scale_table")
            require(torch.is_tensor(table), f"missing per-step A4 table: {name}")
            require(
                tuple(table.shape) == (int(recipe["denoising_steps"]), in_features),
                f"activation-scale table drift: {name}",
            )
            breakdown["activation_scales"] += table.numel() * table.element_size()
            breakdown["permutation_indices"] += in_features * 4

            output_rotation = record.get("duquant_rotation_out")
            require(torch.is_tensor(output_rotation), f"missing output rotation: {name}")
            require(
                tuple(output_rotation.shape) == (out_features, out_features),
                f"output rotation drift: {name}",
            )
            require(output_rotation.dtype == torch.float16, f"output rotation precision drift: {name}")
            breakdown["output_rotations"] += rotation_block_bytes(out_features, output_block, 2)
            for key in ("lowrank_A", "lowrank_B"):
                factor = record.get(key)
                require(torch.is_tensor(factor), f"missing {key}: {name}")
                breakdown["low_rank_factors"] += factor.numel() * factor.element_size()
        else:
            record_classes["llm_gptq"] += 1
            # The preregistered warmup freezes one per-channel LLM A4 scale.
            breakdown["activation_scales"] += in_features * 2

        for value in record.values():
            if torch.is_tensor(value):
                artifact_tensor_bytes += value.numel() * value.element_size()

    require(record_classes == {"llm_gptq": 84, "dit_per_step": 96}, "record-class drift")
    component_bytes = sum(breakdown.values())
    return {
        "schema_version": 1,
        "kind": "omega_qvla_theoretical_packed_storage_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": "Omega-QVLA W4A4",
        "task_set": "atomic_seen",
        "scope": "GR00T LLM+DiT Linear weights/biases (QuantVLA Tables 1/2)",
        "estimate_kind": "theoretical tightly-packed deployment component storage",
        "scope_linear_layers": len(scope_layers),
        "record_classes": record_classes,
        "quantized_layers": len(records),
        "quantized_parameters": quantized_parameters,
        "fp16_reference": {
            "component_bytes": fp16_bytes,
            "component_gib": fp16_bytes / 2**30,
            "weight_bytes": fp16_weight_bytes,
            "bias_bytes": fp16_bias_bytes,
        },
        "packed": {
            "component_bytes": component_bytes,
            "component_gib": component_bytes / 2**30,
            "compression_ratio": fp16_bytes / component_bytes,
            "relative_savings": 1.0 - component_bytes / fp16_bytes,
            "breakdown_bytes": breakdown,
        },
        "evaluation_artifact": {
            "pack_bytes": pack_path.stat().st_size,
            "pack_gib": pack_path.stat().st_size / 2**30,
            "tensor_payload_bytes": artifact_tensor_bytes,
            "note": (
                "The evaluation pack stores dequantized FP16 weights and dense rotations; "
                "its physical size is not used as a packed-W4 deployment claim."
            ),
        },
        "representation": {
            "weight_bits": 4,
            "activation_bits": 4,
            "gptq_scale_dtype": "float16",
            "rotation_dtype": "float16",
            "rotation_block_in": input_block,
            "rotation_block_out": output_block,
            "dit_permutation_dtype": "int32",
            "dit_activation_scale_dtype": "float32",
            "llm_activation_scale_dtype": "float16",
            "low_rank": 0,
        },
        "excluded": [
            "vision modules",
            "activation tensors",
            "CUDA/runtime workspaces",
            "dequantized evaluation caches",
            "simulator",
        ],
        "evidence": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "pack": {"path": str(pack_path), "sha256": sha256_file(pack_path)},
            "pack_attestation": {
                "path": str(attestation_path),
                "sha256": sha256_file(attestation_path),
            },
            "calibration_manifest": {
                "path": str(calibration_path),
                "sha256": sha256_file(calibration_path),
            },
            "builder": {
                "path": str(REPO_ROOT / "scripts/tools/omega_qvla_robocasa365_build.py"),
                "sha256": sha256_file(REPO_ROOT / "scripts/tools/omega_qvla_robocasa365_build.py"),
            },
            "calculator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = calculate(Path(args.manifest), Path(args.summary))
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
