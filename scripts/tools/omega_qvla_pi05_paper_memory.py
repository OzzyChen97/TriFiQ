#!/usr/bin/env python3
"""Audit theoretical packed storage for pi0.5 Omega-QVLA on RoboCasa365."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = re.compile(
    r".*paligemma_with_expert\.(?:gemma_expert\.model|paligemma\.model\.language_model)"
    r"\.layers\.\d+\..*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def verified_path(
    record: dict[str, Any], label: str, *, verify_content_hash: bool = True
) -> Path:
    path = resolve(record["path"])
    require(path.is_file(), f"missing {label}: {path}")
    if verify_content_hash:
        require(sha256_file(path) == record["sha256"], f"{label} SHA drift: {path}")
    return path


def rotation_block_bytes(features: int, block: int) -> int:
    return sum(
        min(block, features - start) ** 2 * 2
        for start in range(0, features, block)
    )


def read_shapes(checkpoint: Path) -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open

    shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            shapes[key] = tuple(int(value) for value in handle.get_slice(key).get_shape())
    return shapes


def load_pack(path: Path) -> dict[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        value = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(value, dict), "Omega-QVLA pack is not a dictionary")
    return value


def calculate(manifest_path: Path, summary_path: Path) -> dict[str, Any]:
    import torch

    manifest_path = resolve(manifest_path)
    summary_path = resolve(summary_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    task_set = str(manifest.get("task_set"))
    require(task_set in {"atomic_seen", "composite_seen", "composite_unseen"}, "unknown task set")
    require(summary.get("task_set") == task_set, "summary task set drift")
    require(summary.get("complete") is True, "task-set summary is incomplete")
    require(summary.get("manifest_sha256") == sha256_file(manifest_path), "summary link drift")
    result = (summary.get("configs") or {}).get("omega_qvla_w4a4") or {}
    expected_episodes = len(manifest["tasks"]) * len(manifest["seeds"])
    require(result.get("episodes") == expected_episodes, "task-set coverage drift")
    require(result.get("formal_failures") == 0, "task set has failures")

    # The 35-GiB dequantized pack was content-hashed during build and again by
    # every formal server. Avoid a third full sequential read while evaluation
    # workers are active; cross-check the two frozen attestations below.
    pack_path = verified_path(manifest["pack"], "pack", verify_content_hash=False)
    attestation_path = verified_path(manifest["pack_attestation"], "pack attestation")
    calibration_path = verified_path(manifest["calibration_manifest"], "calibration manifest")
    checkpoint_path = verified_path(manifest["checkpoint"], "checkpoint")
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    require(attestation.get("pack_sha256") == manifest["pack"]["sha256"], "pack link drift")
    require(
        attestation.get("calibration_manifest_sha256") == sha256_file(calibration_path),
        "calibration link drift",
    )
    recipe = calibration["recipe"]
    require(recipe.get("weight_bits") == 4, "recipe is not W4")
    require(recipe.get("activation_bits") == 4, "recipe is not A4")
    input_block = int(recipe["duquant_block_size"])
    output_block = int(recipe["duquant_block_out"])
    gptq_block = int(recipe["gptq_block_size"])
    flow_steps = int(recipe["denoising_steps"])

    shapes = read_shapes(checkpoint_path)
    scope = sorted(name for name, shape in shapes.items() if len(shape) == 2 and TARGET.fullmatch(name))
    require(len(scope) == 252, f"pi0.5 Omega scope drift: {len(scope)}")
    fp16_weight_bytes = sum(math.prod(shapes[name]) * 2 for name in scope)
    fp16_bias_bytes = sum(
        math.prod(shapes[bias]) * 2
        for name in scope
        if (bias := name[:-7] + ".bias") in shapes
    )
    fp16_bytes = fp16_weight_bytes + fp16_bias_bytes

    pack = load_pack(pack_path)
    records = {name: value for name, value in pack.items() if name != "__meta__"}
    expected = {name[:-7] for name in scope}
    require(set(records) == expected, "pack records do not exactly cover the 252-layer scope")
    classes = {
        "paligemma_gptq": sum(".paligemma." in name for name in records),
        "expert_rtn": sum(".gemma_expert." in name for name in records),
    }
    require(classes == {"paligemma_gptq": 126, "expert_rtn": 126}, "record-class drift")

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
    artifact_tensor_bytes = 0
    quantized_parameters = 0
    for name, record in sorted(records.items()):
        require(isinstance(record, dict), f"invalid record: {name}")
        out_features, in_features = shapes[f"{name}.weight"]
        weight = record.get("weight_res_q", record.get("baseline_q"))
        require(torch.is_tensor(weight), f"missing quantized weight: {name}")
        require(tuple(weight.shape) == (out_features, in_features), f"weight shape drift: {name}")
        require(int(record.get("weight_bits", 0)) == 4, f"non-W4 record: {name}")
        require(int(record.get("a_bits", 0)) == 4, f"non-A4 record: {name}")
        quantized_parameters += out_features * in_features
        breakdown["packed_weights"] += math.ceil(out_features * in_features / 2)
        breakdown["weight_scales"] += out_features * math.ceil(in_features / gptq_block) * 2

        input_rotation = record.get("duquant_rotation")
        output_rotation = record.get("duquant_rotation_out")
        permutation = record.get("duquant_rotation_perm")
        table = record.get("act_scale_table")
        require(torch.is_tensor(input_rotation) and tuple(input_rotation.shape) == (in_features, in_features), f"input rotation drift: {name}")
        require(torch.is_tensor(output_rotation) and tuple(output_rotation.shape) == (out_features, out_features), f"output rotation drift: {name}")
        require(input_rotation.dtype == output_rotation.dtype == torch.float16, f"rotation dtype drift: {name}")
        require(torch.is_tensor(permutation) and tuple(permutation.shape) == (in_features,), f"permutation drift: {name}")
        expected_steps = flow_steps if ".gemma_expert." in name else 1
        require(torch.is_tensor(table) and tuple(table.shape) == (expected_steps, in_features), f"A4 table drift: {name}")
        require(table.dtype == torch.float32, f"A4 table dtype drift: {name}")
        breakdown["input_rotations"] += rotation_block_bytes(in_features, input_block)
        breakdown["output_rotations"] += rotation_block_bytes(out_features, output_block)
        breakdown["permutation_indices"] += in_features * 4
        breakdown["activation_scales"] += table.numel() * table.element_size()
        for key in ("lowrank_A", "lowrank_B"):
            factor = record.get(key)
            require(torch.is_tensor(factor) and factor.numel() == 0, f"nonzero low-rank factor: {name}/{key}")
        for value in record.values():
            if torch.is_tensor(value):
                artifact_tensor_bytes += value.numel() * value.element_size()

    packed_bytes = sum(breakdown.values())
    runtime_files = [resolve(server["path"]) for server in manifest["servers"]]
    runtime_hashes = []
    for path in runtime_files:
        runtime = json.loads(path.read_text(encoding="utf-8"))["openpi_runtime"]["omega_qvla"]
        require(runtime.get("wrapped_layers") == 252 and runtime.get("pack_records") == 252, f"runtime layer drift: {path}")
        require(runtime.get("weight_bits") == 4 and runtime.get("activation_bits") == 4, f"runtime bit drift: {path}")
        require(runtime.get("pack_sha256") == manifest["pack"]["sha256"], f"runtime pack SHA drift: {path}")
        runtime_hashes.append(sha256_file(path))

    return {
        "schema_version": 1,
        "kind": "omega_qvla_pi05_theoretical_packed_storage_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": "Omega-QVLA W4A4",
        "task_set": task_set,
        "scope": "pi0.5 PaliGemma+expert target Linear weights/biases",
        "estimate_kind": "theoretical tightly-packed deployment component storage",
        "scope_linear_layers": len(scope),
        "record_classes": classes,
        "quantized_layers": len(records),
        "quantized_parameters": quantized_parameters,
        "fp16_reference": {
            "component_bytes": fp16_bytes,
            "component_gib": fp16_bytes / 2**30,
            "weight_bytes": fp16_weight_bytes,
            "bias_bytes": fp16_bias_bytes,
        },
        "packed": {
            "component_bytes": packed_bytes,
            "component_gib": packed_bytes / 2**30,
            "compression_ratio": fp16_bytes / packed_bytes,
            "relative_savings": 1.0 - packed_bytes / fp16_bytes,
            "breakdown_bytes": breakdown,
        },
        "evaluation_artifact": {
            "pack_bytes": pack_path.stat().st_size,
            "pack_gib": pack_path.stat().st_size / 2**30,
            "tensor_payload_bytes": artifact_tensor_bytes,
            "note": "The evaluation pack stores dequantized FP16 weights and dense rotations; it is not the packed-deployment size.",
        },
        "representation": {
            "weight_bits": 4,
            "activation_bits": 4,
            "gptq_scale_dtype": "float16",
            "rotation_dtype": "float16",
            "rotation_block_in": input_block,
            "rotation_block_out": output_block,
            "permutation_deployment_dtype": "int32",
            "activation_scale_dtype": "float32",
            "low_rank": 0,
        },
        "excluded": ["vision modules", "activations", "CUDA/runtime workspaces", "dequantized evaluation caches", "simulator"],
        "evidence": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "pack": {"path": str(pack_path), "sha256": manifest["pack"]["sha256"]},
            "pack_attestation": {"path": str(attestation_path), "sha256": sha256_file(attestation_path)},
            "calibration_manifest": {"path": str(calibration_path), "sha256": sha256_file(calibration_path)},
            "runtime_attestations": [{"path": str(path), "sha256": digest} for path, digest in zip(runtime_files, runtime_hashes, strict=True)],
            "calculator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = calculate(Path(args.manifest), Path(args.summary))
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
