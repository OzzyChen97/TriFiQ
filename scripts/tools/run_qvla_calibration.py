#!/usr/bin/env python3
"""Collect released-QVLA Hessian proxies and build real mixed-row packs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT / "code"))

from qvla_actquant.core import (  # noqa: E402
    HessianProxy,
    canonical_hash,
    fixed_parameter_inventory,
    qvla_compute_proxies,
    qvla_greedy_allocate,
    qvla_pack_model,
    sha256_file,
    target_inventory,
)
from qvla_actquant.model_adapters import (  # noqa: E402
    iter_frame_batches,
    load_frozen,
    load_gr00t_policy,
    load_pi05_policy,
    prepare_gr00t_batch,
    prepare_pi05_batch,
    run_backbone_only,
)


QVLA_COMMIT = "26cc4821a3be4c003d09d3c7997b38db2a347982"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def implementation_record() -> dict[str, Any]:
    paths = [
        REPO_ROOT / "code/qvla_actquant/core.py",
        REPO_ROOT / "code/qvla_actquant/model_adapters.py",
        REPO_ROOT / "code/qvla_actquant/runtime.py",
        REPO_ROOT / "code/gr00t/model/policy.py",
        REPO_ROOT / "code/pi05/openpi/src/openpi/policies/policy_config.py",
        REPO_ROOT / "code/pi05/openpi/src/openpi/models_pytorch/gemma_pytorch.py",
        REPO_ROOT / "code/pi05/openpi/scripts/serve_pi05_quant_policy.py",
        REPO_ROOT / "scripts/inference_service.py",
        REPO_ROOT / "scripts/tools/run_qvla_calibration.py",
    ]
    records = [
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
        for path in paths
    ]
    return {"files": records, "sha256": canonical_hash(records)}


def load_policy(model_family: str, checkpoint: Path, device: str):
    if model_family == "gr00t":
        return load_gr00t_policy(checkpoint, device)
    return load_pi05_policy(checkpoint, device)


def model_of(model_family: str, policy):
    return policy.model if model_family == "gr00t" else policy._model


def require_bf16_model(model_family: str, model: torch.nn.Module) -> dict[str, Any]:
    """Bind QVLA calibration and decoded weights to explicit BF16 GEMMs."""
    variable = "GR00T_MODEL_DTYPE" if model_family == "gr00t" else "OPENPI_MODEL_DTYPE"
    requested = os.environ.get(variable, "")
    normalized = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
    }.get(requested.strip().lower())
    if normalized != "bfloat16":
        raise ValueError(f"QVLA requires explicit {variable}=bfloat16; got {requested!r}")
    by_dtype: dict[str, int] = {}
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            dtype = str(module.weight.dtype).removeprefix("torch.")
            by_dtype[dtype] = by_dtype.get(dtype, 0) + 1
    if not by_dtype or set(by_dtype) != {"bfloat16"}:
        raise ValueError(f"QVLA requires all Linear/Conv weights in BF16; observed {by_dtype}")
    return {
        "environment_variable": variable,
        "requested": requested,
        "resolved": "bfloat16",
        "linear_conv_layers_by_weight_dtype": by_dtype,
        "strict_all_linear_conv_bf16": True,
    }


def input_group(name: str) -> str:
    """Deduplicate only projections whose released architectures share input."""
    replacements = (
        (".self_attn.q_proj", ".self_attn.{qkv}_proj"),
        (".self_attn.k_proj", ".self_attn.{qkv}_proj"),
        (".self_attn.v_proj", ".self_attn.{qkv}_proj"),
        (".mlp.gate_proj", ".mlp.{gate_up}_proj"),
        (".mlp.up_proj", ".mlp.{gate_up}_proj"),
        (".self_attn.qkv", ".self_attn.qkv"),
    )
    for suffix, replacement in replacements:
        if suffix in name:
            return name.replace(suffix, replacement)
    return name


def lpt_shards(inventory: list[dict[str, Any]], count: int) -> list[list[str]]:
    bins: list[list[str]] = [[] for _ in range(count)]
    costs = [0] * count
    for row in sorted(
        inventory,
        key=lambda item: (-math.prod(int(value) for value in item["shape"][1:]) ** 2, item["name"]),
    ):
        index = min(range(count), key=lambda i: (costs[i], i))
        bins[index].append(row["name"])
        costs[index] += math.prod(int(value) for value in row["shape"][1:]) ** 2
    return bins


def collect(args) -> None:
    started = time.time()
    frozen = load_frozen(args.frozen_manifest)
    if frozen["model"] != args.model_family:
        raise ValueError("frozen calibration/model family mismatch")
    policy = load_policy(args.model_family, args.checkpoint, args.device)
    model = model_of(args.model_family, policy)
    model.eval()
    precision = require_bf16_model(args.model_family, model)
    inventory = target_inventory(model, args.model_family)
    if not inventory:
        raise RuntimeError("QVLA target inventory is empty")
    shards = lpt_shards(inventory, args.shard_count)
    selected = shards[args.shard_index]
    modules = dict(model.named_modules())
    groups: dict[str, list[str]] = {}
    for name in selected:
        groups.setdefault(input_group(name), []).append(name)
    proxies: dict[str, HessianProxy] = {}
    representative: dict[str, str] = {}
    handles = []
    for group, names in sorted(groups.items()):
        name = sorted(names)[0]
        representative[group] = name
        proxy = HessianProxy(modules[name], device=args.device)
        proxies[group] = proxy

        def hook(_module, inputs, _output, *, target=proxy):
            value = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            target.add_batch(value)

        handles.append(modules[name].register_forward_hook(hook))
    processed = 0
    try:
        with torch.inference_mode():
            for batch_index, frames in enumerate(
                iter_frame_batches(frozen, subset="qvla", batch_size=args.batch_size)
            ):
                if args.model_family == "gr00t":
                    prepared = prepare_gr00t_batch(policy, frames, include_actions=False)
                else:
                    prepared, _ = prepare_pi05_batch(policy, frames, include_actions=False)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    run_backbone_only(args.model_family, policy, prepared)
                processed += len(frames)
                if batch_index % args.log_every == 0:
                    print(
                        f"[qvla] shard={args.shard_index}/{args.shard_count} "
                        f"frames={processed}/{frozen['qvla_frame_count']} groups={len(groups)}",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()
    if processed != int(frozen["qvla_frame_count"]):
        raise ValueError(f"QVLA frame count mismatch: {processed}")
    output_proxies: dict[str, dict[int, torch.Tensor]] = {}
    group_stats = {}
    for group in sorted(groups):
        proxy = proxies[group]
        diagonal = proxy.diag_hinv(percdamp=args.percdamp)
        group_stats[group] = {
            "representative": representative[group],
            "members": sorted(groups[group]),
            "columns": proxy.columns,
            "nsamples": proxy.nsamples,
            "hessian_diagnostics": proxy.last_diagnostics,
        }
        for name in groups[group]:
            output_proxies[name] = {
                bit: value.detach().cpu()
                for bit, value in qvla_compute_proxies(modules[name], diagonal).items()
            }
        del diagonal
        torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "kind": "qvla_proxy_shard",
        "upstream_commit": QVLA_COMMIT,
        "model_family": args.model_family,
        "model_unit": frozen["model_unit"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "frozen_manifest": str(args.frozen_manifest),
        "frozen_manifest_sha256": sha256_file(args.frozen_manifest),
        "calibration_source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "selection_seed": 42,
        "test_results_used": False,
        "model_precision": precision,
        "activation_compute_dtype": "bfloat16",
        "frames": processed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_inventory_sha256": canonical_hash(inventory),
        "selected_names": sorted(selected),
        "selected_names_sha256": canonical_hash(sorted(selected)),
        "input_groups": group_stats,
        "proxies": output_proxies,
        "wall_seconds": time.time() - started,
    }
    atomic_torch(args.output, payload)
    sidecar = {key: value for key, value in payload.items() if key != "proxies"}
    sidecar.update({"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha256_file(args.output)})
    atomic_json(Path(str(args.output) + ".json"), sidecar)
    print(json.dumps(sidecar, indent=2, sort_keys=True))


def merge_pack(args) -> None:
    frozen = load_frozen(args.frozen_manifest)
    expected_frame_count = int(frozen["qvla_frame_count"])
    merged: dict[str, dict[int, torch.Tensor]] = {}
    inventory_sha = None
    shard_records = []
    expected_indices = set(range(args.shard_count))
    observed_indices = set()
    for path in sorted(args.shards):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("kind") != "qvla_proxy_shard" or payload["model_family"] != args.model_family:
            raise ValueError(f"invalid proxy shard: {path}")
        if payload["upstream_commit"] != QVLA_COMMIT or payload["frames"] != expected_frame_count:
            raise ValueError(f"proxy semantic/frame drift: {path}")
        if payload["checkpoint_sha256"] != args.checkpoint_sha256:
            raise ValueError(f"checkpoint SHA drift: {path}")
        if payload["frozen_manifest_sha256"] != sha256_file(args.frozen_manifest):
            raise ValueError(f"calibration SHA drift: {path}")
        if (
            payload.get("activation_compute_dtype") != "bfloat16"
            or (payload.get("model_precision") or {}).get("strict_all_linear_conv_bf16") is not True
        ):
            raise ValueError(f"QVLA precision drift: {path}")
        if int(payload["shard_count"]) != args.shard_count:
            raise ValueError(f"shard-count drift: {path}")
        index = int(payload["shard_index"])
        if index in observed_indices:
            raise ValueError(f"duplicate proxy shard index: {index}")
        observed_indices.add(index)
        inventory_sha = inventory_sha or payload["full_inventory_sha256"]
        if inventory_sha != payload["full_inventory_sha256"]:
            raise ValueError("target inventory differs across QVLA shards")
        for name, records in payload["proxies"].items():
            if name in merged:
                raise ValueError(f"duplicate QVLA target: {name}")
            merged[name] = {int(bit): value.float() for bit, value in records.items()}
        shard_records.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    if observed_indices != expected_indices:
        raise ValueError(f"missing QVLA shards: {sorted(expected_indices - observed_indices)}")
    assignments, allocation_stats = qvla_greedy_allocate(merged, target_average_bits=4.0)
    if allocation_stats["final_average_bits"] > 4.0:
        raise AssertionError("QVLA allocation exceeded Wavg4")
    policy = load_policy(args.model_family, args.checkpoint, args.device)
    model = model_of(args.model_family, policy)
    precision = require_bf16_model(args.model_family, model)
    inventory = target_inventory(model, args.model_family)
    if canonical_hash(inventory) != inventory_sha or set(assignments) != {row["name"] for row in inventory}:
        raise ValueError("live QVLA target inventory differs from proxy shards")
    assignment_payload = {
        "schema_version": 1,
        "kind": "qvla_gate_allocation",
        "bits": [0, 2, 4, 8, 16],
        "assignments": assignments,
        "stats": allocation_stats,
        "proxy_shards": shard_records,
    }
    atomic_json(args.allocation_output, assignment_payload)
    pack_metadata = {
        "model_unit": frozen["model_unit"],
        "checkpoint_sha256": args.checkpoint_sha256,
        "calibration_manifest_sha256": sha256_file(args.frozen_manifest),
        "calibration_source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "target_inventory_sha256": inventory_sha,
        "proxy_shards_sha256": canonical_hash(shard_records),
        "allocation_sha256": sha256_file(args.allocation_output),
        "upstream_commit": QVLA_COMMIT,
        "test_results_used": False,
        "model_precision": precision,
        "activation_compute_dtype": "bfloat16",
    }
    pack = qvla_pack_model(
        model,
        assignments,
        args.pack_output,
        model_family=args.model_family,
        artifact_metadata=pack_metadata,
    )
    baseline_bytes = sum(int(row["parameters"]) * 2 for row in inventory)
    fixed_inventory = fixed_parameter_inventory(model, args.model_family)
    fixed_bytes = sum(int(row["bytes"]) for row in fixed_inventory)
    result = {
        "schema_version": 1,
        "kind": "qvla_frozen_pack_manifest",
        "method": "qvla",
        "display_name": "QVLA-code Wavg4/A16",
        "model_family": args.model_family,
        "model_unit": frozen["model_unit"],
        "checkpoint": str(args.checkpoint),
        **pack_metadata,
        "assignment": {"path": str(args.allocation_output), "sha256": sha256_file(args.allocation_output)},
        "pack": {"path": str(args.pack_output), "sha256": pack["sha256"], "bytes": pack["bytes"]},
        "actual_average_channel_bits": allocation_stats["final_average_bits"],
        "bit_histogram": allocation_stats["bit_histogram"],
        "target_fp16_baseline_bytes": baseline_bytes,
        "target_pack_payload_bytes": pack["target_payload_bytes"],
        "target_static_compression_ratio": baseline_bytes / pack["target_payload_bytes"],
        "fixed_precision_inventory_sha256": canonical_hash(fixed_inventory),
        "exclusion_inventory_sha256": canonical_hash(fixed_inventory),
        "fixed_precision_parameter_count": sum(int(row["parameters"]) for row in fixed_inventory),
        "fixed_precision_native_bytes": fixed_bytes,
        "full_component_native_baseline_bytes": baseline_bytes + fixed_bytes,
        "fixed_precision_pack_payload_bytes": pack["fixed_payload_bytes"],
        "full_component_static_bytes": pack["bytes"],
        "full_component_static_compression_ratio": (baseline_bytes + fixed_bytes) / pack["bytes"],
        "static_size_scope": "real_complete_qpack_with_low_bit_targets_and_native_fixed_parameters",
        "precision_semantics": {
            "target_weights": "released_QVLA_per_output_channel_symmetric_0_2_4_8_16",
            "target_budget": "average_bit_per_output_channel_le_4",
            "activations": "A16_no_activation_quantization",
            "excluded_parameters": "native_checkpoint_precision_encoded_in_pack",
            "load": "decode_once_to_live_native_dtype",
        },
        "runtime": "packed_rows_dequantized_once_to_native_dtype",
        "implementation": implementation_record(),
        "runtime_memory_claim_allowed": False,
        "latency_claim_allowed": False,
    }
    atomic_json(args.manifest_output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--model-family", choices=("gr00t", "pi05"), required=True)
    collect_parser.add_argument("--checkpoint", type=Path, required=True)
    collect_parser.add_argument("--checkpoint-sha256", required=True)
    collect_parser.add_argument("--frozen-manifest", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--shard-index", type=int, required=True)
    collect_parser.add_argument("--shard-count", type=int, required=True)
    collect_parser.add_argument("--device", default="cuda:0")
    collect_parser.add_argument("--batch-size", type=int, default=1)
    collect_parser.add_argument("--percdamp", type=float, default=0.01)
    collect_parser.add_argument("--log-every", type=int, default=25)
    merge = sub.add_parser("merge-pack")
    merge.add_argument("--model-family", choices=("gr00t", "pi05"), required=True)
    merge.add_argument("--checkpoint", type=Path, required=True)
    merge.add_argument("--checkpoint-sha256", required=True)
    merge.add_argument("--frozen-manifest", type=Path, required=True)
    merge.add_argument("--shards", type=Path, nargs="+", required=True)
    merge.add_argument("--shard-count", type=int, required=True)
    merge.add_argument("--allocation-output", type=Path, required=True)
    merge.add_argument("--pack-output", type=Path, required=True)
    merge.add_argument("--manifest-output", type=Path, required=True)
    merge.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in ("checkpoint", "frozen_manifest", "output", "allocation_output", "pack_output", "manifest_output"):
        if hasattr(args, name) and getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.command == "collect":
        if not 0 <= args.shard_index < args.shard_count:
            raise SystemExit("invalid shard")
        collect(args)
    else:
        args.shards = [path.resolve() for path in args.shards]
        merge_pack(args)


if __name__ == "__main__":
    main()
