#!/usr/bin/env python3
"""ActQuant v3 HSIC allocation, α=1 AMF Fisher, and GGUF pack pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import time
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT / "code"))

from qvla_actquant.core import (  # noqa: E402
    actquant_greedy_l2_allocate,
    canonical_hash,
    classify_target,
    fixed_parameter_inventory,
    hsic_reference_modules,
    sha256_file,
    target_inventory,
)
from qvla_actquant.gguf_artifacts import (  # noqa: E402
    block_padded_parameters,
    export_pi05_official_ggufs,
    inspect_quantized_gguf,
    merge_gguf_tensor_overrides,
    merge_pi05_official_llm,
    pi05_official_tensor_maps,
    quantize_component,
    quantize_pi05_official_llm,
    tensor_name_map,
    write_component_gguf,
    write_fisher_gguf,
    write_fixed_component_gguf,
)
from qvla_actquant.model_adapters import (  # noqa: E402
    flow_loss,
    iter_frame_batches,
    load_frozen,
    load_gr00t_policy,
    load_pi05_policy,
    prepare_gr00t_batch,
    prepare_pi05_batch,
    run_backbone_only,
)


ACTQUANT_COMMIT = "b64791125070652fe6b554e244fe809c79ef5246"


def job_implementation_sha256() -> str:
    value = os.environ.get("QVLA_ACTQUANT_JOB_IMPLEMENTATION_SHA256", "")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(
            "ActQuant job is not bound to a valid implementation SHA256"
        )
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def implementation_record() -> dict[str, Any]:
    paths = [
        REPO_ROOT / "code/qvla_actquant/core.py",
        REPO_ROOT / "code/qvla_actquant/model_adapters.py",
        REPO_ROOT / "code/qvla_actquant/gguf_artifacts.py",
        REPO_ROOT / "code/qvla_actquant/runtime.py",
        REPO_ROOT / "code/gr00t/model/policy.py",
        REPO_ROOT / "code/pi05/openpi/src/openpi/policies/policy_config.py",
        REPO_ROOT / "code/pi05/openpi/src/openpi/models_pytorch/gemma_pytorch.py",
        REPO_ROOT / "code/pi05/openpi/scripts/serve_pi05_quant_policy.py",
        REPO_ROOT / "scripts/inference_service.py",
        REPO_ROOT / "scripts/tools/run_actquant_calibration.py",
        REPO_ROOT / "patches/qvla_actquant/actquant_component_quantizer.patch",
    ]
    records = [
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
        for path in paths
    ]
    return {"files": records, "sha256": canonical_hash(records)}


def load_policy(model_family: str, checkpoint: Path, device: str):
    return load_gr00t_policy(checkpoint, device) if model_family == "gr00t" else load_pi05_policy(checkpoint, device)


def model_of(model_family: str, policy):
    return policy.model if model_family == "gr00t" else policy._model


def require_fp16_model(model_family: str, model: torch.nn.Module) -> dict[str, Any]:
    """Fail closed unless every GEMM/convolution weight is actually FP16."""
    variable = "GR00T_MODEL_DTYPE" if model_family == "gr00t" else "OPENPI_MODEL_DTYPE"
    requested = os.environ.get(variable, "")
    normalized = {
        "fp16": "float16",
        "float16": "float16",
    }.get(requested.strip().lower())
    if normalized != "float16":
        raise ValueError(
            f"ActQuant requires explicit {variable}=float16; got {requested!r}"
        )
    by_dtype: dict[str, int] = {}
    layer_count = 0
    for module in model.modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            continue
        layer_count += 1
        dtype = str(module.weight.dtype).removeprefix("torch.")
        by_dtype[dtype] = by_dtype.get(dtype, 0) + 1
    if layer_count == 0 or set(by_dtype) != {"float16"}:
        raise ValueError(
            f"ActQuant requires all Linear/Conv weights in FP16; observed {by_dtype}"
        )
    return {
        "environment_variable": variable,
        "requested": requested,
        "resolved": "float16",
        "linear_conv_layers_by_weight_dtype": by_dtype,
        "strict_all_linear_conv_fp16": True,
    }


def prepare_batch(model_family: str, policy, frames, *, include_actions: bool):
    if model_family == "gr00t":
        return prepare_gr00t_batch(policy, frames, include_actions=include_actions), None
    return prepare_pi05_batch(policy, frames, include_actions=include_actions)


def install_pi05_batched_image_prefix(model: torch.nn.Module) -> dict[str, Any]:
    """Fuse independent SigLIP camera calls into one camera-major batch."""
    original = model.embed_prefix

    def batched_embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        if not images:
            return original(images, img_masks, lang_tokens, lang_masks)
        batch_sizes = [int(image.shape[0]) for image in images]
        if (
            len(set(batch_sizes)) != 1
            or len({tuple(image.shape[1:]) for image in images}) != 1
        ):
            return original(images, img_masks, lang_tokens, lang_masks)

        def image_embed_func(value):
            return self.paligemma_with_expert.embed_image(value)

        combined_emb = self._apply_checkpoint(
            image_embed_func, torch.cat(images, dim=0)
        )
        image_embs = list(combined_emb.split(batch_sizes, dim=0))
        embs = list(image_embs)
        pad_masks = [
            mask[:, None].expand(batch_sizes[index], image_emb.shape[1])
            for index, (mask, image_emb) in enumerate(
                zip(img_masks, image_embs, strict=True)
            )
        ]
        image_tokens = sum(int(image_emb.shape[1]) for image_emb in image_embs)

        def lang_embed_func(value):
            embedded = self.paligemma_with_expert.embed_language_tokens(value)
            return embedded * math.sqrt(embedded.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        prefix_embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        attention_pattern = torch.zeros(
            image_tokens + int(lang_emb.shape[1]),
            dtype=torch.bool,
            device=prefix_pad_masks.device,
        )
        prefix_att_masks = attention_pattern[None, :].expand(
            prefix_pad_masks.shape[0], -1
        )
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    model.embed_prefix = types.MethodType(batched_embed_prefix, model)
    return {
        "name": "pi05_camera_major_siglip_batch",
        "camera_order_restored": True,
        "eval_batch_separable": True,
        "hsic_camera_reduction": "mean_matching_released_repeated_hook_mean",
    }


def pool_activation_tensors(
    value: torch.Tensor,
    batch_size: int,
    *,
    nonfinite_counter: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Pool on device while preserving released repeated-camera call order."""
    value = value.detach()
    if value.ndim < 2:
        raise ValueError(f"cannot pool activation {value.shape}")
    if value.ndim == 2:
        if value.shape[0] == batch_size:
            pooled = value
        elif value.shape[0] % batch_size == 0:
            pooled = value.reshape(batch_size, -1, value.shape[-1]).mean(dim=1)
        else:
            raise ValueError(f"cannot recover batch={batch_size} from activation {value.shape}")
    elif value.ndim == 3:
        pooled = value.mean(dim=1)
    elif value.ndim == 4:
        pooled = value.mean(dim=(2, 3))
    else:
        pooled = value.flatten(1, value.ndim - 2).mean(dim=1)
    # The fused pi0.5 vision call has camera-major rows [camera, batch].
    # The released path called hooks once per camera and collect_hsic averaged
    # those calls, so this reduction is algebraically the same operation.
    if value.ndim >= 3 and pooled.shape[0] != batch_size:
        if pooled.shape[0] % batch_size:
            raise ValueError(
                f"cannot recover camera-major batch={batch_size} from activation {value.shape}"
            )
        camera_pooled = pooled.reshape(-1, batch_size, pooled.shape[-1])
        # Return one tensor per logical released hook call.  The batch drain
        # converts each to FP16 before NumPy averages cameras below.
        result = [item.to(torch.float16) for item in camera_pooled.unbind(0)]
    else:
        result = [pooled.to(torch.float16)]

    # RoboCasa365 contains a pi0.5 frame whose final PaliGemma MLP produces
    # FP16 overflow.  The released LIBERO collector has no policy for this
    # out-of-domain case and consequently emits a NaN RBF score.  Keep the
    # declared A16/FP16 compute path, but make the pooled statistic total and
    # auditable: NaN maps to zero and signed infinity to the finite extrema of
    # the pooled dtype.  Count replacements on device and transfer the small
    # counter vector only once after collection.
    sanitized = []
    for item in result:
        nonfinite = ~torch.isfinite(item)
        if nonfinite_counter is not None:
            nonfinite_counter.add_(nonfinite.sum(dtype=torch.int64))
        limit = torch.finfo(item.dtype).max
        sanitized.append(
            torch.nan_to_num(item, nan=0.0, posinf=limit, neginf=-limit)
        )
    return sanitized


def lpt_names(inventory: list[dict[str, Any]], count: int, cost) -> list[list[str]]:
    result = [[] for _ in range(count)]
    totals = [0] * count
    for row in sorted(inventory, key=lambda item: (-int(cost(item)), item["name"])):
        index = min(range(count), key=lambda i: (totals[i], i))
        result[index].append(row["name"])
        totals[index] += int(cost(row))
    return result


def _rbf_kernel(values: torch.Tensor) -> torch.Tensor:
    count, width = values.shape
    square = (values * values).sum(dim=1)
    distance = values @ values.T
    distance.mul_(-2).add_(square[:, None]).add_(square[None, :]).clamp_min_(0)
    pairs = min(200_000, count * (count - 1) // 2)
    i = torch.randint(0, count, (pairs * 2,), device=values.device)
    j = torch.randint(0, count, (pairs * 2,), device=values.device)
    mask = i < j
    sample = distance[i[mask][:pairs], j[mask][:pairs]]
    estimate = float(sample.median().item())
    if estimate <= 0:
        estimate = float(sample.mean().item())
    estimate = max(estimate, 1e-2)
    distance.div_(-2.0 * estimate * estimate).exp_()
    return distance


def _center(kernel: torch.Tensor) -> torch.Tensor:
    return kernel.sub_(kernel.mean(dim=0, keepdim=True))


def _trace_product(left: torch.Tensor, right: torch.Tensor, chunk: int = 2048) -> float:
    total = 0.0
    for start in range(0, left.shape[0], chunk):
        stop = min(start + chunk, left.shape[0])
        total += float((left[start:stop] * right.T[start:stop]).sum().item())
    return total


def hsic_scores(
    references: Mapping[str, np.ndarray],
    actions: np.ndarray,
    activations: Mapping[str, tuple[np.ndarray, np.ndarray]],
    families: Mapping[str, str],
    *,
    device: str,
) -> dict[str, Any]:
    count = len(actions)
    denominator = (count - 1) ** 2
    y = torch.from_numpy(actions.astype(np.float32)).to(device)
    ky = _center(y @ y.T)  # released code defaults to a linear action kernel
    kx = {
        family: _center(_rbf_kernel(torch.from_numpy(value.astype(np.float32)).to(device)))
        for family, value in references.items()
    }
    result = {}
    for index, name in enumerate(sorted(activations)):
        zin_np, zout_np = activations[name]
        kin = _center(_rbf_kernel(torch.from_numpy(zin_np.astype(np.float32)).to(device)))
        kout = _center(_rbf_kernel(torch.from_numpy(zout_np.astype(np.float32)).to(device)))
        family_kernel = kx[families[name]]
        x_in = _trace_product(family_kernel, kin) / denominator
        y_in = _trace_product(kin, ky) / denominator
        x_out = _trace_product(family_kernel, kout) / denominator
        y_out = _trace_product(kout, ky) / denominator
        f_in, f_out = -x_in + y_in, -x_out + y_out
        numeric = (x_in, y_in, x_out, y_out, f_in, f_out, f_out - f_in)
        if not all(math.isfinite(value) for value in numeric):
            raise FloatingPointError(f"non-finite HSIC score for {name}: {numeric}")
        result[name] = {
            "hsic_x_in": x_in, "hsic_y_in": y_in, "F_in": f_in,
            "hsic_x_out": x_out, "hsic_y_out": y_out, "F_out": f_out,
            "sens": f_out - f_in,
            "d_in": int(zin_np.shape[1]), "d_out": int(zout_np.shape[1]),
            "family": families[name],
        }
        del kin, kout
        # Reuse the released kernel blocks through PyTorch's allocator.  A
        # per-tensor empty_cache() forced hundreds of host/device sync points
        # without releasing any live HSIC matrices.
        print(f"[actquant][hsic] {index + 1}/{len(activations)} {name} F_out={f_out:.6e}", flush=True)
    return result


def collect_hsic(args) -> None:
    seed_everything()
    frozen = load_frozen(args.frozen_manifest)
    policy = load_policy(args.model_family, args.checkpoint, args.device)
    model = model_of(args.model_family, policy)
    model.eval()
    precision = require_fp16_model(args.model_family, model)
    collector_optimization = (
        install_pi05_batched_image_prefix(model)
        if args.model_family == "pi05"
        else None
    )
    inventory = target_inventory(model, args.model_family)
    shards = lpt_names(
        inventory, args.shard_count,
        lambda row: int(row["shape"][0]) + math.prod(int(value) for value in row["shape"][1:]),
    )
    selected = shards[args.shard_index]
    modules = dict(model.named_modules())
    family_for = {row["name"]: row["family"] for row in inventory}
    references = hsic_reference_modules(model, inventory)
    temporary_in: dict[str, list[torch.Tensor]] = {}
    temporary_out: dict[str, list[torch.Tensor]] = {}
    in_buckets = {name: [] for name in selected}
    out_buckets = {name: [] for name in selected}
    ref_buckets = {family: [] for family in references}
    nonfinite_counters: dict[str, torch.Tensor] = {}
    handles = []
    batch_context = {"size": 0}

    def make_hook(name: str, *, reference_family: str | None = None):
        def counter(key: str, value: torch.Tensor) -> torch.Tensor:
            if key not in nonfinite_counters:
                nonfinite_counters[key] = torch.zeros(
                    (), dtype=torch.int64, device=value.device
                )
            return nonfinite_counters[key]

        def hook(_module, inputs, output):
            inp = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            out = output[0] if isinstance(output, (tuple, list)) else output
            if reference_family is not None:
                key = f"@{reference_family}"
                temporary_in.setdefault(key, []).extend(
                    pool_activation_tensors(
                        inp,
                        batch_context["size"],
                        nonfinite_counter=counter(f"in:{key}", inp),
                    )
                )
            if name in in_buckets:
                temporary_in.setdefault(name, []).extend(
                    pool_activation_tensors(
                        inp,
                        batch_context["size"],
                        nonfinite_counter=counter(f"in:{name}", inp),
                    )
                )
                temporary_out.setdefault(name, []).extend(
                    pool_activation_tensors(
                        out,
                        batch_context["size"],
                        nonfinite_counter=counter(f"out:{name}", out),
                    )
                )
        return hook

    hook_names = set(selected) | set(references.values())
    for name in sorted(hook_names):
        reference_family = next((family for family, ref in references.items() if ref == name), None)
        handles.append(modules[name].register_forward_hook(make_hook(name, reference_family=reference_family)))
    y_buckets = []
    frame_count = 0
    try:
        with torch.no_grad():
            for batch_index, frames in enumerate(iter_frame_batches(frozen, subset="actquant", batch_size=args.batch_size)):
                batch_context["size"] = len(frames)
                prepared, actions = prepare_batch(args.model_family, policy, frames, include_actions=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # HSIC supervises frozen target-backbone activations with
                    # the already archived continuous action chunk.  Action
                    # heads/experts are excluded from the target inventory and
                    # do not feed back into these activations, so executing the
                    # flow-matching loss here only burns compute and samples
                    # unrelated noise/time values.
                    run_backbone_only(
                        args.model_family,
                        policy,
                        prepared,
                        preserve_flow_rng=True,
                        actions=actions,
                    )
                # Hooks retain only already-pooled device tensors.  Pack every
                # variable-width row into one contiguous FP16 buffer so each
                # calibration batch incurs one D2H synchronization rather
                # than roughly two per target tensor.
                entries = []
                for kind, buckets in (("in", temporary_in), ("out", temporary_out)):
                    for key in sorted(buckets):
                        for value in buckets[key]:
                            entries.append((kind, key, tuple(value.shape), value.reshape(-1)))
                if not entries:
                    raise RuntimeError("HSIC hooks produced no pooled activations")
                packed = torch.cat([entry[3] for entry in entries])
                host = packed.cpu().numpy()
                host_buckets: dict[tuple[str, str], list[np.ndarray]] = {}
                offset = 0
                for kind, key, shape, value in entries:
                    count = value.numel()
                    host_buckets.setdefault((kind, key), []).append(
                        host[offset : offset + count].reshape(shape)
                    )
                    offset += count
                if offset != host.size:
                    raise RuntimeError("HSIC packed activation offset drift")
                for name in selected:
                    if ("in", name) not in host_buckets or ("out", name) not in host_buckets:
                        raise RuntimeError(f"HSIC target did not execute: {name}")
                    in_buckets[name].append(
                        np.mean(np.stack(host_buckets.pop(("in", name))), axis=0)
                    )
                    out_buckets[name].append(
                        np.mean(np.stack(host_buckets.pop(("out", name))), axis=0)
                    )
                for family in references:
                    key = f"@{family}"
                    if ("in", key) not in host_buckets:
                        raise RuntimeError(f"HSIC reference did not execute: {family}")
                    ref_buckets[family].append(
                        np.mean(np.stack(host_buckets.pop(("in", key))), axis=0)
                    )
                temporary_in.clear()
                temporary_out.clear()
                if host_buckets:
                    raise RuntimeError("unconsumed HSIC hook calls")
                del entries, packed, host, host_buckets
                y_buckets.append(np.stack([np.asarray(frame["actions"], dtype=np.float32).reshape(-1) for frame in frames]))
                frame_count += len(frames)
                if batch_index % args.log_every == 0:
                    print(f"[actquant][hsic] shard={args.shard_index} frames={frame_count}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    counter_names = sorted(nonfinite_counters)
    counter_values = (
        torch.stack([nonfinite_counters[name] for name in counter_names]).cpu().tolist()
        if counter_names
        else []
    )
    nonfinite_pooled_values = {
        name: int(value)
        for name, value in zip(counter_names, counter_values, strict=True)
        if int(value) != 0
    }
    expected = sum(row["frames"] for row in frozen["episodes"] if row["actquant_subset"])
    if frame_count != expected:
        raise ValueError(f"ActQuant HSIC frame mismatch: {frame_count} != {expected}")
    activations = {
        name: (
            np.concatenate(in_buckets[name]).astype(np.float16),
            np.concatenate(out_buckets[name]).astype(np.float16),
        )
        for name in selected
    }
    reference_arrays = {family: np.concatenate(values).astype(np.float16) for family, values in ref_buckets.items()}
    y = np.concatenate(y_buckets).astype(np.float32)
    y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)
    del policy, model
    gc.collect(); torch.cuda.empty_cache()
    seed_everything()
    scores = hsic_scores(reference_arrays, y, activations, family_for, device=args.device)
    payload = {
        "schema_version": 1,
        "kind": "actquant_hsic_shard",
        "upstream_commit": ACTQUANT_COMMIT,
        "job_implementation_sha256": job_implementation_sha256(),
        "model_family": args.model_family,
        "model_unit": frozen["model_unit"],
        "checkpoint_sha256": args.checkpoint_sha256,
        "calibration_manifest_sha256": sha256_file(args.frozen_manifest),
        "calibration_source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "selection_seed": 42,
        "test_results_used": False,
        "model_precision": precision,
        "activation_compute_dtype": "float16",
        "collector_optimization": collector_optimization,
        "forward_scope": "target_backbone_only_action_chunk_external_supervision",
        "flow_rng_semantics": "released_full_flow_rng_advanced_without_action_head_execution",
        "activation_d2h": "one_packed_fp16_transfer_per_batch",
        "nonfinite_pool_policy": "nan_to_zero_inf_to_fp16_finite_extrema",
        "nonfinite_pooled_values": nonfinite_pooled_values,
        "nonfinite_pooled_total": sum(nonfinite_pooled_values.values()),
        "hidden_kernel": "rbf_released_median",
        "action_kernel": "linear_released_default",
        "lx": 1.0,
        "ly": 1.0,
        "frames": frame_count,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_inventory_sha256": canonical_hash(inventory),
        "selected_names": sorted(selected),
        "reference_modules": references,
        "scores": scores,
    }
    atomic_json(args.output, payload)
    print(json.dumps({**{k: v for k, v in payload.items() if k != "scores"}, "sha256": sha256_file(args.output)}, indent=2))


def collect_fisher(args) -> None:
    seed_everything()
    frozen = load_frozen(args.frozen_manifest)
    policy = load_policy(args.model_family, args.checkpoint, args.device)
    model = model_of(args.model_family, policy)
    model.eval()
    precision = require_fp16_model(args.model_family, model)
    collector_optimization = (
        install_pi05_batched_image_prefix(model)
        if args.model_family == "pi05"
        else None
    )
    inventory = target_inventory(model, args.model_family)
    shards = lpt_names(inventory, args.shard_count, lambda row: int(row["parameters"]))
    selected = shards[args.shard_index]
    modules = dict(model.named_modules())
    parameters = [(name, modules[name].weight) for name in selected]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in parameters:
        parameter.requires_grad_(True)
    # Keep empirical-Fisher sums beside the gradients and transfer exactly
    # once at the end.  The previous per-frame ``grad.float().cpu()`` path
    # moved the selected model shard over PCIe for every sample and made the
    # host/PCIe link the dominant cost.  FP32 add order remains one sample at
    # a time, matching the released empirical grad^2 definition.
    accumulators = {
        name: torch.zeros(parameter.shape, dtype=torch.float32, device=args.device)
        for name, parameter in parameters
    }
    frames_seen = 0
    losses = []
    for batch_index, frames in enumerate(iter_frame_batches(frozen, subset="actquant", batch_size=args.batch_size)):
        prepared, actions = prepare_batch(args.model_family, policy, frames, include_actions=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = flow_loss(args.model_family, policy, prepared, actions)
        gradients = torch.autograd.grad(loss, [parameter for _, parameter in parameters], allow_unused=True)
        for (name, _), gradient in zip(parameters, gradients, strict=True):
            if gradient is not None:
                squared = gradient.detach().float()
                squared.square_()
                accumulators[name].add_(squared)
                del squared
        losses.append(float(loss.detach().item()))
        frames_seen += len(frames)
        del loss, gradients, prepared, actions
        # Keep released blocks in PyTorch's allocator for the next empirical-
        # Fisher sample.  empty_cache() here forced a device synchronization
        # on every frame without reducing any live tensor's memory footprint.
        if batch_index % args.log_every == 0:
            print(f"[actquant][fisher] shard={args.shard_index} frames={frames_seen}", flush=True)
    expected = sum(row["frames"] for row in frozen["episodes"] if row["actquant_subset"])
    if frames_seen != expected:
        raise ValueError(f"ActQuant Fisher frame mismatch: {frames_seen} != {expected}")
    fisher_sum = {name: value.cpu() for name, value in accumulators.items()}
    del accumulators
    payload = {
        "schema_version": 1,
        "kind": "actquant_fisher_shard",
        "upstream_commit": ACTQUANT_COMMIT,
        "job_implementation_sha256": job_implementation_sha256(),
        "model_family": args.model_family,
        "model_unit": frozen["model_unit"],
        "checkpoint_sha256": args.checkpoint_sha256,
        "calibration_manifest_sha256": sha256_file(args.frozen_manifest),
        "calibration_source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "selection_seed": 42,
        "test_results_used": False,
        "model_precision": precision,
        "activation_compute_dtype": "float16",
        "collector_optimization": collector_optimization,
        "alpha": 1.0,
        "loss": "native_continuous_action_flow_matching",
        "fisher_accumulator": "cuda_fp32_single_final_d2h",
        "frames": frames_seen,
        "mean_loss": float(np.mean(losses)),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_inventory_sha256": canonical_hash(inventory),
        "selected_names": sorted(selected),
        "fisher_sum": fisher_sum,
    }
    atomic_torch(args.output, payload)
    sidecar = {key: value for key, value in payload.items() if key != "fisher_sum"}
    sidecar.update({"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha256_file(args.output)})
    atomic_json(Path(str(args.output) + ".json"), sidecar)
    print(json.dumps(sidecar, indent=2))


def layer_signature(name: str, family: str) -> str:
    match = re.search(r"^(.*?\.layers\.\d+)\.", name)
    if match:
        return f"{family}:{match.group(1)}"
    match = re.search(r"^(.*?\.blocks\.\d+)\.", name)
    if match:
        return f"{family}:{match.group(1)}"
    return f"{family}:{name.rsplit('.', 1)[0]}"


def allocate(args) -> None:
    policy = load_policy(args.model_family, args.checkpoint, args.device)
    model = model_of(args.model_family, policy)
    precision = require_fp16_model(args.model_family, model)
    inventory = target_inventory(model, args.model_family)
    inventory_by_name = {row["name"]: row for row in inventory}
    scores = {}
    shard_records = []
    indices = set()
    inventory_sha = canonical_hash(inventory)
    expected_frames = None
    for path in sorted(args.hsic_shards):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") != "actquant_hsic_shard" or payload["upstream_commit"] != ACTQUANT_COMMIT:
            raise ValueError(f"invalid HSIC shard: {path}")
        if payload.get("job_implementation_sha256") != job_implementation_sha256():
            raise ValueError(f"HSIC implementation drift: {path}")
        if payload.get("forward_scope") != "target_backbone_only_action_chunk_external_supervision":
            raise ValueError(f"HSIC forward-scope drift: {path}")
        if payload.get("activation_d2h") != "one_packed_fp16_transfer_per_batch":
            raise ValueError(f"HSIC activation-transfer drift: {path}")
        if payload.get("nonfinite_pool_policy") != "nan_to_zero_inf_to_fp16_finite_extrema":
            raise ValueError(f"HSIC non-finite policy drift: {path}")
        if payload["checkpoint_sha256"] != args.checkpoint_sha256 or payload["full_inventory_sha256"] != inventory_sha:
            raise ValueError(f"HSIC checkpoint/inventory drift: {path}")
        if payload["calibration_manifest_sha256"] != sha256_file(args.frozen_manifest):
            raise ValueError(f"HSIC calibration drift: {path}")
        if (
            payload.get("activation_compute_dtype") != "float16"
            or (payload.get("model_precision") or {}).get("strict_all_linear_conv_fp16") is not True
        ):
            raise ValueError(f"HSIC precision drift: {path}")
        optimization = payload.get("collector_optimization")
        if (
            args.model_family == "pi05"
            and (optimization or {}).get("name") != "pi05_camera_major_siglip_batch"
        ) or (args.model_family != "pi05" and optimization is not None):
            raise ValueError(f"HSIC collector-optimization drift: {path}")
        if int(payload["shard_count"]) != args.shard_count:
            raise ValueError("HSIC shard count drift")
        index = int(payload["shard_index"])
        if index in indices:
            raise ValueError("duplicate HSIC shard")
        indices.add(index)
        expected_frames = expected_frames or int(payload["frames"])
        if expected_frames != int(payload["frames"]):
            raise ValueError("HSIC frame count drift")
        for name, record in payload["scores"].items():
            if name in scores:
                raise ValueError(f"duplicate HSIC tensor: {name}")
            score = float(record["F_out"])
            if not math.isfinite(score):
                raise ValueError(f"non-finite HSIC allocation score for {name}: {path}")
            scores[name] = max(score, 0.0)
        shard_records.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    if indices != set(range(args.shard_count)) or set(scores) != set(inventory_by_name):
        raise ValueError("HSIC shard/target coverage mismatch")
    original_counts = {name: int(row["parameters"]) for name, row in inventory_by_name.items()}
    padded_counts = {name: block_padded_parameters(row["shape"]) for name, row in inventory_by_name.items()}
    signatures = {name: layer_signature(name, row["family"]) for name, row in inventory_by_name.items()}
    signature_ids = {signature: index for index, signature in enumerate(sorted(set(signatures.values())))}
    layer_ids = {name: signature_ids[signature] for name, signature in signatures.items()}
    target_adjusted = 4.0 * sum(original_counts.values()) / sum(padded_counts.values())
    assignments, stats = actquant_greedy_l2_allocate(
        scores, padded_counts, layer_ids,
        target_bpw=target_adjusted, base_type="IQ2_XS", max_type="Q4_K",
    )
    estimated_bits = stats["achieved_bpw"] * sum(padded_counts.values())
    estimated_original_bpw = estimated_bits / sum(original_counts.values())
    if estimated_original_bpw > 4.0 + 1e-8:
        raise AssertionError("ActQuant allocation exceeded the 4.0 BPW storage budget")
    result = {
        "schema_version": 1,
        "kind": "actquant_tensor_allocation",
        "method": "actquant",
        "upstream_commit": ACTQUANT_COMMIT,
        "model_family": args.model_family,
        "model_unit": load_frozen(args.frozen_manifest)["model_unit"],
        "checkpoint_sha256": args.checkpoint_sha256,
        "calibration_manifest_sha256": sha256_file(args.frozen_manifest),
        "calibration_source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "selection_seed": 42,
        "test_results_used": False,
        "model_precision": precision,
        "activation_compute_dtype": "float16",
        "score_key": "max(F_out,0)",
        "target_bpw_original_parameters": 4.0,
        "allocator_target_bpw_padded_parameters": target_adjusted,
        "estimated_actual_bpw_original_parameters": estimated_original_bpw,
        "allocator_stats": stats,
        "target_inventory_sha256": inventory_sha,
        "hsic_shards": shard_records,
        "hsic_sha256": canonical_hash(shard_records),
        "layer_signatures_sha256": canonical_hash(signatures),
        "tensor_mapping": (
            pi05_official_tensor_maps(sorted(assignments))[1]
            if args.model_family == "pi05"
            else tensor_name_map(sorted(assignments))
        ),
        "assignments": assignments,
    }
    result["tensor_mapping_sha256"] = canonical_hash(result["tensor_mapping"])
    atomic_json(args.output, result)
    print(json.dumps({**{k: v for k, v in result.items() if k not in ("assignments", "tensor_mapping")}, "sha256": sha256_file(args.output)}, indent=2))


def chunk_names(inventory: list[dict[str, Any]], max_bytes: int) -> list[list[str]]:
    chunks, current, size = [], [], 0
    for row in sorted(inventory, key=lambda item: item["name"]):
        tensor_bytes = int(row["parameters"]) * 2
        if current and size + tensor_bytes > max_bytes:
            chunks.append(current); current, size = [], 0
        current.append(row["name"]); size += tensor_bytes
    if current:
        chunks.append(current)
    return chunks


def pack_pi05_official(
    args,
    *,
    policy,
    model,
    inventory: list[dict[str, Any]],
    allocation: dict[str, Any],
    fisher: dict[str, torch.Tensor],
    fisher_records: list[dict[str, Any]],
    sample_count: int,
) -> None:
    """Released pi0.5 export → llama-quantize → merge, plus vision overlay."""
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    inventory_by_name = {row["name"]: row for row in inventory}
    fixed_inventory = fixed_parameter_inventory(model, "pi05")
    fixed_bytes = sum(int(row["bytes"]) for row in fixed_inventory)
    names = sorted(inventory_by_name)
    standalone_map, final_map = pi05_official_tensor_maps(names)
    language_names = sorted(standalone_map)
    vision_names = sorted(set(names) - set(language_names))
    if not language_names or not vision_names:
        raise ValueError("pi0.5 official pack requires both PaliGemma and SigLIP targets")

    # Write the released-format Fisher inputs before dropping the live model.
    llm_fisher = output / "pali_llm.amf.gguf"
    vision_fisher = output / "siglip.amf.gguf"
    llm_fisher_record = write_fisher_gguf(
        fisher, language_names, standalone_map, llm_fisher,
        num_samples=sample_count, calibration_sha256=sha256_file(args.frozen_manifest),
    )
    vision_fisher_record = write_fisher_gguf(
        fisher, vision_names, final_map, vision_fisher,
        num_samples=sample_count, calibration_sha256=sha256_file(args.frozen_manifest),
    )

    # Vision has no standalone llama.cpp graph.  Export only the released v.*
    # tensors and quantize them with the graph-free CPU component patch.
    vision_source = output / "siglip.f16.gguf"
    vision_quantized = output / "siglip.actquant.gguf"
    vision_source_record = write_component_gguf(
        model, vision_names, final_map, vision_source, model_family="pi05",
        metadata={"checkpoint_sha256": args.checkpoint_sha256,
                  "inventory_sha256": allocation["target_inventory_sha256"]},
    )
    vision_quant_record = quantize_component(
        vision_source, vision_fisher,
        {name: allocation["assignments"][name] for name in vision_names},
        vision_names, final_map, vision_quantized, output / "siglip.types.tsv",
        log_path=output / "siglip.quantize.log",
    )
    vision_inspected = inspect_quantized_gguf(vision_quantized)
    for name in vision_names:
        if vision_inspected[final_map[name]]["quant_type"] != allocation["assignments"][name]:
            raise ValueError(f"pi0.5 vision quant type drift: {name}")

    # The released exporters load the safetensors checkpoint directly, so the
    # GPU policy can be released before their CPU-only conversion pass.
    del policy, model
    gc.collect(); torch.cuda.empty_cache()
    official = export_pi05_official_ggufs(
        args.checkpoint if args.checkpoint.is_dir() else args.checkpoint.parent,
        output / "official_export",
        tokenizer_path=args.pi05_tokenizer,
        python_executable=args.pi05_export_python,
    )
    llm_quantized = output / "pali_llm.actquant.gguf"
    llm_quant_record = quantize_pi05_official_llm(
        official["standalone_llm"]["path"], llm_fisher,
        {name: allocation["assignments"][name] for name in language_names},
        standalone_map, llm_quantized, log_path=output / "pali_llm.quantize.log",
    )
    llm_inspected = inspect_quantized_gguf(llm_quantized)
    for name in language_names:
        if llm_inspected[standalone_map[name]]["quant_type"] != allocation["assignments"][name]:
            raise ValueError(f"pi0.5 llama-quantize type drift: {name}")
    llm_merged = output / "pi05.llm_merged.gguf"
    llm_merge_record = merge_pi05_official_llm(
        official["unified"]["path"], llm_quantized, llm_merged,
        log_path=output / "pali_llm.merge.log", python_executable=args.pi05_export_python,
    )
    final = output / "pi05.actquant.final.gguf"
    final_record = merge_gguf_tensor_overrides(
        llm_merged, vision_quantized, [final_map[name] for name in vision_names], final,
    )
    final_inspected = inspect_quantized_gguf(final)
    tensor_records = []
    total_payload_bytes = 0
    for name in names:
        gguf_name = final_map[name]
        info = final_inspected.get(gguf_name)
        if info is None or info["quant_type"] != allocation["assignments"][name]:
            raise ValueError(f"pi0.5 final merged GGUF drift: {name}")
        total_payload_bytes += int(info["payload_bytes"])
        tensor_records.append({
            "module_name": name,
            "gguf_name": gguf_name,
            "file_id": "pi05_final",
            "shape": inventory_by_name[name]["shape"],
            "quant_type": allocation["assignments"][name],
            "payload_bytes": int(info["payload_bytes"]),
            "stored_shape": info["shape"],
        })
    original_parameters = sum(int(row["parameters"]) for row in inventory)
    actual_bpw = total_payload_bytes * 8 / original_parameters
    if actual_bpw > 4.0 + 1e-6:
        raise ValueError(f"final pi0.5 GGUF target payload exceeds 4.0 BPW: {actual_bpw}")
    target_fp16 = original_parameters * 2
    bundle = {
        "schema_version": 1,
        "format": "actquant_gguf_bundle_v1",
        "method": "actquant",
        "display_name": "ActQuant 4.0 BPW/A16",
        "upstream_commit": ACTQUANT_COMMIT,
        "model_family": "pi05",
        "model_unit": load_frozen(args.frozen_manifest)["model_unit"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "calibration_manifest": str(args.frozen_manifest),
        "calibration_manifest_sha256": sha256_file(args.frozen_manifest),
        "calibration_source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "selection_seed": 42,
        "test_results_used": False,
        "alpha": 1.0,
        "model_precision": allocation["model_precision"],
        "activation_compute_dtype": allocation["activation_compute_dtype"],
        "target_inventory_sha256": allocation["target_inventory_sha256"],
        "target_tensor_count": len(inventory),
        "tensor_mapping_sha256": canonical_hash(final_map),
        "hsic_sha256": allocation["hsic_sha256"],
        "fisher_shards": fisher_records,
        "fisher_sha256": canonical_hash(fisher_records),
        "fisher_imatrix": {"llm": llm_fisher_record, "vision": vision_fisher_record},
        "allocation": {"path": str(args.allocation), "sha256": sha256_file(args.allocation)},
        "achieved_bpw": actual_bpw,
        "allocation_estimated_bpw": allocation["estimated_actual_bpw_original_parameters"],
        "target_payload_bytes": total_payload_bytes,
        "target_fp16_baseline_bytes": target_fp16,
        "target_static_compression_ratio": target_fp16 / total_payload_bytes,
        "fixed_precision_inventory_sha256": canonical_hash(fixed_inventory),
        "exclusion_inventory_sha256": canonical_hash(fixed_inventory),
        "fixed_precision_parameter_count": sum(int(row["parameters"]) for row in fixed_inventory),
        "fixed_precision_native_bytes": fixed_bytes,
        "gguf_static_bytes": final_record["bytes"],
        "full_fp16_gguf_bytes": official["unified"]["bytes"],
        "full_gguf_compression_ratio": official["unified"]["bytes"] / final_record["bytes"],
        "static_size_scope": "actual_official_full_gguf_after_released_llm_and_local_vision_merge",
        "precision_semantics": {
            "target_weights": "official_ActQuant_IQ_QK_tensor_mixture_with_per_weight_AMF",
            "target_budget": "actual_GGUF_target_payload_BPW_le_4",
            "activations": "A16_no_activation_quantization",
            "excluded_parameters": "FP16_in_official_complete_GGUF",
            "load": "GGUF_dequantize_once_to_live_FP16_dtype",
        },
        "files": [{
            "id": "pi05_final", "path": str(final),
            "sha256": final_record["sha256"], "bytes": final_record["bytes"],
        }],
        "tensors": tensor_records,
        "official_pi05_pipeline": {
            "export": official,
            "llama_quantize": llm_quant_record,
            "released_llm_merge": llm_merge_record,
            "vision_component_source": vision_source_record,
            "vision_component_quantize": vision_quant_record,
            "local_vision_merge": final_record,
        },
        "runtime": "gguf_dequantized_once_to_native_dtype",
        "implementation": implementation_record(),
        "runtime_memory_claim_allowed": False,
        "latency_claim_allowed": False,
        "cpp_runtime_claim_allowed": False,
    }
    atomic_json(args.manifest_output, bundle)
    print(json.dumps({**{k: v for k, v in bundle.items() if k not in ("files", "tensors", "official_pi05_pipeline")},
                      "manifest_sha256": sha256_file(args.manifest_output)}, indent=2))


def pack(args) -> None:
    allocation = json.loads(args.allocation.read_text(encoding="utf-8"))
    if allocation.get("kind") != "actquant_tensor_allocation" or allocation["checkpoint_sha256"] != args.checkpoint_sha256:
        raise ValueError("invalid ActQuant allocation")
    if allocation["calibration_manifest_sha256"] != sha256_file(args.frozen_manifest):
        raise ValueError("ActQuant allocation/calibration drift")
    if (
        allocation.get("activation_compute_dtype") != "float16"
        or (allocation.get("model_precision") or {}).get("strict_all_linear_conv_fp16") is not True
    ):
        raise ValueError("ActQuant allocation precision drift")
    fisher = {}
    fisher_records = []
    indices = set()
    sample_count = None
    inventory_sha = allocation["target_inventory_sha256"]
    for path in sorted(args.fisher_shards):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("kind") != "actquant_fisher_shard" or payload["upstream_commit"] != ACTQUANT_COMMIT:
            raise ValueError(f"invalid Fisher shard: {path}")
        if payload.get("job_implementation_sha256") != job_implementation_sha256():
            raise ValueError(f"Fisher implementation drift: {path}")
        if payload["checkpoint_sha256"] != args.checkpoint_sha256 or payload["full_inventory_sha256"] != inventory_sha:
            raise ValueError(f"Fisher checkpoint/inventory drift: {path}")
        if payload["calibration_manifest_sha256"] != sha256_file(args.frozen_manifest) or payload["alpha"] != 1.0:
            raise ValueError(f"Fisher calibration/alpha drift: {path}")
        if (
            payload.get("activation_compute_dtype") != "float16"
            or (payload.get("model_precision") or {}).get("strict_all_linear_conv_fp16") is not True
        ):
            raise ValueError(f"Fisher precision drift: {path}")
        optimization = payload.get("collector_optimization")
        if (
            args.model_family == "pi05"
            and (optimization or {}).get("name") != "pi05_camera_major_siglip_batch"
        ) or (args.model_family != "pi05" and optimization is not None):
            raise ValueError(f"Fisher collector-optimization drift: {path}")
        if payload.get("fisher_accumulator") != "cuda_fp32_single_final_d2h":
            raise ValueError(f"Fisher accumulator drift: {path}")
        if int(payload["shard_count"]) != args.shard_count:
            raise ValueError("Fisher shard count drift")
        index = int(payload["shard_index"])
        if index in indices:
            raise ValueError("duplicate Fisher shard")
        indices.add(index)
        sample_count = sample_count or int(payload["frames"])
        if sample_count != int(payload["frames"]):
            raise ValueError("Fisher frame count drift")
        for name, value in payload["fisher_sum"].items():
            if name in fisher:
                raise ValueError(f"duplicate Fisher tensor: {name}")
            fisher[name] = value.float().div_(sample_count)
        fisher_records.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    if indices != set(range(args.shard_count)) or set(fisher) != set(allocation["assignments"]):
        raise ValueError("Fisher shard/target coverage mismatch")
    policy = load_policy(args.model_family, args.checkpoint, args.device)
    model = model_of(args.model_family, policy)
    precision = require_fp16_model(args.model_family, model)
    inventory = target_inventory(model, args.model_family)
    if canonical_hash(inventory) != inventory_sha:
        raise ValueError("live target inventory drift before ActQuant pack")
    if args.model_family == "pi05":
        pack_pi05_official(
            args, policy=policy, model=model, inventory=inventory, allocation=allocation,
            fisher=fisher, fisher_records=fisher_records, sample_count=sample_count,
        )
        return
    mapping = allocation["tensor_mapping"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    files, tensor_records = [], []
    total_payload_bytes = 0
    chunks = chunk_names(inventory, args.component_max_bytes)
    inventory_by_name = {row["name"]: row for row in inventory}
    for index, names in enumerate(chunks):
        file_id = f"component_{index:03d}"
        source = output / f"{file_id}.f16.gguf"
        imatrix = output / f"{file_id}.amf.gguf"
        quantized = output / f"{file_id}.actquant.gguf"
        type_manifest = output / f"{file_id}.types.tsv"
        source_record = write_component_gguf(
            model, names, mapping, source, model_family=args.model_family,
            metadata={"checkpoint_sha256": args.checkpoint_sha256, "inventory_sha256": inventory_sha},
        )
        fisher_record = write_fisher_gguf(
            fisher, names, mapping, imatrix, num_samples=sample_count,
            calibration_sha256=sha256_file(args.frozen_manifest),
        )
        quant_record = quantize_component(
            source, imatrix, allocation["assignments"], names, mapping, quantized,
            type_manifest, log_path=output / f"{file_id}.quantize.log",
        )
        inspected = inspect_quantized_gguf(quantized)
        if set(inspected) != {mapping[name] for name in names}:
            raise ValueError(f"quantized GGUF coverage drift: {file_id}")
        for name in names:
            info = inspected[mapping[name]]
            if info["quant_type"] != allocation["assignments"][name]:
                raise ValueError(f"quant type drift for {name}")
            total_payload_bytes += int(info["payload_bytes"])
            tensor_records.append({
                "module_name": name,
                "gguf_name": mapping[name],
                "file_id": file_id,
                "shape": inventory_by_name[name]["shape"],
                "quant_type": allocation["assignments"][name],
                "payload_bytes": int(info["payload_bytes"]),
                "stored_shape": info["shape"],
            })
        files.append({
            "id": file_id,
            "path": str(quantized),
            "sha256": quant_record["sha256"],
            "bytes": quant_record["bytes"],
            "source": source_record,
            "fisher": fisher_record,
            "type_manifest": {"path": str(type_manifest), "sha256": sha256_file(type_manifest)},
            "quantizer_log": {"path": quant_record["log"], "sha256": quant_record["log_sha256"]},
        })
        print(f"[actquant][pack] {index + 1}/{len(chunks)} {file_id} tensors={len(names)}", flush=True)
        gc.collect()
    original_parameters = sum(int(row["parameters"]) for row in inventory)
    actual_bpw = total_payload_bytes * 8 / original_parameters
    if actual_bpw > 4.0 + 1e-6:
        raise ValueError(f"final GGUF payload exceeds 4.0 BPW: {actual_bpw}")
    fixed_component = write_fixed_component_gguf(
        model,
        output / "fixed.fp16.gguf",
        model_family=args.model_family,
        metadata={
            "checkpoint_sha256": args.checkpoint_sha256,
            "inventory_sha256": inventory_sha,
            "precision": "F16",
        },
    )
    files.append({key: value for key, value in fixed_component.items() if key != "tensors"})
    static_bytes = sum(record["bytes"] for record in files)
    baseline_bytes = original_parameters * 2
    fixed_inventory = fixed_parameter_inventory(model, args.model_family)
    fixed_fp16_bytes = sum(int(row["parameters"]) * 2 for row in fixed_inventory)
    bundle = {
        "schema_version": 1,
        "format": "actquant_gguf_bundle_v1",
        "method": "actquant",
        "display_name": "ActQuant 4.0 BPW/A16",
        "upstream_commit": ACTQUANT_COMMIT,
        "model_family": args.model_family,
        "model_unit": load_frozen(args.frozen_manifest)["model_unit"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "calibration_manifest": str(args.frozen_manifest),
        "calibration_manifest_sha256": sha256_file(args.frozen_manifest),
        "calibration_source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "selection_seed": 42,
        "test_results_used": False,
        "alpha": 1.0,
        "model_precision": precision,
        "activation_compute_dtype": "float16",
        "target_inventory_sha256": inventory_sha,
        "target_tensor_count": len(inventory),
        "tensor_mapping_sha256": allocation["tensor_mapping_sha256"],
        "hsic_sha256": allocation["hsic_sha256"],
        "fisher_shards": fisher_records,
        "fisher_sha256": canonical_hash(fisher_records),
        "allocation": {"path": str(args.allocation), "sha256": sha256_file(args.allocation)},
        "achieved_bpw": actual_bpw,
        "allocation_estimated_bpw": allocation["estimated_actual_bpw_original_parameters"],
        "target_payload_bytes": total_payload_bytes,
        "gguf_static_bytes": static_bytes,
        "target_fp16_baseline_bytes": baseline_bytes,
        "target_static_compression_ratio": baseline_bytes / total_payload_bytes,
        "fixed_precision_inventory_sha256": canonical_hash(fixed_inventory),
        "exclusion_inventory_sha256": canonical_hash(fixed_inventory),
        "fixed_precision_parameter_count": sum(int(row["parameters"]) for row in fixed_inventory),
        "fixed_precision_fp16_payload_bytes": sum(
            int(record["payload_bytes"]) for record in fixed_component["tensors"]
        ),
        "full_component_native_baseline_bytes": baseline_bytes + fixed_fp16_bytes,
        "full_component_static_bytes": static_bytes,
        "full_component_static_compression_ratio": (baseline_bytes + fixed_fp16_bytes) / static_bytes,
        "static_size_scope": "real_target_component_ggufs_plus_real_fixed_fp16_component_gguf",
        "precision_semantics": {
            "target_weights": "official_ActQuant_IQ_QK_tensor_mixture_with_per_weight_AMF",
            "target_budget": "actual_component_GGUF_target_payload_BPW_le_4",
            "activations": "A16_no_activation_quantization",
            "excluded_parameters": "real_fixed_FP16_component_GGUF",
            "load": "GGUF_dequantize_once_to_live_FP16_dtype",
        },
        "files": files,
        "tensors": sorted(tensor_records, key=lambda row: row["module_name"]),
        "fixed_tensors": fixed_component["tensors"],
        "fixed_tensor_names_sha256": fixed_component["tensor_names_sha256"],
        "runtime": "gguf_dequantized_once_to_native_dtype",
        "implementation": implementation_record(),
        "runtime_memory_claim_allowed": False,
        "latency_claim_allowed": False,
    }
    atomic_json(args.manifest_output, bundle)
    print(json.dumps({**{k: v for k, v in bundle.items() if k not in ("files", "tensors")}, "manifest_sha256": sha256_file(args.manifest_output)}, indent=2))


def common(parser):
    parser.add_argument("--model-family", choices=("gr00t", "pi05"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    hsic = sub.add_parser("collect-hsic"); common(hsic)
    hsic.add_argument("--output", type=Path, required=True)
    hsic.add_argument("--shard-index", type=int, required=True)
    hsic.add_argument("--shard-count", type=int, required=True)
    hsic.add_argument("--batch-size", type=int, default=1)
    hsic.add_argument("--log-every", type=int, default=25)
    fisher = sub.add_parser("collect-fisher"); common(fisher)
    fisher.add_argument("--output", type=Path, required=True)
    fisher.add_argument("--shard-index", type=int, required=True)
    fisher.add_argument("--shard-count", type=int, required=True)
    fisher.add_argument("--batch-size", type=int, default=1)
    fisher.add_argument("--log-every", type=int, default=25)
    allocation = sub.add_parser("allocate"); common(allocation)
    allocation.add_argument("--hsic-shards", type=Path, nargs="+", required=True)
    allocation.add_argument("--shard-count", type=int, required=True)
    allocation.add_argument("--output", type=Path, required=True)
    pack_parser = sub.add_parser("pack"); common(pack_parser)
    pack_parser.add_argument("--allocation", type=Path, required=True)
    pack_parser.add_argument("--fisher-shards", type=Path, nargs="+", required=True)
    pack_parser.add_argument("--shard-count", type=int, required=True)
    pack_parser.add_argument("--output-dir", type=Path, required=True)
    pack_parser.add_argument("--manifest-output", type=Path, required=True)
    pack_parser.add_argument("--component-max-bytes", type=int, default=512 * 1024 * 1024)
    pack_parser.add_argument(
        "--pi05-tokenizer", type=Path,
        default=Path("/home1/gyy/.cache/openpi/big_vision/paligemma_tokenizer.model"),
    )
    pack_parser.add_argument(
        "--pi05-export-python",
        default="/home1/gyy/probe/miniforge3/envs/openpi/bin/python",
    )
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
        elif isinstance(value, list) and value and isinstance(value[0], Path):
            setattr(args, key, [path.resolve() for path in value])
    if hasattr(args, "shard_index") and not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard")
    {"collect-hsic": collect_hsic, "collect-fisher": collect_fisher, "allocate": allocate, "pack": pack}[args.command](args)


if __name__ == "__main__":
    main()
