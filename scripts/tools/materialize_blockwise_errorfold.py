#!/usr/bin/env python3
"""Materialize one coordinate round of selector-free blockwise ErrorFold.

The search unit is an attention-logit layer, not a task or request.  Every
candidate is a fully static deployment artifact.  Linear and attention-head
affines are held at identity; each attention-logit block uses one of the
shared finite gate levels.  This keeps the search training-free and makes the
same method applicable to both model adapters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from fit_softfold_compensation import sha256_file, validate_raw_correction_protocol
from quantvla_errorfold import materialize_entry


LEVELS = (0.0, 0.5, 1.0)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def profile_sha256(profile: Mapping[str, float]) -> str:
    canonical = json.dumps(dict(sorted(profile.items())), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_profile(path: str | None, layer_names: list[str]) -> dict[str, float]:
    if path is None:
        return {name: 0.0 for name in layer_names}
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    raw_profile = payload.get("profile", payload.get("gate_profile", payload))
    if not isinstance(raw_profile, Mapping):
        raise ValueError("base profile must be a mapping")
    unknown = sorted(set(raw_profile) - set(layer_names))
    missing = sorted(set(layer_names) - set(raw_profile))
    if unknown or missing:
        raise ValueError(
            f"base profile inventory drift: missing={missing[:3]} unknown={unknown[:3]}"
        )
    result = {name: float(raw_profile[name]) for name in layer_names}
    if any(value not in LEVELS for value in result.values()):
        raise ValueError(f"blockwise profile gates must be in {LEVELS}")
    return result


def compact_entry(raw_entry: Mapping[str, Any], gate: float) -> dict[str, Any]:
    materialized = materialize_entry(raw_entry, gate)
    gain = [float(value) for value in materialized["effective_gain"]]
    bias = [float(value) for value in materialized["effective_bias"]]
    return {
        "kind": str(raw_entry.get("kind", "linear")),
        "channels": len(gain),
        "effective_gain": gain,
        "effective_bias": bias,
        "block_gate": float(gate),
        "reliability_already_folded": True,
        "softmax_invariant_bias_dropped": bool(
            materialized.get("softmax_invariant_bias_dropped", False)
        ),
        "runtime_selector": False,
    }


def materialize_profile(
    raw_layers: Mapping[str, Mapping[str, Any]],
    profile: Mapping[str, float],
) -> tuple[dict[str, Any], float]:
    layers: dict[str, Any] = {}
    squared_norm = 0.0
    for name, raw_entry in sorted(raw_layers.items()):
        kind = str(raw_entry.get("kind", "linear"))
        gate = float(profile[name]) if kind == "attention_logits" else 0.0
        entry = compact_entry(raw_entry, gate)
        squared_norm += sum((value - 1.0) ** 2 for value in entry["effective_gain"])
        squared_norm += sum(value**2 for value in entry["effective_bias"])
        layers[str(name)] = entry
    return layers, math.sqrt(squared_norm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-correction", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--a8", required=True)
    parser.add_argument("--hessian-w4", required=True)
    parser.add_argument("--base-profile")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    args = parser.parse_args()

    if args.round < 0:
        raise ValueError("round must be non-negative")
    raw_path = Path(args.raw_correction).expanduser().resolve()
    template_path = Path(args.template).expanduser().resolve()
    plan = Path(args.plan).expanduser().resolve()
    a8 = Path(args.a8).expanduser().resolve()
    hessian = Path(args.hessian_w4).expanduser().resolve()
    for required in (raw_path, template_path, plan, a8, hessian):
        if not required.is_file():
            raise FileNotFoundError(required)

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_meta = validate_raw_correction_protocol(raw, source=str(raw_path))
    raw_layers = raw.get("layers") or {}
    layer_names = sorted(
        name
        for name, entry in raw_layers.items()
        if entry.get("kind") == "attention_logits"
    )
    if not layer_names:
        raise ValueError("raw correction has no attention-logit blocks")
    base = load_profile(args.base_profile, layer_names)
    template = json.loads(template_path.read_text(encoding="utf-8"))
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles: dict[str, dict[str, float]] = {f"block_r{args.round:02d}_base": base}
    for index, name in enumerate(layer_names):
        for level in LEVELS:
            if level == base[name]:
                continue
            profile = dict(base)
            profile[name] = level
            profiles[f"block_r{args.round:02d}_l{index:02d}_g{int(level * 2)}"] = profile

    candidates: dict[str, Any] = {}
    for config_id, profile in sorted(profiles.items()):
        layers, correction_norm = materialize_profile(raw_layers, profile)
        profile_hash = profile_sha256(profile)
        artifact = {
            "schema_version": 3,
            "kind": "errorfold_compensation",
            "cross_model_protocol": template["cross_model_protocol"],
            "dynamic_a8_protocol": template.get("dynamic_a8_protocol"),
            "teacher_checkpoint_sha256": template["teacher_checkpoint_sha256"],
            "quant_plan_sha256": template["quant_plan_sha256"],
            "buffer_sha256": template["buffer_sha256"],
            "metric": "d_func_v1_plus_d_pac_v2_block_coordinate",
            "gate": {"type": "static_block_profile", "errorfold": 0.0},
            "gate_profile": profile,
            "gate_profile_sha256": profile_hash,
            "correction_norm": correction_norm,
            "layers": layers,
            "selection": {
                "status": "offline_coordinate_candidate_not_frozen",
                "round": args.round,
                "uses_task_labels": False,
                "uses_rollout_success": False,
                "sample_unit": "paired_shared_buffer_sequence",
            },
            "meta": {
                **raw_meta,
                "method_id": "blocksoftfold-dual-coordinate-v1",
                "raw_correction_sha256": sha256_file(raw_path),
                "template_sha256": sha256_file(template_path),
                "profile_sha256": profile_hash,
                "gate_levels": list(LEVELS),
                "atm_application": "fold_q_weight",
                "errorfold_application": "fold_affine_into_weight_dequant_scale_and_bias",
                "selector_free": True,
                "runtime_branch": False,
            },
        }
        artifact_path = output_dir / f"{config_id}.json"
        atomic_json(artifact_path, artifact)
        candidates[config_id] = {
            "errorfold": str(artifact_path),
            "errorfold_sha256": sha256_file(artifact_path),
            "gate": {"type": "static_block_profile", "errorfold": 0.0},
            "gate_profile": profile,
            "gate_profile_sha256": profile_hash,
            "correction_norm": correction_norm,
        }

    manifest = {
        "schema_version": 1,
        "kind": "blockwise_errorfold_candidate_manifest",
        "method_id": "blocksoftfold-dual-coordinate-v1",
        "round": args.round,
        "raw_correction": str(raw_path),
        "raw_correction_sha256": sha256_file(raw_path),
        "base_profile": base,
        "base_profile_sha256": profile_sha256(base),
        "gate_levels": list(LEVELS),
        "common": {
            "plan": str(plan),
            "plan_sha256": sha256_file(plan),
            "a8": str(a8),
            "a8_sha256": sha256_file(a8),
            "hessian_w4": str(hessian),
            "hessian_w4_sha256": sha256_file(hessian),
        },
        "candidates": candidates,
        "selection": {
            "uses_task_labels": False,
            "uses_rollout_success": False,
            "rule": "coordinate_minimum_worst_endpoint_normalized_regret_d_func_d_pac",
        },
    }
    manifest_out = Path(args.manifest_out).expanduser().resolve()
    atomic_json(manifest_out, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_out),
                "sha256": sha256_file(manifest_out),
                "round": args.round,
                "blocks": len(layer_names),
                "candidates": len(candidates),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
