#!/usr/bin/env python3
"""Score complete π0.5 W4/FP16 masks with LIBERO D_PAC on noise A or B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / "code/pi05/openpi"
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "packages/openpi-client/src"))
sys.path.insert(0, str(ROOT / "scripts/tools"))

from openpi.policies import policy_config  # noqa: E402
from openpi.quant import enable_duquant_if_configured, iter_duquant_layers  # noqa: E402
from openpi.quant.duquant_layers import DuQuantLinear  # noqa: E402
from openpi.training import config  # noqa: E402
from probe_libero_dypac_pi05_outputimpact import configure  # noqa: E402
from quantvla_libero_dypac import (  # noqa: E402
    PROTOCOL,
    PROTOCOL_PATH,
    atomic_json,
    load_records,
    physical_scale,
    run_policy,
    sha256_file,
    summarize_pair,
)
from quantvla_outputimpact import install_fp16_bypass  # noqa: E402


def set_mask(layers: dict, protected: set[str]) -> None:
    unknown = protected - set(layers)
    if unknown:
        raise ValueError(f"mask contains unknown layers: {sorted(unknown)[:3]}")
    for name, layer in layers.items():
        layer._outputimpact_fp16 = name in protected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--plan", required=True, help="All-W4 wrapper plan")
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--hessian", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--noise", choices=("A", "B"), default="A")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid mask score shard")
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    names = [row["name"] for row in inventory["layers"]]
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL["protocol_id"]:
        raise ValueError("mask manifest protocol drift")
    candidates = manifest["candidates"]
    candidate_ids = sorted(candidates)
    selected_ids = [value for index, value in enumerate(candidate_ids) if index % args.shard_count == args.shard_index]
    buffer = Path(args.buffer).resolve()
    records = load_records(buffer, selection_only=True)
    noise_index = 0 if args.noise == "A" else 1
    configure(args, sha256_file(buffer))
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_libero"), Path(args.checkpoint_dir).resolve(), pytorch_device=args.device
    )
    runtime = enable_duquant_if_configured(policy._model)
    policy._model.to(args.device).eval()
    if runtime.get("hessian_w4_loaded") != 180 or not runtime.get("act_dynamic"):
        raise RuntimeError("mask scorer did not load complete Hessian/dynamic-A8 runtime")
    layers = install_fp16_bypass(iter_duquant_layers(policy._model), module_type=DuQuantLinear)
    if list(layers) != names:
        raise RuntimeError("wrapped layer order differs from candidate inventory")
    set_mask(layers, set(names))
    reference = run_policy(policy, records, args.device, noise_index=noise_index, batch_size=args.batch_size)
    scale = physical_scale(reference)

    output = Path(args.out).resolve()
    previous = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else None
    if previous is not None:
        expected_previous = {
            "kind": "dypac_libero_pi05_complete_mask_scores",
            "protocol_id": PROTOCOL["protocol_id"],
            "manifest_sha256": sha256_file(manifest_path),
            "selection_buffer_sha256": sha256_file(buffer),
            "noise": args.noise,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "assigned_candidates": selected_ids,
        }
        drift = {
            key: (previous.get(key), expected)
            for key, expected in expected_previous.items()
            if previous.get(key) != expected
        }
        if drift:
            raise RuntimeError(f"refusing to resume incompatible mask-score shard: {drift}")
    scores = dict(previous.get("scores", {})) if previous else {}
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_pi05_complete_mask_scores",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selection_buffer_sha256": sha256_file(buffer),
        "noise": args.noise,
        "selection_noise": args.noise == "A",
        "uses_success_labels": False,
        "uses_test_rollout_feedback": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned_candidates": selected_ids,
        "scores": scores,
        "complete": False,
    }
    started = time.monotonic()
    for ordinal, identifier in enumerate(selected_ids, start=1):
        if identifier in scores:
            print(f"[mask {args.shard_index}] reuse {ordinal}/{len(selected_ids)} {identifier}", flush=True)
            continue
        protected = set(candidates[identifier]["protected_layers"])
        set_mask(layers, protected)
        candidate_started = time.monotonic()
        actions = run_policy(policy, records, args.device, noise_index=noise_index, batch_size=args.batch_size)
        summary = summarize_pair(reference, actions, records, scale=scale)
        scores[identifier] = {
            "protected_layers": sorted(protected),
            "d_pac": float(summary["d_pac_summary"]["d_pac"]),
            "d_func": float(summary["d_func_summary"]["d_func"]),
            "d_pac_summary": summary["d_pac_summary"],
            "d_func_summary": summary["d_func_summary"],
            "elapsed_s": time.monotonic() - candidate_started,
        }
        payload["scores"] = scores
        atomic_json(output, payload)
        print(f"[mask {args.shard_index}] {ordinal}/{len(selected_ids)} {identifier} D_PAC={scores[identifier]['d_pac']:.6g}", flush=True)
    payload["complete"] = set(scores) == set(selected_ids)
    payload["elapsed_s"] = time.monotonic() - started
    atomic_json(output, payload)
    if not payload["complete"]:
        raise RuntimeError("mask score shard incomplete")


if __name__ == "__main__":
    main()
