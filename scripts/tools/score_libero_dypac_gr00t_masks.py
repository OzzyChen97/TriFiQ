#!/usr/bin/env python3
"""Score complete-network GR00T DyPAC masks on one LIBERO suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from gr00t.quantization.duquant_layers import DuQuantLinear
from gr00t_v2_common import ensure_flash_attn_rpath, load_policy
from probe_libero_dypac_gr00t_outputimpact import DATA_CONFIGS, configure, identity
from quantvla_libero_dypac import (
    PROTOCOL,
    PROTOCOL_PATH,
    atomic_json,
    load_records,
    physical_scale,
    run_gr00t_policy,
    sha256_file,
    summarize_pair,
)
from quantvla_outputimpact import install_fp16_bypass


def set_mask(layers: dict, protected: set[str]) -> None:
    unknown = protected - set(layers)
    if unknown:
        raise ValueError(f"GR00T mask contains unknown layers: {sorted(unknown)[:3]}")
    for name, layer in layers.items():
        layer._outputimpact_fp16 = name in protected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(DATA_CONFIGS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--plan", required=True, help="Suite all-W4 wrapper plan")
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--hessian", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--noise", choices=("A", "B"), default="A")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid GR00T mask-score shard")
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    names = [row["name"] for row in inventory["layers"]]
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL["protocol_id"]:
        raise ValueError("GR00T mask manifest protocol drift")
    candidates = manifest["candidates"]
    candidate_ids = sorted(candidates)
    selected_ids = [
        value
        for index, value in enumerate(candidate_ids)
        if index % args.shard_count == args.shard_index
    ]
    all_records = load_records(args.buffer, selection_only=True, model="gr00t")
    records = [record for record in all_records if record["suite"] == args.suite]
    if len(all_records) != 144 or len(records) != 36:
        raise ValueError("GR00T mask scoring requires 144 global and 36 suite records")
    noise_index = 0 if args.noise == "A" else 1
    global_teacher = np.stack([record["teacher_actions"] for record in all_records])
    scale = physical_scale(global_teacher)
    configure(args)
    ensure_flash_attn_rpath()
    policy = load_policy(
        str(Path(args.checkpoint).resolve()),
        data_config=DATA_CONFIGS[args.suite],
        denoising_steps=10,
        device=args.device,
    )
    runtime = getattr(policy.model, "_gr00t_duquant_runtime", {})
    if runtime.get("hessian_w4_loaded") != len(names):
        raise RuntimeError("GR00T mask scorer did not load the complete Hessian artifact")
    layers = install_fp16_bypass(policy.model.named_modules(), module_type=DuQuantLinear)
    if list(layers) != names:
        raise RuntimeError("wrapped GR00T layer order differs from candidate inventory")
    set_mask(layers, set(names))
    reference = run_gr00t_policy(
        policy, records, noise_index=noise_index, batch_size=args.batch_size
    )
    if args.noise == "A":
        teacher_identity = identity(
            reference,
            np.stack([record["teacher_actions"] for record in records]),
            records,
            scale,
        )
    else:
        teacher_identity = None

    output = Path(args.out).resolve()
    previous = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else None
    if previous is not None:
        expected_previous = {
            "kind": "dypac_libero_gr00t_complete_mask_scores_suite",
            "model": "gr00t",
            "suite": args.suite,
            "protocol_id": PROTOCOL["protocol_id"],
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "selection_buffer_sha256": sha256_file(args.buffer),
            "noise": args.noise,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "assigned_candidates": selected_ids,
            "paired_scoring_batch_size": args.batch_size,
        }
        drift = {
            key: (previous.get(key), expected)
            for key, expected in expected_previous.items()
            if previous.get(key) != expected
        }
        if drift:
            raise RuntimeError(f"refusing to resume incompatible GR00T mask shard: {drift}")
    scores = dict(previous.get("scores", {})) if previous else {}
    unknown_scores = set(scores) - set(selected_ids)
    if unknown_scores:
        raise RuntimeError(
            f"GR00T mask shard contains unassigned scores: {sorted(unknown_scores)[:3]}"
        )
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_gr00t_complete_mask_scores_suite",
        "model": "gr00t",
        "suite": args.suite,
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selection_buffer_sha256": sha256_file(args.buffer),
        "noise": args.noise,
        "selection_noise": args.noise == "A",
        "teacher_identity": teacher_identity,
        "teacher_identity_batch_size": args.batch_size if args.noise == "A" else None,
        "paired_scoring_batch_size": args.batch_size,
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
            print(
                f"[mask/gr00t/{args.suite}/{args.shard_index}] reuse "
                f"{ordinal}/{len(selected_ids)} {identifier}",
                flush=True,
            )
            continue
        protected = set(candidates[identifier]["protected_layers"])
        set_mask(layers, protected)
        candidate_started = time.monotonic()
        actions = run_gr00t_policy(
            policy, records, noise_index=noise_index, batch_size=args.batch_size
        )
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
        print(
            f"[mask/gr00t/{args.suite}/{args.shard_index}] "
            f"{ordinal}/{len(selected_ids)} {identifier} "
            f"D_PAC={scores[identifier]['d_pac']:.6g}",
            flush=True,
        )
    payload["complete"] = set(scores) == set(selected_ids)
    payload["elapsed_s"] = time.monotonic() - started
    atomic_json(output, payload)
    if not payload["complete"]:
        raise RuntimeError("GR00T mask-score suite shard incomplete")


if __name__ == "__main__":
    main()
