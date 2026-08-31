#!/usr/bin/env python3
"""Measure GR00T single-layer W4 interventions with LIBERO D_PAC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from gr00t.quantization.duquant_layers import DuQuantLinear
from gr00t_v2_common import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    ensure_flash_attn_rpath,
    load_policy,
    set_quant_env,
    strip_quant_env,
)
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
from quantvla_outputimpact import install_fp16_bypass, set_single_intervention


DATA_CONFIGS = {
    "goal": "examples.Libero.custom_data_config:LiberoDataConfigMeanStd",
    "spatial": "examples.Libero.custom_data_config:LiberoDataConfig",
    "object": "examples.Libero.custom_data_config:LiberoDataConfig",
    "long": "examples.Libero.custom_data_config:LiberoDataConfig",
}


def configure(args: argparse.Namespace) -> None:
    strip_quant_env()
    set_quant_env(
        DEFAULT_INCLUDE,
        DEFAULT_EXCLUDE,
        str(Path(args.pack_dir).resolve()),
        bits_default=4,
        group=64,
        ls=0.0,
        row_rot="0",
        act_dynamic=True,
    )
    os.environ.update(
        {
            "TORCHDYNAMO_DISABLE": "1",
            "TORCH_COMPILE_DISABLE": "1",
            "GR00T_DUQUANT_PLAN": str(Path(args.plan).resolve()),
            "GR00T_DUQUANT_HESSIAN_W4_PATH": str(Path(args.hessian).resolve()),
            "GR00T_DUQUANT_FUSED": "1",
            "GR00T_DUQUANT_PRECACHE_WEIGHTS": "0",
            "GR00T_ATM_ENABLE": "0",
            "GR00T_OHB_ENABLE": "0",
        }
    )


def identity(reference: np.ndarray, stored: np.ndarray, records: list[dict], scale) -> dict:
    delta = np.abs(reference - stored)
    pair = summarize_pair(stored, reference, records, scale=scale)
    value = {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "p99_abs": float(np.quantile(delta, 0.99)),
        "p999_abs": float(np.quantile(delta, 0.999)),
        "d_pac": float(pair["d_pac_summary"]["d_pac"]),
        "d_func": float(pair["d_func_summary"]["d_func"]),
    }
    # This is a cross-process provenance/replay gate, not the reference used
    # for candidate ranking.  Reloading the same GR00T FP16 checkpoint through
    # wrapped Linear modules changes a small tail of kernel reductions while
    # preserving the action distribution.  Keep the task-aware D_PAC gate at
    # 0.03 and bound the observed numerical distribution; every candidate is
    # still compared against a freshly generated, same-process FP16 bypass.
    thresholds = {"mean_abs": 3e-3, "p99_abs": 2e-2, "d_pac": 3e-2}
    value["thresholds"] = thresholds
    value["role"] = "cross_process_fp16_provenance_only"
    value["passed"] = bool(
        value["mean_abs"] <= thresholds["mean_abs"]
        and value["p99_abs"] <= thresholds["p99_abs"]
        and value["d_pac"] <= thresholds["d_pac"]
    )
    if not value["passed"]:
        raise RuntimeError(f"GR00T all-FP16 bypass identity failed: {value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(DATA_CONFIGS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--hessian", required=True)
    parser.add_argument("--buffer", required=True, help="Merged 144-row selection buffer")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid GR00T OutputImpact shard")
    inventory_path = Path(args.inventory).resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    names = [row["name"] for row in inventory["layers"]]
    selected = [
        name for index, name in enumerate(names) if index % args.shard_count == args.shard_index
    ]
    all_records = load_records(args.buffer, selection_only=True, model="gr00t")
    records = [record for record in all_records if record["suite"] == args.suite]
    if len(all_records) != 144 or len(records) != 36:
        raise ValueError("GR00T OutputImpact requires 144 global and 36 suite records")
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
        raise RuntimeError(f"incomplete GR00T Hessian runtime: {runtime}")
    layers = install_fp16_bypass(policy.model.named_modules(), module_type=DuQuantLinear)
    if list(layers) != names:
        raise RuntimeError("wrapped GR00T layer order differs from candidate inventory")
    set_single_intervention(layers, None)
    stored = np.stack([record["teacher_actions"] for record in records])
    # Use one fresh FP16 bypass for both provenance and all paired candidate
    # comparisons.  This prevents cross-process replay noise from entering a
    # mask benefit while retaining a fail-closed provenance check above.
    reference = run_gr00t_policy(policy, records, batch_size=args.batch_size)
    teacher_identity = identity(reference, stored, records, scale)

    output = Path(args.out).resolve()
    previous = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else None
    completed = dict(previous.get("layers", {})) if previous else {}
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_gr00t_single_layer_outputimpact_suite",
        "model": "gr00t",
        "suite": args.suite,
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "checkpoint_sha256": inventory["checkpoints"][args.suite]["sha256"],
        "inventory_sha256": sha256_file(inventory_path),
        "plan_sha256": sha256_file(args.plan),
        "hessian_sha256": sha256_file(args.hessian),
        "selection_buffer_sha256": sha256_file(args.buffer),
        "selection_rows": 36,
        "global_scale_rows": 144,
        "intervention": "one_hessian_group64_w4_dynamic_a8_layer_all_others_native_fp16",
        "teacher": "original_fp16_checkpoint",
        "teacher_identity": teacher_identity,
        "teacher_identity_batch_size": args.batch_size,
        "paired_scoring_batch_size": args.batch_size,
        "dimension_scale": scale.tolist(),
        "uses_success_labels": False,
        "uses_test_rollout_feedback": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned_layers": selected,
        "layers": completed,
        "complete": False,
    }
    started = time.monotonic()
    for ordinal, name in enumerate(selected, start=1):
        if name in completed:
            print(
                f"[OutputImpact/gr00t/{args.suite}/{args.shard_index}] reuse "
                f"{ordinal}/{len(selected)} {name}",
                flush=True,
            )
            continue
        set_single_intervention(layers, name)
        layer_started = time.monotonic()
        actions = run_gr00t_policy(policy, records, batch_size=args.batch_size)
        summary = summarize_pair(reference, actions, records, scale=scale)
        completed[name] = {
            "d_pac": float(summary["d_pac_summary"]["d_pac"]),
            "d_func": float(summary["d_func_summary"]["d_func"]),
            "d_pac_summary": summary["d_pac_summary"],
            "d_func_summary": summary["d_func_summary"],
            "elapsed_s": time.monotonic() - layer_started,
        }
        payload["layers"] = completed
        atomic_json(output, payload)
        print(
            f"[OutputImpact/gr00t/{args.suite}/{args.shard_index}] "
            f"{ordinal}/{len(selected)} {name} D_PAC={completed[name]['d_pac']:.6g}",
            flush=True,
        )
    payload["complete"] = set(completed) == set(selected)
    payload["elapsed_s"] = time.monotonic() - started
    atomic_json(output, payload)
    if not payload["complete"]:
        raise RuntimeError("GR00T OutputImpact suite shard incomplete")


if __name__ == "__main__":
    main()
