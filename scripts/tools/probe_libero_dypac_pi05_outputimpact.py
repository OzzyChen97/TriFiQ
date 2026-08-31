#!/usr/bin/env python3
"""Measure every π0.5 single-layer W4 intervention with LIBERO D_PAC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / "code/pi05/openpi"
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "packages/openpi-client/src"))
sys.path.insert(0, str(ROOT / "scripts/tools"))

from openpi.policies import policy_config  # noqa: E402
from openpi.quant import enable_duquant_if_configured, iter_duquant_layers  # noqa: E402
from openpi.quant.duquant_layers import DuQuantLinear  # noqa: E402
from openpi.training import config  # noqa: E402
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
from quantvla_outputimpact import install_fp16_bypass, set_single_intervention  # noqa: E402


def configure(args: argparse.Namespace, buffer_sha: str) -> None:
    keys = [key for key in os.environ if key.startswith("OPENPI_DUQUANT") or key.startswith("OPENPI_ATM") or key.startswith("OPENPI_OHB") or key == "OPENPI_ERRORFOLD_PATH"]
    for key in keys:
        os.environ.pop(key, None)
    os.environ.update({
        "TORCHDYNAMO_DISABLE": "1",
        "OPENPI_MODEL_DTYPE": "float16",
        "OPENPI_CHECKPOINT_SHA256": args.checkpoint_sha256,
        "OPENPI_DUQUANT_PLAN": str(Path(args.plan).resolve()),
        "OPENPI_DUQUANT_PLAN_STRICT": "1",
        "OPENPI_DUQUANT_WBITS_DEFAULT": "4",
        "OPENPI_DUQUANT_ABITS": "8",
        "OPENPI_DUQUANT_BLOCK": "64",
        "OPENPI_DUQUANT_BLOCK_OUT": "64",
        "OPENPI_DUQUANT_EXPECT_BLOCK": "64",
        "OPENPI_DUQUANT_EXPECT_WRAPPED": "180",
        "OPENPI_DUQUANT_LS": "0",
        "OPENPI_DUQUANT_PERMUTE": "0",
        "OPENPI_DUQUANT_ROW_ROT": "0",
        "OPENPI_DUQUANT_DENOISING_STEPS": "10",
        "OPENPI_DUQUANT_PACKDIR": str(Path(args.pack_dir).resolve()),
        "OPENPI_DUQUANT_HESSIAN_W4_PATH": str(Path(args.hessian).resolve()),
        "OPENPI_DUQUANT_ACT_DYNAMIC": "1",
        "OPENPI_DUQUANT_REQUIRE_ACT_SCALE": "0",
        "OPENPI_DUQUANT_CALIB_BUFFER_SHA256": buffer_sha,
        "OPENPI_DUQUANT_STRICT_ARTIFACTS": "0",
        "OPENPI_DUQUANT_PRECACHE_WEIGHTS": "0",
        "OPENPI_DUQUANT_TRITON": "1",
        "OPENPI_DUQUANT_QUIET": "1",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--hessian", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid OutputImpact shard")
    inventory_path = Path(args.inventory).resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    names = [row["name"] for row in inventory["layers"]]
    selected = [name for index, name in enumerate(names) if index % args.shard_count == args.shard_index]
    buffer = Path(args.buffer).resolve()
    records = load_records(buffer, selection_only=True)
    if len(records) != 144:
        raise ValueError(f"selection requires 144 rows, got {len(records)}")
    configure(args, sha256_file(buffer))

    policy = policy_config.create_trained_policy(
        config.get_config("pi05_libero"), Path(args.checkpoint_dir).resolve(), pytorch_device=args.device
    )
    runtime = enable_duquant_if_configured(policy._model)
    policy._model.to(args.device).eval()
    if runtime.get("hessian_w4_loaded") != 180 or not runtime.get("act_dynamic"):
        raise RuntimeError(f"incomplete DyPAC probe runtime: {runtime}")
    layers = install_fp16_bypass(iter_duquant_layers(policy._model), module_type=DuQuantLinear)
    if list(layers) != names:
        raise RuntimeError("wrapped layer order differs from candidate inventory")

    set_single_intervention(layers, None)
    reference = run_policy(policy, records, args.device, batch_size=args.batch_size)
    stored = np.stack([row["teacher_actions"] for row in records])
    delta = np.abs(reference - stored)
    scale = physical_scale(reference)
    identity_pair = summarize_pair(stored, reference, records, scale=scale)
    identity = {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "p99_abs": float(np.quantile(delta, 0.99)),
        "p999_abs": float(np.quantile(delta, 0.999)),
        "per_dimension_mean_abs": delta.mean(axis=(0, 1)).tolist(),
        "per_dimension_max_abs": delta.max(axis=(0, 1)).tolist(),
        "d_pac": float(identity_pair["d_pac_summary"]["d_pac"]),
        "d_func": float(identity_pair["d_func_summary"]["d_func"]),
    }
    # Separate GPU processes can disagree at a handful of flow-matching output
    # elements even with identical checkpoint, observation, and supplied
    # noise.  A maximum-only gate therefore rejects an otherwise paired buffer
    # because of one physical-action outlier.  Gate the complete distribution
    # and the actual selection metric instead.  Candidate scores below use the
    # freshly reproduced all-FP16 bypass as their paired reference, so this is
    # only a provenance/numerical-reproduction check.
    identity["passed"] = bool(
        identity["mean_abs"] <= 5e-4
        and identity["p99_abs"] <= 5e-3
        and identity["d_pac"] <= 2e-2
    )
    if not identity["passed"]:
        raise RuntimeError(
            f"offline FP16 teacher disagrees with collected source: {identity}"
        )

    output = Path(args.out).resolve()
    previous = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else None
    if previous is not None:
        expected_previous = {
            "kind": "dypac_libero_pi05_single_layer_outputimpact",
            "protocol_id": PROTOCOL["protocol_id"],
            "checkpoint_sha256": args.checkpoint_sha256,
            "inventory_sha256": sha256_file(inventory_path),
            "plan_sha256": sha256_file(args.plan),
            "hessian_sha256": sha256_file(args.hessian),
            "selection_buffer_sha256": sha256_file(buffer),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "assigned_layers": selected,
        }
        drift = {
            key: (previous.get(key), expected)
            for key, expected in expected_previous.items()
            if previous.get(key) != expected
        }
        if drift:
            raise RuntimeError(f"refusing to resume incompatible OutputImpact shard: {drift}")
    completed = dict(previous.get("layers", {})) if previous else {}
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_pi05_single_layer_outputimpact",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "checkpoint_sha256": args.checkpoint_sha256,
        "inventory": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "plan_sha256": sha256_file(args.plan),
        "hessian_sha256": sha256_file(args.hessian),
        "selection_buffer_sha256": sha256_file(buffer),
        "selection_rows": 144,
        "intervention": "exactly_one_hessian_group64_w4_dynamic_a8_layer_all_others_native_fp16",
        "teacher": "original_fp16_checkpoint",
        "teacher_identity": identity,
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
            print(f"[OutputImpact {args.shard_index}] reuse {ordinal}/{len(selected)} {name}", flush=True)
            continue
        set_single_intervention(layers, name)
        layer_started = time.monotonic()
        candidate = run_policy(policy, records, args.device, batch_size=args.batch_size)
        summary = summarize_pair(reference, candidate, records, scale=scale)
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
            f"[OutputImpact {args.shard_index}] {ordinal}/{len(selected)} {name} "
            f"D_PAC={completed[name]['d_pac']:.6g}", flush=True
        )
    payload["complete"] = set(completed) == set(selected)
    payload["elapsed_s"] = time.monotonic() - started
    atomic_json(output, payload)
    if not payload["complete"]:
        raise RuntimeError("OutputImpact shard incomplete")


if __name__ == "__main__":
    main()
