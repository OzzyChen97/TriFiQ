#!/usr/bin/env python3
"""pi0.5 adapter for the shared single-layer physical OutputImpact probe."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = REPO_ROOT / "code" / "pi05" / "openpi"
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from openpi.quant import enable_duquant_if_configured, iter_duquant_layers  # noqa: E402
from openpi.quant.duquant_layers import DuQuantLinear  # noqa: E402
from pi05_score_configs_against_fp16 import (  # noqa: E402
    CHECKPOINT_SHA256,
    DEFAULT_CHECKPOINT,
    configure_base,
    configure_quant,
    load_policy,
)
from pi05_sensitivity_probe import run_records  # noqa: E402
from quantvla_cross_model_protocol import (  # noqa: E402
    PROTOCOL_SHA256,
    protocol_artifact,
    protocol_attestation,
    sha256_file,
    validate_quant_plan,
)
from quantvla_dynamic_a8_protocol import (  # noqa: E402
    protocol_attestation as dynamic_a8_protocol_attestation,
)
from quantvla_metric_protocol import physical_action_scale, summarize_pair  # noqa: E402
from quantvla_model_adapters import (  # noqa: E402
    canonical_physical_chunk,
    load_model_records,
    record_metadata,
    validate_calibration_artifact,
)
from quantvla_outputimpact import (  # noqa: E402
    ARTIFACT_KIND,
    atomic_json,
    identity_check,
    impact_row,
    install_fp16_bypass,
    set_single_intervention,
    shard_names,
)


DEFAULT_PLAN = REPO_ROOT / (
    "runs/pi05_gdsq_gr00t_aligned/plans/pi05_quantvla_uniform_w4a8_d4.plan.json"
)
DEFAULT_CALIBRATION = REPO_ROOT / "runs/errorfold_v4_iter/calibration/pi05_full_w4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument(
        "--pack-dir", default=str(DEFAULT_CALIBRATION / "identity_pack")
    )
    parser.add_argument("--a8", default=str(DEFAULT_CALIBRATION / "a8_scales.npz"))
    parser.add_argument(
        "--hessian-w4", default=str(DEFAULT_CALIBRATION / "hessian_w4.npz")
    )
    parser.add_argument(
        "--buffer", default=str(protocol_artifact("selection_buffer", verify=False))
    )
    parser.add_argument(
        "--artifact-calibration-buffer",
        default=str(protocol_artifact("calibration_buffer", verify=False)),
    )
    parser.add_argument("--n-obs", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--activation-mode",
        choices=("static_a8", "dynamic_a8", "fp16"),
        default="static_a8",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_obs < 4 or args.n_obs % 4:
        raise ValueError("OutputImpact requires complete four-replan sequences")
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve()
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    a8_path = Path(args.a8).expanduser().resolve()
    hessian_path = Path(args.hessian_w4).expanduser().resolve()
    buffer_path = Path(args.buffer).expanduser().resolve()
    artifact_buffer = Path(args.artifact_calibration_buffer).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    required_paths = [
        checkpoint / "model.safetensors",
        plan_path,
        hessian_path,
        Path(str(hessian_path) + ".json"),
        buffer_path,
        artifact_buffer,
    ]
    if args.activation_mode == "static_a8":
        required_paths.extend((a8_path, Path(str(a8_path) + ".json")))
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(checkpoint / "model.safetensors") != CHECKPOINT_SHA256:
        raise ValueError("pi0.5 checkpoint drift")
    if not pack_dir.is_dir():
        raise FileNotFoundError(pack_dir)

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_quant_plan(plan, model="pi05", source=str(plan_path))
    expected_names = sorted(
        name
        for name, row in (plan.get("layers") or {}).items()
        if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
    )
    if len(expected_names) != 180:
        raise ValueError(f"pi0.5 full-W4 OutputImpact requires 180 layers, got {len(expected_names)}")
    hessian_meta = json.loads(
        Path(str(hessian_path) + ".json").read_text(encoding="utf-8")
    )
    if hessian_meta.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("Hessian protocol drift")
    if set(hessian_meta.get("layer_names") or []) != set(expected_names):
        raise ValueError("Hessian/base-plan inventory mismatch")
    artifact_buffer_sha = sha256_file(artifact_buffer)
    if args.activation_mode == "static_a8":
        validate_calibration_artifact(
            a8_path, model="pi05", expected_buffer_sha256=artifact_buffer_sha
        )
    selected_names = shard_names(expected_names, args.shard_index, args.shard_count)

    records, buffer_provenance = load_model_records(
        buffer_path, args.n_obs, model="pi05"
    )
    details = record_metadata(records)
    payload = {
        "schema_version": 4,
        "kind": ARTIFACT_KIND,
        "cross_model_protocol": protocol_attestation(),
        "model_adapter": "pi05",
        "teacher": "original_fp16",
        "selection_metric": "d_pac_v2",
        "base_full_w4_plan": str(plan_path),
        "base_full_w4_plan_sha256": sha256_file(plan_path),
        "hessian_w4": str(hessian_path),
        "hessian_w4_sha256": sha256_file(hessian_path),
        "activation_mode": args.activation_mode,
        "a8": str(a8_path) if args.activation_mode == "static_a8" else None,
        "a8_sha256": (
            sha256_file(a8_path) if args.activation_mode == "static_a8" else None
        ),
        "selection_buffer": str(buffer_path),
        "selection_buffer_sha256": buffer_provenance["sha256"],
        "calibration_buffer_sha256": artifact_buffer_sha,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "n_obs": args.n_obs,
        "noise": "A",
        "intervention": (
            f"exactly_one_hessian_group64_w4_{args.activation_mode}_linear; every other candidate "
            "uses its original FP16 weight and no A8"
        ),
        "uses_cka": False,
        "uses_cs": False,
        "uses_task_success": False,
        "uses_success_labels": False,
        "uses_gradients": False,
        "uses_extra_training_data": False,
        "deployment_uses_bypass": False,
        "shard": {
            "index": args.shard_index,
            "count": args.shard_count,
            "layer_names": selected_names,
        },
        "records": details,
        "source_sha256": {
            "adapter": sha256_file(Path(__file__)),
            "shared_probe": sha256_file(
                REPO_ROOT / "scripts/tools/quantvla_outputimpact.py"
            ),
            "metric": sha256_file(
                REPO_ROOT / "scripts/tools/quantvla_metric_protocol.py"
            ),
            "rollout_adapter": sha256_file(
                REPO_ROOT / "scripts/tools/pi05_sensitivity_probe.py"
            ),
            "quant_runtime": sha256_file(
                REPO_ROOT
                / "code/pi05/openpi/src/openpi/quant/duquant_layers.py"
            ),
            "w4_kernel": sha256_file(
                REPO_ROOT
                / "code/pi05/openpi/src/openpi/quant/duquant_triton.py"
            ),
        },
        "layers": {},
        "complete": False,
    }
    if args.activation_mode == "dynamic_a8":
        payload["dynamic_a8_protocol"] = dynamic_a8_protocol_attestation()
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        invariant = (
            "cross_model_protocol",
            "dynamic_a8_protocol",
            "model_adapter",
            "base_full_w4_plan_sha256",
            "hessian_w4_sha256",
            "a8_sha256",
            "activation_mode",
            "selection_buffer_sha256",
            "calibration_buffer_sha256",
            "checkpoint_sha256",
            "n_obs",
            "shard",
            "source_sha256",
        )
        if any(previous.get(key) != payload.get(key) for key in invariant):
            raise ValueError("existing OutputImpact shard provenance drift")
        payload["layers"] = previous.get("layers") or {}

    configure_base()
    torch.manual_seed(0)
    teacher_policy = load_policy(checkpoint, args.device)
    _, teacher_native, teacher_timings = run_records(
        teacher_policy, records, args.device, noise_index=0
    )
    teacher_actions = canonical_physical_chunk(teacher_native, model="pi05")
    scale = physical_action_scale(teacher_actions)
    payload["teacher_latency_mean_s"] = float(np.mean(teacher_timings))
    del teacher_policy
    gc.collect()
    torch.cuda.empty_cache()

    spec = {
        "plan": plan_path,
        "a8": a8_path,
        "hessian_w4": hessian_path,
        "wrapped": len(expected_names),
        "atm": None,
    }
    configure_quant(
        spec=spec,
        pack_dir=pack_dir,
        artifact_buffer_hash=artifact_buffer_sha,
        strict_artifacts=True,
        activation_mode=args.activation_mode,
    )
    policy = load_policy(checkpoint, args.device)
    runtime = enable_duquant_if_configured(policy._model)
    policy._model.to(args.device)
    quant_layers = iter_duquant_layers(policy._model)
    if (
        int(runtime.get("wrapped_layers", 0)) != len(expected_names)
        or len(quant_layers) != len(expected_names)
        or not all(
            module._hessian_w4_loaded
            and module._fused_ready
            and module.act_scale_ready
            for _, module in quant_layers
        )
    ):
        raise RuntimeError("live pi0.5 OutputImpact network lacks complete Hessian W4A8")
    layers = install_fp16_bypass(
        policy._model.named_modules(), module_type=DuQuantLinear
    )
    if set(layers) != set(expected_names):
        raise RuntimeError("live pi0.5 OutputImpact layer inventory mismatch")
    set_single_intervention(layers, None)
    _, bypass_native, _ = run_records(policy, records, args.device, noise_index=0)
    bypass_actions = canonical_physical_chunk(bypass_native, model="pi05")
    payload["fp16_bypass_check"] = identity_check(teacher_actions, bypass_actions)
    bypass_pair = summarize_pair(
        teacher_actions, bypass_actions, details, scale=scale
    )
    payload["fp16_bypass_check"]["d_pac"] = float(
        bypass_pair["d_pac_summary"]["d_pac"]
    )

    for name in selected_names:
        if name in payload["layers"]:
            print(f"[outputimpact/pi05] reuse {name}", flush=True)
            continue
        set_single_intervention(layers, name)
        started = time.time()
        _, candidate_native, timings = run_records(
            policy, records, args.device, noise_index=0
        )
        candidate_actions = canonical_physical_chunk(candidate_native, model="pi05")
        pair = summarize_pair(
            teacher_actions, candidate_actions, details, scale=scale
        )
        row = impact_row(pair, time.time() - started)
        row["latency_mean_s"] = float(np.mean(timings))
        payload["layers"][name] = row
        atomic_json(output, payload)
        print(
            f"[outputimpact/pi05] {name}: D_PAC={row['d_pac']:.6g}",
            flush=True,
        )
        del candidate_native, candidate_actions
        gc.collect()
        torch.cuda.empty_cache()

    set_single_intervention(layers, None)
    payload["complete"] = set(payload["layers"]) == set(selected_names)
    payload["completed_layers"] = len(payload["layers"])
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "out": str(output),
                "complete": payload["complete"],
                "completed_layers": payload["completed_layers"],
                "total_shard_layers": len(selected_names),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
