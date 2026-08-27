#!/usr/bin/env python3
"""GR00T adapter for the shared single-layer physical OutputImpact probe."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from gr00t_sensitivity_probe import run_rollouts  # noqa: E402
from gr00t_v2_common import (  # noqa: E402
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    ensure_a8_calibrated,
    ensure_flash_attn_rpath,
    load_policy,
    set_quant_env,
    strip_quant_env,
)
from quantvla_cross_model_protocol import (  # noqa: E402
    PROTOCOL,
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
    gr00t_rollout_inputs,
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
from gr00t.quantization.duquant_layers import DuQuantLinear  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / (
    "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
    "target_posttraining/atomic_seen/checkpoint-60000"
)
DEFAULT_PLAN = REPO_ROOT / "checkpoints/packs/robocasa365/quantvla_v1_uniform_w4a8.json"
DEFAULT_CALIBRATION = REPO_ROOT / "runs/errorfold_v4_iter/calibration/gr00t_full_w4"
DEFAULT_DATA_CONFIG = "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig"


def checkpoint_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no checkpoint safetensors under {path}")
    for candidate in files:
        digest.update(candidate.name.encode("utf-8") + b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--activation-mode",
        choices=("static_a8", "dynamic_a8", "fp16"),
        default="static_a8",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_obs < 4 or args.n_obs % 4:
        raise ValueError("OutputImpact requires complete four-replan sequences")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve()
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    a8_path = Path(args.a8).expanduser().resolve()
    hessian_path = Path(args.hessian_w4).expanduser().resolve()
    buffer_path = Path(args.buffer).expanduser().resolve()
    artifact_buffer = Path(args.artifact_calibration_buffer).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    required_paths = [
        checkpoint / "config.json",
        plan_path,
        hessian_path,
        Path(str(hessian_path) + ".json"),
        buffer_path,
        artifact_buffer,
    ]
    if args.activation_mode == "static_a8":
        required_paths.extend((a8_path, Path(str(a8_path) + ".meta.json")))
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not pack_dir.is_dir():
        raise FileNotFoundError(pack_dir)

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_quant_plan(plan, model="gr00t", source=str(plan_path))
    expected_names = sorted(
        name
        for name, row in (plan.get("layers") or {}).items()
        if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
    )
    if not expected_names:
        raise ValueError("OutputImpact base plan contains no W4 layers")
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
            a8_path, model="gr00t", expected_buffer_sha256=artifact_buffer_sha
        )
    selected_names = shard_names(expected_names, args.shard_index, args.shard_count)

    records, buffer_provenance = load_model_records(
        buffer_path, args.n_obs, model="gr00t"
    )
    observations, noises = gr00t_rollout_inputs(records, noise_index=0)
    details = record_metadata(records)
    payload = {
        "schema_version": 4,
        "kind": ARTIFACT_KIND,
        "cross_model_protocol": protocol_attestation(),
        "model_adapter": "gr00t",
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
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
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
                REPO_ROOT / "scripts/tools/gr00t_sensitivity_probe.py"
            ),
            "quant_runtime": sha256_file(
                REPO_ROOT / "code/gr00t/quantization/duquant_layers.py"
            ),
            "w4_kernel": sha256_file(
                REPO_ROOT / "code/gr00t/quantization/duquant_fused.py"
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

    ensure_flash_attn_rpath()
    strip_quant_env()
    torch.manual_seed(0)
    teacher_policy = load_policy(
        str(checkpoint),
        data_config=args.data_config,
        denoising_steps=int(PROTOCOL["closed_loop"]["flow_steps"]),
        device=args.device,
    )
    _, teacher_actions = run_rollouts(
        teacher_policy.model,
        teacher_policy,
        observations,
        noises,
        args.batch_size,
        return_physical=True,
    )
    scale = physical_action_scale(teacher_actions)
    del teacher_policy
    gc.collect()
    torch.cuda.empty_cache()

    strip_quant_env()
    set_quant_env(
        DEFAULT_INCLUDE,
        DEFAULT_EXCLUDE,
        str(pack_dir),
        row_rot="0",
        act_dynamic=args.activation_mode == "dynamic_a8",
    )
    os.environ.update(
        {
            "GR00T_DUQUANT_FUSED": "1",
            "GR00T_DUQUANT_PLAN": str(plan_path),
            "GR00T_DUQUANT_HESSIAN_W4_PATH": str(hessian_path),
            "GR00T_ATM_ENABLE": "0",
            "GR00T_OHB_ENABLE": "0",
        }
    )
    if args.activation_mode == "static_a8":
        os.environ["GR00T_DUQUANT_ACT_SCALE_PATH"] = str(a8_path)
    else:
        os.environ.pop("GR00T_DUQUANT_ACT_SCALE_PATH", None)
    if args.activation_mode == "fp16":
        os.environ["GR00T_DUQUANT_ABITS"] = "0"
    policy = load_policy(
        str(checkpoint),
        data_config=args.data_config,
        denoising_steps=int(PROTOCOL["closed_loop"]["flow_steps"]),
        device=args.device,
    )
    if args.activation_mode == "static_a8":
        warm_records, _ = load_model_records(
            artifact_buffer, 32 * args.batch_size, model="gr00t"
        )
        warm_obs, warm_noises = gr00t_rollout_inputs(warm_records, noise_index=0)
        ensure_a8_calibrated(
            policy,
            warm_obs,
            warm_noises,
            args.batch_size,
            expected_wrapped=len(expected_names),
            act_scale_path=str(a8_path),
        )
    layers = install_fp16_bypass(
        policy.model.named_modules(), module_type=DuQuantLinear
    )
    if set(layers) != set(expected_names):
        raise RuntimeError("live GR00T OutputImpact layer inventory mismatch")
    set_single_intervention(layers, None)
    _, bypass_actions = run_rollouts(
        policy.model,
        policy,
        observations,
        noises,
        args.batch_size,
        return_physical=True,
    )
    payload["fp16_bypass_check"] = identity_check(teacher_actions, bypass_actions)
    bypass_pair = summarize_pair(
        teacher_actions, bypass_actions, details, scale=scale
    )
    payload["fp16_bypass_check"]["d_pac"] = float(
        bypass_pair["d_pac_summary"]["d_pac"]
    )

    for name in selected_names:
        if name in payload["layers"]:
            print(f"[outputimpact/gr00t] reuse {name}", flush=True)
            continue
        set_single_intervention(layers, name)
        started = time.time()
        _, candidate_actions = run_rollouts(
            policy.model,
            policy,
            observations,
            noises,
            args.batch_size,
            return_physical=True,
        )
        pair = summarize_pair(
            teacher_actions, candidate_actions, details, scale=scale
        )
        payload["layers"][name] = impact_row(pair, time.time() - started)
        atomic_json(output, payload)
        print(
            f"[outputimpact/gr00t] {name}: "
            f"D_PAC={payload['layers'][name]['d_pac']:.6g}",
            flush=True,
        )
        del candidate_actions
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
