#!/usr/bin/env python3
"""Score arbitrary static W4/FP16 masks using one full-W4 GR00T model.

This adapter-only evaluator is used for calibration-only mask refinement.  It
loads the full Hessian-W4A8 network once, executes skipped layers through their
bitwise FP16 bypass, and compares every mask with the original FP16 teacher in
the shared physical D_PAC-v2 space.  Closed-loop success is never observed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from gr00t_probe_outputimpact import (  # noqa: E402
    DEFAULT_CALIBRATION,
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_CONFIG,
    DEFAULT_PLAN,
    checkpoint_sha256,
)
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
    protocol_artifact,
    protocol_attestation,
    sha256_file,
    validate_quant_plan,
)
from quantvla_dynamic_a8_protocol import (  # noqa: E402
    protocol_attestation as dynamic_a8_protocol_attestation,
    require_protocol_attestation as require_dynamic_a8_protocol_attestation,
)
from quantvla_metric_protocol import physical_action_scale, summarize_pair  # noqa: E402
from quantvla_full_context import (  # noqa: E402
    PROTOCOL as FULL_CONTEXT_PROTOCOL,
    candidate_plan_mapping,
    protocol_attestation as full_context_protocol_attestation,
    shard_candidate_mapping,
)
from quantvla_model_adapters import (  # noqa: E402
    gr00t_rollout_inputs,
    load_model_records,
    record_metadata,
    validate_calibration_artifact,
)
from quantvla_outputimpact import atomic_json, identity_check, install_fp16_bypass  # noqa: E402
from gr00t.quantization.duquant_layers import DuQuantLinear  # noqa: E402


def parse_named_plan(value: str) -> tuple[str, Path]:
    identifier, separator, raw_path = value.partition("=")
    if not separator or not identifier or not raw_path:
        raise argparse.ArgumentTypeError("candidate plan must be ID=/absolute/or/relative/path.json")
    return identifier, Path(raw_path).expanduser().resolve()


def validate_flow_artifacts(
    *, hessian_path: Path, a8_path: Path, activation_mode: str, flow_steps: int
) -> None:
    hessian_meta = json.loads(
        Path(str(hessian_path) + ".json").read_text(encoding="utf-8")
    )
    capture_path = Path(hessian_meta["capture_path"]).expanduser().resolve()
    if sha256_file(capture_path) != hessian_meta.get("capture_sha256"):
        raise ValueError("GR00T Hessian/FP16 capture lineage drift")
    with np.load(capture_path, allow_pickle=False) as capture:
        captured_steps = int(
            capture["capture_flow_steps"].item()
            if "capture_flow_steps" in capture
            else 4
        )
    if captured_steps != flow_steps:
        raise ValueError(f"GR00T Hessian flow-step drift: {captured_steps} != {flow_steps}")
    if activation_mode == "static_a8":
        sidecar = Path(str(a8_path) + ".meta.json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            int(metadata.get("denoising_steps", -1)) != flow_steps
            or int(metadata.get("dit_flow_step_tables", -1)) != flow_steps
        ):
            raise ValueError("GR00T static A8 table/native flow-step drift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--base-full-w4-plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--pack-dir", default=str(DEFAULT_CALIBRATION / "identity_pack"))
    parser.add_argument("--a8", default=str(DEFAULT_CALIBRATION / "a8_scales.npz"))
    parser.add_argument("--hessian-w4", default=str(DEFAULT_CALIBRATION / "hessian_w4.npz"))
    parser.add_argument(
        "--activation-mode",
        choices=("static_a8", "dynamic_a8", "fp16"),
        default="static_a8",
        help="Activation-only diagnostic; W4 codes and every other protocol field stay fixed.",
    )
    parser.add_argument("--buffer", default=str(protocol_artifact("selection_buffer", verify=False)))
    parser.add_argument(
        "--artifact-calibration-buffer",
        default=str(protocol_artifact("calibration_buffer", verify=False)),
    )
    parser.add_argument("--candidate-plan", action="append", type=parse_named_plan)
    parser.add_argument("--candidate-manifest")
    parser.add_argument("--candidate-shard-index", type=int, default=0)
    parser.add_argument("--candidate-shard-count", type=int, default=1)
    parser.add_argument("--n-obs", type=int, default=32)
    parser.add_argument(
        "--noise-rule", choices=("A", "B"), default="A",
        help="Noise B is a frozen single-candidate audit; its D_PAC scale remains fixed from teacher noise A.",
    )
    parser.add_argument(
        "--flow-steps",
        type=int,
        default=int(FULL_CONTEXT_PROTOCOL["model_hyperparameters"]["gr00t"]["table1_flow_steps"]),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_obs < 4 or args.n_obs % 4:
        raise ValueError("plan scoring requires complete four-replan sequences")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    base_plan_path = Path(args.base_full_w4_plan).expanduser().resolve()
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    a8_path = Path(args.a8).expanduser().resolve()
    hessian_path = Path(args.hessian_w4).expanduser().resolve()
    buffer_path = Path(args.buffer).expanduser().resolve()
    artifact_buffer = Path(args.artifact_calibration_buffer).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    validate_flow_artifacts(
        hessian_path=hessian_path,
        a8_path=a8_path,
        activation_mode=args.activation_mode,
        flow_steps=args.flow_steps,
    )
    all_candidates, candidate_manifest = candidate_plan_mapping(
        named=args.candidate_plan, manifest_path=args.candidate_manifest
    )
    candidates = shard_candidate_mapping(
        all_candidates,
        shard_index=args.candidate_shard_index,
        shard_count=args.candidate_shard_count,
    )
    if args.noise_rule == "B" and len(candidates) != 1:
        raise ValueError("noise-B audit requires exactly one frozen candidate")
    base_plan = json.loads(base_plan_path.read_text(encoding="utf-8"))
    validate_quant_plan(base_plan, model="gr00t", source=str(base_plan_path))
    full_names = {
        name
        for name, row in base_plan["layers"].items()
        if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
    }
    plan_payloads = {}
    quantized_names = {}
    for identifier, path in candidates.items():
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_quant_plan(document, model="gr00t", source=str(path))
        if int((document.get("meta") or {}).get("flow_steps", -1)) != args.flow_steps:
            raise ValueError(f"{identifier}: candidate flow-step hyperparameter drift")
        if args.activation_mode == "dynamic_a8":
            require_dynamic_a8_protocol_attestation(
                document.get("meta") or {}, source=str(path)
            )
        if set(document.get("layers") or {}) != set(base_plan["layers"]):
            raise ValueError(f"{identifier}: candidate inventory differs from full-W4 plan")
        names = {
            name
            for name, row in document["layers"].items()
            if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
        }
        if not names.issubset(full_names):
            raise ValueError(f"{identifier}: candidate contains out-of-scope layers")
        plan_payloads[identifier] = document
        quantized_names[identifier] = names

    records, buffer_provenance = load_model_records(buffer_path, args.n_obs, model="gr00t")
    noise_index = 0 if args.noise_rule == "A" else 1
    observations, noises = gr00t_rollout_inputs(records, noise_index=noise_index)
    details = record_metadata(records)
    artifact_buffer_sha = sha256_file(artifact_buffer)
    if args.activation_mode == "static_a8":
        validate_calibration_artifact(
            a8_path, model="gr00t", expected_buffer_sha256=artifact_buffer_sha
        )
    payload = {
        "schema_version": 4,
        "kind": "outputimpact_static_mask_scores",
        "cross_model_protocol": protocol_attestation(),
        "full_context_protocol": full_context_protocol_attestation(),
        "model_adapter": "gr00t",
        "teacher": "original_fp16",
        "selection_metric": "d_pac_v2",
        "selection_noise": args.noise_rule,
        "selection_role": (
            "noise_a_selection" if args.noise_rule == "A"
            else "frozen_noise_b_generalization_audit"
        ),
        "noise_b_used_for_selection": False,
        "candidate_manifest": (
            {
                "path": str(Path(args.candidate_manifest).expanduser().resolve()),
                "sha256": sha256_file(Path(args.candidate_manifest).expanduser().resolve()),
                "kind": candidate_manifest.get("kind"),
            }
            if candidate_manifest is not None
            else None
        ),
        "candidate_shard": {
            "index": args.candidate_shard_index,
            "count": args.candidate_shard_count,
            "unsharded_candidate_count": len(all_candidates),
            "candidate_ids": list(candidates),
        },
        "base_full_w4_plan_sha256": sha256_file(base_plan_path),
        "hessian_w4_sha256": sha256_file(hessian_path),
        "activation_mode": args.activation_mode,
        "a8_sha256": (
            sha256_file(a8_path) if args.activation_mode == "static_a8" else None
        ),
        "selection_buffer_sha256": buffer_provenance["sha256"],
        "calibration_buffer_sha256": artifact_buffer_sha,
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "n_obs": args.n_obs,
        "flow_steps": args.flow_steps,
        "uses_cka": False,
        "uses_cs": False,
        "uses_success_labels": False,
        "uses_gradients": False,
        "candidate_plans": {
            identifier: {"path": str(path), "sha256": sha256_file(path)}
            for identifier, path in candidates.items()
        },
        "source_sha256": {
            "scorer": sha256_file(Path(__file__)),
            "metric": sha256_file(REPO_ROOT / "scripts/tools/quantvla_metric_protocol.py"),
            "selection_core": sha256_file(
                REPO_ROOT / "scripts/tools/quantvla_full_context.py"
            ),
            "quant_runtime": sha256_file(REPO_ROOT / "code/gr00t/quantization/duquant_layers.py"),
            "w4_kernel": sha256_file(REPO_ROOT / "code/gr00t/quantization/duquant_fused.py"),
        },
        "scores": {},
    }
    if args.activation_mode == "dynamic_a8":
        payload["dynamic_a8_protocol"] = dynamic_a8_protocol_attestation()
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        invariant = tuple(
            key for key in payload if key not in ("scores", "teacher_latency_mean_s")
        )
        if any(previous.get(key) != payload.get(key) for key in invariant):
            raise ValueError("existing static-mask score provenance drift")
        payload["scores"] = previous.get("scores") or {}

    ensure_flash_attn_rpath()
    strip_quant_env()
    torch.manual_seed(0)
    teacher_policy = load_policy(
        str(checkpoint), data_config=args.data_config,
        denoising_steps=args.flow_steps, device=args.device,
    )
    if args.noise_rule == "B":
        scale_observations, scale_noises = gr00t_rollout_inputs(records, noise_index=0)
        _, scale_actions = run_rollouts(
            teacher_policy.model, teacher_policy, scale_observations, scale_noises,
            args.batch_size, return_physical=True,
        )
        scale = physical_action_scale(scale_actions)
    teacher_t0 = time.time()
    _, teacher_actions = run_rollouts(
        teacher_policy.model, teacher_policy, observations, noises, args.batch_size,
        return_physical=True,
    )
    payload["teacher_latency_mean_s"] = float(
        (time.time() - teacher_t0) / max(1, len(records))
    )
    if args.noise_rule == "A":
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
            "GR00T_DUQUANT_PLAN": str(base_plan_path),
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
        str(checkpoint), data_config=args.data_config,
        denoising_steps=args.flow_steps, device=args.device,
    )
    if args.activation_mode == "static_a8":
        warm_records, _ = load_model_records(
            artifact_buffer, 32 * args.batch_size, model="gr00t"
        )
        warm_obs, warm_noises = gr00t_rollout_inputs(warm_records, noise_index=0)
        ensure_a8_calibrated(
            policy, warm_obs, warm_noises, args.batch_size,
            expected_wrapped=len(full_names), act_scale_path=str(a8_path),
        )
    layers = install_fp16_bypass(policy.model.named_modules(), module_type=DuQuantLinear)
    if set(layers) != full_names:
        raise RuntimeError("live full-W4 layer inventory mismatch")
    for module in layers.values():
        module._outputimpact_fp16 = True
    _, bypass_actions = run_rollouts(
        policy.model, policy, observations, noises, args.batch_size, return_physical=True
    )
    payload["fp16_bypass_check"] = identity_check(teacher_actions, bypass_actions)

    for identifier in candidates:
        if identifier in payload["scores"]:
            print(f"[outputimpact-mask/gr00t] reuse {identifier}", flush=True)
            continue
        active = quantized_names[identifier]
        for name, module in layers.items():
            module._outputimpact_fp16 = name not in active
        started = time.time()
        _, actions = run_rollouts(
            policy.model, policy, observations, noises, args.batch_size, return_physical=True
        )
        pair = summarize_pair(teacher_actions, actions, details, scale=scale)
        payload["scores"][identifier] = {
            "quantized_w4_layers": len(active),
            "retained_fp16_layers": len(full_names) - len(active),
            "d_pac": float(pair["d_pac_summary"]["d_pac"]),
            "d_pac_summary": pair["d_pac_summary"],
            "d_func": float(pair["d_func_summary"]["d_func"]),
            "d_func_summary": pair["d_func_summary"],
            "elapsed_s": time.time() - started,
        }
        atomic_json(output, payload)
        print(
            f"[outputimpact-mask/gr00t] {identifier}: "
            f"D_PAC={payload['scores'][identifier]['d_pac']:.6g}", flush=True,
        )
    payload["complete"] = set(payload["scores"]) == set(candidates)
    atomic_json(output, payload)
    print(json.dumps({"out": str(output), "scores": {key: value["d_pac"] for key, value in payload["scores"].items()}}, indent=2))


if __name__ == "__main__":
    main()
