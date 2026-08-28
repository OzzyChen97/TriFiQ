#!/usr/bin/env python3
"""Jointly score arbitrary pi0.5 W4/FP16 masks with one model load.

The full Hessian-W4 network is loaded once.  Candidate plans only toggle the
calibration-only exact FP16 bypass already used by the single-layer
OutputImpact probe, so mask interaction can be measured without repeatedly
loading the 14 GB checkpoint.  Deployment never retains the bypass weights.
"""

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
from pi05_full_context_adapter import run_records  # noqa: E402
from quantvla_cross_model_protocol import (  # noqa: E402
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
    canonical_physical_chunk,
    load_model_records,
    record_metadata,
    validate_calibration_artifact,
)
from quantvla_outputimpact import atomic_json, identity_check, install_fp16_bypass  # noqa: E402


DEFAULT_PLAN = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_quantvla_uniform_w4a8_d4.plan.json"
)
DEFAULT_CALIBRATION = REPO_ROOT / "runs/errorfold_v4_iter/calibration/pi05_full_w4"


def named_plan(value: str) -> tuple[str, Path]:
    identifier, separator, raw_path = value.partition("=")
    if not separator or not identifier or not raw_path:
        raise argparse.ArgumentTypeError("candidate plan must be ID=PLAN.json")
    if any(not (char.isalnum() or char in "_.-") for char in identifier):
        raise argparse.ArgumentTypeError(f"invalid candidate id: {identifier!r}")
    return identifier, Path(raw_path).expanduser().resolve()


def validate_flow_artifacts(
    *, hessian_path: Path, a8_path: Path, activation_mode: str, flow_steps: int
) -> None:
    hessian_meta = json.loads(
        Path(str(hessian_path) + ".json").read_text(encoding="utf-8")
    )
    capture_path = Path(hessian_meta["capture_path"]).expanduser().resolve()
    if sha256_file(capture_path) != hessian_meta.get("capture_sha256"):
        raise ValueError("pi0.5 Hessian/FP16 capture lineage drift")
    with np.load(capture_path, allow_pickle=False) as capture:
        captured_steps = int(
            np.asarray(capture["capture_flow_steps"]).item()
            if "capture_flow_steps" in capture
            else flow_steps
        )
    if captured_steps != flow_steps:
        raise ValueError(
            f"pi0.5 Hessian flow-step drift: {captured_steps} != {flow_steps}"
        )
    if activation_mode == "static_a8":
        sidecar = Path(str(a8_path) + ".json")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        metadata = payload.get("metadata") or payload
        if (
            int(metadata.get("denoising_steps", -1)) != flow_steps
            or int(metadata.get("dit_flow_step_tables", -1)) != flow_steps
        ):
            raise ValueError("pi0.5 static A8 table/native flow-step drift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--base-full-w4-plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--pack-dir", default=str(DEFAULT_CALIBRATION / "identity_pack"))
    parser.add_argument("--a8", default=str(DEFAULT_CALIBRATION / "a8_scales.npz"))
    parser.add_argument("--hessian-w4", default=str(DEFAULT_CALIBRATION / "hessian_w4.npz"))
    parser.add_argument(
        "--activation-mode",
        choices=("static_a8", "dynamic_a8", "fp16"),
        default="dynamic_a8",
    )
    parser.add_argument(
        "--buffer", default=str(protocol_artifact("selection_buffer", verify=False))
    )
    parser.add_argument(
        "--artifact-calibration-buffer",
        default=str(protocol_artifact("calibration_buffer", verify=False)),
    )
    parser.add_argument("--candidate-plan", action="append", type=named_plan)
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
        default=int(FULL_CONTEXT_PROTOCOL["model_hyperparameters"]["pi05"]["table1_flow_steps"]),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_obs < 4 or args.n_obs % 4:
        raise ValueError("joint mask scoring requires complete four-replan sequences")
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

    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    base_plan_path = Path(args.base_full_w4_plan).expanduser().resolve()
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    a8_path = Path(args.a8).expanduser().resolve()
    hessian_path = Path(args.hessian_w4).expanduser().resolve()
    buffer_path = Path(args.buffer).expanduser().resolve()
    artifact_buffer = Path(args.artifact_calibration_buffer).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    for path in (
        checkpoint / "model.safetensors",
        base_plan_path,
        hessian_path,
        Path(str(hessian_path) + ".json"),
        buffer_path,
        artifact_buffer,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not pack_dir.is_dir():
        raise FileNotFoundError(pack_dir)
    validate_flow_artifacts(
        hessian_path=hessian_path,
        a8_path=a8_path,
        activation_mode=args.activation_mode,
        flow_steps=args.flow_steps,
    )
    if sha256_file(checkpoint / "model.safetensors") != CHECKPOINT_SHA256:
        raise ValueError("pi0.5 checkpoint drift")

    base_plan = json.loads(base_plan_path.read_text(encoding="utf-8"))
    validate_quant_plan(base_plan, model="pi05", source=str(base_plan_path))
    full_names = {
        name
        for name, row in (base_plan.get("layers") or {}).items()
        if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
    }
    if len(full_names) != 180:
        raise ValueError(f"base plan must contain all 180 W4 layers, got {len(full_names)}")

    plan_payloads: dict[str, dict] = {}
    quantized_names: dict[str, set[str]] = {}
    for identifier, path in candidates.items():
        document = json.loads(path.read_text(encoding="utf-8"))
        selection = validate_quant_plan(document, model="pi05", source=str(path))
        if int((document.get("meta") or {}).get("flow_steps", -1)) != args.flow_steps:
            raise ValueError(f"{identifier}: candidate flow-step hyperparameter drift")
        if args.activation_mode == "dynamic_a8":
            require_dynamic_a8_protocol_attestation(document.get("meta") or {}, source=str(path))
        if set(document.get("layers") or {}) != set(base_plan["layers"]):
            raise ValueError(f"{identifier}: candidate inventory differs from the full-W4 plan")
        active = {
            name
            for name, row in document["layers"].items()
            if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
        }
        if not active.issubset(full_names):
            raise ValueError(f"{identifier}: candidate contains an out-of-scope W4 layer")
        if len(active) != int(selection["quantized_w4_layers"]):
            raise ValueError(f"{identifier}: quantized layer attestation mismatch")
        plan_payloads[identifier] = document
        quantized_names[identifier] = active

    records, buffer_provenance = load_model_records(buffer_path, args.n_obs, model="pi05")
    details = record_metadata(records)
    artifact_buffer_sha = sha256_file(artifact_buffer)
    if args.activation_mode == "static_a8":
        validate_calibration_artifact(
            a8_path, model="pi05", expected_buffer_sha256=artifact_buffer_sha
        )

    payload = {
        "schema_version": 1,
        "kind": "outputimpact_joint_mask_scores",
        "cross_model_protocol": protocol_attestation(),
        "full_context_protocol": full_context_protocol_attestation(),
        "model_adapter": "pi05",
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
        "a8_sha256": sha256_file(a8_path) if args.activation_mode == "static_a8" else None,
        "selection_buffer_sha256": buffer_provenance["sha256"],
        "calibration_buffer_sha256": artifact_buffer_sha,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "n_obs": args.n_obs,
        "flow_steps": args.flow_steps,
        "uses_cka": False,
        "uses_cs": False,
        "uses_task_success": False,
        "uses_success_labels": False,
        "uses_gradients": False,
        "uses_extra_training_data": False,
        "calibration_only_fp16_bypass": True,
        "deployment_uses_bypass": False,
        "candidate_plans": {
            identifier: {"path": str(path), "sha256": sha256_file(path)}
            for identifier, path in candidates.items()
        },
        "source_sha256": {
            "scorer": sha256_file(Path(__file__)),
            "single_layer_probe": sha256_file(REPO_ROOT / "scripts/tools/pi05_probe_outputimpact.py"),
            "metric": sha256_file(REPO_ROOT / "scripts/tools/quantvla_metric_protocol.py"),
            "selection_core": sha256_file(
                REPO_ROOT / "scripts/tools/quantvla_full_context.py"
            ),
            "quant_runtime": sha256_file(
                REPO_ROOT / "code/pi05/openpi/src/openpi/quant/duquant_layers.py"
            ),
            "w4_kernel": sha256_file(
                REPO_ROOT / "code/pi05/openpi/src/openpi/quant/duquant_triton.py"
            ),
        },
        "scores": {},
    }
    if args.activation_mode == "dynamic_a8":
        payload["dynamic_a8_protocol"] = dynamic_a8_protocol_attestation()
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        invariant = tuple(key for key in payload if key != "scores")
        if any(previous.get(key) != payload.get(key) for key in invariant):
            raise ValueError("existing joint-mask score provenance drift")
        payload["scores"] = previous.get("scores") or {}

    configure_base()
    torch.manual_seed(0)
    teacher_policy = load_policy(checkpoint, args.device)
    noise_index = 0 if args.noise_rule == "A" else 1
    if args.noise_rule == "B":
        _, scale_native, _ = run_records(
            teacher_policy, records, args.device, noise_index=0,
            flow_steps=args.flow_steps,
        )
        scale = physical_action_scale(
            canonical_physical_chunk(scale_native, model="pi05")
        )
    _, teacher_native, teacher_timings = run_records(
        teacher_policy, records, args.device, noise_index=noise_index,
        flow_steps=args.flow_steps,
    )
    teacher_actions = canonical_physical_chunk(teacher_native, model="pi05")
    if args.noise_rule == "A":
        scale = physical_action_scale(teacher_actions)
    payload["teacher_latency_mean_s"] = float(np.mean(teacher_timings))
    del teacher_policy, teacher_native
    gc.collect()
    torch.cuda.empty_cache()

    spec = {
        "plan": base_plan_path,
        "a8": a8_path,
        "hessian_w4": hessian_path,
        "wrapped": len(full_names),
        "atm": None,
    }
    configure_quant(
        spec=spec,
        pack_dir=pack_dir,
        artifact_buffer_hash=artifact_buffer_sha,
        strict_artifacts=True,
        activation_mode=args.activation_mode,
        flow_steps=args.flow_steps,
    )
    policy = load_policy(checkpoint, args.device)
    runtime = enable_duquant_if_configured(policy._model)
    policy._model.to(args.device)
    quant_layers = iter_duquant_layers(policy._model)
    if (
        int(runtime.get("wrapped_layers", 0)) != len(full_names)
        or len(quant_layers) != len(full_names)
        or not all(module._hessian_w4_loaded and module._fused_ready for _, module in quant_layers)
        or (
            args.activation_mode != "fp16"
            and not all(module.act_scale_ready for _, module in quant_layers)
        )
    ):
        raise RuntimeError("joint scorer lacks the complete live Hessian-W4A8 network")
    layers = install_fp16_bypass(policy._model.named_modules(), module_type=DuQuantLinear)
    if set(layers) != full_names:
        raise RuntimeError("live joint-score layer inventory mismatch")

    for module in layers.values():
        module._outputimpact_fp16 = True
    _, bypass_native, _ = run_records(
        policy, records, args.device, noise_index=noise_index, flow_steps=args.flow_steps
    )
    bypass_actions = canonical_physical_chunk(bypass_native, model="pi05")
    payload["fp16_bypass_check"] = identity_check(teacher_actions, bypass_actions)
    payload["fp16_bypass_check"]["d_pac"] = float(
        summarize_pair(teacher_actions, bypass_actions, details, scale=scale)[
            "d_pac_summary"
        ]["d_pac"]
    )

    for identifier in candidates:
        if identifier in payload["scores"]:
            print(f"[joint-mask/pi05] reuse {identifier}", flush=True)
            continue
        active = quantized_names[identifier]
        for name, module in layers.items():
            module._outputimpact_fp16 = name not in active
        started = time.time()
        _, candidate_native, timings = run_records(
            policy, records, args.device, noise_index=noise_index, flow_steps=args.flow_steps
        )
        actions = canonical_physical_chunk(candidate_native, model="pi05")
        pair = summarize_pair(teacher_actions, actions, details, scale=scale)
        plan = plan_payloads[identifier]
        payload["scores"][identifier] = {
            "quantized_w4_layers": len(active),
            "retained_fp16_layers": len(full_names) - len(active),
            "target_compression": (plan.get("meta") or {}).get("target_compression"),
            "achieved_compression": plan.get("achieved_compression"),
            "d_pac": float(pair["d_pac_summary"]["d_pac"]),
            "d_pac_summary": pair["d_pac_summary"],
            "d_func": float(pair["d_func_summary"]["d_func"]),
            "d_func_summary": pair["d_func_summary"],
            "latency_mean_s": float(np.mean(timings)),
            "elapsed_s": time.time() - started,
        }
        atomic_json(output, payload)
        print(
            f"[joint-mask/pi05] {identifier}: "
            f"D_PAC={payload['scores'][identifier]['d_pac']:.6g}",
            flush=True,
        )
        del candidate_native, actions
        gc.collect()
        torch.cuda.empty_cache()

    for module in layers.values():
        module._outputimpact_fp16 = True
    payload["complete"] = set(payload["scores"]) == set(candidates)
    if args.noise_rule == "A":
        payload["best_noise_a"] = min(
            payload["scores"], key=lambda key: payload["scores"][key]["d_pac"]
        )
    else:
        payload["frozen_noise_b_candidate"] = next(iter(payload["scores"]))
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "out": str(output),
                "best_noise_a": payload.get("best_noise_a"),
                "frozen_noise_b_candidate": payload.get("frozen_noise_b_candidate"),
                "scores": {
                    key: row["d_pac"] for key, row in payload["scores"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
