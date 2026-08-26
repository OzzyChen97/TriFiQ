#!/usr/bin/env python3
"""Audit two already-frozen ErrorFold candidates on held-out paired noise B."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


REPO = Path(__file__).resolve().parents[2]
OPENPI = REPO / "code/pi05/openpi"
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "packages/openpi-client/src"))
sys.path.insert(0, str(REPO / "scripts/tools"))

from quantvla_cross_model_protocol import (  # noqa: E402
    PROTOCOL,
    protocol_artifact,
    protocol_attestation,
    sha256_file,
)
from quantvla_metric_protocol import summarize_noise_a_b  # noqa: E402
from quantvla_model_adapters import (  # noqa: E402
    canonical_physical_chunk,
    gr00t_rollout_inputs,
    load_model_records,
    record_metadata,
)


def _clear() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gr00t_actions(args: argparse.Namespace, records, correction: Path | None):
    from gr00t.quantization import finalize_real_quant
    from gr00t.quantization.duquant_layers import load_act_scales
    from gr00t_sensitivity_probe import run_rollouts
    from gr00t_v2_common import (
        DEFAULT_EXCLUDE,
        DEFAULT_INCLUDE,
        load_policy,
        set_quant_env,
        strip_quant_env,
    )

    strip_quant_env()
    if correction is not None:
        set_quant_env(DEFAULT_INCLUDE, DEFAULT_EXCLUDE, args.pack_dir, row_rot="0")
        os.environ.update(
            {
                "QUANTVLA_ADAPTER_ONLY": "1",
                "GR00T_DUQUANT_PLAN": args.plan,
                "GR00T_DUQUANT_FUSED": "1",
                "GR00T_DUQUANT_ACT_SCALE_PATH": args.a8,
                "GR00T_DUQUANT_HESSIAN_W4_PATH": args.hessian_w4,
                "GR00T_ERRORFOLD_PATH": str(correction),
                "GR00T_ATM_ENABLE": "1",
                "GR00T_OHB_ENABLE": "1",
                "GR00T_ATM_ALPHA_PATH": str(correction),
                "GR00T_ATM_SCOPE": "dit",
                "GR00T_OHB_SCOPE": "dit",
                "GR00T_ATM_PER_STEP": "0",
                "GR00T_ATM_APPLICATION": "fold_q_weight",
                "GR00T_OHB_APPLICATION": "fold_o_weight_perhead",
            }
        )
    policy = load_policy(
        args.checkpoint,
        data_config="examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig",
        denoising_steps=4,
        device=args.device,
    )
    if correction is not None:
        load_act_scales(
            policy.model,
            args.a8,
            require={
                "plan_sha256": sha256_file(args.plan),
                "calibration_buffer_sha256": PROTOCOL["data"]["calibration_buffer"][
                    "sha256"
                ],
            },
        )
        residency = finalize_real_quant(policy.model)
        runtime = getattr(policy.model, "_gr00t_atm_runtime", {})
        if not residency["packed_low_bit_residency"] or not runtime.get("selector_free"):
            raise RuntimeError("GR00T frozen ErrorFold runtime attestation failed")
    observations_a, noises_a = gr00t_rollout_inputs(records, noise_index=0)
    observations_b, noises_b = gr00t_rollout_inputs(records, noise_index=1)
    _, actions_a = run_rollouts(
        policy.model,
        policy,
        observations_a,
        noises_a,
        args.batch_size,
        return_physical=True,
    )
    _, actions_b = run_rollouts(
        policy.model,
        policy,
        observations_b,
        noises_b,
        args.batch_size,
        return_physical=True,
    )
    del policy
    _clear()
    return actions_a, actions_b


def _pi05_actions(args: argparse.Namespace, records, correction: Path | None):
    from openpi.quant import (
        enable_duquant_if_configured,
        enable_pi05_atm_if_configured,
        finalize_real_quant,
    )
    from pi05_score_configs_against_fp16 import configure_quant, load_policy
    from pi05_sensitivity_probe import run_records

    spec: dict[str, Any] = {
        "plan": args.plan,
        "a8": args.a8,
        "wrapped": len(json.loads(Path(args.plan).read_text(encoding="utf-8"))["layers"]),
        "hessian_w4": args.hessian_w4,
    }
    if correction is None:
        from pi05_score_configs_against_fp16 import configure_base

        configure_base()
    else:
        spec.update(
            {
                "errorfold": str(correction),
                "atm": str(correction),
                "atm_enable": True,
                "ohb_enable": True,
                "atm_application": "fold_q_weight",
                "ohb_application": "fold_o_weight_perhead",
            }
        )
        configure_quant(
            spec=spec,
            pack_dir=Path(args.pack_dir),
            artifact_buffer_hash=PROTOCOL["data"]["calibration_buffer"]["sha256"],
            strict_artifacts=True,
        )
    policy = load_policy(Path(args.checkpoint), args.device)
    if correction is not None:
        enable_duquant_if_configured(policy._model)
        policy._model.to(args.device)
        enable_pi05_atm_if_configured(policy._model)
        residency = finalize_real_quant(policy._model)
        if not residency["packed_low_bit_residency"]:
            raise RuntimeError("pi0.5 frozen ErrorFold residency failed")
    _, actions_a, _ = run_records(policy, records, args.device, noise_index=0)
    _, actions_b, _ = run_records(policy, records, args.device, noise_index=1)
    del policy
    _clear()
    return (
        canonical_physical_chunk(actions_a, model="pi05"),
        canonical_physical_chunk(actions_b, model="pi05"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=PROTOCOL["models"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--a8", required=True)
    parser.add_argument("--hessian-w4", required=True)
    parser.add_argument("--selected", action="append", required=True)
    parser.add_argument(
        "--buffer", default=str(protocol_artifact("selection_buffer", verify=False))
    )
    parser.add_argument("--n-obs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [Path(value).resolve() for value in args.selected]
    if len(selected) != 2:
        raise ValueError("noise-B audit requires exactly D_func and D_PAC-v2 candidates")
    for path in selected:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("selection") or {}).get("uses_rollout_success") is not False:
            raise ValueError(f"candidate was not frozen without success labels: {path}")
    records, provenance = load_model_records(
        args.buffer, args.n_obs, model=args.model
    )
    if any(len(row["noises"]) < 2 for row in records):
        raise ValueError("selection buffer does not contain held-out noise B")
    adapter = _gr00t_actions if args.model == "gr00t" else _pi05_actions
    teacher_a, teacher_b = adapter(args, records, None)
    candidates = {}
    heldout = {}
    for path in selected:
        candidate_a, candidate_b = adapter(args, records, path)
        result = summarize_noise_a_b(
            teacher_a,
            candidate_a,
            teacher_b,
            candidate_b,
            record_metadata(records),
        )
        candidates[path.stem] = result
        heldout[path.stem] = result["heldout_noise_B"]
    payload = {
        "schema_version": 3,
        "kind": "errorfold_v3_noise_b_audit",
        "cross_model_protocol": protocol_attestation(),
        "model_adapter": args.model,
        "selection_buffer": provenance,
        "selection_frozen_before_noise_B": True,
        "noise_B_used_for_selection": False,
        "heldout_noise_B": heldout,
        "candidates": candidates,
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
