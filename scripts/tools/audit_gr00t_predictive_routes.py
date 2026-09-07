#!/usr/bin/env python3
"""Compare six predictive superset routes with ordinary static-mask models."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "scripts" / "tools"))

from gr00t_sensitivity_probe import run_rollouts  # noqa: E402
from gr00t_v2_common import (  # noqa: E402
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    ensure_flash_attn_rpath,
    load_policy,
    set_quant_env,
    strip_quant_env,
)
from quantvla_model_adapters import (  # noqa: E402
    gr00t_rollout_inputs,
    load_model_records,
)
from quantvla_outputimpact import identity_check, install_fp16_bypass  # noqa: E402
from quantvla_predictive_validity import (  # noqa: E402
    artifact,
    atomic_json,
    is_w4,
    protocol_attestation,
    require_protocol_attestation,
)
from gr00t.quantization.duquant_layers import DuQuantLinear  # noqa: E402


DEFAULT_DATA_CONFIG = "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig"


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def quant_env(plan: Path, pack: Path, hessian: Path) -> None:
    strip_quant_env()
    os.environ.pop("GR00T_PREDICTIVE_MASK_MANIFEST", None)
    set_quant_env(
        DEFAULT_INCLUDE, DEFAULT_EXCLUDE, str(pack), row_rot="0", act_dynamic=True
    )
    os.environ.update(
        {
            "GR00T_DUQUANT_FUSED": "1",
            "GR00T_DUQUANT_PLAN": str(plan),
            "GR00T_DUQUANT_HESSIAN_W4_PATH": str(hessian),
            "GR00T_DUQUANT_ABITS": "8",
            "GR00T_ATM_ENABLE": "0",
            "GR00T_OHB_ENABLE": "0",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=("atomic_seen", "composite_seen", "composite_unseen"), default="atomic_seen")
    parser.add_argument("--n-obs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.n_obs != 4:
        raise ValueError("equivalence audit is frozen at four observations")
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_protocol_attestation(manifest, source=str(manifest_path))
    chosen = []
    for swap_count in (1, 2, 4, 8, 12, 16):
        rows = [row for row in manifest["candidates"] if int(row["swap_count"]) == swap_count]
        if len(rows) != 10:
            raise ValueError(f"swap-count {swap_count} inventory drift")
        chosen.append(sorted(rows, key=lambda row: row["candidate_id"])[0])
    split = args.split
    source = manifest["split_sources"][split]
    checkpoint = Path(source["checkpoint"]["path"])
    hessian = Path(source["hessian_w4"]["path"])
    pack = Path(source["identity_pack"]["path"])
    buffer = Path(source["selection_buffer"]["path"])
    full_w4 = Path(manifest["full_w4_plan"]["path"])
    records, buffer_provenance = load_model_records(buffer, args.n_obs, model="gr00t")
    observations, noises = gr00t_rollout_inputs(records, noise_index=0)
    ensure_flash_attn_rpath()
    torch.manual_seed(0)

    quant_env(full_w4, pack, hessian)
    superset = load_policy(
        str(checkpoint), data_config=DEFAULT_DATA_CONFIG,
        denoising_steps=4, device=args.device,
    )
    layers = install_fp16_bypass(superset.model.named_modules(), module_type=DuQuantLinear)
    if len(layers) != 116:
        raise RuntimeError(f"superset route has {len(layers)} layers, expected 116")
    superset_actions = {}
    for candidate in chosen:
        plan = json.loads(Path(candidate["path"]).read_text(encoding="utf-8"))
        active = {name for name, row in plan["layers"].items() if is_w4(row)}
        for name, layer in layers.items():
            layer._outputimpact_fp16 = name not in active
        _, actions = run_rollouts(
            superset.model, superset, observations, noises, args.batch_size,
            return_physical=True,
        )
        superset_actions[candidate["candidate_id"]] = actions.clone()
    for layer in layers.values():
        layer._outputimpact_fp16 = True
    _, bypass_teacher = run_rollouts(
        superset.model, superset, observations, noises, args.batch_size,
        return_physical=True,
    )
    del superset, layers
    clear_cuda()

    strip_quant_env()
    for key in list(os.environ):
        if key.startswith("GR00T_DUQUANT_") or key in (
            "GR00T_ATM_ENABLE", "GR00T_OHB_ENABLE", "GR00T_ERRORFOLD_PATH"
        ):
            os.environ.pop(key, None)
    torch.manual_seed(0)
    teacher = load_policy(
        str(checkpoint), data_config=DEFAULT_DATA_CONFIG,
        denoising_steps=4, device=args.device,
    )
    _, native_teacher = run_rollouts(
        teacher.model, teacher, observations, noises, args.batch_size,
        return_physical=True,
    )
    fp16_identity = identity_check(native_teacher, bypass_teacher, atol=2e-3, rtol=2e-3)
    del teacher
    clear_cuda()

    comparisons = {}
    for candidate in chosen:
        plan = Path(candidate["path"])
        subset = Path(args.out).expanduser().resolve().parent / "equivalence_hessians" / (
            f"{split}_{candidate['candidate_id']}.npz"
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/tools/materialize_full_context_hessian_subset.py"),
                "--parent", str(hessian), "--plan", str(plan),
                "--model", "gr00t", "--out", str(subset),
            ],
            cwd=REPO,
            check=True,
        )
        quant_env(plan, pack, subset)
        torch.manual_seed(0)
        static = load_policy(
            str(checkpoint), data_config=DEFAULT_DATA_CONFIG,
            denoising_steps=4, device=args.device,
        )
        _, static_actions = run_rollouts(
            static.model, static, observations, noises, args.batch_size,
            return_physical=True,
        )
        comparisons[candidate["candidate_id"]] = {
            "swap_count": int(candidate["swap_count"]),
            "plan": artifact(plan),
            "hessian_subset": artifact(subset),
            "identity": identity_check(
                static_actions, superset_actions[candidate["candidate_id"]],
                atol=2e-3, rtol=2e-3,
            ),
        }
        del static
        clear_cuda()

    payload = {
        "schema_version": 1,
        "kind": "gr00t_predictive_route_equivalence_audit",
        "predictive_validity_protocol": protocol_attestation(),
        "manifest": artifact(manifest_path),
        "split": split,
        "selection_buffer": artifact(buffer),
        "selection_buffer_provenance": buffer_provenance,
        "n_obs": args.n_obs,
        "noise_rule": "A",
        "atol": 2e-3,
        "rtol": 2e-3,
        "fp16_bypass_identity": fp16_identity,
        "static_mask_comparisons": comparisons,
        "passed": fp16_identity["passed"] and all(
            row["identity"]["passed"] for row in comparisons.values()
        ),
    }
    atomic_json(Path(args.out), payload)
    print(json.dumps({"out": str(Path(args.out).resolve()), "passed": payload["passed"]}, indent=2))


if __name__ == "__main__":
    main()
