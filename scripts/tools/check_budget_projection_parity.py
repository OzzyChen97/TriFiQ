#!/usr/bin/env python3
"""GPU smoke parity: offline full-W4 bypass masks versus native deployment masks.

Uses frozen teacher-context observations, not new evaluation outcomes. Tolerances
are fixed to the existing identity_check defaults (atol=rtol=0.002). Each split
checks every new plan on four contexts; this is a finite smoke check, not proof
of equality over all possible inputs. Writes no precision or selection changes.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/budget_projection_comparison_v1"
sys.path.insert(0, str(ROOT / "scripts/tools"))


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gr00t", "pi05"), required=True)
    parser.add_argument("--split", default="atomic_seen")
    args = parser.parse_args()
    if args.model == "pi05":
        sys.path.insert(0, str(ROOT / "code/pi05/openpi/src"))
        sys.path.insert(0, str(ROOT / "code/pi05/openpi/packages/openpi-client/src"))
    else:
        sys.path.insert(0, str(ROOT / "code"))
    import numpy as np
    import torch
    from quantvla_cross_model_protocol import sha256_file
    from quantvla_model_adapters import load_model_records, gr00t_rollout_inputs
    from quantvla_outputimpact import install_fp16_bypass, identity_check, atomic_json
    torch.set_num_threads(2)
    torch.manual_seed(0)
    manifest = read(OUT / "manifest.json")
    model = manifest["models"][args.model]
    for item in model["plans"].values():
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError("Frozen candidate drift")
    original = read(model["sources"][0]["path"])
    base_plan = Path(original["base_full_w4_plan"])
    if args.model == "gr00t":
        from gr00t_v2_common import DEFAULT_INCLUDE, DEFAULT_EXCLUDE, ensure_flash_attn_rpath, load_policy, set_quant_env, strip_quant_env
        from gr00t_sensitivity_probe import run_rollouts
        from gr00t.quantization.duquant_layers import DuQuantLinear
        calibration = ROOT / f"runs/archive_previous_versions/errorfold_v3_15x20/calibration/gr00t/{args.split}"
        checkpoint = ROOT / f"checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/{args.split}/checkpoint-60000"
        buffer = ROOT / f"runs/full_context_v2/selection_buffer_splits/selection_buffer_{args.split}.npz"
        records, provenance = load_model_records(buffer, 4, model="gr00t")
        observations, noises = gr00t_rollout_inputs(records, noise_index=0)
        ensure_flash_attn_rpath()
        def load(plan, weights):
            strip_quant_env()
            set_quant_env(DEFAULT_INCLUDE, DEFAULT_EXCLUDE, str(calibration / "identity_pack"), row_rot="0", act_dynamic=True)
            os.environ.update({"GR00T_DUQUANT_FUSED": "1", "GR00T_DUQUANT_PLAN": str(plan),
                               "GR00T_DUQUANT_HESSIAN_W4_PATH": str(weights), "GR00T_ATM_ENABLE": "0", "GR00T_OHB_ENABLE": "0"})
            return load_policy(str(checkpoint), data_config="examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig", denoising_steps=4, device="cuda")
        def run(policy):
            return run_rollouts(policy.model, policy, observations, noises, 1, return_physical=True)[1]
        def network(policy):
            return policy.model
    else:
        from pi05_score_configs_against_fp16 import configure_quant, load_policy
        from pi05_full_context_adapter import run_records
        from openpi.quant import enable_duquant_if_configured
        from openpi.quant.duquant_layers import DuQuantLinear
        calibration = ROOT / "runs/archive_previous_versions/errorfold_v4_iter/calibration/pi05_full_w4"
        checkpoint = ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch"
        buffer = ROOT / "runs/full_context_v2/selection_buffer.npz"
        records, provenance = load_model_records(buffer, 4, model="pi05")
        def load(plan, weights):
            payload = read(plan)
            wrapped = sum(not r.get("skip", False) and int(r.get("bits", 0) or 0) == 4 for r in payload["layers"].values())
            configure_quant(spec={"plan":plan,"a8":calibration / "unused.npz", "hessian_w4":weights,"wrapped":wrapped,"atm":None},
                            pack_dir=calibration / "identity_pack",artifact_buffer_hash="c4de41b304d33b15240538a45006f1cf560fb050d0641610a9d38ede91d1999c",
                            strict_artifacts=True,activation_mode="dynamic_a8",flow_steps=4)
            policy = load_policy(checkpoint,"cuda")
            enable_duquant_if_configured(policy._model)
            policy._model.to("cuda")
            return policy
        def run(policy):
            return run_records(policy, records, "cuda", noise_index=0, flow_steps=4)[1]
        def network(policy):
            return policy._model
    parent = calibration / "hessian_w4.npz"
    subsets = {}
    for identifier, row in model["plans"].items():
        subset = OUT / "hessian" / identifier / args.split / "hessian_w4.npz"
        subprocess.run([sys.executable,str(ROOT / "scripts/tools/materialize_full_context_hessian_subset.py"),
                        "--parent",str(parent),"--plan",row["path"],"--model",args.model,"--out",str(subset)],check=True)
        subsets[identifier] = subset
    report = {"kind":"projection_native_deployment_smoke_parity", "manifest_sha256":sha256_file(OUT/"manifest.json"),
              "code_sha256":sha256_file(Path(__file__)), "model":args.model,"split":args.split,
              "buffer":str(buffer),"buffer_sha256":provenance["sha256"],"n_contexts":4,
              "base_plan_sha256":sha256_file(base_plan),"parent_hessian_sha256":sha256_file(parent),
              "atol":0.002,"rtol":0.002,"complete":False,"checks":{}}
    path = OUT / "parity" / f"{args.model}_{args.split}.json"
    atomic_json(path, report)
    policy = load(base_plan,parent)
    layers = install_fp16_bypass(network(policy).named_modules(),module_type=DuQuantLinear)
    reference = {}
    for identifier,row in model["plans"].items():
        protected = set(read(row["path"])["protected_layers"])
        for name,module in layers.items():
            module._outputimpact_fp16 = name in protected
        reference[identifier] = np.array(run(policy),copy=True)
        print("OFFLINE",identifier,flush=True)
    del policy, layers, module
    gc.collect();torch.cuda.empty_cache()
    for identifier,row in model["plans"].items():
        policy = load(Path(row["path"]),subsets[identifier])
        if args.model == "gr00t":
            from gr00t.quantization import finalize_real_quant
        else:
            from openpi.quant import finalize_real_quant
        residency = finalize_real_quant(network(policy))
        if residency.get("packed_low_bit_residency") is not True:
            raise RuntimeError("Deployment parity requires production packed residency")
        actions = run(policy)
        try:
            check = identity_check(reference[identifier],actions,atol=0.002,rtol=0.002)
        except RuntimeError as error:
            report["failed_candidate"]=identifier
            report["failure"]=str(error)
            atomic_json(path,report)
            raise
        report["checks"][identifier] = {**check,"packed_residency":residency,"plan_sha256":row["sha256"],"hessian_sha256":sha256_file(subsets[identifier])}
        atomic_json(path,report)
        print("PARITY_PASS",identifier,check,flush=True)
        del policy,actions
        gc.collect();torch.cuda.empty_cache()
    report["complete"]=True
    atomic_json(path,report)
    print("PARITY_COMPLETE",path,flush=True)


if __name__ == "__main__":
    main()