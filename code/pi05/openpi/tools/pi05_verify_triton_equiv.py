#!/usr/bin/env python3
"""Verify the Triton fused DuQuantLinear path is numerically equivalent to the
eager path on the REAL pi0.5 quantized model (QuantVLA fast-variant check).

Protocol:
  1. Load the pi05_pretrain_human300 model with the QuantVLA quant recipe
     (eager; TORCHDYNAMO_DISABLE=1 as on the serving servers).
  2. Run enough synthetic forwards to freeze every layer's act scale.
  3. Infer once (seed-controlled) with the eager path -> actions_ref,
     capturing a few intermediate layer outputs via hooks.
  4. Flip OPENPI_DUQUANT_TRITON on every DuQuantLinear and infer the same
     obs again with the same seed -> actions_triton.
  5. Report max abs/relative differences.

Usage (openpi env):
  OPENPI_DUQUANT_PACKDIR=... OPENPI_ATM_ENABLE=1 OPENPI_ATM_ALPHA_PATH=... \
  OPENPI_OHB_ENABLE=1 TORCHDYNAMO_DISABLE=1 \
  python tools/pi05_verify_triton_equiv.py \
      --checkpoint checkpoints/robocasa/pi05_pretrain_human300_pytorch \
      --config pi05_pretrain_human300
"""

import argparse
import os
import sys

import numpy as np
import torch

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def make_obs(seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "observation/state": rng.random(12).astype(np.float32) * 0.2 - 0.1,
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/right_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "prompt": "turn on the electric kettle",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home1/gyy/vla/QuantVLA/checkpoints/robocasa/pi05_pretrain_human300_pytorch")
    parser.add_argument("--config", default="pi05_pretrain_human300")
    parser.add_argument("--warm-infers", type=int, default=30,
                        help="synthetic forwards to freeze act scales")
    args = parser.parse_args()

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.quant import enable_duquant_if_configured
    from openpi.quant import enable_pi05_atm_if_configured
    from openpi.quant.duquant_layers import DuQuantLinear

    print("Loading policy ...", flush=True)
    cfg = _config.get_config(args.config)
    import dataclasses
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint, pytorch_device="cuda")
    model = policy._model
    enable_duquant_if_configured(model)
    model.to("cuda")
    enable_pi05_atm_if_configured(model)

    layers = [m for m in model.modules() if isinstance(m, DuQuantLinear)]
    print(f"DuQuantLinear layers: {len(layers)}", flush=True)

    obs = make_obs()

    # 1. Freeze act scales (eager).
    print(f"Warming {args.warm_infers} infers to freeze act scales ...", flush=True)
    for i in range(args.warm_infers):
        policy.infer(obs)
        if (i + 1) % 10 == 0:
            print(f"  warm {i+1}/{args.warm_infers}", flush=True)
    frozen = sum(1 for l in layers if l._act_scale_initialized)
    print(f"act scales frozen: {frozen}/{len(layers)}", flush=True)

    # 2. Reference pass (eager) with hooks.
    hook_outs = {}
    def hook(layer_name):
        def fn(m, inp, out):
            hook_outs.setdefault(layer_name, []).append(out.detach().clone())
        return fn
    handles = [l.register_forward_hook(hook(f"layer_{i}")) for i, l in enumerate(layers)]
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    actions_ref = policy.infer(obs)["actions"]
    hook_ref = {k: v[0] for k, v in hook_outs.items()}
    for h in handles:
        h.remove()
    hook_outs.clear()

    # 3. Triton pass (same seed, same obs).
    for l in layers:
        l._triton_enabled = True
    handles = []
    handles = [l.register_forward_hook(hook(f"layer_{i}")) for i, l in enumerate(layers)]
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    actions_triton = policy.infer(obs)["actions"]
    hook_triton = {k: v[0] for k, v in hook_outs.items()}
    for h in handles:
        h.remove()

    # 4. Compare.
    a_ref = np.asarray(actions_ref)
    a_tri = np.asarray(actions_triton)
    print("\n=== RESULTS ===", flush=True)
    print(f"actions shape: ref={a_ref.shape} triton={a_tri.shape}", flush=True)
    print(f"actions max|diff| = {np.abs(a_ref - a_tri).max():.6e}", flush=True)
    if np.abs(a_ref).max() > 0:
        print(f"actions rel diff  = {np.abs(a_ref - a_tri).max() / np.abs(a_ref).max():.6e}", flush=True)
    print(f"actions exact match ratio: {np.mean(a_ref == a_tri):.4f}", flush=True)
    diverged = []
    for k in sorted(hook_ref, key=lambda x: int(x.split("_")[1])):
        r = hook_ref[k].float()
        t = hook_triton[k].float()
        rel = (r - t).abs().max() / r.abs().max().clamp_min(1e-8)
        if rel.item() > 1e-4:
            diverged.append((k, rel.item()))
    print(f"  first diverging layers (rel>1e-4): {diverged[:8]}", flush=True)
    for k in sorted(hook_ref, key=lambda x: int(x.split("_")[1])):
        r = hook_ref[k].float()
        t = hook_triton[k].float()
        rel = (r - t).abs().max() / r.abs().max().clamp_min(1e-8)
        if rel.item() > 1e-4:
            i = int(k.split("_")[1])
            print(f"  layer_{i}: {layers[i].name} in={layers[i].in_features} out={layers[i].out_features} "
                  f"bias={layers[i].bias is not None} rel={rel.item():.4e} maxdiff={(r-t).abs().max().item():.4e}", flush=True)

    ok = np.abs(a_ref - a_tri).max() < 1e-2
    print("EQUIVALENCE:", "PASS" if ok else "CHECK", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
