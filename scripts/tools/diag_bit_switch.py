#!/usr/bin/env python3
"""决定性测试：set_all_bits(2) vs (8) 时同一层的输出是否真的不同（GR00T-only）。

若 y2 ≈ y8（逐元素），说明 bit 切换未生效（缓存/包装 bug）；
若 y2 ≠ y8，说明 bit 生效，flatness 是 CKA/CS 对该层不敏感的问题。
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))
import torch
import numpy as np
from gr00t_v2_common import (load_policy, make_l1_obs, set_quant_env, strip_quant_env,
                             ensure_flash_attn_rpath, DEFAULT_INCLUDE, DEFAULT_EXCLUDE,
                             stack_obs)

MODEL = str(ROOT / "checkpoints/gr00t/libero-spatial")
PACKDIR = str(ROOT / "checkpoints/packs/gr00t/duquant_packed_libero_spatial_w4a8_b64c32ls015")

ensure_flash_attn_rpath()
strip_quant_env()
set_quant_env(DEFAULT_INCLUDE, DEFAULT_EXCLUDE, PACKDIR, bits_default=4, group=64,
              ls=0.15, act_pct=99.9, calib_steps=32, row_rot="restore")
policy = load_policy(MODEL, denoising_steps=8, device="cuda")
model = policy.model

from gr00t.quantization.duquant_layers import DuQuantLinear
def set_all(b):
    n = 0
    for _, m in model.named_modules():
        if isinstance(m, DuQuantLinear):
            m.weight_bits = b
            n += 1
    return n

rng = np.random.default_rng(0)
obs = make_l1_obs(rng)
norm = policy.apply_transforms(stack_obs([obs]))

def capture(name):
    out = {}
    for n, m in model.named_modules():
        if n == name:
            mod = m
    def hook(module, args, output):
        out["y"] = output.detach().float().cpu()
    h = mod.register_forward_hook(hook)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.get_action(norm)
    h.remove()
    return out["y"]

name = "backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj"
print("wrapped:", set_all(4))
set_all(2); y2 = capture(name)
set_all(8); y8 = capture(name)
rel = (y2 - y8).abs().mean() / (y8.abs().mean() + 1e-9)
print(f"y2 vs y8 相对差异: {float(rel):.6e}")
print(f"y2 范围 [{float(y2.min()):.4f}, {float(y2.max()):.4f}] | y8 范围 [{float(y8.min()):.4f}, {float(y8.max()):.4f}]")
print("BIT-SWITCH-EFFECTIVE" if rel > 1e-4 else "BIT-SWITCH-NOT-EFFECTIVE")
# 再对比 all-0 参照
set_all(0); y0 = capture(name)
rel0 = (y0 - y2).abs().mean() / (y0.abs().mean() + 1e-9)
print(f"y0 vs y2 相对差异: {float(rel0):.6e}")
print("DIAG DONE")
