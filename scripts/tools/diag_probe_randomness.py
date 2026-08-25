#!/usr/bin/env python3
"""诊断：定位 probe 参考 vs 干预之间与 bit 无关的随机性来源（GR00T-only，临时脚本）。

假设：
  H1) DiT 层输出依赖 get_action 的内部随机噪声 → 参考/干预轨迹不同 → CKA 基线低。
  H2) LLM 层输出在两遍相同输入下应确定（bf16/eval）；若不确定 → 有隐藏随机源。
测试：量化模型 all-bits=0，同一 obs 跑两次，对比 LLM 层输出的 CKA/CS；
      再把层 i 设 b=4 跑一次，对比 all-0 参照 → CKA 应 >>0.9。
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))
import torch
import numpy as np
from gr00t_v2_common import (load_policy, make_l1_obs, set_quant_env, strip_quant_env,
                             ensure_flash_attn_rpath, DEFAULT_INCLUDE, DEFAULT_EXCLUDE)
from gr00t.quantization.kernel_scores import LayerScoreBank

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
    for _, m in model.named_modules():
        if isinstance(m, DuQuantLinear):
            m.weight_bits = b

rng = np.random.default_rng(0)
obs = make_l1_obs(rng)
obs_batched = [obs]  # 打包为 batch 列表，capture 内 stack

def capture_layer(name, noise_fixed=None):
    """一次 get_action，返回指定层输出（LLM 层只与 obs 有关）。"""
    out = {}
    mod = None
    for n, m in model.named_modules():
        if n == name:
            mod = m
    def hook(module, args, output):
        out["y"] = output.detach().float().reshape(-1, output.shape[-1]).cpu()
    h = mod.register_forward_hook(hook)
    from gr00t_v2_common import stack_obs
    norm = policy.apply_transforms(stack_obs([obs]))  # B=1 批量
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.get_action(norm)
    h.remove()
    return out["y"]

set_all(0)
name = "backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj"
y1 = capture_layer(name)
y2 = capture_layer(name)
bank = LayerScoreBank(name, max_tokens=4096)
bank.accumulate_ref(y1)
bank.finalize_ref()
s_same = bank.evaluate(y2)
print(f"[H2] 同一 obs、同 bits=0 两次: CKA={s_same['cka']:.6f} CS={s_same['cs']:.6f} (期望 CKA≈1, CS≈0)")

# 单层干预 b=4
from gr00t.quantization.duquant_layers import DuQuantLinear
for n, m in model.named_modules():
    if n == name and isinstance(m, DuQuantLinear):
        m.weight_bits = 4
y3 = capture_layer(name)
s_b4 = bank.evaluate(y3)
print(f"[干预] 该层 b=4 vs 参照0: CKA={s_b4['cka']:.6f} CS={s_b4['cs']:.6f} (期望 CKA>0.9)")

# DiT 层同噪 vs 异噪
name_dit = "action_head.model.transformer_blocks.0.ff.net.0.proj"
set_all(0)
bank2 = LayerScoreBank(name_dit, max_tokens=4096)
bank2.accumulate_ref(capture_layer(name_dit))
bank2.finalize_ref()
y_d2 = capture_layer(name_dit)  # 内部噪声不同
s_dit = bank2.evaluate(y_d2)
print(f"[H1] DiT 层异噪两遍: CKA={s_dit['cka']:.6f} CS={s_dit['cs']:.6f} (若轨迹噪声主导 → CKA 低)")
print("DIAG DONE")
