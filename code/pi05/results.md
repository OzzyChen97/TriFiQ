# OpenPI π0.5 QuantVLA 量化复现结果

日期: 2026-08-11 | 硬件: NVIDIA H20 | 协议: openpi 官方 examples/libero/main.py
(10 任务 × 5 trials = 50 条/套件, num_steps_wait=10, replan_steps=5, seed=7, action_horizon=10, denoise 10 步)
配置: pi05_libero (openroboto-ai/pi05-libero-pytorch 权重) + QuantVLA 方法
量化: DuQuant W4A8 (block=64, calib=160, ls=0.15, permute=0, row_rot=restore, LLM全量+DiT MLP = 180 层) + ATM + OHB (expert 18 层)

## 主结果 (成功率, 每套件 50 条)

| 套件 | 我们 FP16 | 我们 W4A8 | 论文 FP16 | 论文 W4A8 |
|---|---|---|---|---|
| Spatial | 98.0% | 98.0% | 98.5% | 98.5% |
| Goal | 96.0% | 100.0% | 99.0% | 98.0% |
| Object | 96.0% | 96.0% | 97.5% | 98.0% |
| Long | 94.0% | 88.0% | 93.5% | 96.0% |
| **Avg** | **96.0%** | **95.5%** | **97.1%** | **97.6%** |

## 结论

1. **FP16 基线复现成功**：avg 96.0% vs 论文 97.1%（差 -1.1%，50 条样本 σ≈2.8% 统计误差内）
2. **W4A8 量化复现成功**：avg 95.5% vs 论文 97.6%（差 -2.1%，统计误差内；Long 差 8% 略大，
   可能因校准观测为随机图像，长序列激活分布偏差）
3. **量化精度损失极小**：同协议下 W4A8 vs FP16 = 95.5% vs 96.0%（-0.5%），
   与论文结论一致（论文量化版 97.6% 甚至超 FP16 97.1%）
4. **层数验证**：DuQuant dry-run = 180 层（PaliGemma LLM 126 + Gemma expert MLP 54），与论文一致
5. ATM/OHB 校准: 18 层 expert attention (144 heads)，等值单测通过（diff=0）

## 资源
- 权重: /data1/wubohan/openpi/checkpoints/pi05_libero_pytorch (14.5GB FP32)
- 量化代码: /data1/wubohan/openpi/openpi/src/openpi/quant/ (duquant_layers/preprocess + atm_pi05)
- Pack 缓存: /data1/wubohan/openpi/packs/pi05_w4a8_b64c160ls015/ (180 层)
- ATM/OHB JSON: /data1/wubohan/openpi/packs/atm_alpha_beta_pi05.json
- 评估日志: /tmp/pi0_eval_{fp16,w4a8}_*.log
- 环境: pi0 (python 3.11, torch 2.7.1, transformers 4.53.2+补丁, openpi)
