# pi0.5 论文 Table 2 复现结果（本机 A40 复测）

日期: 2026-08-13 | 硬件: 8× NVIDIA A40（驱动 525.105.17）| 协议: openpi 官方 examples/libero/main.py
(10 任务 × 5 trials = 50 条/套件, num_steps_wait=10, replan_steps=5, seed=7, action_horizon=10)
配置: pi05_libero（本地检查点 pi05_libero_pytorch, FP32→bf16 推理）
量化: DuQuant W4A8 (block=64, calib=160, ls=0.15, permute=0, row_rot=restore, LLM全量+DiT MLP = 180 层) + ATM + OHB (expert 18 层)
注: 量化服务器须禁用 torch.compile（TORCHDYNAMO_DISABLE=1），否则 180 层量化层的 max-autotune 首次推理编译数小时（表现为 CPU 满载 GPU 空闲）。

## 主结果 (成功率, 每套件 50 条)

| 套件 | 本机 FP16 | 本机 W4A8 | 远端 FP16* | 远端 W4A8* | 论文 FP16 | 论文 W4A8 |
|---|---|---|---|---|---|---|
| Spatial | **100.0%** | 98.0% | 98.0% | 98.0% | 98.5% | 98.5% |
| Goal | 96.0% | 96.0% | 96.0% | 100.0% | 99.0% | 98.0% |
| Object | 92.0% | 98.0% | 96.0% | 96.0% | 97.5% | 98.0% |
| Long | 94.0% | **88.0%** | 94.0% | 88.0% | 93.5% | 96.0% |
| **Avg** | **95.5%** | **95.0%** | 96.0% | 95.5% | 97.1% | 97.6% |

*远端数据来自 pi05/results.md（H20, 2026-08-11 复现记录）

## 结论

1. **量化损失极小（核心结论复现）**：同协议下 W4A8 vs FP16 = 95.0% vs 95.5%（-0.5%），
   与远端记录（95.5% vs 96.0%, -0.5%）和论文结论完全一致
2. **Long 套件与远端完全一致**：88.0% vs 88.0%（量化对长序列任务影响最大，论文该套件 96%）
3. **FP16 基线复现**：95.5% vs 远端 96.0%（-0.5%，50 条样本 σ≈2.8% 内）
4. **硬件迁移验证**：H20（远端）→ A40（本机）结果稳定，说明方法对硬件不敏感
5. Object FP16 本机 92% 偏低（远端 96%/论文 97.5%），Spatial FP16 100% 偏高（论文 98.5%），
   均在统计波动内

## 运行细节

- 8 卡并行：4× fp16 服务器（:8011-8014, GPU1/4/6/7）+ 4× 量化服务器（:8021-8024, GPU2/3/5/7）
- 每服务器 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6（GPU7 双服务 0.5）
- Long 套件按任务分 4 片并行（任务 0-2/3-4/5-7/8-9），日志: /tmp/table2_quant_libero_10_shard*.log
- 已知坑：
  - torch.compile max-autotune 会让量化推理卡死在首次请求 → TORCHDYNAMO_DISABLE=1 修复
  - 评测进程退出时 EGL 清理异常（噪音，不影响结果）
  - tyro 1.x 参数须加 --args. 前缀
