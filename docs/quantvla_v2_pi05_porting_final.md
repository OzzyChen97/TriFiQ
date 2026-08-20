# GDSQ-VLA / QuantVLA → π0.5 最终移植与评测规范

> 状态：GR00T N1.5 对齐实现已冻结；四配置 sanity 已通过；正式 paired50 manifest 正在生成。  
> 对齐日期：2026-08-20  
> 唯一正式根目录：`runs/pi05_gdsq_gr00t_aligned`  
> 历史 `pretrain / execute-5 / d10 / PCG64` π0.5 rows 全部作废，不得与本轮结果拼接。

## 1. 最终原则

π0.5 的正式测试完全使用 GR00T N1.5 final 的 benchmark 口径、随机性协议、统计方法、
W4/FP16 搜索预算、selector 和 true-mixed TopK 裁决。只保留模型原生且无法机械统一的输入输出
差异，不移植 GR00T 的层 mask、A8 scale 或 ATM/OHB 系数。

正式比较固定为四行：

| ID | 权重/激活 | ATM/OHB | 运行时硬断言 |
|---|---|---|---|
| `fp16` | 严格 FP16 | 关闭 | 0 quant wrappers；458/458 Linear 为 FP16 |
| `quantvla_w4a8_atmohb` | 180/180 candidate layers W4A8 | static | 180 wrappers；18 attention layers |
| `gdsq_vla` | final W4/FP16 plan + A8 | 关闭 | 80 W4 wrappers；100 native-FP16 layers |
| `gdsq_vla_atmohb` | 与 `gdsq_vla` 完全相同的 plan/A8 | static | 80 wrappers；18 attention layers |

ATM/OHB 是独立消融，不预设一定提升。若 `gdsq_vla_atmohb` 不优于 `gdsq_vla`，默认正式方法
关闭 ATM/OHB；这与 GR00T final 上 ATM/OHB 为负增益的观察一致。

## 2. 与 GR00T N1.5 完全一致的正式协议

| 项目 | 冻结值 |
|---|---|
| benchmark | RoboCasa365 |
| split | `target` |
| task sets | 18 atomic_seen + 16 composite_seen + 16 composite_unseen |
| trial seeds | 每任务显式 `0–49` |
| 总规模 | 2,500 episodes/config；四配置共 10,000 |
| environment | 每 trial fresh env；render/EGL 开启 |
| horizon | RoboCasa365 官方 per-task horizon |
| action execution | 每次执行 16 actions 后 replan |
| denoising/flow steps | 4 |
| termination | success、terminated 或 truncated 立即结束；否则到官方 horizon |
| resume key | `(config, task_set, task, seed)` |
| crash | 不写成失败；fresh env 重建并恢复 |

配对 diffusion 初始噪声为：

```text
sha256(task, env_seed, replan_index) / torch-cpu-normal-v1
```

精确 seed 字节串为：

```text
b"quantvla-robocasa365-v1\0" + task + b"\0" + seed + b"\0" + replan_index
```

取 SHA256 前 8 bytes，转换后截为 signed 63-bit；使用 CPU `torch.Generator` 和
`torch.randn(..., dtype=float32)` 生成 `[50,32]` noise。端口、GPU、并发顺序和配置不参与 seed。

统计与 GR00T final 相同：

- per-task SR、task-set macro SR、50-task macro SR、episode SR；
- 成功/失败步数与 policy/env wall time；
- 10,000 次 task-cluster bootstrap 95% CI；
- task-level paired permutation test + Holm 多重比较校正；
- episode-level McNemar 仅作辅助。

预注册比较为：

1. `gdsq_vla` vs `fp16`；
2. `gdsq_vla` vs `quantvla_w4a8_atmohb`；
3. `gdsq_vla_atmohb` vs `gdsq_vla`；
4. `gdsq_vla_atmohb` vs `quantvla_w4a8_atmohb`。

## 3. π0.5 必须保留的原生差异

目标 checkpoint 为：

```text
checkpoints/robocasa/pi05_pretrain_human300_pytorch
```

checkpoint SHA256：

```text
4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c
```

π0.5 原生 action tensor 为 `[50,32]`，RoboCasa 环境实际使用前 12 维；GR00T 对齐的
execute-16 不改变 checkpoint 的 50-step action horizon。输入仍使用 checkpoint 专属三路 224px 图像、
16-D state 顺序和 z-score normalization。不得为了表面一致而改成 GR00T 的 image/state encoder。

服务端显式设置 `OPENPI_MODEL_DTYPE=float16`。允许 75,776 个 normalization 标量保留 FP32
稳定岛；全部 458 个 Linear 权重必须为 FP16。训练配置对象中原始 `dtype=bfloat16` 不代表正式
runtime 精度，正式判据是 runtime attestation：

```text
resolved=float16
linear_layers_by_weight_dtype={float16: 458}
parameter_elements_by_dtype={float16: 3,616,681,744; float32: 75,776}
```

## 4. GR00T final 算法的 π0.5 架构适配

### 4.1 Candidate inventory 与 buffer

- 180 个 candidate Linear；
- PaliGemma language path 126 层；Gemma action expert MLP 54 层；
- DuQuant block64，`permute=false`，row rotation restore；
- canonical calibration buffer：RoboCasa365 seed 0，256 observations；
- A8：32 batches × 8，percentile 99.9，4 flow steps；
- ATM/OHB：16 observations × batch 8，4 flow steps。

canonical buffer SHA256：

```text
c4de41b304d33b15240538a45006f1cf560fb050d0641610a9d38ede91d1999c
```

### 4.2 Sensitivity 与 functional metric

所有 180 层均在两套 canonical noise、`n_obs=16` 下重新 probe。单层介入是 W4；其余 candidate
layers 使用精确 native-FP16 bypass。最终功能距离为：

```text
D_func = D_exec16 + (16/50) * D_chunk50
```

其中只评估部署的 12 个 physical action dimensions；12:32 padded dimensions 不参与评分。

本轮 d4 sensitivity 与真实 `D_func` 的相关性：

| 指标 | Spearman | Pearson |
|---|---:|---:|
| `1 - action CKA` | 0.641 | 0.756 |
| CS | 0.572 | 0.182 |
| `1 - local CKA` | 0.235 | 0.209 |

action CKA 仍是更稳定的主信号；CS 只保留为 16:1 的低权重辅助项。

### 4.3 Ratio 决策

CKA:CS 比例不在 π0.5 RoboCasa closed-loop 上重新调参。正式 port 精确转移 GR00T N1.5 final
四任务开发集选择的 `16:1`：

```text
GR00T development task-macro SR = 80.5% (161/200)
selected ratio = 16:1
pi0.5 ratio-tuning rollouts performed = false
```

π0.5 只重新进行架构相关的 sensitivity、min-max selector 和 true-mixed TopK。这样既保持算法
一致，也避免用 π0.5 正式测试分布二次调超参。

### 4.4 W4/FP16 搜索与 TopK

搜索空间严格为 W4/native-FP16，static-byte budget 与 GR00T final 相同：uniform-W6 reference。
因此 layer count 不是预算；不同 shape 的 layer 对字节预算贡献不同。

新 d4 selector 的 canonical candidate 为 69 W4 + 111 FP16，但这不是最终部署 plan。完整
true-mixed TopK 结果为：

| TopK | 来源 | W4 layers | `D_func` ↓ |
|---:|---|---:|---:|
| 0 | MILP | 69 | 0.360427 |
| 1 | flip12 | 77 | 0.254985 |
| 2 | flip16 | 81 | 0.436619 |
| 3 | flip16 | **80** | **0.225572** |
| 4 | flip20 | 84 | 0.392457 |
| 5 | flip20 | 83 | 0.319903 |
| 6 | CS-only boundary | 107 | 0.690119 |

最终按“最低 `D_func`；5% relative tie set；canonical proxy tie-break”选择 TopK 3：

```text
80 W4 + 100 native FP16
```

80 个 W4 layers 覆盖 1,788,870,656 / 2,208,301,056 = 81.01% 的 candidate 参数；language
candidate 参数覆盖 85.60%，action-expert candidate 参数覆盖 40.74%。这说明 80/180 的层数比例
不能解释为仅量化 45% 参数。

candidate-scope paper-style static bytes 为 2,005,966,848，对 FP16 candidate scope 的
4,416,602,112 为 2.20× 压缩，并低于 uniform-W6 budget 2,042,542,080。

## 5. 冻结产物

| 产物 | SHA256 |
|---|---|
| canonical buffer | `c4de41b304d33b15240538a45006f1cf560fb050d0641610a9d38ede91d1999c` |
| d4 inventory file | `900084089317fec0e8389d8f3f61dba5efd94d1f083db7a3cc5813e7f3e3a79f` |
| merged d4 sensitivity | `9a135507acc4c25bae4df7711cd55a7ad1a91e3749955f1b6e1c64379df56373` |
| 16:1 selector | `d17537193476eac68aa3e6fc6211524906e10ba4f952f41a6149b2d905a34584` |
| true-mixed TopK report | `b4b429dd292edb4b6d086fc2cc21ae73acecf351d9148c69af18328a6f446f2d` |
| frozen GDSQ plan | `fff35a05d09b3e8f93c3c63d28a224dd092356d04dd0aab8dc495cdf8e470f6c` |
| uniform full-W4 plan | `0a970fd109cca226b758907922ccc061d300bdeaab766b3451c7ed3d4888266a` |
| GDSQ A8 scale | `57666af3264bb94367a4b6c5062c54a15e9f036b29ca795b23f71dea45757887` |
| full-W4 A8 scale | `25a02ab4c5e517fcd514d7fd5d1747583346b09ec6d370a9abee2f7b4c56b7ad` |
| GDSQ per-head ATM/OHB | `742b9ed25733408a431b3124f7f4febe05aaefa9b57429950bf1c56a5095577e` |
| full-W4 per-head ATM/OHB | `7e7d3f54b6e93271aac477c5986f164929a89a62b8e23a77d1ed3f52d7f03531` |
| ratio-transfer selection | `b458cbe335e5f0c0c2f80d2da44efaee0026664d1247cba7d471f8ade0cbeb4c` |
| ratio-transfer provenance | `afe572a12fa8f121fcd2bf438c0cf93e286830d78be7cdfa8dadf2c26a0cfaf6` |
| four-server websocket smoke | `a6ce5dbea1f5996027e81c47319e3bbd8c8ab0e40481cdbb35e49ccba55ad702` |

A8 sidecar 对每个配置强制校验：plan/checkpoint/buffer hash、wrapped layers、A8 percentile 99.9、
32 batches、batch size 8、256 observations 和 denoising=4。ATM/OHB 强制校验 plan/A8/pack/buffer
hash、18 attention layers、frames=16、batch=8、flow=4 和 `per_head_pre_projection`。

## 6. Correctness sanity

四个 websocket server 均通过：相同 paired noise 重复两次 inference bitwise identical；输出均为
finite `[50,12]`。runtime attestation 为：

| 配置 | wrappers | ATM/OHB | plan/A8 关系 |
|---|---:|---|---|
| FP16 | 0 | off | 无 quant artifact |
| full-W4 | 180 | on | full-W4 专属 |
| GDSQ | 81 | off | frozen GDSQ plan/A8 |
| GDSQ+ATM/OHB | 81 | on | 与 GDSQ plan/A8 hash 完全相同 |

OpenCabinet seed 0 的 `target/execute-16/d4` fresh-env smoke：

| 配置 | success | steps |
|---|---|---:|
| FP16 | yes | 225 |
| full-W4 + ATM/OHB | yes | 237 |
| GDSQ + ATM/OHB | yes | 244 |
| GDSQ | yes | 248 |

单 seed 只证明协议、环境、动作转换和恢复链路可运行，不能作为 SR 排名证据。

## 7. Efficiency 报告边界

当前闭环 accuracy runtime 使用 `fake_quant_fp16_gemm`：它真实注入 W4/A8 数值误差，但没有
packed int4/int8 GEMM，且 `integer_gemm=false`、`packed_low_bit_residency=false`。因此：

- paper-style static bytes 可以报告为理论 storage；
- 当前 eager CUDA memory 和 latency 必须按 runtime 实测；
- 不得把 2.20× candidate-scope 理论压缩写成当前 eager 显存压缩或部署加速；
- fast packed kernel 必须单独通过数值等价后才可形成 efficiency 行，不能与 accuracy rows 混跑。

## 8. 正式运行、恢复与结果发布

正式目录：

```text
runs/pi05_gdsq_gr00t_aligned/official_target_paired50
```

GPU1–7 全部用于实验，GPU0 和无关进程不触碰。调度按配置顺序使用 7 个 policy replicas；每个
replica 对应两个互斥 seed halves `0–24`、`25–49`，task shard 为 0–6。资源重分片不改变任务、
seed、noise 或结果 key。

命令：

```bash
scripts/run_pi05_faithful_waves.sh prepare
scripts/run_pi05_faithful_waves.sh run-all
scripts/run_pi05_faithful_waves.sh status
scripts/run_pi05_faithful_waves.sh stop
```

`prepare` 会为四配置 × 七 GPUs 实际启动 server，写出 28 个 runtime attestations，再生成
immutable manifest。`run-all` 自动按配置切换下一波；每个 worker 自动按
`atomic_seen → composite_seen → composite_unseen` 切换，并按 committed key 恢复。

只有 10,000/10,000 episodes、artifact graph、runtime metadata、完整 key matrix 和严格统计全部
通过，才允许把结果写入论文主表。partial SR 只能标为进度，不能作为正式结论。

## 9. GR00T N1.5 背景结果

GR00T final 每配置 2,500 episodes：

| 配置 | Atomic | Composite-Seen | Composite-Unseen | 50-task macro SR |
|---|---:|---:|---:|---:|
| FP16 | 75.6% | 41.9% | 45.3% | 55.1% |
| 原版 W4A8 + ATM/OHB | 49.6% | 19.6% | 19.8% | 30.4% |
| GDSQ final | 69.0% | 40.6% | 40.4% | 50.8% |
| GDSQ final + ATM/OHB | 66.7% | 39.1% | 40.3% | 49.4% |

GR00T 中 GDSQ final 相对原版 W4A8 为 `+20.32pp`，相对 FP16 为 `−4.32pp`；ATM/OHB 相对
GDSQ 为 `−1.36pp`。这些数字只作跨模型背景，不与 π0.5 SR 合并统计。

LIBERO v1.4 四套件 macro Avg `89.2%`（W6 `88.2%`、v1.3 `85.2%`）同样只作跨 benchmark
背景，不与 RoboCasa365 SR 混合。

## 10. 已作废的 π0.5 结果

以下任一特征都表示 legacy diagnostic，而非正式证据：

- `split=pretrain`；
- execute/replan=5；
- flow/denoising steps=10；
- `sha256-pcg64-standard-normal-f32-v1`；
- 旧 69-W4 plan 或旧 A8/ATM/OHB；
- `runs/pi05_gdsq_port/official_pretrain_paired50`；
- `runs/pi05_gdsq_final/official_pretrain_paired50`；
- 旧四任务结果 OpenCabinet 32%、OpenStandMixerHead 44%、PickPlaceDrawerToCounter 26%、
  CoffeeSetupMug 18%。

旧结果可用于解释为什么实现被推翻，但禁止与 aligned rows 拼接、禁止填入正式主表、禁止据此
声称 W4A8 或 ATM/OHB 优于 FP16。

## 11. Definition of Done

1. 28 个正式 replica runtime attestation 与 immutable manifest 通过；
2. FP16=0、full-W4=180、GDSQ 两行=80 wrappers；
3. GDSQ 两行 plan/A8 hash 完全相同，ATM enablement 唯一差异；
4. 所有 rows 均为 target/execute-16/d4/Torch paired noise；
5. 四配置各 2,500 episodes，key matrix 无缺失、无重复、无 crash-as-failure；
6. 统计按 GR00T final 的 task-cluster bootstrap、paired permutation、Holm 和 McNemar 输出；
7. accuracy 与 theoretical storage、eager runtime efficiency、packed-kernel efficiency 分开报告；
8. 历史 rows 永不进入 aligned aggregate。
