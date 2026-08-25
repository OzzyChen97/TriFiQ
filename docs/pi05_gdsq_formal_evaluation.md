# GDSQ-VLA → π0.5 旧工程波次记录（已作废）

> **INVALIDATED 2026-08-19:** 本文对应旧 state16/16:1 工程波次，使用的功能指标没有显式
> 区分 execute-5 与 chunk-50，也没有经过新的 ratio/CKA-only faithful selector 与 development
> freeze。文中 `runs/pi05_gdsq_port/official_pretrain_paired50`、旧 plan/A8/ATM hash 和已有 rows
> 只能作历史诊断，禁止引用为 final 或与新矩阵拼接。新正式路径为
> `runs/pi05_gdsq_gr00t_aligned/official_target_paired50`，固定使用
> `target / execute-16 / d4 / sha256→CPU-Torch paired noise`，以
> [quantvla_v2_pi05_porting_final.md](quantvla_v2_pi05_porting_final.md) 及其 immutable manifest 为准。
> 本文其余章节全部是旧协议记录，不得当作当前配置说明。
>
> 状态：**historical / invalidated**  
> 日期：2026-08-19  
> 设计协议来源：[quantvla_v2_pi05_porting_final.md](quantvla_v2_pi05_porting_final.md)

## 正式配置

| ID | 精度/量化 | ATM/OHB | 运行时断言 |
|---|---|---|---|
| `fp16` | 严格 FP16 reference | 关闭 | 0 quant wrappers；458 个 Linear 均为 FP16 |
| `quantvla_w4a8_atmohb` | 原版 QuantVLA，全 180 层 W4A8 | static expert | 180 wrappers；18 个 attention layers |
| `gdsq_vla_atmohb` | GDSQ-VLA，CKA:CS=16:1 | plan-specific static expert | 69 wrappers；18 个 attention layers |
| `gdsq_vla` | GDSQ-VLA（Ours），CKA:CS=16:1 | 关闭 | 69 wrappers |

所有配置使用相同 π0.5 checkpoint、三路 224×224 图像、10 个 flow steps、50-action
chunk、每 5 步 replan，以及相同的 paired deterministic 初始 diffusion noise。

## RoboCasa365 Table 1 口径

- `pretrain` split；
- `atomic_seen` 18 tasks、`composite_seen` 16 tasks、`composite_unseen` 16 tasks；
- 每任务显式 seeds `0..49`，共 2,500 episodes/config、10,000 episodes；
- 每 trial 构造 fresh environment，render 开启，使用官方 task horizon；
- 16 维 state 顺序严格为：EE position 3、EE quaternion 4、base position 3、base
  quaternion 4、gripper qpos 2；
- 输出使用前 12 个 action dimensions；
- paired-noise 协议为 `sha256-pcg64-standard-normal-f32-v1`；
- 结果键为 `(config, task, seed)`，crash 不写成失败样本，重启按 JSONL 原子恢复。

旧 12 维 synthetic buffer 会被 padding 接受，但遗漏了官方 state 的后 4 维，不能作为正式
probe/calibration 数据。正式产物全部基于 state16 buffer 重算，旧 state12 产物不得加载。

## 16:1 mixed plan 的实际量化幅度

正式计划按 **static-byte budget** 而不是层数预算迁移。GR00T N1.5 的 100/116 mask
不能机械映射到 π0.5，因为两种架构的候选层 shape 与参数量分布不同。π0.5 的 69 个 W4
层主要是大矩阵，因此 69/180 不是等参数比例。它们覆盖：

- W4 参数：1,757,413,376；
- 候选参数总量：2,208,301,056；
- W4 参数覆盖率：79.6%；
- language candidate 参数覆盖率：86.1%；
- action-expert candidate 参数覆盖率：22.2%。

在 180-layer candidate scope 内：

| 配置 | paper-style static bytes | GiB | 相对 FP16 压缩 |
|---|---:|---:|---:|
| FP16 | 4,416,602,112 | 4.113 | 1.00× |
| 全层 W4A8 | 1,490,466,816 | 1.388 | 2.96× |
| GDSQ 69 W4 + 111 FP16 | 2,040,756,224 | 1.901 | 2.16× |

GDSQ 计划满足预注册的 uniform-W6 static-byte budget。当前正式准确率使用 eager fake-quant
correctness path；该实现为每个 wrapper 保留 base/transformed/quantized cache，因此 live CUDA
memory 不等于紧密打包后的部署内存，不能把 paper-style 1.901 GiB 误报为当前 eager 显存。

候选层共 2,208,301,056 个参数，只占 checkpoint 全部 3,616,757,520 个参数的 61.1%。因此，
正式 plan 的 W4 参数占全模型 48.6%；若把候选层的理论紧密打包和其余权重的 FP16 存储合并，
全模型权重从 6.737 GiB 降至 4.524 GiB，即理论 1.49×、等效约 10.74 bit。这个数字是更保守、
更接近端到端的 paper-style 权重口径，但仍不是当前 fake-quant runtime 的实测显存。

为审计“更高量化强度”而额外生成了一个独立的 37.5%-of-FP16 budget 计划：121 W4 + 59
FP16，覆盖 93.4% 候选参数和 37/54 个 action-expert MLP。其 SHA256 为
`93bbac31aa2bbb4f6fc334d0e13deaa2acb39dc51ac1a0a971307a535c65c7d8`。该产物未参与正式
矩阵、没有使用 held-out SR 选择，也不能替代预注册的 uniform-W6 final；后续若报告，必须
独立标定 A8/ATM-OHB 并使用新的 immutable run manifest。

## 冻结产物

| 产物 | SHA256 |
|---|---|
| π0.5 checkpoint | `4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c` |
| config | `3673272c04c1d5eb6bc187d087104b4912a3a33624eb6e02c2250904d593b836` |
| norm stats | `4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee` |
| 180-layer inventory | `1b42eed276978560fd9c4c5120ce2847bc7117ed0233c464034809bbdbdd46ac` |
| block64 pack manifest | `53df14c6db4875a5515ff11bf9c40350b4d08c0f11e2e5b9cfde1a8e13bd6138` |
| state16 buffer | `b6a67b7d3e5731890ef4c178542e458d79810fdaba930e9cf747c659803cb9d9` |
| state16 sensitivity merge | `6d7b8c14862d7b3d06f79dc9d5273408920fcd5608a2817b0bc9937113e1d240` |
| full-W4A8 plan | `680a118a0817750f0ad39d8f8e4e52c3ed35e974aeb5ca204d366245ccfe1f9b` |
| GDSQ 16:1 plan | `8e09fe339a1ce0aeddbb1af14891dee0d60bf83fdbfff7ac0b2dc001d3c87b9a` |
| full-W4A8 A8 scale | `c62d480bd1260f4c72747bfca2dd1b2f4a109dbf18846706125ced36b7fe02f2` |
| GDSQ A8 scale | `7cae74e47ff38a1ac0c3014de092ab11be5913280e5278edb71cb6d8330da3c3` |
| full-W4A8 ATM/OHB | `19172cc98cb225d9abc472a44b10d40894195ade7d022c5f708c3f7870d688c7` |
| GDSQ ATM/OHB | `df27e968dd47551763b588c03f29e678338ef15b44c5f2aac9d4cb80457e422d` |
| formal run manifest | `0bd05bf31434cc5f0df505bbe99e9e7782364c9ae84dd9d4023b39173ec4b703` |

## Correctness smoke

- 四配置输出均为 `[50,12]`、finite；
- 相同 `(task, seed, replan)` noise 的重复 websocket 推理 bitwise identical；
- FP16/full-W4A8/GDSQ+ATM/GDSQ wrapper 数为 `0/180/69/69`；
- full-W4A8 与 GDSQ ATM/OHB 均匹配 18 层，GDSQ 两配置使用完全相同的 plan/A8；
- OpenCabinet seed 0 fresh-env smoke：FP16 成功于 724 步；三个量化配置跑满 1,050 步未成功。
  单 seed 仅证明闭环无 crash，不能作为成功率结论。

## 正式运行与恢复

正式目录：

```text
runs/pi05_gdsq_port/official_pretrain_paired50/
```

主命令：

```bash
scripts/run_pi05_formal_matrix.sh status
scripts/run_pi05_formal_matrix.sh start       # 幂等恢复
```

GPU1–6 上运行六台 server。初始每台 4 个 fresh-env worker；schedule v2 将 seed 拆为互斥的
`0–24` 和 `25–49`，提升到 48 个并发 episode。schedule v3 进一步只重分 task shard，并用
config/task-set 全局只读 resume index 严格拒绝跨 JSONL 重复，提升到 72 个 worker：FP16 16、
full-W4A8 20、GDSQ+ATM/OHB 12、GDSQ 24。两次 amendment 均不改变模型、task/seed、noise 或
评测协议，切换前的完整 rows 全部保留，未提交 trial 重新 fresh 构造。GPU0/7 上的外部进程
未被终止。每个 worker 自动按 `atomic_seen → composite_seen → composite_unseen` 切换。

调度 amendment：

```text
runs/pi05_gdsq_port/official_pretrain_paired50/manifest.schedule_v2.json
runs/pi05_gdsq_port/official_pretrain_paired50/manifest.schedule_v3.json
```

schedule-v3 manifest SHA256：
`62a79b13fefb66b39c97ab5b61019407b4a6ac50ba6b2638c12362fc5d154114`。

正式矩阵完整前，聚合器默认拒绝输出正式结论；进度审计可显式使用：

```bash
PYTHONPATH=code/pi05/openpi/packages/openpi-client/src \
  /home1/gyy/probe/miniforge3/envs/robocasa365/bin/python \
  scripts/tools/aggregate_pi05_robocasa365.py \
  --run-dir runs/pi05_gdsq_port/official_pretrain_paired50 \
  --out-dir runs/pi05_gdsq_port/official_pretrain_paired50/aggregate_partial \
  --allow-incomplete
```

最终完整聚合将输出 per-task SR、三 task-set macro SR、50-task macro SR、成功/失败步数、
10,000 次 task-cluster bootstrap、task-level paired permutation、Holm 校正、McNemar 辅助统计、
policy/env 效率、live eager CUDA memory 与 paper-style component memory。
