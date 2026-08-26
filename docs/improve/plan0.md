## 总目标
看看这个项目https://OzzyChen97@github.com/OzzyChen97/GDSQ-VLA.git。我现在想做两个事情：第一，增加一个loss用来衡量长程任务的累积差异，解决长程任务量化等问题；第二，去除selector，将其soft化，loss或其他办法判断是否需要OHB并将其内化，或者选择其他办法弥补量化差异。你可以在这两个目标上发挥想像，看看有没有其他优化和改进的方法，压缩率对标Quant VLA。注意你只能和原始FP16对比来进行量化，不能有额外数据训练

这两个目标不应拆成“一个长程 loss + 一个更复杂的 selector”。更合理的下一版是统一成：

> **原始 FP16 锚定的双时间尺度累积差异 + selector-free 连续校正 + QuantVLA 等字节预算搜索**

也就是：

1. 新指标不只比较一次推理内部的去噪轨迹，还比较动作在 **chunk 内、replan 间和任务前缀上如何累积**；
2. ATM/OHB 不再是 baseline / ATM / OHB 三选一，而是从恒等映射出发的连续系数，最后直接折叠进权重；
3. 搜索预算直接锁死为 QuantVLA 的字节数，而不是先得到一个较保守的 W4/FP16 mask，再解释为什么压缩率低。

我建议暂时把这套方法称为：

* `D_PAC`：Prefix-Accumulated Control Divergence；
* `SoftFold`：selector-free folded compensation。

---

## 一、当前实现真正缺的是什么

### 1. 当前 `D_func` 仍是“单次推理内”的指标

仓库里的 `D_func` 输入是 `(T+1, B, H, D)`，第一维是去噪/流匹配轨迹。它测量最终 action chunk、平移旋转误差、gripper sign mismatch 和 observation-level CVaR。因此它能发现“一次 action inference 的末端输出坏了”，但不能直接发现：

* 连续多个 replan 后，小偏差是否同方向累加；
* chunk 边界是否越来越不连续；
* gripper 开合事件是否逐渐提前或延迟；
* 一个很小但持续存在的 DC bias 是否比同等 MSE 的正负抖动更危险。

换言之，它主要覆盖 **solver time**，还没有覆盖 **control time**。

π0.5 的 adapter 又只保留 50 个预测动作中的前 16 个执行动作，并明确没有 chunk-50 辅助项。因此当前指标也没有利用 π0.5 较长预测 horizon 中潜在的规划漂移信息。

### 2. 当前 v8 selector 实际已经不是“动态 selector”

它对一个模型的所有任务给出相同决策，候选校正不通过门槛就退回 baseline。它不读取运行时成功反馈，也不真正根据当前 observation 改变决策。因此它本质上是一个**离线模型级硬开关**，放在 runtime 中的信息增益接近零。

而且这些门槛回答的是“校正幅度看起来够不够明显”，并没有回答更关键的问题：

> 应用 OHB 后，量化模型相对于原始 FP16 的动作累积差异到底变好了还是变坏了？

### 3. 需要先修正 FP16 教师语义

这是我认为最先要改的地方。当前局部 CKA/CS intervention 的 reference 不是原始 FP16，而是“所有目标层 `weight_bits=0`、但仍经过 rotation / wrapper / A8 路径”的参考管线；原始 FP16 主要用于全局 `D_solver` 配对。这个设计适合消除 wrapper confound，但不满足你现在提出的“所有量化判断只能和原始 FP16 比较”的严格约束。

下一版应规定：

[
\text{Teacher} \equiv M_{\mathrm{FP16}}
]

所有配置、所有层干预、所有校正系数都只优化：

[
D(M_q(x,\epsilon),M_{\mathrm{FP16}}(x,\epsilon))
]

其中 observation、noise、flow steps 完全配对。wrapped reference 最多保留为 debug diagnostic，不能再进入最终优化目标。

---

# 二、目标一：增加真正的长程累积差异 `D_PAC`

## 1. 明确三条时间轴

建议把误差表示成三维结构，而不是简单把 action MSE 拉长：

* (k)：去噪/flow step；
* (h)：一个 action chunk 内的动作位置；
* (r)：环境中的第几个 replan。

当前 `D_func` 主要覆盖 (k) 和单个 (h) chunk。新指标需要增加 (r) 轴。

设同一个任务序列中，FP16 和量化模型在第 (r) 次 replan 输出：

[
A^F_r,A^Q_r\in \mathbb{R}^{H\times D}
]

只对真正执行的前 (m) 个动作定义主损失，其中当前协议通常是 (m=16)。用原始 FP16 动作的 MAD 或 percentile 做逐维尺度归一化：

[
e_{r,h,d}=
\frac{A^Q_{r,h,d}-A^F_{r,h,d}}
{\operatorname{MAD}(A^F_{\cdot,\cdot,d})+\epsilon}
]

这样平移、旋转、gripper、base/torso 不会因为单位不同而互相淹没。

---

## 2. 核心项：全前缀累积漂移

最重要的不是最终时刻误差，而是**每一个前缀的累计误差**。

将所有已执行动作按控制时间展开为 (e_1,\ldots,e_N)，定义：

[
c_n=\sum_{i=1}^{n}e_i
]

[
D_{\mathrm{prefix}}
===================

\frac{1}{Z}
\sum_{n=1}^{N}
\left(\frac{n}{N}\right)^2
\lVert c_n\rVert_W^2
]

这个指标有一个非常需要的性质：

* 误差为 (+\delta,-\delta,+\delta,-\delta) 时，长期漂移较小；
* 误差始终为 (+\delta) 时，即使逐步 MSE 完全相同，前缀损失会快速增大。

这正是长程控制里“持续小偏差比零均值抖动更危险”的数学表达。

对于前六维的末端位姿，最好不要直接做欧式求和，而是用动作 twist 做 (SE(3)) 组合：

[
T_n^F=\prod_{i=1}^{n}\operatorname{Exp}(\xi_i^F),
\qquad
T_n^Q=\prod_{i=1}^{n}\operatorname{Exp}(\xi_i^Q)
]

[
D_{\mathrm{pose}}
=================

\frac{1}{Z}
\sum_{n=1}^{N}w_n
\left|
\operatorname{Log}\left((T_n^F)^{-1}T_n^Q\right)
\right|_W^2
]

这不需要机器人 simulator、真实动力学或额外数据，只使用 FP16 和量化模型输出的 action delta。

对于 π0.5 的 base、torso 或控制模式维度，继续用归一化 prefix sum，不要硬塞进虚构的机械臂 Jacobian。

---

## 3. replan 边界一致性

量化模型可能每个 chunk 单独看都不差，但 chunk 之间出现控制跳变。可以比较量化与 FP16 的“边界跳变量之差”：

[
D_{\mathrm{stitch}}
===================

\frac{1}{R-1}
\sum_r
\left|
\left(A^Q_{r+1,0}-A^Q_{r,m-1}\right)
------------------------------------

\left(A^F_{r+1,0}-A^F_{r,m-1}\right)
\right|^2
]

它不要求量化动作本身绝对平滑，只要求量化不要改变 FP16 原本的边界动态。

π0.5 还可以增加一个低权重的 forecast-overlap 项。由于它预测 50 步、只执行 16 步，前一次预测的 `16:50` 与下一次预测的 `0:34` 在时间上近似重叠：

[
D_{\mathrm{overlap}}
====================

\left|
\big(A^Q_r[16:50]-A^Q_{r+1}[0:34]\big)
--------------------------------------

\big(A^F_r[16:50]-A^F_{r+1}[0:34]\big)
\right|^2
]

它不是环境真值，所以只能作为辅助项，权重应低于已执行 16 步的损失。

---

## 4. gripper 事件时间，而不是去噪过程 sign

当前 `D_func` 的 gripper sign mismatch 主要描述去噪轨迹中 gripper 输出如何变化。长程任务更关心“真正执行时什么时候抓、什么时候放”。

可以把 gripper action 转为软状态：

[
p_n=\sigma(\kappa a^{\mathrm{grip}}_n)
]

同时匹配状态和状态变化：

[
D_{\mathrm{grip-time}}
======================

\frac1N\sum_n
\left[
(p_n^Q-p_n^F)^2
+
\lambda_\Delta
\left(
\Delta p_n^Q-\Delta p_n^F
\right)^2
\right]
]

这样会惩罚抓取事件提前、延迟或漏掉，但不需要人为设一个硬开合阈值。

---

## 5. 最终形式

建议 sequence-level 指标写成：

[
d_s =
\frac1{R_s}\sum_r D_{\mathrm{func}}(r)
+
\lambda_pD_{\mathrm{prefix/pose}}
+
\lambda_sD_{\mathrm{stitch}}
+
\lambda_gD_{\mathrm{grip-time}}
+
\lambda_oD_{\mathrm{overlap}}
]

[
D_{\mathrm{PAC}}
================

\operatorname{Mean}*s(d_s)
+
\lambda*{\mathrm{tail}}
\operatorname{CVaR}_{0.9}{d_s}
]

其中：

* GR00T：(\lambda_o=0)；
* π0.5：`overlap` 只作低权重辅助；
* 各项先用 FP16 统计量归一化，初始全部设为 1；
* sequence 模式下建议关闭当前 `D_func` 的 observation-CVaR，统一在 sequence 层计算 CVaR，避免 tail risk 重复计权；
* 第一版不要调出十几个手工权重，否则它会变成另一种 selector。

仓库里的 π0.5 scorer 已经支持读取 `env_steps` 和 `replan_indices`，并按 task、replan 和 replan bin 汇总，因此加入 sequence grouping 的代码改动不会很大。

### 必须保留的科学边界

固定 observation 下的 FP16/quant 配对仍然是 **teacher-forced open-loop surrogate**。因为量化动作没有真正改变下一帧图像，所以它不能等价于真实 closed-loop covariate shift。

因此：

* 有连续 replan observation 时，可以称为 “replan-sequence drift”；
* 只有单个 observation 时，只能称为 “action-prefix accumulated drift”；
* 不要直接写成“closed-loop loss”，否则审稿人很容易抓住这一点。

最终 closed-loop success 只能在所有校准和选择冻结后用于评测，不能反过来调损失权重。

---

# 三、目标二：去掉 selector，改成 SoftFold

## 1. 不建议训练一个 soft selector 网络

一个 observation-conditioned MLP gate 会带来三个问题：

* 需要额外数据或训练；
* 引入 runtime 分支；
* 很可能只学会任务类别或动作幅度，而不是真正的量化误差。

这与“training-free、无额外数据、无额外运行时结构”的方向相反。

真正需要 soft 化的不是“选择器网络”，而是**校正幅度本身**。

---

## 2. 把 ATM/OHB 写成从 identity 出发的连续路径

保留当前校准得到的原始因子 (\alpha_{\rm raw})、(\beta_{\rm raw})，定义：

[
\alpha_{\mathrm{eff}}
=====================

\exp\left(g_A\log\alpha_{\mathrm{raw}}\right)
]

[
\beta_{\mathrm{eff}}
====================

\exp\left(g_B\log\beta_{\mathrm{raw}}\right)
]

其中：

[
g_A,g_B\in[0,1]
]

于是：

* (g=0)：严格等于 identity，即不应用校正；
* (g=1)：完整 ATM/OHB；
* 中间值：部分校正；
* 不再需要 baseline / ATM / OHB / ATM+OHB 四个离散 variant。

用 log-space 插值比 `1 + g(β−1)` 更合适，因为它保持乘法校正的正值性，并且放大与缩小更加对称。

---

## 3. 让 `D_PAC` 决定 OHB 是否需要

第一版建议采用非常保守的无梯度流程：

1. 通过 sample hash 将同一个冻结 calibration buffer 分为 fit/validation 两半；
2. 在 fit 半上估计原始 (\alpha,\beta)；
3. 在 validation 半上枚举
   [
   g_A,g_B\in{0,0.125,\ldots,1}
   ]
   共 (9\times9) 个组合；
4. 计算每个组合相对于原始 FP16 的 `D_PAC`；
5. 使用 **one-standard-error rule**：在与最优结果统计上不可区分的组合中，选择 (g_A+g_B) 最小的一个。

也可以写成：

[
\min_{g_A,g_B}
D_{\mathrm{PAC}}(g_A,g_B)
+
\lambda_1(g_A+g_B)
+
\lambda_\times g_Ag_B
]

其中：

* (\lambda_1) 把不确定校正收缩回 identity；
* (\lambda_\times) 抑制 ATM 和 OHB 对同一漂移进行双重校正；
* 但不再像当前 selector 那样硬性禁止二者同时存在。

如果 GR00T 上 OHB 无稳定收益，最终自然得到 (g_B\approx0)；如果 π0.5 确实需要 OHB，则得到 (g_B>0)。这比“校正幅度大于某个经验门槛就启用”更直接。

---

## 4. OHB 本身也可以从 RMS ratio 升级

当前 OHB 主要匹配 RMS，只保证能量相似。两个输出即使 RMS 一样，方向也可能完全不一致。

可以将每个 head 的原始因子改为 identity-regularized least squares：

[
\beta^*_{l,h}
=============

\operatorname{clip}
\left(
\frac{
\langle O^Q_{l,h},O^F_{l,h}\rangle+\lambda
}{
\lVert O^Q_{l,h}\rVert^2+\lambda
},
e^{-c},e^c
\right)
]

它是下面问题的闭式解：

[
\min_\beta
\lVert\beta O^Q-O^F\rVert^2
+
\lambda(\beta-1)^2
]

其中 (\lambda) 明确把系数拉向 1。ATM 对 centered attention logits 也可以使用同样的 ridge scale，而不是仅使用 std ratio。

不过实验上不要一次把所有东西都换掉。推荐分两步：

* `SoftFold-v1`：保留当前 ratio-based (\alpha,\beta)，只引入连续 gate；
* `SoftFold-v2`：再将 RMS/std ratio 替换为 identity-ridge factor。

这样消融结果才可解释。

---

## 5. 最后全部折叠，不保留 runtime selector

π0.5 路径已经实现：

* ATM 折叠进 `q_proj`；
* scalar OHB 折叠进 `o_proj`；
* per-head OHB 折叠进 `o_proj` 输入列。

所以这里只需生成 (\alpha_{\mathrm{eff}},\beta_{\mathrm{eff}})，并移除 runtime selector uniformity check。

GR00T 当前已有 ATM 的 `to_q` weight folding，也已有 per-head/per-step OHB 表，但 OHB 仅支持 runtime output scaling。应补一个：

```python
_fold_ohb_perhead_into_o_projection(module, beta)
```

本质上对 `to_out[0].weight` 的输入列按 head 展开后缩放：

[
W_o[:,h\cdot d_h:(h+1)d_h]
\leftarrow
\beta_h
W_o[:,h\cdot d_h:(h+1)d_h]
]

scalar OHB 则可缩放输出行和 bias。

如果保留 denoising-step-specific correction，它不能完全折叠为一个静态矩阵。此时有两个选择：

* 保留一个按 flow step 索引的小型确定性 scale table；它不是 selector，也不依赖任务；
* 或将各 step 系数按 `D_PAC` 权重做 log-space pooling，得到一个静态因子后折叠。

第一版建议用静态折叠；per-step correction 单独作为增强消融。

QuantVLA 本身也强调 ATM/OHB 可在校准后折叠，不增加新算子或 GEMM，因此 SoftFold 不应牺牲这一部署属性。([arXiv][1])

---

# 四、压缩率如何真正对标 QuantVLA

仓库当前内部统计是：

| 模型    |                  当前 GDSQ-VLA |               仓库 QuantVLA 对照 |                   需要减少 |
| ----- | ---------------------------: | ---------------------------: | ---------------------: |
| GR00T | 1.001 GiB，100/116 层 W4，1.99× | 0.898 GiB，116/116 层 W4，2.22× | 0.103 GiB，约当前大小的 10.3% |
| π0.5  |  1.868 GiB，80/180 层 W4，2.20× | 1.388 GiB，180/180 层 W4，2.96× | 0.480 GiB，约当前大小的 25.7% |

这些是当前仓库的理论 candidate-linear storage 口径，不是实际 eager GPU memory。

因此需要直说：

> 删除 selector 几乎不会改善压缩率。真正占空间的是 GR00T 的 16 个 FP16 候选层，以及 π0.5 的 100 个 FP16 候选层。

QuantVLA 原论文的选择布局也是量化 LLM 和 DiT MLP，同时保持 attention projections 为浮点；其表中对应 180 个 π0.5 MLP 层和 116 个 GR00T MLP 层。([arXiv][1])

## 推荐改成 exact-byte optimization

不要继续用“固定 16:1 比例”，而是直接设置：

[
\sum_i C_i(b_i)+C_{\mathrm{metadata}}
\le B_{\mathrm{QuantVLA}}
]

其中 `B_QuantVLA` 使用仓库内同 checkpoint、同 candidate scope、同 rotation/scale accounting 下的字节数。

实际搜索顺序应当是：

1. **从 QuantVLA 全 W4 layout 出发**，而不是从当前保守 mixed mask 出发；
2. 用 `D_PAC` 搜索 clipping percentile、smooth factor、rotation 和 correction；
3. 所有 ATM/OHB/SoftFold correction 直接折叠，额外权重存储近似为零；
4. 只有全 W4 仍明显失败时，才给敏感层分配额外残差预算；
5. 每次增加高精度残差，都必须由精确 byte counter 扣账。

### GR00T

差距只有约 10%，更可能通过下面组合完成：

* 116/116 W4；
* `D_PAC` 指导的 per-layer clipping/smoothing；
* SoftFold ATM/OHB；
* attention-input 的 per-step A8 scale；
* 全配置 FP16-relative adjudication。

### π0.5

这里困难得多。80/180 到 180/180 不是轻微修补，而是要让 100 个当前受保护层进入 W4。若全 W4 + SoftFold 仍失败，优先考虑：

* W4 主矩阵 + 极小的 sparse/outlier residual；
* W4 主矩阵 + rank-1/2 residual，仅用于少数最敏感层；
* 将 residual 的收益统一用 `ΔD_PAC / extra_byte` 排序。

但 low-rank residual 会引入额外 GEMM，除非有融合 kernel。它可以作为恢复精度的实验，不应在没有实际 kernel 和 latency 证据时宣称“无运行时开销”。

---

# 五、另外三个值得加入的优化

## 1. 用误差方向一致性修正 additive mask objective

当前计划搜索使用：

[
\sum_i w_iS_i
]

而完整 `D_func` 只用于 Top-K adjudication。这样会忽略一种长程危险：

> 两个层各自误差不大，但它们在 action space 中方向高度一致，于是完整模型出现持续偏置。

可以缓存每个单层干预相对于 FP16 的低维 action-error sketch (\bar e_i)，增加：

[
P_{ij}
======

\max(0,\cos(\bar e_i,\bar e_j))
\sqrt{d_i d_j}
]

[
D_{\mathrm{proxy}}(M)
=====================

\sum_i m_i d_i
+
\eta\sum_{(i,j)\in\mathcal E}
m_im_jP_{ij}
]

只对 top-sensitive 层构建 pair graph，避免 (O(L^2)) 全量计算。最终仍由完整配置的 `D_PAC` 裁决。当前 selector 已明确承认 additive proxy 不是完整配置行为，因此这里是自然延伸。

## 2. per-step A8 scale，而不是 per-step selector

GR00T 已经能读取 denoising-step-aware ATM/OHB 表，下一步更值得做的是给 AdaLN 后的 attention 输入使用 per-step activation scale：

[
s_{l,k,d}
=========

\frac{P_{99.9}(|x^F_{l,k,d}|)}{127}
]

它仍只来自原始 FP16，同一模型只有一个静态表，不依赖任务和 runtime observation。

但这个方向不能当作核心新颖性。Ω-QVLA 已经系统使用 composite rotation 和 per-step DiT activation scaling，并报告该项对 Long suite 尤其重要。([arXiv][2])

## 3. 同 observation、多配对 noise 的动作分布匹配

不增加 observation，只对同一个冻结输入使用 2–4 个确定性 noise seed：

[
D_{\mathrm{noise}}
==================

\lVert\mu_Q-\mu_F\rVert^2
+
\lambda_\Sigma
\lVert\Sigma_Q-\Sigma_F\rVert_F^2
]

它能避免某个量化配置只在单个 paired noise 下偶然接近 FP16。计算成本较高，因此只用于最终 Top-K adjudication，不进入每层扫描。

---

# 六、代码改造顺序

建议按以下顺序落地，避免同时改太多组件后无法归因。

### P0：先修教师语义

修改 `code/gr00t/quantization/kernel_scores.py` 和 sensitivity probe：

* 原始 FP16 是唯一 teacher；
* wrapped zero-bit reference 只保留 debug 字段；
* 所有 layer/config score 都记录 FP16 checkpoint hash、observation hash 和 paired-noise scheme。

### P1：加入 sequence-level `D_PAC`

在 `scripts/tools/gr00t_func_metrics.py` 中增加：

```python
d_pac_sequence(
    fp16_chunks,
    quant_chunks,
    replan_indices,
    executed_actions=16,
    ...
)
```

`d_func()` 保持兼容，避免破坏旧结果。

`pi05_func_metrics.py` 增加：

* 前 16 步主损失；
* `16:50` forecast 辅助；
* optional overlap loss。

### P2：替换 selector 拟合

新建：

```text
scripts/tools/fit_softfold_compensation.py
```

输出 artifact：

```json
{
  "teacher_checkpoint_sha256": "...",
  "quant_plan_sha256": "...",
  "buffer_sha256": "...",
  "metric": "d_pac_v1",
  "gate": {
    "atm": 0.125,
    "ohb": 0.625
  },
  "layers": {
    "...": {
      "alpha_eff": [],
      "beta_eff": []
    }
  },
  "selection": {
    "rule": "one_standard_error",
    "uses_task_labels": false,
    "uses_rollout_success": false
  }
}
```

不再输出 task → variant 映射。

### P3：完成 fold

* π0.5：复用现有 q-proj/o-proj folding；
* GR00T：补 per-head OHB → `to_out[0]` column folding；
* 删除 `atm_enabled_for_current_request()` / `ohb_enabled_for_current_request()` 依赖；
* selector 文件只保留一段过渡兼容期，正式配置不再加载。

### P4：exact QuantVLA byte cap

修改 `gr00t_select_plan.py` 及 π0.5 对应搜索：

* byte cap 固定为 QuantVLA；
* CKA/CS 仅作 cheap prescreen；
* 最终 mask 用完整配置 `D_PAC`；
* 搜索从全 W4 开始，以“恢复精度/额外字节”为核心，而不是从大量 FP16 层开始逐步压缩。

---

# 七、必须做的关键消融

快速实验(15tasks * 20seeds)至少需要分成两组，避免 mask、loss、correction 和压缩率互相混淆。

第一组固定当前 mask：

| 配置                    | 目的                 |
| --------------------- | ------------------ |
| 当前 GDSQ，无校正           | 基线                 |
| 当前硬 selector          | 复现                 |
| SoftFold，仍用旧 `D_func` | 隔离 soft correction |
| SoftFold + `D_PAC`    | 检查新长程目标            |

第二组固定 QuantVLA exact bytes：

| 配置                       | 目的     |
| ------------------------ | ------ |
| QuantVLA W4A8            | 同预算基线  |
| GDSQ score + exact bytes | 检查选层本身 |
| `D_PAC` + exact bytes    | 检查累积指标 |
| `D_PAC` + SoftFold       | 主方法    |
| + per-step A8            | 工程增强   |

校准阶段只能看 FP16-relative divergence。Atomic、Composite、Long success 全部在配置冻结后评测。

最重要的单元测试包括：

* FP16 与自身：所有项为 0；
* 恒定 (+\delta) 与正负交替误差具有相同逐步 MSE，但前者 `D_prefix` 显著更高；
* 只在 replan 边界制造跳变时，`D_stitch` 点火；
* gripper 事件延迟时，`D_grip-time` 点火；
* runtime scaling 与 folded weight 输出等价；
* 删除 task metadata 后，输出完全不变；
* 两个 calibration split 得到的 gate 和 mask 稳定。

---

## 我建议的第一版收敛方案

第一轮只做三件事：

1. **原始 FP16-only reference 重构**；
2. **`D_PAC`：prefix pose drift + replan stitch + gripper timing + sequence CVaR**；
3. **全局 `g_A,g_B` SoftFold，9×9 无梯度搜索并折叠权重**。

暂时不加入 low-rank residual、动态 gate 或复杂 per-head gate。先验证这三项能否在当前 mask 上稳定降低长程 FP16 divergence；随后再把预算压到 QuantVLA。这个路径的科学问题最明确，代码改动也与现有实现高度对齐，不会再把架构堆成一座量化圣诞树。

[1]: https://arxiv.org/html/2602.20309v1 "QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models"
[2]: https://arxiv.org/html/2605.28803v1 "https://arxiv.org/html/2605.28803v1"
[3]: https://arxiv.org/html/2604.11572v1 "https://arxiv.org/html/2604.11572v1"
[4]: https://arxiv.org/html/2602.03782v1 "https://arxiv.org/html/2602.03782v1"
[5]: https://arxiv.org/html/2603.07904v1 "https://arxiv.org/html/2603.07904v1"
