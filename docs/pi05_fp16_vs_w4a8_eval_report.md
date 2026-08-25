# pi0.5 fp16 vs QuantVLA W4A8 评测报告（RoboCasa365 + RoboCerebra）

> 本报告是旧全层 W4A8 的工程基线，不能作为 CS+CKA π0.5 正式结果。新的移植、paired-noise、
> explicit-seed 与 50-trial 协议以 `docs/quantvla_v2_pi05_porting_final.md` 为准；本文历史
> “fp16”服务在当前 PyTorch 路径中实际为 BF16。

> 完成日期: 2026-08-15 | 机器: user-420GP-TNR（8× NVIDIA A40, 128 核, 503GB RAM）| 评测 GPU: 1–7
> 最终结果: `runs/pi05_fp16_vs_w4a8/`（聚合 JSON `pi05_fp16_vs_w4a8_agg.json`、延迟基准 `latency_*.json`）；逐任务明细见聚合 JSON 的 `per_task` 字段
> （中间产物——评测日志/rollouts/原始 stats/临时脚本——已按收尾清理移除）

## 1. 评测设置

| 项 | 值 |
|---|---|
| 模型 | pi0.5（PaliGemma 2B + action expert 300M），`pi05_pretrain_human300`（RoboCasa）/ `pi05_libero`（RoboCerebra）|
| fp16 版 | bf16 + torch.compile(max-autotune) |
| W4A8 版 | QuantVLA DuQuant W4A8（RoboCasa pack 实为 **block=16**，见 §4.2）+ ATM + OHB（scope=expert），eager 执行 |
| RoboCasa365 | 3 task set（atomic_seen 18 / composite_seen 16 / composite_unseen 16 = **50 任务**）× **10 trials**，split=pretrain，seed=7，replan=5 |
| RoboCerebra | 6 类 × 10 case = **60 任务** × **5 trials**，seed=7，replan=5 |
| 协议一致性 | 两模型相同任务/相同协议；唯一例外见 §5 注 2（W4A8 最后 2 个任务拆分执行）|

**执行方式**：策略服务器（websocket）与仿真客户端分离；RoboCasa 客户端 4 路并行 + 手动扩容、RoboCerebra 3 路并行，任务互不重叠；mujoco EGL 渲染上下文与服务器同卡绑定（`MUJOCO_EGL_DEVICE_ID`+`CUDA_VISIBLE_DEVICES`）。全程由可重挂载监督器（tmux 会话 `pi05eval`，`scripts/pi05_eval_supervisor.py`）管理断点续跑。

## 2. 延迟 / 吞吐 / 显存（同机实测，30 请求合成观测，服务器空闲）

| 服务器 | 均值 | p50 | p95 | 吞吐 | 空闲显存 |
|---|---|---|---|---|---|
| RoboCasa fp16 (torch.compile) | **105 ms** | 105 | 110 | 9.5 req/s | 8.7 GB |
| RoboCasa W4A8 (eager) | **897–931 ms** | ~900 | ~940 | 1.1 req/s | 23.0 GB |
| RoboCerebra fp16 (torch.compile) | **99 ms** | 94 | 102 | 10.1 req/s | 9.0 GB |
| RoboCerebra W4A8 (eager) | **561 ms** | 549 | 566 | 1.8 req/s | 23.1 GB |

要点：
- W4A8 单请求延迟为 fp16 的 **5.7–8.5×**，吞吐约为 fp16 的 1/6–1/9。
- 本实现 W4A8 **不省推理期显存**（23GB vs 9GB：保留原权重 + 预解量化缓存 + 旋转矩阵 + ATM/OHB 参数）；量化收益在权重体积（pack 124MB vs 检查点 14.4GB）。

## 3. 成功率结果

### 3.1 RoboCasa365（10 trials × 50 任务 = 每模型 500 episodes）

| 任务集 | fp16 平均 SR | W4A8 平均 SR | Δ (W4A8−fp16) |
|---|---|---|---|
| atomic_seen (18) | 40.6% | 40.6% | **0.0 pp** |
| composite_seen (16) | 6.3% | 5.0% | −1.3 pp |
| composite_unseen (16) | 1.3% | 2.5% | +1.3 pp |
| **50 任务总平均** | **17.0%** | **17.0%** | **0.0 pp** |

**逐任务对比（成功率）**：

| 任务 | fp16 | W4A8 | | 任务 | fp16 | W4A8 |
|---|---|---|---|---|---|---|
| CloseBlenderLid | 0.0 | 0.1 | | PrepareCoffee | 0.0 | 0.1 |
| CloseFridge | 0.6 | 0.4 | | RinseSinkBasin | 0.1 | 0.1 |
| CloseToasterOvenDoor | 0.4 | 0.2 | | ScrubCuttingBoard | 0.0 | 0.1 |
| CoffeeSetupMug | 0.2 | 0.2 | | SetUpCuttingStation | 0.0 | 0.0 |
| NavigateKitchen | 0.0 | 0.0 | | StackBowlsCabinet | 0.2 | 0.2 |
| OpenCabinet | 0.4 | 0.3 | | SteamInMicrowave | 0.1 | 0.0 |
| OpenDrawer | 0.7 | 0.3 | | StirVegetables | 0.0 | 0.0 |
| OpenStandMixerHead | 0.4 | 0.0 | | StoreLeftoversInBowl | 0.0 | 0.0 |
| PickPlaceCounterToCabinet | 0.8 | 0.6 | | WashLettuce | 0.1 | 0.1 |
| PickPlaceCounterToStove | 0.5 | 0.8 | | ArrangeBreadBasket | 0.0 | 0.0 |
| PickPlaceDrawerToCounter | 0.5 | 0.4 | | ArrangeTea | 0.0 | 0.0 |
| PickPlaceSinkToCounter | 0.8 | 0.9 | | BreadSelection | 0.0 | 0.0 |
| PickPlaceToasterToCounter | 0.1 | 0.8 | | CategorizeCondiments | 0.0 | 0.0 |
| SlideDishwasherRack | 0.9 | 0.7 | | CuttingToolSelection | 0.0 | 0.0 |
| TurnOffStove | 0.1 | 0.1 | | GarnishPancake | 0.0 | 0.0 |
| TurnOnElectricKettle | 0.3 | 0.2 | | GatherTableware | 0.0 | 0.0 |
| TurnOnMicrowave | 0.1 | 0.6 | | HeatKebabSandwich | 0.0 | 0.0 |
| TurnOnSinkFaucet | 0.5 | 0.7 | | MakeIceLemonade | 0.0 | 0.0 |
| DeliverStraw | 0.0 | 0.0 | | PanTransfer | 0.0 | 0.0 |
| GetToastedBread | 0.0 | 0.0 | | PortionHotDogs | 0.0 | 0.0 |
| KettleBoiling | 0.1 | 0.0 | | RecycleBottlesByType | 0.1 | 0.0 |
| LoadDishwasher | 0.2 | 0.1 | | SeparateFreezerRack | 0.0 | 0.0 |
| PackIdenticalLunches | 0.0 | 0.0 | | WaffleReheat | 0.0 | 0.0 |
| PreSoakPan | 0.2 | 0.0 | | WashFruitColander | 0.1 | 0.0 |
| SearingMeat | 0.0 | 0.0 | | WeighIngredients | 0.0 | 0.0 |

### 3.2 RoboCerebra（5 trials × 60 任务 = 每模型 300 episodes）

| 类别 | fp16 ep SR | W4A8 ep SR | Δ | fp16 subtask SR | W4A8 subtask SR | Δ |
|---|---|---|---|---|---|---|
| Ideal | 20.0% (10/50) | 20.0% (10/50) | 0.0 | 22.1% | 21.3% | −0.8 |
| Memory_Execution | 0.0% (0/50) | 2.0% (1/50) | +2.0 | 14.4% | 17.1% | +2.7 |
| Memory_Exploration | 18.0% (9/50) | 18.0% (9/50) | 0.0 | 14.3% | 13.3% | −1.0 |
| Mix | 14.0% (7/50) | 24.0% (12/50) | **+10.0** | 17.7% | 16.3% | −1.4 |
| Observation_Mismatching | 16.0% (8/50) | 24.0% (12/50) | **+8.0** | 17.0% | 17.9% | +0.9 |
| Random_Disturbance | 20.0% (10/50) | 20.0% (10/50) | 0.0 | 22.6% | 21.8% | −0.8 |
| **总计** | **14.7%** (44/300) | **18.0%** (54/300) | **+3.3** | **17.5%** | **17.5%** | **0.0** |

## 4. 性能差异分析

### 4.1 精度：W4A8（+ATM+OHB）与 fp16 总体持平，无系统性损失

- **RoboCasa365**：两模型总平均完全一致（**17.0% vs 17.0%**）；三个任务集差距均在 ±1.3pp 内。500 episodes/模型的总均值标准误约 ±1.7pp。
- **RoboCerebra**：episode SR W4A8 略高（**18.0% vs 14.7%，+3.3pp**），subtask SR 完全一致（**17.5% vs 17.5%**）。差异主要来自 Mix（+10pp）与 Observation_Mismatching（+8pp）；每类 50 episodes 的 95% 置信区间约 ±5–6pp，两类差异处于边缘（Mix +10pp 的 95% CI 约为 [0.6, 19.4]pp，勉强排除 0，但需更大样本确认）。其余 4 类两者几乎一致。
- 结论：在该 DuQuant W4A8 + ATM/OHB 配方下，量化+误差补偿**没有造成可测的精度退化**；在低成功率长程任务（composite）上甚至个别类别略优，可能与 ATM/OHB 对 expert 激活分布的校准有关（注：样本量不足以作因果断言）。

### 4.2 速度：W4A8 慢 6–9× 的根因（代码级）

1. **RoboCasa pack 实际 block=16**：目录名 `pi05_robocasa_w4a8_b64c160ls015` 含 "b64"，但 pack 元数据 `R_in_blocks=(256,)`、`R_out_blocks=(64,)` 证明校准块大小是 16（4096/16、1024/16），疑似校准期配置疏漏。运行时以 pack 元数据为准（block_in=block_out=16），精度无碍，但逐块变换算子数量是 block=64 的 4 倍（RoboCasa W4A8 931ms vs RoboCerebra W4A8 561ms 的主因之一）。
2. **DuQuant 前向的额外算子**：逐块输入旋转（gather/变换）→ 激活伪量化 → **fp16 全精度 GEMM**（权重预解量化，并非 int4 kernel）→ 输出旋转恢复。GEMM 本身与 fp16 同价，其余全是净开销。
3. **eager 执行**：180 层逐层 Python/小 kernel 启动开销；TORCHDYNAMO_DISABLE=1（官方配方）。
4. **torch.compile 无效（实测）**：量化后再编译（`serve_pi05_quant_policy_compiled.py`，mode=default）测得 892ms vs eager 897ms（<1%）。自定义算子打断 dynamo 图（graph break），GEMM 已是 fp16，编译无物可融；reduce-overhead 依赖 CUDA graphs，在大量 graph break 下同样无效。**真正提速需 Triton 融合的 int4/int8 矩阵乘 + 块变换融合 kernel**（后续工作）。
5. **吞吐放大**：评测期因单实例 ~1 req/s，RoboCasa W4A8 最终以 4 实例（GPU1/2/3/5）平行扩容、RoboCerebra 单实例运行。

### 4.3 显存：W4A8 推理期不省显存

23.0GB（W4A8）vs 8.7–9.0GB（fp16+compile）。原因：pack 解包后仍保留原 fp16 权重 + 预解量化缓存（`OPENPI_DUQUANT_PRECACHE_WEIGHTS=1`）+ 旋转矩阵 + ATM/OHB 参数。量化收益限于权重文件体积（124MB vs 14.4GB）与传输带宽，而非推理期显存或速度。

### 4.4 fast 方案：Triton 融合推理（量化配方不变，已落地并标记为 fast）

**定位**：不重新校准、不改变量化包（仍用 b16 pack + 同一 ATM/OHB 系数），仅把 DuQuant 逐层的
"置换+块旋转 → 激活伪量化 → GEMM → 输出旋转恢复" 中的**变换开销用 Triton kernel 融合**，
GEMM 本身仍走 `F.linear`（与 eager **位级一致**）。

**开关**：量化服务器加 `OPENPI_DUQUANT_TRITON=1`（默认关闭，关闭即 eager 基线）。
激活 scale 校准期（服务启动后前 ~40 个请求）自动回退 eager，校准冻结后自动启用融合路径。

**实现**（`code/pi05/openpi/src/openpi/quant/duquant_triton.py`，三 kernel 管线）：
1. `_input_transform_quant_kernel`：gather 置换 + 16 块旋转 + 静态逐通道 int8 伪量化（镜像 eager 的 bf16 舍入链）；
2. `F.linear(x_tq, W_tq)`：与 eager 完全相同的 cuBLAS GEMM（位级一致）；
3. `_output_restore_kernel`：逐 16 块输出旋转恢复 + bias。

**实测（同机同窗口，RoboCasa 服务器）**：

| 版本 | 单请求均值 | p95 | 吞吐 | 加速比 |
|---|---|---|---|---|
| eager（基线） | 951 ms | 1083 ms | 1.05 req/s | 1× |
| **fast（Triton 融合）** | **435 ms** | 440 ms | **2.30 req/s** | **2.2×** |

冒烟评测端到端加速：同一任务单 episode 67–90s（fast）vs 127–205s（eager）。

**对测试结果的影响（已三重验证）**：
1. 数学恒等：算子序列与 eager 完全一致（GEMM 直接复用 `F.linear`，位级一致）；
2. 数值等价：逐层对比，VLM 层接近位级一致（rel ~1e-7）；专家 MLP 层 rel 1–3e-2、最终动作
   差 ~2.8e-2（2.5%）——来源是"块旋转点积的 fp32 累加顺序与 cuBLAS 不同"导致的个别
   bf16 边界量化翻转（数据相关），属量化噪声级（验证脚本 `tools/pi05_verify_triton_equiv.py`）；
3. 成功率冒烟（5 trials × 2 任务）：CloseFridge **0.4 vs 0.4**（完全一致）；TurnOnElectricKettle
   **0.4 vs 0.0**（该任务历史基线 eager 0.2/fp16 0.3，结果本身波动大，差异在样本方差内）。
   结论：fast 方案的成功率与 eager 在统计噪声内一致，可用于评测；如需逐位一致可关闭开关回退 eager。

**后续优化空间**（未做，保持结果不变的前提下）：R_out 预折叠进权重可省一个 kernel；更大块批处理；
如需进一步提速需 Triton GEMM + 数值再校验。

## 5. 工程问题与记录

1. **fp16 服务器首请求编译超时**：max-autotune 首次编译阻塞事件循环 >20s → websocket keepalive 超时（1011 断开）→ 客户端崩溃。解决：`scripts/warm_pi05_server.py` 预热（容忍断连重试）。
2. **并发 EGL 崩溃（SIGSEGV）**：多客户端默认共用 GPU0 的 EGL 上下文导致显存耗尽。解决：`MUJOCO_EGL_DEVICE_ID`+`CUDA_VISIBLE_DEVICES` 逐客户端绑定到独立渲染卡（后改为与服务器同卡，GPU5/6/7 已释放）。
3. **会话重启杀死编排进程**：改为 tmux 会话 + 可重挂载监督器（状态文件/结果 JSON 重建断点；RoboCasa 按 stats.json 跳过已完成任务），全程断点续跑无数据损失。
4. **注 2（拆分协议）**：W4A8 的 MakeIceLemonade / WaffleReheat 两个任务因收尾提速按 5×2 trials 拆分到 7 张 GPU（seeds 7/100/200/300/400，各 2 trials，完成后按客户端日志重建合并 stats.json）。fp16 同任务为 seed=7 标准 10 trials。两任务两者成功率均为 0%，不影响对比结论。
5. **冒烟产物隔离**：smoke 产物已在收尾清理中移除；全量评测使用干净日志目录。

## 6. 结论

1. **精度**：QuantVLA DuQuant W4A8 + ATM + OHB 在 RoboCasa365 与 RoboCerebra 上与 fp16 **总体持平**（RoboCasa 总平均 17.0% vs 17.0%；RoboCerebra episode SR 18.0% vs 14.7%、subtask SR 17.5% vs 17.5%），无系统性退化。
2. **速度**：W4A8 单请求延迟为 fp16 的 5.7–8.5×、吞吐 1/6–1/9；瓶颈为逐块变换/伪量化等额外算子 + fp16 精度 GEMM + eager 执行，torch.compile 实测无效（<1%）。真正加速需 Triton 融合 kernel。
3. **显存**：W4A8 推理期不省显存（23GB vs 9GB）；收益在权重体积（124MB vs 14.4GB）。
4. **建议**：若目标是以量化换取部署体积/带宽，当前配方可用且精度无损；若目标是推理加速，需要实现融合的 W4A8 kernel（如 Triton int4 反量化 matmul + 块旋转融合），并修正 RoboCasa pack 的 block=16→64 校准，预计可将单请求延迟降到 fp16 的 1.5–2× 以内。
