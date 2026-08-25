# RoboCasa365 + RoboCerebra pi0.5 评测环境搭建与冒烟测试记录

日期: 2026-08-13 | 机器: 8× NVIDIA A40 | 本文件记录环境搭建结果与冒烟测试(完整评测由用户后续用脚本启动)

## 环境清单

| 环境 | 用途 | 关键版本 |
|---|---|---|
| `robocasa365` (py3.11) | RoboCasa365 仿真 + eval 客户端 | robocasa 1.0.1, robosuite 1.5.2 (master), mujoco 3.3.1, numpy 2.2.5 |
| `robocerebra_test` (py3.10) | RoboCerebra eval 客户端 (fork LIBERO) | libero 0.1.0 (RoboCerebra fork), robosuite 1.4.0 |
| `openpi` (py3.11) | pi0.5 策略服务器 (fp16 + W4A8) | openpi (vendored), torch 2.6.0 |

## Checkpoint / 量化产物

| 产物 | 路径 | 状态 |
|---|---|---|
| RoboCasa pi0.5 fp16 (pytorch) | `checkpoints/robocasa/pi05_pretrain_human300_pytorch/` (14.4GB fp32, action_horizon=50) | ✅ 由 orbax 转换, 加载+前向验证通过 (50×12 有限动作) |
| RoboCasa DuQuant W4A8 pack | `code/pi05/packs/pi05_robocasa_w4a8_b64c160ls015/` (180 层, 124M) | ✅ 校准完成 (合成 obs, TORCHDYNAMO_DISABLE=1) |
| RoboCasa ATM/OHB | `code/pi05/packs/atm_alpha_beta_pi05_robocasa.json` (18 层) | ✅ 校准完成 |
| RoboCasa 服务器 | :8003 fp16 (GPU2) / :8004 W4A8+ATM+OHB (GPU5) | ✅ 均上线, 合成请求返回 50×12 有限动作 |
| RoboCasa assets | `code/robocasa/robocasa/models/assets/` (24GB) | ✅ 同步完成 (SSH 公钥 + rsync, 实测 25-75MB/s) |
| RoboCerebra benchmark 数据 | `data/RoboCerebra_dl/RoboCerebraBench/` (60 cases + init_files) | ✅ 已有 |
| pi05 libero (fp16/W4A8) | `code/pi05/checkpoints/pi05_libero_pytorch` + 已有 pack | ✅ 复用 (:8001/:8002 均验证) |

## 服务器端口约定

| 端口 | 模型 | 量化 | GPU | 脚本 |
|---|---|---|---|---|
| 8001 | pi05_libero | fp16 | 6 | `scripts/serve_policy.py` |
| 8002 | pi05_libero | W4A8+ATM+OHB | 1 | `scripts/serve_pi05_quant_policy.py` |
| 8003 | pi05_pretrain_human300 (RoboCasa) | fp16 | 2 | `code/pi05/run_robocasa_serve.sh` |
| 8004 | pi05_pretrain_human300 (RoboCasa) | W4A8+ATM+OHB | 5 | `code/pi05/run_robocasa_serve_quant.sh` |

## 关键实现决策

1. **RoboCasa checkpoint 转换**: orbax JAX → PyTorch, 用 repo 现有 `examples/convert_jax_model_to_pytorch.py`;
   config `pi05_pretrain_human300` = `Pi0Config(pi05=True, max_token_len=200)`, **保持默认 action_horizon=50 / discrete_state_input=True**(与官方 fork 训练配置一致, 不能照抄 pi05_libero 的 10/False)。
2. **归一化**: fork 的 `DataConfig.use_quantile_norm` 默认为 False (z-score); 本地新版 openpi 对 PI05 强制 quantile=True 且 robocasa norm_stats 无 q01/q99 → 在 `LeRobotRobocasaDataConfig.create()` 强制 `use_quantile_norm=False`, 与官方训练行为一致。
3. **量化校准**: 复用 libero 的合成观测校准工具 (`OPENPI_OBS_FORMAT=robocasa` 分支: 12 维 state + 3 相机), 无需真实数据集。
4. **RoboCerebra 评测**: 完整移植官方 evaluation/ (episode 分段/resume/动态扰动逻辑保留), 仅替换 OpenVLA → pi0.5 websocket 客户端; 动作不做 OpenVLA 式 gripper 转换 (pi05 输出即 env 空间)。
5. **视频保存**: pyav 在本机崩溃 (`expected bytes, NoneType found`) → 改用 imageio-ffmpeg 插件 (format="FFMPEG")。
6. **Assets 同步**: 参数化 SFTP 增量同步 (仅缺失/大小不同的文件), 不重复拷贝已有内容。

## RoboCerebra 冒烟 (6 类 × case1 × 1 trial, 复现论文协议的 System-1 设置)

| 类别 | fp16 episode 成功 | fp16 子任务完成 | W4A8 episode 成功 | W4A8 子任务完成 |
|---|---|---|---|---|
| Ideal | 0/1 | 1/6 (16.7%) | 0/1 | 1/6 (16.7%) |
| Memory_Execution | 0/1 | 0/7 (0%) | 0/1 | 1/7 (14.3%) |
| Memory_Exploration | 0/1 | 3/12 (25.0%) | 0/1 | 2/12 (16.7%) |
| Mix | 0/1 | 2/11 (18.2%) | **1/1** | 4/11 (36.4%) |
| Observation_Mismatching | 0/1 | 0/5 (0%) | 0/1 | 0/5 (0%) |
| Random_Disturbance | 0/1 | 1/6 (16.7%) | 0/1 | 1/6 (16.7%) |
| **合计** | **0/6** | **7/47 (14.9%)** | **1/6 (16.7%)** | **9/47 (19.1%)** |

注: 单 trial 冒烟以验证管线为主; episode 全成功预期很低 (论文中 System-1 基线 Ideal 仅 4.05%)。
子任务能完成说明 pi0.5 策略驱动、init 文件、分段/resume/扰动逻辑全部生效。

## RoboCasa365 冒烟 (atomic_seen 2 任务 × 3 trials, split=pretrain)

| 任务 | fp16 成功/trials | W4A8 成功/trials | 备注 |
|---|---|---|---|
| CloseFridge | 1/3 (33.3%) | 2/3 (66.7%) | 论文 π0.5 atomic_seen 均值 39.6%, 冒烟结果量级一致 |
| TurnOnElectricKettle | 1/3 (33.3%) | **3/3 (100%)** | W4A8 与 fp16 使用独立日志目录 |
| **合计** | **2/6 (33.3%)** | **5/6 (83.3%)** | 小样本仅供参考, 不具统计意义 |

⚠️ **重要**: fp16 与 W4A8 评测**必须使用不同的 log 目录**(客户端会在任务已有 stats.json 时跳过)。
完整评测建议:
- fp16: 默认 `ROBOCASA_LOG_DIR` (checkpoints/robocasa/pi05_pretrain_human300/...)
- W4A8: `export ROBOCASA_LOG_DIR=checkpoints/robocasa/pi05_pretrain_human300_pytorch_quant`

## 完整评测启动方式 (用户自行运行)

```bash
# RoboCasa365 完整评测 (官方协议 50 rollouts × 50 任务; fp16 与 W4A8 必须用不同日志目录!)
./code/pi05/run_robocasa_eval.sh --args.port 8003 --args.task_set atomic_seen composite_seen composite_unseen --args.num_trials 50
ROBOCASA_LOG_DIR=checkpoints/robocasa/pi05_pretrain_human300_pytorch_quant \
  ./code/pi05/run_robocasa_eval.sh --args.port 8004 --args.task_set atomic_seen composite_seen composite_unseen --args.num_trials 50
# 汇总 (分别对两个日志目录):
# conda activate robocasa365 && python code/pi05/openpi/examples/robocasa/get_eval_stats.py \
#   --dir checkpoints/robocasa/pi05_pretrain_human300/multitask_learning/75000
# conda activate robocasa365 && python code/pi05/openpi/examples/robocasa/get_eval_stats.py \
#   --dir checkpoints/robocasa/pi05_pretrain_human300_pytorch_quant

# RoboCerebra 完整评测 (论文协议 60 tasks × 10 trials)
./scripts/run_robocerebra_eval.sh --num_trials_per_task 10 --port 8001   # fp16
./scripts/run_robocerebra_eval.sh --num_trials_per_task 10 --port 8002   # W4A8
```

## 文件结构 (按功能归类)

```
QuantVLA/
├── code/                              # ===== 全部代码与评测基准 =====
│   ├── gr00t/                         # GR00T N1.5 代码包
│   ├── pi05/                          # ===== pi0.5 全部资产 =====
│   │   ├── run_libero_serve[_quant].sh    # LIBERO 服务器 (fp16 :8001 / W4A8 :8002)
│   │   ├── run_robocasa_serve[_quant].sh  # RoboCasa 服务器 (fp16 :8003 / W4A8 :8004)
│   │   ├── run_robocasa_eval.sh           # RoboCasa 评测客户端 (robocasa365 env)
│   │   ├── checkpoints/               # 模型权重 (pi05_libero_pytorch)
│   │   ├── packs/                     # 量化产物 (libero + robocasa DuQuant pack / ATM-OHB json)
│   │   ├── rollouts/                  # LIBERO 评测视频 (既有)
│   │   └── openpi/                    # 源码; 含 examples/robocasa 客户端、robocasa_policy、校准工具
│   ├── examples/
│   │   ├── RoboCerebra/eval/          # ===== RoboCerebra 评测栈 (pi0.5 版, 9 个文件) =====
│   │   ├── RoboCasa/                  # GR00T tabletop (既有)
│   │   └── Libero/eval/               # GR00T LIBERO 评测 (既有)
│   ├── LIBERO/                        # LIBERO 评测基准库 (两模型共用)
│   ├── robocasa/                      # ===== RoboCasa365 仿真 (v1.0.1) =====
│   │   └── robocasa/models/assets/    # 仿真资产 (24GB, 已通过 rsync 同步完毕)
│   └── third_party/                   # 外部代码: robosuite(master) / RoboCerebra(main)
├── checkpoints/                       # ===== 模型权重与量化缓存 =====
│   ├── gr00t/                         # GR00T 检查点 (libero-* 微调 + gr00t-n1.5-3b 基座)
│   ├── robocasa/                      # RoboCasa pi0.5: orbax 原始 (42G) + pytorch 转换 (14.4G)
│   └── packs/gr00t/                   # GR00T DuQuant pack 缓存 + atm_alpha_beta_*.json
├── data/                              # 数据集: RoboCerebra_dl/ (benchmark+RLDS), RoboCasa-XEmbodiment/, libero_*
├── scripts/                           # ===== 启动/评测/运维脚本 =====
│   ├── run_inference_server.sh / run_libero_eval.sh / run_quantvla.sh   # GR00T (既有)
│   ├── run_robocerebra_eval.sh        # RoboCerebra 评测入口 (本次新增)
│   ├── activate_*.sh                  # 便捷激活 (既有)
│   ├── sync_robocasa_assets.py        # assets 增量同步 (SFTP, 可并行)
│   ├── tarstream_robocasa_assets.py   # assets tar 流式传输 (备选)
│   ├── tools/ + deployment_scripts/   # GR00T 量化校准/TensorRT 工具
│   └── inference_service.py 等        # GR00T 服务 (既有)
├── environments/                      # conda env 定义
├── docs/                              # 文档 (本文档 + 环境说明 + 评测结果 + getting_started)
├── runs/                              # 运行产物 (评测日志/视频/冒烟结果)
└── tests/ assets/ demo_data/          # 上游仓库文件 (未动)

传输备忘: assets 首次同步用 SFTP/tar 流较慢 (~1-4MB/s); 已把本机公钥
(~/.ssh/id_ed25519_robocasa) 装入服务器, 后续增量更新直接:
rsync -a --partial -e "ssh -i ~/.ssh/id_ed25519_robocasa -p 50029" \
  wubohan@112.65.216.193:/data1/wubohan/robocasa/robocasa/models/assets/ \
  /home1/gyy/vla/QuantVLA/code/robocasa/robocasa/models/assets/   # 实测 25-75MB/s
```
