# 🚀 wanUpsampler 项目主线

> **最后更新**：2026-05-15  
> **当前阶段**：Stage 2 — Clean Latent Resizer（LTX2 风格）  
> **核心任务**：Wan 视频扩散模型采样中途的 480p → 720p 分辨率切换

---

## 📌 一、项目定位

`wanUpsampler` 的目标是训练一个 **在 Wan 推理中途使用的 1.5× latent 空间放大器**，替代 LightX2V 中原有的固定三线性插值（trilinear interpolation）分辨率切换方案。

### 推理链路

```text
低分辨率 Wan 采样前半程（480p latent）
         ↓
  x0_pred_lr = x_t - sigma * eps   ← 估计当前 clean latent
         ↓
  WanCleanLatentResizerStage2(x0_pred_lr)  ← 学习到的 1.5× 放大
         ↓
  z0_hr（720p clean latent）
         ↓
  re-noise 到同一 timestep
         ↓
  高分辨率 Wan 继续采样后半程（720p）
         ↓
  VAE decode → 720p 视频
```

### 与 LightX2V 的关系

| 组件 | 说明 |
|------|------|
| LightX2V | 上游推理框架，提供 Wan 模型加载、采样调度、VAE 编解码 |
| `wan_sr/` | 本项目核心 Python 包：模型、数据、损失、训练工具 |
| `changing_resolution/` | Stage 2 的 480p→720p 训练/评估入口和 LightX2V 桥接 |

---

## 🗂️ 二、仓库结构

```text
wanUpsampler/
├── wan_sr/                          # 🔧 核心 Python 包
│   ├── models/
│   │   ├── stage2_resizer.py        # ⭐ WanCleanLatentResizerStage2 模型定义
│   │   ├── blocks.py                # SigmaConditionedResBlock3D / SpatialPixelShuffle2x
│   │   ├── sigma_embedding.py       # FourierFeatures / SigmaEmbedding / AdaGroupNorm3D
│   │   └── factory.py               # build_clean_latent_resizer() 工厂函数
│   ├── data/
│   │   ├── clean_latent_lmdb_dataset.py  # ⭐ LMDB 分片数据集（当前主线）
│   │   ├── clean_latent_pair_dataset.py  # 兼容旧版 safetensors 文件格式
│   │   ├── degradation.py           # 随机退化策略（blur / noise / JPEG）
│   │   └── video_io.py              # 视频读写工具
│   ├── losses/
│   │   ├── clean_latent_losses.py   # ⭐ CleanLatentResizeLoss（四合一损失）
│   │   └── latent_losses.py         # Charbonnier / temporal difference 基础损失
│   ├── schedulers/
│   │   ├── sigma_sampler.py         # SigmaSampler（mid / uniform / clean 模式）
│   │   └── noise_utils.py           # add_flow_noise / spatial_downsample
│   ├── pipelines/
│   │   └── transition.py            # ⭐ transition_lr_to_hr() 推理桥接函数
│   └── training/
│       ├── config.py                # YAML 配置加载 / deep_update
│       ├── checkpoint.py            # save_checkpoint / load_checkpoint
│       └── ema.py                   # EMA 指数移动平均
│
├── changing_resolution/             # 🎯 Stage 2 主线入口
│   ├── configs/
│   │   └── train_clean_480p_to_720p_lmdb_stage2.yaml  # ⭐ 训练配置
│   ├── scripts/
│   │   ├── train/
│   │   │   ├── train_clean_latent_resizer_stage2.py    # ⭐ 训练脚本
│   │   │   ├── run_clean_480p720p_stage2_lmdb_training.sh
│   │   │   └── tmux_run_clean_480p720p_stage2_lmdb_training.sh
│   │   └── eval/
│   │       ├── eval_clean_resizer_operator_compare.py  # ⭐ 算子对比评估
│   │       ├── summarize_operator_compare_table.py     # 评估结果汇总
│   │       ├── run_clean_480p720p_stage2_operator_compare_multigpu.sh
│   │       ├── run_clean_480p720p_stage2_chain_ab_compare_multigpu.sh
│   │       └── run_clean_480p720p_stage2_change_step_sweep.sh
│   ├── diagrams/
│   │   └── ltx2_latent_upsampler.mmd  # LTX2 架构参考图
│   ├── STAGE2_MODEL_PLAN.md
│   └── STAGE2_RUNBOOK.md
│
├── configs/
│   └── local_paths.sh               # 机器相关路径配置
│
├── experiments/                     # 探索性代码（非主线入口）
│   └── wan22_lmdb/                  # LMDB 构建 / clean latent 生成
│
├── .archive/                        # 📦 已归档代码
│   ├── stage1/                      # Stage 1 残差 clean-resizer
│   └── v1/                          # 早期 noisy-to-clean V1 脚本
│
├── codex.md                         # 原始项目设计文档
├── PROGRESS.md                      # 进度追踪
└── requirements.txt
```

---

## 🧱 三、核心模型：`WanCleanLatentResizerStage2`

### 3.1 模型契约

| 项目 | 规格 |
|------|------|
| 输入 | `z0_lr` 或 `x0_pred_lr`，shape `[B, 16, T, 60, 104]` |
| 输出 | `z0_hr`，shape `[B, 16, T, 90, 156]` |
| 缩放倍率 | **1.5×** 空间（时间维度不变） |
| 通道数 | 16（Wan VAE latent 通道） |

### 3.2 架构概览

```text
z0_lr [B, 16, T, H, W]
        ↓
  Conv3d stem (16 → 256) + GroupNorm + SiLU
        ↓
  ResBlock × (num_res_blocks // 2)     ← 前置残差块（LTX2 风格）
        ↓
  SpatialRationalResampler (1.5×)       ← 学习到的有理数重采样
  ├── Conv3d (256 → 256×9)             ← 通道扩展
  ├── PixelShuffle ×3                   ← 空间放大 3 倍
  └── BlurDownsample /2                 ← 抗混叠下采样 2 倍
        ↓
  ResBlock × (num_res_blocks - pre)     ← 后置残差块
        ↓
  Conv3d output (256 → 16)
        ↓
  z0_hr [B, 16, T, 1.5H, 1.5W]
```

### 3.3 关键设计决策

| 决策 | 说明 |
|------|------|
| **LTX2 风格 ResBlock** | Conv3d → GroupNorm → SiLU → Conv3d → GroupNorm → 残差加和 |
| **有理数重采样 3/2** | PixelShuffle ×3 + BlurDownsample /2 = 净 1.5×，避免直接插值 |
| **无 Sigma 条件注入** | Stage 2 输入是 clean latent（`x0_pred`），不需要噪声水平条件 |
| **可选残差跳跃** | `residual_skip=True` 时，输出 = trilinear 上采样 + 残差预测 |

### 3.4 模型规格

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `in_channels` | 16 | Wan VAE latent 通道 |
| `out_channels` | 16 | 输出通道（与输入一致） |
| `hidden_channels` | 256 | 内部特征通道数 |
| `num_res_blocks` | 8 | 残差块总数（前后各半） |
| `scale_factor` | 1.5 | 空间缩放倍率 |
| `resblock_type` | `ltx2` | 残差块类型 |
| `resize_op` | `rational_conv3d_pixel_shuffle` | 重采样算子 |

---

## 📊 四、损失函数：`CleanLatentResizeLoss`

### 损失组成

| 损失项 | 权重 | 公式 | 作用 |
|--------|------|------|------|
| **Latent Charbonnier** | 1.0 | `√((pred - gt)² + ε²)` | HR latent 逐像素 fidelity |
| **Low-Freq Consistency** | 0.2 | `|down(pred) - z0_lr|₁` | 约束下采样后与 LR 一致 |
| **Temporal Difference** | 0.1 | `|Δt_pred - Δt_gt|₁` | 时间连续性约束 |
| **Residual Regularization** | 0.0 | `|pred - up(z0_lr)|₁` | 残差幅度正则（默认关闭） |

```text
L_total = 1.0 × L_latent + 0.2 × L_low + 0.1 × L_temp + 0.0 × L_residual
```

---

## 🔄 五、数据管线

### 5.1 数据流

```text
Wan2.1 720p 生成视频（1000 条 prompt）
        ↓
  下采样到 480p（bicubic）
        ↓
  Wan VAE encode → z0_720p, z0_480p
        ↓
  分片 LMDB 存储（每片 ~250 条）
        ↓
  CleanLatentLMDBDataset 在线读取
```

### 5.2 LMDB 数据格式

| 字段 | 说明 |
|------|------|
| `z0_lr` | 480p clean latent，shape `[16, T, 60, 104]`，dtype float16 |
| `z0_hr` | 720p clean latent，shape `[16, T, 90, 156]`，dtype float16 |
| `prompt` | 生成该视频的文本提示词 |
| `meta` | JSON 元信息（VAE 版本、帧数等） |

### 5.3 数据集类

| 类 | 文件 | 用途 |
|----|------|------|
| [`CleanLatentLMDBDataset`](../wan_sr/data/clean_latent_lmdb_dataset.py:13) | `clean_latent_lmdb_dataset.py` | ⭐ 当前主线：分片 LMDB 读取 |
| [`CleanLatentPairDataset`](../wan_sr/data/clean_latent_pair_dataset.py) | `clean_latent_pair_dataset.py` | 兼容旧版 safetensors 文件格式 |

---

## 🏋️ 六、训练流程

### 6.1 训练配置

```yaml
# changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml
data_dir: data/changing_resolution/lmdb_480p720p_1k
data_format: lmdb

model:
  in_channels: 16
  hidden_channels: 256
  num_res_blocks: 8
  scale_factor: 1.5
  resblock_type: ltx2
  resize_op: rational_conv3d_pixel_shuffle

train:
  max_steps: 50000
  batch_size: 1
  grad_accum: 8          # 等效 batch = 8
  lr: 1e-4
  weight_decay: 0.01
  precision: bf16
  ema_decay: 0.9999
  grad_clip_norm: 1.0
  val_ratio: 0.05
```

### 6.2 训练脚本入口

```bash
# 1. 环境检查
bash changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh check

# 2. 启动训练（单卡）
CUDA_VISIBLE_DEVICES=0 \
bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage2_lmdb_training.sh

# 3. 自定义超参覆盖
MAX_STEPS=100000 LR=5e-5 GRAD_ACCUM=16 \
bash changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh train
```

### 6.3 训练特性

| 特性 | 实现 |
|------|------|
| **混合精度** | bf16 autocast + GradScaler（fp16 时启用） |
| **梯度累积** | 默认 8 步，等效 batch = 8 |
| **EMA** | decay=0.9999，验证默认使用 EMA 权重 |
| **梯度裁剪** | max_norm=1.0 |
| **Checkpoint** | 每 1000 步保存 `step_XXXX.pt` + `latest.pt` |
| **Best Val** | 跟踪验证集最优 loss，保存 `best_val.pt` |
| **续训** | `--resume latest.pt` 恢复 model/optimizer/EMA |
| **指标记录** | `metrics.jsonl` 每 20 步记录 train/val 指标 |

### 6.4 训练监控

```text
outputs/changing_resolution_clean_480p720p_stage2_lmdb/
├── train_config.yaml       # 训练配置归档
├── metrics.jsonl           # 训练/验证指标（每行 JSON）
├── latest.pt               # 最新 checkpoint
├── best_val.pt             # 验证集最优 checkpoint
├── best_val.json           # 最优验证指标
└── step_0001000.pt         # 定期 checkpoint
```

---

## 📈 七、评估体系

### 7.1 算子对比评估（Operator Compare）

对比 **trilinear 插值** vs **Stage 2 学习到的重采样**：

```bash
TOTAL_SAMPLES=32 GPU_IDS=0,1,2,3 \
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_operator_compare_multigpu.sh
```

**评估指标**：

| 指标 | 方向 | 说明 |
|------|------|------|
| Latent L1 | ↓ 越低越好 | latent 空间逐像素差异 |
| PSNR | ↑ 越高越好 | VAE 解码后 RGB 峰值信噪比 |
| SSIM | ↑ 越高越好 | 结构相似性 |
| LPIPS | ↓ 越低越好 | 感知相似度（AlexNet） |
| Temporal L1 | ↓ 越低越好 | 帧间差分一致性 |

**输出**：

```text
outputs/changing_resolution_operator_compare_stage2/
├── metrics_val.jsonl       # 逐样本指标
└── tables/
    ├── samples_val.csv     # 样本明细表
    ├── samples_val.md      # Markdown 表格
    ├── summary_val.csv     # 汇总统计
    ├── summary_val.md
    └── summary_val.json
```

### 7.2 生成链路 A/B 对比（Chain A/B Compare）

在完整 LightX2V 推理链路中对比：

```text
stop480（低分直接结束） vs interp720（三线性插值切换） vs stage2（学习到的切换）
```

```bash
TOTAL_SAMPLES=16 GPU_IDS=0,1,2,3 \
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_chain_ab_compare_multigpu.sh
```

### 7.3 切换步长扫描（Change-Step Sweep）

扫描不同 handoff step，观察切换时机对生成质量的影响：

```bash
STEP_START=10 STEP_END=50 STEP_STRIDE=1 \
bash changing_resolution/scripts/eval/run_clean_480p720p_stage2_change_step_sweep.sh
```

---

## 🔌 八、推理桥接：`transition_lr_to_hr`

[`transition_lr_to_hr()`](../wan_sr/pipelines/transition.py:9) 是连接 Stage 2 模型与 LightX2V 推理管线的核心函数：

```python
@torch.no_grad()
def transition_lr_to_hr(x_t_lr, sigma, upsampler, noise=None):
    """LR noisy latent → HR noisy latent（同一 sigma）"""
    pred_z0_hr = upsampler(x_t_lr)          # Stage 2 预测 HR clean latent
    x_t_hr, _ = add_flow_noise(pred_z0_hr, sigma, noise=noise)  # re-noise
    return x_t_hr, pred_z0_hr
```

**Flow-style 加噪**（[`add_flow_noise()`](../wan_sr/schedulers/noise_utils.py:14)）：

```text
x_sigma = (1 - sigma) × z0 + sigma × ε
```

---

## 🗺️ 九、完整工作流

```text
┌─────────────────────────────────────────────────────────┐
│  Step 1: 构建 LMDB 数据集                                │
│  bash changing_resolution/scripts/data/                 │
│       tmux_build_clean_lmdb_480p720p_1k_multigpu.sh     │
│  输出: data/changing_resolution/lmdb_480p720p_1k/       │
├─────────────────────────────────────────────────────────┤
│  Step 2: 训练 Stage 2 模型                               │
│  bash changing_resolution/scripts/train/                │
│       tmux_run_clean_480p720p_stage2_lmdb_training.sh   │
│  输出: outputs/.../latest.pt                            │
├─────────────────────────────────────────────────────────┤
│  Step 3: 算子对比评估（latent + RGB 指标）                │
│  bash changing_resolution/scripts/eval/                 │
│       tmux_run_clean_480p720p_stage2_operator_compare_  │
│       multigpu.sh                                       │
│  输出: outputs/.../tables/summary_val.md                │
├─────────────────────────────────────────────────────────┤
│  Step 4: 生成链路 A/B 对比（完整推理质量）                │
│  bash changing_resolution/scripts/eval/                 │
│       tmux_run_clean_480p720p_stage2_chain_ab_compare_  │
│       multigpu.sh                                       │
│  输出: outputs/.../compare/ (并排对比视频)               │
├─────────────────────────────────────────────────────────┤
│  Optional: 切换步长扫描                                  │
│  bash changing_resolution/scripts/eval/                 │
│       tmux_run_clean_480p720p_stage2_change_step_sweep_ │
│       multigpu.sh                                       │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 十、归档说明

| 归档路径 | 内容 | 归档原因 |
|----------|------|----------|
| [`.archive/stage1/`](../.archive/stage1/) | Stage 1 残差 clean-resizer 模型 | 被 Stage 2 LTX2 风格替代 |
| [`.archive/v1/`](../.archive/v1/) | 早期 noisy-to-clean V1 脚本 | 架构已重构 |

归档代码保留在仓库中用于追溯，不会被当前主线导入或使用。

---

## 🔧 十一、环境与依赖

### 必需依赖

```text
torch, lmdb, pyyaml, tqdm, numpy, einops
```

### 路径配置

机器相关路径通过 [`configs/local_paths.sh`](../configs/local_paths.sh) 管理：

```bash
# 覆盖默认路径
PATH_CONFIG=/path/to/custom_local_paths.sh \
bash changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh check
```

### 关键环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LIGHTX2V_REPO` | LightX2V 仓库路径 | `/mnt/afs_2/houze/LightX2V` |
| `CR_LMDB_DIR` | LMDB 数据目录 | `data/changing_resolution/lmdb_480p720p_1k` |
| `CR_STAGE2_CONFIG` | 训练配置文件 | `changing_resolution/configs/...yaml` |
| `CR_STAGE2_OUT_DIR` | 训练输出目录 | `outputs/...stage2_lmdb` |
| `CUDA_VISIBLE_DEVICES` | GPU 设备 | `0` |

---

## 📐 十二、架构演进历史

```text
┌──────────────────────────────────────────────────────┐
│  V1 (已归档)                                          │
│  · noisy-to-clean latent upsampler                   │
│  · sigma-conditioned 3D CNN                          │
│  · 输入: x_t_lr + sigma → 输出: z0_hr                │
│  · 问题: 训练分布与推理中间态不完全一致               │
├──────────────────────────────────────────────────────┤
│  Stage 1 (已归档)                                     │
│  · clean-to-clean residual resizer                   │
│  · 输入: z0_lr → 输出: z0_hr                         │
│  · 问题: 固定 trilinear resize 点，无学习能力         │
├──────────────────────────────────────────────────────┤
│  Stage 2 (当前主线) ⭐                                 │
│  · LTX2 风格 clean latent resizer                    │
│  · 输入: z0_lr / x0_pred_lr → 输出: z0_hr            │
│  · 有理数重采样: PixelShuffle ×3 + BlurDownsample /2 │
│  · 目标: 替代 LightX2V 的固定插值切换                 │
└──────────────────────────────────────────────────────┘
```

---

## 📚 十三、关键文件索引

| 类别 | 文件 | 说明 |
|------|------|------|
| **模型** | [`wan_sr/models/stage2_resizer.py`](../wan_sr/models/stage2_resizer.py) | `WanCleanLatentResizerStage2` 完整实现 |
| **模型** | [`wan_sr/models/blocks.py`](../wan_sr/models/blocks.py) | `SigmaConditionedResBlock3D` 等基础模块 |
| **模型** | [`wan_sr/models/sigma_embedding.py`](../wan_sr/models/sigma_embedding.py) | Fourier 特征 + AdaGN 条件注入 |
| **数据** | [`wan_sr/data/clean_latent_lmdb_dataset.py`](../wan_sr/data/clean_latent_lmdb_dataset.py) | LMDB 分片数据集 |
| **损失** | [`wan_sr/losses/clean_latent_losses.py`](../wan_sr/losses/clean_latent_losses.py) | 四合一损失函数 |
| **训练** | [`changing_resolution/scripts/train/train_clean_latent_resizer_stage2.py`](../changing_resolution/scripts/train/train_clean_latent_resizer_stage2.py) | 训练主脚本 |
| **配置** | [`changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml`](../changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml) | 训练超参配置 |
| **推理** | [`wan_sr/pipelines/transition.py`](../wan_sr/pipelines/transition.py) | LR→HR 推理桥接 |
| **评估** | [`changing_resolution/scripts/eval/eval_clean_resizer_operator_compare.py`](../changing_resolution/scripts/eval/eval_clean_resizer_operator_compare.py) | 算子对比评估 |
| **运行手册** | [`changing_resolution/STAGE2_RUNBOOK.md`](../changing_resolution/STAGE2_RUNBOOK.md) | 日常操作命令速查 |
| **设计文档** | [`codex.md`](../codex.md) | 原始项目设计思路 |
| **进度** | [`PROGRESS.md`](../PROGRESS.md) | 项目进度追踪 |

---

> 💡 **快速开始**：直接阅读 [`changing_resolution/STAGE2_RUNBOOK.md`](../changing_resolution/STAGE2_RUNBOOK.md) 获取可执行的命令清单。