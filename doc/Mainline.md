# wanUpsampler 项目主线

> **最后更新**：2026-05-20
> **当前阶段**：Stage 3 - x0-pred Clean Latent Resizer
> **核心任务**：在 Wan / LightX2V 采样中途完成 480p -> 720p 的 latent 分辨率切换
> **主线判断**：Stage 2 解决“学习型 1.5x clean latent resize”，Stage 3 解决“真实推理桥接输入分布”

---

## 一、项目定位

`wanUpsampler` 的目标是训练一个 **在 Wan 推理中途使用的 1.5x latent 空间放大器**，替代 LightX2V 中原有的固定三线性插值分辨率切换方案。

主问题不是单纯把 480p latent 放大到 720p，而是在采样链路中保持语义、结构、时间一致性，并让高分辨率后半程采样能稳定继续。

### 当前推理链路

```text
低分辨率 Wan 前半程采样（480p noisy latent）
        ↓
在 handoff step 做一次 Wan denoiser forward
        ↓
x0_pred_lr = x_t - sigma_t * noise_pred
        ↓
WanCleanLatentResizerStage2(x0_pred_lr)
        ↓
z0_hr_hat（720p clean latent estimate）
        ↓
re-noise 到同一 timestep
        ↓
高分辨率 Wan 后半程采样（720p）
        ↓
VAE decode -> 720p 视频
```

### 阶段关系

| 阶段 | 输入域 | 目标 | 状态 |
|------|--------|------|------|
| V1 | noisy latent + sigma | 直接预测 HR clean latent | 已归档 |
| Stage 1 | clean `z0_lr` | residual clean-to-clean resize | 已归档 |
| Stage 2 | clean `z0_lr` | 学习型 LTX2 风格 1.5x resize | 主要章节 / 架构基线 |
| Stage 3 | one-step `x0_pred_lr` | 对齐真实桥接分布的 1.5x resize | 当前主线 |

---

## 二、仓库结构

```text
wanUpsampler/
├── wan_sr/
│   ├── models/
│   │   ├── stage2_resizer.py              # Stage 2/3 共用模型类
│   │   └── factory.py                     # build_clean_latent_resizer()
│   ├── data/
│   │   ├── clean_latent_lmdb_dataset.py   # Stage 2 clean LMDB
│   │   └── x0pred_latent_lmdb_dataset.py  # Stage 3 x0_pred LMDB
│   ├── losses/
│   │   └── clean_latent_losses.py         # CleanLatentResizeLoss
│   ├── schedulers/
│   │   └── noise_utils.py                 # flow-style add_noise / downsample
│   └── training/
│       ├── checkpoint.py
│       ├── config.py
│       └── ema.py
│
├── changing_resolution/
│   ├── configs/
│   │   ├── train_clean_480p_to_720p_lmdb_stage2.yaml
│   │   ├── train_x0pred_480p_to_720p_lmdb_stage3.yaml
│   │   └── wan_t2v_stage3_x0pred_480p.json
│   ├── scripts/
│   │   ├── data/                          # Stage 3 x0_pred LMDB 构建
│   │   ├── train/                         # Stage 2 / Stage 3 训练入口
│   │   ├── eval/                          # operator / chain / sweep 评估
│   │   └── bridge/                        # LightX2V 桥接运行入口
│   ├── lightx2v_clean_bridge.py           # Wan / LightX2V bridge 注册
│   ├── STAGE2_RUNBOOK.md
│   ├── STAGE2_MODEL_PLAN.md
│   └── STAGE3_RUNBOOK.md
│
├── configs/
│   └── local_paths.sh                     # 机器相关路径
├── experiments/                           # 探索性脚本
├── .archive/
│   ├── stage1/
│   └── v1/
├── codex.md
├── PROGRESS.md
└── requirements.txt
```

---

## 三、Stage 2：Clean-to-Clean 学习型 Resizer

Stage 2 是当前体系的模型与训练架构基线。它把输入从早期 noisy-to-clean 任务收敛到 clean latent resize：

```text
Stage 2: z0_lr -> z0_hr
```

### 3.1 目标

Stage 2 要证明：固定三线性插值不是唯一选择，1.5x latent resize 可以由模型学习，并在 latent/RGB 指标上超过或接近固定插值。

它的训练输入是 VAE 编码得到的干净低分辨率 latent `z0_lr`，目标是对应的高分辨率 clean latent `z0_hr`。

### 3.2 模型契约

| 项目 | 规格 |
|------|------|
| 模型类 | `WanCleanLatentResizerStage2` |
| 输入 | `z0_lr`，shape `[B, 16, T, 60, 104]` |
| 输出 | `z0_hr_hat`，shape `[B, 16, T, 90, 156]` |
| 缩放倍率 | 1.5x 空间，时间维度不变 |
| 训练数据 | `CleanLatentLMDBDataset` |
| 默认配置 | `changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml` |

### 3.3 架构

```text
z0_lr [B, 16, T, H, W]
        ↓
Conv3d stem (16 -> hidden)
        ↓
ResBlock x N/2
        ↓
SpatialRationalResampler 3/2
  ├─ Conv3d channel expansion
  ├─ spatial PixelShuffle x3
  └─ BlurDownsample /2
        ↓
ResBlock x N/2
        ↓
Conv3d output (hidden -> 16)
        ↓
z0_hr_hat [B, 16, T, 1.5H, 1.5W]
```

关键设计：

| 设计 | 说明 |
|------|------|
| LTX2 风格 ResBlock | Conv3d + GroupNorm + SiLU + residual |
| 有理数重采样 3/2 | PixelShuffle x3 后 BlurDownsample /2 |
| 无 sigma 条件 | Stage 2 输入是 clean latent，不需要噪声水平条件 |
| `residual_skip` 可选 | 可用 trilinear 上采样作为 skip，再学习残差 |

### 3.4 损失函数

Stage 2 使用 `CleanLatentResizeLoss`：

```text
L_total =
  1.0 * L_latent_charbonnier
+ 0.2 * L_low_frequency_consistency
+ 0.1 * L_temporal_difference
+ 0.0 * L_residual_regularization
```

| 项 | 作用 |
|----|------|
| Latent Charbonnier | 约束预测 HR latent 接近 GT |
| Low-Freq Consistency | 预测下采样后应回到 `z0_lr` |
| Temporal Difference | 约束帧间变化一致 |
| Residual Regularization | 控制相对插值基线的残差幅度，默认关闭 |

### 3.5 训练入口

```bash
# preflight
bash changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh check

# tmux 单卡训练
CUDA_VISIBLE_DEVICES=0 \
bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage2_lmdb_training.sh

# 覆盖超参
MAX_STEPS=100000 LR=5e-5 GRAD_ACCUM=16 \
bash changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh train
```

默认输出：

```text
outputs/changing_resolution_clean_480p720p_stage2_lmdb/
├── train_config.yaml
├── metrics.jsonl
├── latest.pt
├── best_val.pt
├── best_val.json
└── step_*.pt
```

### 3.6 Stage 2 评估

Stage 2 的主要评估分三层：

| 评估 | 入口 | 目的 |
|------|------|------|
| Operator Compare | `tmux_run_clean_480p720p_stage2_operator_compare_multigpu.sh` | latent/RGB 指标对比 trilinear vs learned |
| Chain A/B Compare | `tmux_run_clean_480p720p_stage2_chain_ab_compare_multigpu.sh` | 完整 LightX2V 链路对比 |
| Change-Step Sweep | `tmux_run_clean_480p720p_stage2_change_step_sweep_multigpu.sh` | 扫描 handoff step |

Stage 2 change-step sweep 的三列面板固定为：

```text
stop480 | interp720 | stage2_720
```

### 3.7 Stage 2 的局限

Stage 2 的核心局限是训练输入域和真实推理输入域不一致：

```text
训练: z0_lr
推理: x0_pred_lr = x_t - sigma_t * noise_pred
```

`x0_pred_lr` 是一次 Wan denoiser forward 后的 clean estimate，仍可能带有残留噪声、结构偏差和 timestep 相关伪影。Stage 2 只在纯 clean latent 上训练，因此链路表现不能只靠 clean-to-clean 指标判断。

---

## 四、Stage 3：当前主线 x0-pred Resizer

Stage 3 保留 Stage 2 的模型架构，但更换训练输入域：

```text
Stage 2: z0_lr      -> z0_hr
Stage 3: x0_pred_lr -> z0_hr
```

这一步是当前项目主线，因为它直接对齐 LightX2V 桥接时真正喂给 resizer 的输入。

### 4.1 目标

Stage 3 要解决的问题：

| 问题 | Stage 3 的处理 |
|------|----------------|
| 推理输入不是纯 clean latent | 用 one-step denoise 得到的 `x0_pred_lr` 作为训练输入 |
| Stage 2 对残留噪声/伪影鲁棒性不足 | 让模型在训练时直接见到 bridge handoff 分布 |
| clean 指标无法代表链路质量 | 评估重点转向 continuation-aware sweep |

### 4.2 Stage 3 数据 recipe

默认 Stage 3 数据构建方式：

```text
clean 480p latent z0_lr
        ↓
在 50-step Wan schedule 的 step 35 加 flow noise
        ↓
运行一次 Wan denoiser forward
        ↓
x0_pred_lr = x_t - sigma_t * noise_pred
        ↓
写入 Stage 3 LMDB
```

训练时：

| 字段 | 用途 |
|------|------|
| `x0_pred_lr` | 模型输入 |
| `z0_hr` | HR 监督目标 |
| `z0_lr` | low-frequency consistency 参考 |
| `prompt` / `meta` | 追溯样本和生成 recipe |

### 4.3 Stage 3 LMDB 构建

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash changing_resolution/scripts/data/tmux_build_x0pred_480p720p_stage3_lmdb.sh
```

多卡：

```bash
TOTAL_SAMPLES=1000 GPU_IDS=0,1,2,3 OVERWRITE=1 \
bash changing_resolution/scripts/data/tmux_build_x0pred_480p720p_stage3_lmdb_multigpu.sh
```

默认路径：

```text
source: data/changing_resolution/lmdb_480p720p_1k
output: data/changing_resolution/lmdb_x0pred_480p720p_stage3_step35
infer_steps: 50
denoise_step: 35
config: changing_resolution/configs/wan_t2v_stage3_x0pred_480p.json
```

快速 schema smoke test：

```bash
MODE=clean_copy MAX_SAMPLES=2 OVERWRITE=1 \
bash changing_resolution/scripts/data/tmux_build_x0pred_480p720p_stage3_lmdb.sh
```

`clean_copy` 只用于检查 LMDB 写入链路，不能作为正式 Stage 3 训练集。

### 4.4 Stage 3 模型与配置

Stage 3 仍使用 `WanCleanLatentResizerStage2`：

| 项目 | Stage 3 默认值 |
|------|----------------|
| 配置 | `changing_resolution/configs/train_x0pred_480p_to_720p_lmdb_stage3.yaml` |
| 数据集 | `X0PredLatentLMDBDataset` |
| 输出目录 | `outputs/changing_resolution_x0pred_480p720p_stage3_lmdb` |
| `hidden_channels` | 256 |
| `num_res_blocks` | 8 |
| `scale_factor` | 1.5 |
| `residual_skip` | false |
| `precision` | bf16 |
| `max_steps` | 50000 |
| `grad_accum` | 8 |
| `eval_use_ema` | true |

虽然类名仍是 Stage2，这是有意保持：Stage3 的 checkpoint schema 和 bridge 加载路径复用 Stage2 架构，主要差异在数据分布。

### 4.5 Stage 3 训练入口

```bash
# preflight
bash changing_resolution/scripts/train/run_x0pred_480p720p_stage3_lmdb_training.sh check

# tmux 单卡训练
CUDA_VISIBLE_DEVICES=0 \
bash changing_resolution/scripts/train/tmux_run_x0pred_480p720p_stage3_lmdb_training.sh

# 覆盖超参
MAX_STEPS=50000 LR=1e-4 GRAD_ACCUM=8 \
bash changing_resolution/scripts/train/run_x0pred_480p720p_stage3_lmdb_training.sh train
```

默认输出：

```text
outputs/changing_resolution_x0pred_480p720p_stage3_lmdb/
├── train_config.yaml
├── metrics.jsonl
├── latest.pt
├── best_val.pt
├── best_val.json
└── step_*.pt
```

### 4.6 Stage 3 评估主线

Stage 3 的评估重点是完整链路中的 handoff 质量，而不是只看 clean latent operator 指标。

单卡 smoke test：

```bash
bash changing_resolution/scripts/eval/run_x0pred_480p720p_stage3_change_step_sweep.sh
```

指定 checkpoint 和切换步：

```bash
CR_STAGE3_CHANGE_STEP_SWEEP_CKPT=outputs/changing_resolution_x0pred_480p720p_stage3_lmdb/step_050000.pt \
CHANGE_STEPS=35 LIMIT=2 \
bash changing_resolution/scripts/eval/run_x0pred_480p720p_stage3_change_step_sweep.sh
```

四卡批量 sweep：

```bash
bash changing_resolution/scripts/eval/tmux_run_x0pred_480p720p_stage3_change_step_sweep_multigpu.sh
```

默认 sweep：

```text
GPU_IDS: 0,1,2,3
TOTAL_PROMPTS: 4
STEP_START / STEP_END / STEP_STRIDE: 10 / 50 / 1
```

默认会生成 164 个三列面板，因为 10..50 inclusive 是 41 个 handoff step，4 个 prompt 对应 164 个 panel。若需要 200 个 panel，用：

```bash
STEP_START=1 STEP_END=50 STEP_STRIDE=1
```

Stage 3 sweep 的三列面板固定为：

```text
stop480 at step N | interp720 step N -> 50 | stage3 720 step N -> 50
```

### 4.7 Stage 3 成功标准

Stage 3 是否成立，应优先看 continuation-aware 结果：

| 维度 | 判断 |
|------|------|
| 主体结构 | stage3 分支不应比 interp720 更容易形变或丢主体 |
| 细节质量 | 纹理、文字、边缘不应出现明显 x0_pred 伪影放大 |
| 时间稳定性 | 不应引入明显闪烁或局部跳动 |
| 切换步鲁棒性 | step 30/35/40 等关键点应稳定 |
| 对比基线 | 至少要和 interp720 可比，最好在结构或细节上胜出 |

---

## 五、LightX2V 桥接

桥接代码位于 `changing_resolution/lightx2v_clean_bridge.py`，当前有三类关键 runner：

| runner | 用途 |
|--------|------|
| `wan2.1_partial_denoise_decode` | 低分辨率采样到 handoff step 后直接 decode，得到 `stop480` |
| `wan2.1_clean_interp_bridge` | handoff 后用固定插值切到 720p，得到 `interp720` |
| `wan2.1_clean_resizer_bridge` | handoff 后用 learned resizer 切到 720p，得到 Stage2/Stage3 分支 |

Stage 3 在 bridge config 里仍使用：

```json
"wan_clean_resizer_model_class": "stage2"
```

这是因为模型类和 checkpoint schema 仍是 Stage2 架构，Stage3 的语义由 checkpoint、训练配置和数据域决定。

---

## 六、完整工作流

```text
1. 构建 Stage 2 clean LMDB
   data/changing_resolution/lmdb_480p720p_1k

2. 训练 Stage 2 clean-to-clean resizer
   outputs/changing_resolution_clean_480p720p_stage2_lmdb/latest.pt

3. 用 Stage 2 观察真实 bridge 输入域问题
   x0_pred_lr vs z0_lr

4. 构建 Stage 3 x0_pred LMDB
   data/changing_resolution/lmdb_x0pred_480p720p_stage3_step35

5. 训练 Stage 3 x0-pred resizer
   outputs/changing_resolution_x0pred_480p720p_stage3_lmdb/latest.pt

6. 运行 Stage 3 change-step sweep
   stop480 | interp720 | stage3_720

7. 根据 continuation-aware panel 判断可用切换步和 checkpoint
```

---

## 七、环境与路径

机器相关路径集中在：

```text
configs/local_paths.sh
```

常用环境变量：

| 变量 | 说明 |
|------|------|
| `LIGHTX2V_REPO` | LightX2V 仓库路径 |
| `MODEL_ROOT` | Wan2.1 模型路径 |
| `CR_STAGE2_CONFIG` | Stage 2 训练配置 |
| `CR_STAGE2_OUT_DIR` | Stage 2 输出目录 |
| `CR_STAGE3_LMDB_DIR` | Stage 3 x0_pred LMDB |
| `CR_STAGE3_CONFIG` | Stage 3 训练配置 |
| `CR_STAGE3_OUT_DIR` | Stage 3 输出目录 |
| `CR_STAGE3_CHANGE_STEP_SWEEP_CKPT` | Stage 3 sweep 使用的 checkpoint |

---

## 八、归档说明

| 路径 | 内容 | 状态 |
|------|------|------|
| `.archive/v1/` | noisy-to-clean V1 脚本 | 已归档，不是当前主线 |
| `.archive/stage1/` | Stage 1 residual clean-resizer | 已归档，不再由 factory 加载 |

`wan_sr/models/factory.py` 当前只允许主线 Stage2/Stage3 架构加载。Stage1 如需追溯，应从 `.archive/stage1/` 查看，不应重新混入主线。

---

## 九、关键文件索引

| 类别 | 文件 | 说明 |
|------|------|------|
| 模型 | `wan_sr/models/stage2_resizer.py` | Stage2/Stage3 共用模型实现 |
| 模型工厂 | `wan_sr/models/factory.py` | checkpoint/config 到模型实例的构建入口 |
| Stage 2 数据 | `wan_sr/data/clean_latent_lmdb_dataset.py` | clean `z0_lr -> z0_hr` LMDB |
| Stage 3 数据 | `wan_sr/data/x0pred_latent_lmdb_dataset.py` | `x0_pred_lr, z0_lr, z0_hr` LMDB |
| 损失 | `wan_sr/losses/clean_latent_losses.py` | clean latent resize 损失 |
| Stage 2 训练 | `changing_resolution/scripts/train/train_clean_latent_resizer_stage2.py` | Stage 2 trainer |
| Stage 3 训练 | `changing_resolution/scripts/train/train_x0pred_latent_resizer_stage3.py` | Stage 3 trainer |
| Stage 3 数据构建 | `changing_resolution/scripts/data/build_x0pred_480p720p_stage3_lmdb.py` | 生成 one-step x0_pred LMDB |
| 桥接 | `changing_resolution/lightx2v_clean_bridge.py` | LightX2V runner 注册和分辨率切换 |
| Stage 2 手册 | `doc/STAGE2_RUNBOOK.md` | Stage 2 命令速查 |
| Stage 3 手册 | `doc/STAGE3_RUNBOOK.md` | Stage 3 命令速查 |

---

## 十、当前阅读顺序

如果要快速理解当前主线，建议按下面顺序读：

```text
1. doc/Mainline.md
2. doc/STAGE3_RUNBOOK.md
3. changing_resolution/configs/train_x0pred_480p_to_720p_lmdb_stage3.yaml
4. wan_sr/data/x0pred_latent_lmdb_dataset.py
5. changing_resolution/scripts/train/train_x0pred_latent_resizer_stage3.py
6. changing_resolution/scripts/eval/run_x0pred_480p720p_stage3_change_step_sweep.sh
7. changing_resolution/lightx2v_clean_bridge.py
```
