# changing_resolution 训练方法整理

本文档按“当前主线优先，历史路径保留”的方式整理
`changing_resolution` 下的训练方法。当前目标是训练一个 clean latent resizer，
替换 LightX2V `changing_resolution` 推理链路里的固定插值算子。

## 1. 核心训练问题

当前任务不是 RGB 超分，也不是 noisy latent 超分，而是 clean latent resize：

```text
z0_lr 或 x0_pred_lr -> z0_hr
```

在当前 480p -> 720p 设置中：

```text
RGB:    480 x 832  -> 720 x 1248
latent:  60 x 104  ->  90 x 156
scale:  1.5x spatial
```

LightX2V 在 changing-resolution 中会先得到 clean latent 估计：

```text
x0_pred = x_t - sigma * eps
```

然后对 clean latent 做 resize，再重新加噪继续 diffusion。因此训练目标应和推理替换点一致：
学习 clean latent 的空间 resize 算子。

## 2. 当前推荐主线：1k LMDB + Stage 1 Residual Resizer

这是当前应优先使用的训练路径。

模型结构图：

```text
changing_resolution/diagrams/wan_clean_latent_resizer.svg
changing_resolution/diagrams/wan_clean_latent_resizer.mmd
```

LTX2 upsampler 参考结构图：

```text
changing_resolution/diagrams/ltx2_latent_upsampler.svg
changing_resolution/diagrams/ltx2_latent_upsampler.mmd
```

### 2.1 数据生成与 LMDB 构建

目标数据：

```text
data/changing_resolution/lmdb_480p720p_1k
```

每条样本包含：

```text
z0_lr, z0_hr, prompt, meta
```

构造逻辑：

```text
source video: Wan2.1 生成的 720p 视频
z0_hr       : Wan VAE encode(video_720p)
z0_lr       : Wan VAE encode(resize(video_720p, 480 x 832))
```

如果原始 720p 视频已经存在，只构建 LMDB：

```bash
TOTAL_SAMPLES=1000 GPU_IDS=0,1,2,3 \
bash changing_resolution/scripts/data/build_clean_480p720p_lmdb_1k_multigpu.sh lmdb
```

如果要从 prompt 生成视频并构建 LMDB，使用 tmux 多卡入口：

```bash
TOTAL_SAMPLES=1000 GPU_IDS=0,1,2,3 \
bash changing_resolution/scripts/data/tmux_build_clean_lmdb_480p720p_1k_multigpu.sh

tmux attach -t wan_cr_lmdb_480p720p_1k_multigpu
```

相关入口：

```text
scripts/data/generate_wan21_720p_dataset.sh
  生成 Wan2.1 720p 源视频。

scripts/data/build_480p720p_lmdb.py
  将 720p 视频编码成 480p/720p clean latent pair，并写入 sharded LMDB。

scripts/data/build_clean_480p720p_lmdb_1k_multigpu.sh
  多卡数据构建入口，按 PROMPT_OFFSET 拆分 prompt 范围。

scripts/data/tmux_build_clean_lmdb_480p720p_1k_multigpu.sh
  长任务 tmux 包装入口，适合 SSH 断开也要继续跑的场景。
```

### 2.2 Stage 1 模型

Stage 1 不改大结构，使用当前 `WanCleanLatentResizer`：

```text
z0_lr
  -> Conv3D 16 -> 256
  -> ResBlock3D x4
  -> trilinear feature resize to target H/W
  -> ResBlock3D x4
  -> Conv3D 256 -> 16 residual
  -> trilinear(z0_lr) + residual
  -> pred_z0_hr
```

对应代码：

```text
wan_sr/models/clean_resizer.py
```

默认模型配置：

```text
hidden_channels: 256
num_res_blocks: 8
scale_factor: 1.5
residual_skip: true
```

这一路径的意义是先确认“围绕 trilinear 的残差修正”是否能稳定超过固定插值。
如果 Stage 1 都不能在 operator compare 或真实链路 A/B 中体现收益，后续更复杂模块的依据不足。

### 2.3 Stage 1 训练配置

配置文件：

```text
changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage1.yaml
```

默认训练设置：

```text
max_steps:       10000
batch_size:      1
grad_accum:      8
effective batch: 8
precision:       bf16
lr:              1e-4
weight_decay:    0.01
train/val split: 95% / 5%
eval_every:      1000
save_every:      1000
best model:      lowest EMA validation loss
```

训练前检查：

```bash
bash changing_resolution/scripts/train/run_clean_480p720p_stage1_lmdb_training.sh check
```

单卡 tmux 训练：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage1_lmdb_training.sh

tmux attach -t wan_cr_stage1_lmdb_train
```

输出目录：

```text
outputs/changing_resolution_clean_480p720p_stage1_lmdb
```

关键输出：

```text
latest.pt
best_val.pt
step_*.pt
metrics.jsonl
train_config.yaml
```

如果 10k checkpoint 的评测结果明显优于插值，可以在同一 Stage 1 配置上延长：

```bash
MAX_STEPS=20000 \
RESUME=outputs/changing_resolution_clean_480p720p_stage1_lmdb/latest.pt \
bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage1_lmdb_training.sh
```

### 2.4 Loss 组成

训练 loss 在 `wan_sr/losses/clean_latent_losses.py`：

```text
latent_loss:
  pred_z0_hr 与 z0_hr 的 Charbonnier loss。

low_freq_loss:
  将 pred_z0_hr 下采样回 z0_lr 尺寸后，与 z0_lr 做 L1。
  作用是约束低频内容仍能对齐 LR source。

temporal_loss:
  对 pred_z0_hr 和 z0_hr 的时间差分做约束。
  作用是减少时间方向的不稳定。

residual_loss:
  可选项，默认权重为 0。
```

默认权重：

```text
latent_weight:   1.0
low_freq_weight: 0.2
temporal_weight: 0.1
residual_weight: 0.0
```

## 3. 评测方法

训练是否有效不能只看 train loss，至少需要两类评测。

### 3.1 Operator Compare：有参考目标

该评测使用 LMDB validation split，因此有 `ori720_decode` 作为真实参考。

比较对象：

```text
lr480_decode
ori720_decode
interp720_decode
trained720_decode
```

指标：

```text
PSNR, SSIM, LPIPS
```

运行：

```bash
TOTAL_SAMPLES=32 GPU_IDS=0,1,2,3 \
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_operator_compare_multigpu.sh

tmux attach -t wan_cr_operator_compare
```

表格汇总：

```bash
python changing_resolution/scripts/eval/summarize_operator_compare_table.py \
  --input outputs/changing_resolution_operator_compare_stage1 \
  --split val
```

判断标准：

```text
trained_psnr  > interp_psnr
trained_ssim  > interp_ssim
trained_lpips < interp_lpips
```

### 3.2 Generation-chain A/B：无参考目标

该评测把模型插入真实 LightX2V `changing_resolution` 链路中。
这里没有 ground truth，因为 native 720p 与 changing-resolution 不是同一条 diffusion 轨迹。

比较对象只有：

```text
interp720
trained720
```

运行：

```bash
TOTAL_SAMPLES=16 GPU_IDS=0,1,2,3 \
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_chain_ab_compare_multigpu.sh

tmux attach -t wan_cr_chain_ab_compare
```

人工判断重点：

```text
sharpness
temporal stability
less flicker
less texture crawling
less subject deformation
```

## 4. 历史训练路径：safetensors 文件数据

这是早期路径，当前不作为主线，但保留用于兼容和回溯。

数据目录：

```text
data/changing_resolution/latent_pairs_480p720p
```

配置文件：

```text
changing_resolution/configs/train_clean_480p_to_720p.yaml
```

训练入口：

```bash
bash changing_resolution/scripts/legacy/run_clean_480p720p_training.sh train
```

或者一条链路执行生成、构建、训练：

```bash
bash changing_resolution/scripts/legacy/run_clean_480p720p_training.sh all
```

相关脚本：

```text
scripts/data/build_480p720p_latents.py
  生成每样本一个目录的 safetensors latent pair。

scripts/legacy/run_clean_480p720p_training.sh
  历史 all-in-one 入口。

scripts/legacy/tmux_run_clean_480p720p_all.sh
  历史 tmux 包装入口。
```

不推荐继续把新实验建立在这个路径上，原因是 1k LMDB 路径更适合多卡构建、
validation split、后续评测和长任务恢复。

## 5. 后续阶段建议

### Stage 2：Learned Feature Upsampler

触发条件：

```text
Stage 1 在 operator compare 或真实 chain A/B 中超过固定插值。
```

建议改动：

```text
保留 residual skip，替换当前固定 feature resize block。
```

候选结构：

```text
LR feature blocks
  -> trilinear feature resize
  -> gated Conv3D residual refinement
  -> HR feature blocks
  -> output residual
```

原因是它比纯残差修正更强，但仍适合 1k 样本和固定 1.5x 比例。
在 Stage 1 证据不足前，不建议直接上大 attention 或 U-Net。

### Stage 3：Conditioning 与泛化

只在 Stage 2 稳定后考虑：

```text
prompt/text conditioning
sigma or step conditioning
multi-resolution training
larger datasets
DDP training
```

这些不是当前 Stage 1 的必要条件。

## 6. 当前推荐执行顺序

```text
1. 构建或确认 data/changing_resolution/lmdb_480p720p_1k。
2. 运行 Stage 1 preflight。
3. 训练 10k steps baseline。
4. 跑 operator compare，先确认有参考指标是否超过插值。
5. 跑 generation-chain A/B，确认真实 LightX2V 链路中是否视觉收益。
6. 如果 10k 有收益，resume 到 20k。
7. 如果 20k 仍有收益，再进入 Stage 2 结构改造。
```
