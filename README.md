# WanTrajectoryUpsampler 项目书

## 1. 项目目标

WanTrajectoryUpsampler 不是普通视频超分项目，而是一个用于 Wan 视频生成采样中途的 latent 分辨率切换模块。

目标是在低分辨率 Wan 已经采样到中间步骤时，接收当前的低分辨率 noisy latent 和噪声水平 sigma，预测高分辨率 clean latent，再重新加噪到同一个采样步，交给高分辨率 Wan 继续生成。

核心链路：

```text
低分辨率 Wan 采样前半程
  -> 得到中间态 x_t_lr
  -> Upsampler(x_t_lr, sigma) 预测 z0_hr
  -> 重新加噪得到 x_t_hr
  -> 高分辨率 Wan 继续采样
  -> VAE 解码成高分辨率视频
```

## 2. 第一版范围

第一版只做最小可验证版本：

- 适配 Wan2.1 VAE。
- 只做 2x 空间 latent 放大，不做时间超分。
- 输入 low-res noisy latent 和 sigma。
- 输出 high-res clean latent。
- 先实现训练、评估和 transition 函数。
- 暂不训练 Wan 主模型，暂不接 text prompt，暂不做 cross-attention。

推荐实验规格：

```text
视频帧数: 17
HR 分辨率: 512 x 512
LR 分辨率: 256 x 256
latent 通道: 16
scale: 2
```

## 3. 数据流程

每个训练样本来自同一段高分辨率视频 clip。

```text
HR clip
  -> 随机退化和下采样，得到 LR clip
  -> Wan2.1 VAE 编码
  -> 得到 z0_hr 和 z0_lr
  -> 训练时在线采样 sigma
  -> 给 z0_lr 加噪得到 x_t_lr
  -> 训练模型预测 z0_hr
```

样本保存结构：

```text
data/latent_pairs_wan21_512/
  000000/
    z0_lr.safetensors
    z0_hr.safetensors
    meta.json
```

`meta.json` 记录 VAE 版本、帧数、分辨率、latent shape 和退化参数，避免后续混用不同 VAE 或不同数据规格。

## 4. 模型设计

模型名：

```text
WanNoisyLatentUpsampler
```

输入：

```text
x_t_lr: [B, 16, T, H, W]
sigma:  [B]
```

输出：

```text
pred_z0_hr: [B, 16, T, 2H, 2W]
```

第一版使用轻量 3D CNN：

```text
3D Conv stem
  -> SigmaConditionedResBlock3D
  -> Spatial PixelShuffle 2x
  -> SigmaConditionedResBlock3D
  -> 3D Conv output
```

sigma 通过 Fourier embedding 和 MLP 注入每个 ResBlock，用 FiLM 或 AdaGN 调制特征。

## 5. 训练目标

第一版训练 noisy-to-clean：

```text
U(x_t_lr, sigma) -> z0_hr
```

不要直接训练 noisy-to-noisy，因为高分辨率随机噪声本身不可恢复，训练会更不稳定。

加噪方式先用简化 flow 形式：

```python
eps = torch.randn_like(z0_lr)
x_t_lr = (1.0 - sigma) * z0_lr + sigma * eps
```

sigma 主要采样中段，因为目标是在 Wan 采样中途切换分辨率：

```text
70%: Uniform(0.35, 0.70)
20%: Uniform(0.20, 0.85)
10%: Uniform(0.00, 0.20)
```

## 6. Loss

总 loss：

```text
loss = latent_loss + 0.2 * low_freq_loss + 0.1 * temporal_loss
```

其中：

- `latent_loss`: Charbonnier(pred_z0_hr, z0_hr)
- `low_freq_loss`: L1(downsample(pred_z0_hr), z0_lr)
- `temporal_loss`: L1(pred temporal diff, gt temporal diff)

第一版不加 VAE decode RGB loss，等 latent 训练跑通后再加入。

## 7. 训练阶段

### Stage A: clean warm-up

先训练 clean upscaler：

```text
U(z0_lr, sigma=0) -> z0_hr
```

占总步数 5% 到 10%，用于让模型先学会 latent 空间的 2x 映射。

### Stage B: noisy-to-clean 主训练

正式训练：

```text
U(x_t_lr, sigma) -> z0_hr
```

占总步数 80% 到 90%。

### Stage C: trajectory fine-tuning

后续再做。收集真实 Wan 低分采样中间 latent，让 upsampler 适配真实 sampler trajectory。

## 8. 项目结构

```text
WanTrajectoryUpsampler/
  README.md
  codex.md
  requirements.txt
  configs/
    train_wan21_x2_512.yaml
    infer_transition.yaml
  wan_sr/
    models/
      upsampler.py
      blocks.py
      sigma_embedding.py
    data/
      latent_pair_dataset.py
      degradation.py
      video_io.py
    vae/
      wan_vae_wrapper.py
    schedulers/
      sigma_sampler.py
      noise_utils.py
    losses/
      latent_losses.py
    pipelines/
      transition.py
      eval_decode.py
  scripts/
    build_latent_pairs.py
    train.py
    eval_latent.py
    eval_decode.py
    infer_transition_wan.py
```

## 9. 第一阶段交付物

第一阶段要完成这些文件和能力：

1. `scripts/build_latent_pairs.py`
   - 从原始视频生成 HR/LR latent pair。
   - 保存 `z0_lr.safetensors`、`z0_hr.safetensors` 和 `meta.json`。

2. `LatentPairDataset`
   - 读取 latent pair。
   - 在线采样 sigma。
   - 构造 `x_t_lr`。

3. `WanNoisyLatentUpsampler`
   - 3D CNN + sigma conditioning + spatial PixelShuffle 2x。

4. `scripts/train.py`
   - 支持 bf16、梯度累积、AdamW、EMA、断点保存和恢复。
   - 支持 clean warm-up。

5. `eval_latent.py`
   - 输出 latent reconstruction 指标。

6. `eval_decode.py`
   - 用 Wan VAE decode 预测结果、GT 和 baseline，导出 mp4。

7. `transition_lr_to_hr`
   - 推理中途完成 LR noisy latent 到 HR noisy latent 的转换。

## 10. 验证标准

第一阶段完成后，需要至少验证：

- clean upscaler loss 能下降。
- noisy-to-clean loss 能下降。
- `downsample(pred_z0_hr)` 和 `z0_lr` 保持低频一致。
- VAE decode 后视频主体一致、运动连续、闪烁可控。
- latent interpolate baseline、clean-only upsampler、noisy-to-clean upsampler 有可视化对比。

最终目标是在真实 Wan 推理中测试：

```text
Wan LR sample 25 steps
  -> transition_lr_to_hr
  -> Wan HR continue 25 steps
```

## 11. 主要风险

- forward noising 构造的 `x_t_lr` 不完全等于真实 Wan 中间采样态。
- re-noise 后的 HR latent 可能无法被高分 Wan 稳定接住。
- 模型可能只学到模糊平均。
- Wan2.1 和 Wan2.2 VAE latent 结构不同，不能混用。

规避方式：

- 第一版锁定 Wan2.1 VAE。
- 先在较晚 timestep 切换。
- 先跑通 forward-noise 版本，再做真实 trajectory fine-tuning。
- 所有样本写入清晰的 `meta.json`。

## 12. 一句话总结

本项目训练一个 sigma-conditioned noisy-to-clean latent upsampler：用 HR/LR 视频对编码得到 `z0_hr` 和 `z0_lr`，对 `z0_lr` 加噪得到 `x_t_lr`，训练模型从 `(x_t_lr, sigma)` 预测 `z0_hr`；推理时在 Wan 低分辨率采样中途调用该模型，把 latent 切到高分辨率后继续采样。
