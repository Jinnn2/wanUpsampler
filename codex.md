下面给你一版**可以直接交给 Codex 落地的训练思路**。核心目标不是普通视频超分，而是训练一个 **Wan 采样中途可用的 noisy latent upsampler**。

---

# 0. 项目定位

项目名可以先叫：

```text
WanTrajectoryUpsampler
```

目标：

> 训练一个在 Wan 推理中途使用的 2× latent upsampler。
> 输入低分辨率 noisy latent (x_t^{LR}) 和当前噪声水平 (\sigma_t)，输出高分辨率 clean latent 估计 (\hat{z}_0^{HR})，然后重新加噪到同一 timestep，交给高分辨率 Wan 继续 denoise。

最终推理链路：

```text
低分辨率 Wan 采样前半程
        ↓
得到中间态 x_t_LR
        ↓
WanTrajectoryUpsampler(x_t_LR, sigma_t)
        ↓
预测 z0_HR_hat
        ↓
re-noise 成 x_t_HR
        ↓
高分辨率 Wan 继续采样后半程
        ↓
VAE decode 得到高分辨率视频
```

Wan2.1 本身已经公开 480P/720P 模型，并且 Diffusers 示例使用 `AutoencoderKLWan`、`WanPipeline` 和 `UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True)`；因此第一版建议优先适配 Wan2.1 的 VAE 和 flow-style scheduler。Wan2.1 14B 支持 480P 和 720P，1.3B 更偏 480P，适合先做低成本实验。([GitHub][1])

---

# 1. 第一版训练目标

不要先训练：

[
z_0^{LR} \rightarrow z_0^{HR}
]

而是训练：

[
U_\theta(x_t^{LR}, \sigma_t) \rightarrow z_0^{HR}
]

也就是：

```text
输入：低分辨率 noisy latent + 当前 sigma
输出：高分辨率 clean latent
```

原因是你的 upsampler 要插在推理中途，例如 50 步中的第 25 步。此时输入 latent 不是干净 latent，而是仍处在采样轨迹中的 noisy latent。

所以第一版推荐做 **noisy-to-clean**，不要直接做 noisy-to-noisy。
noisy-to-clean 更稳定，因为它避免让模型去预测不可恢复的高分辨率随机噪声。

---

# 2. 选择模型版本

第一版建议：

```text
Wan 版本：Wan2.1
VAE：Wan2.1 VAE
任务：2× spatial latent upscaling
时间维度：不做时间超分
训练分辨率：256→512 或 384×224→768×448
clip 长度：17 帧
```

原因：

Wan2.1 的 VAE 公开代码中 `WanVAE` 默认 `z_dim=16`，并且使用固定的 mean/std 做 latent 归一化；encode 输入是 `[C, T, H, W]` 的视频列表，输出是 Wan latent。代码中还可以看到它按时间切片处理，第一帧单独处理，后续按 4 帧一组处理，所以训练 clip 推荐使用 (4n+1) 帧，例如 17、33、49、81 帧。([Hugging Face][2])

不要第一版就用 Wan2.2 TI2V-5B。Wan2.2 的 TI2V-5B 使用高压缩 VAE，官方 README 写到其 (T \times H \times W) 压缩比达到 (4 \times 16 \times 16)，并且 patchification 后总压缩率更高；这会让 latent 结构和 Wan2.1 不同，先做会增加变量。([GitHub][3])

---

# 3. 数据构造总流程

训练样本来自同一段 HR 视频：

```text
HR 视频 clip x_HR
        ↓
退化 + 下采样
LR 视频 clip x_LR
        ↓
Wan-VAE encode
z0_HR = E(x_HR)
z0_LR = E(x_LR)
        ↓
采样 sigma
        ↓
x_t_LR = add_noise(z0_LR, sigma)
        ↓
训练 U(x_t_LR, sigma) ≈ z0_HR
```

也就是每条训练样本保存：

```text
z0_lr.pt
z0_hr.pt
meta.json
```

训练时在线采样 sigma 并加噪。

---

# 4. 数据规格

第一版建议从小规格开始：

```text
frames: 17
fps: 16
HR: 512×512
LR: 256×256
scale: 2×
VAE: Wan2.1
```

对应 latent 大致是：

```text
z0_LR: [16, 5, 32, 32]
z0_HR: [16, 5, 64, 64]
```

如果想更接近真实视频比例，可以用：

```text
HR: 768×448
LR: 384×224
```

但是第一版做方形更容易 debug。

---

# 5. LR 退化策略

不要只做 bicubic downsample。每个 HR clip 生成 LR 时使用随机退化：

```text
HR clip
  ↓
随机 blur
  ↓
随机 resize kernel 下采样
  ↓
随机 noise
  ↓
随机 JPEG / H.264 压缩
  ↓
resize 到精确 LR 尺寸
```

建议实现一个 `degrade_video()`：

```python
def degrade_video(x_hr, lr_size):
    """
    x_hr: torch.Tensor, [T, H, W, C], range [0, 1]
    return: x_lr, [T, h, w, C], range [0, 1]
    """
    # 1. optional blur
    # 2. random downsample kernel: bicubic / bilinear / area / lanczos
    # 3. optional gaussian noise
    # 4. optional codec compression
    # 5. final resize to lr_size
    return x_lr
```

第一版可以先实现：

```text
bicubic / bilinear / area 随机
Gaussian blur
Gaussian noise
JPEG 压缩
```

H.264 压缩可以第二版再加。

---

# 6. latent pair 预处理脚本

让 Codex 先写这个脚本：

```bash
python scripts/build_latent_pairs.py \
  --video_dir data/raw_videos \
  --out_dir data/latent_pairs_wan21_512 \
  --vae_path checkpoints/Wan2.1_VAE.pth \
  --hr_size 512 512 \
  --lr_size 256 256 \
  --num_frames 17 \
  --fps 16
```

输出结构：

```text
data/latent_pairs_wan21_512/
  000000/
    z0_lr.safetensors
    z0_hr.safetensors
    meta.json
  000001/
    z0_lr.safetensors
    z0_hr.safetensors
    meta.json
```

`meta.json`：

```json
{
  "vae": "Wan2.1",
  "frames": 17,
  "fps": 16,
  "hr_size": [512, 512],
  "lr_size": [256, 256],
  "scale": 2,
  "z0_lr_shape": [16, 5, 32, 32],
  "z0_hr_shape": [16, 5, 64, 64],
  "degradation": {
    "resize_kernel": "bicubic",
    "blur": true,
    "noise_std": 0.01,
    "jpeg_quality": 85
  }
}
```

---

# 7. 模型结构

第一版模型：

```text
WanNoisyLatentUpsampler
```

输入：

```text
x_t_lr: [B, 16, T_lat, H, W]
sigma:  [B]
```

输出：

```text
pred_z0_hr: [B, 16, T_lat, 2H, 2W]
```

结构建议：

```text
x_t_lr
  ↓
3D Conv stem
  ↓
若干 SigmaConditionedResBlock3D
  ↓
Spatial PixelShuffle 2×
  ↓
若干 SigmaConditionedResBlock3D
  ↓
3D Conv output
  ↓
pred_z0_hr
```

不要第一版就上 DiT。先用 3D CNN / ResBlock 跑通。

---

## 7.1 Sigma embedding

模型必须知道当前噪声水平。

实现：

```python
sigma_emb = FourierFeatures(sigma)
sigma_emb = MLP(sigma_emb)
```

在 ResBlock 里用 FiLM / AdaGN：

```python
h = norm(h)
h = h * (1 + scale(sigma_emb)) + shift(sigma_emb)
h = silu(h)
```

---

## 7.2 Spatial PixelShuffle

latent 是 5D：

```text
[B, C, T, H, W]
```

2× spatial upsample 可以这样做：

```python
# before: [B, hidden, T, H, W]
h = conv(h)  # [B, hidden * 4, T, H, W]
h = rearrange(h, "b (c r1 r2) t h w -> b c t (h r1) (w r2)", r1=2, r2=2)
```

---

# 8. 加噪方式

Wan / Diffusers 示例使用 flow-style scheduler，`UniPCMultistepScheduler` 里有 `sigmas`。训练时尽量使用同一个 scheduler 的 sigma 分布，避免自己随便定义噪声日程。Wan2.1 的 Diffusers 示例中，720P 使用 `flow_shift=5.0`，480P 使用 `flow_shift=3.0`。([GitHub][1])

第一版训练可以用简化 flow matching 形式：

[
x_\sigma = (1-\sigma) z_0 + \sigma \epsilon
]

即：

```python
eps = torch.randn_like(z0_lr)
x_t_lr = (1.0 - sigma) * z0_lr + sigma * eps
```

注意：

```text
sigma 越大，噪声越重；
sigma 越小，越接近 clean latent。
```

如果后续接入 Diffusers/Wan 原生 scheduler，就把这里替换成 scheduler 官方的 add_noise / scale_noise 逻辑。

---

# 9. sigma 采样策略

由于你计划在 50 步中的第 25 步左右切换，训练不要覆盖全范围平均采样。建议重点采中段。

第一版：

```python
# 假设 sigma in [0, 1]
# 中段为主
70%: sigma ~ Uniform(0.35, 0.70)
20%: sigma ~ Uniform(0.20, 0.85)
10%: sigma ~ Uniform(0.00, 0.20)
```

如果用真实 scheduler timesteps，则：

```text
70%: step 18~32
20%: step 10~40
10%: step 35~49
```

不要大量采最早期高噪声，因为那时低分 latent 里语义还不稳定；不要只采最后期，因为模型会退化成 clean upscaler。

---

# 10. Loss 设计

主 loss：

[
\mathcal{L}_{latent} = |\hat{z}_0^{HR} - z_0^{HR}|_1
]

建议用 Charbonnier：

[
\mathcal{L}_{charb} = \sqrt{(\hat{z}_0^{HR} - z_0^{HR})^2 + \epsilon^2}
]

加一个低频一致性 loss：

```python
pred_down = spatial_downsample(pred_z0_hr, scale=2)
loss_low = L1(pred_down, z0_lr)
```

这能保证预测的 HR latent 下采样后和 LR latent 一致。

再加 temporal consistency loss：

```python
dt_pred = pred[:, :, 1:] - pred[:, :, :-1]
dt_gt = z0_hr[:, :, 1:] - z0_hr[:, :, :-1]
loss_temp = L1(dt_pred, dt_gt)
```

总 loss：

[
\mathcal{L}
===========

1.0 \mathcal{L}*{latent}
+
0.2 \mathcal{L}*{low}
+
0.1 \mathcal{L}_{temp}
]

第一版不要加 VAE decode RGB loss，因为太吃显存。等 latent loss 跑通后，再抽帧 decode 加 RGB loss。

---

# 11. 训练阶段

建议分三阶段。

---

## Stage A：clean warm-up

目标：

[
U(z_0^{LR}, \sigma=0) \rightarrow z_0^{HR}
]

训练 5%~10% 总步数。

作用：

```text
让模型先学会 Wan latent 的空间 2× 映射
避免一上来 noisy 输入导致不收敛
```

---

## Stage B：noisy-to-clean 主训练

目标：

[
U(x_t^{LR}, \sigma_t) \rightarrow z_0^{HR}
]

这是主训练，占 80%~90%。

---

## Stage C：推理轨迹微调，可选

后面再做，第一版先不做。

目标是收集真实 Wan 低分采样第 k 步的 latent，微调 upsampler 适配真实 sampler trajectory。

---

# 12. 训练脚本接口

Codex 实现：

```bash
python train.py \
  --data_dir data/latent_pairs_wan21_512 \
  --out_dir outputs/wan_traj_upsampler_x2 \
  --scale 2 \
  --in_channels 16 \
  --hidden_channels 256 \
  --num_res_blocks 8 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 1e-4 \
  --max_steps 100000 \
  --precision bf16 \
  --sigma_mode mid \
  --warmup_clean_steps 5000
```

第一版超参：

```text
hidden_channels: 256
num_res_blocks: 8
batch_size: 1
grad_accum: 8~16
lr: 1e-4
optimizer: AdamW
weight_decay: 0.01
precision: bf16
ema: 可选，建议开
```

---

# 13. 推理接入逻辑

写一个独立函数：

```python
def transition_lr_to_hr(
    x_t_lr,
    sigma,
    upsampler,
    scheduler,
    noise_mode="hybrid"
):
    """
    x_t_lr: low-res noisy latent at current sigma
    return: high-res noisy latent at same sigma
    """
    pred_z0_hr = upsampler(x_t_lr, sigma)

    eps_hr = make_hr_noise_from_lr(x_t_lr, pred_z0_hr, sigma)

    x_t_hr = (1 - sigma) * pred_z0_hr + sigma * eps_hr

    return x_t_hr
```

第一版 `eps_hr` 可以直接随机：

```python
eps_hr = torch.randn_like(pred_z0_hr)
```

第二版改成 hybrid noise：

```text
低频噪声继承 LR trajectory
高频噪声重新采样
```

hybrid noise：

```python
eps_new = torch.randn_like(pred_z0_hr)
eps_low = upsample(get_noise_estimate_from_lr(...), scale=2)
eps_high = eps_new - upsample(downsample(eps_new), scale=2)
eps_hr = eps_low + lambda_high * eps_high
eps_hr = normalize(eps_hr)
```

第一版不要纠结 hybrid，先跑通。

---

# 14. 评估方式

第一版评估不要只看 loss。要做四类评估。

## 14.1 latent reconstruction

```text
L1(pred_z0_hr, z0_hr)
L1(downsample(pred_z0_hr), z0_lr)
```

## 14.2 VAE decode 可视化

抽样：

```text
pred_z0_hr → WanVAE decode → pred_video
z0_hr → WanVAE decode → gt_video
```

看：

```text
主体是否一致
运动是否连续
边缘是否比 bicubic latent 好
是否闪烁
是否有块状伪影
```

## 14.3 和 baseline 对比

至少三个 baseline：

```text
1. LR RGB bicubic upscale
2. z0_lr latent interpolate → VAE decode
3. clean-only upsampler
4. noisy-to-clean upsampler，也就是你的模型
```

## 14.4 真正插入 Wan 推理

最终必须测试：

```text
Wan LR sample 25 steps
  ↓
upsampler transition
  ↓
Wan HR continue sample 25 steps
```

这个结果最重要，因为你的目标是推理中途切分辨率。

---

# 15. 文件结构建议

让 Codex 按这个项目结构写：

```text
WanTrajectoryUpsampler/
  README.md
  requirements.txt

  configs/
    train_wan21_x2_512.yaml
    infer_transition.yaml

  wan_sr/
    __init__.py
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

---

# 16. 给 Codex 的明确任务清单

你可以直接把下面这段发给 Codex。

```text
请实现一个名为 WanTrajectoryUpsampler 的 PyTorch 项目。

目标：
训练一个 Wan2.1 latent-space 2× spatial upsampler，用于扩散/flow 采样中途的分辨率切换。模型输入 low-res noisy latent x_t_lr 和 sigma，输出 high-res clean latent pred_z0_hr。推理时再把 pred_z0_hr re-noise 到同一 sigma，交给 high-res Wan 继续采样。

第一阶段只实现训练 upsampler，不需要完整接入 Wan 推理。

项目要求：

1. 数据预处理
- 实现 scripts/build_latent_pairs.py。
- 输入 raw video directory。
- 切分连续 clip，默认 17 frames, 16 fps。
- 对 HR clip 做 resize/crop 到 512×512。
- 通过随机退化生成 LR clip，默认 256×256。
- 使用 Wan2.1 VAE encode HR 和 LR clip，得到 z0_hr 和 z0_lr。
- 保存 safetensors：
  sample_id/z0_lr.safetensors
  sample_id/z0_hr.safetensors
  sample_id/meta.json

2. Dataset
- 实现 LatentPairDataset。
- 读取 z0_lr 和 z0_hr。
- 在线采样 sigma。
- 构造 x_t_lr = (1 - sigma) * z0_lr + sigma * noise。
- 返回 x_t_lr, sigma, z0_lr, z0_hr。

3. 模型
- 实现 WanNoisyLatentUpsampler。
- 输入 shape: [B, 16, T, H, W] 和 sigma [B]。
- 输出 shape: [B, 16, T, 2H, 2W]。
- 使用 3D Conv stem、SigmaConditionedResBlock3D、spatial PixelShuffle 2×、output Conv3D。
- sigma 通过 Fourier embedding + MLP 注入 ResBlock。

4. Loss
- latent Charbonnier loss: pred_z0_hr vs z0_hr。
- low-frequency consistency loss: downsample(pred_z0_hr) vs z0_lr。
- temporal difference loss: pred temporal diff vs gt temporal diff。
- total loss = latent + 0.2 * low + 0.1 * temp。

5. Training
- 实现 scripts/train.py。
- 支持 bf16、gradient accumulation、AdamW、EMA、checkpoint save/resume。
- 支持 warmup_clean_steps：前若干步 sigma=0，训练 clean upscaler。
- 之后采样 sigma，训练 noisy-to-clean。

6. Evaluation
- 实现 eval_latent.py：输出 latent loss。
- 实现 eval_decode.py：用 Wan VAE decode pred_z0_hr、z0_hr、latent interpolate baseline，并导出 mp4。
- baseline 包括：
  a. z0_lr interpolate to HR latent size
  b. clean-only mode
  c. noisy-to-clean model

7. Inference transition
- 实现 wan_sr/pipelines/transition.py。
- 函数 transition_lr_to_hr(x_t_lr, sigma, upsampler)：
  pred_z0_hr = upsampler(x_t_lr, sigma)
  eps_hr = torch.randn_like(pred_z0_hr)
  x_t_hr = (1 - sigma) * pred_z0_hr + sigma * eps_hr
  return x_t_hr, pred_z0_hr

注意：
- 第一版不要训练 Wan 主模型。
- Wan VAE 冻结。
- 第一版不要做 temporal upscaling，只做 spatial 2×。
- 第一版不接 text prompt，不加 cross-attention。
- 所有 shape 要打印检查。
- 所有 sample 保存 meta.json。
```

---

# 17. 关键风险和规避

## 风险 1：训练输入分布和真实推理中间态不同

第一版用 forward noising 构造 (x_t^{LR})，不完全等于真实 Wan 采样第 25 步 latent。

规避：

```text
先跑通 forward-noise 版本；
后续加入真实 Wan sampler 中间态做 trajectory fine-tuning。
```

## 风险 2：re-noise 后高分 Wan 接不住

规避：

```text
先在较晚 timestep 切换，例如 50 步中的第 30~35 步；
不要一开始在第 10 步切换。
```

## 风险 3：只学到模糊平均

规避：

```text
加入低频一致性 loss；
加入 temporal loss；
后续加入 VAE decode RGB loss；
增加 Wan-generated domain 数据。
```

## 风险 4：VAE latent 版本混乱

规避：

```text
第一版锁死 Wan2.1 VAE；
不要混 Wan2.2 VAE；
所有 latent pair 的 meta.json 记录 VAE 版本。
```

---

# 18. 最终训练路线

你可以按这个顺序推进：

```text
第 1 步：实现数据预处理，构造 z0_lr / z0_hr
第 2 步：训练 clean upsampler，验证 z0_lr → z0_hr 能收敛
第 3 步：加入 sigma，训练 noisy-to-clean upsampler
第 4 步：VAE decode 可视化，对比 latent interpolate baseline
第 5 步：实现 transition_lr_to_hr
第 6 步：插入 Wan 推理，在较晚 timestep 做 LR→HR 切换
第 7 步：收集真实 Wan 中间 latent，做 trajectory fine-tuning
```

第一篇可展示结果做到第 6 步就足够有价值。

---

# 19. 一句话版本

**第一版不要做“普通 Wan latent 超分”，而要做“sigma-conditioned noisy-to-clean latent upsampler”：先用 HR/LR 视频对编码得到 (z_0^{HR}, z_0^{LR})，再对 (z_0^{LR}) 按 Wan scheduler 加噪得到 (x_t^{LR})，训练模型从 ((x_t^{LR}, \sigma_t)) 预测 (z_0^{HR})。推理时在 Wan 低分辨率采样中途调用该模型，预测高分辨率 clean latent，再重新加噪到当前 timestep，交给高分辨率 Wan 继续采样。**

[1]: https://github.com/Wan-Video/Wan2.1?utm_source=chatgpt.com "GitHub - Wan-Video/Wan2.1: Wan: Open and Advanced Large-Scale Video Generative Models · GitHub"
[2]: https://huggingface.co/spaces/fffiloni/Wan2.1/blob/a0044b510a3c221a7f327789b54a004ceae5e6b8/wan/modules/vae.py?utm_source=chatgpt.com "wan/modules/vae.py · fffiloni/Wan2.1 at a0044b510a3c221a7f327789b54a004ceae5e6b8"
[3]: https://github.com/Wan-Video/Wan2.2?utm_source=chatgpt.com "GitHub - Wan-Video/Wan2.2: Wan: Open and Advanced Large-Scale Video Generative Models · GitHub"
