# 面向高分辨率视频扩散的轨迹对齐潜空间切换

## Trajectory-Aligned Latent Handoff for Efficient High-Resolution Video Diffusion

> 草稿状态：论文骨架 v0.1，2026-07-15。
>
> 内部写作提醒：本文不能将 latent upsampler、PixelShuffle 或 latent 视频超分本身表述为首创。Stage2 架构应明确注明受 LTX-2 LatentUpsampler 启发。建议将核心贡献集中在 Wan 单轨迹内的可学习分辨率 handoff、handoff-step LoRA 轨迹校正，以及二者在 50-step 与 4-step distilled sampler 上的统一验证。

## 摘要

视频扩散模型在高分辨率 latent 上执行全部去噪步骤会产生较高的计算和显存开销。已有动态分辨率推理方法通过在低分辨率完成早期去噪、随后插值到高分辨率继续采样来降低成本，但固定插值无法补偿低分辨率轨迹在切换时残留的去噪误差，也不能学习视频 VAE latent 在不同空间分辨率之间的非线性对应关系。本文提出轨迹对齐潜空间切换框架（Trajectory-Aligned Latent Handoff, TALH），将分辨率切换解耦为两个子问题：首先，仅在 handoff step 激活低秩适配器，使当前低分辨率 clean prediction 对齐完整低分辨率 teacher 轨迹的最终 endpoint；随后，使用轻量 clean-latent Stage2 网络将校正后的低分辨率 latent 映射到高分辨率 latent，并重新加噪至下一时间步以继续高分辨率去噪。该设计避免让单一 upsampler 同时承担轨迹修复和空间升分；在确定性 prefix 及固定 prompt、seed、scheduler 的条件下，单步局部 LoRA 的 cached-prefix 训练状态与推理访问状态一致。我们在 Wan2.1 50-step 模型和 4-step step-distilled 模型上构建统一实现，覆盖 480p 到 720p 的 1.5 倍 latent 放大，以及 368x640 到 720x1248 的近 2 倍 latent 放大。实验将从 handoff endpoint 误差、latent lifting 误差、完整生成质量、时序稳定性和推理效率五个层面验证各模块的贡献。初步 4-step 实验中，handoff LoRA 在 10 个样本上将 endpoint L1 均值降低 10.88%；完整 Stage2 和效率结果将在最终版本中补充。

**关键词：** 视频扩散；动态分辨率；潜空间超分；轨迹蒸馏；低秩适配；Wan

---

## 1. 引言

### 1.1 研究背景

以 Wan 为代表的视频扩散 Transformer 已能够生成具有较强语义一致性和运动表现的视频，但其计算量随视频时空 token 数量快速增长。在全部采样步骤上使用目标分辨率并不总是必要：高噪声或中高噪声阶段主要决定全局布局、主体和粗粒度运动，细节恢复则更多发生在低噪声阶段。因此，先在低分辨率上建立内容和运动，再在采样后段切换到高分辨率，是一种直接的空间加速路径。

LightX2V 已提供 changing-resolution 推理：在指定 step 将当前 latent 插值到目标尺寸，再继续高分辨率去噪。该方法简单且无需训练，但存在两个问题。第一，固定插值假设不同分辨率的 VAE latent 近似尺度等变，而分别编码低分辨率与高分辨率视频得到的 latent 并不严格满足这一假设。第二，切换时送入插值算子的通常是当前 denoiser 给出的单步 clean prediction，而不是完整低分辨率轨迹的最终 clean latent；过早切换会把残留去噪误差和空间升分误差混合在一起。

### 1.2 核心观察

旧的直接映射方案尝试学习

\[
F_\omega(x_{0,s}^{L}) \rightarrow z_0^{H},
\]

其中，\(x_{0,s}^{L}\) 是 handoff step 的单步 clean prediction。这迫使 \(F_\omega\) 同时完成：

1. 修复当前 prediction 与完整低分辨率 teacher endpoint 之间的轨迹误差；
2. 完成低分辨率到高分辨率的 latent lifting。

本文的核心判断是，这两个误差来源应由不同模块处理：denoiser-side LoRA 负责轨迹校正，clean-latent Stage2 只负责空间升分。

### 1.3 方法概览

```text
LR noise
  -> LR base denoising prefix
  -> handoff-step base + LoRA
  -> trajectory-aligned clean LR latent
  -> learned clean-latent Stage2
  -> clean HR latent
  -> re-noise at the next scheduled sigma
  -> HR denoising suffix
  -> Wan VAE decode
```

### 1.4 主要贡献

本文计划主张以下贡献：

1. 提出一种面向视频扩散动态分辨率采样的轨迹对齐 latent handoff，将 handoff 误差显式拆分为低分辨率尾轨迹误差和跨分辨率 latent lifting 误差。
2. 提出 handoff-localized endpoint distillation：仅在切换 step 激活 LoRA，用完整低分辨率 teacher endpoint 监督当前 flow prediction，使 Stage2 接收更接近 clean-latent 训练域的输入。
3. 构建 Wan clean-latent Stage2，将 LTX-2 风格 latent upsampler 适配到 Wan 16 通道视频 latent，并支持 1.5 倍有理数放大与近 2 倍 PixelShuffle-crop 放大。
4. 在标准 50-step Wan2.1 与 4-step distilled Wan 上建立统一的训练、bridge 和评估体系，分析长轨迹与少步轨迹下 handoff 机制的共性和差异。

### 1.5 不应使用的贡献表述

- 不声称“首次提出 latent-to-latent 视频超分”。
- 不声称 PixelShuffle、3D ResBlock 或 LTX-2 upsampler 骨干具有原创性。
- 不将当前 cached-prefix endpoint training 称为 D-OPSD 或完整 on-policy distillation。
- 不声称 LoRA 减少了总 denoising step；它减少的是高分辨率执行的 step 数量，并校正切换状态。
- 不将当前 resize-only 合成数据训练出的模型称为通用 real-world VSR。

---

## 2. 相关工作

### 2.1 潜空间视频生成与级联高分辨率生成

Latent Diffusion Models 将生成过程转移到压缩 latent 空间，为高分辨率图像和视频生成提供了计算基础。Video LDM、Imagen Video 和 LaVie 使用空间或时间超分级联获得高分辨率视频。LTX-Video 与 LTX-2 进一步强调视频 VAE、latent 表示和生成 Transformer 的协同设计，并公开了独立 spatial latent upsampler。近期 LUVE、SimpleGVR 与 Ultra Flash 均研究了低分辨率视频生成后接 latent 或视频超分模块的级联系统。

与这些独立级联系统不同，本文的 Stage2 被插入同一 Wan diffusion trajectory：切换后仍通过原 Wan denoiser 在高分辨率 latent 上完成剩余采样步骤。

### 2.2 视频超分与时序一致性

Upscale-A-Video、SATeCo、MGLD 和 VideoGigaGAN 分别通过 latent diffusion、时空适配模块、运动引导或生成式 upsampler 提升视频细节并控制闪烁。这些工作主要处理已生成或真实低分辨率视频到高分辨率 RGB 视频的恢复。本文处理的是 VAE decode 之前、扩散轨迹内部的 clean-latent resolution transition，输入分布和评价方式均不同。

### 2.3 混合分辨率扩散推理

LightX2V changing-resolution 通过固定三线性插值在采样中途改变 latent 尺寸。RALU 进一步指出，naive latent upsampling 的主要问题包括高频边缘混叠和不同分辨率下的 noise-timestep mismatch。本文通过学习型 clean-latent lifting 处理尺度映射，并在切换后依据目标分辨率 noise bank 重新加噪，而不是直接插值 noisy latent。

### 2.4 扩散模型少步蒸馏

Progressive Distillation、Consistency Models、LCM、DMD/DMD2 和 Phased Consistency Models 从轨迹回归、consistency mapping 或 distribution matching 等角度减少采样步数。LCM-LoRA 表明低秩适配可以作为可插拔的 neural solver。视频方向的 VideoLCM、T2V-Turbo、Motion Consistency Model、Self-Forcing 和 SGMD 分别研究了视频少步生成、奖励优化、运动/外观解耦、训练推理 gap 和 score matching。

本文不重新训练完整 few-step generator，而是对既有 sampler 的一个 handoff step 做局部 endpoint 校正。

### 2.5 On-policy 轨迹学习

DMD2、Self-Forcing、D-OPSD、AnyFlow 和 OPSD-V 均指出：训练状态与模型推理时实际访问状态不一致会导致明显性能退化。D-OPSD 在 student 自己的 rollout state 上，以更强 multimodal context 下的 EMA teacher 提供监督；AnyFlow 学习任意时间区间的 flow-map transition。

当前方法使用 cached teacher prefix，但只在 handoff step 启用 LoRA。由于参数更新不会影响该 step 之前的状态，在确定性 prefix、相同 prompt、seed 和 scheduler 下，缓存输入就是推理实际输入。该性质将在第 4.4 节形式化讨论。

---

## 3. 问题定义

### 3.1 符号

| 符号 | 含义 |
|---|---|
| \(L,H\) | 低分辨率和高分辨率 latent 域 |
| \(T\) | sampler 总 denoising step 数 |
| \(s\) | handoff step，使用 1-based 编号 |
| \(x_s^L\) | 执行第 \(s\) 次 denoiser 前的 LR noisy/intermediate latent |
| \(v_\phi\) | 冻结的 base Wan flow/velocity predictor |
| \(\Delta\theta_s\) | 仅在 handoff step 激活的 LoRA 参数 |
| \(z_T^L\) | base teacher 完整 LR rollout 的最终 endpoint |
| \(z_0^L,z_0^H\) | 同一视频分别经 LR/HR 路径编码得到的 clean latent pair |
| \(U_\psi\) | clean-latent Stage2 resizer |
| \(\sigma_s\) | handoff step 对应 flow sigma |

### 3.2 目标

给定预训练 Wan、总步数 \(T\) 和 handoff step \(s\)，目标是在尽量多的 LR steps 和尽量少的 HR steps 下，得到接近完整 HR 采样质量的结果：

\[
x_1^L \xrightarrow{\text{LR prefix}}
x_s^L \xrightarrow{\text{LoRA handoff}}
\tilde z_s^L \xrightarrow{U_\psi}
\hat z_s^H \xrightarrow{\text{re-noise + HR suffix}}
x_T^H.
\]

本文分别考察三个误差：

\[
E_{\mathrm{handoff}}=d(\tilde z_s^L,z_T^L),
\]

\[
E_{\mathrm{lift}}=d(U_\psi(z_0^L),z_0^H),
\]

\[
E_{\mathrm{chain}}=Q(y_{\mathrm{mixed-res}},y_{\mathrm{reference}}),
\]

其中 \(Q\) 包括无参考生成指标、同算子 reference 指标和人工盲测。

---

## 4. 方法

### 4.1 总体框架

**图 1（待绘制）：** TALH 总览。左侧为 LR Wan prefix，中间为仅在 step \(s\) 生效的 LoRA 和 Stage2，右侧为 re-noise 后的 HR Wan suffix。图中应分别标出 50-step 的 \(s=40/45,T=50\) 和 distilled 4-step 的 \(s=3,T=4\)。

方法包含两个独立训练阶段：

1. 用 clean latent pairs 训练 \(U_\psi:z_0^L\rightarrow z_0^H\)。
2. 冻结 Wan base model，在缓存的 handoff state 上训练 \(\Delta\theta_s\)，使单步 clean prediction 接近 \(z_T^L\)。

二者当前不做联合反向传播，从而避免 14B denoiser 与 Stage2 同时驻留训练图带来的显存开销。

### 4.2 Clean-latent Stage2

输入和输出均为 Wan VAE latent：

\[
z_0^L\in\mathbb R^{B\times16\times F\times h\times w},\qquad
\hat z_0^H\in\mathbb R^{B\times16\times F\times H\times W}.
\]

网络采用 LTX-2-inspired encoder-resizer-decoder 结构：

```text
Conv3D stem
  -> GroupNorm + SiLU
  -> N/2 x 3D ResBlock
  -> spatial resampler
  -> N/2 x 3D ResBlock
  -> Conv3D output
```

默认配置为 hidden width 256、8 个 ResBlock、16 输入/输出通道，不改变 latent 时间维。

#### 4.2.1 1.5 倍有理数重采样

480x832 到 720x1248 对应 latent `60x104 -> 90x156`。网络先通过 Conv3D 将通道扩展 9 倍，再执行 spatial PixelShuffle x3，最后用固定二项式模糊核 stride 2 下采样：

\[
60\times104\xrightarrow{\times3}180\times312
\xrightarrow{/2}90\times156.
\]

#### 4.2.2 近 2 倍重采样

368x640 到 720x1248 对应 latent `46x80 -> 90x156`。由于目标不是严格 2 倍，网络采用：

\[
46\times80\xrightarrow{\text{PixelShuffle }\times2}92\times160
\xrightarrow{\text{center crop}}90\times156.
\]

该路径避免预先将输入插值到旧 1.5 倍模型要求的尺寸。

### 4.3 Stage2 训练目标

Stage2 总损失为：

\[
\mathcal L_{\mathrm{S2}}=
\lambda_z\mathcal L_{\mathrm{charb}}
+\lambda_l\mathcal L_{\mathrm{low}}
+\lambda_t\mathcal L_{\mathrm{temp}}
+\lambda_r\mathcal L_{\mathrm{res}}.
\]

其中：

\[
\mathcal L_{\mathrm{charb}}=
\mathbb E\sqrt{(\hat z_0^H-z_0^H)^2+\epsilon^2},
\]

\[
\mathcal L_{\mathrm{low}}=
\|D(\hat z_0^H)-z_0^L\|_1,
\]

\[
\mathcal L_{\mathrm{temp}}=
\|\Delta_F\hat z_0^H-\Delta_F z_0^H\|_1.
\]

当前权重为 \((\lambda_z,\lambda_l,\lambda_t,\lambda_r)=(1.0,0.2,0.1,0)\)，\(\epsilon=10^{-3}\)。

### 4.4 Handoff-localized endpoint LoRA

冻结 base Wan 参数，仅在目标线性层注入 rank-\(r\) LoRA：

\[
W'=W+\alpha BA/r.
\]

当前 target modules 为 attention 的 `q,k,v,o` 与 FFN 的 `ffn.0,ffn.2`，默认 rank 和 alpha 均为 32。

在 handoff step，模型输出 flow prediction：

\[
\tilde z_s^L=x_s^L-\sigma_s
v_{\phi+\Delta\theta_s}(x_s^L,s,c).
\]

teacher target 是相同 prompt、seed 和 LR 初始噪声下，base model 完整运行至 \(T\) 得到的 \(z_T^L\)。默认 endpoint loss 为：

\[
\mathcal L_{\mathrm{endpoint}}=
\|\tilde z_s^L-z_T^L\|_1
+0.1\|\tilde z_s^L-z_T^L\|_2^2.
\]

50-step 的 step40 temporal variant 额外使用：

\[
\mathcal L_{\mathrm{endpoint-temp}}=
\mathcal L_{\mathrm{endpoint}}
+0.05\|\Delta_F\tilde z_s^L-\Delta_Fz_T^L\|_1.
\]

代码也支持直接监督 target flow：

\[
v^*=\frac{x_s^L-z_T^L}{\sigma_s},
\]

但当前主配置中 velocity loss 权重为 0，应作为消融而不是默认方法描述。

### 4.5 Cached-prefix 一致性

设 LoRA 只在 handoff step \(s\) 生效，且在所有 \(k<s\) 的步骤中 \(\Delta\theta_k=0\)。对于确定性 sampler 和固定 prompt、seed，有：

\[
x_k^{\mathrm{student}}=x_k^{\mathrm{base}},\quad \forall k\le s.
\]

因此：

\[
x_s^{\mathrm{student}}=x_s^{\mathrm{cached\ teacher}}.
\]

这说明当前 cached prefix 不是对 student-visited handoff state 的近似，而是在上述约束下的精确输入。该结论不适用于以下情况：

- LoRA 在 handoff 之前的步骤激活；
- LoRA 同时训练多个连续 step；
- prefix 使用不同 scheduler、CFG、prompt embedding 或噪声；
- sampler 本身包含未对齐的随机操作。

多步 LoRA 扩展应采用 student rollout、same-state teacher prediction 和 EMA teacher，届时才能称为 on-policy 训练。

### 4.6 Resolution lifting 与 re-noise

得到 \(\tilde z_s^L\) 后，Stage2 预测：

\[
\hat z_s^H=U_\psi(\tilde z_s^L).
\]

若 handoff 不是最后一步，使用目标分辨率 noise bank 重新加噪：

\[
x_{s+1}^H=(1-\sigma_{s+1})\hat z_s^H
+\sigma_{s+1}\epsilon^H.
\]

随后由相同 Wan denoiser 在 HR latent 上执行剩余步骤。`resize_flow` 将 LR flow 直接插值到 HR 的策略仅保留为消融；主方法使用 random/fixed target-resolution noise。

### 4.7 50-step 实例

标准路线使用 Wan2.1 T2V-1.3B：

```text
infer_steps = 50
sample_shift = 8
CFG scale = 6
handoff s = 45 or 40
teacher endpoint T = 50
```

以 step45 为例：

```text
LR steps 1..44 base
  -> LR step45 base + LoRA predicts teacher50-like clean latent
  -> Stage2 LR->HR
  -> re-noise at step46 sigma
  -> HR steps 46..50 base
```

### 4.8 4-step distilled 实例

distill 路线使用 Wan2.1 T2V-14B StepDistill-CfgDistill：

```text
infer_steps = 4
denoising timesteps = [1000, 750, 500, 250]
sample_shift = 5
handoff s = 3
teacher endpoint T = 4
```

完整链路为：

```text
LR steps 1..2 base
  -> LR step3 base + LoRA predicts teacher4-like clean latent
  -> Stage2 LR->HR
  -> re-noise at step4 sigma
  -> HR step4 base
```

### 4.9 参数与计算量

按当前网络定义计算，Stage2 约包含：

| Stage2 路径 | 参数量 | 备注 |
|---|---:|---|
| 1.5x rational | 44.47M | Conv3D 通道扩展 9 倍 |
| 2x + crop | 35.62M | Conv3D 通道扩展 4 倍 |

按 Wan 官方维度和 rank 32 粗略计算，若所有配置 target module 均成功匹配：

| LoRA base | 估算可训练参数 | 最终需以 trainer 日志核对 |
|---|---:|---|
| Wan2.1 1.3B | 31.95M | 30 层，dim 1536，FFN 8960 |
| Wan2.1 14B | 100.93M | 40 层，dim 5120，FFN 13824 |

最终论文应报告实际 `trainable_params`、Stage2 FLOPs、DiT FLOPs、wall-clock latency 和峰值显存，不仅报告理论空间 token 比例。

---

## 5. 实验设置

### 5.1 研究问题

- **RQ1：** learned Stage2 是否优于固定三线性插值？
- **RQ2：** handoff LoRA 是否使当前 LR clean prediction 更接近完整 LR teacher endpoint？
- **RQ3：** endpoint 对齐是否能转化为完整 Stage2+HR suffix 的视频质量提升？
- **RQ4：** 该分解是否同时适用于 50-step 和 4-step distilled sampler？
- **RQ5：** 更早 handoff 带来的效率收益与质量退化之间如何权衡？

### 5.2 数据集

#### 50-step 路线

- 上游模型：Wan2.1 T2V-1.3B。
- teacher 视频：约 1,000 个 720x1248、81 帧、16 fps 生成视频。
- Stage2 pair：HR teacher 视频与其 bicubic LR 版本分别经 Wan VAE encode。
- LR 尺寸：480x832 或 368x640。
- HR 尺寸：720x1248。

#### 4-step distill 路线

- 上游模型：Wan2.1 T2V-14B StepDistill-CfgDistill。
- teacher 视频：约 5,000 个 720x1248、81 帧、16 fps 生成视频。
- Stage2 pair：368x640 与 720x1248 clean latent pair。
- LoRA pair：`x_pre_step3_lr` 与 `z4_lr_teacher`。

#### 数据限制

当前退化仅为 resize-only，且训练样本来自生成视频。最终实验需要说明这是一项 model-specific generative upscaling 任务，并考虑加入以下泛化测试：

- 未见 prompt 与不同 seed；
- 不同内容类别和运动强度；
- 自然视频编码得到的 latent；
- 不同生成模型或不同 Wan checkpoint；
- bicubic 之外的 area/bilinear/codec degradation。

### 5.3 训练细节

| 模块 | 配置 |
|---|---|
| Stage2 optimizer | AdamW |
| Stage2 LR | 1e-4 |
| Stage2 max steps | 50k |
| Stage2 precision | bf16 |
| Stage2 EMA | 0.9999 |
| LoRA optimizer | AdamW |
| LoRA LR | 5e-5 |
| LoRA max steps | 10k |
| LoRA precision | bf16 |
| LoRA rank/alpha | 32/32 |

最终版本补充硬件、GPU 数量、有效 batch size、训练时长、checkpoint 选择规则和随机种子。

### 5.4 对比方法

#### 核心基线

1. Full-HR：全部步骤在 720p latent 上运行。
2. Full-LR：全部步骤在 LR latent 上运行并直接 decode。
3. Interp handoff：LightX2V 固定三线性插值。
4. Base + Stage2：无 handoff LoRA 的 learned latent lifting。
5. LoRA + Interp：只验证轨迹校正，不使用 learned Stage2。
6. LoRA + Stage2：完整 TALH。
7. Teacher endpoint + Stage2：Stage2 的理想 clean-input 上界。
8. 旧 Stage3：单模型 `x0_pred_lr -> z0_hr`。

#### 外部方法

在条件允许时加入 LTX-2 spatial upsampler、独立 VSR 或 cascaded baseline。若 latent 通道和 VAE 不兼容，应只做系统层面的定性讨论，不进行不公平的直接权重比较。

### 5.5 指标

#### Handoff 指标

- latent L1 / MSE：\(\tilde z_s^L\) 对 \(z_T^L\)。
- temporal latent L1：相邻 latent 帧差分。
- decode LPIPS / PSNR / SSIM。
- 10-prompt 或更大测试集上的 win rate。

#### Stage2 operator 指标

- latent Charbonnier/L1。
- decoded PSNR、SSIM、LPIPS。
- temporal difference error。
- 高频能量仅作为诊断，不作为独立成功标准。

#### 完整生成指标

- VBench / VBench2 子项。
- CLIP 或文本-视频对齐指标。
- optical-flow warping error 或 temporal LPIPS。
- 人工盲测：细节、artifact、时序稳定性、结构/身份。

#### 效率指标

- 单视频 wall-clock latency。
- peak VRAM。
- LR/HR denoiser step 分布。
- Stage2 和 LoRA 动态加载开销。
- 相对 full-HR 的端到端加速比。

### 5.6 公平性控制

- 每组使用完全相同的 prompt、seed、初始 noise 和 scheduler。
- 比较 LoRA 时，优先使用相同 Stage2 operator 的 `Base+Stage2` 与 `LoRA+Stage2`。
- 比较 Stage2 与 interpolation 时，二者从相同 teacher endpoint 开始。
- 无真实 HR ground truth 的完整生成不得仅依据 PSNR 排名。
- 人工评估隐藏方法名称并随机左右顺序。

---

## 6. 主实验结果

### 6.1 Stage2 operator comparison

**表 1（待填）：Clean latent operator comparison。**

| Setting | Method | Latent L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Temp. L1 ↓ |
|---|---|---:|---:|---:|---:|---:|
| 480p->720p | Trilinear | TBD | TBD | TBD | TBD | TBD |
| 480p->720p | Stage2 1.5x | TBD | TBD | TBD | TBD | TBD |
| 368p->720p | Trilinear | TBD | TBD | TBD | TBD | TBD |
| 368p->720p | Stage2 2x-crop | TBD | TBD | TBD | TBD | TBD |

待回答：Stage2 的优势来自真实细节恢复，还是单纯增加锐度和高频能量？

### 6.2 Handoff LoRA endpoint alignment

**表 2（待填）：LR endpoint alignment。**

| Sampler | Handoff | Method | L1 ↓ | MSE ↓ | LPIPS ↓ | Temp. L1 ↓ | Wins |
|---|---:|---|---:|---:|---:|---:|---:|
| 50-step | 45->50 | Base | TBD | TBD | TBD | TBD | TBD |
| 50-step | 45->50 | LoRA | TBD | TBD | TBD | TBD | TBD |
| 50-step | 40->50 | Base | TBD | TBD | TBD | TBD | TBD |
| 50-step | 40->50 | LoRA+temp | TBD | TBD | TBD | TBD | TBD |
| 4-step | 3->4 | Base | 0.03207 | TBD | TBD | TBD | 0/10 |
| 4-step | 3->4 | LoRA | 0.02858 | TBD | TBD | TBD | 10/10 |

当前 4-step 数值为开发文档中的 10 样本初步结果，最终版本必须用固定 checkpoint 和独立测试集重新生成。

### 6.3 完整 2x2 因子实验

**表 3（待填）：分离 LoRA 主效应、Stage2 主效应及交互效应。**

| Handoff | Resizer | VBench ↑ | LPIPS to anchor ↓ | Temp. error ↓ | Human pref. ↑ | Time ↓ |
|---|---|---:|---:|---:|---:|---:|
| Base | Interp | TBD | TBD | TBD | TBD | TBD |
| Base | Stage2 | TBD | TBD | TBD | TBD | TBD |
| LoRA | Interp | TBD | TBD | TBD | TBD | TBD |
| LoRA | Stage2 | TBD | TBD | TBD | TBD | TBD |

重点报告：

- LoRA 是否独立改善 handoff；
- Stage2 是否独立优于 interpolation；
- LoRA 是否使 Stage2 的收益更稳定；
- 两个模块是否出现负交互或 distribution shift。

### 6.4 与 Full-HR 和旧 Stage3 比较

**表 4（待填）：质量-效率 Pareto。**

| Method | LR steps | HR steps | Total time | Peak VRAM | VBench | Human pref. |
|---|---:|---:|---:|---:|---:|---:|
| Full-HR 50-step | 0 | 50 | TBD | TBD | TBD | TBD |
| Interp@45 | 45 | 5 | TBD | TBD | TBD | TBD |
| Stage3@45 | 45 | 5 | TBD | TBD | TBD | TBD |
| TALH@45 | 45 | 5 | TBD | TBD | TBD | TBD |
| Full-HR distill | 0 | 4 | TBD | TBD | TBD | TBD |
| TALH distill | 3 | 1 | TBD | TBD | TBD | TBD |

---

## 7. 消融实验

### 7.1 Handoff step

50-step 扫描 `30,35,40,45,50`；distill 扫描 `1,2,3`。绘制质量、延迟和 handoff error 随切换步变化的曲线。

### 7.2 LoRA target modules

| Variant | qkvo | FFN | Rank | Endpoint L1 | Temp. | Time/step |
|---|---|---|---:|---:|---:|---:|
| A | yes | no | 8 | TBD | TBD | TBD |
| B | yes | no | 16 | TBD | TBD | TBD |
| C | yes | yes | 16 | TBD | TBD | TBD |
| D | yes | yes | 32 | TBD | TBD | TBD |

### 7.3 LoRA loss

- endpoint L1；
- endpoint L1 + MSE；
- velocity MSE；
- endpoint + velocity；
- endpoint + temporal difference。

### 7.4 Cached-prefix 与 on-policy rollout

- step-only cached prefix；
- recomputed base prefix；
- LoRA-active prefix；
- LoRA-active prefix + EMA same-state teacher。

该实验用于验证：单步局部 LoRA 下 cached prefix 是否已经足够，以及多步 LoRA 何时需要 on-policy。

### 7.5 Stage2 架构

- trilinear；
- residual-to-trilinear；
- direct prediction；
- 1.5x rational resampler；
- 2x PixelShuffle + crop；
- 2D per-frame resampler；
- 3D temporal-aware resampler。

### 7.6 Stage2 loss

- Charbonnier only；
- + low-frequency consistency；
- + temporal difference；
- + decode perceptual loss（若显存允许）。

### 7.7 Re-noise

- independent/fixed HR noise bank；
- resized LR flow；
- shared low-frequency + new high-frequency noise；
- 不 re-noise，直接从 clean latent 继续。

### 7.8 数据分布

- 1k vs 5k；
- standard Wan teacher vs distilled teacher；
- clean VAE latent vs LoRA-produced clean prediction；
- resize-only vs richer degradation；
- 同模型训练/测试 vs cross-model transfer。

---

## 8. 定性分析

### 8.1 建议展示案例

- 人脸、手部、文字、小物体和重复纹理；
- 快速主体运动和镜头运动；
- 低纹理平面，用于观察噪点和色带；
- 细密结构，用于观察 ringing、checkerboard 和纹理爬动；
- 失败案例，而不仅是成功案例。

### 8.2 图版设计

**图 2：** 同 prompt/seed 的 `Full-LR | Interp | Stage2 | LoRA+Stage2 | Full-HR`。

**图 3：** handoff 前后 latent 或 decode crop，展示 LoRA 对 endpoint 的校正。

**图 4：** 连续 5 帧局部 crop 或时空切片，展示闪烁和纹理稳定性。

**图 5：** 质量-延迟 Pareto 曲线。

**图 6：** 不同 handoff step 的 trajectory error 曲线。

---

## 9. 讨论

### 9.1 为什么解耦优于直接 `x0_pred -> z_hr`

待结合实验论证以下假设：直接 Stage3 对特定 timestep 和 denoiser error 分布高度耦合；LoRA+Stage2 将 timestep-specific correction 留在具备语义建模能力的 Wan DiT 内，而 Stage2 学习更稳定、可复用的 clean-latent spatial mapping。

### 9.2 为什么单步 LoRA 不一定需要 on-policy

关键不在于是否使用缓存，而在于参数更新是否改变到达训练 state 的 prefix。step-only LoRA 不改变 prefix；多步或全程 LoRA 会改变 prefix。该条件化结论比笼统地比较 on-policy/off-policy 更准确。

### 9.3 50-step 与 4-step 的差异

50-step handoff 可以在较宽的 step 区间权衡效率和质量；4-step sampler 的每一步承担更大轨迹跨度，单步误差更敏感。若同一框架在两者上均成立，可支持“方法针对 handoff，而非绑定某一 scheduler”的论点。

### 9.4 Stage2 的模型专属性

分别 VAE encode 的 latent 对不具备严格尺度等变性。当前 Stage2 可能学习了 Wan VAE、teacher 生成分布和 resize recipe 的联合映射。跨模型、跨 VAE 和自然视频测试将决定其可迁移程度。

---

## 10. 局限性

1. Stage2 架构主要源自 LTX-2 latent upsampler，架构创新有限。
2. 当前训练数据是模型生成视频且退化为 resize-only，不覆盖真实视频退化。
3. LoRA target 是 base teacher endpoint，性能上限受 teacher 质量约束。
4. LoRA 输出与 Stage2 clean-pair 输入仍可能存在 distribution shift。
5. 当前方法针对固定 handoff step；不同 step 通常需要独立数据和 checkpoint。
6. 当前 cached-prefix 方法不能直接推广为多步 on-policy distillation。
7. 完整链路缺乏真实 720p ground truth，必须依赖多指标和人工盲测。
8. Stage2 与 Wan 尚未端到端联合优化。

---

## 11. 结论

本文提出 TALH，一种面向视频扩散动态分辨率采样的轨迹对齐潜空间切换方法。该方法通过 handoff-step LoRA 将当前低分辨率 clean prediction 对齐完整 teacher endpoint，再由 clean-latent Stage2 完成空间升分和高分辨率轨迹续接。与把去噪修复和升分压入单一模型的方案相比，TALH 提供了更清晰的模块职责和评估分解。最终结论需在 50-step 和 4-step distilled Wan 的完整质量-效率实验、消融和人工评估完成后确定。

---

## 附录 A：代码实现映射

| 论文模块 | 当前实现 |
|---|---|
| Stage2 model | `wan_sr/models/stage2_resizer.py` |
| Stage2 loss | `wan_sr/losses/clean_latent_losses.py` |
| Stage2 trainer | `changing_resolution/scripts/train/train_clean_latent_resizer_stage2.py` |
| Clean pair builder | `changing_resolution/scripts/data/build_480p720p_lmdb.py` |
| 50-step LoRA trainer | `changing_resolution/scripts/train/train_tail_skip_lora.py` |
| 50-step bridge | `changing_resolution/lightx2v_clean_bridge.py` |
| 4-step LoRA trainer | `changing_resolution_distill/scripts/train/train_last_step_skip_lora.py` |
| 4-step bridge | `changing_resolution_distill/lightx2v_distill_bridge.py` |
| Four-way protocol | `doc/360P_FOUR_WAY_EVAL_PROTOCOL.md` |

## 附录 B：投稿前必须补齐的材料

- [ ] Stage2 operator 指标原始 CSV/JSONL。
- [ ] 50-step LoRA 独立测试集指标。
- [ ] 4-step LoRA 固定 checkpoint 的正式复现。
- [ ] 完整 2x2 因子实验。
- [ ] Full-HR 质量和效率基线。
- [ ] handoff step sweep。
- [ ] LoRA loss、rank、target-module 消融。
- [ ] Stage2 loss 和架构消融。
- [ ] 至少两名或更多评审者的盲测。
- [ ] 失败案例与局限性图版。
- [ ] 实际参数量、FLOPs、延迟和峰值显存。
- [ ] LTX-2 代码来源、引用和许可证核对。

---

## 参考文献初表

[1] Wan Team. [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314). 2025.

[2] Rombach et al. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752). CVPR 2022.

[3] Blattmann et al. [Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818). CVPR 2023.

[4] Ho et al. [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303). 2022.

[5] Wang et al. [LAVIE: High-Quality Video Generation with Cascaded Latent Diffusion Models](https://arxiv.org/abs/2309.15103). 2023.

[6] HaCohen et al. [LTX-Video: Realtime Video Latent Diffusion](https://arxiv.org/abs/2501.00103). 2024.

[7] Lightricks. [LTX-2: Efficient Joint Audio-Visual Foundation Model](https://arxiv.org/abs/2601.03233). 2026; [LatentUpsampler source](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core/src/ltx_core/model/upsampler).

[8] Zhao et al. [LUVE: Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts](https://arxiv.org/abs/2602.11564). 2026.

[9] Ultra Flash Team. [Ultra Flash: Scaling Real-Time Streaming Video Generation to High Resolutions](https://arxiv.org/abs/2606.09150). 2026.

[10] SimpleGVR Team. [SimpleGVR: A Simple Baseline for Latent-Cascaded Video Super-Resolution](https://arxiv.org/abs/2506.19838). 2025.

[11] Jeong et al. [Training-free Mixed-Resolution Latent Upsampling for Spatially Accelerated Diffusion Transformers](https://arxiv.org/abs/2507.08422). 2025.

[12] Zhou et al. [Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution](https://arxiv.org/abs/2312.06640). CVPR 2024.

[13] SATeCo Team. [Learning Spatial Adaptation and Temporal Coherence in Diffusion Models for Video Super-Resolution](https://arxiv.org/abs/2403.17000). CVPR 2024.

[14] Yang et al. [Motion-Guided Latent Diffusion for Temporally Consistent Real-World Video Super-Resolution](https://arxiv.org/abs/2312.00853). ECCV 2024.

[15] Yu et al. [VideoGigaGAN: Towards Detail-rich Video Super-Resolution](https://arxiv.org/abs/2404.12388). 2024.

[16] Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.

[17] Salimans and Ho. [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512). ICLR 2022.

[18] Song et al. [Consistency Models](https://arxiv.org/abs/2303.01469). ICML 2023.

[19] Luo et al. [Latent Consistency Models](https://arxiv.org/abs/2310.04378). 2023.

[20] Luo et al. [LCM-LoRA: A Universal Stable-Diffusion Acceleration Module](https://arxiv.org/abs/2311.05556). 2023.

[21] Sauer et al. [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828). 2024.

[22] Yin et al. [Improved Distribution Matching Distillation for Fast Image Synthesis](https://arxiv.org/abs/2405.14867). 2024.

[23] Wang et al. [Phased Consistency Models](https://arxiv.org/abs/2405.18407). 2024.

[24] Zhai et al. [Motion Consistency Model](https://arxiv.org/abs/2406.06890). 2024.

[25] Huang et al. [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009). 2025.

[26] Jiang et al. [D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models](https://arxiv.org/abs/2605.05204). 2026.

[27] Gu et al. [AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation](https://arxiv.org/abs/2605.13724). 2026.

[28] Wu et al. [SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion Distillation](https://arxiv.org/abs/2605.30116). 2026.

[29] Li et al. [One Diffusion Step to Real-World Super-Resolution via Flow Trajectory Distillation](https://arxiv.org/abs/2502.01993). 2025.

[30] ModelTC. [LightX2V Variable Resolution Inference](https://github.com/ModelTC/LightX2V/blob/main/docs/EN/source/method_tutorials/changing_resolution.md) and [Step Distillation](https://github.com/ModelTC/LightX2V/blob/main/docs/EN/source/method_tutorials/step_distill.md). Accessed 2026-07-15.
