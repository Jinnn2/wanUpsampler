# 面向高分辨率视频扩散的轨迹对齐潜空间切换

**Trajectory-Aligned Latent Handoff for Efficient High-Resolution Video Diffusion**

> AAAI-27 中文工作稿，版本 v0.3，2026-07-15。
>
> 本文按 AAAI-27 主赛道 7 页正文的论证密度组织。AAAI 官方模板不支持中文正文，本文件用于完成内容与论证；定稿时翻译到 `main.tex`，并以官方 `aaai2027.sty` 编译匿名投稿版。

## 摘要

视频扩散模型在目标分辨率的潜空间上执行全部去噪步骤，需要较高的计算和显存开销。混合分辨率采样通过在低分辨率形成视频结构与运动、随后切换到高分辨率恢复纹理来降低成本，但现有基于固定插值的切换方式混合了三个连续问题：切换时刻的低分辨率纯净预测尚未到达完整轨迹终点，固定重采样无法刻画视频 VAE 的跨尺度潜变量对应关系，升分结果还可能包含需要高分辨率生成先验修复的高频偏差。本文提出轨迹对齐潜空间切换框架（Trajectory-Aligned Latent Handoff，TALH）。首先，仅在 handoff step 激活 LoRA，使当前低分辨率纯净预测对齐完整低分辨率 teacher 轨迹的最终 endpoint；随后，使用轻量 clean-latent Stage2 将校正后的潜变量映射到高分辨率潜空间；最后，在下一采样时刻重新加噪，并由原视频扩散模型完成高分辨率去噪后缀。Stage2 与 LoRA 均使用冻结生成器自身产生的数据和轨迹训练，不引入外部配对视频、外部超分权重或额外 teacher。由于 LoRA 在 handoff 之前始终关闭，在确定性采样器以及固定提示词、随机种子和调度器的条件下，cached-prefix 训练状态与推理时实际访问的 handoff 状态一致。本文分别在 Wan2.1 50-step 模型和 4-step 蒸馏模型上实现该框架，并将 step40 与 step45 分别作为质量优先和效率优先的工作点。初步 10 样本实验中，4-step handoff LoRA 将低分辨率 endpoint L1 从 0.03207 降低至 0.02858，相对下降 10.88%；完整结论仍需通过 Stage2、端到端生成质量、时序一致性和效率实验验证。

## 1 引言

以 Wan 为代表的潜空间视频扩散模型已经能够生成具有较强语义一致性、运动表现和视觉细节的视频 [Wan Team et al., 2025]。这类模型通常在压缩后的视频潜变量上执行扩散或流匹配采样，但计算量仍会随空间分辨率、帧数和去噪步数快速增长。尤其对于扩散 Transformer，每个去噪步骤都需要处理完整的时空 token 序列；当全部步骤均在目标分辨率上运行时，早期主要用于决定全局布局与粗粒度运动的计算也必须承担高分辨率 token 成本。

一种直接的加速思路是混合分辨率采样：先在低分辨率潜空间中建立主体、布局和运动，再在采样后段将潜变量扩大到目标分辨率，只用少量高分辨率步骤恢复局部细节。LightX2V 的 changing-resolution 推理已经展示了这一工程路径：在指定步骤对当前潜变量进行三线性插值，然后继续执行高分辨率去噪 [ModelTC, 2026]。MrFlow 在图像流匹配模型上进一步表明，低分辨率采样不仅降低单步 token 成本，还可能以更短的低频轨迹形成全局结构；其低通轨迹长度约占完整轨迹的 58%，并且低噪声端的单个高分辨率 refinement step 已能接近多步 refinement [Zheng et al., 2026]。与减少总采样步数的蒸馏方法不同，混合分辨率采样主要改变不同步骤承担的空间计算量，还可以与既有 few-step sampler 组合。

然而，中途切换分辨率并不是一个单纯的张量 resize 操作。当前实现通常先由 denoiser 在切换步骤预测一次纯净潜变量，再将该预测插值到目标尺寸。这里至少存在两类误差。第一，在第 s 步得到的单步纯净预测不等于完整低分辨率轨迹运行至第 T 步得到的最终潜变量；当切换较早时，残余去噪误差会被一并送入升分算子。第二，将同一视频分别缩放到低、高分辨率后再经视频 VAE 编码，所得潜变量并不严格满足尺度等变性。固定插值可以改变网格尺寸，却无法学习 VAE、视频内容与缩放过程共同决定的跨尺度对应关系。

一个看似自然的方案是直接训练网络，将切换时刻的低分辨率纯净预测映射为高分辨率最终潜变量。本文前期的 Stage3 路线即采用这一目标。与相同骨干的 clean-to-clean Stage2 相比，Stage3 在现有 sweep 对比中没有表现出稳定提升，并且更容易产生模糊。该现象与其任务定义一致：Stage3 既要消除随 timestep、scheduler 和 prompt 变化的尾轨迹残差，又要推断低分辨率输入中无法唯一确定的高分辨率细节。在逐点回归目标下，这种条件不确定性容易使输出向中心趋势收缩，表现为高频衰减。因而，简单让 upsampler 见到真实 handoff 输入，并不足以解决 trajectory mismatch。

为此，本文提出轨迹对齐潜空间切换 TALH。其核心思想是先对齐轨迹，再进行升分，最后让高分辨率生成先验修复升分残差。TALH 冻结 Wan 主模型，仅在指定 handoff step 激活低秩适配器，使该步的纯净预测接近相同提示词、种子和初始噪声下完整低分辨率 teacher 轨迹的 endpoint。校正后的潜变量再进入 clean-latent Stage2；Stage2 只学习分别编码的低、高分辨率纯净潜变量之间的映射。得到高分辨率纯净潜变量后，系统在下一调度时刻重新加噪，并使用原 Wan denoiser 完成剩余高分辨率步骤。两个可训练模块由同一个冻结模型提供监督：Stage2 使用模型自身生成的高分辨率视频及其降采样版本构造 latent pair，LoRA 使用完整低分辨率 rollout 的 endpoint 作为 teacher target，从而形成一套模型内生的分辨率与轨迹自蒸馏方案。

本文的贡献可以概括为三点：

1. 将视频扩散的动态分辨率切换分解为切换前的低分辨率尾轨迹误差、跨分辨率 latent lifting 误差，以及切换后的高频 refinement，并分析三者随 handoff step 变化形成的质量--效率权衡。
2. 提出 handoff-localized endpoint distillation：只在切换步骤启用 LoRA，以完整低分辨率 teacher endpoint 监督当前纯净预测；同时给出 cached-prefix 与推理访问状态一致的适用条件和失效边界。
3. 构建不依赖外部配对视频、外部 SR 权重和额外 teacher 的模型内生训练体系，并在标准 50-step Wan2.1 与 4-step distilled Wan 上统一验证 Stage2、LoRA 及其交互作用。

需要强调的是，本文不将 latent 视频超分、PixelShuffle 或 Stage2 骨干作为首创。Stage2 的 encoder-resizer-decoder 结构明显受到 LTX-2 LatentUpsampler 的启发 [HaCohen et al., 2026]。本文的主要创新点是面向同一 Wan diffusion trajectory 的可学习 resolution handoff，以及利用 step-localized LoRA 将轨迹修复与空间升分解耦。

## 2 相关工作

### 2.1 潜空间视频生成与级联升分

Latent Diffusion Models 将生成过程转移到压缩潜空间，为高分辨率图像和视频生成提供了可扩展基础 [Rombach et al., 2022]。Video LDM、Imagen Video 和 LaVie 采用空间或时间级联，从低分辨率生成逐步获得高分辨率视频 [Blattmann et al., 2023; Ho et al., 2022; Wang et al., 2023]。LTX-Video 与 LTX-2 进一步协同设计视频 VAE、潜变量表示与生成 Transformer，其中 LTX-2 公开了独立的 spatial latent upsampler [HaCohen et al., 2024, 2026]。SimpleGVR、LUVE 等近期工作同样研究低分辨率生成器与 latent/video refinement 模块组成的级联系统 [Xie et al., 2025; Zhao et al., 2026]。

上述方法通常把升分器作为生成器之后的独立阶段。本文的 Stage2 则插入同一 Wan 采样轨迹：升分前后的潜变量仍由同一个 Wan denoiser 处理，并通过 re-noise 与后续调度时刻重新连接。因此，本文关注的不只是最终视频升分质量，还包括 handoff 状态是否位于高分辨率去噪器可继续处理的轨迹分布上。

### 2.2 视频超分与时序一致性

Upscale-A-Video、SATeCo、MGLD 与 VideoGigaGAN 分别通过扩散先验、时空适配、运动引导和生成式细节恢复提高视频分辨率并抑制闪烁 [Zhou et al., 2024; Chen et al., 2024; Yang et al., 2024; Xu et al., 2024]。这些方法主要处理已生成或真实低分辨率 RGB 视频到高分辨率 RGB 视频的恢复。TALH 的输入和输出均位于 VAE decode 之前，且后续仍有扩散去噪步骤。因此，RGB VSR 的指标和模型可以作为参考，但不能在潜变量接口、VAE 或输入分布不一致时直接比较权重。

### 2.3 混合分辨率扩散推理

LightX2V changing-resolution 使用固定插值在采样中途改变 latent 尺寸 [ModelTC, 2026]。RALU 指出，训练自由的 latent upsampling 会受到高频混叠与分辨率相关 noise-timestep mismatch 的影响，并针对扩散 Transformer 设计混合分辨率处理 [Jeong et al., 2025]。MrFlow 采用“低分辨率结构生成--像素域 GAN 超分--低强度加噪--高分辨率细节修复”的 staged sampling，在图像模型上获得显著端到端加速，并证明该策略可与 timestep distillation 叠加 [Zheng et al., 2026]。与依赖外部像素域 SR 模型的 training-free 路线不同，TALH 面向视频，在目标生成模型自身的数据与潜空间中学习 resolution lifting，并在切换前显式修复低分辨率尾轨迹误差。

### 2.4 少步蒸馏与 on-policy 轨迹学习

Progressive Distillation、Consistency Models、LCM 和 Distribution Matching Distillation 从轨迹回归、一致性映射与分布匹配等角度减少扩散模型采样步数 [Salimans and Ho, 2022; Song et al., 2023; Luo et al., 2023; Yin et al., 2024]。VideoLCM 与 Motion Consistency Model 将相关思想扩展到视频 [Wang et al., 2023; Zhai et al., 2024]。Self-Forcing、D-OPSD 和 AnyFlow 进一步强调训练状态与 student 推理时实际访问状态的一致性 [Huang et al., 2025; Jiang et al., 2026; Gu et al., 2026]。

本文不重新蒸馏完整 few-step 生成器，而是校正既有 sampler 的一个 handoff evaluation。由于 LoRA 在 handoff 之前关闭，其参数更新不会改变到达该步骤的 prefix；这使当前单步训练与通常的多步 student rollout 存在关键差异。本文在第 3.5 节给出这一结论的条件化说明，并不将当前 cached-prefix endpoint training 表述为完整 on-policy distillation。

## 3 方法

### 3.1 问题定义与总体框架

设冻结的 Wan flow predictor 为 v_phi(x_s, s, c)，其中 x_s 是执行第 s 次 denoiser forward 前的中间潜变量，c 为文本条件，T 为总去噪步数。上标 L 与 H 分别表示低分辨率和高分辨率潜空间。记 z_T^L 为相同 prompt、seed、初始噪声和低分辨率 scheduler 下，base teacher 完整运行 T 步所得的最终低分辨率潜变量；记 (z_0^L, z_0^H) 为同一视频的低、高分辨率版本分别经过 Wan VAE 编码后得到的 clean latent pair。

TALH 学习两个相互独立的模块：handoff step LoRA 参数 Delta theta_s，以及 clean-latent Stage2 算子 U_psi。完整数据流为

\[
x_s^L
\xrightarrow{v_{\phi+\Delta\theta_s}}
\widetilde z_s^L
\xrightarrow{U_\psi}
\widehat z_s^H
\xrightarrow{\text{re-noise}}
x_{s+1}^H.
\]

其中，LoRA 只负责使当前低分辨率纯净预测接近 teacher endpoint；Stage2 只负责 clean latent 的跨分辨率映射。两个模块分别训练，不进行联合反向传播。一方面，这避免了 14B denoiser 与 Stage2 同时驻留训练图带来的显存开销；另一方面，它为两种误差提供了可分离的实验分析。

从系统角度看，handoff step 同时控制三类误差。记切换前的尾轨迹误差为 E_tail(s)，跨分辨率升分误差为 E_lift(s)，剩余高分辨率步骤未能修复的误差为 E_refine(s)，则总误差可以概念性地写为

\[
E_{\mathrm{handoff}}(s)
\approx E_{\mathrm{tail}}(s)
+E_{\mathrm{lift}}(s)
+E_{\mathrm{refine}}(s).
\]

该式不是对非线性网络误差的严格加法分解，而是用于刻画三种相反趋势：切换越晚，当前 clean prediction 越接近完整 LR endpoint，Stage2 输入也越接近 clean 训练域；但可用的 HR suffix 越短，对高频细节和升分残差的修复能力越弱。LoRA 主要降低第一项，Stage2 处理第二项，re-noise 与 HR suffix 处理第三项。

**图 1：TALH 总体流程（待绘制）。** 图中应从左到右展示 LR base prefix、handoff-step base+LoRA、trajectory-aligned clean LR latent、Stage2 latent lifting、re-noise 以及 HR suffix。图下方分别标注 50-step 的 s=40/45、T=50，以及 distilled 4-step 的 s=3、T=4。

### 3.2 模型内生的双重监督数据

TALH 不使用外部配对视频、外部超分模型权重或额外教师模型。Stage2 与 LoRA 的监督均由冻结的目标生成器产生，但对应两种不同关系。

对于 Stage2，本文首先使用一套现有 prompt 和固定 seed 运行 Wan，生成高分辨率 teacher 视频；随后在 RGB 空间对视频做空间降采样，并将高、低分辨率版本分别送入同一个 Wan VAE 编码，得到严格对齐的 clean latent pair (z_0^L,z_0^H)。该构造保持帧数、语义内容与运动轨迹不变，使 Stage2 学习目标模型、目标 VAE 和目标生成分布下的跨尺度对应，而不是通用真实世界退化。

对于 LoRA，本文在相同 prompt、seed、初始噪声和 scheduler 下运行完整低分辨率 teacher trajectory，缓存进入 handoff step 前的 x_s^L，并以完整 LR rollout endpoint z_T^L 作为监督。由此，Stage2 获得跨分辨率表示监督，LoRA 获得跨时间步轨迹监督，两者共同构成模型内生的 resolution-and-trajectory self-distillation。

这两套监督同源但不等价。Stage2 的 LR 输入来自“高分辨率 teacher 视频降采样后再编码”，而 LoRA 输出对齐“原生低分辨率完整 rollout”。二者可能存在分布差异。本文将在相同 Stage2 下比较 downsample-encoded LR latent、native LR endpoint 和 LoRA endpoint，并仅在差异显著时使用 LoRA-produced latent 对 Stage2 做保守微调。

### 3.3 Clean-latent Stage2

Stage2 的输入和输出均为 16 通道 Wan VAE latent，且不改变时间维：

\[
U_\psi:\mathbb{R}^{B\times16\times F\times h\times w}
\rightarrow
\mathbb{R}^{B\times16\times F\times H\times W}.
\]

网络采用受 LTX-2 启发的 encoder-resizer-decoder 结构：首先使用 Conv3D stem 将输入投影到 hidden width 256，随后执行 4 个 3D ResBlock；空间 resampler 位于网络中部，升分后再执行 4 个 3D ResBlock，并通过 Conv3D 输出 16 通道潜变量。3D residual processing 使模型能够在不改变帧数的前提下利用相邻 latent frame 的信息。

本文实现两种空间重采样配置。对于 480x832 到 720x1248，latent 网格从 60x104 变为 90x156。网络先将通道扩展 9 倍，再执行空间 PixelShuffle x3，最后通过固定二项式模糊核以 stride 2 下采样，从而实现 3/2 倍有理数放大：

\[
60\times104 \xrightarrow{\times3} 180\times312
\xrightarrow{/2} 90\times156.
\]

对于 368x640 到 720x1248，latent 网格从 46x80 变为 90x156。由于目标尺寸不是严格 2 倍，网络先执行 PixelShuffle x2 得到 92x160，再中心裁剪至 90x156。这一路径避免先将输入插值到旧 1.5 倍模型要求的网格。

Stage2 的训练目标为

\[
\mathcal L_{\mathrm{S2}}=
\mathcal L_{\mathrm{charb}}
+\lambda_l\mathcal L_{\mathrm{low}}
+\lambda_t\mathcal L_{\mathrm{temp}},
\]

其中

\[
\mathcal L_{\mathrm{charb}}
=\mathbb E\sqrt{(\widehat z_0^H-z_0^H)^2+\epsilon^2},
\]

\[
\mathcal L_{\mathrm{low}}
=\lVert D(\widehat z_0^H)-z_0^L\rVert_1,
\qquad
\mathcal L_{\mathrm{temp}}
=\lVert\Delta_F\widehat z_0^H-\Delta_Fz_0^H\rVert_1.
\]

D 表示将预测的高分辨率 latent 映射回低分辨率网格，Delta_F 表示相邻 latent frame 的一阶差分。当前配置取 lambda_l=0.2、lambda_t=0.1、epsilon=1e-3。三项损失分别约束逐点重建、低频内容保持和时间变化一致性。由于从 LR latent 到 HR latent 并非一一映射，确定性 Stage2 仍可能产生模型特定的系统性升分残差；本文不要求 upsampler 独立完成全部纹理生成，而是利用后续 HR diffusion prior 继续修复这些残差。

### 3.4 Handoff-localized endpoint LoRA

本文冻结 Wan base model，仅对 attention 的 q、k、v、o 投影与 FFN 的 ffn.0、ffn.2 线性层注入 rank-r LoRA：

\[
W'=W+\frac{\alpha}{r}BA.
\]

在 flow parameterization 下，第 s 步的 LoRA 纯净预测为

\[
\widetilde z_s^L
=x_s^L-\sigma_s
v_{\phi+\Delta\theta_s}(x_s^L,s,c).
\]

teacher target 是相同 prompt、seed、LR 初始噪声与 scheduler 下，冻结 base model 完整运行至 T 后得到的 z_T^L。默认 endpoint objective 为

\[
\mathcal L_{\mathrm{end}}
=\lVert\widetilde z_s^L-z_T^L\rVert_1
+0.1\lVert\widetilde z_s^L-z_T^L\rVert_2^2.
\]

对于更早的 step40 handoff，本文还考察权重为 0.05 的 temporal difference loss，以减少相邻帧误差被提前带入 Stage2。代码同时支持 target flow

\[
v^*=\frac{x_s^L-z_T^L}{\sigma_s}
\]

的直接监督，但当前主配置的 velocity loss 权重为 0，因此该目标只作为消融，不作为默认方法。

该 LoRA 的作用不是生成高分辨率纹理，也不是独立减少 sampler 的总 evaluation 数。它将原本可能被 Stage2 被动吸收的 tail-denoising residual 放回 denoiser 侧处理，相当于在 handoff evaluation 中压缩部分 LR 尾轨迹，使 Stage2 的输入更接近其 clean latent 训练域。系统效率来自把更多 denoising evaluations 放在低分辨率执行；LoRA 的直接收益是在固定 handoff step 下改善质量，并且切换越早、尾轨迹 gap 越大，其潜在收益越明显。

### 3.5 Cached-prefix 状态一致性

设 LoRA 只在 handoff step s 生效，并且对所有 k<s 均有 Delta theta_k=0。在确定性 sampler，以及固定 prompt embedding、seed、scheduler 和 CFG 的条件下，adapted process 与 base process 在 handoff 之前使用完全相同的转移函数。因此可以按步骤归纳得到

\[
x_k^{\mathrm{adapted}}=x_k^{\mathrm{base}},
\qquad \forall k\leq s.
\]

于是，预先缓存的 base prefix state x_s^L 就是推理时 step-localized LoRA 实际访问的输入，而不是对 student state 的近似。该结论说明：对于当前单步局部干预，重新执行未改变的 prefix 不会产生新的训练分布。

该性质有明确边界。当 LoRA 在 handoff 之前启用、连续作用于多个步骤、训练和推理使用不同 scheduler/CFG/condition，或 prefix 中含有未对齐的随机操作时，student 到达 handoff 的状态会随参数更新而改变，此时 cached base state 不再等于 student-visited state。多步扩展需要采用 student rollout、same-state teacher prediction 和 EMA teacher 等机制，届时才适合称为 on-policy training。

### 3.6 Resolution lifting、re-noise 与两种实例

LoRA 得到 trajectory-aligned LR prediction 后，Stage2 输出

\[
\widehat z_s^H=U_\psi(\widetilde z_s^L).
\]

若 handoff 不是最后一步，则按照下一时刻的 sigma 和目标分辨率 noise bank 重新加噪：

\[
x_{s+1}^H
=(1-\sigma_{s+1})\widehat z_s^H
+\sigma_{s+1}\epsilon^H.
\]

随后由冻结的 Wan denoiser 在 HR latent 上完成 s+1 到 T 的剩余步骤。直接将 LR flow 插值到 HR 的 resize-flow 策略只作为消融；主方法使用 random 或固定的 target-resolution noise，以避免新空间频率完全由插值决定。

50-step 路线使用 Wan2.1 T2V-1.3B，infer steps=50、sample shift=8、CFG=6，并将 s=40 与 s=45 作为两个不同工作点。step40 保留 10 个 HR steps，现有视觉 sweep 显示其最终高分辨率生成质量更高；但此时当前预测离 teacher50 endpoint 更远，Stage2 输入域差距更大，因此 LoRA 的改善也更明显。step45 只保留 5 个 HR steps，推理更快，当前预测更接近 clean latent，Stage2 本身表现更稳定，而 LoRA 所需承担的校正较小。本文不将二者简化为唯一最优切换步，而是分别作为质量优先和效率优先的 Pareto 工作点。

4-step 路线使用 Wan2.1 T2V-14B StepDistill-CfgDistill，四个 nominal timestep 为 [1000, 750, 500, 250]，sample shift=5，handoff s=3。步骤 1--2 使用 LR base，步骤 3 使用 LR base+LoRA 预测 teacher4-like clean latent，随后 Stage2 升分、在 step4 sigma 重新加噪，并用一个 HR base evaluation 完成生成。

## 4 实验

### 4.1 研究问题

实验围绕以下五个问题展开：

- **RQ1：** learned Stage2 是否优于固定三线性插值？
- **RQ2：** handoff LoRA 是否使当前 LR clean prediction 更接近完整 LR teacher endpoint？
- **RQ3：** endpoint 对齐能否转化为完整 Stage2+HR suffix 的生成质量提升？
- **RQ4：** 该分解能否同时适用于 50-step 与 4-step distilled sampler？
- **RQ5：** handoff step 如何同时改变尾轨迹误差、Stage2 输入域差距、HR refinement 预算和端到端延迟？

### 4.2 数据与训练设置

50-step 路线使用约 1,000 个由 Wan2.1 T2V-1.3B 生成的 teacher 视频，每段 81 帧、720x1248、16 fps。对 HR teacher 视频做 bicubic resize 得到 480x832 或 368x640 版本，并将 LR/HR 视频分别经过 Wan VAE 编码以构造 Stage2 clean latent pair。LoRA 数据记录指定 handoff step 前的 LR state 与相同样本完整 teacher50 endpoint。

4-step 路线使用约 5,000 个由 Wan2.1 T2V-14B StepDistill-CfgDistill 生成的视频。Stage2 训练使用 368x640 与 720x1248 clean latent pair；LoRA 训练使用进入 step3 的 x_pre_step3_lr 与完整 step4 的 z4_lr_teacher。训练、验证与测试应按照 prompt 和 seed 划分，避免同一语义样本跨集合泄漏。

Stage2 使用 AdamW，学习率 1e-4，最多训练 50k steps，精度 bf16，EMA decay 0.9999。LoRA 使用 AdamW，学习率 5e-5，最多训练 10k steps，精度 bf16，默认 rank/alpha=32/32。最终版本还必须补充 GPU 型号与数量、有效 batch size、训练时长、随机种子、checkpoint 选择准则以及 trainer 实际报告的 trainable parameter count。

当前训练视频来自 teacher generator，退化过程只有 resize，不包含真实相机、压缩、运动模糊与噪声退化。因此本文将任务限定为 model-specific generative latent upscaling，而不是通用 real-world VSR。该设置的目标不是引入新的世界知识，而是把冻结 Wan 已有的高分辨率生成能力蒸馏到轻量跨分辨率算子和局部轨迹适配器中。泛化实验将逐步加入未见 prompt、不同运动强度、自然视频 latent、codec degradation 以及跨 Wan checkpoint 测试。

### 4.3 基线与因子实验

本文设置九类核心基线：

1. **Full-HR：** 全部步骤在 720p latent 上执行。
2. **Full-LR：** 全部步骤在低分辨率执行并直接解码。
3. **Interp handoff：** 在 handoff 使用 LightX2V 三线性插值。
4. **Base+Stage2：** 不使用 LoRA，只替换为 learned latent lifting。
5. **LoRA+Interp：** 只验证轨迹修正，不使用 learned Stage2。
6. **LoRA+Stage2：** 完整 TALH。
7. **Teacher endpoint+Stage2：** Stage2 在理想 clean input 下的参考上界。
8. **Direct Stage3：** 直接学习 x0_pred_lr 到 z0_hr 的旧混合目标。
9. **Full-LR endpoint+Stage2+1HR：** 完整运行 LR 轨迹后升分，并以低强度 re-noise 执行一个 HR refinement step；该设置对应 staged sampling 的直接强基线。

核心实验在 step40 和 step45 分别采用 {Base, LoRA} x {Interpolation, Stage2} 的 2x2 因子设计。LoRA 的效果必须在相同空间算子下比较；Stage2 与 interpolation 的效果必须从相同 handoff state 开始。只有这种设计才能分离 LoRA 主效应、Stage2 主效应和两者交互效应，并检验 LoRA 的收益是否随尾轨迹 gap 增大而提高。Direct Stage3 与 Stage2 使用相同骨干、参数量、样本数量、训练步数和 handoff step 比较，避免将模糊归因于不公平训练预算。

### 4.4 指标与公平性

LR endpoint alignment 报告 latent L1/MSE、相邻 latent frame 的 temporal L1、解码后的 LPIPS/PSNR/SSIM 和逐样本 win rate。Stage2 operator 报告 clean latent Charbonnier/L1、解码 PSNR/SSIM/LPIPS、时间差分误差以及高频能量诊断。高频能量不能单独代表质量提升，因为过锐化同样会提高该数值。

完整生成使用 VBench/VBench2 子项、文本视频对齐、temporal LPIPS 或 optical-flow warping error，并进行匿名人工盲测。人工评价至少覆盖细节、artifact、时序稳定性和结构/身份保持。由于生成样本不存在唯一真实 HR ground truth，完整生成不能仅依据 PSNR 排名。所有成对比较共享 prompt、seed、初始 noise、scheduler 和 guidance；人工比较隐藏方法名并随机左右顺序。

效率指标包括单视频 wall-clock latency、峰值显存、LR/HR denoiser step 数量、Stage2 与 LoRA 动态加载开销，以及相对 Full-HR 的端到端加速比。最终结果应报告运行次数、均值以外的离散程度或置信区间，并对关键成对比较使用合适的统计检验。

## 5 结果与分析

### 5.1 Handoff-step sweep 的定性观察

50-step sweep 显示，handoff step 不是单调的质量超参数。step40 和 step45 形成两个具有不同优势的工作点。step40 在切换后保留 10 个 HR steps，能够提供更充分的高分辨率生成与细节修复，当前视觉比较中最终质量更高；但 step40 的单步 clean prediction 离完整 teacher50 endpoint 更远，未经校正时 Stage2 更容易受到残余去噪误差影响。step45 只保留 5 个 HR steps，端到端推理更快，同时其 clean prediction 更接近 Stage2 的 clean-latent 训练域，因此 Stage2 直接输出更稳定，但 HR suffix 的细节修复预算较少。

| 工作点 | LR/HR steps | Tail gap | Stage2 输入 | HR refinement | 当前定位 |
|---|---:|---|---|---|---|
| step40 | 40/10 | 较大 | 离 clean 域较远 | 较充分 | 质量优先，LoRA 收益更明显 |
| step45 | 45/5 | 较小 | 更接近 clean 域 | 较少 | 效率优先，Stage2 更稳定 |

**表 1：step40 与 step45 的定性 Pareto 关系。** 上述观察来自现有 sweep 视频，投稿版本需在独立 validation prompts 上通过 endpoint error、VBench/时序指标、人工盲测和端到端延迟进行量化，并使用独立 test prompts 报告最终结果。

### 5.2 初步 endpoint alignment

目前唯一可以写入正文的量化结果来自 4-step sampler 的 10 样本开发实验。相同 step3 state 上，base clean prediction 到 teacher4 endpoint 的 L1 均值为 0.0320653，LoRA prediction 的 L1 均值为 0.0285758，相对下降 10.88%；LoRA 在 10 个样本上全部取得更低 L1。

| 方法 | Endpoint L1 ↓ | 相对变化 | 逐样本胜出 |
|---|---:|---:|---:|
| Base step3 | 0.03207 | - | 0/10 |
| LoRA step3 | 0.02858 | -10.88% | 10/10 |

**表 2：4-step handoff endpoint alignment 的开发集初步结果。** 该实验只说明局部 LoRA 能够缩小 LR endpoint error，不能证明最终 720p 视频质量已经提升。投稿版本必须固定 checkpoint，并在更大且独立的测试集上重新生成结果、方差与显著性统计。

### 5.3 Stage2 operator 与 Direct Stage3

| 设置 | 方法 | Latent L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Temp. L1 ↓ |
|---|---|---:|---:|---:|---:|---:|
| 480p -> 720p | Trilinear | TBD | TBD | TBD | TBD | TBD |
| 480p -> 720p | Stage2 1.5x | TBD | TBD | TBD | TBD | TBD |
| 368p -> 720p | Trilinear | TBD | TBD | TBD | TBD | TBD |
| 368p -> 720p | Stage2 2x-crop | TBD | TBD | TBD | TBD | TBD |

**表 3：Clean latent operator comparison。** 该表需要回答 Stage2 是否恢复了更接近 HR encoded latent 的结构，而不只是提高锐度或高频能量。除整体指标外，还应报告人脸、文字、重复纹理和快速运动等类别的局部结果。

现有等骨干对比中，Direct Stage3 相对 Stage2 没有观察到稳定提升，并且更容易产生模糊。该结果支持本文的任务纠缠假设，但尚不能仅凭视觉观察形成最终结论。正式实验将比较两者的高频功率谱、temporal LPIPS、清晰度人工偏好和跨 handoff-step 泛化，并检查 Stage3 是否出现训练损失下降但感知细节退化的回归中心化现象。

### 5.4 两个工作点上的完整因子实验

| Step | Handoff | Resizer | VBench ↑ | Anchor LPIPS ↓ | Temp. error ↓ | Human pref. ↑ | Time ↓ |
|---:|---|---|---:|---:|---:|---:|---:|
| 40 | Base | Interpolation | TBD | TBD | TBD | TBD | TBD |
| 40 | Base | Stage2 | TBD | TBD | TBD | TBD | TBD |
| 40 | LoRA | Interpolation | TBD | TBD | TBD | TBD | TBD |
| 40 | LoRA | Stage2 (TALH) | TBD | TBD | TBD | TBD | TBD |
| 45 | Base | Interpolation | TBD | TBD | TBD | TBD | TBD |
| 45 | Base | Stage2 | TBD | TBD | TBD | TBD | TBD |
| 45 | LoRA | Interpolation | TBD | TBD | TBD | TBD | TBD |
| 45 | LoRA | Stage2 (TALH) | TBD | TBD | TBD | TBD | TBD |

**表 4：分离 LoRA、Stage2、handoff step 及其交互作用的核心实验。** 预期但尚待验证的趋势是：step40 的 LoRA 绝对收益大于 step45；step45 的 Base+Stage2 更稳定；完整 TALH 在两个工作点上分别给出质量优先与效率优先结果。本文的主要主张只有在模块收益能够通过 HR suffix 保留，且没有明显负交互或新的输入分布偏移时才得到支持。

### 5.5 质量与效率比较

| 方法 | LR steps | HR steps | 总时间 | 峰值显存 | VBench | 人工偏好 |
|---|---:|---:|---:|---:|---:|---:|
| Full-HR 50-step | 0 | 50 | TBD | TBD | TBD | TBD |
| TALH@40 | 40 | 10 | TBD | TBD | TBD | TBD |
| Interp@45 | 45 | 5 | TBD | TBD | TBD | TBD |
| Direct Stage3@45 | 45 | 5 | TBD | TBD | TBD | TBD |
| TALH@45 | 45 | 5 | TBD | TBD | TBD | TBD |
| Full-LR50+Stage2+1HR | 50 | 1 | TBD | TBD | TBD | TBD |
| Full-HR distill | 0 | 4 | TBD | TBD | TBD | TBD |
| TALH distill | 3 | 1 | TBD | TBD | TBD | TBD |

**表 5：相对 Full-HR、完整 LR staged sampling 与旧 Stage3 的质量--效率 Pareto。** TALH 不减少原 sampler 的总 denoising evaluation 数，因此必须明确报告其收益来自 HR step 比例下降。`Full-LR50+Stage2+1HR` 用于检验：完整运行廉价 LR 轨迹后仅做一次 HR refinement，是否已经能够替代提前 handoff 与 LoRA。Stage2、LoRA、re-noise、VAE 和动态加载开销均计入端到端延迟。

### 5.6 消融实验

主要消融包括：50-step 的 handoff step 30/35/40/45 与 distill 的 step 1/2/3；LoRA 的 qkvo-only 与 qkvo+FFN、rank 8/16/32；endpoint L1、L1+MSE、velocity、endpoint+temporal 等训练目标；cached base prefix、recomputed base prefix 与 LoRA-active student prefix；Stage2 的 trilinear、residual-to-interpolation、direct prediction、1.5x rational resampler、2x-crop、2D per-frame 与 3D temporal-aware 结构；以及 fixed/random HR noise、resized LR flow、共享低频噪声和不 re-noise 等轨迹续接方式。

其中两项消融具有直接的论证价值。第一，在相同样本上分别将 teacher endpoint 与 LoRA prediction 输入 Stage2，可以测量 LoRA 是否引入新的 clean-latent distribution shift。第二，比较 step-only adapter 与 LoRA-active prefix，可以验证 cached-prefix 一致性何时成立、何时必须转向 on-policy rollout。

## 6 讨论与局限性

TALH 的核心假设是：与 timestep 和 scheduler 强相关的误差应由具备语义建模能力的 denoiser 处理，空间升分器应工作在更稳定的 clean latent 域，而升分后仍不确定的高频成分应交回 HR diffusion prior 修复。该三阶段分工能提高可解释性，但模块损失下降并不自动等价于最终视频质量提升。特别是，LoRA 可能在 L1 意义下接近 teacher endpoint，却丢失会影响 Stage2 或 HR suffix 的统计；Stage2 也可能产生锐利但不真实、或跨帧不稳定的细节。因此，两个 handoff 工作点上的端到端因子实验比单独的 endpoint 或重建指标更重要。

Stage2 与 LoRA 还构成一种闭环的模型内生自蒸馏。冻结 Wan 一方面生成 HR 视频及其跨分辨率 latent pair，另一方面提供完整 LR rollout endpoint；student 模块学习的不是额外外部知识，而是如何更低成本地重用 base model 已有的生成能力。该表述并不意味着系统完全不依赖外部知识：预训练 Wan 本身仍包含其原始训练数据和生成先验。本文所强调的是训练 TALH 时不新增外部配对视频、外部 SR 权重或额外 teacher model。

当前方法还存在以下限制。第一，Stage2 骨干主要来自 LTX-2 latent upsampler 的设计启发，架构创新有限。第二，Stage2 可能学习了 Wan VAE、teacher 生成分布和特定 resize recipe 的联合映射，跨 VAE 和跨生成模型迁移能力未知。第三，Stage2 训练使用 downsample-encoded LR latent，而 LoRA 对齐 native LR rollout endpoint，两种模型内生分布仍可能存在差距。第四，LoRA 的 target 是 base teacher endpoint，其性能上限受 teacher 本身质量约束。第五，当前方法针对固定 handoff step，不同步骤通常需要独立数据或 checkpoint。第六，cached-prefix 结论只适用于 handoff 之前模型行为不变的单步局部干预，不能直接推广到多步 LoRA。第七，训练数据由模型生成且只使用 resize degradation，不覆盖真实视频退化。第八，完整生成没有唯一 HR ground truth，必须结合参考、无参考、时间一致性与人工评价。

从应用角度看，混合分辨率采样降低生成成本，有助于减少高分辨率视频生成的计算和能耗；但更低成本也可能扩大合成视频的滥用规模。本文不引入新的训练数据采集或人物识别机制，风险主要继承自 base video generator。最终版本应在 Ethical Statement 中说明模型许可、训练视频来源、生成内容标识与使用限制。

## 7 结论

本文提出面向高分辨率视频扩散的轨迹对齐潜空间切换 TALH。该方法在 handoff step 使用局部 LoRA 将当前低分辨率纯净预测对齐完整 teacher endpoint，再由 clean-latent Stage2 完成低到高分辨率潜变量映射，并通过 re-noise 接回高分辨率生成先验以修复升分残差。Stage2 与 LoRA 分别接受冻结模型自身产生的跨分辨率表示监督和跨步骤轨迹监督，形成不新增外部配对数据、SR 权重或 teacher 的模型内生自蒸馏。step40 与 step45 分别刻画质量优先和效率优先的工作点，4-step 实例则验证该框架与 timestep distillation 的可组合性。现有初步结果支持 LoRA 的 endpoint correction 能力，但论文的最终结论仍取决于 Stage2/Stage3 对照、两个工作点上的完整因子实验、强 staged-sampling 基线、质量--效率 Pareto、泛化测试和人工盲测。

## 参考文献

正式参考文献由 `references.bib` 和 AAAI 2027 BibTeX 样式生成。当前正文引用范围包括：Wan、Latent Diffusion、Video LDM、Imagen Video、LaVie、LTX-Video/LTX-2、SimpleGVR、LUVE、Upscale-A-Video、SATeCo、MGLD、VideoGigaGAN、LightX2V、RALU、MrFlow、LoRA、Progressive Distillation、Consistency Models、LCM、DMD/DMD2、VideoLCM、Motion Consistency Model、Self-Forcing、D-OPSD、AnyFlow、LPIPS 与 VBench。作者、标题、年份和 arXiv/会议元数据已在 BibTeX 文件中统一核验，近期预印本在投稿前仍需检查版本更新。
