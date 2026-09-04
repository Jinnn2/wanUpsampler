预实验不应直接从五维 Controller 开始，而应按“执行是否正确 → 单模块是否有价值 → 信息能否预测价值 → 联合策略能否兑现收益”四道门推进。

最终建议拆成 12 个预实验。

## 一、统一实验原则

所有预实验共享以下控制条件：

- 相同 prompt、seed、scheduler、目标帧数和目标分辨率；
- 每次只改变被研究模块；
- 同时记录中间状态和最终视频；
- 质量与实际 latency 分开保存，不提前强行合成一个分数；
- 所有比例记录“抽象请求值”和“模型实际映射值”；
- Oracle、learned controller 和真实部署结果严格区分；
- 架构和超参数只在 validation 上选择；
- 最终 test 只用于一次锁定确认。

建议分三级规模：

| 级别 | 规模 | 用途 |
|---|---:|---|
| Smoke | 4–8 prompts × 2 seeds | 检查能否运行、shape 和成本是否正确 |
| Pilot | 24–40 prompts × 2–3 seeds | 判断趋势、筛掉明显无效配置 |
| Oracle | 100–200 prompts | 研究样本间最优策略是否不同 |
| Controller | 约 500 prompts 起 | 训练和验证信息预测能力 |

所有运行应写入统一记录：

```text
prompt / seed / model hash
abstract action
actual shapes / frames / timesteps
probe state
switch state
transition output
final video
quality vector
predicted cost
measured latency
peak VRAM
environment provenance
```

这样后续模块可以复用已有 trajectory，避免重复生成。

---

# 第一阶段：执行基础是否成立

## E0：Model/Scheduler Adapter 一致性

### 目的

确认抽象 schedule 能被具体 backbone 正确执行。这个实验不追求加速收益，只验证框架语义。

### 变量

选择少量典型配置：

- full resolution/full frames/full NFE；
- 低空间分辨率；
- 低 temporal resolution；
- reduced NFE；
- 一次 LR→HR transition；
- 一次 recovery。

### 检查内容

- Prompt embedding 是否与原模型一致；
- 相同 seed 是否可复现；
- shape 和 frame count 是否合法；
- timestep subset 是否严格递减、无重复、无越界；
- `predict_clean` 是否符合模型 prediction type；
- re-noise 后能否继续 scheduler；
- Adapter 包装前后 full-compute 结果是否一致；
- 记录的 NFE、latency、显存是否可信。

### 通过条件

- Adapter 下的 full-compute 输出与原始 pipeline 数值或视觉等价；
- 所有候选动作都能映射到合法配置；
- 不发生隐式增加 NFE、帧数或分辨率；
- 同样配置下运行成本可重复。

如果 E0 没通过，后面所有质量和效率实验都没有意义。

---

## E1：Cost Model 标定

### 目的

建立硬预算需要的真实成本模型。

### 测量网格

对第一个 backbone 测量：

\[
L_m(H,W,F,N,\mathrm{dtype},\mathrm{kernel})
\]

覆盖：

- 4–5 个空间比例；
- 3–4 个合法帧数；
- 多个低分辨率 NFE；
- transition；
- recovery presets；
- decode/encode；
- Controller overhead。

### 实验方法

每个配置：

1. 模型和权重保持 resident；
2. 预热若干次；
3. CUDA synchronize；
4. 重复测量；
5. 保存 median、p90、p95；
6. 单独记录 peak VRAM；
7. 不把模型加载时间混入生成 latency。

### 需要验证

- CostModel 在已测网格上的拟合误差；
- 对未参与拟合的 shape/NFE 组合的预测误差；
- transition 和 recovery 是否能简单相加；
- 不同 shape 下是否存在 kernel 性能断点；
- 理论 token reduction 与真实 latency reduction 的偏差。

### 通过条件

- CostModel 足以稳定判断候选是否满足预算；
- 成本估计误差不会频繁改变候选的可行性；
- 最终硬约束应使用保守上界或安全裕量；
- budget violation 必须结构性接近零，而不是依靠训练学出来。

---

## E2：Fixed Common Probe 预实验

这个实验分两步，不能一次完成。

### E2-A：运行可行性筛选

候选 probe 可以包含：

- 不同空间比例；
- 不同 temporal ratio；
- 2/3/4/6 个 probe NFE；
- 不同 normalized diffusion progress。

固定要求：

\[
a_{\mathrm{probe}}=g(m,B)
\]

同一个模型和预算下，不能依赖 prompt 内容。

检查：

- probe 后是否仍有足够预算完成至少一个合法 schedule；
- early state 是否已经包含可测的空间和时间结构；
- probe 是否过早导致所有 observation 接近随机噪声；
- probe 是否过晚导致节省空间过小；
- 不同 seed 下基本统计是否数值稳定；
- probe 本身占总预算的比例。

E2-A 只筛选出 2–4 个候选 probe，不决定最终最优 probe。

### E2-B：信息有效性选择

必须等单轴 Oracle label 构造完成后再做。

对每个候选 probe，使用完全相同的小型 predictor，测试其预测：

- spatial sensitivity；
- temporal sensitivity；
- NFE sensitivity；
- switch/recovery sensitivity；
- schedule utility ranking。

主指标不是 feature reconstruction，而是：

- validation policy regret；
- pairwise ranking accuracy；
- harmful-action rate；
- 相同成本下的 Oracle-gap closure。

最终 probe 只在 validation 上选择。

### 关键风险

如果 probe 太激进，便宜但没有足够信息；如果太保守，信息充分但已经花掉大量预算。因此目标不是 observation accuracy 最大，而是：

\[
\text{Value of information} - \text{probe cost}
\]

---

## E3：Transition Operator 正确性

### 目的

证明低成本状态能够被可靠地转换为目标高保真状态。

### Latent 路径至少比较

1. 直接插值 noisy latent，作为负面 baseline；
2. predict-clean → resize → scheduler-native re-noise；
3. 不同 coordinate-consistent noise 构造；
4. 可选 learned latent resizer。

### Pixel/SR 路径

只需要验证其作为可插拔路径是否能够正常运行，不需要把“RGB 与 latent 谁更优”设为研究结论。

### 需要两层指标

切换后立即测量：

- clean estimate 偏差；
- latent mean/std；
- 频谱变化；
- temporal difference；
- re-noise 后的噪声统计。

固定 recovery 后测量：

- 最终视频质量；
- 闪烁；
- 结构漂移；
- 语义变化；
- transition latency；
- peak VRAM。

### 控制变量

- 相同 LR trajectory；
- 相同 switch；
- 相同 target shape；
- 相同 recovery；
- 相同 seed/noise-field protocol。

### 通过条件

- transition 后 scheduler 稳定；
- 不产生系统性数值偏移；
- 相比直接插值 noisy latent 有明确优势；
- transition 成本能够被 CostModel 可靠描述。

---

# 第二阶段：五个动作轴分别有没有价值

## E4：Spatial Compression Sweep

### 目标

测量不同内容对空间压缩的敏感程度。

### 控制

固定：

- temporal ratio = 1；
- NFE ratio = 1；
- switch；
- recovery；
- prompt、seed、scheduler。

只改变：

\[
r_s\in\{0.5,0.625,0.75,0.875,1.0\}
\]

### 同时测量两个阶段

切换前：

- LR clean-estimate 与 teacher downsample 的距离；
- 高频结构损失；
- 空间 feature 偏差。

最终视频：

- appearance；
- subject/background consistency；
- text alignment；
-细节和结构质量；
- latency。

### 关键分析

- 每个 prompt 的最佳 \(r_s\)；
- 不同 prompt 的 spatial sensitivity 分布；
- 更强 recovery 能否修复空间压缩损失；
- spatial compression 的真实加速是否符合 \(r_s^2\)；
- 哪些损失是可恢复的，哪些已经破坏语义结构。

### 输出

为每个样本构造：

\[
d_s=\Delta Q/\Delta C_s
\]

它表示额外空间计算对该样本的边际价值。

---

## E5：Temporal Compression Sweep

### 目标

测量不同视频对帧数/时间 token 压缩的敏感程度。

### 控制

固定空间、NFE、switch、recovery，只改变合法 temporal ratio：

\[
r_\tau\in\{0.5,0.67,0.8,1.0\}
\]

实际帧数由 Adapter 映射，并记录实际比例。

### 重点指标

不能只看静态图像质量，应重点测量：

- motion smoothness；
- subject identity across frames；
- camera-motion consistency；
- temporal feature difference；
-运动幅度保持；
- flicker；
- temporal aliasing。

### 关键问题

- 快速运动是否普遍更需要 temporal compute；
- 静态视频是否可以安全压缩帧数；
- temporal compression 造成的运动错误能否被 recovery 修复；
- 简单抽帧、时间插值和模型原生合法帧数映射的差别；
- Prompt 和 early latent 中是否能预测 temporal sensitivity。

### 输出

\[
d_\tau=\Delta Q/\Delta C_\tau
\]

---

## E6：Low-resolution NFE Sweep

### 目标

验证在低分辨率阶段减少 denoising evaluations 的收益。

### 第一部分：先确定 NFE placement

相同 NFE ratio 可能对应不同 timestep subset，因此先比较：

- uniform in normalized sigma；
- early-heavy；
- late-heavy；
- scheduler-native subsampling。

如果 placement 对质量影响很大，那么动作不能只定义为 `r_NFE`，而应定义成 NFE preset：

```text
NFE ratio + placement policy
```

如果一种 placement 在多数条件下稳定最好，可以把 placement 固定在 SchedulerAdapter 中，Controller 只输出比例。

### 第二部分：比例 sweep

\[
r_{\mathrm{NFE}}\in\{0.4,0.55,0.7,0.85,1.0\}
\]

固定：

- spatial；
- temporal；
- switch；
- recovery。

### 必须加入的 baseline

- uniform step skipping；
-相同总 NFE 但不同 LR/HR 分配；
- 少跑 LR steps；
- 少跑 HR steps。

### 关键问题

- 哪些样本对 low-resolution NFE reduction 敏感；
- 减少 NFE 是否主要损害语义、运动还是纹理；
- 空间压缩和 NFE reduction 是否有乘法收益；
- “更低分辨率跑更多步”和“更高分辨率跑更少步”哪个更优。

### 输出

\[
d_{\mathrm{NFE}}=\Delta Q/\Delta C_{\mathrm{NFE}}
\]

---

## E7：Switch Sweep

### 目标

研究 cheap trajectory 应在什么时候切回高保真阶段。

使用 normalized sigma/logSNR，而不是绝对 step：

\[
t_{\mathrm{switch}}\in\{0.80,0.90,1.00\}
\]

### 控制

固定：

- spatial ratio；
- temporal ratio；
- low-resolution NFE policy；
- transition；
- recovery preset。

### 比较

- 最早 switch；
- 中间 switch；
- 最晚 switch；
- Best Global Switch；
- Per-Sample Oracle Switch。

### 关键分析

- 不同 prompt 的最优 switch 是否有明显差异；
- 差异是否跨 seed 稳定；
- Prompt 是否能预测复杂纹理、运动和人物对 switch 的需求；
- early latent 是否能修正 Prompt 的判断；
- switch 是否只是 recovery effort 的替代变量。

### 特别注意

晚 switch 更快，但可能使高频细节无法恢复；早 switch 更稳，但可能没有效率收益。需要同时报告：

\[
Q(t_{\mathrm{switch}}),\quad C(t_{\mathrm{switch}})
\]

而不是只报告“最佳 step accuracy”。

---

## E8：Recovery Effort Sweep

### 目标

先构造合理的恢复 preset，再判断 recovery 是否值得成为可学习动作。

### 第一步：建立恢复 Pareto 集合

对以下变量做小规模组合：

- restart noise level；
- HR NFE；
- timestep placement；
- 可选 guidance 设置。

去掉被支配组合，形成：

```text
R0: minimal
R1: light
R2: standard
R3: aggressive
```

这些 preset 应按实际质量—成本 Pareto 定义，而不是人为规定 1/2/4/6 步。

### 第二步：测试不同损伤等级

Recovery 不应只在一种 LR 配置上测试。至少选择：

- 轻度 spatial/NFE compression；
- 中度 compression；
- 激进 compression；
- 早 switch；
- 晚 switch。

### 必须加入等成本对照

如果 `R3` 比 `R1` 好，可能只是因为它用了更多计算。必须比较相同额外成本用于：

- 更多 LR NFE；
- 更早 switch；
- 更多 HR recovery。

这才能回答：

> 计算应该用于避免损伤，还是用于事后修复？

### 保留条件

只有当 adaptive recovery 相比最佳固定 recovery 和等成本替代方案仍有稳定收益时，才把 \(e_{\mathrm{HR}}\) 保留为最终动作轴。

---

# 第三阶段：联合空间是否真的存在

## E9：Multi-Axis Oracle Schedule Study

这是整个项目最关键的 go/no-go 实验。

### 不运行完整笛卡尔积

五维 action space 可能有 1200 种组合。建议构造有结构的候选集合：

- 单轴 Pareto 配置；
- DVG-like；
- spatial-heavy；
- temporal-heavy；
- NFE-heavy；
- early-switch；
- recovery-heavy；
- balanced；
- strong compositional baselines；
- 少量 fractional-factorial/Latin-hypercube 组合。

每个 prompt 运行约 30–100 个 schedule。

### 至少三个预算

例如：

\[
B/C_{\mathrm{full}}\in\{0.3,0.5,0.7\}
\]

同时保留多个 `lambda`，但 Oracle 分析时应把质量和成本原始值保存下来，避免被单一 `lambda` 定义绑死。

### 比较对象

1. Best Global Schedule；
2. Best Fixed per Budget；
3. DVG；
4. DVG + uniform NFE reduction；
5. DVG + tuned switch；
6. DVG + fixed recovery；
7. Best Compositional Baseline；
8. Per-Sample Sampled Oracle。

这里必须称为“sampled Oracle upper bound”，因为没有穷举全部 1200 个动作。

### 核心分析

- Oracle–Global gap；
- Oracle–DVG gap；
- Oracle–Strong Composition gap；
- 最优动作分布和熵；
- 各 action axis 的使用频率；
- 不同轴之间的 interaction；
- 相同 prompt 跨 seed 的最优动作稳定性；
- near-tie 的数量和幅度。

### Go/no-go 条件

只有同时满足以下条件才进入 Controller 训练：

- 至少多个预算下 Oracle 明显优于强组合 baseline；
- paired prompt-bootstrap 区间支持正向收益；
- 最优动作具有真实多样性；
- 多样性不是由极小质量差或随机 seed 噪声造成；
- 至少前三个轴中存在可预测的 sample-dependent variation。

如果 Oracle gap 接近零，训练更复杂的 Controller 没有意义。

---

# 第四阶段：信息是否能预测 Oracle 差异

## E10：Prompt / Latent Information Study

### 目标

回答 Prompt prior 和 early latent evidence 分别提供了多少可部署信息。

### 模型组

使用相同训练数据和容量控制：

1. Budget only；
2. Prompt + Budget；
3. Latent + Budget；
4. Prompt + Latent + Budget；
5. Prompt-prior + gated latent correction；
6. 可选 schedule-only/context control。

### 不推荐的训练目标

不要把 Oracle Action ID 作为唯一分类标签，因为多个动作可能质量近似。

优先预测：

- 候选质量曲线；
- pairwise action ranking；
- stop/switch advantage；
- per-axis compute demand；
- expected regret。

### 指标

- validation policy regret；
- Oracle-gap closure；
- pairwise ranking accuracy；
- top-k feasible utility；
- harmful-action rate；
- predicted quality calibration；
- 不同预算和 `lambda` 下的稳定性；
- Controller overhead。

### Prompt 实验

测试：

- pooled text embedding；
- token attention pooling；
- 是否需要完整 token sequence；
- Prompt 对不同 action axis 的预测能力。

不额外调用 LLM。

### Latent 实验

从最简单特征开始：

1. mean/std + spatial/temporal statistics；
2. DVG-style frequency/motion statistics；
3. lightweight Conv3D；
4. Conv3D + statistics。

只有简单模型不足时才增加容量。

### 融合实验

验证：

\[
h=h_{\mathrm{prior}}+g(t)\Delta h_{\mathrm{latent}}
\]

是否优于普通 concat，以及不同 probe progress 下 Prompt/latent 的相对作用。

所有模型选择必须只看 validation。

---

## E11：Budget 与 Lambda 行为实验

### Budget 测试

给定预算后：

- 可行集合是否正确；
- 是否保留 minimum future cost；
- measured cost 是否始终不超过预算；
- 很低预算下是否有安全 fallback；
- 预算边界附近是否发生频繁错误切换。

预算违反应该由结构避免，而不是统计意义上的“很少发生”。

### Lambda 测试

固定相同 quality predictions 和 feasible set：

\[
a^*(\lambda)=
\arg\max_a[\hat Q(a)-\lambda C(a)]
\]

随着 `lambda` 增大，选择成本理论上应非递增。如果出现更大的 `lambda` 反而选择更昂贵动作，通常说明：

- Cost normalization 不一致；
- 候选集合发生变化；
- tie-breaking 不稳定；
- 实现存在错误。

需要测试：

- 训练使用的 lambda；
- 未见过的插值 lambda；
- 极端 lambda；
- 相邻 lambda 的动作变化是否平滑；
- Pareto 曲线是否连续覆盖不同质量—效率区域。

---

# 第五阶段：最终 Controller

## E12：Joint Utility Controller

### 训练顺序

不要直接五维训练：

```text
2D: spatial + temporal
            ↓
3D: + low-resolution NFE
            ↓
4D: + switch
            ↓
5D: + recovery
```

每次扩展都必须回答：

> 新增 action axis 是否在 validation 上产生独立的 Pareto 增益？

### 数据规模

第一轮可以从约 500 prompts 开始：

- prompt-disjoint train/validation/test；
- 每个 prompt 一个 full teacher；
- 每个 prompt 8–16 个有结构的候选；
- 对 validation/test 使用额外 seeds 检查稳定性；
- split 在生成数据前冻结。

### Loss

\[
\mathcal L=
\mathcal L_{\mathrm{value}}
+\alpha\mathcal L_{\mathrm{rank}}
+\beta\mathcal L_{\mathrm{demand}}
\]

必要时加入 uncertainty/risk head，但第一版不使用 RL。

### 最终比较

- Best Fixed；
- DVG；
- DVG + uniform skip；
- DVG + fixed recovery；
- Best Compositional Baseline；
- Prompt-only；
- Latent-only；
- Prompt + latent；
- Joint Controller；
- Sampled Oracle。

### 主要成功指标

- 多预算、多 lambda 下的 policy regret；
- Learned Controller 关闭的 Oracle gap 比例；
- Quality–Latency Pareto；
- harmful-action rate；
- prompt-disjoint、seed-disjoint 泛化；
- Controller latency和显存；
- budget violation rate。

Action classification accuracy只能作为辅助结果。

---

# 第六阶段：框架可移植性

## E13：第二 Backbone

这个实验验证框架，而不是比较两个生成模型谁更好。

### Setting A：独立 Controller

- 相同 abstract action schema；
- 相同模块接口；
- 每个 backbone 独立 CostProfile；
- 每个 backbone 独立 Oracle 数据和 Controller。

如果成功，可以声明：

> 相同框架和 Controller architecture 能够通过 Adapter 迁移到不同 backbone。

### Setting B：Shared Controller

再尝试：

\[
h_m=E_m(\text{model capability})
\]

训练 shared controller + model embedding。

只有 Setting B 确实有效，才能声称共享 policy。否则保留 Setting A，不影响模型可插拔框架成立。

---

# 推荐实际执行顺序

```text
E0 Adapter correctness
        ↓
E1 Cost calibration
        ↓
E2-A Fixed probe shortlist
        ↓
E3 Transition correctness
        ↓
E4 Spatial sweep
E5 Temporal sweep
E6 NFE sweep
E7 Switch sweep
E8 Recovery sweep
        ↓
E9 Multi-axis sampled Oracle
        ↓
E2-B Final probe selection
        ↓
E10 Prompt/latent information study
        ↓
E11 Budget/lambda behavior
        ↓
E12 Joint Controller
        ↓
E13 Second-backbone portability
```

其中真正的三个总闸门是：

1. **Transition 能否可靠执行**；
2. **Multi-Axis Oracle 是否明显优于强组合 baseline**；
3. **Prompt + latent 是否能在未见 prompt 上关闭一部分 Oracle gap**。

只有这三点依次成立，完整五维加速框架才具有充分的研究价值。
