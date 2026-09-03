# Universal Video Generation Acceleration Framework

## 1. 项目定位

本项目从零构建一套**模型可插拔、观测方式可插拔、转换路径可插拔**的视频生成加速框架。框架不预设 RGB observation 或 latent observation 更优，也不把某一种视频生成 backbone 写死在核心算法中。

项目的核心研究问题是：

> 对于一个给定 prompt 和生成过程中的在线状态，有限计算预算应该如何分配到空间分辨率、时间分辨率、低成本阶段 NFE、分辨率切换时机以及高分辨率恢复上？

核心学习组件不是简单的 timestep router，也不是五个独立 heuristic 的拼接，而是一个 **Compute Utility Controller**：它学习当前样本在不同计算维度上的边际计算价值，并对候选生成轨迹进行质量评分。

一句话概括：

> Learn the sample-conditioned marginal utility of computation, then allocate a hard budget jointly across the video generation trajectory.

---

## 2. 主线

整个方法遵循一条统一主线：

```text
Prompt
  ↓
Content-independent fixed common sketch
  ↓
Early online observation (latent / RGB / hybrid)
  ↓
Prompt prior + online evidence + budget + lambda
  ↓
Predict the quality utility of candidate schedules
  ↓
Model legality mask + hard budget mask
  ↓
Choose space/time/NFE/switch/recovery schedule
  ↓
Low-cost generation → transition → HR recovery
  ↓
Video
```

其中：

- **Prompt 是生成前的计算需求先验**：描述运动、纹理、人物、镜头和语义结构等潜在计算需求；
- **在线 observation 是生成过程中的内容证据**：反映当前样本实际形成的空间细节、运动强度和生成难度；
- **预算 `B` 是硬约束**：任何被执行的轨迹都不能突破预算；
- **`lambda` 是软偏好**：在预算以内调节质量和效率的权衡；
- **Controller 预测 schedule utility**：不直接把输入分类成一个固定 Action ID；
- **执行框架负责合法性和可移植性**：模型尺寸、scheduler、帧数和硬件约束不进入通用算法定义。

---

## 3. 为什么采用 fixed common sketch

第一版只进行一次学习型在线决策。在获得 observation 之前，所有相同模型、相同预算档位的样本使用相同的 probe 配置：

\[
a_{\mathrm{probe}}=g(m,B)
\]

`a_probe` 可以依赖：

- 当前生成模型和 scheduler；
- 模型支持的合法空间尺寸和帧数；
- 总预算档位；
- 预先标定的 cost profile。

但它**不能依赖 prompt 或当前内容**。因此，同一个模型和预算条件下，不同样本的 early observation 具有统一、可比较的生成条件。

这带来四个直接好处：

1. Prompt-only、observation-only 和 prompt+observation 的比较是公平的；
2. Oracle trajectory 和训练状态可以稳定缓存；
3. 不会产生“Prompt 先改变 sketch，sketch 再改变 observation 分布”的 credit assignment 问题；
4. 第一版只需学习一次完整 remaining-schedule decision，训练和部署都更简单。

Prompt 的作用没有被削弱。Prompt 仍然作为 prior 参与 joint decision，只是不负责改变第一次 observation 之前的生成条件。

未来只有在 joint controller 已经成立，并且 Oracle 证明自适应初始 sketch 仍有明显收益时，才增加第二个 Prompt Prior 决策：

\[
\pi_{\mathrm{prior}}(p,B,\lambda,m)\rightarrow a_{\mathrm{sketch}}
\]

---

## 4. 统一问题定义

### 4.1 用户请求

一次生成请求定义为：

\[
x=(p,\mathrm{seed},m,H,W,F,B,\lambda)
\]

其中：

- `p`：文本 prompt；
- `seed`：随机种子；
- `m`：选用的视频生成模型；
- `H,W,F`：目标空间尺寸和帧数；
- `B`：硬计算预算；
- `lambda`：质量—效率偏好。

### 4.2 在线状态

在固定 probe 后得到：

\[
s_t=(h_p,o_t,t,B_{\mathrm{remain}},\lambda,h_m)
\]

其中：

- `h_p`：复用生成模型 text encoder 得到的 prompt feature；
- `o_t`：ObservationAdapter 提取的在线 observation feature；
- `t`：归一化 diffusion time、sigma 或 logSNR；
- `B_remaining`：扣除 probe 已用成本后的剩余预算；
- `lambda`：在线质量—效率偏好；
- `h_m`：模型、scheduler 和能力描述。

核心公式不把 `o_t` 写死为 latent：

\[
o_t=O_{\mathrm{obs}}(\mathrm{runtime\ state})
\]

### 4.3 轨迹动作

完整 remaining schedule 定义为：

\[
a=(r_s,r_\tau,r_{\mathrm{NFE}},t_{\mathrm{switch}},e_{\mathrm{HR}})
\]

- `r_s`：空间边长比例，而不是空间 token 比例；
- `r_tau`：时间或帧数压缩比例；
- `r_NFE`：实际执行 NFE 与参考 NFE 的比例；
- `t_switch`：cheap trajectory 结束的归一化噪声位置；
- `e_HR`：高分辨率恢复 preset。

第一版采用离散 action space，由具体 adapter 将抽象比例映射成合法尺寸、帧数和 timestep subset。

### 4.4 硬预算与 lambda

预算和 `lambda` 承担不同职责。

硬预算首先构造可行集合：

\[
\mathcal A_B=\left\{a:\ C_m(a)+C_{\min,\mathrm{future}}\le B_{\mathrm{remain}}\right\}
\]

Controller 对候选动作预测质量：

\[
\hat Q_\theta(s_t,a)
\]

最终决策为：

\[
a^*=\underset{a\in\mathcal A_B}{\arg\max}
\left[
\hat Q_\theta(s_t,a)
-\lambda\frac{C_m(a)}{C_{\mathrm{full}}}
\right]
\]

因此：

- `B` 保证不会超预算；
- `lambda=0` 时，在预算内最大化预测质量；
- `lambda` 增大时，允许主动少用预算以换取更高效率；
- Controller 不需要自己学习如何遵守预算；
- CostModel 与质量预测解耦，因此推理时可以自由改变 `B` 和 `lambda`。

---

## 5. 总体推理 Pipeline

### Stage 0：运行配置

用户选择：

- `GeneratorAdapter`；
- `SchedulerAdapter`；
- `ObservationAdapter`；
- `TransitionAdapter`；
- 目标输出规格；
- 硬预算 `B`；
- 偏好参数 `lambda`；
- 对应模型、硬件和精度的 CostProfile。

Framework Registry 解析配置并检查模块能力是否兼容。

### Stage 1：Prompt encoding

复用生成模型已有的 T5、CLIP 或其他 text encoder：

```text
Prompt tokens
    ↓
Generator text encoder
    ↓
Attention pooling / lightweight projection
    ↓
Prompt feature hp
```

不额外调用 LLM 分析 prompt，以避免新的延迟、依赖和不可控变量。

### Stage 2：Fixed common probe

ProbePolicy 根据模型和预算选择内容无关的 preset：

```text
(model capabilities, scheduler, budget tier)
                      ↓
       fixed probe shape / frames / NFE / stop level
```

随后使用原 prompt 和 seed 运行低成本 sketch，得到统一 probe 位置的 runtime state。

### Stage 3：Online observation

ObservationAdapter 从 runtime state 提取 feature：

```text
Runtime state
    ↓
Latent / Pixel / Hybrid ObservationAdapter
    ↓
Observation feature ho
```

Observation 只负责向 controller 提供证据，不负责执行分辨率转换。

### Stage 4：候选轨迹构造

ActionSpace 枚举离散 schedule，并依次执行：

1. model capability mask；
2. scheduler legality mask；
3. target shape/frame validity mask；
4. hard budget mask；
5. minimum-future-cost reservation。

得到合法候选集合 `A_B`。

### Stage 5：Joint utility decision

Compute Utility Controller 使用：

```text
Prompt feature
Observation feature
Normalized timestep
Remaining budget
Lambda
Model/capability feature
Candidate action feature
```

对每个合法 schedule 预测质量保持程度。Selector 再结合 CostModel 和 `lambda` 选择最终动作。

### Stage 6：Low-fidelity execution

ScheduleExecutor 根据动作执行：

- 空间压缩；
- temporal compression；
- 低成本阶段 NFE reduction；
- scheduler-compatible timestep placement；
- 运行到 `t_switch`。

### Stage 7：Transition

TransitionAdapter 将 cheap state 转换到目标高保真状态。它与 ObservationAdapter 相互独立。

可选路径包括：

- latent clean-estimate resize + scheduler-consistent re-noise；
- pixel/video SR + encode + re-noise；
- 未来其他模型原生 transition。

### Stage 8：Adaptive HR recovery

RecoveryExecutor 根据 `e_HR` 执行高分辨率恢复。`e_HR` 是一个组合 preset，而不是单独的“HR 步数”：

```text
RecoveryPreset
├── restart noise level
├── HR NFE
├── timestep placement
└── optional guidance configuration
```

### Stage 9：Decode 与 accounting

完成视频解码，同时记录：

- 实际 end-to-end latency；
- Transformer latency；
- 实际 NFE；
- transition/recovery/decode 成本；
- peak VRAM；
- Controller overhead；
- 预测成本与实际成本误差；
- budget violation。

---

## 6. 模块设计

### 6.1 Core types

统一核心数据结构：

```python
@dataclass
class GenerationRequest:
    prompt: str
    seed: int
    model_id: str
    output_height: int
    output_width: int
    num_frames: int
    budget: float
    lambda_value: float


@dataclass
class RuntimeState:
    model_state: object
    timestep: object
    normalized_time: float
    spent_cost: float
    prompt_condition: object


@dataclass
class ScheduleAction:
    spatial_ratio: float
    temporal_ratio: float
    nfe_ratio: float
    switch_level: float
    recovery_preset: str
```

### 6.2 GeneratorAdapter

GeneratorAdapter 是自由选择模型的入口：

```python
class VideoGeneratorAdapter:
    def encode_prompt(self, prompt): ...
    def initialize_state(self, seed, shape, condition): ...
    def model_forward(self, state, timestep, condition): ...
    def decode(self, state): ...

    def valid_spatial_shapes(self): ...
    def valid_temporal_shapes(self): ...
    def capabilities(self): ...
```

它屏蔽：

- 不同模型的 forward API；
- latent channel 与 VAE 压缩比例；
- CFG、MoE、单塔或双塔差异；
- 合法空间尺寸和帧数；
- prompt encoder 差异；
- decode/encode 能力。

“模型无关”在第一阶段的含义是：相同框架和 controller architecture 可通过 adapter 接入不同模型。每个 backbone 可以单独校准成本、构造数据和训练 controller。

### 6.3 SchedulerAdapter

```python
class SchedulerAdapter:
    def normalized_time(self, timestep): ...
    def select_nfe_subset(self, nfe_ratio, interval): ...
    def predict_clean(self, state, prediction, timestep): ...
    def renoise(self, clean_state, noise, target_time): ...
    def solver_step(self, state, prediction, t_cur, t_next): ...
```

它统一处理：

- epsilon、v、flow velocity 或 x0 prediction；
- step index 到 normalized sigma/logSNR 的映射；
- NFE ratio 到 timestep subset 的映射；
- scheduler-native predict-clean 与 re-noise。

Controller 不预测具体“跳过第几步”，只预测抽象、可移植的 `r_NFE`。

### 6.4 ObservationAdapter

```python
class ObservationAdapter:
    def extract(self, runtime_state) -> object:
        ...
```

首批实现：

```text
LatentObservationAdapter
PixelObservationAdapter
HybridObservationAdapter
```

Latent option 可以使用：

- adaptive 3D pooling；
- lightweight depthwise Conv3D；
- channel mean/std；
- 高频能量；
- temporal difference；
- predicted-clean statistics；
- 当前 sigma/logSNR。

Pixel option 可以只解码少量低分辨率关键帧，再通过轻量图像或视频 encoder 获取 feature。

本项目不以判断 RGB 和 latent 谁更优为目标。它们是可替换 observation 能力；默认路径可以使用 latent，但不将其声明为普遍最优。

### 6.5 ProbePolicy

ProbePolicy 是确定性、内容无关模块：

```python
class ProbePolicy:
    def resolve(self, model_capability, budget, target_spec): ...
```

输出：

- probe spatial ratio；
- probe temporal ratio；
- probe NFE；
- probe stop level；
- probe 预计成本。

必须保证 probe 后仍存在至少一个可完成生成的合法 future schedule。

### 6.6 ActionSpace

ActionSpace 负责抽象动作枚举与合法映射。初始候选可以采用：

\[
r_s\in\{0.50,0.625,0.75,0.875,1.0\}
\]

\[
r_\tau\in\{0.50,0.67,0.80,1.0\}
\]

\[
r_{\mathrm{NFE}}\in\{0.40,0.55,0.70,0.85,1.0\}
\]

\[
t_{\mathrm{switch}}\in\{0.65,0.50,0.35\}
\]

\[
e_{\mathrm{HR}}\in\{R_0,R_1,R_2,R_3\}
\]

这会产生约 1200 个理论组合，但它们仅用于合法性过滤和低成本评分，不会全部执行。

ActionSpace 必须同时记录：

- 抽象请求比例；
- adapter 映射后的实际尺寸和帧数；
- 实际 timestep subset；
- recovery preset 展开结果；
- 是否发生非精确比例映射。

### 6.7 CostProfiler 与 CostModel

实际 latency 不能简单写成 `H*W*F*NFE`。CostProfiler 针对每个模型、GPU、精度和 kernel 组合测量：

\[
L_m(H,W,F,\mathrm{dtype},\mathrm{kernel})
\]

Schedule cost 定义为：

\[
C_m(a)=\sum_i N_iL_m(H_i,W_i,F_i)
+C_{\mathrm{transition}}
+C_{\mathrm{recovery}}
+C_{\mathrm{decode}}
\]

该子系统包含：

- `Profiler`：运行基准测试；
- `CostTable`：保存测量值和环境 provenance；
- `CostModel`：估算完整 schedule；
- `FeasibleSet`：执行硬预算过滤；
- `RuntimeAccountant`：更新实际已用与剩余预算。

### 6.8 PromptEncoder

PromptEncoder 复用 generator text embedding：

```text
Generator text tokens
        ↓
Attention pooling
        ↓
Two-layer MLP
        ↓
Prompt feature
```

B4 类型的 prompt guidance 在新框架中被解释为**计算需求先验**，而不是一个独立 timestep classifier。

### 6.9 Compute Utility Controller

这是主要训练组件。

输入包括：

- prompt feature；
- observation feature；
- normalized time；
- remaining budget ratio；
- `lambda`；
- model/capability embedding；
- candidate action embedding。

推荐采用 action-conditioned scoring：

```text
Prompt / observation / time / budget / model
                       ↓
                  State query q(s)

Spatial / temporal / NFE / switch / recovery
                       ↓
                 Action embedding ea

Score = q(s)^T ea + InteractionMLP([q(s), ea])
```

Prompt 与 observation 不应只做无结构拼接。推荐：

\[
h_{\mathrm{prior}}=F_p(h_p,h_B,h_\lambda,h_m)
\]

\[
h=h_{\mathrm{prior}}+g(t)\Delta h_o
\]

这表达：

- Prompt 提供稳定 prior；
- observation 提供 online correction；
- correction 强度随生成进度变化。

Controller 主输出为候选质量预测。可选辅助输出包括：

- 五个维度的 marginal compute demand；
- 质量不确定性；
- potential harm/risk。

### 6.10 BudgetConstrainedSelector

Selector 是确定性模块，不是神经网络：

```text
Enumerate schedules
    ↓
Capability mask
    ↓
Scheduler legality mask
    ↓
Hard budget mask
    ↓
Predict candidate quality
    ↓
Apply lambda cost preference
    ↓
Select best schedule
```

### 6.11 ScheduleExecutor

ScheduleExecutor 忠实执行动作，而不自行修改策略：

- 映射并构造实际 spatial shape；
- 映射合法 frame count；
- 构造 NFE subset；
- 运行 cheap phase；
- 在 normalized switch level 停止；
- 调用 TransitionAdapter；
- 调用 RecoveryExecutor。

### 6.12 TransitionAdapter

Observation 和 transition 是两个正交选择。

Latent transition：

```text
Noisy latent zt
    ↓
SchedulerAdapter.predict_clean
    ↓
Clean estimate z0
    ↓
Spatial/temporal resize
    ↓
Coordinate-consistent target noise
    ↓
Scheduler-native re-noise
    ↓
Target-resolution state
```

不直接插值 noisy latent，以免破坏噪声统计。

Pixel/video-SR transition：

```text
Low-resolution state
    ↓
Decode
    ↓
Pixel/video super-resolution
    ↓
Encode target-resolution latent
    ↓
Low-strength noise injection
    ↓
HR refinement state
```

框架可以提供两种实现，但不需要将二者优劣作为论文结论。

### 6.13 RecoveryExecutor

`e_HR` 定义为恢复 preset：

```python
@dataclass
class RecoveryPreset:
    restart_noise_level: float
    hr_nfe: int
    timestep_placement: str
    guidance_config: dict | None
```

Controller 决定当前样本需要的恢复强度，而不是只回归几个 HR steps。

---

## 7. 训练数据与 Oracle

### 7.1 数据定义

对于同一个 prompt、seed 和 backbone：

1. 运行 full-compute teacher；
2. 保存关键 trajectory states；
3. 执行若干 accelerated schedules；
4. 记录最终视频、关键状态、质量向量和实际成本；
5. 在不同 `B` 和 `lambda` 下计算 Oracle schedule。

单条训练数据定义为：

```text
(prompt, seed, model, probe observation, candidate action)
    → quality vector
    → measured cost
    → quality-efficiency utility
```

目标不是学习“什么视频本身更好”，而是学习：

> 对同一个生成器、prompt 和 seed，哪一种加速轨迹最能保留 full-compute generation 的行为和质量？

### 7.2 Teacher trajectory cache

每个 teacher trajectory 保存若干关键状态。候选在 switch point 可以先比较中间 clean estimate 与 teacher state；明显失败的候选可以提前淘汰，只对有潜力的候选完成 transition、recovery 和视频解码。

### 7.3 Candidate sampling

不执行全部笛卡尔积。每个样本选择：

- 一个 DVG-like action；
- 一个 spatial-heavy action；
- 一个 temporal-heavy action；
- 一个 NFE-heavy action；
- 一个 recovery-heavy action；
- 若干随机合法动作；
- 当前 controller 认为最优的动作；
- 当前模型难以区分的 hard negatives。

随着 controller 迭代，逐步采用 active sampling。

### 7.4 Counterfactual compute demand

固定其他因素，只改变一个计算轴：

\[
\Delta Q_s,\Delta Q_\tau,\Delta Q_{\mathrm{NFE}},
\Delta Q_{\mathrm{switch}},\Delta Q_{\mathrm{HR}}
\]

得到：

\[
d=(d_s,d_\tau,d_{\mathrm{NFE}},d_{\mathrm{switch}},d_{\mathrm{HR}})
\]

该向量表示当前样本在不同计算维度上的边际计算需求，可以用于辅助监督和可解释性分析。

### 7.5 训练目标

第一版使用离线监督学习，不使用 RL：

\[
\mathcal L=
\mathcal L_{\mathrm{value}}
+\alpha\mathcal L_{\mathrm{rank}}
+\beta\mathcal L_{\mathrm{demand}}
\]

其中：

- `L_value`：候选质量或质量保持程度的 Huber/MSE regression；
- `L_rank`：同一样本候选动作之间的 pairwise ranking；
- `L_demand`：各计算轴 counterfactual demand 的辅助监督。

Cost 不需要由 Controller 重新学习；它来自 CostModel，并在推理时与任意 `lambda` 组合。

---

## 8. 评价系统

### 8.1 框架正确性

- Adapter conformance；
- 空间尺寸和帧数合法性；
- scheduler timestep 合法性；
- fixed probe 一致性；
- transition 状态和噪声统计检查；
- 同 prompt/seed 的可复现性；
- predicted cost 与 measured cost 偏差；
- budget violation rate。

### 8.2 Baselines

必须包含：

- Full-compute generation；
- Best Global Schedule；
- fixed spatial/temporal/NFE schedule；
- DVG；
- DVG + uniform step skipping；
- DVG + fixed HR recovery；
- DVG + tuned switch；
- Prompt-only Controller；
- Observation-only Controller；
- Prompt + Observation Controller；
- Per-Sample Oracle upper bound。

目标不是只超过原始 DVG，而是超过合理调优后的 strong compositional baseline。

### 8.3 Quality

- VBench total 和分维度结果；
- motion-related quality；
- appearance-related quality；
- text alignment；
- teacher-relative latent/video fidelity；
- 小规模 blind human preference。

### 8.4 Efficiency

- end-to-end latency；
- Transformer latency；
- NFE；
- peak VRAM；
- transition/recovery overhead；
- Controller batch-1 latency；
- speedup versus matched full compute；
- budget violation rate。

### 8.5 核心结果

最重要的结果是：

```text
Quality–Latency Pareto Curve
```

同时报告：

- Best Global Schedule 与 Per-Sample Oracle 的差距；
- DVG 与 Per-Sample Oracle 的差距；
- Learned Controller 关闭了多少 Oracle gap；
- 不同预算和 `lambda` 下的稳定性；
- 不同模型上使用相同框架和 controller architecture 的结果。

---

## 9. 核心消融

### 9.1 信息来源

| Controller input | 30% budget | 50% budget | 70% budget |
|---|---:|---:|---:|
| Budget only |  |  |  |
| Prompt + Budget |  |  |  |
| Observation + Budget |  |  |  |
| Prompt + Observation + Budget |  |  |  |
| Prompt-prior + gated observation correction |  |  |  |

这里的 Observation 可以选择 latent、RGB 或 hybrid，但项目不以比较这些 observation 的普遍优劣为主要结论。

### 9.2 动作轴价值

逐步扩大 action space：

```text
space
space + time
space + time + NFE
space + time + NFE + switch
space + time + NFE + switch + recovery
```

如果某个 axis 没有稳定的独立收益，就不把它保留为最终方法贡献。

### 9.3 Prompt 与 online evidence

分析不同 probe progress 下：

```text
Early:  Prompt prior 更重要
Middle: Prompt 与 observation 互补
Late:   Online observation 更可靠
```

主要指标应为 policy regret 和 Oracle-gap closure，而不只是 Action ID accuracy。

---

## 10. 分阶段研究计划

### P0：Oracle Schedule Study

目标：判断 dynamic joint allocation 是否真实存在。

- 一个 backbone；
- 100–200 prompts；
- 固定 common probe；
- 30–100 个有结构的候选 schedules；
- 30%、50%、70% 至少三个预算；
- 不训练 Controller；
- 比较 Best Global、DVG、strong composition 和 Per-Sample Oracle。

如果相同预算下，大部分样本的最优 schedule 都相同，或者 Oracle 与 strong baseline 的差距接近零，则不继续训练动态 Controller。

### P1：二维 Controller

动作只包含：

\[
(r_s,r_\tau)
\]

比较 DVG heuristic、Prompt-only、observation-only、Prompt+observation 和 Oracle。

### P2：三维 Controller MVP

加入：

\[
(r_s,r_\tau,r_{\mathrm{NFE}})
\]

这是第一版完整核心方法，因为空间、时间和 NFE 共同决定主要计算量：

\[
C\propto r_s^2r_\tau r_{\mathrm{NFE}}
\]

### P3：Adaptive switch

加入归一化 `t_switch`，验证 Prompt 和 online observation 是否能比固定/tuned switch 更好地决定 cheap trajectory 的结束位置。

### P4：Adaptive recovery

最后加入 `e_HR`。只有在独立 Oracle 和 learned-policy 实验中产生稳定 Pareto 增益时才保留。

### P5：第二 backbone

接入第二个模型，验证：

- Adapter 复用；
- 抽象 action 语义一致；
- 实测 CostModel 的必要性；
- 相同 controller architecture 可工作。

第一阶段允许每个 backbone 单独训练 policy。Shared controller + model embedding 只有在实验证明有效后再升级为贡献。

### P6：Prompt-adaptive initial sketch

只有前述系统成立后，才将 fixed common probe 扩展为 prompt-adaptive sketch，形成真正的 two-decision policy。该阶段必须单独证明其增益不是来自额外预算、不同 observation quality 或训练数据分布差异。

---

## 11. 建议目录结构

```text
UNIV_adaptor/
├── README.md
├── core/
│   ├── request.py
│   ├── state.py
│   ├── action.py
│   ├── registry.py
│   └── pipeline.py
├── adapters/
│   ├── generator/
│   │   ├── base.py
│   │   ├── hunyuan_video.py
│   │   └── wan.py
│   ├── scheduler/
│   │   ├── base.py
│   │   └── ...
│   ├── observation/
│   │   ├── base.py
│   │   ├── latent.py
│   │   ├── pixel.py
│   │   └── hybrid.py
│   └── transition/
│       ├── base.py
│       ├── latent_clean_renoise.py
│       └── pixel_sr_renoise.py
├── probe/
│   ├── preset.py
│   └── fixed_policy.py
├── actions/
│   ├── space.py
│   ├── temporal.py
│   ├── nfe.py
│   ├── switch.py
│   └── recovery.py
├── budget/
│   ├── profiler.py
│   ├── cost_table.py
│   ├── cost_model.py
│   ├── feasible_set.py
│   └── accountant.py
├── controller/
│   ├── prompt_encoder.py
│   ├── observation_encoder.py
│   ├── model_encoder.py
│   ├── action_encoder.py
│   ├── fusion.py
│   └── utility_controller.py
├── runtime/
│   ├── probe_executor.py
│   ├── schedule_executor.py
│   └── recovery_executor.py
├── oracle/
│   ├── teacher_cache.py
│   ├── candidate_sampler.py
│   ├── trajectory_metric.py
│   ├── final_metric.py
│   └── pareto.py
├── train/
│   ├── build_dataset.py
│   ├── losses.py
│   ├── train_controller.py
│   └── active_sampling.py
├── eval/
│   ├── adapter_conformance.py
│   ├── latency.py
│   ├── quality.py
│   ├── pareto.py
│   └── ablation.py
└── tests/
```

---

## 12. 推理伪代码

```python
request = GenerationRequest(
    prompt=prompt,
    seed=seed,
    model_id=model_id,
    output_height=height,
    output_width=width,
    num_frames=num_frames,
    budget=budget,
    lambda_value=lambda_value,
)

model = registry.build_generator(request.model_id)
scheduler = registry.build_scheduler(request.model_id)
observer = registry.build_observer(observer_id)
transition = registry.build_transition(transition_id)
cost_model = registry.load_cost_model(model, hardware_profile)

condition = model.encode_prompt(request.prompt)
prompt_feature = prompt_encoder(condition)

# Content-independent probe: depends on model/budget, not prompt content.
probe_action = probe_policy.resolve(
    model_capabilities=model.capabilities(),
    budget=request.budget,
    target_spec=request,
)

state = model.initialize_state(
    seed=request.seed,
    shape=probe_action.initial_shape,
    condition=condition,
)
state = probe_executor.run(
    model=model,
    scheduler=scheduler,
    state=state,
    condition=condition,
    action=probe_action,
)

observation_feature = observer.extract(state)
remaining_budget = request.budget - runtime_accountant.spent_cost

candidates = action_space.enumerate(
    model=model,
    scheduler=scheduler,
    current_state=state,
    target_spec=request,
)

feasible = feasible_set.filter(
    candidates=candidates,
    remaining_budget=remaining_budget,
    cost_model=cost_model,
    reserve_minimum_future_cost=True,
)

quality_scores = controller.predict_quality(
    prompt_feature=prompt_feature,
    observation_feature=observation_feature,
    normalized_time=scheduler.normalized_time(state.timestep),
    remaining_budget=remaining_budget,
    model_feature=model.capabilities(),
    actions=feasible,
)

action = selector.select(
    actions=feasible,
    quality_scores=quality_scores,
    action_costs=cost_model.cost_many(feasible),
    lambda_value=request.lambda_value,
)

state = schedule_executor.run_low_fidelity_phase(
    model=model,
    scheduler=scheduler,
    state=state,
    condition=condition,
    action=action,
)

state = transition.to_target_state(
    model=model,
    scheduler=scheduler,
    state=state,
    target_spec=request,
    recovery=action.recovery_preset,
)

state = recovery_executor.run(
    model=model,
    scheduler=scheduler,
    state=state,
    condition=condition,
    recovery=action.recovery_preset,
)

video = model.decode(state)
runtime_accountant.finalize(video)
```

---

## 13. 实现顺序

建议按照依赖关系实现，而不是先搭建所有模型的“万能框架”：

1. 定义 Core types、Adapter contracts 和 capability schema；
2. 接入第一个 Generator/Scheduler Adapter；
3. 实现 fixed ProbePolicy 和完整 schedule executor；
4. 实现离散 ActionSpace、CostProfiler 和硬预算 FeasibleSet；
5. 实现至少一条可靠 TransitionAdapter 和 RecoveryExecutor；
6. 构造 P0 Oracle Schedule Study，不训练 Controller；
7. Oracle 证明存在 sample-dependent headroom 后，再训练二维/三维 Controller；
8. 接入第二种 ObservationAdapter，验证框架可替换性；
9. 加入 switch 和 recovery action；
10. 接入第二个 backbone；
11. 最后评估 shared controller 和 prompt-adaptive initial sketch。

---

## 14. Claim 边界

本项目应明确区分以下结论。

可以作为目标验证的 claim：

- Prompt prior 与 online evidence 对计算分配具有互补价值；
- 联合计算分配优于固定 schedule 和强组合 baseline；
- 硬预算约束可以结构性满足；
- 相同框架和 controller architecture 可以通过 adapter 支持多个 backbone；
- 可变 `lambda` 能连续控制质量—效率偏好；
- 学习 action utility 比单独设计多个 heuristic 更统一。

在没有对应实验前不能宣称：

- RGB observation 或 latent observation 普遍更优；
- 同一套 controller 权重可以零样本跨所有 backbone；
- 五个 action axis 都有独立贡献；
- FLOPs/token reduction 等价于真实 latency speedup；
- Per-Sample Oracle upper bound 等价于 learned-controller generalization；
- Prompt-adaptive sketch 一定优于 fixed common probe。

---

## 15. 当前版本的最终定义

第一版正式方法定义为：

> 在用户选择的视频生成模型上，先以内容无关的固定低成本 probe 获得统一在线 observation；随后使用 Prompt prior、online evidence 和模型状态预测候选生成轨迹的质量，在硬预算过滤后，通过可调 `lambda` 联合选择空间压缩、时间压缩、低成本 NFE、切换位置和高分辨率恢复强度，并由可插拔执行与转换模块完成视频生成。

第一版最关键的科学验证不是“能否训练一个 Controller”，而是：

> 在相同预算下，不同视频的最优计算分配是否真的显著不同，并且这种差异是否能够由 Prompt prior 与统一 probe 后的在线 evidence 共同预测？

---

## 16. 当前可执行 Pipeline

第一版 Wan2.1 50-step 生成链已经实现在本目录中。它支持：

- LR 空间比例、时间比例和精确 full-DiT NFE 比例；
- `0.6 / 0.8 / 1.0` 三个 reference-trajectory 切换点；
- 未重计算 LR step 的 residual cache reuse；
- 两个互斥 transition baseline：严格按 DVG 式 (11)-(12) 在 latent T/H/W
  上恢复的 `dvg_latent_anchor`，以及旧 Stage 2 的 Wan VAE decode、
  Real-ESRGAN x2、像素域时间插值、Wan VAE encode 的 `rgb_sr_vae`；
- coordinate-hash HR 重加噪；
- 清空 LR cache/solver history 后的 full-compute HR suffix；
- Wan flow clean/re-noise 解析公式测试；
- mean/std、频谱、temporal difference，以及可选 native HR state distance 诊断；
- 每次运行的 action、实际 shape、step mask、boundary sigma 和阶段耗时 sidecar。

具体运行契约、配置和命令见 [`PIPELINE.md`](PIPELINE.md)。

## 17. Controller 数据生成协议

Prompt + common-probe latent controller 的数据契约、稀疏反事实采样、
sampled-Oracle split、质量/成本标签和恢复门禁见
[`DATA_GENERATION.md`](DATA_GENERATION.md)。当前 fixed-action runner 不能生成
训练所需的同源 common-probe 分叉数据，因此协议入口暂时只开放 immutable
`plan/check`；common-probe branch runner 完成后才会开放正式采集。
