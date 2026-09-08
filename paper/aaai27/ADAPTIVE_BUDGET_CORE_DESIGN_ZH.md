# 自适应视频生成预算框架：核心设计工作稿

> 状态：研究设计草案。本文档用于固定当前核心逻辑，不代表已完成的实现或实验结论。

## 一句话定义

我们将高分辨率视频生成表述为一个**证据逐步增强的序贯预算分配问题**：prompt 提供生成前的语义先验，一段低空间成本但保留完整时间轴的生成轨迹提供实例级后验，切换状态进一步决定高分辨率精修的边际投入。

## 1. 问题与核心观察

视频扩散的计算量同时来自去噪步数、空间 token 和时间 token。固定采样策略对所有 prompt 和随机生成实例分配相同计算，忽略了三类差异：

- prompt 在生成前已经描述了对象数量、动作幅度、镜头运动和纹理复杂度；
- prompt 只能提供先验，同一 prompt 在不同 seed 下仍可能产生不同的实际运动与结构；
- 不同低分辨率结果在升至目标分辨率后，需要的高分辨率修复量也不同。

因此，计算策略不应由单一启发式一次决定，而应随着可用证据增加而逐步修正。

## 2. 统一决策目标

设策略 \(\pi\) 控制空间分辨率、时间分辨率、分辨率切换时刻以及高分辨率精修预算。目标为

\[
\pi^*=\arg\max_{\pi}\;\mathbb{E}\left[Q(y,p)-\lambda C(\pi)\right],
\]

其中 \(Q\) 是生成质量，\(C\) 是实测推理成本，\(\lambda\) 是在线输入的质量--效率偏好。模型应预测候选动作的质量或效用，而不是只预测一个含义不明确的“复杂度分数”。

## 3. 三阶段证据更新

### 3.1 生成前：Semantic Budget Prior

输入 prompt、目标分辨率、帧数、基础模型配置和 \(\lambda\)，预测各候选计算策略的先验效用分布。该阶段给出初始预算与保守的 probe 配置，但不做不可撤销的最终决策。

### 3.2 生成中：Trajectory Posterior

先执行一段**低空间分辨率、完整时间轴**的 probe trajectory。完整时间支持用于观察实际运动，避免先降帧再估计运动造成混叠或低估。随后融合 prompt 与跨步 latent 证据，更新各候选策略的效用，并选择后续：

- 空间分辨率或空间 token 密度；
- 时间采样率或时间 token 密度；
- 继续低成本生成或进入高分辨率阶段的时刻。

### 3.3 切换时：Marginal HR Refinement

根据当前低分辨率 clean estimate 分析结构误差与高频缺失，预测每增加一次 HR refinement 的边际质量收益。当预测收益不足以覆盖其成本时停止：

\[
K^*=\min\left\{K:\widehat{\Delta Q}_{K+1}<\lambda\Delta C_{K+1}\right\}.
\]

开发版本首先只把离散 HR 步数 \(K\) 作为精修强度，例如 \(K\in\{0,1,2,5\}\)。暂不同时引入连续 re-noise strength 和区域精修，以控制动作空间和标签成本。

## 4. 方法的统一性

三个阶段不是三个互不相关的 analyzer，而是同一个 action-value 接口在不同观测条件下的更新：

\[
\widehat U_0(a)=f_p(p,a,\lambda),
\]

\[
\widehat U_k(a)=\widehat U_0(a)+f_x(p,\phi(x_{1:k}),a,\lambda),
\]

\[
\widehat{\Delta U}_{K}=f_e(p,\phi(x_{1:k}),z^L,K,\lambda).
\]

这里 \(f_p\) 提供语义先验，\(f_x\) 用实际轨迹进行后验修正，\(f_e\) 估计 HR 精修的边际价值。三个网络是否共享参数是实现问题；论文层面的统一点是它们预测同一质量--成本目标下的反事实动作价值。

## 5. 当前拟定贡献

1. 将视频生成加速统一为从 prompt prior 到 trajectory posterior 的序贯计算分配问题，同时决定样本总预算及其空间、时间和 HR 精修分配。
2. 提出 temporally complete low-resolution probing，在压缩时间轴前先观察实际运动，并以生成轨迹修正 prompt 的不确定性。
3. 提出基于反事实边际收益的 HR refinement stopping，为不同样本动态选择 HR 计算量。

“通用 prompt-budget 分析器”暂不作为核心贡献。只有在跨模型、跨分辨率的留一泛化实验成立后，才使用 universal 或 model-agnostic 表述；否则称为 pretrained 或 transferable prompt prior。

## 6. 与近邻工作的边界

- [AdaDiff](https://arxiv.org/abs/2311.14768) 已根据 prompt 为图像和视频选择样本级去噪步数。
- [DVG](https://arxiv.org/abs/2605.21042) 已根据早期 latent sketch，在固定总预算下选择联合时空压缩策略，是最接近的工作。
- [DLFR-Gen](https://arxiv.org/abs/2504.12259) 已根据生成中 latent 的运动频率动态压缩时间帧率。
- [AViTS](https://arxiv.org/abs/2608.17995) 已联合 text--latent alignment 与跨步 latent variation 指导选择性升分辨率。
- [AdaDiffSR](https://arxiv.org/abs/2410.17752) 已根据 latent 信息增益动态调整图像超分 timestep。

因此，本文不能声称首次使用 prompt、首次进行自适应时空压缩或首次进行 latent-guided refinement。拟保留的区别是：**样本级可变总预算、完整时间轴 probe、跨阶段后验更新，以及统一目标下的末端边际精修决策。**

## 7. 必须锁定的术语与路径

需要在方法实现前确定以下二选一：

- **In-trajectory 路径**：在低分辨率轨迹结束前切换。输入应称为 transition-time clean estimate \(\hat z_{0|s}^{L}\)，不能称为 LR endpoint。该路径与现有 InTraScale 一致，HR refinement 对应剩余 HR suffix。
- **Endpoint-first 路径**：先完成低分辨率轨迹，再从 \(z_{\mathrm{end}}^{L}\) 升至目标分辨率并追加 \(K\) 个 HR 步骤。该路径属于 latent cascade，不再保持固定总去噪步数。

当前建议优先采用 **in-trajectory 路径**，以复用现有 ITU/TTD 和 handoff 体系；完成的 LR endpoint 可作为训练反事实标签，而不是推理时必经状态。

## 8. 最小验证闭环

- 先建立匹配 prompt/seed 的候选动作 oracle，证明样本间最优预算确实不同；
- 比较 prompt-only、latent-only、prompt+latent 与 oracle，报告 utility regret 和 harmful-action rate；
- 在相同端到端成本下比较完整时间轴 probe 与先降帧 probe；
- 比较固定 \(K\)、预测 \(K\) 和 oracle \(K\)；
- 使用 prompt-disjoint train/validation/test、多 seed、held-out \(\lambda\) 和实测 latency；
- 分开报告语义对齐、空间细节、运动幅度、时间一致性、平均延迟和 P90 延迟。

在 oracle headroom、在线轨迹增益和末端精修增益三项未分别成立前，不扩大到复杂联合策略或“通用分析器”声明。

