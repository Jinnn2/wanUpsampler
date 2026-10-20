# 加速质量评测 v2：固定层与动态层

v1 的做法：每个 prompt 手写 4 道题，全部交给 Qwen2.5-VL-7B 打分，P(Yes) 等权平均，每个视频单独评分后再与 FULL 相减。清单在 `prompts/univ_controlled_factor_*_requirements.jsonl`。

v2 把评测分成两层，划分标准是"需不需要读懂 prompt"：

| 层 | 回答的问题 | 适用范围 | 判定方式 |
|---|---|---|---|
| **固定层** | 与原版相比丢了什么 | 任何 prompt、任何视频，只需同 seed 的 FULL 视频 | 信号测量；固定问句的成对 VLM 判定 |
| **动态层** | prompt 的要求还满足吗 | 按 prompt 出题；冻结子集人工审核，新 prompt 自动出题 | 按题型分派：VLM 或测量 |

固定层能脱离 prompt，是因为加速评测天然带一个参考：同 seed 的 FULL 视频已经隐含了"这个视频应该是什么样"。因此固定层的指标在任何 prompt 集上都可比，可以直接用到 VBench 的 prompt 或用户的真实 prompt 上。代价是固定层奖励"像 FULL"：FULL 本身做错的地方，加速视频照抄也不扣分。是否满足 prompt 交给动态层判断。

依据的相关工作：

- DSG、ETVA 的拆题与依赖结构。
- MLLM-as-a-Judge、UVE：成对比较更可靠，大模型优于小模型。
- Time Blindness、MotionInsight：VLM 看不出时间维度的信息。
- MoGAN、SGMD：蒸馏会改变运动量，而 VBench 不够敏感。

## 0. 固定层

### 0.1 信号指标（已实现）

实现：`scripts/data/score_fixed_metrics.py`，运行脚本 `scripts/run_univ_fixed_metrics.sh`。每个视频先单独计算原始量，再与同一来源、同一 prompt、同一 seed 的 FULL 视频配对，得到差值或对数比。

| 维度 | 原始量 | 与 FULL 配对后 |
|---|---|---|
| 运动保真 | 全部 80 个相邻帧对的 RAFT 光流：全局幅度、镜头分量（光流中位数向量）、局部分量（扣除中位数后的残差） | 三者的对数比 `logratio_*`。运动多了、少了都偏离 0：rCM 给静止画面加运动时大于 0，蒸馏让画面变静止时小于 0 |
| 卡顿 | 光流幅度在相邻帧对间的交替程度 `flow_alternation`；停帧比例 `flow_hold_fraction`、`pixel_hold_fraction` | 差值。合成测试中，重复帧视频的交替度为 1.98，平滑运动为 0.18 |
| 闪烁 | 逐帧平均亮度的二阶差分 `flicker` | 差值 |
| 纹理沸腾 | 近静止区域（光流小于 1 像素）做运动补偿后，高频分量的残差占比 `boil` | 差值 |
| 光流中断 | 运动补偿误差的峰值与中位数之比 `warp_spike` | 差值 |
| 细节 | 逐帧 Laplacian 清晰度的均值，以及其随时间的波动 | 差值 |
| 色彩 | 饱和度、亮度、colorfulness、饱和度加权色相 | 差值；色相偏移取圆周距离 |

计算按视频哈希去重：多个结果目录共用的 FULL 视频只算一次。locked test 在计划阶段就被排除，不会被解码。

### 0.2 固定问句（待实现，依赖 VLM 选型）

对每个加速视频都问同样的问题，不读 prompt 的具体内容，并与 FULL 成对判定：

- "Do any objects deform, melt, merge, or appear and disappear abruptly?"（形状完整，单视频判定）
- "Do video A and video B depict the same scene and content?"（内容一致）
- "Given the description, does video B follow it at least as well as video A?"（整体忠实度，只把 prompt 原文作为上下文）

### 0.3 表示类指标（待评估）

V-JEPA 2 的视频表示对运动和物理很敏感，原版 V-JEPA 用预测误差（surprise）在 IntPhys 上零样本检测物理违例达到 98%。计划加入两项：

- 加速视频与 FULL 在 V-JEPA 特征空间的距离。
- 加速视频自身的 surprise。

两项都不依赖 prompt。

## 动态层

以下第 1–6 节定义动态层。动态层的题目由 LLM 按本规范自动生成，经机器校验后即可用于任何新 prompt。进入 benchmark 冻结子集（dev、dev2、test）的清单，还要经过人工审核。

## 1. 题目格式

每个 prompt 一行 JSON，`questions` 中每道题包含：

| 字段 | 含义 |
|---|---|
| `id` | `q0`、`q1`……按顺序编号 |
| `type` | 题型，见第 2 节；判定器只由题型决定 |
| `question` | 英文是非题，回答 Yes 表示要求被满足（否定约束也这样写，如"Is the frame free of human hands?"） |
| `evidence` | prompt 原文中的一段，必须逐字出现，用来追溯题目来源 |
| `depends_on` | 前置题的 id；前置题不满足时，本题计为不满足 |
| `region`、`expect`、`grounding` | 光流类题的测量区域、期望结果和分割用的名词短语 |

出题规则：

- 只问 prompt 明确写出的要求，不问"好不好看"。唯一例外是 `integrity`：运动物体保持形状完整，是隐含要求，`evidence` 指向该物体的名词短语。
- 一题只问一件事。"银色的、占满画面的怀表"要拆成属性题和空间题。
- prompt 里的否定和静止约束必须出题，例如"no other objects""no engraving""nothing moves""lighting never changes"。这类要求是加速方法最容易违反的，现有 benchmark 很少系统地问。
- 只要 prompt 约束了镜头，就必须有 `camera` 题；低运动 prompt 必须有 `static` 题；高运动 prompt 必须有 `motion` 或 `action` 题。以上由校验脚本检查。
- 按单帧判定的题型（entity、count、attribute、detail、spatial）不能问"是否在动"，如"球是否滚过桌面"。单帧看不出运动，这类要求应写成 action 或 motion 题。校验脚本按问句中的运动动词检查。

## 2. 题型与判定器

| 题型 | 问什么 | 判定器 | 是否成对判定 |
|---|---|---|---|
| entity | 主体是否存在 | VLM，原分辨率帧 | 是 |
| count | 数量（一个、几十个） | VLM，原分辨率帧 | 是 |
| attribute | 颜色、材质、形状 | VLM，原分辨率帧 | 是 |
| detail | 细小结构是否清晰可辨 | VLM，原分辨率帧 | 是 |
| spatial | 位置、构图、视角、遮挡 | VLM，原分辨率帧 | 是 |
| negative | 否定约束：无纹理、无其他物体、无文字 | VLM，原分辨率帧 | 是 |
| action | 语义动作：摆动、翻滚、盘旋 | VLM，密集抽帧的视频 | 是 |
| integrity | 运动中形状是否完整、不融化 | VLM，密集抽帧的视频 | 是 |
| motion | 运动幅度和速度（"快速""跨越大半画面"） | RAFT 光流，在 `grounding` 分割出的主体区域上测 | 否，与 FULL 的测量值比较 |
| static | 主体或全局保持静止 | RAFT 光流，主体区域或全局 | 否 |
| camera | 镜头锁定（`expect: still`），或按 prompt 运动，如环绕、横扫（`expect: moving`） | RAFT 光流，背景区域 | 否 |
| focus | 对焦不变 | 逐帧清晰度随时间的变化 | 否 |
| lighting | 光照恒定 | 逐帧亮度随时间的变化 | 否 |

映射写在 `scripts/data/requirements_v2.py` 的 `JUDGE` 表里，题目里不能单独指定判定器。

## 3. 出题流程

1. **起草**：由 LLM 按本规范起草。试点的 16 个 prompt 由 Claude 起草，见 `prompts/univ_requirements_v2_pilot.jsonl`。
2. **机器校验**：
   ```bash
   python UNIV_adaptor/scripts/data/requirements_v2.py validate --checklist prompts/univ_requirements_v2_pilot.jsonl \
       --prompts prompts/univ_controlled_factor_v1.jsonl prompts/univ_controlled_factor_dev2.jsonl
   ```
3. **人工审核**：逐题检查以下几点：是否原子、是否忠于原文、是否漏掉了原文中的可检验要求、依赖关系是否正确。
4. **冻结**：审核通过后冻结。之后不再修改，否则 dev 与 dev2 的结果不可比。

locked test 的清单最后单独起草，在最终评估前冻结。校验脚本会拒绝 test split 的 prompt。

## 4. 判定协议

**VLM 单视频判定**

- 模型暂定为 72B 级（Qwen2.5-VL-72B 或 Qwen3-VL）。7B 保留作为 v1 的基线。
- 帧类题：取 5 帧原分辨率图片，P(Yes) 取平均。
- 视频类题：均匀抽 32 帧，短边 448。帧数和分辨率受视觉 token 数限制；运动快慢和卡顿不靠 VLM 判断，由光流测量。
- 分数取第一个 token 上 Yes 相对 No 的概率。

**VLM 成对判定（与 FULL 比较）**

只对 FULL 满足的要求做：单视频判定下 FULL 的 P(Yes) ≥ 0.5，再由人工标注校准这一阈值。

- 把 FULL（A）和加速视频（B）放进同一个上下文，提问："Video A and video B were generated from the same description. Requirement: {question} Does video B satisfy this requirement at least as well as video A? Answer Yes or No."
- 再交换 A、B 的位置，问 A 是否不差于 B。
- 两个顺序都给出"保持"时，才算保持。

**测量类判定**

- `static` 和 `expect: still` 的 `camera`：对应区域的光流幅度低于阈值。
- `motion` 和 `expect: moving` 的 `camera`：对应区域的光流幅度达到 FULL 同 seed 视频的一定比例。
- 校验脚本拒绝"依据是镜头运动、却期望镜头静止"的 camera 题。全量出题时，模型曾把 p42 的环绕镜头写成"镜头是否锁定"。
- 阈值只在 dev 的 FULL 视频上标定。

## 5. 评分聚合

- **依赖门控**：前置题不满足时，后续题计为不满足。
- **满足度**：每个视频先在每个题型内平均，再在题型间平均。这样题数多的题型不会主导总分。
- **保持率**：只看 FULL 满足的要求，计算加速视频保住的比例。与 v1 一样，按题型、格子分别报告。
- 单视频满足度与成对保持率都报告。论文以成对保持率为主指标，前提是它在人工验证中与人的一致性更高。

## 6. 人工验证

人工标注直接用 v2 的题目，分层抽样方案见阶段性报告。每种判定器与人对比，按题型选定与人最一致的判定器，以人与人的 κ 为上限。基线包括 v1（7B、单视频）和 VideoScore2。

## 7. 待定

- 主体分割：`grounding` 短语交给检测加跟踪模型（如 Grounding DINO 加 SAM2）得到主体掩码。需要确认服务器能否安装。
- 光流阈值的标定方法，以及 `motion` 题"一定比例"的具体取值。
- 72B 模型的显存与吞吐：约 2900 个视频 × 约 9 题 × 5 帧，需要先测速。
- **VideoJev**（南京大学 MCG，[arXiv 2610.18615](https://arxiv.org/abs/2610.18615)，[GitHub](https://github.com/MCG-NJU/VideoJev)）：暂不接入，记录备选。
    - 它用打分头代替逐字生成：视频编码一次后可缓存，多个问题在一次前向中同时打分。
    - QA 准确率与底座相当，速度提升 100 倍以上；有 4B 和 35B-A3B 两个版本。
    - 适合"一个视频回答约 10 道是非题"的动态层：主要价值是成本，不解决准确率问题。它的时间定位能力还可以用来定位"要求在第几秒开始被违反"。
    - 据二手资料，输入为 2 fps、最大 640×640，5 秒视频只有约 10 帧，不能用于运动、卡顿和细节类题。
    - 发布于 2026-10，尚无生成视频评测上的验证。接入前需读原文核实输入规格，并在人工标注集上与 72B VLM 对比。

## 8. 动态层实施计划

五个里程碑。前四个只用已有视频，不重新生成；locked test 的清单和评分放到最后，规则和判定器定稿后只跑一次。

| 里程碑 | 产出 | 服务器开销 | 依赖 |
|---|---|---|---|
| M1 自动出题 | `requirements_v2.py generate`：用开源 LLM 按本规范出题，再经校验和自动修复；产出 dev 与 dev2 共 128 个 prompt 的清单 | 约 10 分钟 | 选定 LLM |
| M2 判定器 | `score_dynamic.py`：VLM 单视频判定、成对判定、测量类判定、报告 | 跑一遍试点 16 个 prompt：约 30 分钟 | 选定 VLM |
| M3 人工标注 | 分层抽样、标注页面、一致性分析 | 标注约 4 小时人工 | M1 冻结、M2 跑通 |
| M4 判定器选型与全量评分 | 每类题选与人最一致的判定器，在 dev 与 dev2 上全量评分，重算路由结论 | 约 3–5 小时 | M3 |
| M5 locked test | test 清单起草、审核、冻结，与最终评估一起运行一次 | 与最终评估一起 | 全部定稿 |

### M1 自动出题

- **出题模型**（已定）：复用 M2 的大 VLM，只输入文本。不再引入单独的 LLM，权重与环境只维护一套，结果也可复现（固定模型版本，贪心解码）。第一个候选是 Qwen3-VL-32B-Instruct，单卡可放下。
- **输入**：本规范第 1、2 节的规则（英文改写，见 `requirements_v2.py` 的 `RULES`），以及试点中 watchmaking 一族的 4 个清单作为示例（覆盖四个格子），再加上待出题的 prompt。
- **输出**：一个 JSON 对象。prompt 的元数据从 prompt 文件取，不信任模型的输出。先交给 `validate` 校验；不通过时，把错误信息附回去重新生成，最多两次。仍不通过的 prompt 列出来交给人工。每次尝试的原始回复都保存在 `raw/<prompt_id>.json`。
- **生成质量检验**：用模型重出试点中其余 3 族的 12 个 prompt（与示例不重叠），以人工审核后的试点清单为参照，计算两项：
    - 召回：参照题中，有同题型且依据片段重叠的生成题的比例。
    - 多余：与参照题都不匹配的生成题，列表交给人判断是否合理。

  召回不到 90% 时，修改出题指令或换模型。
- **运行**：
  ```bash
  bash UNIV_adaptor/scripts/run_univ_requirements_v2.sh setup   # 建环境、下载模型（约 65GB）
  bash UNIV_adaptor/scripts/run_univ_requirements_v2.sh check   # 单卡出 1 题，确认环境
  bash UNIV_adaptor/scripts/run_univ_requirements_v2.sh pilot   # 12 个留出 prompt + 召回报告
  bash UNIV_adaptor/scripts/run_univ_requirements_v2.sh full    # dev + dev2 共 128 个 prompt
  ```
- **冻结子集**：dev 与 dev2 的清单由人审核，修改记入变更日志，然后冻结并记录 sha256。
- **已冻结（2026-10-21）**：
    - 草稿 `prompts/univ_requirements_v2_draft.jsonl`（Qwen3-VL-32B 生成，sha256 `7654534a…`）。
    - 审核修改 `prompts/univ_requirements_v2_edits.json`：删除 9 题、改写 5 题，全部是"用单帧题问运动"的问题。
    - 冻结版 `prompts/univ_requirements_v2_frozen.jsonl`：128 个 prompt，1247 题，sha256 `5f8bb6567d520faa8f67aebde1f0dcd36885991791ca9e87e6d50f8679abcc51`。
    - 由 `requirements_v2.py freeze` 从前两个文件重新生成，结果逐字节一致。
    - 审核范围：一人审核。p0–p6 逐题审核；49 道自动标记的疑点全部检查；其余题目只经过机器校验。M3 的人工标注会顺带检验题目质量，发现问题时只能发新版本（v2.1），不能改动 v2.0。

### M2 判定器

**实现**：`scripts/data/score_dynamic.py`，运行脚本 `scripts/run_univ_dynamic.sh`（`plan pilot` → `check` → `run`）。判定模型先用 Qwen3-VL-32B-Instruct，与出题模型相同。

- 视频题传入 32 帧，同时传入帧号和 fps：Qwen3-VL 据此为每帧标注时间戳，并且不会再按默认的 2 fps 重新抽帧。默认设置会把 32 帧压到总共约 78 万像素，即每帧约 120×210，所以显式放宽了像素上限，每帧保持 448×768。
- 成对判定：两种顺序问的都是加速视频（"B 是否不差于 A"和"A 是否不差于 B"），两次的 P(Yes) 都 ≥ 0.5 才算保持。
- 测量类题的阈值写在 `PROVISIONAL` 里，是 M3 标定前的临时值。原始测量值逐题写入 `dynamic_by_question.csv`，以后改阈值不需要重新运行。

**VLM 单视频判定**

- 帧类题：5 帧原分辨率图片。
- 视频类题：32 帧，短边 448。
- 每题记录 Yes 相对 No 的概率 P(Yes)。
- 推理用 vLLM 并开启前缀缓存，同一视频的多道题共用视觉编码。

**VLM 成对判定**

- 只对 FULL 满足的题做（FULL 的 P(Yes) ≥ 0.5，阈值由 M3 校准）。
- FULL 与加速视频放进同一个上下文，交换左右顺序各问一次。

**测量类判定**

- 直接复用固定层缓存中的逐帧对光流和亮度、清晰度序列，不重复计算。
- 第一版不做主体分割：`static`、`motion` 题的主体区域先用固定层的局部分量（扣除镜头运动后的残差）近似。
- M3 显示这种近似与人不一致时，再引入 Grounding DINO 加 SAM2 做主体分割。

**报告**

逐视频、逐题输出：判定器、P(Yes) 或测量值、成对结果、依赖门控后的最终判定。再按方法 × 格子 × 题型汇总单视频满足度和成对保持率。

### M3 人工标注

- **抽样**：沿用阶段性报告的分层抽样，约 220 组（FULL 与加速视频同 prompt、同 seed），全部来自 dev 和 dev2。
- **标注**：标注者不知道视频来自哪个方法，逐题回答是、否或不确定。动作和卡顿类题另加成对模式。
- **分析**：每类题分别计算判定器与人、人与人的 Cohen's κ；校准 P(Yes) 与测量阈值；用人工标注重算关键结论。
- **事先登记**：抽样清单和分析方案在开标前提交到仓库。

### M4 判定器选型

| 候选 | 角色 |
|---|---|
| Qwen2.5-VL-7B，单视频判定 | v1 基线 |
| Qwen3-VL-8B | 小模型对照 |
| Qwen3-VL-32B | 单卡可部署，8 张卡可并行 8 个实例 |
| Qwen2.5-VL-72B | 大模型，需 2 张卡一个实例 |
| VideoJev | 只在人工标注集上测，不进主流程，见第 7 节 |

- 每类题选与人一致性最高的判定器。一致性相差不大时，选更便宜的。
- 选定后在 dev 与 dev2 上全量评分，重算论证一至三的全部结论。
- Qwen3-VL 的具体型号和权重以 Hugging Face 上实际可下载的为准，下载前需核实。

### 需要决定的事

1. ~~**出题模型**~~：已定为复用大 VLM。
2. **VLM 候选能否下载**：72B 约 150GB，32B 约 65GB。需确认服务器磁盘空间和下载渠道。
3. **标注人力**：至少两人，其中一人重复标注约 25%，用来计算人与人之间的一致性。
