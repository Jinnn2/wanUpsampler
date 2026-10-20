# 需求清单 v2：出题规范、判定器分派与评分协议

v1 清单（`prompts/univ_controlled_factor_*_requirements.jsonl`）的做法：每个 prompt 手写 4 道题，全部交给 Qwen2.5-VL-7B 打分，P(Yes) 等权平均，每个视频单独评分后再与 FULL 相减。v2 在四处升级：

1. **出题**：题目原子化，每题对应 prompt 原文的一段，题与题之间有依赖关系。
2. **分派**：按题型交给最合适的判定器，运动类题不再交给 VLM。
3. **成对判定**：VLM 判定的题，同时与同 seed 的 FULL 视频成对比较。
4. **伪影信号**：卡顿、闪烁等不依赖 VLM 的信号，见本文末尾"待定"。

依据的相关工作：DSG、ETVA 的拆题与依赖结构；MLLM-as-a-Judge、UVE 关于成对比较更可靠、大模型优于小模型的结论；Time Blindness、MotionInsight 关于 VLM 看不出时间维度信息的结论。

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
| camera | 镜头锁定 | RAFT 光流，背景区域 | 否 |
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
- 视频类题：至少 16 fps 抽帧，分辨率不低于 480p 短边。
- 分数取第一个 token 上 Yes 相对 No 的概率。

**VLM 成对判定（与 FULL 比较）**

只对 FULL 满足的要求做：单视频判定下 FULL 的 P(Yes) ≥ 0.5，再由人工标注校准这一阈值。

- 把 FULL（A）和加速视频（B）放进同一个上下文，提问："Video A and video B were generated from the same description. Requirement: {question} Does video B satisfy this requirement at least as well as video A? Answer Yes or No."
- 再交换 A、B 的位置，问 A 是否不差于 B。
- 两个顺序都给出"保持"时，才算保持。

**测量类判定**

- `static` 和 `camera`：对应区域的光流幅度低于阈值。
- `motion`：主体区域的光流幅度达到 FULL 同 seed 视频的一定比例。
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
- 伪影信号：重复帧和卡顿、亮度闪烁、高频纹理沸腾。v2a 已证明光流能区分 DUP 和 RIFE，这一层计划作为独立指标，不并入需求分数。
- 72B 模型的显存与吞吐：约 2900 个视频 × 约 9 题 × 5 帧，需要先测速。
