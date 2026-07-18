# TALH 论文图片规划（AAAI-27）

本文档只定义图片的论证任务、接入位置、版式和数据来源，不生成具体图像或绘图脚本。规划遵循 AAAI 双栏版式：单栏宽度 3.3 in，跨双栏宽度 7.0 in；数据图最终导出 PDF 矢量文件，视频帧拼图导出 300 DPI 以上 PNG/PDF。

## 1. 图片承担的论证链

论文的三项贡献需要分别得到视觉支撑：

1. **系统贡献：** TALH 将大部分去噪计算转移到低分辨率潜空间，并形成 TALH-Q/TALH-E 两个质量—效率工作点。
2. **方法贡献：** 动态分辨率切换被分解为 TAA 轨迹对齐、CLL 跨分辨率提升和 HTR 高分辨率轨迹重入。
3. **训练贡献：** TAA 与 CLL 共享冻结生成模型提供的模型内生监督，不依赖外部配对视频、外部超分权重或额外教师模型。

主文图片应按“方法总览 → 系统收益 → 模块机制 → 视觉结果”的顺序展开。推荐主文保留 4 张图，补充材料保留 4 张图。

## 2. 主文图片总览

| 编号 | 暂定名称 | 核心问题 | 接入位置 | 版式 | 类型 | 优先级 |
|---|---|---|---|---|---|---|
| 图 1 | TALH 总体框架与模型内生训练 | TALH 如何工作，三个模块如何对应三类误差，训练监督从何而来？ | 引言贡献点之后，首次引用放在引言方法概述段 | 跨双栏，约 7.0×3.0 in | 架构图 | P0 |
| 图 2 | 质量—效率 Pareto 与质量维度变化 | TALH 相对 Native-HR 是否真正加速，质量代价集中在哪里？ | 5.1 节首段之后、表 1 附近 | 跨双栏，约 7.0×2.4 in | 数据图 | P0 |
| 图 3 | TALH 机制证据 | CLL、TAA 是否分别完成预期功能，二者组合后如何作用？ | 5.4 节因子分析之后 | 跨双栏，约 7.0×2.7 in | 三联数据图 | P0 |
| 图 4 | 同提示同种子的定性比较 | 插值、CLL、TAA 和高分辨率后缀在视觉上分别带来什么？ | 5.5 节人工盲评之后，或 5.7 节之后 | 跨双栏，约 7.0×3.5–4.0 in | 视频帧拼图 | P0 |

## 3. 图 1：TALH 总体框架与模型内生训练

### 3.1 接入点

- **正文源位置：** 引言中“TALH 将动态分辨率切换分解为 TAA、CLL 和 HTR”段落之后、三项贡献之前或之后。
- **正文引用：** 引言首次介绍 TALH 时加入“如图 1 所示”；第 3.1、3.2 和 3.6 节分别回指图 1 的推理、训练和重入部分。
- **LaTeX 形式：** `figure*`，页顶浮动。

### 3.2 论证任务

一张图同时回答三件事：

1. TALH 为什么能加速：低分辨率前缀替代大部分高分辨率去噪计算。
2. TALH 为什么需要三个模块：轨迹差距、跨尺度误差和高分辨率残差由不同环节处理。
3. TALH 如何训练：两类训练对均由同一个冻结 Wan 模型产生。

### 3.3 推荐布局

采用两条水平带状区域，整体从左向右阅读。

#### 上半部分：推理与工作点

1. **采样时间轴：**
   - Native-HR：50 个高分辨率步骤，全部使用暖红/深灰块。
   - TALH-Q：40 个低分辨率步骤 + 10 个高分辨率步骤，在第 40 步标出切换点。
   - TALH-E：45 个低分辨率步骤 + 5 个高分辨率步骤，在第 45 步标出切换点。
   - 时间轴上方以“Structure & Motion → Texture & Detail”标注生成职责的阶段性变化。

2. **切换流水线：**
   - `LR Prefix` → `TAA at step s` → `Aligned Clean LR Latent` → `CLL` → `Lifted Clean HR Latent` → `HTR` → `HR Suffix`。
   - TAA 框内标注 `LoRA, base model frozen`。
   - CLL 框内标注 `Clean LR → Clean HR`。
   - HTR 框内标注 `Target-resolution re-noise`。
   - 在三个位置分别标出 `E_traj(s)`、`E_lift(s)` 和 `E_refine(s)`；误差标记使用细虚线，不使用大面积警示色。

#### 下半部分：模型内生监督

1. **Trajectory Alignment Pairs：**
   - 冻结 Wan 低分辨率 rollout 产生缓存状态 `x_s^L` 和完整轨迹终点 `z_T^L`。
   - 两者形成 TAA 训练对。
   - 在 `x_s^L` 旁标注 `cached prefix = inference state`，并用小型条件标签注明 `same prompt / seed / scheduler / CFG`。

2. **Cross-Resolution Lifting Pairs：**
   - 冻结 Wan 生成 HR 视频。
   - HR 视频一支直接经过 Wan VAE 得到 `z_0^H`；另一支先在 RGB 空间降采样，再经同一 Wan VAE 得到 `z_0^L`。
   - 两者形成 CLL 训练对。

3. **共同边界：**
   - 用一个外框标注 `Model-Internal Supervision`。
   - 在外框底部以短句标注 `No external paired videos, SR weights, or extra teacher`。

### 3.4 视觉规范

- **工作流：** academic-plotting Workflow 1。
- **风格：** Classic Accent Bar，适合 AAAI 正式技术论文，并可在灰度打印中保持层次。
- **颜色语义：**
  - 低分辨率计算：蓝色 `#4A90D9`。
  - TAA：紫红色 `#CC79A7`。
  - CLL：绿色 `#009E73`。
  - HTR 与高分辨率后缀：朱红色 `#D55E00`。
  - 冻结 Wan、VAE 和基线：灰色 `#7B8794`。
- **禁止项：** 不放网络结构小图、真实视频帧、阴影、渐变和装饰性图标；图 1 的目标是解释系统关系，而不是展示全部实现细节。

### 3.5 图注草案

> **TALH 的推理流程与模型内生训练。** TALH 将大部分去噪计算置于低分辨率潜空间，并在切换步骤依次执行 TAA 轨迹对齐、CLL 跨分辨率提升和 HTR 高分辨率轨迹重入。TALH-Q 与 TALH-E 分别保留 10 个和 5 个高分辨率步骤。TAA 与 CLL 的监督均由冻结 Wan 生成，分别对应轨迹对齐对和跨分辨率提升对。

## 4. 图 2：质量—效率 Pareto 与质量维度变化

### 4.1 接入点

- **正文源位置：** 5.1 节首段之后，表 1 之前或之后。
- **推荐关系：** 图 2 负责展示趋势，表 1 保留精确延迟、加速比和 VBench-5 数值；正文避免重复逐项朗读两者。
- **LaTeX 形式：** `figure*`。如果版面紧张，可只保留左侧 Pareto 图并缩为单栏 `figure`。

### 4.2 面板设计

#### (a) 质量—延迟 Pareto

- 横轴：单视频端到端冷启动延迟（秒，越低越好）。
- 纵轴：VBench-5（越高越好）。
- 四个点：
  - Native-HR Sampling：253.10 s，0.82836。
  - TALH-Q：138.36 s，0.80983。
  - TALH-E：114.26 s，0.80792。
  - Endpoint Re-entry Baseline：86.45 s，0.80093。
- 在 TALH-Q/TALH-E 旁直接标注 `1.83×` 和 `2.22×`，避免额外图例解释。
- 用浅灰虚线连接工作点，表现速度—质量前沿；不要用平滑拟合曲线，因为只有四个离散配置。
- 点形同时编码方法：Native-HR 为方形，TALH-Q 为圆形，TALH-E 为三角形，Endpoint Re-entry 为菱形，确保灰度可辨。

#### (b) 相对 Native-HR 的五维 VBench 变化

- 横轴：Subject、Background、Motion、Aesthetic、Imaging。
- 纵轴：相对 Native-HR 的绝对分数变化，零线居中。
- 两组柱或两条带标记折线：TALH-Q 与 TALH-E。
- 该面板应客观显示质量变化主要集中在主体一致性和成像质量，而运动平滑度变化较小。
- 不画雷达图：差值柱状图更容易比较方向和幅度，也不会夸大面积差异。

### 4.3 数据来源

- `results/integrated_20260718/compiled_tables/quality_efficiency_summary.csv`
- `results/integrated_20260718/compiled_tables/wan50_final_quality_efficiency.csv`
- 如需配对差值置信区间：`wan50_final_vbench_paired_statistics.csv`

### 4.4 视觉规范

- **工作流：** academic-plotting Workflow 2，matplotlib。
- **输出：** PDF 矢量图 + 300 DPI PNG 预览。
- **颜色：** Native-HR 与 Endpoint Re-entry 使用不同深浅灰；TALH-Q 使用蓝色 `#0072B2`，TALH-E 使用橙色 `#E69F00`。
- **字体：** Times，坐标标题 9–10 pt，刻度与标注不低于 7 pt。

### 4.5 图注草案

> **TALH 相对 Native-HR Sampling 的质量—效率权衡。** (a) TALH-Q 和 TALH-E 分别将端到端延迟降至 138.36 s 和 114.26 s，形成质量优先与效率优先工作点；单步高分辨率重入可进一步提速，但质量下降更大。(b) 相对 Native-HR 的逐维 VBench 差值，展示两个 TALH 工作点的质量变化来源。

## 5. 图 3：TALH 机制证据

### 5.1 接入点

- **正文源位置：** 5.4 节因子分析结果之后。
- **正文引用：** 5.2 节引用面板 (a)，5.3 节引用面板 (b)，5.4 节引用面板 (c)。跨双栏图可在 5.4 节后统一浮动到页顶。
- **表格调整建议：** 主文可保留表 4 的精确因子结果；表 2 和表 3 的完整指标移至补充材料，正文保留关键数字和图 3。

### 5.2 面板设计

#### (a) CLL 的跨分辨率提升效果

- 使用相对误差下降率，避免把不同量纲直接放在同一坐标轴。
- 横轴指标：Latent L1、LPIPS、Temporal L1。
- 两组柱：480×832→720×1248 与 368×640→720×1248。
- 数值：
  - 480 路线：48.60%、73.30%、30.81%。
  - 368 路线：33.36%、51.07%、15.54%。
- 所有柱均表示“相对三线性插值的误差下降”，方向统一为越高越好。

#### (b) TAA 的轨迹终点对齐

- 使用哑铃图或成对柱状图，展示 Unaligned 与 TAA-Aligned 的终点 L1。
- 三组配置：Wan50@40、Wan50@45、Distill4@3-of-4。
- 数值：
  - @40：0.03215 → 0.02385（-25.82%）。
  - @45：0.02363 → 0.01866（-21.03%）。
  - Distill4：0.04286 → 0.04070（-5.03%）。
- 在连接线末端标注相对下降率；如空间允许，用细误差线表示改善量的 95% bootstrap 置信区间。

#### (c) 2×2 因子交互

- 采用 3×4 热力图，行分别为 Wan50@40、Wan50@45、Distill4@3-of-4。
- 列分别为：
  - Unaligned + Trilinear。
  - TAA-Aligned + Trilinear。
  - Unaligned + CLL。
  - TAA-Aligned + CLL（TALH）。
- 单元格显示相对本行 `Unaligned + Trilinear` 的 VBench-5 增量，而不是原始分数。
- 使用从白色到蓝色的单向色图；负增量以浅橙描边或负号表示，不使用红绿对立色。
- 该面板的主要视觉结论应是：CLL 提供主要端到端增益，TAA 的聚合 VBench 增量较小，但其轨迹对齐作用在面板 (b) 中独立成立。

### 5.3 数据来源

- 面板 (a)：`operator_480p.csv`、`operator_368p.csv`。
- 面板 (b)：`wan50_step40_endpoint_paired_statistics.csv`、`wan50_step45_final_endpoint_paired_statistics.csv`、`distill_transfer_paired_statistics.csv`。
- 面板 (c)：`vbench_case_summary.csv`、`factorial_vbench_effects.csv`。

### 5.4 视觉规范

- **工作流：** academic-plotting Workflow 2，matplotlib/seaborn。
- **颜色：** CLL 使用绿色系，TAA 使用紫红色系，未对齐/插值基线使用灰色；图 1 到图 3 保持同一模块颜色语义。
- **面板标签：** `(a) CLL Lifting`、`(b) TAA Alignment`、`(c) Factorial Interaction`。
- **限制：** 不使用双 y 轴，不将 PSNR、LPIPS 和 L1 原始值强行画在同一尺度，不使用雷达图。

### 5.5 图注草案

> **TALH 的模块作用与交互。** (a) CLL 在两种缩放比例上均显著降低三线性插值的潜变量、感知与时间误差。(b) TAA 缩小第 40 步、第 45 步及 4 步蒸馏模型的低分辨率轨迹终点差距。(c) 2×2 因子结果显示，CLL 提供主要端到端 VBench-5 增益，而 TAA 主要承担局部轨迹校正。

## 6. 图 4：同提示同种子的定性比较

### 6.1 接入点

- **正文源位置：** 5.5 节人工盲评之后；若版面更适合统一展示，可放在 5.7 节末尾。
- **LaTeX 形式：** `figure*`，跨双栏。
- **目标：** 将表格中的“CLL 改善清晰度、TAA 存在细节—时序权衡、TALH-Q/E 对应不同后缀预算”转化为直观视觉证据。

### 6.2 推荐内容

推荐采用一个跨双栏、左右双面板的真实视频帧图，而不是仅放一组 Native-HR/TALH 单帧 pair。

#### (a) 空间细节与模块作用

- 使用一个细节丰富的 TALH-Q（第 40 步切换）候选提示。
- 四行依次为 Native-HR、Trilinear@40、CLL-only@40、TALH-Q。
- 每行展示同一帧的完整画面，并附两个相同坐标的 2× 或 3× 局部裁剪。
- 该面板同时展示系统参考、固定插值、CLL 主效应和完整 TALH，避免只比较最终方法而无法判断改进来源。

#### (b) 连续帧与时序表现

- 使用一个运动明显的 TALH-E（第 45 步切换）候选提示。
- 四行依次为 Native-HR、Trilinear@45、CLL-only@45、TALH-E。
- 每行先放一个完整上下文帧，再展示同一局部区域的 5 帧短序列，例如 `t-4, t-2, t, t+2, t+4`。
- 该面板用于观察主体形状、纹理附着和局部闪烁，真实呈现 TAA 的细节—时序权衡。

裁剪区域优先覆盖：

- 人脸、动物头部或稳定身份区域。
- 细密材质、边缘、文字或小物体。
- 运动过程中容易闪烁或形变的区域。

两个面板使用不同提示，但各自内部必须共享文本提示、随机种子、调度器、帧索引和裁剪坐标。图中不绘制相对 Native-HR 的像素误差热力图，因为不同分辨率轨迹的生成结果并非像素级真值对齐。

### 6.3 样本选择规则

- 候选范围限定为主评测的 10 个匹配提示，不从额外未评测样本中挑选。
- 在查看方法差异前，先按语义属性选择“细节丰富”和“明显运动”两个提示，或使用 Native-HR 指标的中位样本。
- 不只展示 TALH 最优样本；补充材料同时给出失败案例。
- 所有方法使用相同提示、随机种子和帧索引，裁剪框坐标保持一致。
- 原始帧优先使用无损 PNG，避免 JPEG 压缩掩盖高频差异。

### 6.4 视觉规范

- 方法名置于每行左侧，帧索引置于每列上方。
- 放大区域使用细实线框与同色连接线；不使用粗箭头遮挡内容。
- 每张小图之间留 1–2 mm 白边，禁止阴影和圆角卡片效果。
- Native-HR 不使用强调色；TALH-Q/TALH-E 的行标签分别使用与图 2 一致的蓝色和橙色。

### 6.5 图注草案

> **相同提示与随机种子下的真实视频帧比较。** (a) 第 40 步切换的空间细节对比，完整帧与放大区域展示三线性插值、CLL 和 TALH-Q 的差异。(b) 第 45 步切换的连续局部帧，展示 CLL-only 与 TALH-E 的细节和时序表现。每个面板内的所有方法使用相同调度器、帧索引和裁剪位置。

### 6.6 需要收集的真实视频组

与其寻找孤立的两两 pair，优先提供同一提示和种子下的四路视频组。最低需要以下两组：

#### 组 A：TALH-Q 空间细节

1. `full_hr50`：Native-HR Sampling。
2. `step40_base_interp`：Unaligned + Trilinear。
3. `step40_base_stage2`：Unaligned + CLL。
4. `step40_lora_s0p75_stage2` 或最终 `talh40`：TALH-Q。

#### 组 B：TALH-E 运动与时序

1. `full_hr50`：Native-HR Sampling。
2. `step45_base_interp`：Unaligned + Trilinear。
3. `step45_base_stage2`：Unaligned + CLL。
4. `step45_lora_stage2` 或最终 `talh45`：TALH-E，TAA strength=0.75。

每组最好提供 2–3 个候选提示，便于按照预先声明的选择规则确定最终示例。若只能先找到两两 pair，优先级如下：

1. `Trilinear` vs `CLL-only`：展示 CLL 的空间提升作用。
2. `CLL-only` vs `TALH`：展示 TAA 的局部作用与时序权衡。
3. `Native-HR` vs `TALH-Q/E`：展示完整系统的最终视觉接近程度。

另外提供一组 TALH 表现不理想的真实失败案例，用于补充材料图 S4。

### 6.7 视频交付格式

推荐按以下结构整理，文件名可以不同，但方法映射必须明确：

```text
qualitative_candidates/
├── group_a_step40_<prompt_id>_<seed>/
│   ├── native_hr.mp4
│   ├── trilinear.mp4
│   ├── cll_only.mp4
│   └── talh_q.mp4
├── group_b_step45_<prompt_id>_<seed>/
│   ├── native_hr.mp4
│   ├── trilinear.mp4
│   ├── cll_only.mp4
│   └── talh_e.mp4
└── failure_<prompt_id>_<seed>/
    ├── native_hr.mp4
    ├── cll_only.mp4
    └── talh.mp4
```

同时附一份简单的 `manifest.csv`：

```text
group_id,prompt_id,prompt,seed,handoff_step,case_name,video_path
```

视频应满足：

- 原始实验输出 MP4，不经过聊天软件、剪辑软件或二次压缩。
- 输出分辨率 720×1248、81 帧、16 fps；如个别文件不同，在 manifest 中注明。
- 保留完整 prompt、seed、case 名、切换步和 TAA strength。
- 暂时不需要手工截帧或画框；拿到完整视频后再用确定性脚本统一提取帧、选择裁剪坐标并生成拼图。

### 6.8 最终帧选择协议

收到视频后按以下流程确定图中内容：

1. 对每个候选组生成统一的全时段 contact sheet，不先单独查看 TALH 最优帧。
2. 空间面板优先选取主体和高频区域均清晰可见的中间帧，并对所有方法使用完全相同的帧号与裁剪坐标。
3. 时序面板选择发生明显运动的中心帧 `t`，固定抽取 `t-4, t-2, t, t+2, t+4`；不为不同方法单独调整时间位置。
4. 主文采用代表性或中位表现样本；最优案例和失败案例同时放入补充材料，降低挑样偏差。
5. 保存最终 prompt、seed、帧号、裁剪框和源视频 SHA-256，保证图稿可复现。

## 7. 补充材料图片

### 图 S1：切换步 sweep 与工作点选择

- **内容：** 固定提示和种子，按切换步从早到晚排列输出视频帧；至少包含 @40、@45，并尽可能加入更早和更晚的切换点。
- **目的：** 直接支撑“早期高分辨率后缀改善生成质量，晚期切换更快且 CLL 输入更接近纯净域”的工作点选择逻辑。
- **升级条件：** 若能为多个切换步补齐一致的延迟与 VBench-5，可升级为主文数据图；当前只有视觉 sweep 时留在补充材料。

### 图 S2：JTSL 与 CLL 的定性对比

- **内容：** JTSL、CLL-only、TALH 的同帧对比和高频区域放大。
- **目的：** 支撑联合轨迹—尺度回归更易模糊的动机。
- **表述边界：** 仅作为定性机制观察，不绘制缺乏等预算数据的排名柱状图。

### 图 S3：人工盲评与提示级分布

- **内容：** 以发散堆叠条形图展示细节、整体质量、时序稳定性的提示级胜/负/平；另附每个提示的 VBench-5 配对点图。
- **数据：** `human_review_prompt_wan50_step40_strength.csv`、`human_review_prompt_wan50_step45.csv`、`wan50_final_quality5_paired_statistics.csv`。
- **目的：** 呈现 TAA 的细节—时序权衡，并避免只报告聚合均值。

### 图 S4：失败案例与适用边界

- **内容：** 至少展示三类失败：主体一致性下降、局部过锐或伪细节、跨帧细节闪烁。
- **方法：** Native-HR、CLL-only、TALH-Q/E 使用相同提示、种子和裁剪位置。
- **目的：** 与第 6 节局限性对应，提高论文可信度。

## 8. 主文表格与图片的分工

为控制 AAAI 7 页正文的视觉密度，建议：

- **保留主文表 1：** 精确报告 Native-HR、TALH-Q/E 和 Endpoint Re-entry 的延迟、加速比和 VBench-5；图 2 展示趋势与逐维变化。
- **保留主文表 4：** 精确报告 2×2 因子数值；图 3(c) 展示主效应和交互结构。
- **表 2、表 3 移至补充材料：** 主文由图 3(a,b) 和正文关键数字承担模块证据。
- **表 5 移至补充材料或压缩：** 主文使用图 4 定性结果，补充材料使用图 S3 完整展示人工盲评。

这一安排可将主文压缩为“2 张核心表 + 4 张主图”，并使每张图承担独立论证任务，而不是重复表格。

## 9. 全文统一视觉系统

### 9.1 颜色映射

| 语义 | 颜色 | 线型/标记补充 |
|---|---|---|
| Native-HR / 冻结模块 | `#7B8794` 灰 | 方形、实线 |
| 低分辨率前缀 | `#4A90D9` 蓝 | 实线 |
| TAA | `#CC79A7` 紫红 | 三角或虚线 |
| CLL | `#009E73` 绿 | 圆形或实线 |
| HTR / 高分辨率后缀 | `#D55E00` 朱红 | 菱形或点划线 |
| TALH-Q | `#0072B2` 深蓝 | 圆形 |
| TALH-E | `#E69F00` 橙 | 三角形 |
| 普通基线 | `#B0B0B0` 浅灰 | 空心标记 |

颜色不能作为唯一编码；所有数据图同时使用标记形状、线型或直接标签。

### 9.2 字体与输出

- 图中文字使用 Times 或与 AAAI 模板兼容的 serif 字体。
- 最终打印尺寸下，任何文字不得小于 7 pt。
- 数据图输出 `fig_<name>.pdf` 与 300 DPI `fig_<name>.png`。
- 架构图输出宽度至少 2100 px 的 PNG；生成时保留 3 个候选版本和生成脚本。
- 定性帧拼图使用确定性脚本生成，记录提示、种子、帧索引与裁剪坐标。

### 9.3 文件命名

```text
paper/aaai27/figures/
├── gen_fig_talh_overview.py
├── fig_talh_overview.png
├── gen_fig_quality_efficiency.py
├── fig_quality_efficiency.pdf
├── gen_fig_component_evidence.py
├── fig_component_evidence.pdf
├── gen_fig_qualitative.py
├── fig_qualitative.pdf
└── supplementary/
    ├── fig_handoff_sweep.pdf
    ├── fig_jtsl_comparison.pdf
    ├── fig_human_review.pdf
    └── fig_failure_cases.pdf
```

### 9.4 生成式工具使用记录

- 若图 1 按 academic-plotting Workflow 1 使用 Gemini 生成候选架构图，应保存生成脚本、完整提示词和全部候选版本。
- 最终图中的模块关系、公式、数字和文字必须逐项人工核对，不允许生成模型补充论文中不存在的结构或实验结论。
- 投稿前按照届时有效的 AAAI 作者指南，在论文或提交材料中如实记录生成式工具在图稿制作中的作用。

## 10. 后续生成顺序

1. 先生成图 2 与图 3：数据已经闭环，可直接验证论文核心叙事。
2. 再整理图 4 的提示、种子、帧索引和裁剪坐标，完成定性选择协议后生成拼图。
3. 最后生成图 1：在正文术语、图号和两条训练数据流完全冻结后制作架构图，避免反复修改标签。
4. 主文排版稳定后，再按剩余页数决定图 S1 是否升级为主文切换步图。
