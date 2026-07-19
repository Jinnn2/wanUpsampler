# AAAI-27 七页正文删减计划

> 执行状态（2026-07-20）：已将原正文 Table 3 移至 Supplementary Table S2，
> 删除 Figure 4，将系统级图表前移至第 6 页，并将人评表改为单栏布局。
> 当前正文严格止于第 7 页，第 8 页仅包含参考文献，目标已达成。

## 当前状态

- 最终正文共 8 页：前 7 页为技术内容，第 8 页仅为参考文献。
- 引用已全部解析，剩余图表均通过视觉可读性检查。
- 并排详细表的测试曾造成横向重叠，最终未采用；人评表改用可读的单栏布局。

## 必须保留的核心证据

1. Figure 1：真实帧细节对比和 TALH 推理流程。
2. Figure 2：TAA/CLL 的模型内生监督与方法闭环。
3. Figure 3 + Table 1：相对 Native-HR 的质量—效率主结果。
4. Table 4：2x2 factorial，直接解释 TAA 在 VBench-5 总分上变化较小、但 Imaging Quality 有稳定增益。
5. Table 5：人评细节与总体质量偏好，是“更多细节、更精致”论点的关键支撑。
6. TAA 的三层结论：endpoint alignment 改善、Imaging Quality 恢复、人评细节偏好。

## 推荐方案 A：保留 TAA 叙事与综合图

预计释放 0.48--0.62 页，风险最低。

1. 将正文 Table 2（CLL operator）移至补充材料，仅保留正文中的 48.6%、73.3%、30.8% 等主结果。补充材料已有 Table S1。预计节省 0.10--0.13 页。
2. 将正文 Table 3（TAA endpoint）移至补充材料，正文保留 25.82%、21.03%、5.03% 以及全胜/显著性结论。补充材料已有 Table S2。预计节省 0.09--0.12 页。
3. 保留 Figure 4，作为 CLL、TAA 和 factorial 三类证据的单一视觉汇总；正文引用 Figure 4(a--c)，精确置信区间留在补充材料。
4. Related Work 三个小节合计压缩约 120--150 词，删除对常规 latent diffusion 和 temporal distillation 的教科书式背景，只保留与 mixed-resolution handoff 的差异。预计节省 0.12--0.16 页。
5. Method 中压缩约 80--100 词：合并 3.1 的误差分解解释和 3.5 cached-prefix 条件的重复表述，不删公式与四项一致性条件。预计节省 0.08--0.11 页。
6. Discussion 与 Conclusion 合计压缩约 50--70 词，避免再次复述三模块定义和 CLL/TAA 分工。预计节省 0.05--0.08 页。

## 备选方案 B：尽量少删正文文字

预计释放 0.42--0.50 页，但会失去模块综合图。

1. 删除 Figure 4；其三块信息分别已由 Table 2、Table 3、Table 4 和相邻正文覆盖。预计节省 0.20--0.24 页。
2. 将正文 Table 2 移至补充材料 Table S1，保留正文主数字。预计节省 0.10--0.13 页。
3. 将 Related Work 和 Conclusion 合计压缩约 80--100 词。预计节省 0.08--0.11 页。
4. 保留 Table 3、Table 4、Table 5，使 TAA 的 endpoint、VBench 分项解释和人评证据仍全部留在主文。

## 不建议的做法

- 不再缩小正文字号或修改页边距；这会违反模板意图。
- 不采用两个宽表强行并排；当前测试已出现单元格重叠。
- 不删 Table 4 或 Table 5；二者分别支撑“VBench-5 为什么变化小”和“TAA 的细节改善为什么真实可感知”。
- 不只报告 VBench-5 总分；必须同时保留 Imaging Quality 分项和人评结果，避免把 TAA 描述成对总体自动指标有大幅提升。

## 建议执行顺序

先执行方案 A 的 Table 2/3 迁移并编译；若仍越界，再压缩 Related Work 和 Method；最后仅在需要时压缩 Discussion/Conclusion。每一步都重新检查：第 8 页是否从页首开始即为 References、引用是否完整、图表字号是否可读、是否存在 overfull box。
