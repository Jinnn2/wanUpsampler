# TALH 真实视频帧候选审计

审计对象：`C:\Users\jinho\Downloads\talh_figure_video_candidates.tar.gz`

- SHA-256：`4430723D932663DF793DC6C7CCCED81B1F912E613959E4B0AF1CBB6DD0E46134`
- 候选组：6 组。
- 视频文件：24 个。
- 每组方法：Native-HR (estimated)、Trilinear、CLL-only、TALH。
- 帧数与帧率：全部为 81 帧、16 fps。
- 接触表帧号：第 1、21、41、61、81 帧。

## 1. Native-HR (estimated) 的定义

定性图中的 `Native-HR (estimated)` 是同一 prompt、seed 和低分辨率生成轨迹下完成 50 步采样的 640×368 参考视频，即实验中的 `ori_50`。它提供与三种 720p handoff 输出一致的场景、主体和运动参照。

之所以不在该图中直接使用真实 720p Native-HR 视频，是因为扩散采样对空间分辨率敏感：即使 prompt 和 seed 相同，368p 与 720p 的原生轨迹也会产生不同的场景实例与运动细节，因而不能形成内容对齐的逐帧定性比较。将低分辨率完整轨迹标为 `estimated`，比把内容不一致的 720p 样本并排展示更忠实于该图的比较目的。

这里必须与定量实验区分：表 1 和质量—效率图中的 `Native-HR Sampling` 仍指真实的 1248×720 全程采样；`Native-HR (estimated)` 只出现在定性图中，不参与延迟、VBench 或 720p 细节质量的定量计算。

## 2. 媒体一致性结论

- 6 个 `Native-HR (estimated)` 视频均为 640×368，符合内容对齐代理的预期尺寸。
- 18 个 Trilinear、CLL-only 和 TALH 视频均为 1248×720，符合论文目标输出分辨率。
- 24 个视频的帧数、帧率和候选帧索引均通过一致性检查。

审阅用接触表会将 `Native-HR (estimated)` 放大到统一版面尺寸，但不会把该行用于 720p 清晰度或纹理恢复的优劣判断。空间细节结论只比较 Trilinear、CLL-only 与 TALH 三种 720p 输出；estimated 行用于说明内容、结构和运动是否保持一致。

自动审计结果位于：

- `outputs/aaai27_figure_work/review/video_audit.csv`
- `outputs/aaai27_figure_work/review/audit_summary.json`
- `outputs/aaai27_figure_work/review/contact_*.png`

## 3. 候选可读性评估

以下排序只衡量论文图片中的视觉可读性，不表示方法质量排名。

### TALH-Q 空间细节

1. **prompt 05 / seed 9705，机器人装配电路板。** 电路板针脚、机械臂边缘和小型元件提供明确高频区域；Trilinear 的模糊与 CLL/TALH 的恢复最容易在有限版面中辨认。推荐作为主文空间面板首选。
2. **prompt 08 / seed 9708，热带花卉温室。** 叶片、花瓣、玻璃框架和光束具有丰富纹理，适合作为补充材料或第二候选。
3. **prompt 00 / seed 9700，雨后夜市。** 视觉内容丰富，但人群遮挡、反射和高动态范围会增加裁剪解释难度。

### TALH-E 运动与时序

1. **prompt 07 / seed 9707，金毛穿过海浪。** 主体边界、水花和肢体运动清晰，适合连续局部帧与身份稳定性展示。推荐作为主文时序面板首选。
2. **prompt 02 / seed 9702，林间山地自行车。** 运动幅度大，轮胎、骑手和尘土适合展示动态细节，但高速相机运动会同时引入自然运动模糊。
3. **prompt 06 / seed 9706，霓虹灯下舞者。** 色彩表现强，但主体遮挡、暗部剪影和手持镜头使细节差异更难归因。

## 4. 暂定主文组合

- **空间细节面板：** prompt 05 / seed 9705，Native-HR (estimated)、Trilinear@40、CLL-only@40、TALH-Q。
- **时序面板：** prompt 07 / seed 9707，Native-HR (estimated)、Trilinear@45、CLL-only@45、TALH-E。
- **补充材料候选：** prompt 08 用于第二个细节案例；prompt 02 用于高速运动案例；prompt 00 或 prompt 06 可用于复杂场景或失败边界。

最终图中，`Native-HR (estimated)` 行展示完整上下文帧，不将其低分辨率局部放大图与 720p 方法并列作为锐度证据。放大裁剪和纹理结论集中在三种 720p 方法之间；时序面板则可使用 estimated 行辅助判断主体结构与运动语义是否一致。

## 5. 推荐图注说明

> `Native-HR (estimated)` 表示与各 handoff 方法内容对齐的 368p 完整采样轨迹，仅作为定性参照；其帧在图中经过等比例放大以统一版面。定量实验中的 Native-HR Sampling 为真实 720p 全程采样。由于不同原生分辨率下即使使用相同随机种子也不会产生逐帧对齐的内容，我们不使用 estimated 参考进行 720p 锐度比较。

## 6. 可复现工具

候选审计与 contact sheet 由以下脚本生成：

```text
paper/aaai27/figures/prepare_qualitative_candidates.py
```

该脚本分别检查 368p estimated 参考和 720p 方法输出的预期尺寸，统一检查帧率与帧数，记录 SHA-256，并以固定帧号生成四路 contact sheet。它只用于候选审阅，不直接生成最终投稿图片。
