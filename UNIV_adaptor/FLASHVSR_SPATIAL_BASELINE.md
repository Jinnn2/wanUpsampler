# FlashVSR spatial baseline: asset-only screening

论文主张保持为待检验假设：**prompt 对不同加速机制与 budget 的容忍度是否具有稳定、可学习的差异，足以支持 prompt-only 选择。** 本轮只验证空间基线的质量评价是否有区分度，不证明选择器收益，也不预设 VBench 一定失效。

空间基线采用已有的 Wan2.1 低分辨率 **416×240、81 帧、50 步**生成的干净 endpoint，经原生 VAE 解码，再用 FlashVSR v1.1 Tiny x4、去 padding、Lanczos 到 832×480；不使用旧 RealESRGAN / VAE 回编码 / HR4。这是我们的组合 pipeline，不是 FlashVSR 官方加速实验的直接复现。B025 指主生成像素密度 0.25，不是总耗时比例。

本轮直接复用 `outputs/flashvsr_asset_diagnostic_v1` 的 **4 prompt × 2 seed × 5 track = 40 个 33 帧片段**。不新增生成、不下载模型、不重编译 kernel、不改变原诊断脚本及其冻结哈希。也可把已经导出的诊断压缩包解压后作为 source，不需要 NPZ、权重或原服务器绝对路径。

## 运行

在服务器仓库中更新这些新增文件后：

```bash
EVAL_SCRIPT=UNIV_adaptor/scripts/run_univ_flashvsr_spatial_baseline_eval_8gpu.sh
bash "$EVAL_SCRIPT" all
```

依次冻结计划、核验实际视频帧数/FPS/画布/哈希、打包匿名人评、运行七个 VBench 维度、生成配对报告并打包研究者分析包。视频核验需要系统 `ffprobe`，或运行 Python 环境已有的 `imageio-ffmpeg`。不会安装依赖或改环境，不下载生成/SR 模型；VBench 自身若缺评估权重，仍可能按其原逻辑下载。默认评分使用 `/opt/conda/bin/python` 和 `/mnt/afs_2/houze/VBench`，VBench 锁定 commit `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。

若评分使用单独的已有环境，设置 `VBENCH_PYTHON`；若目录不同，设置 `SOURCE_FLASH_DIAG_ROOT` / `FLASH_EVAL_ROOT` / `VBENCH_ROOT`。不读取旧流程的 `DATASET_ROOT`，避免误写原实验目录。可以单独执行 `plan check blind export-rater score report export-analysis`；`all` 不运行人评分析，因为尚无评审文件。

各维度顺序运行，每次调用现有严格评分器占用 `NGPUS=8`，核验逐视频覆盖和内容绑定的评分缓存。**不要与生成任务抢同一批 GPU。** 断点重跑复用严格匹配的完成结果；不会覆盖不匹配 / 中断的文件。修改协议或脚本后必须使用新的评测输出目录。

## 四类配对

|配对|用途|
|---|---|
|FULL / NATIVE_LR_FLASH|整个低分辨率生成 + SR pipeline 的视觉与 prompt 风险|
|NATIVE_LR_BICUBIC / NATIVE_LR_FLASH|相同原生 LR 输入上 SR 的增益与代价|
|HR_DOWN4_BICUBIC / HR_DOWN4_FLASH|同一 FULL 降采样输入上恢复能力|
|FULL / HR_DOWN4_FLASH|已对齐 SR 控制的细节保真风险|

每组全部纳入，不按评分或主观观感筛选。32 个主配对，另加 4 个 A/B 反转的重复配对检查一致性；重复不进入主统计。A/B 左右平衡，公开材料隐藏方法、seed、源文件名、自动分数与配对类型。不是双盲试验：研究者知道映射，观看过诊断结果的评审也不能当作未接触标签的新评审。

## 本地人评（无需端口转发）

从新输出目录 `exports/` 下载 `flashvsr_spatial_raters_*.tgz`，解压后打开内含 `index.html`。两段视频保留原 MP4 字节，不再转码；HTML 离线运行，不需要 localhost、VS Code tunnel 或 CDN。

输入自己的匿名 ID，并声明此前是否见过标签 / 诊断结果。分别评价 prompt 符合度、细节 / 结构正确性、清晰度、时序稳定性、总体质量，填写偏好与具体伪影。对无法判断的细节选“不确定”，不把构图不同直接视为失败，也不假设 FULL 永远正确。浏览器自动保存，仍建议定期导出 JSON。

优先邀请 **至少 3 位此前未看过标签的独立评审**完成全部 36 组。仅有研究者本人评分是探索性诊断。相同评审的多次导出可以合并不重叠的组，但出现冲突时程序会拒绝静默覆盖；请只保留该评审最新、完整的一份。

把 JSON 放在新输出目录的 `ratings/`，再运行：

```bash
bash "$EVAL_SCRIPT" human-report
bash "$EVAL_SCRIPT" export-analysis
```

人评报告写入由输入内容哈希命名的 `human_reports/` 子目录，保留每组/seed 的轴向差值及 prompt 内 seed 均值。未知项和未评组不补零；不把投票当作独立 prompt 样本做显著性检验、不自动删除不一致评审、不构造人评总分，也不自动生成选择器训练标签。评审 ID 与未接触标签声明无法自动证明独立性。

研究者分析包 `flashvsr_spatial_analysis_PRIVATE_*.tgz` 含私有映射、逐视频与配对分数、人评（若有），**不可发给尚未评审的人员**。人评包只含公开 HTML/JSON/匿名媒体。已有同内容包核验后复用，源目录完全只读。

## 结果边界与下一步

七维分别报告，不是官方完整 VBench 总分，不称为“full VBench”。不把 dynamic degree 越高视为越好；静态 prompt 不应因静止受罚。不得混入此前 81 帧分数。两种原生生成轨迹不对齐，不使用它们之间的 PSNR/LPIPS 作为质量真值。

SR 耗时保留，但端到端加速比为空：81 帧生成和解码耗时不能与 33 帧 SR 耗时直接拼成正式速度结论。当前仅有 B025、四个选定 prompt 和两个 seed，不能据此训练 / 宣称 method+budget 分类收益。若确有可复核的人评差异，下一轮再扩大同源 prompt、增加 B050 和完整片长的同协议对照，冻结新计划后比较 prompt 间效应与 seed 内稳定性；若无差异，也如实保留结论，不调指标制造差异。
