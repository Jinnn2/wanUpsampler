# 投稿前阻塞项与风险

## P0：必须处理

- [ ] 确认已在 2026-07-21 AoE 前完成 OpenReview abstract registration。
  若没有有效条目，AAAI-27 官方规则不允许新建 full-paper submission。
- [x] 主论文为 9 页；第 1–7 页是技术内容，第 8–9 页仅为参考文献。
- [x] 主论文 PDF 已移除 Type 3、CID 和 Identity-H 图内字体；当前
  `pdffonts` 仅报告嵌入的 Type 1 字体。
- [x] 两个结果表已从 7pt `\scriptsize` 调整为 Author Kit 允许的
  9pt `\small`，并保持单层展开维度表头。
- [x] 已找回 `368x640 -> 720x1248` ITU operator 的 50 条逐样本
  JSONL。源文件 SHA-256 为
  `e9dccf84dc386b91616e3151d43d5ef19c29f5e4bf8dcb33eb4b862eceaf2c85`，
  与旧导出清单一致。由原始记录重新计算的 latent L1、LPIPS 和 temporal
  L1 相对降幅分别为 33.3614%、51.0682% 和 15.5441%。匿名原始记录、
  逐样本 CSV、汇总表和确定性恢复脚本已加入代码数据包。
- [x] 已从原实验机补录基础设施：NVIDIA H100 80GB HBM3（81,559 MiB）、
  双路 Intel Xeon Platinum 8462Y+、2,159,612,928,000 B RAM、Ubuntu
  22.04.5、driver 550.90.07、CUDA 12.8、cuDNN 9.10.2、Python 3.11.13、
  PyTorch 2.8.0+cu128，以及 LightX2V、DiffSynth-Studio 和 VBench
  commit。导出时容器可见 1 卡；训练 launchers 和作者记录对应 4 卡任务，
  两种口径已在证据摘要中明确区分。
- [x] 已提供轻量导出脚本
  `code_data_package/tools/export_missing_repro_metadata.sh`：不复制权重、
  视频或 latent tensor，只导出环境、Git revision、LMDB shard/sample/
  schema/shape/seed 与确定性 split 证据。远程入口和确切目录见
  `REMOTE_EVIDENCE_LOCATIONS_ZH.md`。2026-07-28 的首轮导出已执行并通过
  27 项 metadata 内部 SHA-256 校验；四个 realized LMDB 集合已核实。
- [ ] 运行 `remote_export/export_temp_followup.sh`，补查 Distill4 TTD3
  legacy LMDB 的实际 shards/samples，并尝试从 conda 环境清单定位已经
  迁移的 VBench Python。首轮固定路径 `/opt/conda/envs/vbench/bin/python`
  已不存在，不能伪造其 freeze。
- [ ] 在原实验机运行 `tools/export_full_repro_bundle.sh`，取回 5 个最终
  checkpoint：Wan50 ITU、Wan50 TTD step40/45、Distill4 ITU、Distill4
  TTD step3。脚本已经写入每个文件的精确大小和 SHA-256，缺失、截断或
  错误权重都会拒绝导出。
- [ ] 不要使用旧下载目录中两个约 1.5 MiB 的 Distill 权重。它们与
  run manifest 记录的 570,100,417 B 和 306,801,824 B 不符，SHA-256
  也不匹配，已判定为截断文件。

## P1：提交前强烈建议

- [ ] 确认公开发布计划和许可证。当前不能承诺所有自研代码、权重和生成数据
  在录用后使用何种许可证。
- [ ] 把 `requirements.txt` 的最低版本约束补充为导出脚本生成的
  `pip freeze --all`、conda environment 和 explicit package list。
- [x] 已按 Submission 21838 上传字段校验：Technical 10 MB、Media
  50 MB、Code and Data 50 MB。
- [ ] 对补充 PDF 和 ZIP 做人工匿名检查：PDF properties、注释、绝对路径、
  用户名、邮件、实验室标志、可识别声音和校园场景。
- [x] 已逐项核对 31 个 checklist 问题；数据描述、开发范围、
  预处理/实验源码、代码映射、基础设施、指标定义和最终参数均有证据并答为
  `yes`。未选择公开许可证仍为 `no`。
- [x] `main.tex` 已移除 `\g@addto@macro\@maketitle`、`\captionof` 和
  手工标题间距，改用标准 `figure*`；teaser 经编译和逐页视觉检查确认位于
  第二页顶部。`origin.tex` 保持不变，正文仍为 7 页、参考文献仍为 2 页。

## 已有但应保持边界的证据

- VBench-5 是五个 custom-input 维度的等权均值，不是官方
  VBench Quality、Semantic 或 Total。
- warm latency 是同一 resident process 内 1 次不计时 warm-up +
  5 次 CUDA 同步测量；不要与 cold-start 时间混用。
- RALU 是 Wan adaptation，应写 `RALU-style`，不能宣称为官方 RALU 最优实现。
- Distill4 主路线是 3 LR + 1 HR；不要把 token 比例当作实测 wall-clock speedup。
- 公开基础模型已经固定 Hugging Face revision，并提供约 58 GB 的下载与
  SHA-256 校验脚本；公开模型与五个自研 checkpoint 不应混为一类。
