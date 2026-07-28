# AAAI-27 投稿材料工作区

本目录独立于 `paper/aaai27/rewrite`。创建和打包过程中不会修改论文主目录。

## 准备结果

- `supplementary_document/`：匿名技术补充 PDF 的 LaTeX 源文件。
- `reproducibility_checklist/`：AAAI-27 官方模板填写版，单独编译和上传。
- `code_data_package/`：匿名化后的代码、配置、提示词、派生表格和原始逐样本指标。
- `media_archive/`：两组严格匹配 prompt/seed 的 720p 视频比较。
- `packages/`：最终 PDF、ZIP 与 SHA256 清单。
- `OFFICIAL_REQUIREMENTS_ZH.md`：投稿要求和北京时间换算。
- `OPEN_ISSUES_ZH.md`：投稿前必须人工解决或确认的问题。
- `FULL_E2E_REPRO_ZH.md`：公开模型、五个自研权重、环境导出和训练数据的
  完整复现清单。
- `EVIDENCE_RECOVERY_ZH.md`：368p 逐样本原始文件、SHA、重算方法与旧
  `MISSING` 状态的来源解释。

## 一键构建

在本目录运行：

```powershell
.\build_packages.ps1
```

脚本会：

1. 编译补充 PDF 和 reproducibility checklist；
2. 验证媒体文件、匿名性关键词和禁止的绝对路径；
3. 生成代码数据包和媒体包；
4. 生成 `packages/SHA256SUMS.txt`；
5. 比较 `rewrite` 在构建前后的 Git diff，确保该目录未被修改。

## 状态

当前材料是“可审阅草案”，不是无条件可上传的最终包。368p 原始逐样本证据
已经找回并加入数据包。剩余阻塞项见 `OPEN_ISSUES_ZH.md`，其中最重要的是
正文页数、OpenReview 摘要注册状态、实验机环境快照和五个完整自研权重。
