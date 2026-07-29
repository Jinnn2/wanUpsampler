# AAAI-27 投稿材料工作区

本目录独立于 `paper/aaai27/rewrite`。构建脚本会在打包前后比较论文目录快照，
确保打包过程本身不会改动终稿。

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

1. 编译 Technical Supplement 和独立 Reproducibility Checklist；
2. 复制已编译主论文并检查 PDF 字体；
3. 验证媒体文件、匿名性关键词、ZIP 路径和上传大小上限；
4. 生成 Code and Data、Media 两个 ZIP；
5. 生成 `packages/SHA256SUMS.txt`；
6. 比较 `rewrite` 在构建前后的内容快照，确保打包期间未被修改。

## 状态

当前材料是可直接上传的保守版候选包：主论文、Checklist、Technical、
Media、Code and Data 均已生成。368p 原始逐样本证据已经找回并加入数据包。
仍需作者确认的事项见 `OPEN_ISSUES_ZH.md`，其中最重要的是 OpenReview
摘要注册状态和公开许可证计划。实验机环境、全部五个 realized LMDB
集合及五个自研权重的完整 SHA-256 均已取证；权重本体因总计约 1.62 GB
不能放入 50 MB 投稿 ZIP。Checklist 4.8 已由 `partial` 补为 `yes`；
无法由证据代替的公开许可证承诺仍保持 `no`。
