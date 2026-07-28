# AAAI 2027 Submission 21838 提交审计

审计日期：2026-07-28（Asia/Shanghai）

## 可上传文件

| 上传字段 | 文件 | 大小 | SHA-256 |
|---|---|---:|---|
| Main Paper | `AAAI27_InTraScale_Main_Paper.pdf` | 4,175,547 B | `b84b8d6e0a034d2e3271cd324ebabbfb97b1d0ae8b497c9ce3630cc5dbc827b3` |
| Reproducibility Checklist | `AAAI27_InTraScale_Reproducibility_Checklist.pdf` | 95,670 B | `7d9251e2b04c55f346bcde430b6a2fe5f6b7da16d459a03edb5206b540dc1a8f` |
| Technical Supplement | `AAAI27_InTraScale_Technical_Supplement.pdf` | 248,652 B | `6b52844226eea1b0f35d5689c8b6b7999b3c76f615bfc84c9b1b16820cbcf79a` |
| Media Supplement | `AAAI27_InTraScale_Media_Supplement.zip` | 15,160,649 B | `0201c11916dffa5141c2a1a14ed420f85812317f3d566cc9e5e3e5456b96649d` |
| Code and Data Supplement | `AAAI27_InTraScale_Code_and_Data_Supplement.zip` | 537,749 B | `6918945f3c29879e3e42ecdc8370ed881295a1f6340c54fbf2b0f3ab4cb152d7` |

以上哈希也保存在 `packages/SHA256SUMS.txt`。Technical、Media 和 Code
and Data 分别低于 Submission 21838 界面显示的 10 MB、50 MB 和 50 MB
上限。

## 审计结论

- 主论文为 US Letter、9 页；第 1--7 页为正文，第 8--9 页仅参考文献。
- 主论文、checklist 和技术补充的字体均已嵌入；未检出 Type 3、CID 或
  Identity-H 字体。
- 主论文未检出未定义引用、overfull box、作者、单位、邮箱或致谢。
- `main.tex` 的论文内容以 `origin.tex` 为锁定基准；正文、公式、表格
  数值、图注和引用未改写。两个结果表使用 9pt `\small`，未使用 7pt
  `\scriptsize`。
- 全宽 teaser 使用标准 `figure*`，位于第二页顶部；`main.tex` 不再使用
  `\g@addto@macro\@maketitle`、`\captionof` 或手工标题间距，原有模板
  合规风险已关闭。
- standalone checklist 保留官方 31 个问题：19 个 `yes`、0 个
  `partial`、2 个 `no`、4 个 `NA`；理论贡献总问题为 `no`，其 6 个
  条件项按官方 “If yes” 逻辑留空。Dataset Usage 明确区分实验中生成的
  模型内生样本与外部文献数据集。
- 技术补充为 US Letter，给出训练配置、协议、完整扩展表和
  checklist 中所有非 `yes` 项的证据边界。
- Media ZIP 含 6 个 H.264 MP4；每个都已实际解码为 1248x720、
  5.0625 秒，无解码错误。两组内 prompt、seed 和 transition step
  严格匹配，所有 6 个视频均已抽帧目检。
- Code and Data ZIP 含 232 个成员，包括 108 个 paper-specific
  Python 源文件、75 个 shell 脚本、8 个 JSON、1 个 JSONL 和 14 个
  CSV；内部导入闭包检查通过，ZIP 不含 `__pycache__` 或 `.pyc`。
  Media ZIP 含 9 个成员。两个 ZIP 均通过
  CRC、路径穿越、绝对成员路径和内部 SHA-256 清单验证。
- 当前两个 ZIP 的文本成员未检出个人姓名、邮箱、机器绝对路径、
  机构名称或外部论文源码/数据仓库链接。媒体仅含编码器自身的
  x264 项目信息，不是论文源码/数据链接。
- 打包脚本对 `paper/aaai27/rewrite` 做前后快照，确认打包过程未修改
  终稿目录。

## 尚需作者人工确认

1. 确认 Submission 21838 对应的 OpenReview abstract registration 已在
   截止时间前有效完成。
2. Checklist 4.8 计算基础设施已由 2026-07-28 原实验机导出补为
   `yes`：GPU/显存、CPU/内存、OS、driver/CUDA/cuDNN、Python/PyTorch
   与 LightX2V、DiffSynth-Studio、VBench revision 均已记录。VBench
   独立 Python 环境的旧固定路径已不存在，但框架 commit 已精确固定；
   Distill4 TTD3 的 legacy LMDB 路径仍需运行小型补导。5 个完整自研
   checkpoint 仍用于完整端到端复现包。
3. 确认录用后的源码、权重和生成数据公开许可证。未确认前，checklist
   中相关公开承诺保持 `no`。
4. 上传后在网页端重新下载一次五个文件，并与 `SHA256SUMS.txt` 比对，
   防止选错旧版 `submission_materials_20260726` 文件。
