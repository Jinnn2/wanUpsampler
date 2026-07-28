# 368p 原始证据恢复审计

## 为什么此前被判断为缺失

较新的 `wan50_pareto_v2_core_20260721_171638` 快照中，
`compiled_tables/operator_368p.csv` 是 `MISSING` 状态文件；它只表示该次
快照没有携带 368p operator 结果。

`integrated_20260718/integration_manifest.json` 明确把真实
`operator_368p.csv` 标记为 `base_fallback`，来源是更早的完整导出：

`C:\Users\jinho\Downloads\aaai27_final_20260717`

## 找回的原始文件

原始逐样本文件：

`evidence/canonical/operator_368p/metrics_val_offset0_limit50.jsonl`

- 文件大小：87,572 B
- SHA-256：
  `e9dccf84dc386b91616e3151d43d5ef19c29f5e4bf8dcb33eb4b862eceaf2c85`
- 记录数：50
- 唯一 sample ID：50
- 与完整导出的 `SHA256SUMS`：一致

同目录的旧 `summary_val.json/csv` 和 `samples_val.csv` 是零字节文件，
对应 task state 记录的汇总命令失败。它们不能作为证据；可用证据是通过
SHA 校验的逐样本 JSONL。

## 从原始记录重新计算

| 指标 | Trilinear | ITU | 相对改善 | Wins |
|---|---:|---:|---:|---:|
| Latent L1 ↓ | 0.244369110 | 0.162844184 | 33.3614% | 50/50 |
| PSNR ↑ | 23.423814392 | 24.102052135 | 2.8955% | 37/50 |
| SSIM ↑ | 0.674044469 | 0.759204099 | 12.6341% | 48/50 |
| LPIPS ↓ | 0.335217976 | 0.164028294 | 51.0682% | 50/50 |
| Temporal L1 ↓ | 0.023880088 | 0.020168135 | 15.5441% | 49/50 |
| HF-energy error ↓ | 0.007921320 | 0.002334041 | 70.5347% | 50/50 |

均值、population standard deviation、win count 均由
`tools/recover_operator_368p.py` 直接从 50 条 JSONL 记录计算。输出与
`integrated_20260718/compiled_tables/operator_368p.csv` 在其六位小数精度内
完全一致。

## 已进入匿名数据包

代码数据包新增：

- `data/operator_368p/operator_368p_raw_sanitized.jsonl`
- `data/operator_368p/operator_368p_samples.csv`
- `data/operator_368p/operator_368p_summary.csv`
- `data/operator_368p/operator_368p_provenance.json`
- `tools/recover_operator_368p.py`

原始记录中的机器绝对路径已替换为
`<GENERATED_OUTPUT_ROOT>/<basename>`；指标、sample index 和 sample ID
保持不变。
