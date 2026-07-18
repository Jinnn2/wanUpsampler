# AAAI-27 统一结果集

该目录由 `integrate_result_snapshots.py` 生成，合并 2026-07-17 base core 与 2026-07-18 closure incremental archive。

- 增量归档校验：全部 747 个文件通过 SHA-256；其中 48 个原始文件名含冒号，只能从 tar 流读取。
- 汇总 VBench case：24 个。
- 论文因子效应：9 组。
- 最终质量—效率 case：4 个。
- 最终 TAA 配置（LoRA 实现）：step40 strength=0.75，step45 strength=0.75。
- 训练资源：4×NVIDIA H100；TAA 约 33 小时，CLL 约 8 小时（wall-clock）。

## 有意不纳入的实验

- `sources.generalization`
- `sources.lora_architecture_loss`
- `sources.stage2_architecture_loss`

## 使用约束

- `wan50_step45_final_endpoint_paired_statistics.csv` 从归档内 TAA strength=0.75 原始逐样本表重新计算；不要引用旧的 strength=1.0 paired table 作为最终配置。
- TAA endpoint 指标与最终视频质量必须分别表述。
- 人工盲评以 10 个 prompt-majority 为统计单位；30 个 individual votes 仅作描述。
- 最终效率表不支持显存下降主张。
