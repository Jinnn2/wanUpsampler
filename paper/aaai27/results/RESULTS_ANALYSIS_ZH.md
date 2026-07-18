# AAAI-27 全量实验结果整合与论文口径

> 整合日期：2026-07-18  
> 数据源：`aaai27_final_20260717/core/core` + `aaai27_closure_20260718_incremental.tar.gz`  
> 统一结果：`integrated_20260718/`

## 1. 数据闭环状态

增量归档中的 747 个文件全部通过 SHA-256 校验，无缺失或哈希不一致。归档内有 48 个 VBench 原始 JSON 的文件名包含 `:`，Windows 无法直接落盘，但整合脚本从 tar stream 读取，不影响统计。旧下载目录的大型 evidence 不完整，因此只使用其经过校验的 `core/core` 汇总；旧 core 用于补齐增量包有意省略的 368p CLL、distill transfer、旧因子实验与盲评表。

合并后有 24 个 VBench case、9 组可分离因子效应、4 个最终 Wan50 质量—效率点。训练使用 4 块 NVIDIA H100 GPU；作者报告的 wall-clock 训练时间为 TAA 约 33 小时、CLL 约 8 小时。以下三类扩展实验确定不再开展，并作为论文范围限制：

- TAA module/rank/loss 消融；
- CLL architecture/loss 消融；
- prompt/domain/checkpoint 泛化。

最终配置固定为 step40 strength=0.75、step45 strength=0.75、distill strength=0.75。旧的 `wan50_endpoint_paired_statistics.csv` 对应 step45 strength=1.0，不能作为最终配置引用；最终 step45 配对统计已从归档内 strength=0.75 原始逐样本表重算。

论文正式术语与实验内部标签的对应关系如下。CSV、case name 和脚本参数保留内部名，以免破坏复现；正文只使用功能名。

| 内部标签 | 论文术语 |
|---|---|
| LoRA / handoff LoRA | Trajectory Alignment Adapter（TAA，LoRA implementation） |
| Stage2 | Clean Latent Lifter（CLL） |
| re-noise + HR suffix | High-Resolution Trajectory Re-entry（HTR） |
| Stage3 | Joint Trajectory-Scale Lifter（JTSL） |
| Full-HR | Native-HR Sampling |
| Full-LR+Stage2+1HR | Endpoint Re-entry Baseline |
| Quality5 | VBench-5 |

## 2. 核心结论

当前证据最稳妥的论文叙事是：

1. **论文第一主张是 Native-HR Sampling 加速。** 相对 50-step Native-HR，TALH-Q 与 TALH-E 分别降低 45.33% 和 54.85% 延迟，达到 1.83× 和 2.22× 加速；VBench-5 分别保留 97.76% 和 97.53%。
2. **这是一条质量—效率 Pareto，而不是无损或等价质量。** 两个 TALH 工作点的配对 VBench-5 置信区间均跨零，但 n=10 不足以证明等价；subject consistency 相对 Native-HR 显著下降。
3. **CLL 是系统质量保持的主要来源。** 它在两种缩放比例、潜变量误差、感知质量和时序误差上都显著优于固定插值，并贡献几乎全部端到端 VBench-5 增益。
4. **TAA 是确定的 trajectory alignment adapter。** step40 和 step45 的终点 L1 分别下降 25.82% 和 21.03%，均为 10/10 胜出；但其端到端作用表现为细节偏好提高与时序稳定性下降的权衡。
5. **TALH-Q/E 是两个可控工作点。** TALH-Q 的 VBench-5、subject consistency 和 aesthetic quality 更高；TALH-E 更快。
6. **当前实现没有显存收益。** 最终四个 Wan50 case 的峰值均约 26 GiB，只能声称延迟加速。

论文中的比较应按以下层级组织：

| 层级 | 对比 | 回答的问题 |
|---|---|---|
| 第一主对比 | Native-HR vs TALH-Q/E | 核心提速是否成立，质量代价是多少 |
| Pareto 强基线 | Endpoint Re-entry Baseline | 更激进的 LR 计算能否进一步提速，以及额外质量代价 |
| 方法对照 | TALH vs Trilinear Handoff | learned handoff 相对普通动态分辨率切换是否值得 |
| 模块消融 | CLL vs Trilinear；TAA-aligned vs Unaligned | 系统质量来自哪个模块，以及模块间是否存在负交互 |

因此，`TALH vs Trilinear Handoff` 不能替代 `TALH vs Native-HR`：前者解释 handoff 设计，后者才支撑论文的生成加速主张。

## 3. 核心主对比：TALH 与 Native-HR Sampling

| 方法 | LR/HR evals | 时间（秒） | 延迟下降 | 加速 | 显存 GiB | VBench-5 | 保留率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native-HR Sampling | 0/50 | 253.097 ± 1.534 | — | 1.000× | 26.173 | **0.828364** | 100.00% |
| TALH-Q (TALH@40) | 40/10 | 138.359 ± 0.499 | **45.33%** | **1.829×** | 26.255 | 0.809830 | 97.76% |
| TALH-E (TALH@45) | 45/5 | 114.264 ± 1.517 | **54.85%** | **2.215×** | 25.981 | 0.807920 | 97.53% |

这是论文系统层面的主表。它回答核心问题：将大部分 denoising evaluations 移到低分辨率后，能否显著减少高分辨率视频生成时间，同时控制质量下降。TALH-Q 节省 114.74 秒，TALH-E 节省 138.83 秒；两者都没有减少原 sampler 的总 denoising evaluation 数，速度收益来自 LR evaluation 更便宜。

逐 prompt 的复合 VBench-5 配对统计为：

| 对比 | VBench-5 差值（TALH−Native-HR） | 95% bootstrap CI | TALH胜/Native-HR胜 | sign p |
|---|---:|---:|---:|---:|
| TALH-Q vs Native-HR | -0.01853（-2.24%） | [-0.04303, 0.00336] | 2/8 | 0.1094 |
| TALH-E vs Native-HR | -0.02044（-2.47%） | [-0.04606, 0.00170] | 3/7 | 0.3438 |

两个复合指标的置信区间均跨零，说明当前 10-prompt 测试没有检出稳定的 VBench-5 总体差异；但这不是等价性检验，不能据此声称“无损”或“与 Native-HR 等价”。逐维上，TALH-Q 和 TALH-E 的 subject consistency 均显著低于 Native-HR（各 1/9，p=0.02148），而 motion、aesthetic 和 imaging 的置信区间大多跨零。因此最可靠的论文表述是：**TALH 以约 2.2%–2.5% 的 VBench-5 均值代价换取 1.8×–2.2× 加速，并形成可调的质量—效率 Pareto。**

该 Native-HR 主对比目前只在 Wan50 上闭环。Distill4 没有对应的 Native-HR distill 计时与质量 case，因此它只用于证明 TALH 可与 timestep distillation 组合，不能单独承担相对 Native-HR 的提速主张。

## 4. Clean Latent Lifter（CLL）

| 输入→输出 | 方法 | Latent L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Temp. L1 ↓ |
|---|---|---:|---:|---:|---:|---:|
| 480p→720p | Trilinear | 0.192880 | 24.2927 | 0.715875 | 0.235754 | 0.021213 |
| 480p→720p | CLL | **0.099134** | **29.9980** | **0.869612** | **0.062938** | **0.014676** |
| 368p→720p | Trilinear | 0.244369 | 23.4238 | 0.674044 | 0.335218 | 0.023880 |
| 368p→720p | CLL | **0.162844** | **24.1021** | **0.759204** | **0.164028** | **0.020168** |

每种设置 n=50。480p 路线的 latent L1、LPIPS、temporal L1 相对下降 48.60%、73.30%、30.81%；368p 路线分别下降 33.36%、51.07%、15.54%。368p 的 PSNR 只在 37/50 上改善，因此正文应把感知与时序指标作为该设置的主要证据，而不是只突出 PSNR。

## 5. Trajectory Alignment Adapter（TAA）

| 工作点 | Unaligned L1 | TAA-Aligned L1 | 相对改善 | 95% bootstrap CI（改善量） | 胜/负 | sign p |
|---|---:|---:|---:|---:|---:|---:|
| step40→50，s=0.75 | 0.032155 | **0.023853** | 25.82% | [0.004791, 0.013023] | 10/0 | 0.001953 |
| step45→50，s=0.75 | 0.023634 | **0.018664** | 21.03% | [0.002597, 0.008395] | 10/0 | 0.001953 |
| distill step3→4，368p transfer | 0.042859 | **0.040704** | 5.03% | [0.001481, 0.002993] | 10/0 | 0.001953 |

step40 的 strength=0.5 和 1.0 在 L1 上都没有可靠优势，0.75 是明显最佳点。step45 的最终 0.75 还带来 +2.225 dB PSNR、44.79% MSE 下降和 21.24% latent temporal L1 下降。注意这里的 temporal L1 是相对完整轨迹低分辨率终点的潜变量误差；它与最终 720p 视频的人工时序偏好不是同一量，二者方向可以不同。TAA 以 rank-32 LoRA 实现，但论文按其轨迹对齐功能命名。

## 6. 端到端因子效应

| Sampler | Trilinear Handoff | TAA+Trilinear | CLL-only | TALH | CLL Gain | TAA Gain under CLL | Overall TALH Gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wan50 @40 (TALH-Q) | 0.778116 | 0.779010 | 0.808963 | **0.809830** | +0.030846 | +0.000868 | +0.031714 |
| Wan50 @45 (TALH-E) | 0.767761 | 0.767335 | **0.808420** | 0.807920 | +0.040659 | -0.000500 | +0.040158 |
| Distill4 @3-of-4 (TALH-D4) | 0.817963 | 0.819085 | 0.856699 | **0.856799** | +0.038736 | +0.000100 | +0.038836 |

VBench-5 是 subject/background consistency、motion smoothness、aesthetic quality、imaging quality 的非加权均值。step40 strength sweep 中，0.75 是最终 TALH 的最佳 VBench-5；相对 CLL-only Handoff，subject、background、motion 和 imaging 分别为小幅正增益，但 aesthetic 下降 0.00455。所有 TAA 下游效应都远小于 CLL 主效应。

论文表述建议：

- 可写：“TAA 稳定缩小 residual trajectory gap，并在人工评价中恢复更多细节。”
- 不宜写：“TAA 显著提高端到端 VBench”或“TAA 全面提升视频质量”。
- 完整系统的端到端优势应归因于“CLL 提供主要质量保持，TAA 提供轨迹对齐与细节—时序控制”。

## 7. 人工盲评

统计单位为 10 个 prompt-majority，每个 prompt 有 3 位评价者。step45 的结果最完整：

| 比较 | 细节 胜/负/平 | 整体 胜/负/平 | 时序 胜/负/平 |
|---|---:|---:|---:|
| TAA vs Unaligned，均用 CLL | **9/0/1** | **6/0/4** | 0/8/2 |
| CLL-only vs Trilinear Handoff | **10/0/0** | **10/0/0** | **8/0/2** |
| TALH-E vs Trilinear Handoff | **10/0/0** | **10/0/0** | **9/0/1** |

TAA 的 detail、overall、temporal 双侧 sign p 分别为 0.003906、0.03125、0.007812。step40 strength=0.75 同方向：detail 8/2/0，overall 6/2/2，temporal 1/9/0；只有时序差异达到 p<0.05。最准确的解释是 TAA 将结果推向更强细节，但可能引入跨帧高频变化；本文将其作为方法现有的质量权衡，不再追加 temporal loss 或 strength schedule 实验。

Fleiss kappa 多数低或为负，但 observed agreement 与 expected agreement 在一边倒/高平票任务中非常接近，存在 prevalence paradox。正文不应只给 kappa；应同时给 prompt-level 胜负和平局。

## 8. 切换点与 Endpoint Re-entry Baseline

Endpoint Re-entry Baseline 将时间进一步降至 86.454±2.874 秒，达到 2.928× 加速，但 VBench-5 降至 0.800927，较 Native-HR 低 3.31%。它的配对 VBench-5 差值为 -0.02744，95% CI [-0.05128, -0.00577]，并在 9/10 个 prompt 上更低（p=0.02148）。因此一个 HR refinement step 是有效的极限速度基线，却比 TALH-Q/E 付出更明确的质量代价。

TALH-Q 相对 TALH-E 的 VBench-5 高 0.00191，95% CI [-0.00049, 0.00425]。逐维 bootstrap 显示 TALH-Q 的 subject +0.00354、aesthetic +0.01419；TALH-E 的 imaging 高 0.00939，但 CI 跨零。

同一混合分辨率 schedule 下，Wan50 从 Trilinear Handoff@45 的 104.397 秒到 TALH-E 的 112.534 秒，VBench-5 增加 0.04016；distill4 从 167.095 秒到 170.294 秒，VBench-5 增加 0.03884。这组数字衡量的是 TAA+CLL 相对廉价三线性提升的额外开销，不应与 Native-HR 加速比混为一谈。

## 9. Joint Trajectory-Scale Lifter（JTSL）证据边界

现有 sweep 视频支持“JTSL 相对 CLL 没有明显提升且更容易模糊”的观察，但闭环结果中没有等骨干、等预算的正式 CLL/JTSL 质量表。旧 timing summary 的 direct-720p 264.75 秒与 joint bridge 161.37 秒只证明旧 bridge 有 1.64× 时间收益，不能证明其质量，也不能替代最终 TALH 的效率表。

因此正文只讨论信息论动机与定性观察，不把“JTSL 必然损失信息”或“JTSL 显著差于 CLL”写成已由统计验证的结论，也不再追加 JTSL 定量对照。

## 10. 可复现文件

- `integrated_20260718/integration_manifest.json`：输入快照、归档哈希、最终 checkpoint/strength 和剩余缺口。
- `integrated_20260718/compiled_tables/operator_*.csv`：CLL operator 主表（文件名保留内部标签）。
- `integrated_20260718/compiled_tables/wan50_step40_endpoint_paired_statistics.csv`：step40 strength sweep 配对统计。
- `integrated_20260718/compiled_tables/wan50_step45_final_endpoint_paired_statistics.csv`：从 strength=0.75 原始样本重算的最终 step45 表。
- `integrated_20260718/compiled_tables/factorial_vbench_effects.csv`：三套 sampler 的可分离 VBench-5 效应。
- `integrated_20260718/compiled_tables/quality_efficiency_summary.csv`：相对 Native-HR 的加速与质量损失。
- `integrated_20260718/compiled_tables/wan50_final_quality5_paired_statistics.csv`：Native-HR 与 TALH 的逐 prompt 复合 VBench-5 配对统计（文件名保留内部标签）。
- `integrated_20260718/compiled_tables/human_review_prompt_*.csv`：prompt-majority 盲评统计。
