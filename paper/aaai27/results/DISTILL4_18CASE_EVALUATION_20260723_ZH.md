# Distill4 18-case 质量–效率评估

评估归档：`distill4_quality_efficiency_final_20260722T174918Z.tar.gz`

## 1. 数据与协议审计

- 归档内 `SHA256SUMS` 声明的 108 个文件均通过流式哈希校验。
- 18 个 case × 10 个 prompt，共 180 个视频；归档验证为 `complete`，无缺失、额外或损坏视频。
- 质量使用 10 个配对 prompt 的 VBench：subject/background consistency、motion smoothness、aesthetic quality、imaging quality；`quality5` 是五项等权平均。Dynamic degree 单独报告。
- 效率是单卡 resident-model warm 测量：1 个 warm-up、5 个配对重复、CUDA 同步；模型与 checkpoint 初始化不计入 pipeline latency。
- `pipeline_mean_s` 是可用于论文的端到端 warm latency。当前名为 `denoise_mean_s` 的字段实际是整个 `run_segment`，RGB case 中也包含 VAE decode、Real-ESRGAN 和 VAE encode，因此不能解释成纯 DiT 时间。

## 2. 主要结果

| Case | 计算路径 | Quality5 | Warm latency (s) | Speedup vs native | Dynamic degree |
|---|---|---:|---:|---:|---:|
| Native-HR4 | 4 HR | 0.860406 | 42.491 | 1.00× | 0.2 |
| Interp-D4@2 | 2 LR + 2 HR | **0.862099** | 26.350 | 1.61× | 0.5 |
| CLL-D4@3 | 3 LR + lift + 1 HR | 0.856699 | **18.274** | **2.33×** | 0.5 |
| TrajScale-D4@3 | 3 LR + TAA + lift + 1 HR | 0.856799 | 19.922 | 2.13× | 0.5 |
| Endpoint-Stage2-0HR | 4 LR + latent lift | 0.848409 | 10.228 | 4.15× | 0.5 |
| Endpoint-Stage2-1HR | 4 LR + latent lift + 1 HR | 0.856065 | 19.490 | 2.18× | 0.5 |
| Endpoint-RGB-0HR | 4 LR + RGB SR | 0.858839 | **17.790** | 2.39× | 0.5 |
| Endpoint-RGB-1HR | 4 LR + RGB SR + 1 HR | 0.859972 | 27.258 | 1.56× | 0.6 |

严格 Pareto frontier 还包含低质量但略快的 Endpoint-Interp-0HR；论文主图更适合标出 practical frontier：Stage2-0HR、RGB-0HR、CLL/Stage2-1HR 邻域和 Interp@2。

## 3. 能成立的结论

### 3.1 相对 Native-HR4

- Interp@2 将 latency 降低 37.99%，同时 Quality5 点估计提高 0.00169；10-prompt 配对 bootstrap CI 为 `[-0.0053, 0.0093]`，应表述为质量持平，而不是显著提升。
- CLL@3 将 latency 降低 56.99%（2.33× speedup），Quality5 仅下降 0.00371；配对 CI `[-0.0110, 0.0047]` 包含 0。
- TALH@3 将 latency 降低 53.11%（2.13×），Quality5 下降 0.00361；配对 CI 同样包含 0。

### 3.2 提前 lift 与相同 learned-lifter endpoint 的比较

- CLL@3 比 Endpoint-Stage2-1HR 快 1.216 s，即以 endpoint 为分母降低 6.24%；warm 配对时间 CI `[1.164, 1.268]` s。
- 两者 Quality5 分别为 0.856699 和 0.856065，差 0.000634；配对 CI 包含 0。因此“提前 learned latent lift 在同等质量下节省约 6% latency”可以成立。
- TALH@3 为 19.922 s，反而比 Endpoint-Stage2-1HR 慢 0.432 s（2.22%）。因此该结论只能由 CLL 支撑，不能由当前 TALH 支撑。

### 3.3 RGB 往返成本

- 同为 endpoint-0HR，RGB 比 Stage2 增加 7.561 s（+73.9%）。这是 VAE decode → Real-ESRGAN → VAE encode 的直接 warm 开销证据。
- 同为 endpoint-1HR，RGB 比 Stage2 增加 7.768 s（+39.9%）。
- 相对 Endpoint-RGB-1HR，CLL@3 快 8.984 s，以 RGB endpoint 为分母降低 32.96%；TALH@3 快 7.336 s，降低 26.91%。两者与 RGB-1HR 的 Quality5 差异均很小，prompt-paired Quality5 CI 包含 0。

这组结果可以用来说明：一旦 endpoint-first 流程还需要 latent 回编码并继续 HR refinement，提前在 latent 轨迹中 lift 明显更便宜。

### 3.4 LUVE-style 对比

LUVE-style 的受控对应项定义为 Endpoint-Stage2-2HR，即完成 4 个 LR
step、使用 learned Stage2 latent lift，再执行 2 个 HR step：

- Endpoint-Stage2-2HR：28.762 s，Quality5 0.859806。
- TALH@3：19.922 s，Quality5 0.856799。
- TALH 降低 8.840 s，即以 LUVE-style 为分母降低 **30.73%** latency；
  配对时间 CI `[8.663, 9.017]` s。
- LUVE-style 的 Quality5 高 0.00301；post-hoc prompt-paired Quality5
  bootstrap CI `[0.00078, 0.00570]`。应表述为 TALH 以很小质量差换取约 31%
  延迟下降，而不是严格质量无损。

## 4. MRFlow 对比口径

MRFlow 在 RGB 超分后仍包含高分辨率生成，因此受控对应项是
Endpoint-RGB-1HR，而不是 Endpoint-RGB-0HR。RGB-0HR 只用于分解 RGB
往返开销，不进入主对比；Interp@2 也不进入主表。

- Endpoint-RGB-1HR：27.258 s，Quality5 0.859972。
- CLL@3：18.274 s，Quality5 0.856699；比 RGB-1HR 快 8.984 s，以
  RGB endpoint 为分母降低 **32.96%** latency。
- TALH@3：19.922 s，Quality5 0.856799；比 RGB-1HR 快 7.336 s，降低
  **26.91%** latency。
- CLL/TALH 与 RGB-1HR 的 prompt-paired Quality5 差异均很小，95% CI
  包含 0。

因此当前实验支持：与 MRFlow-style 的低分辨率完成 → RGB 超分 →
高分辨率生成流程相比，提前在 latent 轨迹中升分辨率可降低约
27%–33% warm 端到端延迟，同时维持相近的 VBench Quality5。

## 5. 方法消融结论

- CLL@3 与 TALH@3 的 Quality5 几乎相同：差 0.00010，配对 CI `[-0.00165, 0.00194]`。
- TALH 比 CLL 慢 1.648 s（+9.02%），时间 CI `[1.467, 1.830]` s。
- TAA+Interp@3 相对 Interp@3 仅提高 0.00112 Quality5，但增加 1.684 s（+9.18%）。

当前 4-step checkpoint、step-3 LoRA 和 strength=0.75 下，TAA 没有提供可测的质量收益，却带来稳定额外时间。论文主效率点应优先使用 CLL@3；若 TALH 是核心方法名，则需要重新调 step-3 LoRA/strength，或解释它在其他证据上的必要性。

## 6. Endpoint budget 发现

- Stage2：Quality5 随 0→1→2 HR 从 0.8484→0.8561→0.8598，提高到 2HR 后饱和；4HR 降到 0.8557。
- Interp：0.7684→0.8165→0.8600→0.8557；2HR 才恢复到高质量。
- RGB：0.8588→0.8600→0.8597→0.8557；0HR 已经很强，追加 HR step 的收益很小而成本很大。
- 三种 4HR case 的 10 个输出逐 prompt **字节级 SHA256 完全相同**。实现从四步 suffix 的第一个高噪声状态重新加噪，lift 域信息被完全消除。4HR 不应作为普通 refinement 点解释，应重命名为 full-restart control，或从主曲线移到附录。

## 7. 统计与复现风险

- 质量仅有 10 个 prompt；时间仅有 5 个重复。当前时间差稳定（所有 case pipeline CV 约 0.17%–1.14%），但质量 CI 较宽。
- 41 个 comparison × 5 个维度共 205 次检验，没有 multiple-comparison correction。主文应预注册少量核心 contrast，其余放附录，或增加 Holm/FDR。
- Dynamic degree 是 0.1 粒度的离散比例，不应与连续五项直接平均；当前单独报告是正确的。
- 归档没有记录 GPU 型号、驱动、CUDA/PyTorch/LightX2V revision。正式表格前需要补一份硬件/软件元数据；无需重跑时间，只要确认测量机器配置并冻结记录。
- VBench 原始 JSON 文件名包含 `:`。tar 在 Linux 上完整，但 Windows 解包会改写为 `_`，导致解包后的 `SHA256SUMS` 路径无法直接匹配。发布 supplementary 前应将原始文件名规范化为跨平台格式。

## 8. 推荐论文叙事

主对比表建议只保留 Native-HR4、Interp@3、TALH@3、
Endpoint-Stage2-2HR（LUVE-style）和 Endpoint-RGB-1HR（MRFlow-style）。
CLL@3 与 TAA+Interp@3 放入消融表。核心叙事应为：

1. Interp@3 是无学习的直接插值基线，快但质量明显下降；
2. TALH@3 是主方法：相对 Native-HR4 为 2.13× 加速；
3. 相对 LUVE-style，TALH 降低 30.73% latency，Quality5 低 0.00301；
4. 相对 MRFlow-style，TALH 降低 26.91% latency，Quality5 低 0.00317，
   配对 CI 包含 0；
5. CLL 与 TAA+Interp 只用于分解 learned lift 和 TAA 的贡献；RGB-0HR
   与 Interp@2 不进入主对比表。
