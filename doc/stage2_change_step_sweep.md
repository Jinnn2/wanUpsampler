# Stage 2 Change-Step Sweep 实验：原因、思路与结果分析

> **脚本路径**: [`run_clean_480p720p_stage2_change_step_sweep.sh`](../changing_resolution/scripts/eval/run_clean_480p720p_stage2_change_step_sweep.sh)
>
> **文档版本**: 2026-05-20 · 初始撰写

---

## 1 背景：Changing Resolution 中的切换步（Change Step）

在 Wan2.1 的 changing resolution 流程中，视频生成分为两个阶段：

1. **低分辨率阶段**（480p, latent `60×104`）：模型在低分辨率下执行前 N 步去噪
2. **高分辨率阶段**（720p, latent `90×156`）：在第 N 步将 latent 上采样至高分辨率，继续去噪至结束

**切换步（change step）** 就是这个 N——它决定了低分辨率去噪到哪一步时，执行分辨率切换。

切换步的选择直接影响：

| 切换步过早 | 切换步过晚 |
|-----------|-----------|
| 低分辨率去噪尚未收敛，clean estimate 质量差 | 高分辨率去噪步数不足，细节恢复不充分 |
| 上采样输入噪声大，后续高分辨率修正负担重 | 计算效率高（更多步在低分辨率完成） |

因此，**找到最优切换步是 changing resolution 流程的关键超参数问题**。

---

## 2 实验动机

### 2.1 为什么需要 Sweep？

之前的评估脚本（[`run_clean_480p720p_stage2_chain_ab_compare.sh`](../changing_resolution/scripts/eval/run_clean_480p720p_stage2_chain_ab_compare.sh)）只在**固定切换步**（默认 `CHANGE_STEP=35`）下比较 interp 与 Stage 2 的效果。这无法回答：

- Stage 2 模型在不同切换步下的鲁棒性如何？
- 最优切换步是否与 interp baseline 一致？
- 是否存在一个切换步区间，Stage 2 相对 interp 的优势特别显著？

### 2.2 核心问题

> **Stage 2 learned resizer 在不同切换步下的表现，是否始终优于 trilinear interp？最优切换步是多少？**

---

## 3 实验设计

### 3.1 三路对比方案

对每个 `(prompt, seed, change_step)` 组合，脚本生成三路视频：

| 标签 | 模型类 | 含义 |
|------|--------|------|
| **stop480** | `wan2.1_partial_denoise_decode` | 在第 N 步停止去噪，直接解码 480p 结果（低分辨率基线） |
| **interp720** | `wan2.1_clean_interp_bridge` | 在第 N 步用 **trilinear 插值** 上采样至 720p，继续去噪至第 50 步 |
| **stage2_720** | `wan2.1_clean_resizer_bridge` | 在第 N 步用 **Stage 2 learned resizer** 上采样至 720p，继续去噪至第 50 步 |

三路视频通过 ffmpeg 水平拼接为对比面板：

```
┌──────────────┬──────────────┬──────────────┐
│  stop480     │  interp720   │  stage2_720  │
│  step N      │  step N→50   │  step N→50   │
└──────────────┴──────────────┴──────────────┘
```

### 3.2 切换步范围

脚本通过三个参数控制切换步的遍历范围：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `STEP_START` | 10 | 起始切换步 |
| `STEP_END` | 50 | 结束切换步（含） |
| `STEP_STRIDE` | 5 | 步长 |

默认配置生成切换步序列：`10, 15, 20, 25, 30, 35, 40, 45, 50`（共 9 个点）。

也可通过 `CHANGE_STEPS` 显式指定任意步列表：

```bash
CHANGE_STEPS="20 30 35 40" bash run_clean_480p720p_stage2_change_step_sweep.sh
```

### 3.3 Prompt 与 Seed

- Prompt 来源：[`wan_t2v_generate_720p_prompts.txt`](../changing_resolution/configs/wan_t2v_generate_720p_prompts.txt)
- 默认选取 4 条 prompt（`LIMIT=4`），可通过 `PROMPT_OFFSET` 偏移
- Seed 由 `START_SEED + global_index` 计算，确保同一 prompt 在不同切换步下使用相同 seed

### 3.4 输出结构

```
outputs/changing_resolution_stage2_change_step_sweep/
├── configs/           # 每个样本的推理配置 JSON
│   ├── 000_step10_stop480.json
│   ├── 000_step10_interp720.json
│   ├── 000_step10_stage2_720.json
│   └── ...
├── stop480/           # 480p 停止去噪视频
│   └── 000_step10_stop480.mp4
├── interp720/         # 插值上采样视频
│   └── 000_step10_interp720.mp4
├── stage2_720/        # Stage 2 上采样视频
│   └── 000_step10_stage2_720.mp4
└── compare/           # 三路对比面板
    ├── 000_step10_panel_stop480.mp4
    ├── 000_step10_panel_interp720.mp4
    ├── 000_step10_panel_stage2_720.mp4
    └── 000_step10_step_sweep_compare.mp4
```

### 3.5 关键配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HR_H / HR_W` | 720 / 1248 | 高分辨率尺寸 |
| `LR_H / LR_W` | 480 / 832 | 低分辨率尺寸 |
| `INFER_STEPS` | 50 | 总推理步数 |
| `GUIDE_SCALE` | 6 | CFG 引导系数 |
| `SAMPLE_SHIFT` | 8 | 噪声调度偏移 |
| `USE_EMA` | 0 | Stage 2 短训练时 EMA 可能滞后，默认使用原始权重 |
| `STAGE2_RESIDUAL_SKIP` | checkpoint | 残差跳跃连接策略，默认从 checkpoint 读取 |
| `SKIP_EXISTING` | 1 | 跳过已生成的视频，支持断点续跑 |

### 3.6 多 GPU 并行

[`run_clean_480p720p_stage2_change_step_sweep_multigpu.sh`](../changing_resolution/scripts/eval/run_clean_480p720p_stage2_change_step_sweep_multigpu.sh) 将 prompt 按数量均分到多个 GPU，每个 GPU 运行单 GPU 脚本的子集。

```bash
# 4 GPU × 4 prompt，步长 1（41 个切换步）
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_change_step_sweep_multigpu.sh
```

默认 4 GPU 配置下，41 个切换步 × 4 prompt = 164 个三路面板。

---

## 4 技术细节：切换步在 Bridge 中的实现

### 4.1 调度器拦截

[`WanScheduler4CleanResizerBridge`](../changing_resolution/lightx2v_clean_bridge.py:35) 继承自 `WanScheduler4ChangingResolution`，在 `step_post_upsample()` 中执行：

```python
# 1. 估计 clean latent
x0_pred = sample - sigma_t * model_output

# 2. 用 learned resizer 上采样
clean_sample = self._resize_clean_latent_to_next_stage(x0_pred, target_shape)

# 3. 重新加噪，继续高分辨率去噪
noisy_sample = self.add_noise(clean_sample, ..., timesteps[step_index + 1])
```

### 4.2 配置注入

脚本通过 [`write_config()`](../changing_resolution/scripts/eval/run_clean_480p720p_stage2_change_step_sweep.sh:101) 函数动态生成推理配置 JSON，将 `changing_resolution_steps` 设为当前遍历的切换步值，并注入 Stage 2 模型路径、checkpoint、训练配置等信息。

### 4.3 三种模式的配置差异

| 配置项 | stop480 | interp720 | stage2_720 |
|--------|---------|-----------|------------|
| `target_height/width` | LR 尺寸 | HR 尺寸 | HR 尺寸 |
| `changing_resolution` | — | true | true |
| `resolution_rate` | — | [0.667] | [0.667] |
| `changing_resolution_steps` | — | [N] | [N] |
| `wan_clean_resizer_*` | — | — | ✅ 注入 |
| `stop_after_steps` | N | — | — |

---

## 5 结果分析

> ⚠️ **本节为预留模板，待实验完成后填写。**

### 5.1 定性评估

#### 5.1.1 切换步对画面质量的影响趋势

| 切换步区间 | 预期观察 | 实际观察 |
|-----------|---------|---------|
| 10–15（极早切换） | 低分辨率去噪不足，clean estimate 噪声大；interp 和 stage2 均可能出现模糊/伪影 | _待填写_ |
| 20–30（较早切换） | 低分辨率已初步收敛，高分辨率有充足步数修正 | _待填写_ |
| 30–40（中间区间） | 平衡点区域，预期 stage2 优势最明显 | _待填写_ |
| 45–50（极晚切换） | 高分辨率步数极少，细节恢复不足；可能仅 1–5 步高分辨率去噪 | _待填写_ |

#### 5.1.2 Stage 2 vs Interp 在不同切换步下的视觉差异

| 切换步 | Stage 2 优势描述 | Interp 劣势描述 |
|--------|-----------------|----------------|
| 10 | _待填写_ | _待填写_ |
| 15 | _待填写_ | _待填写_ |
| 20 | _待填写_ | _待填写_ |
| 25 | _待填写_ | _待填写_ |
| 30 | _待填写_ | _待填写_ |
| 35 | _待填写_ | _待填写_ |
| 40 | _待填写_ | _待填写_ |
| 45 | _待填写_ | _待填写_ |
| 50 | _待填写_ | _待填写_ |

#### 5.1.3 时序一致性

- **闪烁（flicker）**：_待填写_
- **纹理爬行（texture crawl）**：_待填写_
- **主体变形（subject deformation）**：_待填写_

### 5.2 定量评估

> 以下指标需基于 `stop480` 解码后的 480p 视频、`interp720` 和 `stage2_720` 解码后的 720p 视频计算。

#### 5.2.1 指标汇总表

| 切换步 | interp PSNR↑ | stage2 PSNR↑ | interp SSIM↑ | stage2 SSIM↑ | interp LPIPS↓ | stage2 LPIPS↓ |
|--------|-------------|-------------|-------------|-------------|--------------|--------------|
| 10 | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ |
| 15 | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ |
| 20 | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ |
| 25 | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ |
| 30 | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ |
| 35 | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ |
| 40 | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ |
| 45 | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ |
| 50 | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ | _待填写_ |

#### 5.2.2 Stage 2 相对 Interp 的增益曲线

| 切换步 | ΔPSNR (stage2 − interp) | ΔSSIM (stage2 − interp) | ΔLPIPS (interp − stage2) |
|--------|------------------------|------------------------|------------------------|
| 10 | _待填写_ | _待填写_ | _待填写_ |
| 15 | _待填写_ | _待填写_ | _待填写_ |
| 20 | _待填写_ | _待填写_ | _待填写_ |
| 25 | _待填写_ | _待填写_ | _待填写_ |
| 30 | _待填写_ | _待填写_ | _待填写_ |
| 35 | _待填写_ | _待填写_ | _待填写_ |
| 40 | _待填写_ | _待填写_ | _待填写_ |
| 45 | _待填写_ | _待填写_ | _待填写_ |
| 50 | _待填写_ | _待填写_ | _待填写_ |

> 📊 **建议绘图**：以切换步为 X 轴，ΔPSNR / ΔSSIM / ΔLPIPS 为 Y 轴，绘制增益随切换步变化的折线图。

### 5.3 最优切换步分析

#### 5.3.1 推荐切换步

- **仅考虑质量**：_待填写_（切换步 = N，此时 stage2 的 PSNR/SSIM 最高，LPIPS 最低）
- **质量-效率权衡**：_待填写_（切换步 = M，在可接受质量损失下最大化低分辨率步数占比）
- **与 interp 最优切换步对比**：_待填写_（Stage 2 是否改变了最优切换步的位置？）

#### 5.3.2 鲁棒性分析

- Stage 2 在切换步 20–40 区间内是否保持稳定优势？_待填写_
- 是否存在切换步使得 Stage 2 劣于 interp？_待填写_
- Stage 2 对切换步的敏感度是否低于 interp？_待填写_

### 5.4 典型案例

#### 案例 1：Stage 2 显著优于 Interp

- **Prompt**：_待填写_
- **切换步**：_待填写_
- **对比视频**：`compare/XXX_stepYY_step_sweep_compare.mp4`
- **分析**：_待填写_

#### 案例 2：Stage 2 与 Interp 效果接近

- **Prompt**：_待填写_
- **切换步**：_待填写_
- **对比视频**：`compare/XXX_stepYY_step_sweep_compare.mp4`
- **分析**：_待填写_

#### 案例 3：Stage 2 劣于 Interp（如有）

- **Prompt**：_待填写_
- **切换步**：_待填写_
- **对比视频**：`compare/XXX_stepYY_step_sweep_compare.mp4`
- **分析**：_待填写_

### 5.5 结论与后续行动

| 项目 | 内容 |
|------|------|
| 推荐默认切换步 | _待填写_ |
| Stage 2 是否在所有切换步下优于 interp | _待填写_ |
| 是否需要调整训练策略 | _待填写_ |
| 后续实验方向 | _待填写_ |

---

## 6 快速复现

### 6.1 单 GPU 运行

```bash
# 默认配置（步长 5，4 prompt）
bash changing_resolution/scripts/eval/run_clean_480p720p_stage2_change_step_sweep.sh

# 密集扫描（步长 1）
STEP_START=10 STEP_END=50 STEP_STRIDE=1 \
bash changing_resolution/scripts/eval/run_clean_480p720p_stage2_change_step_sweep.sh

# 指定切换步
CHANGE_STEPS="20 30 35 40" \
bash changing_resolution/scripts/eval/run_clean_480p720p_stage2_change_step_sweep.sh
```

### 6.2 多 GPU 运行

```bash
# 4 GPU × 4 prompt，步长 1
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_change_step_sweep_multigpu.sh

# 自定义 GPU 和 prompt 数
GPU_IDS=0,1 TOTAL_PROMPTS=8 \
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_change_step_sweep_multigpu.sh
```

### 6.3 常用覆盖参数

```bash
CR_STAGE2_CHAIN_COMPARE_CKPT=/path/to/latest.pt  # 指定 checkpoint
CR_STAGE2_CONFIG=/path/to/train_config.yaml       # 指定训练配置
USE_EMA=0                                          # 使用原始权重（短训练推荐）
STAGE2_RESIDUAL_SKIP=checkpoint                    # 残差跳跃策略
LIMIT=8                                            # 增加 prompt 数量
```

---

## 附录 A：脚本调用链

```
tmux_run_clean_480p720p_stage2_change_step_sweep_multigpu.sh
  └─ run_clean_480p720p_stage2_change_step_sweep_multigpu.sh
       └─ run_clean_480p720p_stage2_change_step_sweep.sh  (per GPU)
            ├─ write_config()  → 生成 JSON 配置
            ├─ run_infer()     → 调用 run_lightx2v_clean_bridge_infer.py
            │    └─ WanCleanResizerBridgeRunner / WanCleanInterpBridgeRunner
            ├─ make_labeled_panel()  → ffmpeg 添加标签
            └─ ffmpeg hstack          → 三路水平拼接
```

## 附录 B：相关文件索引

| 文件 | 说明 |
|------|------|
| [`stage2_resizer.py`](../wan_sr/models/stage2_resizer.py) | Stage 2 模型定义（`WanCleanLatentResizerStage2`） |
| [`lightx2v_clean_bridge.py`](../changing_resolution/lightx2v_clean_bridge.py) | LightX2V Bridge 集成（调度器拦截 + resizer 调用） |
| [`run_lightx2v_clean_bridge_infer.py`](../changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_infer.py) | 推理入口脚本 |
| [`train_clean_480p_to_720p_lmdb_stage2.yaml`](../changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml) | Stage 2 训练配置 |
| [`STAGE2_MODEL_PLAN.md`](../changing_resolution/STAGE2_MODEL_PLAN.md) | Stage 2 模型设计文档 |
| [`STAGE2_RUNBOOK.md`](../changing_resolution/STAGE2_RUNBOOK.md) | Stage 2 运维手册 |
