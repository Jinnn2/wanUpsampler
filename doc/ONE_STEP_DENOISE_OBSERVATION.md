# 🔍 一步去噪效果观察

> **背景**：Stage 2 模型在训练阶段接收的是 **纯 clean latent**（`z0_lr`），
> 但在 LightX2V 推理桥接中，实际输入是 **一步去噪估计值**（`x0_pred`），
> 两者之间存在分布差异。本文档记录该差异的观察方法与结论。

---

## 一、问题来源

推理时，LightX2V 桥接代码 [`step_post_upsample()`](../changing_resolution/lightx2v_clean_bridge.py:71) 执行以下逻辑：

```python
# 一步去噪：从当前 noisy latent 估计 clean latent
x0_pred = sample - sigma_t * model_output    # x_t → z0_hat
denoised_sample = x0_pred.to(sample.dtype)

# 调用 Stage 2 进行 1.5× 空间放大
clean_sample = self._resize_clean_latent_to_next_stage(
    denoised_sample, target_latent_shape
)

# 重新加噪，继续高分辨率采样
noisy_sample = self.add_noise(clean_sample, ...)
```

> ⚠️ **关键矛盾**：训练输入是 `z0_lr`（纯净 latent），推理输入是 `x0_pred`（带残留噪声/伪影的一步估计值）。

---

## 二、观察方案

### 2.1 对比维度

| 对比项 | 输入来源 | 含义 |
|--------|----------|------|
| **纯净基线** | `z0_lr`（训练域） | VAE 直接编码得到的真实 clean latent |
| **一步去噪** | `x0_pred`（推理域） | 从 noisy latent 经一步去噪得到的估计值 |
| **多步去噪** | `x0_pred_N` | 从 noisy latent 经 N 步去噪得到的估计值 |

### 2.2 观察方法

借助 [`WanPartialDenoiseDecodeRunner`](../changing_resolution/lightx2v_clean_bridge.py:271)，在 **不切换分辨率** 的前提下：

1. 运行 Wan 采样 N 步（`stop_after_steps`），取得中间态 `x_t` 和噪声预测 `eps_pred`
2. 计算 `x0_pred = x_t - sigma_t * eps_pred`
3. 将 `x0_pred` 送入 Stage 2 resizer，VAE decode 得到 RGB 视频
4. 对照组：用 `z0_lr`（真实 clean latent）走同样的 resizer + decode
5. 并排对比，观察：

```text
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  x_t (noisy) │ → │ x0_pred 估计  │ → │ Stage 2 → decode │  ← 推理域
└─────────────┘    └──────────────┘    └─────────────────┘

┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  z0_lr (纯净)│ → │     直接      │ → │ Stage 2 → decode │  ← 训练域（对照）
└─────────────┘    └──────────────┘    └─────────────────┘
```

---

## 三、关注要点

| 观察维度 | 具体检查内容 |
|----------|-------------|
| **主体保真度** | `x0_pred` 输入下，放大后的主体是否变形/模糊 |
| **边缘质量** | 高频细节（文字、纹理）是否受残留噪声影响 |
| **时间一致性** | 帧间是否因噪声残留引入额外闪烁 |
| **色彩偏移** | VAE decode 后 RGB 色彩是否有可察觉漂移 |
| **伪影** | 是否出现块状、网格状或振铃伪影 |
| **Sigma 敏感度** | 不同切换步数（不同 sigma 水平）下的退化程度 |

---

## 四、预期结论框架

```
┌────────────────────────────────────────────────────────────┐
│  ✅ 可接受：x0_pred 与 z0_lr 输入下 decode 结果无明显差异    │
│     → Stage 2 对输入中的轻微残留噪声具有鲁棒性              │
│     → 推理桥接可以安全使用一步去噪估计                      │
├────────────────────────────────────────────────────────────┤
│  ⚠️ 需注意：在较大 sigma（较早切换步数）下出现可见退化       │
│     → 建议限制切换步数范围（如 step ≥ 20）                  │
│     → 或考虑对 x0_pred 做轻量后处理再送入 resizer           │
├────────────────────────────────────────────────────────────┤
│  ❌ 不可接受：几乎所有 sigma 下均出现严重退化                │
│     → 需重新训练：训练数据中加入一步去噪样本                 │
│     → 或在 Stage 2 模型中加入 sigma 条件注入                │
└────────────────────────────────────────────────────────────┘
```

---

## 五、相关代码索引

| 文件 | 作用 |
|------|------|
| [`lightx2v_clean_bridge.py`](../changing_resolution/lightx2v_clean_bridge.py) | 桥接调度器：`step_post_upsample()` 一步去噪 + 重采样 + re-noise |
| [`lightx2v_clean_bridge.py:271`](../changing_resolution/lightx2v_clean_bridge.py:271) | `WanPartialDenoiseDecodeRunner`：在 N 步后解码 x0_pred |
| [`transition.py`](../wan_sr/pipelines/transition.py) | 独立推理桥接函数 `transition_lr_to_hr()` |
| [`noise_utils.py`](../wan_sr/schedulers/noise_utils.py) | `add_flow_noise()` flow-style 加噪 |
