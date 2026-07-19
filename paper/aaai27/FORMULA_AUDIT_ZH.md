# 英文稿公式与符号核对记录

核对范围：`main_zh.md`、`main_polished.md` 与最终排版源 `body_polished.tex`。

## 结论

未发现由翻译造成的变量遗漏、上下标颠倒、系数变化或分辨率尺寸错误。英文 Markdown 有 13 个独立展示公式，而中文稿有 14 个；数量差异仅因为速度监督目标

\[
v^*=\frac{x_s^L-z_T^L}{\sigma_s}
\]

在英文稿中被保留为行内公式，并非内容缺失。LaTeX 为节省 AAAI 正文空间，将三个 CLL 损失分量合并进同一个 `align` 环境，但数学定义不变。

## 已逐项核对

1. TALH 数据流：`x_s^L -> TAA -> \widetilde z_s^L -> CLL -> \widehat z_s^H -> HTR -> x_{s+1}^H`，上下标和模块顺序一致。
2. 切换误差分解：`E_handoff ≈ E_traj + E_lift + E_refine`，三项含义一致。
3. CLL 张量映射：输入、输出均为 16 通道，时间维 `F` 不变，仅空间网格从 `h x w` 变为 `H x W`。
4. `480x832 -> 720x1248` 的 latent 网格路径：`60x104 -> 180x312 -> 90x156`，即先 `3x` PixelShuffle、再 `/2` 下采样。
5. CLL 总损失、Charbonnier 重建项、低频内容项和一阶时间差分项的目标变量与权重一致；最终 LaTeX 记号 `lambda_c/lambda_t` 分别对应 Markdown 的 `lambda_content/lambda_temp`。
6. LoRA 更新：`W' = W + (alpha/r)BA`，矩阵顺序和缩放系数一致。
7. 流匹配 clean prediction：`\widetilde z_s^L = x_s^L - sigma_s v_{phi+Delta theta_s}(x_s^L,s,c)`，符号未被翻译改变。
8. TAA 对齐损失：`L1 + 0.1 L2^2`，系数保持为 `0.1`；step 40 的时间残差附加权重保持为 `0.05`。
9. cached-prefix 一致性：TAA 仅在切换计算激活时，`x_k^adapted = x_k^base, k <= s`；这里 `x_s` 是第 `s` 次计算前的状态，因此边界 `<= s` 正确。
10. CLL 输出：`\widehat z_s^H = U_CLL(\widetilde z_s^L)`，输入为 TAA 对齐后的低分辨率 clean latent。
11. HTR 重加噪：`x_{s+1}^H = (1-sigma_{s+1})\widehat z_s^H + sigma_{s+1} epsilon^H`，采用下一调度时刻和目标分辨率噪声。

## 语义修正

中文稿在解释 `E_refine` 时同时使用了“未修复误差”和“修复预算”两种说法，二者方向相反。公式将 `E_refine` 定义为 HTR 与高分辨率后缀之后仍未修复的误差，因此正式英文表述已统一为：更晚切换留下更少的高分辨率计算，可能增大 `E_refine`。这是一处概念口径修正，不是公式改动。

此外，跨尺度下采样算子统一记为 `D`，避免在 `D` 与 `\mathcal D` 之间混用。
