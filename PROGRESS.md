# 理论进度

最后更新：2026-05-08

## 当前阶段

项目已经从 V1 noisy-to-clean 验证，推进到 V2 changing_resolution clean-latent 路线。

当前主线目标：

```text
z0_lr 或 x0_pred_lr -> z0_hr
480p clean latent -> 720p clean latent
```

核心判断是：在 LightX2V 的 changing_resolution 链路中，用训练得到的 clean latent resizer 替换固定插值后，是否比插值算子更接近真实 720p clean latent，并在生成链路中表现出更好的视觉稳定性。

## 已完成

- V1 最小训练闭环：完成模型、数据构建、训练、decode 评估和 LightX2V 对比脚本。
- V2 clean-latent 方案确认：changing_resolution 的接口本质上是 clean latent resize，因此训练目标改为 `z0_lr -> z0_hr`。
- 1k 数据构建机制：支持 Wan2.1 生成 720p 视频、多卡并行构建 480p/720p latent pair，并写入 LMDB。
- Stage1 训练脚本：支持 LMDB 读取、固定 train/val split、EMA、`best_val.pt`、`latest.pt` 和 step checkpoint。
- 双评估体系：
  - operator compare：以 validation LMDB 的 `ori720_decode` 为参考，比较 `interp720_decode` 和 `trained720_decode`。
  - chain A/B compare：在真实 LightX2V changing_resolution 链路中比较 `interp720` 和 `trained720`。
- 项目结构整理：V2 主线集中在 `changing_resolution/`，V1 历史流程集中到 `scripts/v1/` 和 `configs/v1/`。

## 当前要回答的问题

1. Stage1 模型是否在 operator compare 中稳定优于 trilinear interpolation。
2. 这种优势是否能传导到真实生成链路，而不是只在 VAE decode 的重建任务中成立。
3. 如果 Stage1 有效，下一阶段是否需要替换当前 “trilinear + residual” 结构为更强的 learned upsampling operator。

## 判断标准

operator compare 中期望：

```text
trained_psnr  > interp_psnr
trained_ssim  > interp_ssim
trained_lpips < interp_lpips
```

chain A/B 中主要看：

```text
边缘和主体结构是否更稳定
纹理是否减少爬动
运动是否减少闪烁
局部细节是否比插值更自然
```

## 下一步

优先级：

1. 在远端机器运行 stage1 operator compare，汇总 `summary_val.json`。
2. 对同一 checkpoint 运行 chain A/B compare，人工筛查视频质量。
3. 若 Stage1 优于插值，设计 Stage2：更大数据、更强上采样结构、更接近 LTX2 风格的 latent upsampler。
