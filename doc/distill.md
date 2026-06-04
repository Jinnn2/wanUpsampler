# Distill 训练记录

> 当前链路：`changing_resolution_distill`  
> 当前状态：10k 训练已完成，30k 训练进行中  
> 当前主线：5k 14B CfgDistill latent pair，30k steps，EMA = 0.999

## 目标

`changing_resolution_distill` 是独立于原 50-step Stage 3 的新链路，面向 4-step Wan CfgDistill 模型做 480p -> 720p 的 latent handoff。

核心目标是在 4-step distill 采样中，把固定插值换成学习型 latent resizer：

```text
480p distill latent
  -> one-step x0_pred at handoff step
  -> learned latent resizer
  -> 720p clean latent estimate
  -> re-noise
  -> continue 720p 4-step distill sampling
```

这里的重点不是 RGB 超分，而是在 distill denoiser 的真实 handoff 域里学习 `x0_pred_lr -> z0_hr` 的桥接算子。

## 当前阶段

10k 训练已经完成，会附带以下视频用于观察：

| 视频 | 含义 |
| --- | --- |
| ORI480 | 4-step distill 原始 480p 结果 |
| interp | 4-step distill 固定插值升分辨率基线 |
| step1 | handoff step 1 对应模型的 bridge 结果 |
| step2 | handoff step 2 对应模型的 bridge 结果 |
| step3 | handoff step 3 对应模型的 bridge 结果 |
| with_ema | 使用 EMA 权重后的 bridge 结果 |

30k 训练正在进行，是当前主线。原因是当前数据是 5k latent pair，训练到 30k 时约等价 49 个 epoch；10k 更像链路验证点，30k 更适合作为主线观察模型是否稳定学到 distill handoff 分布。

当前 30k launcher：

```text
changing_resolution_distill/scripts/train/tmux_run_x0pred_480p720p_stage3_distill_5k_30k_ema999_steps_1_2_3_training.sh
```

配置：

```text
STEPS=1,2,3
MAX_STEPS=30000
EMA_DECAY=0.999
CR_DISTILL_STAGE3_TAG=14b_cfgdistill_5k
```

## latent pair 构建

distill 的数据构建入口在：

```text
changing_resolution_distill/scripts/data/build_x0pred_480p720p_stage3_distill_lmdb.py
```

上游 clean latent LMDB 默认来自 14B CfgDistill 720p 视频：

```text
data/changing_resolution_distill/lmdb_clean_480p720p_14b_cfgdistill_5k
```

一次构建 step1/2/3 的脚本：

```text
changing_resolution_distill/scripts/data/tmux_build_x0pred_480p720p_stage3_distill_lmdb_steps_1_2_3.sh
```

构建后输出：

```text
data/changing_resolution_distill/lmdb_x0pred_480p720p_stage3_14b_cfgdistill_5k_step1
data/changing_resolution_distill/lmdb_x0pred_480p720p_stage3_14b_cfgdistill_5k_step2
data/changing_resolution_distill/lmdb_x0pred_480p720p_stage3_14b_cfgdistill_5k_step3
```

pair 的合同是：

```text
clean LR latent z0_lr
  -> add 4-step distill flow noise at handoff step k
  -> run one wan2.1_distill denoiser forward
  -> x0_pred_lr = x_t - sigma_k * flow_pred
  -> train x0_pred_lr -> clean z0_hr
```

LMDB 字段语义：

- `x0_pred_lr`：distill handoff step 上的一步 clean estimate，是训练输入。
- `z0_lr`：clean LR latent，保留给低频约束和分析。
- `z0_hr`：clean HR latent，是训练 target。
- `meta.stage3_recipe`：记录 `mode=lightx2v_distill`、`recipe=distill_4step`、`model_cls=wan2.1_distill`、`handoff_step`、`sigma`、`sigma_next`、`denoising_step_list` 等。

## x0_pred 实现

`x0_pred` 由 `LightX2VDistillX0PredGenerator.make_x0_pred()` 生成。

关键代码：

```python
sigma = scheduler.sigmas[step_index].to(device=self.device, dtype=torch.float32)
x_t = scheduler.add_noise(z0_device, noise, sigma)

scheduler.latents = x_t
scheduler.step_pre(step_index=step_index)
self.runner.model.infer(self.runner.inputs)
flow_pred = scheduler.noise_pred.to(torch.float32)
x0_pred = scheduler.latents.to(torch.float32) - sigma * flow_pred
```

对应语义：

```text
z0_lr
  -> scheduler.add_noise(z0_lr, noise, sigma_k)
  -> one wan2.1_distill forward
  -> x0_pred_lr = x_t - sigma_k * flow_pred
```

默认 distill 参数：

```text
model_cls: wan2.1_distill
infer_steps: 4
denoising_step_list: 1000 750 500 250
sample_shift: 5
sample_guide_scale: 6
model: lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill
```

## 模型不重复加载

数据构建时，`LightX2VDistillX0PredGenerator.__init__()` 只初始化一次 LightX2V runner：

```python
self.runner = RUNNER_REGISTER[config["model_cls"]](config)
self.runner.init_modules()
```

之后每个样本只更新 input/run 状态并复用同一个 14B CfgDistill 模型：

```python
self.runner.input_info = input_info
self.runner.inputs = self.runner.run_input_encoder()
self.runner.init_run()
self.runner.model.infer(self.runner.inputs)
self.runner.end_run()
```

同样，720p raw video 生成脚本也采用常驻 runner：

```text
changing_resolution_distill/scripts/data/generate_wan21_distill_720p_dataset.py
```

其中 `runner.init_modules()` 只执行一次，然后循环生成 prompt shard。这个设计避免 5k 数据构建时反复加载 14B checkpoint。

## 训练实现

distill 训练 wrapper：

```text
changing_resolution_distill/scripts/train/run_x0pred_480p720p_stage3_distill_lmdb_training.sh
changing_resolution_distill/scripts/train/tmux_run_x0pred_480p720p_stage3_distill_lmdb_steps_1_2_3_training.sh
```

训练器复用现有 Stage 3 trainer，但由 distill wrapper 传入 distill LMDB 和 `--denoise_step ${HANDOFF_STEP}`，用于保证 checkpoint 与 handoff step 对齐。

配置文件：

```text
changing_resolution_distill/configs/train_x0pred_480p_to_720p_lmdb_stage3_distill.yaml
```

默认配置中的核心项：

```yaml
stage3:
  denoise_step: 2
  recipe: distill_4step

model:
  in_channels: 16
  out_channels: 16
  scale_factor: 1.5
  residual_skip: false
  resblock_type: ltx2
  resize_op: rational_conv3d_pixel_shuffle
```

30k 主线脚本会覆盖：

```text
MAX_STEPS=30000
EMA_DECAY=0.999
```

step1/2/3 各自对应独立 LMDB、独立输出目录、独立 checkpoint。

## 桥接细节

distill runtime bridge 在：

```text
changing_resolution_distill/lightx2v_distill_bridge.py
```

注册的 runner：

```text
wan2.1_distill_clean_resizer_bridge
wan2.1_distill_interp_bridge
```

scheduler 是 4-step distill 专用：

```python
class WanStepDistillScheduler4CleanResizerBridge(WanStepDistillScheduler):
```

handoff 时执行：

```python
flow_pred = self.noise_pred.to(torch.float32)
sample = self.latents.to(torch.float32)
sigma = self.sigmas[self.step_index].to(device=sample.device, dtype=torch.float32)
x0_pred = sample - sigma * flow_pred
clean_sample = self._resize_clean_latent_to_next_stage(x0_pred.to(sample.dtype))
```

如果不是最后一步，会 re-noise 到下一步继续 HR distill：

```python
sigma_next = self.sigmas[self.step_index + 1].to(device=sample.device, dtype=torch.float32)
target_noise = self.latents_list[self.changing_resolution_index + 1].to(torch.float32)
noisy_sample = self.add_noise(clean_sample.to(torch.float32), target_noise, sigma_next)
self.latents = noisy_sample.to(dtype=self.latents.dtype)
```

默认 re-noise 模式是 `random`：

```text
random: x_next_hr = add_noise(x0_hr, fixed_hr_noise, sigma_next)
```

另有 `resize_flow` 作为 ablation（我测试了，效果极差，舍弃）：

```text
resize_flow: x_next_hr = x0_hr + sigma_next * trilinear_resize(flow_pred_lr)
```

如果配置 `wan_clean_resizer_use_ema=True` 且 checkpoint 内有 `ema`，会把 EMA 权重 copy 到模型用于推理。

## 简单分析

10k 的价值是证明 distill 新链路可跑通，包括 14B CfgDistill 数据生成、clean latent LMDB、step1/2/3 x0_pred LMDB、训练、bridge 推理和视频对比。

30k 是当前主线，因为 5k latent pair 在 30k 下约 49 个 epoch，更适合作为最终阶段观察点。需要重点看两件事：

- 30k EMA 相比 10k EMA 是否稳定提升。
- 30k 是否开始出现过拟合，例如纹理记忆、局部锐化、重复细节或时序抖动。

对视频判断时，优先看 `with_ema`。如果 raw 更锐但 EMA 的主体边缘、运动连续性和高频稳定性更好，应优先把 EMA 当成候选模型。

