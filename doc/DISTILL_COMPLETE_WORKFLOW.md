# WanUpsampler Distill 完整工作流

> 日期：2026-07-06  
> 范围：`changing_resolution`、`changing_resolution_distill`、`wan_sr`  
> 目标：把 Stage2 clean latent 升分、Stage3 `x_pred -> z` 尝试、切换点选择、LoRA 最后一步平滑，以及当前推荐工作流整理成一条完整逻辑链。

## 1. 总结

当前结论是：`x_pred_lr -> z_hr` 不再作为 distill 分支的最终主线。它可以保留为 baseline 和对照，但不应该继续承担最终方案。

推荐工作流拆成两层：

```text
LR denoising correction:
  Wan 4-step distill prefix
    -> step3 / final-step LoRA
    -> z_lr_clean

Clean latent upsample:
  z_lr_clean
    -> Stage2 clean latent upsampler
    -> z_hr_clean
    -> VAE decode
```

这个拆分的核心原因是：升分算子应该只处理 `z_lr_clean -> z_hr_clean`，不要同时承担“修复 denoiser 没去干净的误差”和“空间升分”两个任务。LoRA 放在 distill denoiser 的最后有效步上，用来把最后一步的去噪误差提前消化掉，使 Stage2 仍工作在 clean latent 域。

## 2. 起点：Stage2 的 z -> z 升分设计

Stage2 是整个项目最稳定的基础层。它定义的问题很干净：

```text
z0_lr -> z0_hr
```

其中 `z0_lr` 和 `z0_hr` 都是 VAE clean latent。模型只需要学习 480p latent 到 720p latent 的空间映射，不需要理解 timestep，也不需要修复 denoiser 的残留误差。

对应实现：

```text
model:
  wan_sr/models/stage2_resizer.py

trainer:
  changing_resolution/scripts/train/train_clean_latent_resizer_stage2.py

non-distill config:
  changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml

distill config:
  changing_resolution_distill/configs/train_clean_480p_to_720p_lmdb_stage2_distill.yaml
```

Stage2 的设计价值有三点：

| 价值 | 说明 |
| --- | --- |
| 任务边界清楚 | 只做 latent 空间 1.5x 升分 |
| 数据合同稳定 | 输入和目标都是 clean latent |
| 可复用性强 | 非 distill 50-step 和 distill 4-step 都能复用同一 upsampler 架构 |

Stage2 的局限也很明确：真实推理切换时，模型拿到的未必是严格 clean latent，而是 denoiser 当前步估出来的 `x0_pred`。这引出了 Stage3。

## 3. Stage3：从 x_pred -> z 的尝试

Stage3 的初衷是对齐真实 bridge 输入域。在 50-step Wan 链路中，handoff 时会先做一次 denoiser forward：

```text
x0_pred = x_t - sigma_t * noise_pred
```

因此 Stage3 把训练输入从 `z0_lr` 换成 `x0_pred_lr`：

```text
x0_pred_lr -> z0_hr
```

在非 distill 50-step 链路中，默认数据构建围绕 `denoise_step=45`，也支持 `45/46/47` 等多个 checkpoint。评估时通过 change-step sweep 扫描：

```text
stop480 at step N | interp720 step N -> 50 | stage3 720 step N -> 50
```

在 distill 4-step 链路中，对应改成 handoff step 1/2/3：

```text
step1 model: x0_pred_lr(step1) -> z0_hr
step2 model: x0_pred_lr(step2) -> z0_hr
step3 model: x0_pred_lr(step3) -> z0_hr
```

对应旧 distill Stage3 数据合同：

```text
clean LR latent z0_lr
  -> add 4-step distill flow noise at handoff step k
  -> run one wan2.1_distill denoiser forward
  -> x0_pred_lr = x_t - sigma_k * flow_pred
  -> train x0_pred_lr -> clean z0_hr
```

这条链路的工程价值是完整的：数据生成、训练、bridge、step1/2/3 对照、EMA/raw 对比都能跑通。它仍然是很重要的实验基线。

## 4. 为什么 x_pred -> z 会混入直接去噪损失

问题出在 `x0_pred_lr` 的语义。它不是纯 clean latent，而是当前 denoiser 在某个 timestep 上的一步 clean estimate：

```text
x0_pred_lr = clean_lr + residual_denoising_error
```

因此训练 `x0_pred_lr -> z0_hr` 时，小型 resizer 被迫同时学习：

```text
1. 修复 residual_denoising_error
2. 完成 LR latent -> HR latent 升分
```

这对 4-step distill 更明显。few-step distill 每一步跨度更大，`x0_pred` 对轨迹状态、step 位置和 scheduler 细节更敏感。一旦 `x0_pred_lr` 的残差模式和真实推理时不完全一致，upsampler 学到的就不是单纯升分，而是混合了“去噪修补”的补偿函数。

结果上会出现两个风险：

| 风险 | 表现 |
| --- | --- |
| 分布耦合 | checkpoint 只对某个 handoff step 特别敏感，跨 step 不稳 |
| 目标污染 | upsampler 可能用空间纹理去补 denoising error，导致细节、时序或主体结构不稳定 |

所以主线需要把 denoising correction 从 upsampler 里拆出去。

## 5. 切换点选择：50-step 与 4-step distill

切换点选择的本质是权衡：

```text
切得早:
  HR 后续步数多，但 LR clean estimate 更不稳定

切得晚:
  LR 轨迹更接近 clean，但 HR 后续修正步数少
```

### 5.1 50-step Wan

非 distill 50-step 链路有足够多的候选点，所以用 sweep 观察。历史文档和脚本里有两类策略：

| 策略 | 用途 |
| --- | --- |
| `CHANGE_STEPS=10..50` | 大范围评估 Stage2/Stage3 在不同 handoff step 的鲁棒性 |
| `45/46/47` | 在高效 late handoff 区间训练独立 Stage3 checkpoint 做细比较 |

50-step 里，handoff step 与训练数据必须匹配。Stage3 trainer 会读取 LMDB metadata 中的 `stage3_recipe.denoise_step`，发现不一致时提前失败。

### 5.2 4-step distill

4-step distill 的有效候选更少：

```text
denoising_step_list = [1000, 750, 500, 250]
handoff candidates = step1, step2, step3
```

step4 已经是结束步，不适合作为“切换后继续 HR denoise”的普通 handoff。因此旧 distill Stage3 同时构建 step1/2/3 三套数据和 checkpoint。

LoRA 路线里，切换点选择进一步收敛到 step3 / last-step-skip：

```text
base prefix:
  step1 -> step2

trainable final effective step:
  step3 + LoRA

target:
  teacher final clean latent after step4
```

这样做的原因是它直接消除“最后一步缺失或最后一步误差”对后续 clean upsampler 的影响。它不是在每个 handoff step 都训练一个 upsampler，而是先让 LR 分支在结束前得到更可靠的 `z_lr_clean`。

## 6. LoRA：平滑最后一步影响

LoRA 的目标不是升分，而是修正 Wan distill denoiser 的最后有效步：

```text
x_pre_step3_lr -> z_teacher_final
```

或者用 velocity 形式写：

```text
target_flow = (x_current - z_teacher_final) / sigma
loss = MSE(flow_pred_lora, target_flow)
```

它的作用是把原来 `x_pred -> z` 中混入 upsampler 的直接去噪损失，搬回 denoiser 自己身上。这样后续 Stage2 clean upsampler 看到的仍是 clean-ish latent，而不是强行带着 denoising error 的 `x0_pred`。

当前仓库里存在两类 LoRA 数据/训练思路：

| 路线 | 数据字段 | 当前定位 |
| --- | --- | --- |
| last-step-skip LMDB | `x_pre_step3_lr`, `z4_lr_teacher`, `z0_hr` | 主路径下仍有构建器和训练入口，用于 cached teacher 验证 |
| teacher-trajectory LMDB | `x_pre_train_step`, `z_teacher_final` | Plan E 方向，配置和 dataset 在主路径，builder/trainer/launcher 当前在 `old/` |

## 7. 当前选取的工作流

当前推荐的完整工作流如下：

```text
1. 保留 distill Stage2 clean upsampler
   z0_lr -> z0_hr

2. 保留旧 distill Stage3 x0_pred -> z0_hr 作为 baseline
   step1/2/3 checkpoints 用于证明 x_pred 分支的上限和问题

3. 训练最后一步 LoRA
   input:  x_pre_step3_lr or x_pre_train_step
   target: z4_lr_teacher or z_teacher_final

4. 用 LoRA 生成 clean LR latent
   noise -> step1 -> step2 -> step3 + LoRA -> z_lr_clean

5. 先接现有 distill Stage2 clean upsampler
   z_lr_clean -> z_hr_clean

6. 如果出现明显 distribution shift，再构建 LoRA-clean pair 微调 Stage2
   z_lr_lora_clean -> z_hr_clean
```

推荐的第一优先评估不是继续扩展 `x0_pred_lr -> z0_hr`，而是比较：

```text
teacher4 LR
3-step + LoRA LR
3-step + LoRA + existing Stage2 720p
teacher4 clean LR + existing Stage2 720p
interp720
```

如果 `3-step + LoRA LR` 已经接近 `teacher4 LR`，而 Stage2 接上后质量稳定，就说明拆分策略成立。

## 8. 当前代码状态

截至当前工作树：

```text
active docs:
  doc/distill.md
  doc/DISTILL_LAST_STEP_SKIP_LORA_PLAN.md
  doc/DISTILL_COMPLETE_WORKFLOW.md

active distill Stage2:
  changing_resolution_distill/configs/train_clean_480p_to_720p_lmdb_stage2_distill.yaml
  changing_resolution/scripts/train/train_clean_latent_resizer_stage2.py

active last-step-skip LoRA:
  changing_resolution_distill/scripts/data/build_last_step_skip_lora_lmdb.py
  changing_resolution_distill/scripts/train/train_last_step_skip_lora.py
  changing_resolution_distill/configs/train_last_step_skip_lora_distill.yaml

teacher trajectory Plan E config/dataset:
  changing_resolution_distill/configs/train_teacher_trajectory_lora_distill.yaml
  wan_sr/data/teacher_trajectory_lora_lmdb_dataset.py

teacher trajectory Plan E archived scripts:
  changing_resolution_distill/scripts/data/old/build_teacher_trajectory_lora_lmdb.py
  changing_resolution_distill/scripts/data/old/build_teacher_trajectory_lora_lmdb_multigpu.sh
  changing_resolution_distill/scripts/train/old/train_teacher_trajectory_lora.py
  changing_resolution_distill/scripts/train/old/run_teacher_trajectory_lora_training.sh
  changing_resolution_distill/scripts/train/old/run_last_step_skip_lora_plan_e_on_policy_rank16_qkvo_ffn.sh
```

这个状态意味着：如果下一步要恢复 Plan E 作为可直接运行入口，需要先把 `old/` 里的 teacher-trajectory builder/trainer/launcher 重新提升到活动路径，或者明确用 `old/` 入口运行。否则当前主路径更偏向 last-step-skip cached teacher 的 LoRA 工作流。

## 9. Plan E 的真实语义

`train_teacher_trajectory_lora_distill.yaml` 当前写的是：

```yaml
training_mode: on_policy
on_policy_loss_type: velocity_target
on_policy_rollout_source: cached_prefix
on_policy_active_steps: none
grad_accum: 1
ema_decay: null
```

这要谨慎解释。

它的目标 loss 是 velocity target，确实不再直接训练 `pred_clean` 的 L1/MSE：

```text
target_flow = (x_current - z_teacher_final) / sigma
velocity_mse(flow_pred, target_flow)
```

但默认 `on_policy_rollout_source: cached_prefix` 表示 train step 前的 `x_current` 直接来自 teacher-trajectory LMDB 缓存的 `x_pre_train_step`，不是每次都从 noise 用当前 LoRA 闭环滚到 step3。默认 `on_policy_active_steps: none` 也表示前缀步不启用 LoRA。

因此当前 Plan E 更准确的名字是：

```text
cached teacher prefix + final-step LoRA velocity-target training
```

它比旧的 `x_pred -> z` 更合理，因为 loss 回到了 denoiser velocity 上；但它还不是完整 D-OPSD 式的“student 自己访问状态 + EMA teacher”闭环训练。

要升级成更完整的 on-policy，需要：

```text
on_policy_rollout_source: recompute_prefix
on_policy_active_steps: all_before_train 或指定 step 列表
EMA teacher / shadow LoRA
teacher-student same-state velocity alignment
```

## 10. 需要优化的问题

### 10.1 入口整理

teacher-trajectory Plan E 的核心脚本当前在 `old/`，但 config 和 dataset 在主路径。这会让“当前模式”不够清晰。建议二选一：

```text
方案 A: 恢复 Plan E 到活动路径，作为正式路线
方案 B: 保持 old/，文档明确 Plan E 是归档实验，主线回到 last-step-skip cached teacher
```

### 10.2 on-policy 名称与实现要对齐

如果默认仍使用 `cached_prefix`，不要在报告里直接说它是完整 on-policy。更精确的表述应该是：

```text
on-policy-style velocity target with cached teacher prefix
```

如果要真正验证 D-OPSD 思路，需要引入 closed-loop student rollout 和 EMA teacher。

### 10.3 EMA 缺失

当前 teacher-trajectory trainer 没有 EMA shadow state。配置中的 `ema_decay: null` 不是“EMA 关闭但可用”，而是当前 trainer 没有 EMA 更新逻辑。

如果 LoRA 训练出现抖动，EMA 是优先补的稳定性工具。

### 10.4 LoRA 输出与 Stage2 分布差异

即使 LoRA 让 LR decode 接近 teacher4，`z_lr_lora_clean` 也可能和 Stage2 训练时的 `z0_lr` 有分布差异。需要用同一批 prompt/seed 比较：

```text
teacher4 clean LR -> existing Stage2
lora3 clean LR    -> existing Stage2
```

只有当差异明显时，再构建 `z_lr_lora_clean -> z_hr_clean` pair 微调 Stage2。

### 10.5 切换点与 checkpoint 绑定

Stage3 的 step1/2/3 checkpoint、50-step 的 45/46/47 checkpoint 都必须和各自数据 recipe 绑定。后续评估脚本应该继续保留 metadata guard，避免拿 step2 checkpoint 跑 step3 handoff。

### 10.6 评估矩阵需要统一

建议固定最终对比面板：

```text
ORI480
Interp720
Stage2 clean from teacher4 LR
Old Stage3 x0pred baseline
LoRA3 LR
LoRA3 + Stage2
```

这样能同时回答三个问题：

```text
1. x_pred -> z baseline 是否仍有价值
2. LoRA 是否真的消除了最后一步 denoising loss
3. Stage2 是否能直接吃 LoRA 生成的 clean LR latent
```

## 11. 下一步建议

最短闭环：

```text
1. 确认当前要走 active last-step-skip 还是恢复 teacher-trajectory Plan E
2. 跑 16-64 样本 LoRA overfit
3. 生成 teacher4 LR vs lora3 LR 对比
4. 接 existing distill Stage2 生成 720p 对比
5. 根据分布差异决定是否微调 Stage2
```

如果目标是写阶段报告，推荐结论写成：

```text
Stage2 证明了 clean latent 升分可行；
Stage3 证明了 handoff 输入域对齐的重要性，但 x_pred -> z 会把 denoising repair 固定混入 upsampler；
因此最终路线改为 denoiser-side LoRA 先平滑最后一步，再把 clean latent upsample 留给 Stage2。
```
