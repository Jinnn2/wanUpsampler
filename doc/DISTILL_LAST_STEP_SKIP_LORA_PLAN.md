# Distill 最后一步跳过 LoRA 方案

> 日期：2026-06-17  
> 范围：`changing_resolution_distill`  
> 状态：下一条主线方案，尚未实现

## 1. 决策

当前 distill Stage 3 路径训练的是：

```text
x0_pred_lr -> z0_hr
```

这不应该再作为 4-step Wan distill 分支的主要方向。它把两个不同任务混进了一个小型 resizer 里：

```text
1. 修复 x0_pred_lr 中残留的一步去噪误差
2. 将 LR latent 上采样为 HR latent
```

对 few-step distill 模型来说，这条路比较脆弱。4-step 模型对真实推理链中经过的轨迹状态非常敏感。如果训练输入来自外部重建的 `x0_pred_lr`，而这些状态并不是真实推理链中稳定出现的 clean LR 状态，那么 upsampler 就会被迫同时补偿去噪器不匹配和分辨率变化。这很可能就是 distill Stage 3 结果反而弱于 clean Stage 2 baseline 的原因。

新的主线应当拆开问题：

```text
Phase 1: 训练 Wan distill denoiser LoRA，使其跳过最后一步
Phase 2: 训练或复用 clean latent upsampler
```

最终目标链路：

```text
LR chain:
  noise -> Wan step1 -> Wan step2 -> Wan step3 + LoRA -> z_lr_clean

Upscale:
  z_lr_clean -> clean latent upsampler -> z_hr_clean

Decode:
  z_hr_clean -> Wan VAE decode -> HR video
```

旧的 `x0_pred_lr -> z0_hr` Stage 3 代码仍然适合作为实验分支和对比 baseline，但不应作为接下来继续扩展的主分支。

## 2. Phase 1：最后一步跳过 LoRA

LoRA 训练在 Wan 4-step distill denoiser 本身上，而不是训练在 upsampler 上。

它的任务是：

```text
让 step3 + LoRA 完成原本 step3 + step4 才能完成的事情
```

训练样本对：

```text
input : x3_lr
target: z4_lr_teacher
```

其中：

```text
x3_lr:
  原始 4-step distill teacher 到达 step-3 状态后的 LR latent。
  这仍然是一个 noisy/intermediate latent。

z4_lr_teacher:
  原始 4-step teacher 完成全部 4 步之后得到的 clean LR latent。
```

训练计算：

```text
v3_lora     = denoiser_with_lora(x3_lr, t3, prompt_cond)
z_lr_lora   = x3_lr - sigma3 * v3_lora
loss        = L1(z_lr_lora, z4_lr_teacher)
```

可选的第一版 loss：

```text
loss = L1(z_lr_lora, z4_lr_teacher) + 0.1 * MSE(z_lr_lora, z4_lr_teacher)
```

这一阶段不使用 HR latents，也不接触 `z0_hr`。

## 3. Version A：缓存 Teacher x3

这是最快的验证路线，应该优先实现。

新建一个 LMDB：

```text
data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3
```

每个样本保存：

```text
x3_lr
z4_lr_teacher
prompt
seed
meta
```

teacher rollout 固定为：

```text
noise -> teacher step1 -> teacher step2 -> x3_teacher
noise -> teacher step1 -> teacher step2 -> teacher step3 -> teacher step4 -> z4_teacher
```

Version A 推荐的推理/训练策略：

```text
step1: original model
step2: original model
step3: original model + LoRA
```

这样可以保持训练和推理时的 `x3` 对齐，因为 LoRA 只影响最后一次实际执行的 step。它避开了缓存轨迹的主要弱点：如果 LoRA 也改变早期 step，student 推理时到达的 `x3` 就会不同于训练时缓存的 teacher `x3`。

### Version A 验收检查

1. Latent 重建：

```text
L1(z_lr_lora, z4_lr_teacher)
```

2. LR decode 对比：

```text
3-step + LoRA decode vs original 4-step teacher decode
```

3. 运行策略检查：

```text
LoRA disabled on steps 1/2, enabled only on step 3
```

如果 3-step + LoRA 生成的 LR 视频足够接近原始 4-step teacher，使缺失最后一个去噪 step 不再是主要 artifact 来源，就认为 Version A 成功。

## 4. Version B：Student On-Policy Rollout

仅在 Version A 跑通后再启动这一版。

Version B 让当前 student 轨迹生成自己的 `x3`：

```text
noise -> step1 + LoRA -> step2 + LoRA -> x3_student_current
```

然后训练：

```text
x3_student_current -> z4_teacher
```

这更接近 D-OPSD 使用的 on-policy 原则：在 student 自己实际访问的状态上训练 student。它计算开销更高，实现也更难，因为训练循环必须先用当前 LoRA 权重跑一段 partial rollout，再计算 step-3 监督。

把 Version B 保留为正式 follow-up，不作为第一版实现。

## 5. Phase 2：Clean Latent Upsampler

Phase 1 跑通后，从下面的链路生成 clean LR latents：

```text
noise -> Wan step1 -> Wan step2 -> Wan step3 + LoRA -> z_lr_lora_clean
```

然后训练或微调：

```text
z_lr_lora_clean -> z_hr_clean
```

这一阶段应尽可能复用现有 clean-latent Stage 2 基础设施：

```text
config:
  changing_resolution_distill/configs/train_clean_480p_to_720p_lmdb_stage2_distill.yaml

trainer:
  changing_resolution/scripts/train/train_clean_latent_resizer_stage2.py

current launcher style:
  changing_resolution_distill/scripts/train/run_clean_480p720p_stage2_distill_lmdb_training.sh
```

第一轮测试不要立刻重训 upsampler，而是先做：

```text
3-step + LoRA z_lr_clean
  -> existing distill Stage 2 clean upsampler
  -> decode
```

只有当这个测试显示出明显 distribution shift 时，才新建 clean pair LMDB 并微调 Stage 2。

可能的新 LMDB：

```text
data/changing_resolution_distill/lmdb_clean_lora3step_480p720p_14b_cfgdistill_5k
```

字段：

```text
z0_lr: z_lr_lora_clean
z0_hr: clean HR latent from the matched 720p teacher/video
prompt
seed
meta.lora_ckpt
meta.teacher_model
meta.recipe = lora3step_clean_pair
```

## 6. 实现任务

### Task 1：LoRA 数据集构建器

新增：

```text
changing_resolution_distill/scripts/data/build_last_step_skip_lora_lmdb.py
changing_resolution_distill/scripts/data/build_last_step_skip_lora_lmdb.sh
changing_resolution_distill/scripts/data/tmux_build_last_step_skip_lora_lmdb.sh
```

复用下面文件中的 persistent LightX2V runner 模式：

```text
changing_resolution_distill/scripts/data/build_x0pred_480p720p_stage3_distill_lmdb.py
```

但保存 LR-only 的 step-skip 样本对，而不是 `x0_pred_lr/z0_hr`。

数据集契约：

```text
x3_lr: tensor produced by teacher rollout before the final step
z4_lr_teacher: tensor produced by full teacher rollout
prompt: source prompt
seed: source seed
meta:
  mode: last_step_skip_lora
  model_cls: wan2.1_distill
  denoising_step_list: [1000, 750, 500, 250]
  train_step_index: 2
  train_step_name: step3
  target_step_name: teacher_step4_clean
```

### Task 2：LoRA 训练脚本

新增：

```text
changing_resolution_distill/scripts/train/train_last_step_skip_lora.py
changing_resolution_distill/scripts/train/run_last_step_skip_lora_training.sh
changing_resolution_distill/scripts/train/tmux_run_last_step_skip_lora_training.sh
changing_resolution_distill/configs/train_last_step_skip_lora_distill.yaml
```

初始默认值：

```text
LoRA target: Wan 4-step distill denoiser
rank: 16 or 32
alpha: same as rank
lr: 1e-4 for LoRA params only
precision: bf16
batch_size: 1
grad_accum: 8
max_steps: 10000 first, then 30000 if stable
ema: optional, disabled for the first smoke test
```

trainer 必须冻结 base Wan 权重，只更新 LoRA 参数。checkpoint 应只包含 LoRA 权重，以及足够重建 base model 和 recipe 的 metadata。

### Task 3：Step3-only bridge/eval 模式

新增 bridge 支持：

```text
step1/step2: original Wan distill denoiser
step3: Wan distill denoiser + LoRA
stop after step3 and decode clean LR
```

评估输出：

```text
outputs/changing_resolution_distill_last_step_skip_lora_eval/
  teacher4_lr/
  lora3_lr/
  compare/
```

第一版对比列：

```text
teacher 4-step LR | 3-step + LoRA LR
```

### Task 4：现有 Stage 2 兼容性测试

LR decode 通过后，运行：

```text
3-step + LoRA clean LR
  -> existing distill Stage 2 clean upsampler
  -> 720p decode
```

对比：

```text
ORI480 teacher4
interp720
stage2 clean from teacher4 clean LR
stage2 clean from lora3 clean LR
```

这个测试用于判断 Phase 1 是否已经足够，或者 Stage 2 是否需要在 LoRA 生成的 LR clean latents 上继续 fine-tune。

### Task 5：可选 clean-pair 重建与 Stage 2 微调

如果 Stage 2 兼容性测试显示存在 distribution shift，则构建：

```text
z_lr_lora_clean -> z_hr_clean
```

然后用保守学习率微调现有 Stage 2 clean upsampler。

## 7. 里程碑

### Milestone M1：数据 sanity

交付物：

```text
LMDB with x3_lr/z4_lr_teacher
sample metadata dump
tensor shape/range report
```

通过条件：

```text
x3_lr and z4_lr_teacher can both be decoded or consumed by the Wan VAE/runtime
without shape/dtype surprises.
```

### Milestone M2：LoRA 小样本过拟合 smoke test

交付物：

```text
tiny dataset overfit run, 16-64 samples
training loss curve
teacher4 vs lora3 LR compare videos
```

通过条件：

```text
LoRA can visibly close the gap to teacher4 on the tiny set.
```

### Milestone M3：5k Version A 训练

交付物：

```text
10k LoRA checkpoint
teacher4 vs lora3 held-out compare
```

通过条件：

```text
3-step + LoRA is close enough to original 4-step on LR videos to be used as the
input producer for the clean upsampler test.
```

### Milestone M4：Stage 2 兼容性

交付物：

```text
720p compare using existing Stage 2 clean upsampler
decision: reuse Stage 2 or build lora-clean fine-tune data
```

通过条件：

```text
Artifacts are no longer dominated by last-step denoising error.
```

### Milestone M5：可选 on-policy Version B

仅在 M1-M4 完成后启动。

交付物：

```text
student rollout training loop
on-policy LoRA checkpoint
Version A vs Version B compare
```

## 8. 接下来不要做什么

不要继续把旧 distill Stage 3 目标作为主路径扩展：

```text
x0_pred_lr -> z0_hr
```

不要训练 upsampler 同时修复 denoiser error 和完成 upscale。

Version A 中不要在 steps 1/2 启用 LoRA。那会破坏缓存 teacher `x3` 的假设。

在缓存 teacher 的 Version A 证明最后一步跳过目标可学习之前，不要先做 on-policy rollout。

## 9. 一句话总结

先用 step3-only LoRA 把 Wan 4-step distill 变成可靠的 3-step-to-clean LR 生成器，再让 clean latent upsampler 只负责 `z_lr_clean -> z_hr_clean`。
