# Distill LoRA 前置环境

> 范围：`changing_resolution_distill`  
> 目标：为 `last-step-skip LoRA` Phase 1 准备 DiffSynth-Studio 训练环境。

## 参考结论

DiffSynth-Studio 的 Wan 训练入口是：

```text
examples/wanvideo/model_training/train.py
```

官方 Wan LoRA 示例使用的核心参数是：

```text
--lora_base_model "dit"
--lora_target_modules "q,k,v,o,ffn.0,ffn.2"
--lora_rank 32
--remove_prefix_in_ckpt "pipe.dit."
```

因此本仓库的 Phase 1 LoRA 环境先复用 DiffSynth-Studio 的 Wan LoRA
基础设施，再在后续任务中接入自定义的 `x3_lr -> z4_lr_teacher` 数据集和
loss。

## 新增文件

```text
changing_resolution_distill/scripts/train/setup_last_step_skip_lora_env.sh
changing_resolution_distill/scripts/train/check_last_step_skip_lora_env.sh
changing_resolution_distill/configs/train_last_step_skip_lora_distill.yaml
```

`configs/local_paths.sh` 新增：

```text
DIFFSYNTH_REPO
DIFFSYNTH_REF
CR_DISTILL_LORA_LMDB_DIR
CR_DISTILL_LORA_CONFIG
CR_DISTILL_LORA_OUT_DIR
```

## 安装

在 Linux GPU 环境运行：

```bash
DIFFSYNTH_REF=main \
bash changing_resolution_distill/scripts/train/setup_last_step_skip_lora_env.sh install
```

默认会把 DiffSynth-Studio 放到：

```text
/mnt/afs_2/houze/DiffSynth-Studio
```

可覆盖：

```bash
DIFFSYNTH_REPO=/path/to/DiffSynth-Studio \
DIFFSYNTH_REF=main \
bash changing_resolution_distill/scripts/train/setup_last_step_skip_lora_env.sh install
```

如果机器上已经有 DiffSynth-Studio，不想更新 git：

```bash
SKIP_GIT_UPDATE=1 \
DIFFSYNTH_REPO=/path/to/DiffSynth-Studio \
bash changing_resolution_distill/scripts/train/setup_last_step_skip_lora_env.sh install
```

## 检查

```bash
bash changing_resolution_distill/scripts/train/setup_last_step_skip_lora_env.sh check
```

检查项：

- DiffSynth-Studio、LightX2V、14B CfgDistill 模型路径存在。
- DiffSynth Wan `train.py` 和 14B accelerate config 存在。
- Python 可以 import `torch`、`accelerate`、`diffsynth`、`modelscope`、
  `yaml`、`safetensors`。
- DiffSynth Wan `train.py` 暴露 LoRA/offload 参数。

`CR_DISTILL_LORA_LMDB_DIR` 目前允许不存在，因为 LoRA 数据集构建器仍是下一步
任务。检查脚本会把它标记为 pending。

## 构建 Version A 数据

```bash
TOTAL_SAMPLES=5000 GPU_IDS=0,1,2,3 OVERWRITE=1 \
bash changing_resolution_distill/scripts/data/build_last_step_skip_lora_lmdb_multigpu.sh

python changing_resolution_distill/scripts/data/check_last_step_skip_lora_lmdb.py \
  --expect_samples 5000
```

输出默认写到：

```text
data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3
```

字段：

```text
x3_lr
z4_lr_teacher
z0_hr
prompt
seed
meta
```

其中 `x3_lr` 和 `z4_lr_teacher` 来自同一次 LR teacher rollout；`z0_hr`
复用现有 5000 条 clean LMDB 里的 HR latent。

## 当前默认 LoRA 配置

```text
config: changing_resolution_distill/configs/train_last_step_skip_lora_distill.yaml
target: x3_lr -> z4_lr_teacher
rank: 32
alpha: 32
lr: 1e-4
precision: bf16
batch_size: 1
grad_accum: 8
max_steps: 10000
```

训练器实现时必须保持：

```text
step1/step2: base Wan distill, LoRA disabled
step3: base Wan distill + LoRA
checkpoint: LoRA-only weights plus recipe metadata
```
