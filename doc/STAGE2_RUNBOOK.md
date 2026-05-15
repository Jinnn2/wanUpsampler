# Stage 2 Runbook

## Preflight

```bash
bash changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh check
```

## Train

```bash
CUDA_VISIBLE_DEVICES=0 \
bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage2_lmdb_training.sh
```

## Operator Compare

The default checkpoint is:

```text
outputs/changing_resolution_clean_480p720p_stage2_lmdb/latest.pt
```

Run:

```bash
TOTAL_SAMPLES=32 GPU_IDS=0,1,2,3 \
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_operator_compare_multigpu.sh
```

Useful overrides:

```bash
CR_STAGE2_OPERATOR_COMPARE_CKPT=/path/to/latest.pt
CR_STAGE2_CONFIG=/path/to/train_clean_480p_to_720p_lmdb_stage2.yaml
USE_EMA=0
STAGE2_RESIDUAL_SKIP=checkpoint
```

## Chain A/B Compare

Run:

```bash
TOTAL_SAMPLES=16 GPU_IDS=0,1,2,3 \
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_chain_ab_compare_multigpu.sh
```

Useful overrides:

```bash
CR_STAGE2_CHAIN_COMPARE_CKPT=/path/to/latest.pt
CR_STAGE2_CONFIG=/path/to/train_clean_480p_to_720p_lmdb_stage2.yaml
USE_EMA=0
STAGE2_RESIDUAL_SKIP=checkpoint
```

For short Stage 2 runs, raw weights are the default because EMA may lag behind
the current checkpoint.

## Change-Step Sweep

This compares the same prompt and seed across different handoff steps. For each
selected step, it writes a horizontal three-way panel:

```text
stop480 at step N | interp720 step N -> 50 | stage2 720 step N -> 50
```

Run:

```bash
bash changing_resolution/scripts/eval/run_clean_480p720p_stage2_change_step_sweep.sh
```

Defaults:

```text
STEP_START / STEP_END / STEP_STRIDE: 10 / 50 / 5
LIMIT: 4
output: outputs/changing_resolution_stage2_change_step_sweep/compare
```

Use a denser sweep:

```bash
STEP_START=10 STEP_END=50 STEP_STRIDE=1 \
bash changing_resolution/scripts/eval/run_clean_480p720p_stage2_change_step_sweep.sh
```

Or pass explicit steps:

```bash
CHANGE_STEPS="20 30 35 40" \
bash changing_resolution/scripts/eval/run_clean_480p720p_stage2_change_step_sweep.sh
```

Run four prompts in parallel on four GPUs:

```bash
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_change_step_sweep_multigpu.sh
```

Default 4-GPU sweep:

```text
GPU_IDS: 0,1,2,3
TOTAL_PROMPTS: 4
STEP_START / STEP_END / STEP_STRIDE: 10 / 50 / 1
```

That exact range has 41 handoff points, so it produces 164 three-way panels for
4 prompts. To produce exactly 200 panels with 4 prompts, use 50 handoff points,
for example `STEP_START=1 STEP_END=50 STEP_STRIDE=1`.
