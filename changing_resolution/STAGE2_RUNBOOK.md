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
