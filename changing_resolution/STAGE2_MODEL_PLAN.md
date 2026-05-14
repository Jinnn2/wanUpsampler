# Stage 2 Model Notes

Stage 2 is the current mainline clean-latent resizer.

## Contract

```text
input:  z0_lr or x0_pred_lr, [B, 16, T, 60, 104]
output: z0_hr,              [B, 16, T, 90, 156]
scale:  1.5x spatial, T unchanged
```

The LightX2V bridge applies Stage 2 to the clean estimate, then re-noises the
result before continuing the high-resolution diffusion trajectory.

## Model Path

```text
wan_sr/models/stage2_resizer.py
```

Exported class:

```text
WanCleanLatentResizerStage2
```

The default resize operator is:

```text
Conv3d expansion -> spatial PixelShuffle x3 -> BlurDownsample /2
```

The model keeps the existing clean-latent target and bridge contract while
making the spatial resize itself learned.

## Config

```text
changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml
```

The config should keep:

```text
model_type: stage2
resize_op: rational_conv3d_pixel_shuffle
resblock_type: ltx2
scale_factor: 1.5
```

## Entry Points

```text
changing_resolution/scripts/train/train_clean_latent_resizer_stage2.py
changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh
changing_resolution/scripts/train/tmux_run_clean_480p720p_stage2_lmdb_training.sh
changing_resolution/scripts/eval/run_clean_480p720p_stage2_operator_compare_multigpu.sh
changing_resolution/scripts/eval/run_clean_480p720p_stage2_chain_ab_compare.sh
changing_resolution/scripts/eval/run_clean_480p720p_stage2_chain_ab_compare_multigpu.sh
```

For current command examples, use `changing_resolution/STAGE2_RUNBOOK.md`.
