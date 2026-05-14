# Progress

Last updated: 2026-05-14

## Current Stage

The repository mainline is Stage 2 `changing_resolution` clean-latent resizing:

```text
z0_lr or x0_pred_lr -> z0_hr
480p clean latent -> 720p clean latent
```

Stage 2 replaces the Stage 1 fixed trilinear resize point with an LTX2-style
learned operator:

```text
Conv3d expansion -> spatial PixelShuffle x3 -> BlurDownsample /2
```

## Current Entry Points

```bash
bash changing_resolution/scripts/data/tmux_build_clean_lmdb_480p720p_1k_multigpu.sh
bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage2_lmdb_training.sh
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_operator_compare_multigpu.sh
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_chain_ab_compare_multigpu.sh
```

## Archived

Stage 1 and V1 are no longer mainline entrypoints:

```text
.archive/stage1/
.archive/v1/
```

They remain in the repo for reference and checkpoint archaeology, but the
public model package and wrappers now point at Stage 2.
