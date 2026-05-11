# Stage 2 Model Plan

This document defines the planned Stage 2 clean-latent resizer for
`changing_resolution`.

Stage 2 keeps the same training target and LightX2V integration contract as
Stage 1:

```text
z0_lr or x0_pred_lr -> z0_hr
[B, 16, T, 60, 104] -> [B, 16, T, 90, 156]
```

The goal is to make the resize operator closer to the LTX2 latent upsampler
while keeping the surrounding Wan training and evaluation flow stable.

## Current Stage 1 Baseline

Stage 1 is implemented by `WanCleanLatentResizer` in:

```text
wan_sr/models/clean_resizer.py
```

Its effective path is:

```text
z0_lr
  -> Conv3d 16 -> 256
  -> PlainResBlock3D x4
  -> F.interpolate(..., mode="trilinear")
  -> PlainResBlock3D x4
  -> GroupNorm + SiLU + Conv3d 256 -> 16
  -> learned residual
  -> trilinear(z0_lr) + residual
  -> pred_z0_hr
```

The actual size change in Stage 1 is still fixed trilinear interpolation. The
network learns a residual correction around that interpolation baseline.

## Stage 2 Scope

Stage 2 changes only the model internals first. The data contract, trainer,
losses, checkpoint format, and LightX2V clean-latent bridge should remain as
compatible as possible.

The two intended model changes are:

```text
1. Replace the current Wan PlainResBlock3D with an LTX2-compatible ResBlock.
2. Replace the feature-path F.interpolate with a learned rational spatial
   upsampler: Conv3d expansion -> spatial PixelShuffle x3 -> BlurDownsample /2.
```

The residual skip path should remain enabled for the first Stage 2 experiment:

```text
output = trilinear(z0_lr) + learned_residual
```

Keeping this skip path makes Stage 2 a controlled comparison against Stage 1:
only the feature refinement and feature resize operator change.

## LTX2-Compatible ResBlock

LTX2's upsampler `ResBlock` uses this order:

```text
x
  -> Conv
  -> GroupNorm
  -> SiLU
  -> Conv
  -> GroupNorm
  -> add residual
  -> SiLU
```

Stage 1's `PlainResBlock3D` currently uses a pre-activation style:

```text
x
  -> GroupNorm
  -> SiLU
  -> Conv3d
  -> GroupNorm
  -> SiLU
  -> Conv3d
  -> add residual
```

Stage 2 should introduce a Wan-local block with LTX2's operation order but
using `Conv3d` for Wan video latents:

```text
class LTX2StyleResBlock3D:
    Conv3d(channels, channels, kernel_size=3, padding=1)
    GroupNorm(32-compatible, channels)
    SiLU
    Conv3d(channels, channels, kernel_size=3, padding=1)
    GroupNorm(32-compatible, channels)
    residual add
    SiLU
```

Expected behavior:

```text
[B, hidden, T, H, W] -> [B, hidden, T, H, W]
```

This block changes feature refinement behavior but does not change tensor
shape.

## SpatialRationalResampler3D

LTX2 implements 1.5x rational spatial scaling as:

```text
Conv2d -> PixelShuffle x3 -> BlurDownsample stride 2
```

For Wan Stage 2, the intended version is a 3D-convolution variant that keeps
time unchanged:

```text
SpatialRationalResampler3D(scale=1.5)

input:
  [B, hidden, T, H, W]

learned expansion:
  Conv3d(hidden, hidden * 3 * 3, kernel_size=3, padding=1)
  [B, hidden * 9, T, H, W]

spatial pixel shuffle:
  rearrange [B, hidden * 9, T, H, W]
         -> [B, hidden, T, H * 3, W * 3]

anti-aliased reduction:
  BlurDownsample spatial stride 2
  [B, hidden, T, H * 3 / 2, W * 3 / 2]
```

For the current 480p -> 720p latent sizes:

```text
[B, 256, T, 60, 104]
  -> Conv3d expansion
[B, 2304, T, 60, 104]
  -> spatial pixel shuffle x3
[B, 256, T, 180, 312]
  -> blur downsample /2
[B, 256, T, 90, 156]
```

The blur downsample should only operate on H/W. Implementation options:

```text
Option A:
  flatten T into batch, apply fixed depthwise Conv2d blur with stride=2,
  then restore [B, C, T, H, W].

Option B:
  use depthwise Conv3d with kernel shape [1, 5, 5] and stride [1, 2, 2].
```

Option A is closest to LTX2's existing `BlurDownsample` implementation. Option B
keeps the operator fully 5D but must be checked carefully against PyTorch group
conv behavior and padding.

## Planned Stage 2 Model Path

The target Stage 2 forward path is:

```text
z0_lr
  -> stem Conv3d 16 -> hidden
  -> LTX2StyleResBlock3D x4
  -> SpatialRationalResampler3D scale 1.5
       -> Conv3d hidden -> hidden * 9
       -> spatial PixelShuffle x3
       -> BlurDownsample /2
  -> LTX2StyleResBlock3D x4
  -> GroupNorm + SiLU + Conv3d hidden -> 16
  -> learned residual
  -> trilinear(z0_lr) + residual
  -> pred_z0_hr
```

The initial default config should stay close to Stage 1:

```yaml
model:
  in_channels: 16
  out_channels: 16
  hidden_channels: 256
  num_res_blocks: 8
  scale_factor: 1.5
  dropout: 0.0
  residual_skip: true
  resblock_type: ltx2
  resize_op: rational_conv3d_pixel_shuffle
```

## Shape Rules

Stage 2 should validate shapes explicitly:

```text
input ndim must be 5
input channel count must match in_channels
scale_factor must currently be 1.5 for the rational path
output_size, if provided, must equal round(input H/W * 1.5)
time dimension must be unchanged
```

The first implementation should prefer failing loudly over silently falling back
inside the model. Fallbacks are acceptable in bridge or evaluation wrappers, but
the training model should expose shape mistakes early.

## Compatibility Boundary

Keep stable:

```text
WanCleanLatentResizer forward signature:
  forward(z0_lr, output_size=None)

training script:
  changing_resolution/scripts/train/train_clean_latent_resizer.py

loss:
  CleanLatentResizeLoss

data:
  data/changing_resolution/lmdb_480p720p_1k

LightX2V bridge contract:
  clean latent resize inside changing_resolution
```

Allowed Stage 2 additions:

```text
new model blocks under wan_sr/models/
new model config under changing_resolution/configs/
new Stage 2 train/tmux wrapper scripts
new Stage 2 output directories
new Stage 2 eval output directories
```

Avoid changing the dataset builder or bridge scheduler unless Stage 2 proves the
model interface needs it.

## Training Plan

Stage 2 should reuse the 1k LMDB data first:

```text
data/changing_resolution/lmdb_480p720p_1k
```

Recommended first run:

```text
max_steps: 10000
batch_size: 1
grad_accum: 8
precision: bf16
lr: 1e-4
eval_every: 1000
save_every: 1000
```

Recommended output:

```text
outputs/changing_resolution_clean_480p720p_stage2_lmdb
```

Implemented Stage 2 entrypoints:

```text
wan_sr/models/stage2_resizer.py
changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml
changing_resolution/scripts/train/train_clean_latent_resizer_stage2.py
changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh
changing_resolution/scripts/train/tmux_run_clean_480p720p_stage2_lmdb_training.sh
```

Preflight:

```bash
bash changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh check
```

Train:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage2_lmdb_training.sh
```

Do not increase model width or add attention in the first Stage 2 run. The first
question is whether the LTX2-style block and learned rational resize beat Stage
1 under the same data and loss.

## Evaluation Plan

Stage 2 should be compared against both fixed interpolation and Stage 1.

Operator compare should include:

```text
lr480_decode
ori720_decode
interp720_decode
stage1_trained720_decode
stage2_trained720_decode
```

Primary metric expectation:

```text
stage2_psnr  > max(interp_psnr, stage1_psnr)
stage2_ssim  > max(interp_ssim, stage1_ssim)
stage2_lpips < min(interp_lpips, stage1_lpips)
```

Generation-chain A/B should compare:

```text
interp720
stage1_trained720
stage2_trained720
```

Manual review should focus on:

```text
sharpness
temporal stability
texture crawling
subject deformation
new aliasing from pixel shuffle
over-smoothing from blur downsample
```

## Main Risks

Pixel shuffle artifacts:
The learned expansion can create checkerboard-like latent artifacts. The blur
downsample is meant to reduce this, but visual chain A/B is still required.

Residual skip masking:
Keeping `trilinear(z0_lr) + residual` stabilizes training, but it can hide weak
learned upsampling. If Stage 2 only improves marginally, run an ablation with
`residual_skip=false`.

Temporal consistency:
The proposed resampler changes H/W only, but the expansion uses `Conv3d`, so it
can mix temporal context before spatial rearrangement. This should help temporal
stability, but it also makes artifacts potentially time-correlated.

Output-size rigidity:
The 3/2 rational path naturally matches 60x104 -> 90x156. Other ratios should
not be claimed until they are explicitly implemented and tested.

## Implementation Order

1. Add `LTX2StyleResBlock3D`.
2. Add `SpatialPixelShuffle3x` for `[B, C, T, H, W]`.
3. Add `SpatialBlurDownsample3D` or reuse a frame-flattened 2D blur.
4. Add `SpatialRationalResampler3D(scale=1.5)`.
5. Add Stage 2 model selection flags while preserving Stage 1 defaults.
6. Add Stage 2 config and train wrappers.
7. Run a shape-only smoke test on synthetic latents.
8. Run short training over the LMDB.
9. Run operator compare.
10. Run generation-chain A/B only after operator compare is sane.
