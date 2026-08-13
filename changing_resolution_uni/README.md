# Universal clean-latent U-ITU (independent module)

This directory is the first implementation stage of U-ITU: one checkpoint
maps clean Wan LR latents to multiple target spatial latent grids. It preserves
the temporal latent length and deliberately does not include a diffusion step,
EAA, scheduler re-noising, or HR denoising suffix yet.

## Contract

```text
[B,16,T,h,w] + output_size=(H,W) -> [B,16,T,H,W]
```

The model uses a shared Conv3D encoder followed by a dynamic learned 3x3x3
subpixel resampler. Each target position gathers an integer-indexed 27-point
temporal/spatial neighborhood from the LR feature grid and predicts content-
and geometry-conditioned mixing weights. The model directly reconstructs HR
latents: its forward path contains no `F.interpolate`, `grid_sample`, fixed
PixelShuffle branch, or trilinear output skip.

## Build clean multi-scale pairs

The RGB video is resized first and each resolution is independently encoded by
the frozen Wan VAE. Latent interpolation is not used to construct training LR
inputs.

```bash
VIDEO_DIR=/path/to/hr_videos \
MODEL_ROOT=/path/to/Wan2.1-T2V-1.3B \
VAE_PATH=/path/to/Wan2.1_VAE.pth \
WAN_REPO=/path/to/LightX2V \
bash changing_resolution_uni/scripts/build_clean_pairs.sh
```

For a remote multi-GPU VAE build, use one deterministic video range per GPU:

```bash
VIDEO_DIR=/path/to/hr_videos GPU_IDS=0,1,2,3 TOTAL_SAMPLES=1000 \
bash changing_resolution_uni/scripts/build_clean_pairs_multigpu.sh
```

The writer creates a single-shard `wan_uni_clean_v1` LMDB containing HR latent
and LR variants for 1.5x, 2x, and 3x. The current builder intentionally fixes
one HR shape and one clip length per run, because tensor-shape bucketing should
be explicit before mixing larger datasets.

When deployment uses a non-nominal latent grid, pass explicit RGB LR sizes in
the same order as `--scales`, for example:

```bash
python -m changing_resolution_uni.build_latent_pairs \
  --hr_size 720 1248 --scales 1.5 2.0 3.0 \
  --lr_sizes 480x832 368x640 240x416 ...
```

## Train locally or on the remote machine

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
bash changing_resolution_uni/scripts/run_train.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 \
bash changing_resolution_uni/scripts/tmux_run_train.sh
```

Preflight the model on the remote environment:

```bash
bash changing_resolution_uni/scripts/check.sh
```

The remote launcher follows the existing repository convention: `torchrun` for
DDP and a separate tmux wrapper for long jobs. Checkpoint files contain the
model configuration, optimizer, and EMA state. Use
`changing_resolution_uni.checkpoint.load_universal_upsampler` for inference.

For a direct latent check on the remote machine:

```bash
python -m changing_resolution_uni.infer \
  --checkpoint outputs/changing_resolution_uni_clean/last.pt \
  --input /path/to/lr_latent.npy \
  --target_h 90 --target_w 156 \
  --output /path/to/hr_prediction.npy
```

## First acceptance matrix

Compare one shared checkpoint against trilinear interpolation and the existing
fixed Stage2 specialists on the same clean pairs and target grids:

```text
1.5x: 60x104 -> 90x156
2.0x: 46x80  -> 92x160 or a declared crop target
3.0x: declared source/target latent grid from the same VAE geometry
```

Report latent L1/MAE/MSE, temporal-difference error, decoded RGB metrics when
VAE decoding is available, and latency. Unseen ratios such as 1.75x and 2.5x
are validation-only until they are included in a controlled scale-jitter run.
