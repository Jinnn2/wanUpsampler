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

The default 1K-video pilot schedule is 10K optimizer updates with 500 updates
of linear warmup followed by cosine decay from `1e-4` to `1e-6`. Validation is
run every 500 updates on a source-level split. `metrics.jsonl` contains both
global metrics and separate `scale_1p5x/*`, `scale_2x/*`, and `scale_3x/*`
metrics. A longer run should be selected from held-out validation rather than
precommitting the 1K-video dataset to 50K updates.

For a direct latent check on the remote machine:

```bash
python -m changing_resolution_uni.infer \
  --checkpoint outputs/changing_resolution_uni_clean/last.pt \
  --input /path/to/lr_latent.npy \
  --target_h 90 --target_w 156 \
  --output /path/to/hr_prediction.npy
```

## Validate a completed run

The evaluator freezes the exact source-level validation split into a manifest,
compares raw and EMA weights on identical source/scale pairs, and writes
per-video metrics plus source-clustered 95% bootstrap confidence intervals.
Interpolation exists only as an external baseline; it is never inserted into
the U-ITU forward path.

Run the complete latent benchmark on four GPUs:

```bash
GPU_IDS=0,1,2,3 \
TRAIN_OUT_DIR="$PWD/outputs/changing_resolution_uni_clean_v1_1k_fresh_ema0999" \
DATA_DIR="$PWD/data/changing_resolution_uni/lmdb_clean_v1_1k" \
MODE=latent \
bash changing_resolution_uni/scripts/run_evaluate_multigpu.sh
```

Sweep all saved optimizer-step checkpoints using raw and EMA weights:

```bash
GPU_IDS=0,1,2,3 \
TRAIN_OUT_DIR="$PWD/outputs/changing_resolution_uni_clean_v1_1k_fresh_ema0999" \
MODE=sweep \
bash changing_resolution_uni/scripts/run_evaluate_multigpu.sh
```

Run decoded RGB metrics, comparison videos, and a separate single-GPU timing
pass. RGB evaluation defaults to 20 validation sources because Wan VAE decode
is substantially more expensive than latent metrics:

```bash
GPU_IDS=0,1,2,3 \
MODE=all \
SAVE_VISUALS=1 \
DECODE_MAX_SOURCES=20 \
bash changing_resolution_uni/scripts/run_evaluate_multigpu.sh
```

Common environment overrides are:

```text
CHECKPOINT / CHECKPOINT_DIR / CHECKPOINT_GLOB
DATA_DIR / OUT_DIR / METHODS / PRECISION
SPLIT / MANIFEST / MAX_SOURCES / DECODE_MAX_SOURCES / BOOTSTRAP_SAMPLES
MODEL_ROOT / VAE_PATH / LIGHTX2V_REPO / VAE_BACKEND
SPECIALIST_CHECKPOINT / SPECIALIST_CONFIG / SPECIALIST_USE_EMA
```

The output directory contains:

```text
manifest_val.json            frozen source-level validation split
environment*.json            command, Git, CUDA, dataset and checkpoint hashes
latent_coverage.json         expected/observed/status count integrity gate
latent_samples.jsonl         one row per checkpoint/method/source/scale
latent_summary.{json,csv}    mean/std/source-bootstrap CI
latent_paired.csv             paired improvements and win rates
checkpoint_sweep.csv          raw/EMA checkpoint-selection table
best_checkpoint.json          best raw, EMA and overall macro-Charbonnier row
rgb_samples.jsonl
rgb_summary.{json,csv}
rgb_paired.csv
rgb_coverage.json
timing.{json,csv}             isolated batch-1 latency and peak memory
visuals/                      optional MP4 panels and keyframe PNGs
```

`RESUME=1` resumes per-rank JSONL evaluation. The merged results are
deduplicated by checkpoint, method, source, and scale, so a changed GPU count
does not bias the aggregate.

## First acceptance matrix

Compare one shared checkpoint against trilinear interpolation and the existing
fixed Stage2 specialists on the same clean pairs and target grids:

```text
1.5x: 60x104 -> 90x156
2.0x: 46x80  -> 90x156 (actual anisotropic ratio is recorded)
3.0x: 30x52  -> 90x156
```

Report latent L1/MAE/MSE, temporal-difference error, decoded RGB metrics when
VAE decoding is available, and latency. Unseen ratios such as 1.75x and 2.5x
are validation-only until they are included in a controlled scale-jitter run.

## Generate the 1,500-prompt timestep-router videos on eight GPUs

The formal launcher keeps all eight GPUs occupied and uses two physical runs
to avoid unnecessary model reloads:

- train: 1,000 prompts, one seed;
- evaluation: 500 prompts, three seeds, with prompt IDs splitting into 200
  validation prompts and 300 test prompts.

Native-HR is enabled by default for every trajectory so the generated videos
remain compatible with the current strict VBench scorer. The default plan
therefore generates 35,000 videos instead of the old 84,000-video layout.
All 13 candidate videos are retained, and every candidate switch point also
stores a `wan_taa_free_oracle_latent_v1` archive containing `x_t_lr` and
`x0_pred_lr`. With fp16 storage the 32,500 latent files require approximately
150 GiB before filesystem overhead; fp32 roughly doubles that budget.

Inspect the exact offsets, seeds, and counts without starting any work:

```bash
PLAN_ONLY=1 \
bash changing_resolution_uni/scripts/data/build_oracle_dataset_1500_8gpu.sh
```

Start the resumable run in tmux:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
bash changing_resolution_uni/scripts/data/tmux_build_oracle_dataset_1500_8gpu.sh
```

The default logical split is `1000/200/300`. Validation and test share the
physical `eval` directory to load each seed's resident model only once; their
non-overlapping prompt-ID ranges are recorded in `generation_plan.json`.
`SKIP_EXISTING=1` makes restarts idempotent.
`CLEAN_VIDEOS=1` and `SAVE_LATENTS=0` are rejected by this formal launcher so
an accidental restart cannot remove or omit the requested intermediate data.

`TRAIN_INCLUDE_NATIVE_HR=0` reduces the plan to 34,000 videos, but this is an
explicit generation-only mode: the current strict VBench scorer requires a
matched Native-HR video for every train trajectory. Keep the default value of
`1` unless a calibrated native-latency training profile is used downstream.
