# Changing Resolution Clean Latent

This folder contains the V2 path for replacing LightX2V's native
`changing_resolution` interpolation with a learned clean-latent resizer.

The target contract is:

```text
x0_pred_lr or z0_lr -> z0_hr
```

For the current 480p -> 720p setting:

```text
LR RGB: 480 x 832
HR RGB: 720 x 1248
latent: 60 x 104 -> 90 x 156
scale: 1.5x spatial
```

LightX2V first estimates a clean latent with `x0_pred = x_t - sigma * eps`,
resizes that clean estimate, and re-noises it before continuing diffusion.
Therefore the training target is clean latent resizing, not noisy latent
super-resolution.

## Recommended Flow

Run from the Linux machine:

```bash
cd /mnt/afs_2/houze/wanUpsampler
git pull
pip install -r requirements.txt
```

### 1. Build 1k LMDB Data

If raw 720p videos already exist in:

```text
data/changing_resolution/raw_wan21_720p_1k/part_00 ... part_03
```

build only the LMDB:

```bash
TOTAL_SAMPLES=1000 GPU_IDS=0,1,2,3 \
bash changing_resolution/scripts/data/build_clean_480p720p_lmdb_1k_multigpu.sh lmdb
```

To generate raw videos and build LMDB in one tmux run:

```bash
TOTAL_SAMPLES=1000 GPU_IDS=0,1,2,3 \
bash changing_resolution/scripts/data/tmux_build_clean_lmdb_480p720p_1k_multigpu.sh

tmux attach -t wan_cr_lmdb_480p720p_1k_multigpu
```

Output:

```text
data/changing_resolution/lmdb_480p720p_1k
```

### 2. Train Stage 1 Baseline

Stage 1 keeps the current residual resizer model:

```text
trilinear(z0_lr) + learned residual
```

Preflight:

```bash
bash changing_resolution/scripts/train/run_clean_480p720p_stage1_lmdb_training.sh check
```

Train in tmux:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage1_lmdb_training.sh

tmux attach -t wan_cr_stage1_lmdb_train
```

Default training:

```text
max_steps: 10000
effective batch: 8
train/val split: 95% / 5%
eval_every: 1000
best checkpoint: best_val.pt
```

Output:

```text
outputs/changing_resolution_clean_480p720p_stage1_lmdb
```

Extend to 20k only after the 10k comparison looks useful:

```bash
MAX_STEPS=20000 \
RESUME=outputs/changing_resolution_clean_480p720p_stage1_lmdb/latest.pt \
bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage1_lmdb_training.sh
```

### 3. Evaluate Stage 1

Use two separate suites.

Operator compare has a real reference from the validation LMDB:

```text
lr480_decode | ori720_decode | interp720_decode | trained720_decode
```

It computes PSNR, SSIM, and LPIPS against `ori720_decode`.

```bash
TOTAL_SAMPLES=32 GPU_IDS=0,1,2,3 \
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_operator_compare_multigpu.sh

tmux attach -t wan_cr_operator_compare
```

Output:

```text
outputs/changing_resolution_operator_compare_stage1/part_*/compare
outputs/changing_resolution_operator_compare_stage1/metrics_val.jsonl
outputs/changing_resolution_operator_compare_stage1/summary_val.json
```

Generation-chain A/B has no reference. It compares only the two resize
operators inside the same LightX2V changing-resolution path:

```text
interp720 | trained720
```

```bash
TOTAL_SAMPLES=16 GPU_IDS=0,1,2,3 \
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_chain_ab_compare_multigpu.sh

tmux attach -t wan_cr_chain_ab_compare
```

Output:

```text
outputs/changing_resolution_chain_ab_stage1/compare
```

## Script Structure

### Data Build

```text
scripts/data/generate_wan21_720p_dataset.sh
  Generate Wan2.1 720p source videos from prompts.

scripts/data/build_480p720p_lmdb.py
  Convert 720p videos into sharded LMDB clean latent pairs.

scripts/data/build_clean_480p720p_lmdb_1k.sh
  Single-worker 1k prompt/video/LMDB entrypoint.

scripts/data/build_clean_480p720p_lmdb_1k_multigpu.sh
  Multi-GPU data build. Splits prompt ranges across GPUs.

scripts/data/tmux_build_clean_lmdb_480p720p_1k.sh
  tmux wrapper for the single-worker data build.

scripts/data/tmux_build_clean_lmdb_480p720p_1k_multigpu.sh
  tmux wrapper for the multi-GPU data build.
```

### Training

```text
scripts/train/train_clean_latent_resizer.py
  Generic clean latent resizer trainer. Supports files and LMDB backends.

scripts/train/run_clean_480p720p_stage1_lmdb_training.sh
  Stage 1 LMDB baseline training entrypoint.

scripts/train/tmux_run_clean_480p720p_stage1_lmdb_training.sh
  tmux wrapper for Stage 1 training.
```

### Evaluation

```text
scripts/eval/eval_clean_resizer_operator_compare.py
  Decode validation LMDB samples and compute reference metrics.

scripts/eval/run_clean_480p720p_operator_compare_multigpu.sh
  Multi-GPU operator compare wrapper.

scripts/eval/tmux_run_clean_480p720p_operator_compare_multigpu.sh
  tmux wrapper for operator compare.

scripts/eval/run_clean_480p720p_chain_ab_compare.sh
  Single-worker LightX2V chain A/B compare.

scripts/eval/run_clean_480p720p_chain_ab_compare_multigpu.sh
  Multi-GPU chain A/B compare wrapper.

scripts/eval/tmux_run_clean_480p720p_chain_ab_compare_multigpu.sh
  tmux wrapper for chain A/B compare.
```

### LightX2V Bridge

```text
scripts/bridge/run_lightx2v_clean_bridge_infer.py
  Local LightX2V inference wrapper that registers the clean-resizer bridge.

../lightx2v_clean_bridge.py
  Runtime integration: replaces clean-latent interpolation in LightX2V.
```

### Legacy / Historical

```text
scripts/data/build_480p720p_latents.py
  Older per-sample safetensors latent-pair builder.

scripts/legacy/run_clean_480p720p_training.sh
  Older generate/build/train wrapper for the safetensors path.

scripts/legacy/tmux_run_clean_480p720p_all.sh
  tmux wrapper for the older all-in-one safetensors path.

scripts/legacy/run_clean_480p720p_compare_batch10.sh
  Older four-way compare: ori480 / ori720 / interp720 / trained720.

scripts/legacy/apply_clean_resizer_to_video.py
  Offline video-to-video utility for applying a checkpoint through VAE.
```

## Configs

```text
changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage1.yaml
  Stage 1 LMDB training config.

changing_resolution/configs/train_clean_480p_to_720p.yaml
  Older safetensors training config.

changing_resolution/configs/wan_t2v_generate_720p.json
  Wan2.1 720p generation config.

configs/local_paths.sh
  Machine-specific path defaults. Override with environment variables.
```

## Metrics

For operator compare, success means:

```text
trained_psnr  > interp_psnr
trained_ssim  > interp_ssim
trained_lpips < interp_lpips
```

For chain A/B compare, there is no reference target. Judge:

```text
sharpness
temporal stability
less flicker
less texture crawling
less subject deformation
```

Native 720p generation is not a valid reference for changing-resolution output,
because native 720p and changing-resolution follow different diffusion
trajectories.
