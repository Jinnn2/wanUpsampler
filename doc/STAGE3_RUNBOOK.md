# Stage 3 Runbook

Stage 3 keeps the Stage 2 model architecture, but changes the training input
domain:

```text
Stage 2: z0_lr       -> z0_hr
Stage 3: x0_pred_lr  -> z0_hr
```

The default Stage 3 data recipe is aligned with the real bridge handoff:

```text
clean 480p latent z0_lr
  -> add flow noise at step 35 in a 50-step Wan schedule
  -> run one Wan denoiser forward pass
  -> x0_pred_lr = x_t - sigma_t * noise_pred
```

During training, `x0_pred_lr` is the model input, `z0_hr` is the HR target, and
the low-frequency consistency loss still compares the prediction downsampled to
the clean `z0_lr`. This makes the model learn both inference-domain cleanup and
1.5x latent resizing.

## Build Stage 3 LMDB

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash changing_resolution/scripts/data/tmux_build_x0pred_480p720p_stage3_lmdb.sh
```

Multi GPU:

```bash
TOTAL_SAMPLES=1000 GPU_IDS=0,1,2,3 OVERWRITE=1 \
bash changing_resolution/scripts/data/tmux_build_x0pred_480p720p_stage3_lmdb_multigpu.sh
```

Defaults:

```text
source: data/changing_resolution/lmdb_480p720p_1k
output: data/changing_resolution/lmdb_x0pred_480p720p_stage3_step35
infer_steps: 50
denoise_step: 35
model: /mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B
config: changing_resolution/configs/wan_t2v_stage3_x0pred_480p.json
```

Useful overrides:

```bash
DENOISE_STEP=35 MAX_SAMPLES=32 OVERWRITE=1 \
bash changing_resolution/scripts/data/tmux_build_x0pred_480p720p_stage3_lmdb.sh
```

For a multi-GPU partial build, use `TOTAL_SAMPLES` instead of `MAX_SAMPLES`:

```bash
DENOISE_STEP=35 TOTAL_SAMPLES=32 GPU_IDS=0,1,2,3 OVERWRITE=1 \
bash changing_resolution/scripts/data/tmux_build_x0pred_480p720p_stage3_lmdb_multigpu.sh
```

For a fast plumbing check that does not run Wan, use:

```bash
MODE=clean_copy MAX_SAMPLES=2 OVERWRITE=1 \
bash changing_resolution/scripts/data/tmux_build_x0pred_480p720p_stage3_lmdb.sh
```

`clean_copy` is only a schema smoke test. It should not be used as a real Stage
3 training set.

## Preflight

```bash
bash changing_resolution/scripts/train/run_x0pred_480p720p_stage3_lmdb_training.sh check
```

## Train

```bash
CUDA_VISIBLE_DEVICES=0 \
bash changing_resolution/scripts/train/tmux_run_x0pred_480p720p_stage3_lmdb_training.sh
```

Output:

```text
outputs/changing_resolution_x0pred_480p720p_stage3_lmdb
```

## Notes

The Stage 3 checkpoint remains loadable through the existing Stage 2 bridge
model path because the neural architecture and checkpoint schema are unchanged.
The main difference is the LMDB schema and training distribution.
