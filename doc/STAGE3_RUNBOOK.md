# Stage 3 Runbook

Stage 3 keeps the Stage 2 model architecture, but changes the training input
domain:

```text
Stage 2: z0_lr       -> z0_hr
Stage 3: x0_pred_lr  -> x0_pred_hr
```

The default Stage 3 data recipe is aligned with the real bridge handoff:

```text
clean 480p latent z0_lr
  -> add flow noise at step 35 in a 50-step Wan schedule
  -> run one Wan denoiser forward pass
  -> x0_pred_lr = x_t - sigma_t * noise_pred

clean 720p latent z0_hr
  -> add flow noise at the same step
  -> run one Wan denoiser forward pass
  -> x0_pred_hr = x_t - sigma_t * noise_pred
```

During training, `x0_pred_lr` is the model input and `x0_pred_hr` is the HR
target. For compatibility the LMDB stores the HR target in the existing `z0_hr`
slot, but metadata marks `hr_target_domain: x0_pred_hr`. The low-frequency
consistency loss now defaults to comparing the prediction downsampled to the
input-domain `x0_pred_lr`; use `--low_freq_reference clean_lr` only for the old
clean-target recipe.

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
output: data/changing_resolution/lmdb_x0pred_480p720p_stage3_x0predhr_step45
infer_steps: 50
denoise_step: 45
model: /mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B
config: changing_resolution/configs/wan_t2v_stage3_x0pred_480p.json
hr_target_mode: x0_pred
```

Useful overrides:

```bash
DENOISE_STEP=35 MAX_SAMPLES=32 OVERWRITE=1 \
bash changing_resolution/scripts/data/tmux_build_x0pred_480p720p_stage3_lmdb.sh
```

To rebuild the older `x0_pred_lr -> clean z0_hr` recipe for ablation:

```bash
HR_TARGET_MODE=clean DENOISE_STEP=35 MAX_SAMPLES=32 OVERWRITE=1 \
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

## Change-Step Sweep

Stage 3 sweep uses the same three-way panel format as Stage 2, but the learned
branch loads the Stage 3 x0-pred checkpoint:

```text
stop480 at step N | interp720 step N -> 50 | stage3 720 step N -> 50
```

The single-GPU script is intentionally small by default so you can smoke-test the
50k checkpoint first:

```bash
bash changing_resolution/scripts/eval/run_x0pred_480p720p_stage3_change_step_sweep.sh
```

Defaults:

```text
LIMIT: 1
CHANGE_STEPS: 35
checkpoint: outputs/changing_resolution_x0pred_480p720p_stage3_lmdb/latest.pt
USE_EMA: 1
output: outputs/changing_resolution_stage3_change_step_sweep
```

Useful test runs:

```bash
CHANGE_STEPS="30 35 40" LIMIT=1 \
bash changing_resolution/scripts/eval/run_x0pred_480p720p_stage3_change_step_sweep.sh

CR_STAGE3_CHANGE_STEP_SWEEP_CKPT=outputs/changing_resolution_x0pred_480p720p_stage3_lmdb/step_050000.pt \
CHANGE_STEPS=35 LIMIT=2 \
bash changing_resolution/scripts/eval/run_x0pred_480p720p_stage3_change_step_sweep.sh
```

Four-GPU batch sweep:

```bash
bash changing_resolution/scripts/eval/tmux_run_x0pred_480p720p_stage3_change_step_sweep_multigpu.sh
```

Default batch parameters:

```text
GPU_IDS: 0,1,2,3
TOTAL_PROMPTS: 4
STEP_START / STEP_END / STEP_STRIDE: 10 / 50 / 1
```

That default produces 164 three-way panels because 10..50 inclusive contains 41
handoff steps. Use `STEP_START=1 STEP_END=50 STEP_STRIDE=1` for 200 panels with
4 prompts.
