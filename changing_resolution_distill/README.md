# Changing Resolution Distill

This directory keeps the 4-step Wan distill changing-resolution path separate
from the original 50-step Stage 3 workflow.

## Current Direction

As of 2026-06-17, the next mainline is no longer to keep scaling the old
`x0_pred_lr -> z0_hr` Stage 3 objective. That path remains in this directory as
a runnable baseline and ablation surface.

The new plan is documented in:

```text
doc/DISTILL_LAST_STEP_SKIP_LORA_PLAN.md
```

The DiffSynth-Studio based LoRA preflight environment is documented in:

```text
doc/DISTILL_LORA_ENV.md
```

New target:

```text
Phase 1: train a step3-only last-step-skip LoRA on the Wan 4-step distill denoiser
         x3_lr -> z4_lr_teacher

Phase 2: reuse or fine-tune the clean latent upsampler
         z_lr_lora_clean -> z_hr_clean
```

The old Stage 3 contract mirrors Stage 3 while changing the denoiser domain:

```text
clean LR latent z0_lr
  -> add 4-step distill flow noise at handoff step k
  -> run one wan2.1_distill denoiser forward
  -> x0_pred_lr = x_t - sigma_k * flow_pred
  -> train x0_pred_lr -> z0_hr
```

The first version deliberately reuses the existing Stage 2/Stage 3 resizer
architecture and LMDB dataset reader. The main difference is the data recipe
metadata: `stage3_recipe.mode=lightx2v_distill`, `model_cls=wan2.1_distill`,
`distill_model_id=lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill`,
`infer_steps=4`, and `denoising_step_list=[1000,750,500,250]`.

## Layout

```text
changing_resolution_distill/
  configs/
    wan_t2v_distill_stage3_x0pred_480p.json
    wan_t2v_distill_stage3_bridge_720p.example.json
    train_last_step_skip_lora_distill.yaml
    train_clean_480p_to_720p_lmdb_stage2_distill.yaml
    train_x0pred_480p_to_720p_lmdb_stage3_distill.yaml
  scripts/
    bridge/run_lightx2v_distill_bridge_infer.py
    data/generate_wan21_distill_720p_dataset.sh
    data/build_clean_480p720p_14b_cfgdistill_lmdb_1k.sh
    data/build_clean_480p720p_14b_cfgdistill_lmdb_1k_multigpu.sh
    data/build_14b_cfgdistill_720p_clean_and_x0pred_lmdb_1k.sh
    data/build_x0pred_480p720p_stage3_distill_lmdb.py
    data/build_x0pred_480p720p_stage3_distill_lmdb.sh
    data/build_x0pred_480p720p_stage3_distill_lmdb_multigpu.sh
    data/build_last_step_skip_lora_lmdb.py
    data/build_last_step_skip_lora_lmdb.sh
    data/build_last_step_skip_lora_lmdb_multigpu.sh
    data/tmux_build_last_step_skip_lora_lmdb.sh
    data/check_last_step_skip_lora_lmdb.py
    data/tmux_build_x0pred_480p720p_stage3_distill_lmdb_steps_1_2_3.sh
    eval/run_clean_480p720p_stage2_distill_chain_ab_compare.sh
    eval/run_clean_480p720p_stage2_distill_operator_compare_multigpu.sh
    train/run_clean_480p720p_stage2_distill_lmdb_training.sh
    train/setup_last_step_skip_lora_env.sh
    train/check_last_step_skip_lora_env.sh
    train/train_last_step_skip_lora.py
    train/run_last_step_skip_lora_training.sh
    train/tmux_run_last_step_skip_lora_training.sh
    train/tmux_run_clean_480p720p_stage2_distill_lmdb_training.sh
    train/tmux_run_clean_480p720p_stage2_distill_5k_10k_training.sh
    train/tmux_run_clean_480p720p_stage2_distill_5k_30k_ema999_training.sh
    train/run_x0pred_480p720p_stage3_distill_lmdb_training.sh
    train/tmux_run_x0pred_480p720p_stage3_distill_lmdb_steps_1_2_3_training.sh
  lightx2v_distill_bridge.py
```

## Build Last-Step-Skip LoRA LMDB

Version A reuses the existing 5k clean 480p/720p latent LMDB and only generates
the cached teacher `x3_lr` state:

```text
source: data/changing_resolution_distill/lmdb_clean_480p720p_14b_cfgdistill_5k
output: data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3
fields: x3_lr, z4_lr_teacher, z0_hr, prompt, seed, meta
```

`x3_lr` and `z4_lr_teacher` are generated from the same LR teacher rollout:
the builder saves the latent before step 3, then continues the original 4-step
teacher to get the clean LR target. `z0_hr` is copied from the existing 5000 HR
latents and is not regenerated.

Small smoke build:

```bash
MAX_SAMPLES=8 OVERWRITE=1 \
bash changing_resolution_distill/scripts/data/build_last_step_skip_lora_lmdb.sh

python changing_resolution_distill/scripts/data/check_last_step_skip_lora_lmdb.py \
  --expect_samples 8
```

Full 5k multi-GPU build:

```bash
TOTAL_SAMPLES=5000 GPU_IDS=0,1,2,3 OVERWRITE=1 \
bash changing_resolution_distill/scripts/data/build_last_step_skip_lora_lmdb_multigpu.sh
```

Or in tmux:

```bash
TOTAL_SAMPLES=5000 GPU_IDS=0,1,2,3 OVERWRITE=1 \
bash changing_resolution_distill/scripts/data/tmux_build_last_step_skip_lora_lmdb.sh
```

Final check:

```bash
python changing_resolution_distill/scripts/data/check_last_step_skip_lora_lmdb.py \
  --data_dir data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3 \
  --expect_samples 5000
```

Decode-preview 5 samples:

```bash
python changing_resolution_distill/scripts/eval/preview_last_step_skip_lora_lmdb_decode.py \
  --data_dir data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3 \
  --num_samples 5
```

This writes per-sample videos for `x3_lr`, `z4_lr_teacher`, and `z4_hr`
(`z0_hr` in the LMDB), plus a three-column compare panel under
`outputs/changing_resolution_distill_last_step_skip_lora_preview`.

## Train Last-Step-Skip LoRA

The first trainer uses DiffSynth-Studio's trainable Wan module and the cached
latent LMDB. It does not use the LightX2V inference runner for backpropagation.

Preflight:

```bash
bash changing_resolution_distill/scripts/train/run_last_step_skip_lora_training.sh check
```

Tiny overfit smoke:

```bash
MAX_STEPS=200 MAX_SAMPLES=64 \
bash changing_resolution_distill/scripts/train/run_last_step_skip_lora_training.sh smoke
```

Full 10k run:

```bash
MAX_STEPS=10000 \
bash changing_resolution_distill/scripts/train/run_last_step_skip_lora_training.sh train
```

If DiffSynth cannot infer the local model files, pass either `MODEL_PATHS` or
`MODEL_ID_WITH_ORIGIN_PATHS` through the wrapper environment. By default the
wrapper uses `CR_DISTILL_MODEL_ROOT`, which defaults to
`/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill`. The output
directory contains `latest.safetensors`, step safetensors, `latest.pt`, and
`metrics.jsonl`.

DiffSynth expects `MODEL_PATHS` to be a JSON list string:

```bash
MODEL_PATHS='["/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill"]' \
bash changing_resolution_distill/scripts/train/run_last_step_skip_lora_training.sh smoke
```

## Build LMDB

Full rebuild from new 14B CfgDistill 720p videos:

```bash
TOTAL_SAMPLES=5000 GPU_IDS=0,1,2,3 STEPS=1,2,3 OVERWRITE_LMDB=1 OVERWRITE_X0PRED=1 \
bash changing_resolution_distill/scripts/data/build_14b_cfgdistill_720p_clean_and_x0pred_lmdb_1k.sh all
```

This produces:

```text
data/changing_resolution_distill/raw_wan21_14b_cfgdistill_720p_5k
data/changing_resolution_distill/lmdb_clean_480p720p_14b_cfgdistill_5k
data/changing_resolution_distill/lmdb_x0pred_480p720p_stage3_14b_cfgdistill_5k_step1
data/changing_resolution_distill/lmdb_x0pred_480p720p_stage3_14b_cfgdistill_5k_step2
data/changing_resolution_distill/lmdb_x0pred_480p720p_stage3_14b_cfgdistill_5k_step3
```

Video generation uses a persistent LightX2V runner: each GPU worker loads the
14B CfgDistill model once, then loops over its prompt shard.

To reuse an existing 1k raw-video dataset and only generate the remaining 4k:

```bash
TOTAL_SAMPLES=5000 GPU_IDS=0,1,2,3 \
bash changing_resolution_distill/scripts/data/copy_14b_cfgdistill_1k_raw_to_5k.sh

TOTAL_SAMPLES=5000 GPU_IDS=0,1,2,3 OVERWRITE_LMDB=0 \
bash changing_resolution_distill/scripts/data/build_clean_480p720p_14b_cfgdistill_lmdb_1k_multigpu.sh generate

TOTAL_SAMPLES=5000 GPU_IDS=0,1,2,3 OVERWRITE_LMDB=1 \
bash changing_resolution_distill/scripts/data/build_clean_480p720p_14b_cfgdistill_lmdb_1k_multigpu.sh lmdb

TOTAL_SAMPLES=5000 GPU_IDS=0,1,2,3 STEPS=1,2,3 OVERWRITE_X0PRED=1 \
bash changing_resolution_distill/scripts/data/build_14b_cfgdistill_720p_clean_and_x0pred_lmdb_1k.sh x0pred
```

```bash
HANDOFF_STEP=2 MAX_SAMPLES=32 OVERWRITE=1 \
bash changing_resolution_distill/scripts/data/build_x0pred_480p720p_stage3_distill_lmdb.sh
```

Build handoff steps 1/2/3 in one tmux session:

```bash
STEPS=1,2,3 TOTAL_SAMPLES=5000 OVERWRITE=1 \
bash changing_resolution_distill/scripts/data/tmux_build_x0pred_480p720p_stage3_distill_lmdb_steps_1_2_3.sh
```

Build one handoff step with multiple GPUs:

```bash
HANDOFF_STEP=2 TOTAL_SAMPLES=5000 GPU_IDS=0,1,2,3 OVERWRITE=1 \
bash changing_resolution_distill/scripts/data/build_x0pred_480p720p_stage3_distill_lmdb_multigpu.sh
```

Defaults:

```text
source: data/changing_resolution_distill/lmdb_clean_480p720p_14b_cfgdistill_5k
output: data/changing_resolution_distill/lmdb_x0pred_480p720p_stage3_14b_cfgdistill_5k_step2
model_path: /mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill
distill_model_id: lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill
model_cls: wan2.1_distill
infer_steps: 4
denoising_step_list: 1000 750 500 250
sample_shift: 5
sample_guide_scale: 6
config: changing_resolution_distill/configs/wan_t2v_distill_stage3_x0pred_480p.json
dit_original_ckpt: /mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt
```

## Train Stage 2

Stage 2 trains the clean-latent resizer directly on the distill clean LMDB:
`z0_lr -> z0_hr`. It reuses the shared
`changing_resolution/scripts/train/train_clean_latent_resizer_stage2.py`
trainer and keeps the checkpoint/config/output paths under the distill tree.

Single-run entrypoint:

```bash
MAX_STEPS=10000 \
bash changing_resolution_distill/scripts/train/run_clean_480p720p_stage2_distill_lmdb_training.sh train
```

Defaults:

```text
source: data/changing_resolution_distill/lmdb_clean_480p720p_14b_cfgdistill_5k
config: changing_resolution_distill/configs/train_clean_480p_to_720p_lmdb_stage2_distill.yaml
output: outputs/changing_resolution_distill_clean_480p720p_stage2_14b_cfgdistill_5k_lmdb
```

The maintained tmux launcher mirrors the distill Stage 3 launcher style: it
writes a runnable script under `logs/`, records a top-level run log plus a worker
log, and resumes from `latest.pt` when `AUTO_RESUME=1`.

```bash
GPU_IDS=0 MAX_STEPS=10000 \
bash changing_resolution_distill/scripts/train/tmux_run_clean_480p720p_stage2_distill_lmdb_training.sh
```

For the current 5k 14B CfgDistill dataset, the dedicated 10k-step launcher is:

```bash
bash changing_resolution_distill/scripts/train/tmux_run_clean_480p720p_stage2_distill_5k_10k_training.sh
```

For the current 5k 14B CfgDistill dataset, the dedicated 30k-step EMA-0.999
launcher is:

```bash
bash changing_resolution_distill/scripts/train/tmux_run_clean_480p720p_stage2_distill_5k_30k_ema999_training.sh
```

Quick checks after a checkpoint exists:

```bash
TOTAL_SAMPLES=32 GPU_IDS=0,1,2,3 \
bash changing_resolution_distill/scripts/eval/run_clean_480p720p_stage2_distill_operator_compare_multigpu.sh

LIMIT=8 CHANGE_STEP=2 USE_EMA=0 \
bash changing_resolution_distill/scripts/eval/run_clean_480p720p_stage2_distill_chain_ab_compare.sh
```

## Train Stage 3

```bash
HANDOFF_STEP=2 MAX_STEPS=10000 \
bash changing_resolution_distill/scripts/train/run_x0pred_480p720p_stage3_distill_lmdb_training.sh train
```

Train handoff steps 1/2/3 in parallel, one GPU per step:

```bash
STEPS=1,2,3 GPU_IDS=0,1,2 MAX_STEPS=10000 \
bash changing_resolution_distill/scripts/train/tmux_run_x0pred_480p720p_stage3_distill_lmdb_steps_1_2_3_training.sh
```

For the current 5k 14B CfgDistill dataset, the dedicated 10k-step launcher is:

```bash
bash changing_resolution_distill/scripts/train/tmux_run_x0pred_480p720p_stage3_distill_5k_10k_steps_1_2_3_training.sh
```

The trainer is still the existing Stage 3 trainer. The distill wrapper passes
`--denoise_step ${HANDOFF_STEP}` so the current LMDB metadata guard keeps the
handoff-step data and checkpoint aligned.

## Bridge

The runtime bridge registers:

```text
wan2.1_distill_clean_resizer_bridge
```

It runs 4-step distill at LR until `changing_resolution_steps`, resizes the
current clean `x0` estimate, then continues in HR. The bridge supports two
renoise modes:

```text
random      : x_next_hr = (1 - sigma_next) * x0_hr + sigma_next * fixed_hr_noise
resize_flow : x_next_hr = x0_hr + sigma_next * trilinear_resize(flow_pred_lr)
```

`random` is the default and mirrors the non-distill bridge contract: infer the
current clean `x0`, resize only that clean latent, then re-noise with the
target-resolution noise bank. `resize_flow` is kept as an explicit ablation mode
for experiments that intentionally resize the predicted flow.

Use `configs/wan_t2v_distill_stage3_bridge_720p.example.json` as the starting
point for a real infer config. Replace the `wan_clean_resizer_*` placeholder
paths with either the trained Stage 2 checkpoint above or a
`stage3_14b_cfgdistill_5k` step checkpoint and local repo path before running
`scripts/bridge/run_lightx2v_distill_bridge_infer.py`.
