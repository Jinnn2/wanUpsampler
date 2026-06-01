# Changing Resolution Distill

This directory keeps the 4-step Wan distill changing-resolution path separate
from the original 50-step Stage 3 workflow.

The intended contract mirrors Stage 3 while changing the denoiser domain:

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
`infer_steps=4`, and `denoising_step_list=[1000,750,500,250]`.

## Layout

```text
changing_resolution_distill/
  configs/
    wan_t2v_distill_stage3_x0pred_480p.json
    wan_t2v_distill_stage3_bridge_720p.example.json
    train_x0pred_480p_to_720p_lmdb_stage3_distill.yaml
  scripts/
    bridge/run_lightx2v_distill_bridge_infer.py
    data/build_x0pred_480p720p_stage3_distill_lmdb.py
    data/build_x0pred_480p720p_stage3_distill_lmdb.sh
    data/tmux_build_x0pred_480p720p_stage3_distill_lmdb_steps_1_2_3.sh
    train/run_x0pred_480p720p_stage3_distill_lmdb_training.sh
    train/tmux_run_x0pred_480p720p_stage3_distill_lmdb_steps_1_2_3_training.sh
  lightx2v_distill_bridge.py
```

## Build LMDB

```bash
HANDOFF_STEP=2 MAX_SAMPLES=32 OVERWRITE=1 \
bash changing_resolution_distill/scripts/data/build_x0pred_480p720p_stage3_distill_lmdb.sh
```

Build handoff steps 1/2/3 in one tmux session:

```bash
STEPS=1,2,3 TOTAL_SAMPLES=1000 OVERWRITE=1 \
bash changing_resolution_distill/scripts/data/tmux_build_x0pred_480p720p_stage3_distill_lmdb_steps_1_2_3.sh
```

Defaults:

```text
source: data/changing_resolution/lmdb_480p720p_1k
output: data/changing_resolution_distill/lmdb_x0pred_480p720p_stage3_distill_step2
model_path: /mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill
model_cls: wan2.1_distill
infer_steps: 4
denoising_step_list: 1000 750 500 250
sample_shift: 5
sample_guide_scale: 6
config: changing_resolution_distill/configs/wan_t2v_distill_stage3_x0pred_480p.json
dit_original_ckpt: lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt
```

## Train

```bash
HANDOFF_STEP=2 MAX_STEPS=10000 \
bash changing_resolution_distill/scripts/train/run_x0pred_480p720p_stage3_distill_lmdb_training.sh train
```

Train handoff steps 1/2/3 in parallel, one GPU per step:

```bash
STEPS=1,2,3 GPU_IDS=0,1,2 MAX_STEPS=10000 \
bash changing_resolution_distill/scripts/train/tmux_run_x0pred_480p720p_stage3_distill_lmdb_steps_1_2_3_training.sh
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

`resize_flow` preserves the distill step-post shape more closely, while
`random` is a conservative baseline aligned with the original changing
resolution idea of switching to a target-resolution noise bank.

Use `configs/wan_t2v_distill_stage3_bridge_720p.example.json` as the starting
point for a real infer config. Replace the `wan_clean_resizer_*` placeholder
paths with the trained step checkpoint and local repo path before running
`scripts/bridge/run_lightx2v_distill_bridge_infer.py`.
