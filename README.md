# wanUpsampler

`wanUpsampler` is now organized around the Stage 2 `changing_resolution` path:
a learned clean-latent resizer for replacing LightX2V's fixed 480p -> 720p
interpolation step.

## Mainline

Read first:

```text
changing_resolution/README.md
changing_resolution/STAGE2_RUNBOOK.md
```

Recommended remote entrypoints:

```bash
# 1. Build the 1k 480p/720p clean-latent LMDB.
bash changing_resolution/scripts/data/tmux_build_clean_lmdb_480p720p_1k_multigpu.sh

# 2. Train Stage 2.
bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage2_lmdb_training.sh

# 3. Decode-level operator compare against the validation LMDB reference.
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_operator_compare_multigpu.sh

# 4. LightX2V generation-chain A/B compare.
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_chain_ab_compare_multigpu.sh

# Optional: sweep the handoff step and compare stop480 / interp720 / stage2.
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_change_step_sweep_multigpu.sh

# Stage 3 x0-pred checkpoint sweep smoke test.
bash changing_resolution/scripts/eval/run_x0pred_480p720p_stage3_change_step_sweep.sh
```

## Repository Layout

```text
wan_sr/
  Main Python package: data, losses, schedulers, training utilities, VAE wrapper,
  and the Stage 2 model.

changing_resolution/
  Current Stage 2 clean-latent 480p -> 720p route and LightX2V bridge.

configs/
  Machine-specific path defaults. Stage 2 training configs live under
  changing_resolution/configs/.

experiments/
  External references and exploratory code, not current entrypoints.

.archive/
  Hidden in-repo archive for retired Stage 1 and V1 code.
```

## Path Config

Machine-specific defaults live in:

```text
configs/local_paths.sh
```

Override it with `PATH_CONFIG` when needed:

```bash
PATH_CONFIG=/path/to/local_paths.sh \
bash changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh check
```

## Archive

Retired code is kept in the repo for traceability:

```text
.archive/stage1/
  Stage 1 residual clean-resizer model, configs, wrappers, and old docs.

.archive/v1/
  Early noisy-to-clean V1 scripts, configs, and model implementation.
```

The archive is not imported or used by the current mainline.

## Install

```bash
pip install -r requirements.txt
```

Full training and inference require the Linux GPU environment plus working
LightX2V, Wan2.1 model, and Wan VAE paths.
