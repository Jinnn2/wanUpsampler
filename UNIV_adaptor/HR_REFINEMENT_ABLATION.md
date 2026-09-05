# Fixed-boundary HR refinement: 10 / 6 / 4 / 2 steps

This is a four-video quality ablation for one prompt and seed. It asks whether
larger HR solver steps can replace the original 10-step refinement suffix.

The LR prefix and DVG anchor transition execute **once**. Each branch clones the
same saved HR tensor and starts at the same sigma, approximately 0.666389.
Every retained HR position performs a complete DiT evaluation; no HR prediction
cache is used. The final solver endpoint is sigma=0 for all four branches.

## Fixed settings

- Wan2.1 T2V 1.3B, output 720 x 1248, 81 frames, seed 42.
- Default prompt: a tracking shot of a red fox walking through a snowy forest.
- Reference: 50 steps, sample_shift=8, CFG=6.
- Prefix: spatial_ratio=0.75, temporal_ratio=0.8, lr_nfe_ratio=1.0.
- Switch: reference boundary 40/50, after 40 full LR evaluations.
- Transition: dvg_latent_anchor, then Wan flow re-noising.
- Refinement: resident weights, full DiT evaluations, restarted UniPC history.

The LR latent is [16,17,68,118]; the shared HR latent is [16,21,90,156].
This isolates HR discretization from LR caching. It does not modify the existing
B30-B70 data protocol and is not a native-HR50 baseline comparison.

## HR grids

Reduced grids linearly interpolate the original sigma sequence at uniformly
spaced fractional **reference step indices**. sample_shift is not applied twice.
HR10 preserves the original sigma values and model timesteps exactly. The
runtime sidecars contain actual float32 sigmas and quantized model timesteps.

| Video | HR evaluations | Approximate sigma sequence, including terminal zero |
| --- | ---: | --- |
| HR10 | 10 | .666389, .636886, .603489, .565371, .521455, .470311, .409993, .337790, .249805, .140228, 0 |
| HR06 | 6 | .666389, .614621, .550732, .470311, .361858, .213279, 0 |
| HR04 | 4 | .666389, .584430, .470311, .293797, 0 |
| HR02 | 2 | .666389, .470311, 0 |

The runner updates both solver sigmas and model timesteps, clears LR multistep
history, and uses the new adjacent intervals in step_post. It restores the
50-step reference length before preparing the next branch.

## Server commands

Run from the inner project checkout. Defaults use the same model and LightX2V
paths as the existing UNIV launchers. Choose an available GPU; this experiment
uses one GPU and loads model weights once.

```bash
cd /mnt/afs_2/houze/wanUpsampler

# No model load: materialize and inspect the comparison plan.
bash UNIV_adaptor/scripts/run_univ_hr_refinement_ablation.sh plan

# Validate model files, runtime imports and CUDA.
GPU_ID=0 bash UNIV_adaptor/scripts/run_univ_hr_refinement_ablation.sh check

# Generate all four videos sequentially.
GPU_ID=0 bash UNIV_adaptor/scripts/run_univ_hr_refinement_ablation.sh run
```

Set MODEL_ROOT, LIGHTX2V_REPO and WAN_PYTHON if the server paths differ. The
default Python is /opt/conda/bin/python. No Real-ESRGAN dependency is needed.

For a custom prompt, use the same environment settings for plan/check/run:

```bash
export PROMPT='A red fox walks steadily through a snowy forest, detailed fur and drifting snowflakes, smooth tracking camera.'
export SEED=42
export OUT_DIR=/mnt/afs_2/houze/wanUpsampler/outputs/univ_hr_fox_custom_v1
GPU_ID=0 bash UNIV_adaptor/scripts/run_univ_hr_refinement_ablation.sh run
```

A different prompt/config/seed requires a new OUT_DIR. Existing videos and the
shared checkpoint are not overwritten. This minimal script has no partial-run
resume; after an interrupted generation, use a fresh OUT_DIR.

## Artifacts and interpretation

Default directory: outputs/univ_hr_refinement_ablation_v1/

```text
comparison_plan.json
resolved_config.json
shared_hr_boundary.pt
HR10.mp4                HR10.mp4.univ.json
HR06.mp4                HR06.mp4.univ.json
HR04.mp4                HR04.mp4.univ.json
HR02.mp4                HR02.mp4.univ.json
comparison_summary.json
```

The checkpoint stores the actual transition tensor, reference sigmas, prompt,
seed, action and tensor SHA-256. Each branch verifies its actual input tensor
against that hash. The summary is written after each completed video and marks
complete=true only after all four outputs exist.

Use hr_seconds to compare synchronized HR execution time. Whole-pipeline time
for the first branch includes the shared LR prefix/transition; later branches
reuse that work. These are single-pass exploratory timings without a separate
warmup, so they are not publication-grade latency estimates.

Compare fur/branch detail, footprints, motion continuity, flicker and semantic
drift against HR10. A successful single-prompt run demonstrates execution and a
local quality tradeoff; it does not establish a universal optimal HR step count.

## Local validation

```bash
python -m unittest UNIV_adaptor.tests.test_hr_refinement
```

With Torch and a local LightX2V checkout, these tests execute the actual Wan
UniPC source on CPU with analytic/mock velocity fields. They verify HR10 output
equivalence, clean endpoint recovery on all four grids, history reset, one-time
prefix execution, state cloning and actual evaluation counts. Source classes
are loaded without the GPU inference backend imports. Set LIGHTX2V_REPO if the
checkout is not adjacent to this project. Pure planning tests also run without
Torch. Real-model CUDA video generation remains a separate server validation.
