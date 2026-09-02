# UNIV Generation Pipeline v2

This is the first executable implementation of the research design in
`README.md`. It targets Wan2.1 T2V with the canonical 50-step trajectory and a
single GPU.

## Runtime contract

The action is:

```text
(spatial_ratio, temporal_ratio, lr_nfe_ratio, switch_ratio)
```

`switch_ratio` is restricted to `0.6`, `0.8`, and `1.0`, corresponding to
reference boundaries 30, 40, and 50. `lr_nfe_ratio` controls the exact number
of LR positions that run a full DiT recomputation. All other LR positions keep
the reference scheduler trajectory but reuse a cached model-output residual.
The first and final LR positions are always recomputed.

The shared execution path is:

```text
coordinate-aligned LR noise
  -> LR solver prefix (exact top-k full DiT + residual cache reuse)
  -> fresh flow clean estimate at switch
  -> selected transition baseline
  -> coordinate-hash HR re-noise at the reference boundary sigma
  -> reset all LR cache and multistep solver history
  -> full-compute HR reference suffix
  -> Wan VAE decode and save
```

`univ_transition_baseline` selects exactly one of two non-equivalent baselines:

- `dvg_latent_anchor`: applies DVG equations (11)-(12) directly and
  sequentially to latent T, H, and W. Source index `i` is placed at
  `round(i*(N-1)/(K-1))`; every target position is interpolated between its
  neighboring rounded anchors. It does not decode through the VAE or invoke
  Real-ESRGAN.
- `rgb_sr_vae`: decodes the LR clean latent, applies frame-wise Real-ESRGAN x2
  and pixel-space endpoint-aligned temporal interpolation, then re-encodes with
  the Wan VAE. This is the old Stage 2 bridge and is not labeled as DVG.

For a target of `720x1248x81`, the example action
`(0.512, 0.5, 0.5, 0.6)` resolves to:

```text
target latent: 16x21x90x156
LR latent:     16x11x46x80
LR RGB frames: 41
LR solver positions: 30
LR full DiT recomputations: 15
HR full DiT suffix: 20
total full-compute positions: 35
```

With CFG enabled, each full-compute position invokes conditional and
unconditional model paths; the sidecar records this physical pass multiplier
separately.

## Paper conformance boundary

- DVG equations (11)-(12): implemented by `dvg_latent_anchor`, including
  rounded anchors and sequential T/H/W reconstruction for arbitrary `K -> N`.
- DVG equation (13): coordinate-hash Box-Muller noise is shared between LR and
  HR at the same rounded anchors.
- Wan rectified-flow adaptation: the clean endpoint and next state are
  `z0 = x_sigma - sigma*v` and
  `x_sigma = (1-sigma)*z0 + sigma*epsilon`. This is the scheduler-specific
  form used in place of copying a diffusion-model re-noise equation blindly.
- ReCache: exact top-k full-recomputation budget rather than threshold-based
  approximate budget control.
- OnlineCache: model-output residual reuse as the first cache implementation.
- RAPID3: step skipping, cache reuse, and sparse attention remain separate
  executor policies. This v2 implements cache reuse only; it does not conflate
  cache reuse with numerical timestep deletion.

The content-aware DVG sketch/demand selector and the learned RAPID3, ReCache,
and OnlineCache policies are still outside this fixed-action executor.

## Transition diagnostics

Every sidecar records full-state population mean/std/RMS/range, temporal first
difference MAE/RMS, and temporal/spatial spectral power, normalized centroid,
and high-frequency ratio for `clean_lr`, `clean_hr`, and `renoised_hr`.
Spectra are computed on the channel-mean latent; high frequency means at least
half Nyquist. FFTs use orthonormal normalization, so reported mean spectral
power is comparable across different grid sizes by Parseval's identity.

An actual native HR trajectory state at the same boundary can be supplied as a
tensor or tensor dictionary:

```json
{
  "univ_native_hr_state_path": "/path/to/native_hr_boundary_30.pt",
  "univ_native_hr_state_key": "state"
}
```

The sidecar then reports RMSE, MAE, relative L2, and cosine distance from the
re-noised transition state. Without that external teacher state, the metric is
explicitly marked `available: false`; it is never replaced by a self-distance.

## Run

On the Linux inference machine:

```bash
cd /mnt/afs_2/houze/wanUpsampler

LIGHTX2V_REPO=/mnt/afs_2/houze/LightX2V \
REALESRGAN_REPO=/mnt/afs_2/houze/Real-ESRGAN \
MODEL_ROOT=/path/to/Wan2.1-T2V-1.3B \
REALESRGAN_X2_CKPT=/path/to/RealESRGAN_x2plus.pth \
bash UNIV_adaptor/scripts/run_wan_univ_rgb_pipeline.sh check
```

Run one sample:

```bash
TRANSITION_BASELINE=rgb_sr_vae \
SPATIAL_RATIO=0.512 \
TEMPORAL_RATIO=0.5 \
LR_NFE_RATIO=0.5 \
SWITCH_RATIO=0.6 \
PROMPT='A cinematic shot of a red fox walking through a snowy forest.' \
bash UNIV_adaptor/scripts/run_wan_univ_rgb_pipeline.sh run
```

The paper-geometry baseline does not require Real-ESRGAN:

```bash
TRANSITION_BASELINE=dvg_latent_anchor \
bash UNIV_adaptor/scripts/run_wan_univ_rgb_pipeline.sh run
```

Outputs:

```text
outputs/univ_adaptor_smoke/sample.mp4
outputs/univ_adaptor_smoke/sample.mp4.univ.json
outputs/univ_adaptor_smoke/resolved_config.json
```

The sidecar records requested and realized ratios, full-compute/cache step
indices, transition shapes, boundary sigma, CFG multiplier, and per-stage
timings.

## Validation commands

The planning/noise tests do not import Torch and can run locally:

```bash
python -m unittest discover -s UNIV_adaptor/tests -v
python -m py_compile UNIV_adaptor/*.py
```

Full runtime validation requires the Linux CUDA environment, Wan weights,
LightX2V, and the full Wan VAE. The `rgb_sr_vae` baseline additionally requires
Real-ESRGAN/BasicSR and the x2 checkpoint.
