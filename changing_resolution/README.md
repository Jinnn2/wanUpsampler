# Changing Resolution Clean Latent Training

This folder contains the V2 training path for matching LightX2V's native
`changing_resolution` interface.

The target is clean-latent resizing:

```text
x0_pred_lr or z0_lr -> z0_hr
```

This differs from the V1 noisy bridge:

```text
x_t_lr + sigma -> z0_hr
```

LightX2V's stock changing-resolution scheduler first estimates a clean latent
with `x0_pred = x_t - sigma * eps`, resizes that clean estimate, then re-noises
it before continuing diffusion. Therefore this training path uses clean LR/HR
Wan VAE latent pairs.

Default pixel sizes are Wan-friendly approximations of 480p to 720p:

```text
LR RGB: 480 x 832
HR RGB: 720 x 1248
latent: 60 x 104 -> 90 x 156
scale: 1.5x spatial
```

Run on the Linux training machine:

```bash
cd /data/yongyang/Jin/wanUpsampler

bash changing_resolution/scripts/run_clean_480p720p_training.sh download
bash changing_resolution/scripts/run_clean_480p720p_training.sh build
bash changing_resolution/scripts/run_clean_480p720p_training.sh train
```

Or run all steps:

```bash
bash changing_resolution/scripts/run_clean_480p720p_training.sh all
```

Common overrides:

```bash
MAX_STEPS=20000 LR=5e-5 GRAD_ACCUM=16 \
bash changing_resolution/scripts/run_clean_480p720p_training.sh train
```
