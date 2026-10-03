# Wan2.1 published methods + existing endpoint S/T pipeline

This is a NEW frozen experiment. The original `published_wan21_pilot_v3`
driver, worker, plan, calibration and Jenga diagnostic assets are not modified.

Research claim under test: aggregate quality scores may conceal perceptible,
content-dependent acceleration degradation. Neither metric inadequacy nor
prompt-only method/budget routing benefit is assumed. Published caching arms,
custom S/T arms and mechanistic controls must be reported separately.

## Existing implementation reused

- `mrflow_ablation_runner.py`: finish the actual LR trajectory at sigma zero,
  then use an independent direct HR schedule (not the final four main steps).
- `flow.py::wan_renoise`: `(1-sigma)*clean + sigma*noise` for Wan rectified flow.
- `hr_refinement.py::direct_hr_sigmas`: `[0.2, 0.15, 0.1, 0.05, 0]`.
- `transition.py::WanRGBSRTransition` and `rgb_super_resolution.py`: native
  VAE decode, framewise RealESRGAN x2, deterministic VAE encode.
- `transition.py::WanDVGAnchorTransition`: rounded-anchor latent time
  reconstruction. This is DVG-inspired restoration, NOT an official DVG
  reproduction or its content-adaptive selector. RGB SR is our existing
  endpoint recipe, not attributed to DVG.
- `hy15_runtime.py`: earlier RGB restoration + sigma 0.2/HR4 implementation.

The new `wan21_endpoint_runtime.py` only adapts the native Wan VAE API and
native UniPC scheduler to those helpers, avoiding a LightX2V dependency. Native
main sigmas retain shift 8. HR uses a fresh UniPC solver with shift 1 and four
explicit adjacent intervals; no LR solver history is carried across.

## Preregistered arms

Output for every scored clip is **832x480, 81 frames, 16 fps**. Shared CFG=6,
negative prompt, BF16 model, float32 native latent, FA2 dense attention, no
offload, two seeds 42/3407, same 12 public/stress prompts from the original
pilot. Nominal motion/detail cells still require FULL observability review.

| Arm | Main trajectory | Restoration + refinement | Main token density |
| --- | --- | --- | --- |
| FULL50 | Native 50, 832x480x81 | none | 1 |
| STEP25 | Native 25, 832x480x81 | none | 1 (half the steps) |
| TEA008 | Published TeaCache 0.08, 50 | unchanged published pipeline | 1 |
| SCALING10 | Published ScalingCache preset, 50 | unchanged published pipeline | 1 |
| FULL50_HR4 | Native 50, full shape | re-noise + HR4; **no VAE roundtrip** | 1 |
| FULL50_RT_HR4 | Native 50, full shape | VAE roundtrip, no SR, re-noise + HR4 | 1 |
| S_B050 | Native 50, 592x336x81 | RGB SR -> VAE encode -> re-noise + HR4 | 0.4981 |
| S_B025 | Native 50, 416x240x81 | RGB SR -> VAE encode -> re-noise + HR4 | 0.25 |
| T_B050 | Native 50, 832x480x41 | latent time anchors -> re-noise + HR4 | 11/21 = 0.5238 |
| T_B025 | Native 50, 832x480x17 | latent time anchors -> re-noise + HR4 | 5/21 = 0.2381 |

These are **main-pass proxy budgets**, not total compute or promised speedups.
Custom arms have 54 solver updates, with two CFG forwards per update. Measured
latency includes prompt encoding, main denoising, RGB SR/restoration, all VAE
operations, HR4 and final decode. Weight loading, excluded same-shaped warmup,
MP4 encoding and explicit audit/clean-latent saving are recorded separately.
SR model loading is part of excluded process initialization, then reused.

Jenga BASE/OFF remain a separate stratum because BEFORE formal generation its
disabled token-permutation path differed from native FULL and its preset was
slower on this hardware. No changed official forward or relaxed calibration
threshold is used to make Jenga share a false FULL baseline.

## Calibration and resource reuse

There are **240 formal clips** (24 matched groups x 10 arms), and 26 calibration
records (two calibration prompts x 13 arms). Up to **12 old calibration clips**
are reused: FULL50, STEP25, TEA008, SCALING10, TEA_OFF, SCALING_OFF. Exact old
plan/script hashes, model identity, source commits, arm/sampling/prompt/seed,
sample/video hashes and provenance are checked. GPU assignment may differ.
Nothing is copied over the old failed Jenga audit; a NEW audit is mandatory.
If the source root is missing, generate all 26 calibration clips.

The remaining **14 calibration clips** cover six endpoint arms plus
NATIVE_ADAPTER_OFF. Adapter OFF must match native FULL within unchanged
preregistered subsample tolerances. All full-main controls must have identical
clean-main hashes. Main endpoints must reach sigma zero after 50 actual
intervals/100 CFG forwards. HR must have the exact fresh grid and eight full
forwards; paired custom arms must share their separate HR noise.

Different geometries use nested iid samples of the same full noise field.
The original receipt's `noise` binds that full field. `executed_main_noise`
binds the actual smaller tensor, and anchor indices are explicit. Same field
does not mean identical trajectories; it removes an avoidable unmatched-noise
confound, not intrinsic seed variation.

Every new custom-arm main clean endpoint is saved as a float32 `.pt` artifact,
with a hash in its receipt. This supports later restoration/refinement reuse
without re-running the expensive main pass or encoding from lossy MP4. Audit
saving time is excluded. These server-side artifacts are NOT packed into the
small analysis archive. Inspect/move an orphan artifact/video after a crash;
the script will not silently overwrite it.

## Run on the eight-A800 server

Existing Wan native weights are reused. Default SR checkpoint is the previously
downloaded Hunyuan endpoint asset; specify its actual path if different.

```bash
git pull --ff-only
export PUBLISHED_WAN21_STUDY_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/published_wan21_study_v2
export SOURCE_PUBLISHED_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/published_wan21_pilot_v3
export SR_CHECKPOINT=/mnt/afs_2/houze/models/hy15_endpoint_v1/RealESRGAN_x2plus.pth

# Only if this generation venv lacks BasicSR/RealESRGAN:
bash UNIV_adaptor/scripts/run_univ_published_wan21_study_8gpu.sh setup-sr

bash UNIV_adaptor/scripts/run_univ_published_wan21_study_8gpu.sh check
bash UNIV_adaptor/scripts/run_univ_published_wan21_study_8gpu.sh calibrate
bash UNIV_adaptor/scripts/run_univ_published_wan21_study_8gpu.sh audit
# Run this ONLY if the new audit passes:
bash UNIV_adaptor/scripts/run_univ_published_wan21_study_8gpu.sh generate
bash UNIV_adaptor/scripts/run_univ_published_wan21_study_8gpu.sh finalize
bash UNIV_adaptor/scripts/run_univ_published_wan21_study_8gpu.sh score
```

`check` runs imports, argument parsing, FA2 small CUDA smoke tests and, for S
arms, a real small RealESRGAN checkpoint/CUDA test. It does not generate full
videos. `calibrate` DOES fully generate its missing clips, plus excluded
per-process/arm warmup. Expect nontrivial cost. `generate` refuses to start
without the new audit passing. Ctrl+C terminates this launcher's workers;
valid receipts can be resumed. `status` is read-only. Existing unrelated
`DATASET_ROOT`/`PUBLISHED_WAN21_ROOT` variables are deliberately ignored.

For a source-free calibration: `export SOURCE_PUBLISHED_ROOT=none`. If a prior
root exists but is incompatible/tampered, importing raises, not silent reuse.
Native Wan weights are bound by path/size/mtime; this is not a SHA256 claim for
multi-GB weights. The SR checkpoint is SHA256-bound. Do not edit implementation
files after freezing the new plan; use a new output root for code changes.

Scoring uses the existing strict VBench runner and original dimensions. The
five-quality-dimension raw mean is **not official VBench Total**. Reports retain
all methods/control rows, measured wall speedups and prompt-cluster intervals.
Use `blind`, `blind-score`, `blind-report`, `export` as before: 216 preregistered
FULL-vs-arm pairs plus six repeats, including mechanistic controls; no
score-based pair picking. Private role labels distinguish controls, published
cache arms and custom S/T arms. Require multiple independent raters and heldout
prompt families before claiming a benchmark failure or trained router gains.

Local tests exercise protocol binding, reuse, geometry, schedules and mocked
orchestration. **Real native-model/sampler numerical equivalence, RealESRGAN
and full CUDA execution require the server calibration.**
