# UNIV Video, Timing, and VBench Validation

The validation suite generates paired videos with identical prompts and seeds,
measures synchronized warm-model latency, evaluates VBench-5 plus dynamic
degree, and reports speedup and quality change against native Wan2.1.

The native case uses the ordinary 50-step HR Wan scheduler and DiT execution,
but initializes it with the same coordinate-hash HR Gaussian field used by
UNIV. Therefore the native HR tensor and every UNIV low-grid anchor share the
same random field; paired comparisons do not silently compare different noise
samples.

## Default paths

```text
Project:       /mnt/afs_2/houze/wanUpsampler
Wan model:     /mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B
LightX2V:      /mnt/afs_2/houze/LightX2V
Real-ESRGAN:   /mnt/afs_2/houze/Real-ESRGAN
VBench:        /mnt/afs_2/houze/VBench
Wan Python:    /opt/conda/bin/python
VBench Python: auto-detect isolated vbench env, then /opt/conda/bin/python
```

The launcher tests actual `torch` and `vbench` imports before selecting an
interpreter. `VBENCH_PYTHON=/custom/env/bin/python` overrides auto-detection.

## Profiles

- `smoke`: native, DVG joint compression, and RGB joint compression.
- `core`: native, DVG identity control, and DVG/RGB joint compression at
  switch ratios 0.6, 0.8, and 1.0.
- `full`: core plus isolated spatial, temporal, and LR-NFE/cache ablations.

The exact actions are versioned in
`configs/univ_validation_cases.json`. Every profile contains exactly one native
baseline. The protocol manifest freezes the prompt file hash, selected prompts,
seeds, cases, actions, and generated config hashes.

## Quick smoke run

```bash
cd /mnt/afs_2/houze/wanUpsampler

PROFILE=smoke \
LIMIT=3 \
TIMING_WARMUP=1 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_validation_smoke \
bash UNIV_adaptor/scripts/run_univ_validation_suite.sh check
```

Generate, score, and summarize:

```bash
PROFILE=smoke \
LIMIT=3 \
TIMING_WARMUP=1 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_validation_smoke \
ENABLE_TRANSITION_DIAGNOSTICS=1 \
bash UNIV_adaptor/scripts/run_univ_validation_suite.sh all
```

## Core evaluation

The recommended first comparison uses ten paired prompts. The first generated
video in each case warms kernels and lazy transition models; the remaining nine
are used for latency and speedup.

Transition FFT/state diagnostics are disabled by default in this timing suite,
because they are research instrumentation absent from the native production
path. To generate diagnostic sidecars, run a separate smoke output with
`ENABLE_TRANSITION_DIAGNOSTICS=1`; do not combine those timings with the formal
speedup table.

```bash
cd /mnt/afs_2/houze/wanUpsampler

PROFILE=core \
LIMIT=10 \
TIMING_WARMUP=1 \
SEED=9700 \
GPU_ID=0 \
VBENCH_GPU_IDS=0 \
VBENCH_NGPUS=1 \
RESUME=1 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_validation_core_10p \
bash UNIV_adaptor/scripts/run_univ_validation_suite.sh all 2>&1 | \
  tee /mnt/afs_2/houze/wanUpsampler/outputs/univ_validation_core_10p/run.log
```

For multi-GPU VBench after single-GPU generation:

```bash
PROFILE=core \
LIMIT=10 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_validation_core_10p \
VBENCH_GPU_IDS=0,1,2,3 \
VBENCH_NGPUS=4 \
bash UNIV_adaptor/scripts/run_univ_validation_suite.sh vbench

PROFILE=core \
LIMIT=10 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_validation_core_10p \
bash UNIV_adaptor/scripts/run_univ_validation_suite.sh summarize
```

The `core` profile contains exactly eight independent cases and can run one
resident model per GPU:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
PROFILE=core \
LIMIT=10 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_validation_core_10p_8gpu \
bash UNIV_adaptor/scripts/run_univ_validation_8gpu.sh plan

GPU_IDS=0,1,2,3,4,5,6,7 \
PROFILE=core \
LIMIT=10 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_validation_core_10p_8gpu \
bash UNIV_adaptor/scripts/run_univ_validation_8gpu.sh all
```

The launcher writes one log per case under `logs/8gpu_generation`, rejects a
second concurrent writer through an atomic output-root lock, and preserves
completed lanes for `RESUME=1`. VBench and report generation run only after all
eight lanes finish. This is fixed-action E3/E7 discovery evidence, not the
common-probe controller training dataset defined in `DATA_GENERATION.md`.

## Staged and resumed execution

Each phase can run independently:

```bash
bash UNIV_adaptor/scripts/run_univ_validation_suite.sh prepare
bash UNIV_adaptor/scripts/run_univ_validation_suite.sh generate
bash UNIV_adaptor/scripts/run_univ_validation_suite.sh visualize
bash UNIV_adaptor/scripts/run_univ_validation_suite.sh vbench
bash UNIV_adaptor/scripts/run_univ_validation_suite.sh summarize
```

With `RESUME=1`, generation skips only cases whose timing rows, prompt indices,
seeds, videos, and UNIV sidecars are complete and consistent. VBench reuses a
score only when the video and prompt-map content fingerprint matches. Changing
the profile, seed, prompt selection, or action while retaining `OUT_ROOT`
fails before overwriting evidence; use a new output directory.

## Outputs

```text
<OUT_ROOT>/run_manifest.json
<OUT_ROOT>/configs/<case>.json
<OUT_ROOT>/videos/<case>/*.mp4
<OUT_ROOT>/videos/<case>/*.mp4.univ.json
<OUT_ROOT>/timings/<case>.jsonl
<OUT_ROOT>/comparisons/<group>/*.mp4
<OUT_ROOT>/comparisons/layout.json
<OUT_ROOT>/metrics/vbench_scores.json
<OUT_ROOT>/reports/summary.csv
<OUT_ROOT>/reports/per_video.csv
<OUT_ROOT>/reports/paired_vbench_vs_native.csv
<OUT_ROOT>/reports/summary.json
<OUT_ROOT>/reports/SUMMARY.md
```

`pipeline_mean_s` is synchronized warm-model wall time for the complete
generation pipeline, including final VAE decode and video save. Model
initialization is recorded separately. `speedup_vs_native` is the native mean
divided by the candidate mean; `paired_speedup_mean` averages prompt-matched
native/candidate ratios. Transition diagnostics have their own timing field and
remain visible rather than being silently excluded.

VBench quality is the arithmetic mean of subject consistency, background
consistency, motion smoothness, aesthetic quality, and imaging quality.
Dynamic degree is reported separately because it measures motion magnitude,
not monotonic visual quality.
