# Complete the Wan2.1 FlashVSR spatial comparison (four GPUs)

This adds **24 complete videos**: p00–p11 × seeds 42/3407, each **81 frames,
832×480, 16 fps**. It fills the spatial arm of the existing FULL50 / STEP25 /
TEA008 / SCALING10 comparison, giving 24 matched groups and **120 videos**.
The previous eight 33-frame native-LR FlashVSR screening clips remain untouched;
they cannot substitute for full-length clips, so all 24 full clips are processed.

## Run on the existing remote server

```bash
bash UNIV_adaptor/scripts/run_univ_flashvsr_wan21_complete_4gpu.sh all
```

To use the other four cards:

```bash
GPU_IDS=4,5,6,7 bash UNIV_adaptor/scripts/run_univ_flashvsr_wan21_complete_4gpu.sh all
```

The script uses existing `.venv_published_wan21` and `.venv_flashvsr_diagnostic`;
**do not rerun setup/build/download**. Override `NATIVE_PYTHON`, `FLASH_PYTHON`,
`BASE_PYTHON`, `SOURCE_STUDY_ROOT`, `FLASH_ROOT`, `FLASH_KERNEL_ROOT`, or
`FLASH_WEIGHTS` only if the working installation lives elsewhere.

Default source: `outputs/published_wan21_study_v2`. New independent output:
`outputs/flashvsr_wan21_complete_v1`. Override it with `FLASH_COMPLETE_ROOT`.
No old source, diagnostic or evaluation directory is written.

`all` runs plan → environment check → four-card native VAE decode → four-card
FlashVSR → finalization → export. Each card receives six clips. A worker loads
its model once, performs an excluded **full 81-frame** SR warmup and then runs
its assigned clips. Native decode and SR use separate processes/environments.
The environment check alone is not evidence of successful inference.

```bash
bash UNIV_adaptor/scripts/run_univ_flashvsr_wan21_complete_4gpu.sh status
# After a worker failure: inspect its printed log, fix the cause, then resume:
bash UNIV_adaptor/scripts/run_univ_flashvsr_wan21_complete_4gpu.sh all
```

Completed, hash-verified receipts are skipped. Interrupted **unreceipted files**
are not overwritten: inspect them and move the indicated artifacts aside first.
Ctrl+C terminates workers. Frozen code/input changes require a new output root.

## What is reused and what is actually computed

Reuse each S_B025 **50-step sigma-zero clean main latent**, before its old
RealESRGAN/HR4 branch. Decode the entire native 416×240×81 clip with the same
float32 Wan VAE used in the successful diagnostic. There is **no DiT/T5 rerun**,
old ST final-video input, RealESRGAN, VAE re-encode, HR4 or temporal interpolation.

Use the proven pinned FlashVSR v1.1 Tiny x4 implementation and sparse backend.
Edge-pad 416×240 to 416×256, repeat the last frame to 89 input frames, require
85 raw output frames, and remove only padded borders/tail. The content is x4
1664×960, then Lanczos-resized to the common 832×480 canvas. This preserves the
previous diagnostic recipe, **not a newly claimed native x2 model**.

Each result requires finite full-shape output, executed sparse attention,
successful color correction, and the exact frozen input hash. The actual
81-frame runtime and VRAM are recorded; they have not been measured locally.

## Outputs and interpretation

- Final videos: `media/p00_s42_S_B025/NATIVE_LR_FLASH.mp4` (and the other 23).
- Per-clip results: `results/*_NATIVE_LR.json`; lossless arrays/PNGs stay locally.
- Complete dataset index: `dataset_manifest.json`, with prompt/seed, input
  endpoint hash, SR timing and bindings to the existing four comparison arms.
- Download package: `outputs/flashvsr_wan21_complete_v1_videos.tgz`, containing
  the **24 final MP4s**, PNGs and provenance, but not weights, endpoints, NPZs or
  copies of the 96 existing comparison videos. Its paths are remote provenance;
  export is not a new portable scoring/human-rating package.

The method ID is `LR_FLASH_B025`: an **adapted native-LR + published FlashVSR
composite spatial baseline**, not an official published Wan acceleration preset.
Identical seed does not guarantee FULL/LR pixel alignment. Restoration timing
is not fresh end-to-end generation timing; no speedup is fabricated by adding
old generation times. Scoring and human assessment remain separate next steps.

Working paper claim: similar quality scores may hide explicit prompt/local-detail
failures. Completing these assets supplies a matched comparison set; it does not
yet establish metric insufficiency or prompt-router benefit.
