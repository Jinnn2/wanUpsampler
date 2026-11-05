# Attention v1: two sparse-attention arms on the designed prompts (pre-registration)

Written 2026-11-05, before any v1/dev2 attention video exists. Runner: `UNIV_adaptor/scripts/run_attention_v1.sh`.

## What runs

- **Prompts.** The platform v1 set (cf000–cf063, train + validation; the locked test prompts 64–79 are not in it)
  and the dev2 set (cf100–cf163), with seeds 42, 100 and 2024. That is 128 prompts × 3 seeds = 384 groups.
- **Arms.** Both use the platform's official-Wan path (`attention_wan.py`). Only the video self-attention changes.
  - **SVG2**: Sparse-VideoGen 2, with the official Wan2.1 SAP settings. The first 20% of steps are dense.
  - **SPARGE50**: SpargeAttn plug-in (`spas_sage2_attn_meansim_topk_cuda`) at 50% sparsity, on every step.
- **Reference.** The platform's own FULL50 clips and VBench rows, reused unchanged. This is allowed because on H100
  this path reproduces FULL50 bitwise: 4/4 natural_ext clips match by sha256 (`results/attention/attention_wan_h100`).
  The runner repeats this gate on 2 prompts of each set before generating.
- **Scoring.**
  - Same three layers as the dev analyses: fixed, dynamic retention and quality v2. They are computed on a combined
    root holding every dev source plus the two attention roots. Old clips come from the cache.
  - VBench: five quality dimensions plus two diagnostics.
- **Speed.** `pipeline_seconds` relative to the platform FULL50 on the same H100.

## Why these two arms

- The probe (4 natural_ext prompts, seed 42) gave:
  - SVG2: 1.08x, nearly the same clip as FULL (PSNR 22 dB).
  - SPARGE50: 1.47x, recomposed (13 dB); in one clip the main object changes mid-clip.
- **Dropped.**
  - SPARGE65 stalls the GPU kernel.
  - SPARGE80 hard-cuts and degrades.
  - SAGE2 and RADIAL were dropped by decision.

## Hypotheses (confirmatory)

- **Unit.** A prompt. Its score is `min(retention, quality_kept)` of the arm against FULL, averaged over the 3 seeds.
  This is the `min_ret_q` of the route table.
- **Uncertainty.** 95% intervals come from a bootstrap over prompt families: 10,000 resamples, fixed seed.
- **Pooling.** All 128 prompts are pooled. v1 and dev2 are also reported separately, descriptively.

**A1. SPARGE50 loses more on high-motion prompts.**
- Δmotion(SPARGE50) = mean score on low-motion prompts − mean score on high-motion prompts.
- Holds if the 95% CI is > 0.
- Rationale: the block mask comes from mean Q/K similarity per block. Large motion breaks the within-block similarity
  it relies on. It acts from step 0, where the layout is set.

**A2. SVG2 has no cell interaction.**
- Δmotion(SVG2) and Δdetail(SVG2) are defined as in A1, with detail cells in place of motion cells.
- Holds if both 90% CIs lie inside ±0.05.
- This is an equivalence test. Failing to reject a difference does not count as support.

**A3. SPARGE50 fails the temporal checks more often than SVG2.**
- Unit: clips paired by (prompt, seed).
- Count the pairs where exactly one arm fails a quality-v2 judder or flicker gate.
- Holds if SPARGE50 is the failing arm in more of them than SVG2, by a two-sided exact sign test, p < 0.05.

## Descriptive, not confirmatory

- VBench5 against the layered scores for the two new arms, next to the old arms. Does VBench rank SPARGE50 near FULL
  while the layered score does not?
- Speed at matched realized quality. The dev → dev2 cell router (route_v1 evaluate) is rerun with the new arms
  added to the pool. Do the attention arms enter any cell's choice at any τ?
- All results are per set (v1, dev2) and per cell.

## What does not change

- The frozen v1 driver and worker, `sample.json`, the checklist and the quality protocol stay as they are.
- The test split stays locked.
