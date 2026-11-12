# Quality layer v3: three-way pairwise judge (pre-registration)

Written 2026-11-12, after the pilots below and before any v3 score of the full plan exists.
Runner: `UNIV_adaptor/scripts/run_quality_v3.sh`; scorer: `UNIV_adaptor/scripts/data/quality_tri.py`.

## Why

`quality_kept` (v1/v2) asks the judge "is X at least as good as Y" (Yes/No) and keeps an aspect only when both
presentation orders say Yes. That rule charges for being different. Two FULL clips of one prompt at different
seeds keep only 0.71 (0.49 on high-motion prompts) (`results/attention/quality_null`).

Pilots (`results/attention/quality_tri`, `quality_tri_seeds`; 32 prompts, 16 high + 16 low motion) of a three-way
question (A better / B better / Same) on the same judge, frames and aspects:
- **Identity.** Same on 32/32 clips, no position bias. In-format token mass 1.00.
- **Near-identical arm.** SCALING10: net −0.01, Same 0.86.
- **Known-bad arms.** RCM1 and LR_BICUBIC: net −0.76.
- **Transitivity.** The seed triangle (42/100/2024) is transitive.
  - corr(direct, via the third seed) = 0.95.
  - 0/32 cycles; 0.25 is expected by chance.
- **Seed effect.** Seed 100 is judged worse on high-motion prompts: −0.33 (se 0.09).
  - A seed is the same initial noise for every prompt, so a seed can shift all prompts together.
  - Seed 100 also has more local flow and flicker in the fixed layer.
  - Seeds 42 and 2024 are even: net +0.005.
- **Same is rare for different videos.** For two clips with different content the judge rarely says Same: 0.03 on
  high motion. So per-clip v3 scores of recomposed clips are noisy, and only means are interpreted.

## Metric

- For an accelerated clip X against its same-seed FULL Y, per aspect, average over both presentation orders (and over
  the 5 frame pairs for frame aspects):
  - P(X better), P(Y better), P(Same).
  - `net_aspect = P(X better) − P(Y better)`.
- `net` = mean over the five aspects.
- `quality_tri = 1 + net`. FULL is 1. The score can exceed 1, and is not clipped, because clipping would bring back
  the bias for different clips.
- The judge, frames, aspects and questions are those of `score_quality.py`; only the answer format changes.
- Aggregates are means over clips, then over seeds. Uncertainty comes from a bootstrap over prompt families (10,000
  resamples, seed 0), as in ATTENTION_V1.md.

## What is run

- Every accelerated clip of the attention-combined quality plan, against its same-seed FULL. That is 17 arms × 384
  (prompt, seed) groups on v1 + dev2.
- **Null.** All three FULL seed pairs (100–42, 2024–42, 2024–100) of every prompt. Their mean over the three pairs
  is 0 by construction. The null is used for:
  - the judge's spread on different clips, by motion level;
  - the seed main effect, now on all 128 prompts.

## Checks (reported first; they qualify everything after)

- **C1. Seed effect replicates.** Seed 100's mean score against the other two seeds is < 0 on high-motion prompts
  (95% CI). If not, the pilot effect was chance at n = 16.
- **C2. Low motion is exchangeable.** On low-motion prompts every seed's mean score is inside ±0.05.

## Questions (re-asked on v3)

- **Q1. Cell interaction per arm.**
  - Δmotion(arm) = mean `quality_tri` on low-motion prompts − mean on high-motion prompts.
  - An arm has a motion interaction if the 95% CI of Δmotion is > 0.
  - Reported for all 17 arms, and compared with the v1 `quality_kept` Δmotion. The v1 version was inflated by the
    "different" penalty, which is larger on high motion.
- **Q2. Attention arms.** A1 and A2 of ATTENTION_V1.md, recomputed with `min(retention, quality_tri)` in place of
  `min(retention, quality_kept)`. The decision rules are unchanged. These are reported as a re-analysis, not as a
  replacement of the original verdicts.
- **Q3. Routing.**
  - Rerun the dev → dev2 router (route_v1 table/evaluate) with `quality_tri` as the quality column.
  - Same τ grid, classifiers and dev2 split as before.
  - Reported: does the router still beat the blind policy on speed at matched retention and quality, and which arms
    enter which cells?

## Not changed

- The retention layer. It has the same one-directional pairwise design. Its null control is the next step and is
  not part of this run.
- The frozen v1 driver and worker, sample.json, and the locked test split (prompts 64–79 are not in the plan).
- **Human agreement.** Requested from human-line: about 40 pairs, three-way, judged by people. That check, not this
  file, is the external validation of the v3 judge.
