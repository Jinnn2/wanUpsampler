# Natural set v1, extended: verdict

Inputs: `results/natural/natural_ext_v1_results` (server run, 2026-11-04). Pre-registration:
`UNIV_adaptor/NATURAL_EXT_V1.md`. The prompts are the 256 natural v1 prompts rewritten by Wan2.1's extender, with
seeds 42 and 100. 250 have a valid checklist, and all 250 are scored.

## Pre-registered decisions: none holds

Speed gain at matched realized quality, CI above 0 at 3 or more of the 6 τ:

| | τ 0.95 | 0.90 | 0.85 | 0.80 | 0.70 | 0.60 | holds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H1 cell vs blind | −1% [−3, +1] | +1% [−1, +4] | +3% [−2, +7] | +4% [−2, +8] | +0% | +0% | no (0/6) |
| H2 t5_knn vs cell | −2% | +3% [−1, +8] | +2% | −4% | +0% | +0% | no (0/6) |
| H3 qwenmid_ridge vs cell | −100% | −6% | −2% | −6% | +0% | +0% | no (0/6) |

At τ ≤ 0.70 every policy reaches 7.34x. On extended prompts RCM4 alone already keeps min(ret, Q) at about 0.70,
so there is nothing left to route there.

## Descriptive pairs (`compare.py`, output in `compare.txt`)

| | short (seed 42) | ext (seed 42) | ext (both seeds) |
| --- | --- | --- | --- |
| FULL flow, median / 85th percentile (px) | 0.28 / 1.45 | 0.98 / 4.41 | 0.94 / 5.5 |
| text → log(1+flow), held-out R², T5 / Qwen / Qwen-mid | 0.10 / 0.17 / 0.11 | 0.45 / 0.49 / 0.57 | 0.46 / 0.54 / 0.60 |
| text → flow, predicted motion cell | 0.07 | 0.05 | 0.05 |
| cell × arm share, retention / quality_kept | 0.011 / 0.006 | 0.011 / 0.005 | 0.009 / 0.004 |
| RCM4 retention / quality_kept | 0.61 / 0.48 | 0.71 / 0.74 | 0.70 / 0.73 |

Contrast of the top 15% vs the bottom 15% FULL flow, using min(ret, Q) relative to the prompt mean, both seeds:

| arm | short (v1) | ext |
| --- | --- | --- |
| RCM4 | +0.12 [−0.00, +0.24] | **+0.19 [+0.11, +0.28]** |
| SCALING10 | +0.02 | **+0.09 [+0.05, +0.14]** |
| TEA015 | **−0.14 [−0.24, −0.04]** | **−0.09 [−0.16, −0.02]** |
| F41_RIFE | +0.02 | **−0.08 [−0.14, −0.02]** |
| LR41_FLASH_RIFE | −0.08 | **−0.09 [−0.17, −0.01]** |

- **The extremes contrast replicates.** The RCM4 and TEA015 directions were set by natural v1 before this run, and
  they hold in each seed separately.
- **Seed noise is still large.** Prompt × arm residuals correlate 0.23 (retention) and 0.31 (quality_kept) between
  seed 42 and seed 100.

## Exploratory: routing by motion instead of by cell (`motion_groups.py`, `motion_router/`)

We replaced `cell` in the route table by motion quartiles and ran the same `route_embed.py evaluate`. Gain vs blind:

| groups | τ 0.95 | 0.90 | 0.85 | 0.80 | 0.70 | 0.60 |
| --- | --- | --- | --- | --- | --- | --- |
| quartiles of motion predicted from text (Qwen-mid ridge, out-of-fold R² 0.59) | +1% [−2, +6] | +2% [−1, +5] | +5% [−1, +8] | +4% [−4, +11] | +0% | +0% |
| quartiles of the actual FULL flow (ceiling, not available before generation) | +0% [−5, +7] | **+7% [+3, +10]** | **+9% [+3, +13]** | **+10% [+1, +18]** | +0% | +0% |

## Reading

1. **Extension works on the text side.**
   - Videos move about 3x more.
   - The prompt embedding now predicts how much the video moves (R² up to 0.60, against 0.17 before).
   - The motion-extreme effects are larger and replicate across seeds.
2. **The motion × detail cell is the wrong label after extension.**
   - The Wan extender adds motion to every prompt, so 92% are "high motion". The predicted cell then predicts
     nothing (R² 0.05).
3. **Routing by motion has a ceiling of about +10% at τ 0.80–0.90.**
   - Knowing the true motion gives a CI above 0 at 3 τ. Text-predicted motion gets about half of that (+2–5%) and
     does not clear 0.
   - Below τ 0.70, RCM4 alone suffices.
4. **Arm main effects still dominate.**
   - Arms explain 46–57% of the variance; cell × arm explains under 1%.
   - The prompt-specific part is mostly seed noise.
   - On natural prompts, prompt → method routing is worth at most about 10%. That gain sits in the moderate-τ band
     and needs motion that is only partly predictable from the text.
