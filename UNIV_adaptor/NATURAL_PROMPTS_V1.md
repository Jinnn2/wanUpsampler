# Natural prompt set v1

Written 2026-10-27, before any natural prompt was drawn or generated.

## Why

On the 125 controlled-factor dev/dev2 prompts, a router built on prompt vectors (Wan umT5 + kNN, `route_embed.py`)
matches the router built on the hand-made motion x detail cells, at matched realized quality (−2 to −6%, CIs touch
0). It does not beat that router. The umT5 neighbours share the designed cell 79% of the time, and kNN restricted
to one cell does not beat the cell mean. Those prompts vary only the two designed factors, so they cannot show
whether method preferences have regions beyond them. This set tests that on prompts nobody designed.

## Prompts (`natural_prompts.py`)

- **Source:** VidProM `VidProM_semantic_unique.csv`, real Pika user prompts (CC BY-NC 4.0, research use only).
- **Filters:**
  - English, 6–70 words.
  - All six Detoxify probabilities ≤ 0.02, plus a keyword block list (sexual, gore, minors, extremist terms).
  - No URLs, handles or hashtags; Pika flags such as `-gs 12 -motion 2` are stripped.
  - Exact and 6-word-prefix duplicates are dropped.
- **Draw:** 768 candidates, stratified over three length bins (6–11, 12–24, 25–70 words), seed 20261027.
- **Screen:** Qwen3-VL-32B text-only, P(safe, renderable scene, not text/logo/meme/named person) ≥ 0.5, then 256
  prompts (≈85 per length bin), in candidate order. Prompts are used verbatim, with no prompt extension.
- **Ids and nominal cells:**
  - Prompt ids are 1000–1255; each prompt is its own family; the split is `natural`.
  - motion/detail are the zero-shot route_v1 questions (P(Yes) ≥ 0.5), marked `cell_source: predicted`. They exist
    so that the frozen drivers' four-cell check passes, and as the "predicted cell" router baseline.
  - The checklist generator does not force static or motion questions on predicted cells.
- **Review:** the prompt file and the generation configs are reviewed and committed before generation:
  `prompts/univ_natural_v1.jsonl`, `configs/published_wan21_{platform,twostage}_natural.json`.

## Generation and scoring

- **Seeds:** one seed (42).
- **Arms (8):** FULL50, STEP20, TEA015, SCALING10 (frozen v1 driver); LR_FLASH, F41_RIFE, RCM4, LR41_FLASH_RIFE
  (two-stage driver, FULL50 reused).
- **Arms dropped:** STEP25 and TEA008 (dominated by SCALING10); DVG (dominated on every cell); LR_BICUBIC, F41_DUP,
  RCM2 and RCM1 (controls, or too low quality).
- **Scoring:** the same layers as the main experiment:
  - fixed layer;
  - dynamic layer with a machine-only checklist (`requirements_natural/checklist.jsonl`, never human-reviewed);
  - quality layer v2, with judder/flicker thresholds taken from the main experiment's FULL clips.
- **Entry point:** `run_univ_natural_8gpu.sh` (`prompts`, then `all`).

## Pre-registered comparison

`route_embed.py evaluate` on the natural route table: 8 folds by prompt × 10 repeats, speed at matched realized
quality for τ ∈ {0.95, 0.90, 0.85, 0.80, 0.70, 0.60}.

| Outcome | Reading |
| --- | --- |
| t5_knn beats `cell` (the Qwen-predicted cell) with a CI above 0 at ≥ 3 of the 6 τ | Regions exist beyond motion × detail; a vector router is needed |
| t5_knn ≈ `cell` | The two designed factors explain prompt → method routing on natural prompts too |
| `cell` ≈ `blind` | Cell routing gains do not transfer to natural prompts (e.g. natural prompts rarely ask for the extremes) |

Also reported, but not part of the decision:
- whether the arm × predicted-cell interaction signs from dev/dev2 replicate;
- the speed gain of each router over `blind`.

The decision rule above is fixed now and does not change after results arrive.
