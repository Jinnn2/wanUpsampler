# Natural prompt set v1, extended

Written 2026-10-29, before any prompt was extended or any video generated.

## Why

Natural v1 (`NATURAL_PROMPTS_V1.md`, results in `results/derived/natural_v1/README.md`) came out "`cell` ≈ `blind`"
on 236 short VidProM prompts: the cell router was +1% over blind, and t5_knn beat cell at 0 of 6 τ. A post-hoc look
found three things:

- **Method effects show only at the extremes.** They appear only at the motion extremes of the generated videos
  (top vs bottom 15% FULL flow: TEA015 −0.14 [−0.24, −0.04], RCM4 +0.12, the same signs as dev2).
- **The short prompts barely predict motion.** They have a median of 15 words and were used verbatim. They hardly
  predict how much the video moves: held-out R² ≤ 0.19 from any prompt feature.
- **Wan is not meant to be used this way.** Wan's own inference rewrites short prompts with an LLM first.

This set asks whether routing regions come back when the same user prompts are extended the way Wan is meant to be
used.

## Prompts (`natural_extend.py extend`)

- **Source:** the 256 prompts of `prompts/univ_natural_v1.jsonl`, each rewritten once.
- **Extender:**
  - Model: Qwen3-VL-32B-Instruct, text only, greedy decoding, at most 320 new tokens.
  - System prompt: Wan2.1's English system prompt `LM_EN_SYS_PROMPT` from `wan/utils/prompt_extend.py`, copied
    verbatim (a test checks the copy).
  - Wan's default local extender is Qwen2.5-14B-Instruct. We use the judge model already on the machine, so one
    model does the screen, the checklist, the judging and the extension.
- **Ids:** prompt id = short id + 1000 (2000–2255). The family is the short prompt's family (`nat1xxx`), so short
  and extended versions pair up. The split is `natural_ext`.
- **Cells:** motion/detail come from the same route_v1 zero-shot questions, asked on the extended text, and are
  marked `cell_source: predicted`.
- **Review:** the extended prompts and configs are reviewed and committed before generation. An extension outside
  40–200 words is listed for review. The rule is to fix only empty or clearly broken outputs (a refusal, or a reply
  that is not a prompt), and each such fix is recorded in a `FIXES` table here.

## Generation and scoring

- **Seeds:** 42 and 100. Natural v1 had one seed. Seed noise is large: on controlled prompts, prompt × arm residuals
  correlate 0.31 across seeds.
- **Arms:** the same 8 as natural v1, from the same drivers and configs (built by `natural_prompts.build_configs`).
- **Scoring:** the same layers and code as natural v1 (fixed; dynamic with a machine-only checklist on the extended
  text; quality v2).
- **Entry point:** `NATURAL_SET=natural_ext bash UNIV_adaptor/scripts/run_univ_natural_8gpu.sh prompts`, then `all`.

## Pre-registered comparisons

All use `route_embed.py evaluate` on the natural_ext route table (seeds averaged per prompt × arm): 8 folds by
family × 10 repeats, speed at matched realized quality, τ ∈ {0.95, 0.90, 0.85, 0.80, 0.70, 0.60}, family bootstrap
95% CI. A comparison "holds" when the CI is above 0 at ≥ 3 of the 6 τ.

| # | Comparison | Holds → reading |
| --- | --- | --- |
| H1 (primary) | `cell` vs `blind` | Extension restores cell routing: on short prompts the text, not the absence of regions, was the bottleneck |
| H2 | t5_knn vs `cell` | Regions exist beyond motion × detail on extended natural prompts |
| H3 | qwenmid_ridge vs `cell` | The natural v1 exploratory lead (+7..+14% at τ ≤ 0.85) replicates on new videos |

If H1 does not hold, the reading is that extension does not make natural prompts routable by cell.

## Reported, not part of any decision

- **Pairs (seed 42 only, to match natural v1):**
  - held-out R² of log(1 + FULL flow) from each prompt feature, extended vs short;
  - the FULL flow distribution (median, 85th percentile) per predicted cell;
  - the cell × arm interaction share of retention and quality_kept;
  - the arm × cell interaction sign agreement with dev2.
- **The same comparisons with both seeds,** to show how much the second seed changes them.

Extension changes the generated videos as well as the text. So an extended-vs-short difference mixes "more
informative text" with "a different video distribution". It is not a pure test of either.

The decision rules above are fixed now and do not change after results arrive.

## Review of the extended prompts (2026-10-29, before any video was scored)

The server run used `natural_ext_v1_prompts.tgz` as-is; the files here are byte-identical to it.

- **Length:** 256 extensions, median 103 words (range 82–137). None falls outside 40–200, and none is a refusal or
  meta text.
- **Text kept from the user:** the extensions keep quoted text and titles from the user prompts (11). In 18 of them
  the extender added sound or music; Wan cannot render sound, so this has no effect on the video.
- **FIXES:** none.
- **Predicted cells shift strongly towards high:** hh 158, hl 74, lh 18, ll 6, against 56 / 67 / 55 / 78 for the
  short prompts.
  - 148 of the 256 prompts change cell.
  - The cause is rules 5–6 of Wan's system prompt, which tell the extender to add natural motion and camera
    movement to every prompt.
  - With only 24 low-motion prompts, the cell router has little to split on. H1 therefore has less power than it
    did on natural v1.
  - The rule is not changed. Whether the videos actually move more is reported under the pairs.
