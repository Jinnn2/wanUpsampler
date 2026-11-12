# Human check of the three-way quality judge (pre-registration)

Written 2026-11-12, before any kit is built or any label exists. This check is the external validation of the
three-way quality judge in `UNIV_adaptor/QUALITY_V3.md`. It is part of the human line, separate from M3 / M3q, and
it does not change them.

Code: `UNIV_adaptor/scripts/data/annotation_tri.py` (select / kit / analyze). Runner:
`UNIV_adaptor/scripts/run_univ_annotation.sh tri-kit | tri-analyze`. Pair list: `prompts/annotation_tri_v1/pairs.json`,
drawn by `select` with seed 20261112 from `results/attention/quality_tri_seeds/tri_by_pair.csv`, the seed-triangle
pilot (32 prompts, 16 high and 16 low motion). Locked test prompts 64–79 are not in the pilot.

## Pairs (40, plus 6 repeats)

| set | n | content | expectation |
| --- | --- | --- | --- |
| seed100 | 12 | FULL seed 100 against another FULL seed, high motion. The 5 prompts where the judge and the single-clip scores disagree most (9, 25, 147, 155, 163) are against seed 42. Seven more high-motion prompts are drawn at random; for each, the partner seed (42 or 2024) alternates between the one the judge separates from seed 100 more strongly and the one it separates less strongly. | open: (a) |
| null | 4 | FULL seed 2024 against seed 42, the 4 high-motion prompts left over | no direction |
| svg2 | 12 | SVG2 against FULL seed 42, 12 random high-motion prompts | open: (b) |
| scaling | 6 | SCALING10 against FULL, 3 high- and 3 low-motion prompts | mostly "about the same" |
| bad | 6 | RCM1 (3) and LR_BICUBIC (3) against FULL, both motion levels | FULL clearly better |

- Sides are random per pair.
- Six pairs are shown a second time with sides swapped (4 seed100, 2 svg2), at least 10 tasks apart, to measure
  self-consistency.
- Each annotator gets an own task order. Method and seed are never shown.

## Task

- Same page as M3 (`annotate.html`, kind `tri`), the same CRF16 re-encode, and the same prompt text in Chinese and
  English.
- Four questions per pair. Each is answered "left better / about the same / right better":
  - **overall**: sharpness, artifacts and temporal stability together;
  - **sharp**, **artifact**, **temporal**: the judge's wording for these three aspects.
- There is no "unsure" answer. Near-equal pairs are "about the same", which matches the judge's Same.
- Annotators: A (the user), and B if available. The analysis works with one or two annotators.

## Judge side

- The judge's answer on a question = argmax of P(x better), P(y better), P(Same), each averaged over aspects (and
  over both presentation orders, as in `tri_by_pair.csv`).
- **overall** is compared with the mean over all five aspects, which is the v3 `net`. The mean over sharp, artifact
  and temporal (`overall3`) is reported as secondary.
- **sharp**, **artifact**, **temporal** are compared with the judge on that aspect alone.

## Analysis (`human_tri_analysis.md`, `human_tri_by_pair.csv`)

Only the first presentation of each pair is used, except in step 2.

1. **Annotator check.** On the bad set, overall question: FULL better on ≥ 5 of 6. An annotator who fails is
   excluded from steps 3–5 and reported.
2. **Self-consistency.** Agreement between the two presentations of the repeated pairs.
3. **Annotator agreement** (two annotators). Raw agreement and three-way Cohen's κ per question, on all pairs and on
   the hard sets (seed100, null, svg2).
4. **Human vs judge.** Per question:
   - raw agreement and three-way κ, hard sets (primary) and all pairs;
   - the share of "same" for human and for judge;
   - sign agreement on the pairs where both the human and the judge pick a side.
   - With two annotators, this is computed per annotator and on the consensus (pairs where both agree).
5. **Directional questions** (overall question): counts of worse / same / better, and a one-sided exact sign test
   over the pairs where a side is picked.
   - For seed100, the counts are from seed 100's side.
   - For svg2, scaling and bad, they are from the arm's side.

## Decision rules (fixed now; not changed after labels are seen)

- **D1. Judge agrees with people on overall.** On the hard sets, the judge counts as validated on the overall
  question if both hold:
  - sign agreement ≥ 0.75;
  - with two annotators, κ(human, judge) ≥ κ(A, B) − 0.1 for each annotator; with one annotator, κ(human, judge)
    ≥ 0.2.

  The same rule, applied to each aspect question, says which aspects the human check supports.
- **D2. (a) Seed 100 worse on high motion.** Confirmed if, on the consensus (or the single valid annotator):
  - the sign test for "seed 100 worse" gives p < 0.05;
  - and "worse" outnumbers "better".

  Contradicted if the sign test for "seed 100 better" gives p < 0.05. Otherwise: not settled by people at n = 12;
  reported as such.
- **D3. (b) SVG2 worse than FULL on high motion.** The same rule as D2 on the svg2 set.
- **D4. "Same" calibration.** If people answer "same" on more than half of the hard-set pairs where the judge picks
  a side, the judge is over-separating different clips. This is reported as a limit of v3 and does not by itself
  fail D1.
- Power: with 12 pairs, p < 0.05 needs at least 10 of 12 on one side (9 of 10, 7 of 8, 6 of 6 when some are "same"). The check is meant to
  catch a judge that disagrees with people, not to estimate effect sizes.
