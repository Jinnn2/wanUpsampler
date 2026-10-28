# Natural set v1: verdict

Inputs: `results/natural/natural_v1_results` (server run 2026-10-27). Pre-registered rule: `UNIV_adaptor/NATURAL_PROMPTS_V1.md`.

## Which prompts are scored

- 256 VidProM prompts were drawn, and 248 have a valid machine checklist.
- 8 have no valid checklist: 1010, 1169, 1182, 1189, 1200, 1208, 1221, 1231. `route_table.csv` drops them.
  - `table_inputs.json` (server copy) lists 136 dropped ids. 128 of those (0-63, 100-163) only ever had FULL clips
    and were never in the table; commit 58b1c00 narrows the list.
- `route_embed.py evaluate` drops 12 more prompts for undefined scores. This happens, for example, when FULL meets
  none of the requirements, so retention is undefined. This leaves **236 prompts**.
- Nominal (Qwen-predicted) cells: hh 52, hl 60, lh 54, ll 70.

## Pre-registered decision

The rule tests whether t5_knn beats `cell` with a CI above 0 at 3 or more of the 6 τ. It does at **0 of 6**.

| τ | 0.95 | 0.90 | 0.85 | 0.80 | 0.70 | 0.60 |
| --- | --- | --- | --- | --- | --- | --- |
| t5_knn vs cell | −4% [−10, +2] | −0% [−5, +3] | +1% [−4, +5] | +4% [−3, +9] | +7% [−3, +14] | +7% [−3, +15] |
| cell vs blind | +1% [−2, +3] | +1% [−1, +4] | +1% [−0, +5] | +1% [−0, +6] | +1% [−1, +8] | +1% [−2, +7] |

**Reading:** "`cell` ≈ `blind`". On natural prompts the predicted motion × detail cell carries almost no routing
gain (+1%), and the umT5 kNN router does not beat it either. The cell router's gains on the controlled prompts do not
transfer to natural prompts.

## Not part of the decision (exploratory)

- **qwenmid_ridge** (mid-layer Qwen embedding, ridge) beats `cell` with CI > 0 at τ ≤ 0.85: +7%, +9%, +12%, +14%.
  It also beats `blind` at τ ≤ 0.90.
  - This was not the pre-registered comparator, and 9 feature sets were tried, so it is a lead, not a result.
  - It would need its own pre-registered test on fresh prompts.
- **Arm × cell interaction replication** (`interaction_signs.py`, output in `interaction_signs.txt`):
  - Method: two-way interaction residuals of min(mean ret, mean Q) per (arm, cell), natural vs dev2, over the 7
    shared arms.
  - The signs agree in **14/28** cells (chance level), and in 8/17 where the dev2 interaction is at least 0.05.
    Pearson r = −0.24.
  - The natural interactions are all small (|x| ≤ 0.06); dev2 has up to ±0.33 (RCM4 hh).
  - On natural prompts the arms' quality ranking is nearly the same in every predicted cell. The dev2
    interactions do not replicate.
- **Caveats:**
  - One seed.
  - The cells are predicted (zero-shot Qwen), not designed.
  - The checklist is machine-only.
  - Natural prompts may rarely hit the extremes that the controlled factors were built to vary.

## Are the cells the problem? (cluster_signal.py, difficulty_signal.py)

Embeddings are the npz files inside the server's `route_natural_embed_analysis.tgz` (not in the repo). Run:
`python cluster_signal.py route_table.csv emb_t5.npz,emb_qwen.npz,emb_qwenmid.npz labels.json [fixed_by_video.csv]`.

- **Groupings tried:**
  - k-means (k = 2, 4, 8) on each prompt embedding (L2-normalized, PCA 32).
  - As an upper bound that no prompt router can reach: k-means on the FULL clip's measured features (flow,
    flicker, sharpness, color, ...), plus quartiles of flow and sharpness.
- **Statistics, each against 300 or 100 label permutations:**
  - the share of variance in group × arm;
  - held-out (8-fold) R² of the group's arm profile over the global arm profile.
- **Result:**
  - No grouping beats its permutation null in a way that matters. Every group × arm share is at most 0.019,
    against nulls of 0.002–0.014.
  - Held-out R² is at most +0.004 and mostly below 0.
  - The few p < 0.05 entries are on quality_kept only, out of 45 tests.
  - The FULL-video groupings are no better than the prompt clusters.
- **Difficulty** (a prompt's mean over the 7 arms) is barely predictable either: held-out ridge R² ≤ 0.09 for cell,
  T5, Qwen and Qwen-mid.
- **Reading:**
  - On these 236 prompts × 1 seed there is no group-level prompt → arm structure to find, whether the groups come
    from cells, embedding clusters or the generated video itself.
  - Any remaining structure is per-prompt and at the seed-noise level: on controlled prompts the prompt × arm
    residuals correlate only 0.31 across seeds.
