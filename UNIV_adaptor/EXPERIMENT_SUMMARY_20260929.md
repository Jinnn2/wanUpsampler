# Prompt-guided video acceleration: experiment summary

Status: 2026-09-29. This report concerns prompt-only selection of acceleration
method and budget. It distinguishes development evidence from a confirmed
held-out policy result. The working decision target is cross-seed expected
utility, not the best action for an observed seed:

Scope: experiments that inform this method/budget-selection claim, including
the earlier budget-routing comparator. Independent super-resolution component
ablations and other paper tracks are outside this summary.

`U(p, a; lambda) = E_seed[Q(p, seed, a) - Q(p, seed, reference)] - lambda * C(a)`.

Here `Q` has usually been VBench5, the unweighted mean of subject consistency,
background consistency, motion smoothness, aesthetic quality, and imaging
quality. `C` has varied by experiment (measured time or train-calibrated time
ratio); utilities from different protocols must not be pooled. VBench5 is
neither official VBench Quality nor Total. Dynamic Degree and Overall
Consistency were generally diagnostics, not terms in `Q`.

## Executive conclusion

The experiments establish that spatial, temporal, and cache acceleration can
have different effects on different content, and that an in-sample prompt-level
oracle has positive headroom over a fixed action. They have **not** established
that the current prompt-only learner extracts that headroom on unseen semantic
families. Seed-by-configuration interactions, small prompt-level effect margins,
limited family diversity, and a metric that can reward lost motion all remain
plausible contributors. The evidence does not imply that prompt information is
useless, nor that seed is pure irreducible noise.

## Evidence by stage

| Stage | Comparable units and protocol | Main result | Evidential limit |
|---|---|---|---|
| Phase 1 budget pilot | Six standalone acceleration candidates without a paired native teacher | Established the initial quality/cost scoring workflow and the VBench5 working measure. | No locally supplied completed score artifact supports a numerical method-routing conclusion here; it did not measure native-HR quality loss. |
| Phase 2 fixed-lambda reuse | Three existing presets, `lambda=0.05`, train-calibrated cost | Both T5 and TF-IDF selected the same action as the train-fixed baseline on development validation; gain `0`. Hindsight utility oracle gain `+0.00458`. | Presets combine several changes; validation had already informed development. Oracle is an upper bound, not achievable policy gain. |
| Phase 3 sparse actions | 267 prompt-seed groups, 1,068 videos, one reference and three sampled probes per group | Faster combinations incurred nonuniform VBench5 loss; e.g. `SA_111` mean `-0.01993` at `0.753` reference time. | Sparse observations do not identify a full `2^d` action oracle or a prompt-specific winner. Reference was itself accelerated. |
| Phase 3 prompt + state audit | Reused 267 groups; completed-reference-video state proxy | At `lambda=0.05`, T5 prompt-only net gain vs reference `-0.000249`; prompt+state `+0.001143`, exactly equal to action-main and shuffled-state controls. TF-IDF prompt/state variants also collapsed to action-main. | The state is post-hoc, not an online early latent; no seed-specific added value was demonstrated. |
| Phase 4 matched star | 113 prompts, 339 prompt-seed groups, 1,356 videos; one-axis changes from the same accelerated reference | Mean delta VBench5: `STAR_S -0.014185`, `STAR_T -0.002204`, `STAR_C -0.002500`; time ratios `0.8427`, `0.9044`, `0.9042`. | These are reference-relative effects, not gaps to native HR. Only 69 of 339 groups had bound reusable Phase 2 native HR in the earlier audit. |
| Targeted S/T contrast | Eight authored prompts, three seeds, near-matched measured runtime (0.36% gap) | High-motion/large-subject prompts favored spatial over temporal: mean `Q(T)-Q(S)=-0.019386`. Low-motion/high-detail prompts were nearly tied: `+0.001279`. Difference-in-differences `0.020665`, prompt-bootstrap 95% CI `[0.005470, 0.038180]`. | Development-only, four prompts per class; the low-motion class did not show a robust predicted advantage. Prompt group was chosen to expose a contrast. |
| Controlled-factor S/T/C | 80 authored prompts, 20 semantic families, three seeds, four arms = 960 videos; 48/16/16 prompt train/validation/test split by family; native 50-step reference | At `lambda=0.05`, T5 validation gain over train-selected fixed temporal action `-0.003697`, family-bootstrap CI `[-0.009218, 0.001825]`; TF-IDF `-0.002229`, CI `[-0.006201, 0.000816]`. | Test was not used for model selection. Negative point estimates and intervals crossing zero do not establish a useful learned prior. |
| Seed-value audit of controlled-factor data | 16 validation prompts, four families, three observed seeds | For `{FULL,S,T,C}` at `lambda=0.05`, in-sample prompt oracle gain `+0.005565`, instance oracle `+0.010283`, but a two-observed-seed cross-seed selector gained `-0.000624` vs fixed (CI `[-0.003558, 0.003380]`). | Oracle values have selection optimism; cross-seed selector is not prompt-only. Four validation families and three seeds limit precision. |
| HunyuanVideo-1.5 endpoint pilot | Planned 16 prompts x four seeds x eight arms = 512 videos; scored complete snapshot: 25 groups, 200 videos, seven prompts | Full 50-step mean runtime `2008.8 s`; `S_B025 395.8 s`, `T_B025 399.6 s`, `C_B025 693.8 s`. Mean VBench5: FULL `0.827667`, S025 `0.833361`, T025 `0.839780`, C025 `0.826884`. | Incomplete, heavily imbalanced snapshot: 16/25 groups are low-motion/high-detail; no router was trained or tested. Hunyuan and Wan numbers are not directly comparable. |

The historical timestep/B4 budget classifier did show that a trained router can
beat a fixed-step baseline in its own validation setting: reported utility
`0.794861` for B4 versus `0.792435` for fixed step 48. However, that audit
explicitly labels its latency profile as legacy/unprovenanced and its result as
non-formal evidence. It studies a different decision space; it cannot establish
S/T/C method-selection generalization.

The online decision/early-state pilot has a protocol and implementation, but
no completed result artifact was among the evidence reviewed for this report.
The Phase 3 prompt+state audit used a finished reference video, so it must not
be counted as an online early-latent result.

## What the controlled studies say about seed

The paired S/T order is not stable for many prompts: in the controlled-factor
training split, 68.75% of prompts flipped the S-versus-T sign across three
seeds; in validation, 56.25% did. At `lambda=0.05`, 12/16 validation prompts
changed the winning action across the three seeds when `{FULL,S,T,C}` were
available. Nonetheless, prompt-level oracle headroom remained positive in the
observed sample. Thus seed interactions reduce the reliability of hard
per-seed winner labels; they do not prove that cross-seed expected action
effects are absent. The variance audit does not separate stochastic seed
noise from seed-by-method interaction or metric error.

## Measurement and protocol cautions

1. VBench5 can mask temporal-content failures. In the Hunyuan snapshot,
   `T_B025` had higher mean VBench5 than FULL but Dynamic Degree positives
   fell from 14/25 to 4/25. This aggregate is confounded by the snapshot's
   low-motion concentration; inspect the high-motion paired cases and videos.
2. Nominal proxy density (`0.5`, `0.25`) is not elapsed-time ratio. Hunyuan
   RGB recovery and four full-resolution refinement steps add almost fixed
   overhead; compare measured pipeline seconds when evaluating budgets.
3. `FULL50_HR4` is needed to isolate restoration effects, but it is not the
   same baseline as `FULL50`. The Hunyuan cache arm reuses last guided velocity,
   not TeaCache. Keep these definitions visible in comparisons.
4. Official VBench Total requires its 16 dimensions and dimension-specific
   prompt protocol. The locked VBench custom-input path does not support all
   semantic dimensions. A custom-prompt VBench5 or seven-dimension Quality
   calculation must not be presented as leaderboard-equivalent Total.
5. Prompts, not seeds, are independent selection/evaluation units. Keep test
   semantic families locked. Reusing development validation for action,
   metric, lambda, or model tuning makes it development evidence only.

## Paper claim and next validation gate

The defensible present claim is: **content factors alter the relative
quality-cost effects of acceleration axes, but prompt-only routing beyond a
strong train-selected fixed action has not yet been demonstrated.** The next
claim is conditional: a prompt prior can choose method and budget to improve
cross-seed expected utility at a declared quality-risk level.

To test it without changing direction:

1. Keep one explicit primary estimand, e.g. paired `Q(action)-Q(FULL)` minus
   `lambda` times measured time ratio, with lambda and action set frozen before
   confirmation. Also report quality-loss exceedance, mean regret, action
   distribution, and speedup; do not rely on classification accuracy alone.
2. Complete the missing `temporal_flickering` diagnostic on existing videos,
   and audit motion adherence on paired high-motion cases. Keep VBench5 as a
   separate legacy measure; do not silently replace labels after inspecting
   confirmation data.
3. Use the current Hunyuan snapshot only for development. It is not balanced
   by motion/detail or complete across prompts. Any new generation should
   first fill missing factor cells and seeds for the locked eight-arm protocol,
   subject to compute budget, before fitting a new prompt selector.
4. Evaluate a frozen prompt-only model against train-selected fixed actions,
   prompt-blind action-mixture control, and shuffled-prompt assignment on
   unseen semantic families. Compare both per-prompt cross-seed means and
   seed-level harm. Only then claim prompt-derived value.

## Source artifacts

- `C:/Users/jinho/Downloads/utility_prior_{t5,tfidf}_three_lambda_0p05/report.md`
- `C:/Users/jinho/Downloads/metric_v3/report.md`
- `C:/Users/jinho/Downloads/920/prompt_state_{t5,tfidf}_lambda_0p05/report.md`
- `C:/Users/jinho/Downloads/univ_matched_star_phase4_analysis.tgz`
- `C:/Users/jinho/Downloads/targeted_st_vbench/{report.md,analysis.json}`
- `C:/Users/jinho/Downloads/controlled_factor_analysis.tgz`
- `C:/Users/jinho/Downloads/prompt_prior_tfidf/{report.md,policy_summary.csv}`
- `C:/Users/jinho/Downloads/seed_value_audit.tgz`
- `C:/Users/jinho/Downloads/02e9dfba339e3555/{snapshot.json,quality_by_video.csv,report.md}`
- `C:/Users/jinho/Downloads/selection/router_validation_summary.json`
- Frozen protocols: `UNIV_adaptor/configs/univ_controlled_factor_v1.json`,
  `UNIV_adaptor/configs/hy15_endpoint_prior_v1.json`.
