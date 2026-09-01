# Timestep Router Experiments 1-3

This protocol turns the exploratory lambda sweep into a leakage-controlled,
prompt-level confirmation experiment. The primary operating point is
`lambda=0.08`; lambda changes require a new validation selection run.

## Evidence boundary

- Development data may use `quality_valid_legacy_vbench5_v1` only with the
  explicit `--allow_estimated_latency` flag. Its speedups remain estimates.
- Confirmation requires `strict_vbench5_v1`, `formal_evidence=true`, and
  `warm_pipeline_seconds` for Native-HR and every candidate branch.
- The prompt is the independent statistical unit. The three generation seeds
  are averaged into one prompt label; they are not three independent samples.
- The current router is prompt-only. These experiments do not establish a
  latent-conditioned sequential stopping policy.

## Experiment 1: multi-seed robustness and prompt bootstrap

Hypothesis: the learned router reduces policy regret relative to the best
training-selected fixed timestep across training initializations.

Protocol:

1. Freeze one prompt-disjoint split with `SPLIT_SEED=42`.
2. Train B1, B3, and B4 with five initialization seeds:
   `42 100 2024 31415 27182`.
3. Selection runs evaluate validation only and emit one row per validation
   prompt in `router_validation_predictions.csv`.
4. Average each prompt across training seeds, then bootstrap prompts 10,000
   times. Report the 95% interval and the standard deviation of run means.
5. Use paired prompt deltas against the best fixed timestep. Positive deltas in
   `multiseed_paired_deltas.csv` always mean that the learned router is better.

Run:

```bash
PRIMARY_LAMBDA=0.08 \
DATASET_DIR=/mnt/afs_2/houze/wanUpsampler/data/changing_resolution_uni/oracle_dataset_500_quality_valid \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/router_selection_500_quality_valid_lambda008 \
ALLOW_ESTIMATED_LATENCY=1 \
bash changing_resolution_uni/scripts/router/run_multiseed_router_selection.sh
```

Acceptance for proceeding to confirmation:

- all configured seeds finish with identical split/evidence metadata;
- B4 or another architecture is selected by mean validation policy regret;
- the paired regret improvement has a positive point estimate;
- test access remains `false` in `architecture_selection.json`.

Do not require Top-1 timestep accuracy to improve. Regret is primary; VBench-5,
latency, step MAE, and Top-1/Top-3 are secondary diagnostics.

## Experiment 1b: B4 soft distillation versus relative quality curves

After the original B4 passes Experiment 1, compare it against B4-Q without
changing the MLP widths, prompt split, initialization seeds, optimizer, or
evaluation metrics. B4-Q regresses the 13-point target
`candidate_vbench5 - step50_vbench5` with SmoothL1 loss. At evaluation time it
combines the predicted relative quality curve with the selected lambda and the
recorded normalized candidate latency. This isolates the supervision change
and does not access the test split.

Run in a new output root so the accepted B4 selection is not overwritten:

```bash
PRIMARY_LAMBDA=0.08 \
DATASET_DIR=/mnt/afs_2/houze/wanUpsampler/data/changing_resolution_uni/oracle_dataset_500_quality_valid \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/router_b4_quality_curve_comparison_lambda008 \
ALLOW_ESTIMATED_LATENCY=1 \
bash changing_resolution_uni/scripts/router/run_multiseed_quality_curve_comparison.sh
```

Keep B4-Q only if its validation policy regret improves over B4, or is
statistically indistinguishable while providing the desired lambda-independent
quality representation. Estimated-latency results remain development evidence.
Use `selection/multiseed_reference_paired_deltas.csv` for the direct B4-Q
minus B4 comparison; positive values always mean B4-Q is better.

## Experiment 1c: utility-aligned relative quality curves

B4-QA keeps the B4-Q architecture and relative quality output, but aligns it
with the routing objective using:

```text
KL(soft utility) + 0.5 * Wasserstein(soft utility)
+ alpha * SmoothL1(relative quality), alpha=1
```

Predicted utility logits are
`(predicted_relative_quality - lambda * normalized_latency) / tau`, using the
same `tau=0.02` as the target distribution. This is the only change from B4-Q.
Within each train seed, B4 and B4-QA reset the same initialization, DataLoader,
and dropout RNG streams before training so the loss is the isolated variable.

```bash
PRIMARY_LAMBDA=0.08 \
DATASET_DIR=/mnt/afs_2/houze/wanUpsampler/data/changing_resolution_uni/oracle_dataset_500_quality_valid \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/router_b4_quality_aligned_comparison_lambda008 \
ALLOW_ESTIMATED_LATENCY=1 \
bash changing_resolution_uni/scripts/router/run_multiseed_quality_aligned_comparison.sh
```

Use the direct reference-paired delta against B4. If B4-QA does not reduce
validation policy regret without degrading speed, stop prompt-only loss
experiments and proceed to latent-conditioned features.

## Variable-lambda sequential router on the 1,500-prompt dataset

The 1,500-prompt generator has two physical roots rather than one uniform
router directory:

```text
oracle_dataset_1500_8gpu/
  generation_plan.json
  generation_complete.json
  train/  # prompt IDs 0..999, one base seed
    t5_embeddings/
    _parts/part_*/raw_samples/seed_42/{manifests,latents,videos}/
  eval/   # prompt IDs 1000..1499, three base seeds
    t5_embeddings/
    _parts/part_*/raw_samples/seed_{42,100,2024}/{manifests,latents,videos}/
```

The actual stored seed is `base_seed + prompt_id`. Validation is prompt IDs
`1000..1199`; test is `1200..1499`. Do not pass these roots to the legacy
single-directory loader.

After generation completes, score the physical train/eval roots into isolated
strict datasets and extract lambda-independent state features for train and
validation only:

```bash
bash changing_resolution_uni/scripts/router/run_prepare_1500_variable_lambda.sh check
bash changing_resolution_uni/scripts/router/run_prepare_1500_variable_lambda.sh score
bash changing_resolution_uni/scripts/router/run_prepare_1500_variable_lambda.sh profile
bash changing_resolution_uni/scripts/router/run_prepare_1500_variable_lambda.sh features
```

The state builder resolves each latent through the authoritative sample
manifest, validates `wan_taa_free_oracle_latent_v1`, and emits one 13-state
feature file per trajectory. It does not read test record or manifest content
during selection preparation.

Train the multi-lambda prompt-only reference and prompt+state router with
matched initialization seeds:

```bash
bash changing_resolution_uni/scripts/router/run_multiseed_variable_lambda_selection.sh
```

Run the B4 extension through its separate launcher so it writes a fresh
four-way selection rather than overwriting the earlier two-model result:

```bash
bash changing_resolution_uni/scripts/router/run_multiseed_variable_lambda_b4_selection.sh
```

- `prompt_only`: sequential prompt+schedule regret/harm router;
- `prompt_state`: the same router plus 158-dimensional online latent state;
- `b4_offline`: generation-independent prompt+lambda B4 distribution over all
  13 candidates, trained with soft utility KL plus ordered EMD;
- `b4_prompt_state`: the exact selected B4 prior for that initialization seed is
  copied and frozen, then combined with schedule and latent state in a trainable
  sequential correction head.

The B4 target is
`softmax((VBench5 - lambda * normalized_cost) / B4_TEMPERATURE)`. The default
temperature remains `0.02` and the ordered-EMD weight is `0.5`; neither may be
tuned from test. The old development B4 checkpoint is not loaded because its
dataset and latency provenance differ from the strict 1,500-prompt protocol.
Pure B4 uses only prompt and lambda at inference and selects the probability
argmax. The hybrid B4 output is computed without latent input and serves only as
a prior; online latent state may advance or delay the actual handoff.

`paired_reference_deltas.csv` compares every candidate to `prompt_only`.
`paired_b4_deltas.csv` uses `b4_offline` as the paired reference, so its
`b4_prompt_state` rows directly measure the value of online latent correction.
Positive deltas always mean the candidate is better. Use a new `OUT_ROOT` for
this adaptive extension, and keep the earlier selection immutable.

### B4-anchored signed-advantage residual

The first hybrid is an unconstrained fusion model: B4 features are inputs, but
its stop decision is not a residual around the B4 argmax. The deterministic
five-seed result selected `b4_offline`; the hybrid switched systematically
later, especially at low lambda. Preserve that run as the V0 diagnostic.

The V1 residual model makes one isolated change. Its frozen B4 argmax defines an
anchor continuation logit, and the latent-state network can apply only a bounded
residual. A zero residual therefore reproduces B4 exactly. The supervised
target is the signed future continuation advantage

```text
max_{j > k} utility[j] - utility[k]
```

rather than the non-negative stop regret. Negative values retain the margin by
which stopping now is better. The action head predicts whether that signed
advantage exceeds `HARM_EPSILON`; the advantage regression head is diagnostic
and auxiliary. Epoch zero is evaluated and eligible for checkpoint selection,
so state training cannot be selected on validation when it is worse than the
exact B4 anchor.

Run the B4 reference and V1 residual with matched data, initialization seeds,
lambda grid, B4 loss, and H100 profile:

```bash
bash changing_resolution_uni/scripts/router/run_multiseed_variable_lambda_b4_residual_selection.sh
```

The launcher compares `b4_offline`, `b4_residual_prompt`, and
`b4_residual_state`. The prompt residual has the same B4 anchor, signed target,
and bounded correction but omits the 158-dimensional state input. Only the
state-minus-prompt residual comparison isolates the value of online latent
features. It is written to `paired_secondary_reference_deltas.csv` with
`b4_residual_prompt` as the secondary reference.

Keep V1 only when `paired_reference_deltas.csv` reports a positive macro
policy-regret
delta and the low-lambda rows no longer show the V0 late-switch failure. Epoch
zero or a selected macro delta of zero means that the current compact latent
statistics did not justify changing the B4 decision; it is a valid fallback,
not evidence that the residual model improved.

### V0.88 causal soft suffix-margin correction

The anchored signed-action result did not establish an online-state gain. Its
five-initialization state delta over B4 was effectively zero, while the prompt
residual always selected the exact epoch-zero fallback. V0.88 changes the
supervision and state conditioning without generating new videos or modifying
the prepared state dataset.

For B4 logits `l[k]`, define the offline sequential margin

```text
m_off[k] = l[k] - max(l[j] for j > k)
```

The first non-negative margin is exactly the B4 argmax. This preserves the
epoch-zero B4 policy while retaining B4 confidence, unlike the fixed-distance
argmax anchor. The online model applies a zero-initialized residual:

```text
m_online[k] = m_off[k] + delta_state[k]
```

The target is continuous rather than thresholded:

```text
m_target[k] = (utility[k] - max(utility[j] for j > k)) / temperature
y_target[k] = sigmoid(m_target[k])
```

Training uses survival-weighted soft BCE plus a small residual penalty. There is
no regret head, hard harmful-stop head, or `advantage > 0.001` training label.
The complete candidate sequence is one training example. State normalization is
fit on train separately for every candidate step, and a causal GRU carries only
past and current state evidence. The residual branch receives no T5 prompt
embedding; prompt semantics remain owned by the frozen B4 prior.

First run the isolated 32-trajectory, single-lambda overfit check:

```bash
bash changing_resolution_uni/scripts/router/run_soft_margin_overfit_sanity.sh
```

Both V0.88 launchers default to the completed strict state dataset
`router_variable_lambda_states_selection_20260829_h100_profile_v1`; the older
unversioned directory may exist as an empty placeholder. Override `DATASET_DIR`
explicitly only when using another complete directory with `dataset_manifest.json`.

This is a train-only diagnostic and must not be summarized as validation
evidence. Inspect `minimum_train_soft_margin_excess` rather than total soft BCE:
soft targets have irreducible entropy, while excess BCE has a zero optimum.
Then run the five-initialization validation comparison:

```bash
bash changing_resolution_uni/scripts/router/run_multiseed_variable_lambda_soft_margin_selection.sh
```

The comparison contains `b4_offline`, `soft_margin_control`, and
`soft_margin_state`. The control has the same B4 margin, state/schedule encoders,
GRU, soft target, bounded residual, and matched initialization, but feeds a zero
tensor instead of real state. Only the paired
`soft_margin_state - soft_margin_control` result isolates online-state value.
The first V0.88 experiment intentionally reuses the existing 158-dimensional
features. Structured latent tokens remain a later ablation so target semantics,
step normalization, temporal modeling, and representation are not all changed
in one run.

### Online factor relevance audit

Before adding another router or expanding the state encoder, measure whether each
existing online factor has prompt-disjoint and seed-specific relation to the
utility-optimal handoff. This audit reuses the completed state dataset and the
five frozen B4 checkpoints; it does not generate videos, train a router, or read
test content.

```bash
bash changing_resolution_uni/scripts/router/run_factor_relevance_audit.sh
```

The primary target is the continuous signed suffix-best utility margin. Oracle
argmax step and the sequential policy induced by predicted margins are secondary
targets. `schedule+B4 ensemble margin` is the main baseline. Ridge and one fixed,
small histogram-gradient-boosting model measure linear and nonlinear incremental
value for every feature group. Hyperparameter selection is prompt-grouped and
train-only; final metrics use the formal `200 prompts x 3 seeds` validation.

Two negative controls distinguish real trajectory evidence from proxies:

- train features are shuffled across trajectories independently within each
  candidate step;
- validation features are shuffled only among the three seeds of the same
  prompt before recomputing within-prompt association.

Use group-level predictive gains for decisions. Individual-feature correlations
are diagnostic: a factor is useful only when validation MAE/Brier or policy
regret improves, the prompt-bootstrap interval supports the gain, within-prompt
seed association exceeds its shuffle control, and the train within-step shuffle
removes the predictive gain. Raw correlation alone is not sufficient.

### Frozen-B4 residual correction selection

The factor audit found weak seed-specific state signal but no stable policy
gain, and also showed that unconstrained margin refitting degrades the exact B4
decision boundary. The follow-up therefore freezes the five-seed B4 ensemble
and learns only a bounded additive correction:

```text
corrected_margin = b4_margin + alpha * clipped_state_residual
```

The correction is applied only when `abs(b4_margin) <= tau`. `alpha=0` is an
explicit candidate and must reproduce every frozen-B4 margin and decision
exactly. Run selection with:

```bash
bash changing_resolution_uni/scripts/router/run_b4_residual_correction_selection.sh
```

Only three compact state choices are admitted: the global plus per-channel
`trajectory.delta_rms_per_sigma` features, the two global x0 temporal-gradient
features, and their union. A schedule+B4 residual model without state is kept
as the matched control. The residual predictor may use current-step schedule,
lambda, frozen B4 margin, and the selected current-step state; it cannot access
future states.

Ridge regularization, correction scale, and low-confidence gate threshold are
selected by prompt-grouped out-of-fold policy regret on the 1,000-prompt train
split. HistGradientBoosting uses one fixed small configuration while its scale
and gate are selected the same way. Validation then compares each selected
candidate directly with the untouched frozen B4 policy and with its matched
schedule-only correction using paired prompt bootstrap. Report margin error as
a diagnostic, but select the next router from macro policy regret, changed
better/worse counts, stability across lambda, and the paired intervals.

This remains validation-only architecture selection. The saved sklearn
correction checkpoints are candidates, not locked test-confirmed routers. B4
train predictions are in-sample because cross-fitted B4 checkpoints do not
exist; the prompt-disjoint three-seed validation is therefore the decisive
generalization check.

### Candidate steps 40-50 overall retraining

To test whether the coarse early candidates dominate online supervision and
irreversible stopping errors, retrain the current B4-anchored residual and
causal soft-margin suites after removing steps 30 and 35:

```bash
bash changing_resolution_uni/scripts/router/run_steps40_50_overall_selection.sh
```

The candidate set must be exactly
`40,41,42,43,44,45,46,47,48,49,50`. This is a training-time subset, not an
evaluation filter: state normalization, B4 soft utility targets, residual and
soft-margin labels, online sequences, train-selected fixed steps, checkpoints,
and validation decisions all use only these 11 candidates. Step 50 remains the
forced final decision. Candidate costs and seconds are selected from the same
locked full H100 profile, while the Native-HR denominator and profile hash stay
unchanged.

Removing 30 and 35 means they are not admissible handoff actions. The existing
step-40 `trajectory_delta` feature may still reference the observed step-35 LR
state, because that causal history is available before the step-40 decision; it
does not restore step 35 as a switch candidate.

The overall launcher trains five seeds for two matched suites:

- `b4_offline`, `b4_residual_prompt`, and `b4_residual_state`;
- `b4_offline`, `soft_margin_control`, and `soft_margin_state`.

Both suites use the same prompt/seed splits, train and interpolation lambdas,
feature groups, epochs, batch size, optimizer values, B4 temperature, and B4
EMD weight. For each training seed, B4 is trained once by the residual suite and
its hash-bound checkpoint/history are reused by the soft-margin suite. The
cross-suite summarizer still fails unless B4 choices and metrics match row by
row, then keeps one B4 copy and reports all five unique methods.
It writes macro and per-lambda intervals, paired deltas against B4, paired state
deltas against the matched no-state controls, and `overall_selection.json`.
This remains validation-only selection and does not access test states or
regenerate videos/latents.

All online validation runs must record
`evaluation_protocol=deterministic_eval_mode_v1`. Evaluation switches every
model to `eval()` before best-epoch selection and prediction export. Each run
writes one `*_training_history.csv` per model with separated train loss
components plus deterministic validation regret and harmful-stop rate. The
multi-seed summarizer rejects runs without this protocol or with missing or
modified history/checkpoint hashes; results produced by the earlier
dropout-active evaluation bug must remain diagnostic-only and cannot be mixed
into this selection.

Training lambdas default to `0.01,0.02,0.04,0.06,0.08,0.10`. Validation also
contains held-out interpolation points `0.03,0.05,0.07,0.09`. Lambda, sigma,
timestep, and a locked train-calibrated static cost profile are online inputs.
The same profile defines train/validation stop-regret labels, oracle/fixed-step
utilities, realized calibrated latency, and speedup; raw per-trajectory manifest
latency is diagnostic-only. The profile stores train-record and scored-manifest
hashes, hardware identity, distribution diagnostics, prompt bootstrap intervals,
and its own file hash is bound into the state dataset, checkpoints, five-seed run
summaries, and architecture selection. Per-trajectory VBench remains label and
evaluation-only. Model selection uses macro validation policy regret across
lambdas. Prompt bootstrap remains the statistical protocol despite multiple
seeds, states, and lambdas.

### Train-pool 800/200 control and OOD diagnostic

Use the train-pool control to distinguish router overfitting from a prompt-domain
shift in the existing validation set. The control is an isolated derived state
dataset: it does not regenerate videos, rescore VBench, re-extract latent
features, or modify the original state dataset.

```bash
PREPARE_ONLY=1 \
bash changing_resolution_uni/scripts/router/run_multiseed_train800_control_selection.sh

bash changing_resolution_uni/scripts/router/run_multiseed_train800_control_selection.sh
```

`prepare_train800_control_split.py` ranks prompt IDs `0..999` by
`SHA256("train800_control200_v1:" + prompt_id)`, assigns the first 200 hashes to
control validation, and assigns the remaining 800 to control train. It fails on
normalized duplicate prompt text, records both prompt-ID lists and source
hashes, binds every reused feature/T5 file by SHA256, and never opens the source
validation index or any test content. The five-seed launcher retains all four
model types, lambdas, hyperparameters, and the existing frozen H100 hardware
profile. Model state normalization and fixed-step selection use only the new
800-prompt train split.

After control selection completes, evaluate every frozen checkpoint on the
existing prompt IDs `1000..1199`:

```bash
bash changing_resolution_uni/scripts/router/run_train800_control_ood_diagnostics.sh
```

The OOD command evaluates base seed 42 for the matched one-seed domain
comparison and all three base seeds for robustness. It writes
`ood_base42_intervals.csv`, `ood_all3_intervals.csv`,
`control_vs_ood_base42_deltas.csv`, and, when the original train1000 run root is
available, `train800_vs_train1000_ood_paired_deltas.csv`. The OOD split is
diagnostic-only: it cannot select an architecture, and prompt IDs `1200..1499`
remain unread. Positive values in the train800-vs-train1000 paired file mean
train800 is better, but that comparison is descriptive because the two sets of
checkpoints were early-stopped on different validation domains. OOD paired
model/fixed files report whether improvements survive the domain shift.
Control-vs-OOD domain deltas are explicitly raw
`control_mean - ood_mean` because the prompt sets are disjoint.

## Experiment 2: formal quality, measured latency, and router overhead

Hypothesis: the locked router remains useful after replacing branch estimates
with measured resident-process latency and including router inference overhead.

The confirmation command fails closed unless every record has formal VBench-5
provenance and measured `warm_pipeline_seconds`. It measures the selected
router at batch size 1 after warm-up, synchronizes CUDA around every repeat, and
adds median router latency to learned-policy latency and utility. Fixed policies
and the prompt oracle do not pay router overhead.

Required report fields:

- VBench-5 and the five individual dimensions from the formal dataset;
- measured branch latency plus router median/p90/p95 overhead;
- peak router memory;
- speedup versus matched measured Native-HR;
- prompt-bootstrap intervals and paired deltas versus best fixed.

Router overhead is measured separately from model/checkpoint loading. The same
GPU type, scheduler, frame count, resolution, prompt, and generation seeds must
be used for all branch timings.

## Experiment 3: validation-only architecture selection, one locked test

`summarize_multiseed_selection.py` writes `architecture_selection.json` using
only validation predictions. The confirmation launcher reads the model type,
lambda, and split seed from that file. It refuses to overwrite an existing
`router_benchmark_summary.json`. Immediately before the first test iteration it
also writes `test_access_guard.json`; a failed partial run therefore still
blocks an accidental second test read unless the explicit override is supplied.

Run exactly once after the selection manifest is frozen:

```bash
DATASET_DIR=/mnt/afs_2/houze/wanUpsampler/data/changing_resolution_uni/oracle_dataset_1k_strict \
SELECTION_JSON=/mnt/afs_2/houze/wanUpsampler/outputs/router_selection_500_quality_valid_lambda008/selection/architecture_selection.json \
OUT_DIR=/mnt/afs_2/houze/wanUpsampler/outputs/router_confirmation_1k_strict_lambda008 \
bash changing_resolution_uni/scripts/router/run_locked_router_confirmation.sh
```

These paths follow `configs/local_paths.sh` (`PROJECT_ROOT` defaults to
`/mnt/afs_2/houze/wanUpsampler`) and the existing legacy/strict router scripts.
The confirmation command still fails if `oracle_dataset_1k_strict` contains
`estimated_warm_pipeline_seconds`; its path is real and traceable, but it is not
declared confirmation-ready until every branch latency is measured.

The newer 1,500-prompt generator writes raw physical datasets to
`data/changing_resolution_uni/oracle_dataset_1500_8gpu/train` and `.../eval`.
They are not interchangeable with the single-directory `DATASET_DIR` above:
the current router loader expects one scored manifest with uniform seed
coverage, while the 1,500 design intentionally uses one train seed and three
evaluation seeds.

The confirmation output includes:

- `router_test_predictions.csv`;
- `router_overhead.json`;
- `confirmation_test_intervals.csv`;
- `confirmation_test_paired_deltas.csv`;
- `confirmation_bootstrap_report.json`;
- B4 token attribution under `token_attribution_b4/` when B4 is selected.

## B4 token attribution

The old B3 attribution `w^T h_i` is invalid for nonlinear B4. B4 attribution now
targets the probability-weighted expected candidate timestep. For token `i`,
the reported value is:

```text
expected_step(full prompt) - expected_step(prompt pooled without token i)
```

Positive values push toward a later switch/staying LR; negative values push
toward an earlier HR switch. Leave-one-out effects are local perturbation
effects and are not additive Shapley values. Rankings therefore support
exploratory interpretation only; they are not causal claims.
## 40--50 single-factor geometry audit

Before training another online router, run the lightweight model-free audit with:

```bash
bash changing_resolution_uni/scripts/router/run_steps40_50_factor_geometry_audit.sh
```

It restricts actions and oracle labels to steps 40--50, detrends each state
factor with train-only per-step statistics, and measures whether factor values,
finite differences, two-step slopes, or trajectory change points identify the
validation oracle boundary. The single-factor threshold is diagnostic only;
the audit is validation-only, does not access test, and does not train or select
a deployable router.

## 40--50 B4-relative correction headroom

Before fitting a sparse temporal verifier, measure whether one-sided B4
corrections have seed-consistent oracle headroom:

```bash
bash changing_resolution_uni/scripts/router/run_steps40_50_b4_preemption_headroom.sh
```

The audit freezes the five-seed B4 probability ensemble and evaluates lower
(earlier HR, slower) and higher (later HR, faster) corrections with radii one,
two, three, and unrestricted. It separates an undeployable per-generation-seed
oracle from one-common-action, majority-positive, and all-three-positive upper
bounds. The output is validation-only diagnostic evidence; it does not select a
verifier or access test.
