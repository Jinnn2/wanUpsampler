# Controlled prompt-factor v1

## Prompt versus seed diagnostic (no new GPU work)

```bash
bash UNIV_adaptor/scripts/run_univ_controlled_factor_seed_audit.sh
```

Requires the existing `metrics/controlled_factor_vbench/relative_to_full.csv`.
The earlier compact download containing only prompt means is insufficient.
Outputs and the return archive are in `metrics/controlled_factor_vbench/seed_value_audit_v2`.
Test rows in the source CSV are skipped before numeric score parsing.

The audit evaluates validation using a train-selected fixed action and train
mean timing profile. It compares the fixed policy, the optimistic same-sample
prompt oracle, the instance oracle, and a two-seed selector on the third seed.
It reports ST quality-only, FST, and FSTC utility comparisons, pairwise variance,
winner margins, and actual regret of cross-seed choices. Existing T5/TF-IDF
validation predictions are included when present. Family bootstrap intervals
condition on the observed seeds and are exploratory with four validation families.
Large instance-oracle gaps do not establish early-state predictability.

This experiment tests whether prompt semantics predict the expected loss of
spatial, temporal, and cache acceleration relative to an uncompressed common
reference. It does not enumerate operation combinations.

## Design

- 20 semantic families, each with four independently authored prompts covering
  the complete `motion low/high x spatial-detail low/high` factorial.
- Family-disjoint split: 48 train prompts (12 families), 16 validation prompts
  (4 families), and 16 locked test prompts (4 families).
- Three seeds and four arms per prompt: FULL, spatial-only, calibrated
  temporal-only, and cache-only. Total: 960 videos.
- The first eight prompt ids exactly match targeted S/T v2, permitting 48 exact
  prompt/seed/action artifacts to be imported when `REUSE_ROOT` is available.
- Targets are three-seed mean `Q_action - Q_FULL`; lambda is applied after
  prediction using train-calibrated time ratios.

## Run

```bash
bash UNIV_adaptor/scripts/run_univ_controlled_factor_8gpu.sh check
bash UNIV_adaptor/scripts/run_univ_controlled_factor_8gpu.sh plan
bash UNIV_adaptor/scripts/run_univ_controlled_factor_8gpu.sh generate
bash UNIV_adaptor/scripts/run_univ_controlled_factor_8gpu.sh finalize
bash UNIV_adaptor/scripts/run_univ_controlled_factor_8gpu.sh score
bash UNIV_adaptor/scripts/run_univ_controlled_factor_8gpu.sh embed
bash UNIV_adaptor/scripts/run_univ_controlled_factor_8gpu.sh train
```

`all` performs the same sequence through validation training. It deliberately
does not evaluate the locked test split.

For a fast lexical baseline that does not require T5 extraction:

```bash
TRAIN_FEATURES=tfidf \
bash UNIV_adaptor/scripts/run_univ_controlled_factor_8gpu.sh train
```

Only after model, features, alpha, lambdas, and validation conclusions are
frozen:

```bash
CONFIRM_TEST_ACCESS=1 \
bash UNIV_adaptor/scripts/run_univ_controlled_factor_8gpu.sh confirm
```

The confirmation command creates `test_access_guard.json` and refuses a second
test access in the same training output directory.

## Outputs

- `controlled_factor_dataset.json`: generated artifact identity.
- `metrics/controlled_factor_vbench/prompt_targets_train_validation.csv`:
  development prompt-mean regression targets and time ratios.
- `metrics/controlled_factor_vbench/prompt_targets_test.csv`: locked confirmation
  targets, not loaded by the training mode.
- `factor_summary.csv`: controlled factor diagnostics.
- `prompt_prior_t5/validation_summary.json`: frozen validation evidence.
- `prompt_prior_t5/model.npz`: three-output Ridge model.
- `prompt_prior_t5/test_confirmation.json`: created only by explicit test
  confirmation.
